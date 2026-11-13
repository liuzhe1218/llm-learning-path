# 3.7 分布式与 PD 分离服务（Distributed & Disaggregated Serving）

!!! abstract "本章你将学到"
    - 什么时候、为什么一个模型必须跨多张 GPU 来服务，以及三种切分方式：**张量并行（tensor parallelism）**、**流水线并行（pipeline parallelism）**和**专家并行（expert parallelism）**，以及它们各自的通信开销。
    - 为什么**预填充/解码分离（prefill/decode disaggregation，PD 分离）**能提升 goodput，以及如何估算在不同池之间传输 KV cache 的开销。
    - 什么是**以 KV cache 为中心**的架构和**分层 KV 存储**（GPU → CPU → SSD），以及什么时候加载已缓存的前缀比重新计算更划算。
    - 万亿参数级的 MoE 模型在生产中如何服务：宽专家并行、双 batch 重叠（dual-batch overlap）、专家负载均衡（DeepSeek 的设计）。
    - 请求如何在多个副本间路由（包括**前缀感知路由**），以及服务集群如何自动扩缩容。

## 为什么重要 {#why-this-matters-the-problem}

一张 H100 只有 80 GB 的 HBM。Llama-3-70B 在 BF16 下，仅权重就需要约 140 GB；DeepSeek-V3（总参数 671B）即使用 FP8 也需要约 671 GB。而且就算模型放得下，单张 GPU 的带宽（≈ 3.35 TB/s）也限制了解码速度（见 3.1 章），剩余的显存则限制了 KV cache，从而限制了 batch 大小（见 3.2 章）。

所以大模型要部署在**一组 GPU** 上——可能在一台服务器内（8 张 GPU 通过 NVLink 互联），可能跨一个机架，也可能跨多台通过 InfiniBand/RoCE 网络相连的服务器。模型怎么切分，决定了每一步有多少数据要穿过这些链路，而链路比 HBM 慢得多。此外，大规模部署还会在路由器背后运行**许多副本**，并且必须决定每个请求交给哪个副本。本章覆盖这两个层次：把单个模型实例切分到多张 GPU 上，以及运行一整个实例集群。

## 直觉 {#intuition}

- **张量并行**就像几位厨师*同时做同一道菜*，每人负责切每种食材的一部分；他们得不停地互相递送切好的食材（需要高速链路）。
- **流水线并行**就像一条流水线；每个工位负责几层，然后把半成品传给下一个工位。通信很少，但一道菜从头到尾的耗时等于整条流水线的长度。
- **专家并行**就像一群专科厨师；每份订单（token）被送到它需要的两位或八位专家那里，结果再送回来（一次 all-to-all 洗牌）。
- **PD 分离**就像把备料（预填充）和摆盘（解码）放在不同的厨房，这样一份巨大的备料订单永远不会堵住摆盘流水线——代价是要把备好的食材（KV cache）在厨房之间搬运。

## 工作原理 {#how-it-works}

### 互联链路（近似数字） {#interconnects-approximate-figures}

| 链路 | 典型带宽 | 用途 |
|---|---|---|
| H100 SXM 上的 HBM3 | ≈ 3.35 TB/s | 权重、KV、激活 |
| NVLink 4（H100 SXM，8 卡服务器内） | 每 GPU ≈ 900 GB/s（双向合计） | 张量/专家并行 |
| PCIe Gen5 ×16 | 每方向 ≈ 64 GB/s | GPU ↔ CPU 内存 |
| InfiniBand NDR（每块网卡 400 Gb/s） | 每块网卡每方向 ≈ 50 GB/s | 跨服务器；H100 服务器通常每张 GPU 配一块网卡 |

NVLink 与网络带宽之间的比值（约一个数量级）解释了大部分部署选择：通信量大的并行方式留在 NVLink 域内；通信量小的并行方式才跨服务器。NVIDIA GB200/GB300 NVL72 这样的机架级系统把 72 张 GPU 放进同一个 NVLink 域，所以特别适合宽专家并行。

### 张量并行（TP） {#tensor-parallelism-tp}

Megatron 式 TP 把每一层的权重矩阵切分到 $N$ 张 GPU 上：

- **MLP。** $W_{\text{up}}$（$d \times d_{\text{ff}}$）按**列**切分：GPU $i$ 计算 $h_i = \sigma(x W_{\text{up}}^{(i)})$，即隐藏激活的一个切片，无需通信。$W_{\text{down}}$（$d_{\text{ff}} \times d$）按**行**切分：GPU $i$ 计算一个部分输出 $h_i W_{\text{down}}^{(i)} \in \mathbb{R}^{d}$，完整输出是各 GPU 结果的**求和**——一次 **all-reduce**。
- **注意力。** 注意力头在各 GPU 之间划分（每张 GPU 拥有 $n_q/N$ 个 query 头和 $n_{\text{kv}}/N$ 个 KV 头，以及这些头对应的 KV cache）；输出投影按行切分，之后同样做一次 all-reduce。

```text
TP=2 的 MLP:   x（在两张 GPU 上各有一份副本）
             │                              GPU 0                      GPU 1
             ├── x @ W_up[:, :d_ff/2] ─▶ h0 ─▶ act ─▶ h0 @ W_down[:d_ff/2, :] ─▶ y0 ─┐
             └── x @ W_up[:, d_ff/2:] ─▶ h1 ─▶ act ─▶ h1 @ W_down[d_ff/2:, :] ─▶ y1 ─┤
                                                                       all-reduce: y = y0 + y1
                                        （到这里之前都没有通信；y 又在各卡上各有一份完整副本）
```

所以一次前向**每层有两次 all-reduce**，每次作用在形状为 `[tokens, d_model]` 的 tensor 上。在 $N$ 张 GPU 上对 $M$ 字节做 ring all-reduce，每张 GPU 大约发送并接收 $2\frac{N-1}{N}M$ 字节。对解码来说每步的量不大（token 少），但很频繁；每次 all-reduce 还有固定的延迟，所以 TP 带来了随 $N$ 增长的逐层同步开销。TP 的最大好处是**每张 GPU 只读取 $1/N$ 的权重**，所以总带宽——也就是 batch 为 1 时的解码速度——随 $N$ 扩展（直到通信成为主导）。

### 流水线并行（PP） {#pipeline-parallelism-pp}

PP 把连续的若干层分给不同的 GPU（stage）。只有 stage 边界处的激活需要传输（点对点，`[tokens, d_model]`），所以 PP 能容忍慢速链路，常用于跨服务器。但单个请求仍然要依次经过所有 stage，所以 PP **并不能降低单请求延迟**；它只增加容量。为了让所有 stage 都忙起来，引擎必须同时有多个 micro-batch 在飞行中；否则各 stage 会闲置（"流水线气泡 pipeline bubble"）。

### 面向 MoE 的专家并行（EP） {#expert-parallelism-ep-for-moe}

在 MoE 层中（见 [第 1 部分](../part1/index.md)），路由器为每个 token 从 $E$ 个专家里挑出 top-$k$ 个。使用 EP 时，这 $E$ 个专家分布在多张 GPU 上（例如 256 个专家分到 32 张 GPU，每张 8 个专家）。于是每个 MoE 层需要两次 **all-to-all** 交换：

1. **分发（dispatch）：** 每张 GPU 把自己各个 token 的隐藏向量发给承载该 token 所选专家的那些 GPU。
2. **合并（combine）：** 专家的输出被送回 token 所在的"本家"GPU，并按路由权重求和。

```text
 GPU 0 的 token [t1 t2]   GPU 1 的 token [t3 t4]        专家：E0,E1 在 GPU 0 · E2,E3 在 GPU 1
 路由器 (top-2):  t1→E0,E2   t2→E1,E3   t3→E0,E1   t4→E2,E3

 分发 (all-to-all)            专家计算                    合并 (all-to-all)
 GPU0 ─ t1,t2 的副本 ─▶ GPU1   GPU0: E0{t1,t3} E1{t2,t3}   输出返回各 token 的
 GPU1 ─ t3   的副本 ─▶ GPU0    GPU1: E2{t1,t4} E3{t2,t4}   本家 GPU，按路由概率
                                                           加权求和
```

每个 token 在每个 MoE 层大约有 $k \cdot d \cdot b$ 字节发出去又收回来（如果几个被选中的专家恰好在同一张 GPU 上则更少；分发时还可以用 FP8 发送）。该层的其余部分（注意力）通常在同样这些 GPU 上用**数据并行**来运行（"attention DP"），对 MLA 模型尤其如此，否则它们小小的 KV cache 会被 TP 复制多份。难点在于**负载不均衡**：如果很多 token 都选中同一个"热门"专家，它所在的 GPU 就会成为落后者，下一次 all-to-all 时所有人都得等它。

### 预填充/解码分离（PD 分离） {#prefilldecode-disaggregation}

3.1 章讲过：预填充是计算受限且耗时长的；解码是带宽受限且对延迟敏感的。把它们混跑在同一批 GPU 上会产生**干扰**：一个预填充 batch 会让解码停顿（损害 TPOT），而分块预填充（chunked prefill，见 3.3 章）只能限制损害的程度。这两个阶段也需要**不同的配置**：预填充喜欢高计算利用率和适中的并行度；解码喜欢大 batch、充足的 KV 显存，以及能让每步延迟最小的并行方式。

**PD 分离**把预填充和解码放在不同的 GPU 池里运行：

1. 请求被送到一个**预填充实例**，它计算 prompt 的 KV cache 以及第一个 token。
2. KV cache 被**传输**（通过 NVLink 或 RDMA 的 GPU 到 GPU 传输）到一个**解码实例**。
3. 解码实例继续生成；它的 batch 里从不包含预填充。

```text
 请求 ──▶ 路由器
               │
               ▼
 预填充池  （按 prompt token/s 配置规模；计算受限；例如 TP=8）
   预填充实例: prompt ─▶ KV cache + 第一个 token
               │
               │  KV 传输：RDMA / NVLink，逐层流水线化
               ▼
 解码池   （按并发序列数配置规模；带宽受限；大 batch，例如宽 EP）
   解码实例: 数百条序列每步各出 1 个 token ─▶ 流式返回给客户端
```

两个池可以独立确定规模（例如输出很长的负载就多配一些解码 GPU），以最大化 **goodput**——即同时满足 TTFT 和 TPOT 两个 SLO 的每秒请求数，这一定义出自 DistServe。Splitwise 表明，同样的思路甚至可以让两个阶段使用不同代际的 GPU。代价是 KV 传输、额外的调度复杂度，以及对高速网络的需求。传输通常按层流水线化，即一层的 KV 在下一层还在计算的时候就已经发出去了。

### 以 KV cache 为中心的架构与分层存储 {#kv-cache-centric-architectures-and-tiered-storage}

一旦 KV cache 能够在机器之间移动，它们就成了**共享资源**。Mooncake（Moonshot AI 的 Kimi 背后的系统）围绕这一点构建了整套架构：一个横跨集群中 GPU HBM、CPU DRAM 和 SSD 的分布式 KV cache 池、一个 RDMA 传输引擎，以及一个把请求安排到其前缀 KV 所在位置（或者去取回它）的调度器。Llumnix 会在实例之间对正在运行的请求连同它们的 KV 做热迁移（live migration），以重新平衡负载。开源的构件包括 **LMCache**（在引擎实例之间做 KV 卸载和共享）以及 Mooncake 的传输引擎，SGLang 和 vLLM 都在使用。

分层 KV 的基本决策是**加载还是重算**：只要满足下式，从较慢的层级复用已缓存的前缀就划算：

$$
\frac{\text{KV bytes of the prefix}}{\text{tier bandwidth}} < \text{time to recompute the prefix (prefill)} .
$$

由于 KV 字节数随前缀长度线性增长，而预填充的 FLOPs 至少是线性增长（注意力部分是二次增长），对长前缀来说通常是加载更划算，算例会展示这一点。

### 服务大型 MoE 模型：DeepSeek 的设计 {#serving-large-moe-models-deepseeks-design}

DeepSeek 发布了一份 V3/R1 推理系统的概览，它已成为一份参考设计：

- 预填充与解码**分离**。
- **预填充：** 路由专家用 EP32，注意力（MLA）和共享专家用 DP32；一个部署单元横跨 4 个节点。**解码：** EP144，注意力用 DP144，横跨 18 个节点。两者都为热门专家使用**冗余**副本（32 个冗余的路由专家）。
- **双 batch 重叠（dual-batch overlap）：** 把 batch 切成两个 micro-batch，这样当一个在做 all-to-all 通信时，另一个在做计算——把通信隐藏在计算之后。
- **EPLB**（专家并行负载均衡器）：根据观测到的专家负载，决定复制哪些专家以及如何把专家装箱到各 GPU 上，使每张 GPU 的工作量相近。
- **DeepEP**：经过优化的 all-to-all 分发/合并 kernel（预填充用高吞吐 kernel，解码用低延迟 kernel）。

LMSYS 在 SGLang 中，于 96 张 H100 GPU 上复现了这种风格的部署（PD 分离 + 大规模 EP）。共同的教训是：在这个规模上，系统是围绕**通信**（all-to-all、KV 传输）来设计的，和围绕计算来设计一样重要。

### 如何选择布局 {#choosing-a-layout}

| 场景 | 典型选择 | 原因 |
|---|---|---|
| 模型能放进一张 GPU 且给 KV 留有余量 | 每个副本 1 张 GPU，多个副本（数据并行） | 完全没有通信；通过增加副本来扩展 |
| 稠密模型需要 2–8 张 GPU | 在一台 NVLink 服务器内用 TP | 切分权重和 KV；all-reduce 都留在 NVLink 上 |
| 稠密模型大于一台服务器 | 服务器内 TP × 服务器间 PP | PP 的点对点流量能容忍较慢的网络 |
| 大型 MoE（数百个专家） | 专家用 EP，注意力用 DP（或小 TP），通常跨服务器或在机架级 NVLink 域内 | 专家占据了主要显存；对稀疏层来说 all-to-all 比 TP 扩展得更好 |
| 长 prompt + 大规模下严格的 TTFT 与 TPOT SLO | 在上述基础上加 PD 分离 | 消除干扰，分别确定各阶段规模 |

### 副本间的路由与负载均衡 {#routing-and-load-balancing-across-replicas}

一个集群会运行许多独立的副本（每个副本可能本身是多卡的）。路由器为每个请求选择一个副本：

- **轮询 / 随机：** 简单；忽略负载和缓存状态。
- **最小负载：** 按队列长度、运行中的请求数或 KV cache 利用率；对延迟友好。
- **前缀感知（缓存感知）：** 把请求发给已经在缓存中持有其前缀的副本（见 3.3 章），例如通过对 prompt 的前几个块做哈希，或者为每个副本的缓存维护一棵近似的基数树（radix tree）。这可以成倍提高聊天和 agent 场景下的前缀缓存命中率，但纯粹的亲和性会制造热点，所以实际的路由器会把缓存亲和性与负载阈值结合起来。
- **会话亲和（session affinity）：** 让一段对话固定在同一个副本上——这是前缀感知路由的一种粗糙形式。

实现了 KV 感知路由的生产级方案包括 NVIDIA **Dynamo**、**llm-d**（Kubernetes 原生、基于 vLLM）和 SGLang router。

### 自动扩缩容 {#autoscaling}

流量随小时和日期波动，所以集群要上下伸缩副本数。有用的信号是队列长度、KV cache 利用率，以及观测到的 TTFT/TPOT 与 SLO 的对比（GPU 利用率则是个糟糕的信号：访存受限的解码副本在任何负载下看起来都可能"很忙"）。难点在于**冷启动**：新副本必须分配 GPU、加载几十到几百 GB 的权重，并捕获 CUDA graph，然后才能提供服务，这需要几十秒到几分钟。缓解办法包括保留热备容量、从本地 NVMe 或对等 GPU 快速加载权重，以及在 PD 分离系统中独立地伸缩预填充池与解码池。

## 代码 {#code}

下面是一个对应本章各项决策的粗算规划器：TP 下每张 GPU 的显存、每个解码步的 TP all-reduce 流量、PD 分离下的 KV 传输时间，以及分层 KV 的"加载 vs 重算"。所有硬件数字都是近似输入，你应该换成自己的实测值。

```python
GB, GiB = 1e9, 2**30

def kv_bytes_per_token(layers, kv_heads, head_dim, b=2):
    return 2 * layers * kv_heads * head_dim * b

# Llama-3-70B，BF16
P, L, KVH, HD, D = 70e9, 80, 8, 128, 8192
kv_tok = kv_bytes_per_token(L, KVH, HD)

for tp in (2, 4, 8):
    w = P * 2 / tp                                   # 每张 GPU 的权重字节数
    print(f"TP={tp}: weights/GPU {w / GB:5.1f} GB, KV room on 80 GB GPU "
          f"{(72 * GB - w) / GB:5.1f} GB, batch-1 bound {1000 * P * 2 / (tp * 3.35e12):.1f} ms/token")

# 每个解码步的 TP all-reduce 流量：每层 2 次 all-reduce，作用于 [batch, d_model] 的 BF16
tp, batch = 8, 64
msg = batch * D * 2
per_gpu = 2 * L * 2 * (tp - 1) / tp * msg            # ring all-reduce：每次 2(N-1)/N * M
print(f"\nTP=8, batch 64: {2 * L} all-reduces/step, {per_gpu / 1e6:.0f} MB sent per GPU "
      f"-> {1000 * per_gpu / 450e9:.2f} ms at ~450 GB/s per direction (NVLink 4)")

# PD 分离：传输一个 8K token prompt 的 KV cache
ctx = 8192
kv = ctx * kv_tok
prefill_flops = 2 * P * ctx + 2 * ctx**2 * HD * 64 * L   # 权重部分 + 因果注意力（64 个 q 头）
prefill_s = prefill_flops / (8 * 989e12 * 0.5)       # 8 张 H100，按 BF16 峰值的约 50% 计
print(f"\nKV for {ctx} tokens: {kv / GB:.2f} GB; prefill ~{prefill_s * 1000:.0f} ms on 8xH100")
for name, bw in [("1 x 400Gb/s NIC", 50e9), ("8 x 400Gb/s NICs", 400e9), ("PCIe5 from CPU", 50e9)]:
    print(f"  transfer over {name:17s}: {1000 * kv / bw:6.1f} ms")
```

## 实例演算 {#worked-example}

**1. 装下 Llama-3-70B。** 在 BF16 下，权重约 140 GB。在 80 GB 的 H100 上用 TP=2，每张 GPU 放 70 GB 权重，几乎不剩给 KV cache 的空间——技术上放得下，实际上没法用。TP=4 时每张 GPU 剩约 37 GB 给 KV（假设可用约 72 GB），TP=8 时剩约 54 GB。batch 为 1 时的带宽上界，在不计通信开销时，从 ≈ 21 ms/token（TP=2）改善到 ≈ 5 ms/token（TP=8）。另一个办法是用 FP8 权重（≈ 70 GB）加 TP=2，每张 GPU 约有 37 GB 的 KV 空间，而且 GPU 数量只有一半。

**2. TP 通信。** 在 TP=8、解码 batch 为 64 时，每次 all-reduce 的消息大小是 $64 \times 8192 \times 2 = 1$ MiB；每步有 160 次 all-reduce，每张 GPU 发送约 294 MB，在每方向约 450 GB/s 下耗时约 0.65 ms。这在一个约 10–20 ms 的解码步里占比明显但不是主导；160 次同步的固定延迟和字节数本身一样重要。如果跨服务器、带宽约 50 GB/s，同样的流量每步要约 6 ms——这就是 TP 要留在 NVLink 域内的原因。

**3. PD 分离的 KV 传输。** 一个 8K token 的 prompt 在 Llama-3-70B 上产生 $8192 \times 320\ \text{KiB} \approx 2.68$ GB 的 KV cache。预填充的权重部分约需 $2 \times 70\times10^9 \times 8192 \approx 1.15\times10^{15}$ FLOPs，注意力部分约需 $0.09\times10^{15}$——在 8× H100、BF16 峰值约 50% 的利用率下，大约 **0.3 s**。传输时间：

| 路径 | 带宽 | 传输 2.68 GB 的时间 |
|---|---|---|
| 一块 400 Gb/s 网卡 | ≈ 50 GB/s | ≈ 54 ms |
| 八块 400 Gb/s 网卡（每张 GPU 一块，各自发送自己的 KV 头） | ≈ 400 GB/s | ≈ 6.7 ms |

有了每 GPU 一块网卡以及逐层流水线化，传输只会给 TTFT 增加几个百分点。如果只有一条慢速链路且没有流水线化，则会增加约 20%。FP8 KV 能让这两个数字都减半。

**4. 加载 vs 重算。** 同样这 2.68 GB，如果从 CPU 内存经 PCIe Gen5 加载（实际约 50 GB/s），耗时约 54 ms，而重算要约 300 ms：加载快约 5 倍，而且还把 GPU 腾出来做别的预填充。对于一个 500 token 的前缀，对比是约 3 ms 对约 18 ms 的预填充——仍然划算，但绝对数值很小。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **服务器内的 TP（常见为 TP=8）或 TP×EP**，是单卡放不下的模型的默认方案；**PP** 用于跨服务器或超大的稠密模型。
- **宽 EP + attention DP + PD 分离**，是 DeepSeek-V3/R1 级和 Kimi-K2 级 MoE 模型的标准配方；SGLang、vLLM、TensorRT-LLM 和 Dynamo 都提供支持。机架级 NVLink 系统（GB200/GB300 NVL72）把 all-to-all 保持在同一个 NVLink 域内，使宽 EP 更便宜。
- **开源栈：** NVIDIA Dynamo（带 KV 感知路由和跨层级 KV 块管理的 PD 分离服务）、llm-d（Kubernetes 原生、基于 vLLM、带缓存感知调度）、LMCache（KV 卸载与共享）、Mooncake 的传输引擎与 KV 存储。
- 模型厂商提供的 **API "prompt caching"**，是以 KV cache 为中心的服务在用户面前的体现：被缓存的前缀会保留（通常放在较低的层级）若干分钟，并以折扣价计费。

## 常见误区 {#common-pitfalls-misconceptions}

- **"GPU 越多 = 延迟越低。"** TP 在一定程度内会降低每 token 延迟；PP 完全不会降低；超过某个 TP 度之后，通信会把收益抵消掉。
- **跨服务器做 TP。** 每层两次 all-reduce 走约 50 GB/s 的网络，会让解码变得很慢；跨服务器应改用 PP 或 EP。
- **TP 度大于 KV 头的数量。** 这时 KV 头必须被复制，浪费 KV 显存（见 3.2 章）。
- **忽视专家负载不均衡。** 平均专家负载看起来很正常，而最热门专家所在的 GPU 决定了步耗时；要监控每 GPU 的最大负载，而不是平均值。
- **过早做 PD 分离。** 对小模型、短 prompt 或流量不大的场景，在混跑的 GPU 上用分块预填充更简单，效果也几乎一样好；PD 分离是在大规模、长 prompt 和严格 SLO 的场景下才划算。
- **前缀感知路由没有负载上限，** 会让每个带有热门 system prompt 的请求都落到同一个副本上。

## 本章要点 {#key-takeaways}

- TP 切分每一个矩阵，每层需要两次 all-reduce → 要把它留在 NVLink 域内；它能为单个模型实例扩展总带宽。
- PP 切分层，只需廉价的点对点传输，但不会降低延迟；EP 分布 MoE 专家，需要 all-to-all 分发/合并以及负载均衡。
- PD 分离消除预填充与解码之间的干扰，让每个阶段都能为 goodput 单独确定规模和并行方式；有了每 GPU 的 RDMA 和逐层流水线化，KV 传输的开销是可以承受的。
- KV cache 可以分层存放（HBM → DRAM → SSD → 远端）；加载一个长前缀通常比重算便宜得多。
- 集群层面的路由应该把负载均衡与缓存亲和性结合起来；自动扩缩容必须考虑缓慢的冷启动。

## 自测 {#check-your-understanding}

??? question "为什么张量并行在行并行矩阵之后需要 all-reduce，而在列并行矩阵之后不需要？"
    把 $W_{\text{up}}$ 按列切分，每张 GPU 得到的是隐藏激活的一个完整切片，后面逐元素的激活函数可以在本地处理。把 $W_{\text{down}}$ 按行切分，每张 GPU 得到的是对隐藏维度中它那一片的部分和，真正的输出是所有部分和之和。

??? question "一个 MoE 层把每个 token 路由到 256 个专家中的 top-8，d = 7168，分发用 FP8，合并用 BF16。假设所选专家都在其他 GPU 上，每个 token 在每个 MoE 层大约有多少字节离开一张 GPU？"
    分发：$8 \times 7168 \times 1 \approx 57$ KB；合并（收回来）：$8 \times 7168 \times 2 \approx 115$ KB。每个 token 每个 MoE 层大约发出 57 KB、收到 115 KB——再乘以 token 数和层数，这就是 EP 需要快速的 all-to-all kernel 和计算通信重叠的原因。

??? question "为什么 PP 不能降低单个请求的延迟？"
    请求的 token 必须按顺序经过每一个 stage；把层分到不同 GPU 上，并不能让任何一个 stage 在该 token 的上一个 stage 完成之前就开始。PP 是靠让不同的 micro-batch 同时处在不同的 stage 来提高吞吐的。

??? question "PD 分离对哪些请求最没用？"
    低负载下的短 prompt 加短输出：几乎没有干扰可消除，而 KV 传输和调度开销纯属额外成本。

??? question "前缀感知路由把你的缓存命中率从 20% 提到了 70%，但 P99 TTFT 反而变差了。可能发生了什么？怎么解决？"
    热门前缀把流量集中到了少数几个副本上，这些副本过载了。把缓存亲和性与负载感知的回退结合起来（当首选副本的队列超过阈值时，路由到负载较低的副本），或者把热门前缀复制到多个副本上。

## 延伸阅读 {#further-reading}

| 论文 / 资源 | 为什么读 |
|---|---|
| :material-star: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024 | 分离预填充/解码实例，按阶段分别配置并行方式，以最大化 goodput。 |
| :material-book-open-variant: [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — Patel et al., ISCA 2024 | 来自微软的同一思路，包括为每个阶段使用不同的硬件。 |
| :material-star: [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Qin et al. (Moonshot AI), FAST 2025 | Kimi 背后以 KV 为中心的生产级设计：横跨 GPU/CPU/SSD 的分布式 KV 池，配合 RDMA 传输。 |
| :material-book-open-variant: [Llumnix: Dynamic Scheduling for LLM Serving](https://arxiv.org/abs/2406.03243) — Sun et al., OSDI 2024 | 在实例之间对请求及其 KV cache 做热迁移，以实现负载均衡。 |
| :material-book-open-variant: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | 列并行/行并行的张量并行方案，推理中也在用。 |
| :material-book-open-variant: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Pope et al., 2022 | 对推理中各种切分策略及其通信开销的系统化分析。 |
| :material-star: [DeepSeek-V3/R1 inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | 预填充 EP32 / 解码 EP144、双 batch 重叠、冗余专家：大型 MoE 的参考设计。 |
| :material-book-open-variant: [DeepEP](https://github.com/deepseek-ai/DeepEP) | 面向专家并行的 all-to-all 分发/合并 kernel。 |
| :material-book-open-variant: [EPLB](https://github.com/deepseek-ai/EPLB) | 带冗余热门专家的专家并行负载均衡器。 |
| :material-new-box: [Large-scale EP in SGLang (LMSYS blog)](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | 在 96 张 H100 上复现 DeepSeek 风格的 PD 分离 + 大规模 EP。 |
| :material-book-open-variant: [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | 开源的 PD 分离服务框架，带 KV 感知路由。 |
| :material-book-open-variant: [llm-d](https://github.com/llm-d/llm-d) | 基于 vLLM 构建的 Kubernetes 原生分布式服务方案。 |
| :material-book-open-variant: [LMCache](https://github.com/LMCache/LMCache) | 在引擎实例之间做 KV cache 卸载与共享。 |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 这些服务设计所针对的 MoE 架构（256 个路由专家、top-8、MLA）。 |
