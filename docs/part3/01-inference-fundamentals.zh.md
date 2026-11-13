# 3.1 推理基础

!!! abstract "本章你将学到"
    - 写出自回归生成循环，并解释**预填充（prefill）**与**解码（decode）**两个阶段的区别。
    - 了解对推理真正重要的 GPU 知识：HBM 与片上 SRAM、张量核心（tensor core）、峰值 FLOP/s 与显存带宽。
    - 用**算术强度（arithmetic intensity）**和 **roofline 模型**说明解码受显存带宽限制，并算出它变成计算受限时的批大小。
    - 定义并测量 TTFT、TPOT/ITL、端到端延迟、吞吐、goodput 和 SLO。
    - 实现 temperature、top-k、top-p、min-p 和重复惩罚，并解释约束（结构化）解码的原理。

## 为什么重要 {#why-this-matters-the-problem}

像 NVIDIA H100 SXM 这样的现代 GPU，每秒大约能做 $10^{15}$ 次 BF16 浮点运算。一个 80 亿参数的模型生成一个 token 大约需要 $2 \times 8\times10^{9} = 1.6\times 10^{10}$ FLOPs。两者相除，你可能以为单个用户能拿到约 60,000 token/s。而现实中，单个用户只能拿到一两百 token/s。这几百倍的差距去哪了？

答案是：文本生成的瓶颈不在算术运算，而在**把字节从 GPU 显存搬到计算单元**。理解这一点——并且能把它量化——就能解释推理引擎为什么长成现在这个样子：为什么它们拼命做批处理、为什么对 KV cache 如此执着、为什么量化能加速、为什么投机解码有效。本章给你第三部分通篇使用的术语和算术方法。

## 直觉 {#intuition}

把 GPU 想象成一个巨大的厨房：有成千上万个飞快的厨师（计算单元），却只有一扇通往食材库的窄门（显存带宽）。做一道菜（为一个用户生成一个 token），厨师们需要食材库里的*每一样*食材——也就是模型的全部权重。如果只点了一道菜，厨师们几乎所有时间都花在门口等食材，只有一瞬间在做菜。如果同时点了 200 道用同样食材的菜，每样食材搬过门就能用 200 次，厨师们终于忙起来了。

这就是 LLM 推理的全部故事：

- **预填充（prefill）**（读 prompt）就像一张 2,000 道菜的大单：所有 prompt token 一起处理，每个权重被复用很多次。厨师们很忙 → *计算受限*。
- **解码（decode）**（写回答）每一步只产出一个 token，因为 token $t+1$ 依赖 token $t$。只有一个用户时，每个权重搬来只做一道菜 → *显存带宽受限*。
- 把许多用户的解码步骤**批处理**到一起，就能把复用找回来。

## 工作原理 {#how-it-works}

### 自回归循环 {#the-autoregressive-loop}

语言模型定义了 $p_\theta(x_{t+1} \mid x_1, \dots, x_t)$：给定到目前为止的 token，输出下一个 token 在大小为 $V$ 的词表上的概率分布。生成就是重复三步：

1. 在当前序列上运行模型，得到下一个位置的 **logits** $z \in \mathbb{R}^{V}$。
2. 用**采样规则**把 logits 变成一个 token（贪心 argmax，或带 temperature、top-p 等的随机采样）。
3. 把 token 追加到序列后再重复，直到遇到结束符（EOS）或达到长度上限。

朴素做法下，第 1 步每次都要重新处理整个序列。3.2 章会讲到，通过缓存每一层的 key 和 value（即 **KV cache**），每一步只需要处理那*一个*新 token。有了 KV cache，生成分成两个阶段：

| 阶段 | 每个序列的输入 | 发生了什么 | 瓶颈 | 延迟指标 |
|---|---|---|---|---|
| **预填充（Prefill）** | 整个 prompt，$L_{\text{in}}$ 个 token，并行处理 | 为每个 prompt token 计算 key/value，产出第一个输出 token | 计算（大的矩阵–矩阵乘法） | **TTFT** |
| **解码（Decode）** | 1 个新 token | 读取全部权重 + KV cache，追加一条 K/V，产出下一个 token | 显存带宽（矩阵–向量乘法） | **TPOT / ITL** |

### 你真正需要的 GPU 基础 {#gpu-basics-you-actually-need}

GPU 有**内存层级**：靠近计算单元的少量极快内存，和离得较远的大量较慢内存。以 H100 SXM 为例（以下数字均为近似值，来自 NVIDIA 公开规格）：

| 层级 | 大小 | 带宽 / 速度 | 存放什么 |
|---|---|---|---|
| 寄存器 | 每个 SM 256 KB | 最快 | 每线程的值 |
| 共享内存 / L1（**SRAM**） | 每个 SM 最多约 228 KB 可用共享内存，共 132 个 SM | 非常高（聚合带宽数十 TB/s） | kernel 正在处理的矩阵分块 |
| L2 缓存 | 约 50 MB | 高 | 最近用过的数据 |
| **HBM**（HBM3） | 80 GB | **≈ 3.35 TB/s** | 权重、KV cache、激活值 |

**SM**（流式多处理器，streaming multiprocessor）是 GPU 约 132 个独立处理器中的一个。每个 SM 包含**张量核心（tensor core）**：一种硬件单元，一条指令就能完成一个小矩阵乘法（例如 16×16 的分块）。GPU 几乎所有的 FLOP/s 都来自张量核心：H100 SXM 在张量核心上的额定性能约为 **≈ 989 稠密 BF16 TFLOP/s** 和 **≈ 1,979 稠密 FP8 TFLOP/s**，而普通 FP32 算术只有约 67 TFLOP/s。（NVIDIA 宣传的数字往往是这些的两倍，因为它们假设了 2:4 结构化稀疏；我们始终使用稠密数字。）作为对比，A100 SXM 80 GB 的 HBM 带宽约 2.0 TB/s，稠密 BF16 约 312 TFLOP/s。

关键的比值：

$$
\frac{\text{peak FLOP/s}}{\text{memory bandwidth}} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35\times 10^{12}\ \text{B/s}} \approx 295\ \text{FLOP/byte} \quad (\text{H100 SXM, BF16}).
$$

换句话说：要让 H100 忙于计算而不是等数据，kernel 从 HBM 每读一个字节，就必须做大约 **295 次浮点运算**。

### 算术强度与 roofline 模型 {#arithmetic-intensity-and-the-roofline-model}

一次计算的**算术强度（arithmetic intensity）** $I$ 定义为

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to/from HBM}}.
$$

**roofline 模型**说，一个 kernel 能达到的最佳吞吐是

$$
\text{attainable FLOP/s} = \min\big(P_{\text{peak}},\; I \cdot B_{\text{mem}}\big),
$$

其中 $P_{\text{peak}}$ 是峰值算力（FLOP/s），$B_{\text{mem}}$ 是显存带宽（字节/秒）。在双对数坐标（FLOP/s 对 $I$）上画出来，就是一条斜线（“带宽屋顶”）和一条水平线（“算力屋顶”），它们在**拐点（ridge point）** $I^{\star} = P_{\text{peak}}/B_{\text{mem}}$ 处相交。$I < I^{\star}$ 的 kernel 是**显存受限**的：把算术算得再快也没用，只有少读字节才有用。$I > I^{\star}$ 的 kernel 则是**计算受限**的。

```text
 FLOP/s
   ^                 算力屋顶 (989 TFLOP/s)
   |            _______________________________
   |           /
   |          /  <- 拐点 I* ≈ 295 FLOP/B
   |         /
   |        /   带宽屋顶：斜率 = 3.35 TB/s
   |  decode (B=1, I≈1)         prefill (I≈ 数千)
   +----------------------------------------------> 算术强度（对数）
```

现在把它用到 Transformer 的主力——线性层 $Y = XW$ 上：权重 $W \in \mathbb{R}^{d \times d}$ 以 BF16 存储（每个 2 字节），输入 $X \in \mathbb{R}^{B \times d}$ 包含 $B$ 个 token 向量。

- FLOPs：$B \cdot d$ 个输出中的每一个都是长度为 $d$ 的点积（每项一次乘法加一次加法），所以共 $2Bd^2$ FLOPs。
- 字节数：读 $W$（$2d^2$）、读 $X$（$2Bd$）、写 $Y$（$2Bd$）：共 $2d^2 + 4Bd$。

$$
I(B) = \frac{2Bd^2}{2d^2 + 4Bd} \approx B \quad \text{when } B \ll d.
$$

换句话说：**线性层的算术强度大致等于一起处理的 token 数。** 预填充时 $B$ 是 prompt 的 token 数（几百到几千）→ 计算受限。单用户解码时 $B = 1$ → 约 1 FLOP/byte，比 H100 的拐点低了大约 300 倍。当 $B$ 个用户一起解码时 $I \approx B$，所以在 H100 上用 BF16，解码的线性层大约在 $B \approx 295$ 时变成计算受限。

!!! note "解码时的 attention 无法靠批处理提升"
    上面的论证针对的是权重矩阵，它们在 batch 内所有序列之间是**共享**的。而解码时的 attention 读取的是每个序列**各自**的 KV cache，无法共享。对标准多头注意力，它的强度不论批大小都停留在 1–2 FLOP/byte 左右（用分组查询注意力时约为 $g$ FLOP/byte，其中 $g$ 个查询头共享一个 KV 头）。这就是为什么长上下文即使在大批量下也会让解码变慢，也是 3.2 章专门讲 KV cache 的原因。

### 延迟指标 {#latency-metrics}

对于一个产生 $N$ 个输出 token 的请求：

- **TTFT**（time to first token，首 token 延迟）：从请求到达到第一个输出 token 的时间。包括排队 + 预填充。用户把它感知为“响应快不快”。
- **ITL**（inter-token latency，token 间延迟）：相邻两个输出 token 之间的间隔；这是一个分布，每个 token 一个值。
- **TPOT**（time per output token，每输出 token 时间）：一个请求内的平均间隔，$\text{TPOT} = \dfrac{\text{E2E} - \text{TTFT}}{N - 1}$。它的倒数就是每个用户的流式输出速度（“每用户 tokens/s”）。
- **E2E 延迟**：总时间，$\text{E2E} = \text{TTFT} + (N-1)\cdot\text{TPOT}$。
- **吞吐（Throughput）**：全系统每秒输出 token 数（或每秒请求数），对所有用户求和。
- **SLO**（service-level objective，服务等级目标）：形如“P99 TTFT ≤ 2 s 且 P99 TPOT ≤ 50 ms”的目标。一定要说明分位数；平均值会掩盖长尾。
- **Goodput**：**满足** SLO 的请求速率（DistServe 的定义）。一个系统可能吞吐很高、goodput 却很糟——因为它是靠让延迟爆炸换来的吞吐。

吞吐和延迟互相权衡：更大的批能提高吞吐（每次权重读取被更多复用），但每一步的工作量更大，所以 TPOT 会上升；而准入新的预填充会打断正在运行的解码。3.3 章里所有的调度技术，都是在这条权衡曲线上移动，或者把这条曲线向外推。

### 采样：从 logits 到 token {#sampling-from-logits-to-a-token}

给定 logits $z \in \mathbb{R}^V$，模型的分布是 $p_i = \operatorname{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$。常见的变换，大致按以下顺序应用：

- **重复惩罚（Repetition penalty）**（来自 CTRL 论文）：对上下文中已出现的每个 token $i$，若 $z_i > 0$ 则把 $z_i$ 除以 $\theta > 1$，若 $z_i < 0$ 则乘以 $\theta$。两种做法都会让已出现过的 token 概率变低。（OpenAI 风格的 API 则是*减去*“频率惩罚 × 出现次数”，并在出现次数非零时减去一个“存在惩罚”。）
- **Temperature** $T$：使用 $\operatorname{softmax}(z/T)$。$T<1$ 让分布更尖锐，$T>1$ 让分布更平坦，$T \to 0$ 就是贪心 argmax。
- **Top-k**：只保留概率最高的 $k$ 个 token，重新归一化。
- **Top-p（nucleus）**：按概率对 token 排序，保留累积概率 $\ge p$ 的最小前缀。候选数量会随模型的自信程度自适应变化。
- **Min-p**：保留 $p_i \ge p_{\min} \cdot \max_j p_j$ 的 token。阈值随最高概率 token 的概率缩放：模型自信时严格，不自信时宽松。

过滤的做法是：在最后一次 softmax 之前把被剔除的 logits 设为 $-\infty$，然后采样。

### 结构化 / 约束解码 {#structured-constrained-decoding}

应用常常需要可解析的输出：符合某个 schema 的合法 JSON、一次函数调用、一个正则表达式。**约束解码（constrained decoding）**通过在每一步屏蔽所有会让输出无法合法完成的 token 来保证这一点：

1. 把约束（正则、JSON schema 或上下文无关文法）编译成自动机——正则语言用有限状态机，带嵌套的文法用下推自动机。
2. 每一步，计算词表中哪些 token 能让自动机保持在合法状态。这件事有些微妙，因为 token 是多字符的字符串：token `"},` 会让自动机一下前进三个字符。
3. 把不允许的 token 的 logits 设为 $-\infty$，采样，再推进自动机。

代价在于：要对 100K+ 词表的每一步都算出允许 token 的掩码，同时还不能拖慢解码。Outlines 预先计算了从 FSM 状态到允许 token 的索引；XGrammar 则把词表分成两部分：一部分的合法性可以按文法位置预先算好，另一部分较小、必须在运行时检查，并且让掩码计算与 GPU 前向计算重叠进行。掩码会改变分布（它在合法 token 上重新归一化），所以约束输出的质量可能与无约束输出不同；好的 schema 会在自由文本字段中给模型留出“思考”的空间。

## 代码 {#code}

下面是一个自包含的生成循环，配了一个玩具因果 Transformer（随机权重——重点是循环结构，而不是生成的文本）。注意它有多朴素：每一步都在整个序列上重新运行模型。3.2 章会修正这一点。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TinyLM(nn.Module):
    """单层因果 Transformer：足以展示生成循环。"""
    def __init__(self, vocab=100, d=64, heads=4):
        super().__init__()
        self.emb, self.h = nn.Embedding(vocab, d), heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):                      # ids: [batch, seq]
        x = self.emb(ids)                        # [B, S, d]
        B, S, d = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # [B, heads, S, d_head]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, d))
        x = x + self.mlp(x)
        return self.head(x)                      # logits: [B, S, vocab]

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=10, eos_id=None):
    ids = prompt_ids                             # [1, L_in]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]            # 只有最后一个位置有用
        next_id = logits.argmax(-1, keepdim=True)   # 贪心；见下面的 sample()
        ids = torch.cat([ids, next_id], dim=1)   # 追加后重复
        if eos_id is not None and next_id.item() == eos_id:
            break
    return ids

torch.manual_seed(0)
model = TinyLM().eval()
print(generate(model, torch.tensor([[1, 2, 3]])))
```

下面的采样器实现了“工作原理”中的每一条规则。它处理单个序列的 logits；真实的引擎则是在 GPU 上批量运行同样的逻辑。

```python
import torch

def sample(logits, prev_ids=(), temperature=1.0, top_k=0, top_p=1.0,
           min_p=0.0, repetition_penalty=1.0, generator=None):
    """logits: [V] 浮点张量。返回采样得到的 token id（int）。"""
    z = logits.clone().float()
    # 1) 重复惩罚（CTRL 风格），作用于上下文中已出现的 token
    if repetition_penalty != 1.0 and len(prev_ids) > 0:
        idx = torch.tensor(sorted(set(prev_ids)))
        zi = z[idx]
        z[idx] = torch.where(zi > 0, zi / repetition_penalty, zi * repetition_penalty)
    # 2) temperature（T -> 0 即贪心）
    if temperature == 0:
        return int(z.argmax())
    z = z / temperature
    # 3) top-k
    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]
        z[z < kth] = float("-inf")
    probs = torch.softmax(z, dim=-1)
    # 4) min-p：丢弃概率低于 min_p * (最大概率) 的 token
    if min_p > 0:
        z[probs < min_p * probs.max()] = float("-inf")
        probs = torch.softmax(z, dim=-1)
    # 5) top-p（nucleus）：保留累积概率 >= top_p 的最小集合
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = cum - sp >= top_p          # 前面已累积的概率质量 >= top_p 的 token
        z[si[drop]] = float("-inf")
        probs = torch.softmax(z, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))

logits = torch.tensor([3.0, 2.5, 1.0, 0.5, -1.0])
g = torch.Generator().manual_seed(0)
print([sample(logits, prev_ids=[0], temperature=0.7, top_p=0.9,
              repetition_penalty=1.3, generator=g) for _ in range(10)])
```

## 实例演算 {#worked-example}

**问题。** 在一张 H100 SXM 上部署 Llama-3-8B（≈ 8.0 B 参数，BF16）。(a) 单用户的解码速度上限是多少？(b) 批大小多大时线性层变成计算受限？(c) 2,000 token 的 prompt 预填充要多久？

**(a) Batch 1 解码。** 每个解码步必须把每个权重读一遍：$8.0\times10^9 \times 2\ \text{B} = 16\ \text{GB}$。按 ≈ 3.35 TB/s 计算：

$$
t_{\text{step}} \ge \frac{16 \times 10^9}{3.35 \times 10^{12}} \approx 4.8\ \text{ms} \;\Rightarrow\; \le \approx 209\ \text{tokens/s}.
$$

所需的计算量只有 $2 \times 8\times10^9 = 1.6\times10^{10}$ FLOPs，只需 $1.6\times10^{10}/989\times10^{12} \approx 0.016$ ms。GPU 的算术单元约 99.7% 的时间是空闲的。真实引擎大概能达到这个带宽上限的 70–85%，而 KV cache 还会再增加字节数。

**(b) 临界批大小。** 当 $B$ 个用户一起解码时，权重的字节数每步仍约为 16 GB，而 FLOPs 增长为 $1.6\times10^{10} \cdot B$。每步时间大致是

$$
t(B) \approx \max\left(\frac{16\times10^9}{3.35\times10^{12}},\; \frac{1.6\times10^{10}\,B}{989\times10^{12}}\right),
$$

两项在 $B \approx 4.8\ \text{ms} \times 989\times10^{12} / 1.6\times10^{10} \approx 297$ 处相等。在约 300 个并发序列以下，多加一个用户对步时间几乎没有影响（吞吐几乎随 $B$ 线性增长）；超过之后，步时间随 $B$ 增长。实际中会更早到达这个临界点，因为真实 kernel 达不到峰值 FLOP/s，也因为 KV cache 的读取（随 $B$ 和上下文长度增长）会增加字节数。这就是为什么推理引擎要尽量**把批开到显存允许的最大**。

**(c) 预填充。** 一个 2,000 token 的 prompt 大约需要 $2 \times 8\times 10^9 \times 2000 = 3.2\times10^{13}$ FLOPs（忽略在这个长度下很小的 attention）。按峰值算是 32 ms；按实际 40–60% 的峰值利用率算，大约 55–80 ms。2,000 token 的预填充消耗的 GPU 时间约等于 ~15 个解码步——却只产出一个输出 token。这种不对称就是为什么把预填充混进解码批时必须谨慎处理（3.3 章，分块预填充）。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- 每个推理引擎（vLLM、SGLang、TensorRT-LLM）都会报告 TTFT、TPOT/ITL 和吞吐，它们的 benchmark 工具会扫描并发度或请求速率（3.8 章）。
- 同样的带宽论证解释了三个行业趋势：**FP8 和 FP4 权重**（每个权重的字节数更少，3.6 章）、**MoE 模型**（每个 token 只读被激活专家的权重）、**GQA/MLA attention**（每个 token 的 KV 字节更少，3.2 章）。
- 生产环境的 SLO 通常按产品档位用 P90/P99 TTFT 和 TPOT 来表述：交互式聊天产品关心 TTFT 和稳定的流式速度；批量/离线任务只关心吞吐和成本。
- 结构化输出是标准的 API 功能（JSON 模式 / JSON-schema 式“structured outputs”）；开源引擎自带 XGrammar、Outlines 和 llguidance 等文法后端。
- 聊天模型的默认采样通常是 temperature 约 0.6–1.0、top-p 约 0.9–0.95；推理模型的提供方常会公布推荐设置——请照着用，因为这些模型在贪心解码下可能陷入循环或质量下降。

## 常见误区 {#common-pitfalls-misconceptions}

- **“更快的 GPU（更多 FLOP/s）会让单用户解码更快。”** 基本不会：单用户解码受显存带宽限制。比较 GPU 时，延迟看 TB/s，预填充和大批量吞吐看 FLOP/s。
- **混淆 TPOT 和 ITL。** TPOT 是单个请求的平均值；ITL 是每个间隔的分布。一个系统可能 TPOT 均值很好，但由于周期性的卡顿（例如一个长预填充阻塞了解码），P99 ITL 很糟。
- **只报吞吐不报延迟。** 只要不断增大批大小，直到延迟无法接受，吞吐总能被抬高。应该报告 goodput，或在给定 SLO 下的吞吐。
- **在 roofline 计算中使用稀疏 TFLOP/s。** 稠密矩阵乘法用不上 2:4 稀疏的吞吐。
- **不清楚 top-p 与 temperature 谁先谁后。** 不同库的顺序不同，结果也不同。要了解你所用引擎的顺序。
- **以为约束解码是免费的。** 在大词表和复杂文法上，掩码计算如果不与 GPU 重叠，可能成为 CPU 瓶颈。

## 本章要点 {#key-takeaways}

- 生成 = 对 prompt 做一次并行的**预填充**，然后是每步一个 token 的串行**解码**循环。
- 线性层的算术强度 ≈ 一起处理的 token 数。batch 1 的解码约为 1 FLOP/byte，远低于 H100 约 295 FLOP/byte 的拐点 → **受带宽限制**。
- Batch 1 解码速度 ≤ 带宽 ÷ 每步读取的字节数；批处理会摊薄权重读取，直到大约 $B \approx P_{\text{peak}}/B_{\text{mem}}$。
- **按分位数**测量 TTFT、TPOT/ITL、E2E 和吞吐，并在 SLO 约束下优化 goodput。
- 采样规则（temperature、top-k/p、min-p、惩罚项）和文法掩码，本质上都是最终 softmax 并采样之前对 logits 的变换。

## 自测 {#check-your-understanding}

??? question "某个引擎从 A100（≈2.0 TB/s，≈312 BF16 TFLOP/s）升级到 H100 SXM（≈3.35 TB/s，≈989 TFLOP/s）。batch-1 解码大约快多少？大批量预填充呢？"
    batch-1 解码受带宽限制，所以大约快 3.35/2.0 ≈ 1.7 倍。预填充受计算限制，所以最多约快 989/312 ≈ 3.2 倍（实际会略低一些）。注意 H100 的拐点（~295 FLOP/B）高于 A100 的（~156 FLOP/B）：更新的 GPU 需要*更大*的批才能变成计算受限。

??? question "为什么解码时 attention 的算术强度无法通过批处理提高，而 MLP 可以？"
    MLP 的权重矩阵被 batch 里的每个序列共享，所以读一次就能服务 $B$ 个 token。而 attention 中，每个序列关注的是它*自己*的 key 和 value，没有可跨序列共享的东西，所以每读一个 KV 字节只用于一个查询（用 GQA 时是 $g$ 个查询）。

??? question "一个请求生成了 201 个 token；TTFT 为 0.5 s，E2E 为 10.5 s。TPOT 和每用户流式速度是多少？"
    TPOT = (10.5 − 0.5)/(201 − 1) = 0.05 s = 50 ms，即每用户约 20 tokens/s。

??? question "min-p = 0.1，最高概率的 token 概率为 0.6。哪些 token 会保留？如果最高概率的 token 概率是 0.05 呢？"
    第一种情况下，$p_i \ge 0.06$ 的 token 会保留。第二种情况下，阈值是 0.005，所以会保留多得多的 token——min-p 恰好在模型不确定时放宽候选集。

??? question "为什么 JSON 约束必须在 token 层面执行，而不是检查生成完的字符串？"
    事后检查就得拒绝并重新生成（代价无界，也没有保证）。逐步掩码能保证每个前缀始终可以被补全，所以最终输出天然合法。之所以必须在 token 层面，是因为模型输出的是多字符 token，所以每个 token 都要作为字符串对照自动机检查。

## 延伸阅读 {#further-reading}

| 论文 / 资源 | 为什么读 |
|---|---|
| :material-star: [LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook) | 面向实践者的第三部分全部概念概览，附指标定义。 |
| :material-star: [Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/) | 对 Transformer 推理中 FLOPs、字节数和延迟做估算的经典文章。 |
| :material-book-open-variant: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Pope et al., 2022 | 对大规模推理中显存/计算/通信权衡的严谨分析（TPU 上的 PaLM）。 |
| :material-book-open-variant: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al., 2019 | 提出 nucleus（top-p）采样，并解释了为什么贪心/束搜索解码会退化。 |
| :material-book-open-variant: [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082) — Nguyen et al., 2024 | min-p 规则及其在高 temperature 下的评估。 |
| :material-book-open-variant: [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) — Keskar et al., 2019 | 许多引擎所用的乘法式重复惩罚的出处。 |
| :material-book-open-variant: [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) — Willard & Louf, 2023 | Outlines 的方法：正则/文法 → FSM → 预计算的 token 索引。 |
| :material-new-box: [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100) — Dong et al., 2024 | vLLM 和 SGLang 所用的快速上下文无关文法解码。 |
