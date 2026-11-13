# 3.2 KV cache

!!! abstract "本章你将学到"
    - 为什么不用缓存的生成会浪费平方级的计算，以及 **KV cache** 到底存了什么。
    - 实现一个向 KV cache 追加的解码步，并验证它与完整重算的结果一致。
    - 推导 KV cache 大小公式，并把它用于 Llama-3-70B（8K 和 128K 上下文）以及 MHA、GQA、MQA、MLA 的对比。
    - 解释为什么是 KV 内存——而不是算力——限制了批大小，以及为什么长上下文会让解码变慢。
    - 从直觉层面描述压缩 cache 的主要手段：量化、驱逐（StreamingLLM、H2O、SnapKV）和架构改动。

## 为什么重要 {#why-this-matters-the-problem}

在 [3.1 章的生成循环](01-inference-fundamentals.md#code)里，每一步都在整个序列上重新运行模型。为了生成 token $t+1$，我们把 token $1 \dots t$ 的隐状态又算了一遍，尽管它们在上一步已经算过，而且**不会再变**：在因果（decoder-only）Transformer 中，位置 $i$ 只关注位置 $\le i$，所以它的表示从不依赖后面的 token。

用这种方式在 $L$ token 的 prompt 之后生成 $N$ 个 token，大约要做 $\sum_{t=L}^{L+N} t \approx N L + N^2/2$ 次 token 前向计算，而不是 $L + N$ 次。对于 1,000 token 的 prompt 加 1,000 个输出 token，这就是约 150 万次对约 2,000 次——浪费了 750 倍。KV cache 消除了这种浪费，但同时它也成为**推理系统中最大的 GPU 显存消耗者**。

## 直觉 {#intuition}

看看 attention 需要什么。位置 $t$ 的新 token 关注过去时，它需要**自己的 query**，以及**每个更早 token 的 key 和 value**——在每一层、每一个头里都是如此。更早 token 的其他东西（它们的 MLP 激活、它们的 query）再也不会被用到。

所以我们给每一层准备一个笔记本：每处理一个 token，就把它的 key 和 value 向量记下来。下一步只为那一个新 token 计算 Q、K、V，把新的 K 和 V 追加到笔记本里，然后在整个笔记本上做 attention。笔记本每层每个 token 增长一条：这个笔记本就是 KV cache。

```text
步骤 t:   新 token x_t ──▶ q_t, k_t, v_t   （每层每个头各一行）
                                   │    │
KV cache（第 ℓ 层）:  K = [k_1 k_2 ... k_{t-1}] + k_t   ──▶ attention(q_t, K, V)
                      V = [v_1 v_2 ... v_{t-1}] + v_t
```

## 工作原理 {#how-it-works}

### Attention 回顾 {#attention-recap}

对于一个头，头维度为 $d_h$，一个 token 的隐状态 $x \in \mathbb{R}^{d}$ 被投影为 $q = x W_Q$、$k = x W_K$、$v = x W_V$，每个都在 $\mathbb{R}^{d_h}$ 中。把所有 key 和 value 堆叠成 $K, V \in \mathbb{R}^{t \times d_h}$，最新 token 的输出是

$$
o_t = \operatorname{softmax}\!\left(\frac{q_t K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{d_h}.
$$

换句话说：用新的 query 对每个已存的 key 打分，把分数变成权重，再用这些权重对已存的 value 求加权平均。只有 $q_t$ 是新的；$K$ 和 $V$ 就是 cache 再追加一行。

### 带 cache 的预填充与解码 {#prefill-and-decode-with-a-cache}

- **预填充。** 用因果掩码，一次并行地把 prompt $x_{1..L}$ 跑过模型。顺带地，每一层会把每个 KV 头的 $K, V \in \mathbb{R}^{L \times d_h}$ 写入 cache。输出：位置 $L$ 的 logits → 第一个生成的 token。
- **解码。** 对每个新 token：做 embedding，在每一层计算 $q_t, k_t, v_t$（形状 `[batch, heads, 1, d_h]`），把 $k_t, v_t$ 追加到 cache（现在是 `[batch, kv_heads, t, d_h]`），做 attention，再继续经过 MLP。现在每一步的开销只有在 attention 内部才与上下文长度成正比（读取 $K,V$）；所有矩阵乘法只处理一个 token。

### 估算 cache 大小 {#sizing-the-cache}

定义：

- $n_{\text{layers}}$ = Transformer 层数，
- $n_{\text{kv}}$ = **key/value 头**的数量（经典多头注意力中等于查询头数，GQA 中更少，MQA 中为 1），
- $d_h$ = 头维度，
- $b$ = 每个存储元素的字节数（BF16/FP16 为 2，FP8/INT8 为 1），
- $L$ = 上下文中的 token 数，$B$ = batch 中的序列数。

每个 token 在每一层的每个 KV 头上存一个长度为 $d_h$ 的 key **和**一个 value 向量：

$$
\text{KV bytes per token} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_h \cdot b,
\qquad
\text{KV bytes total} = B \cdot L \cdot (\text{KV bytes per token}).
$$

换句话说：cache 随上下文长度和批大小**线性增长**，而它每个 token 的成本完全由 attention 架构决定。

### 为什么 KV 内存限制批大小（和速度） {#why-kv-memory-limits-batch-size-and-speed}

GPU 显存里装着权重、几 GB 的激活值/工作区，以及 KV cache。权重之后剩下的就是 KV 预算，能同时运行的序列数是

$$
B_{\max} \approx \frac{\text{KV budget}}{L_{\text{avg}} \cdot \text{KV bytes per token}}.
$$

3.1 章说明了吞吐在几百个序列之内几乎随 $B$ 线性增长。如果 KV 内存把 $B$ 限制在比如 20，你就浪费了 GPU 大部分算力。所以**压缩 KV cache 能直接提高吞吐**。

还有第二个效应。每个解码步都会把每个序列的整个 cache 读一遍（对所有过去的 token 做 attention）。因此每步读取的字节数大约是

$$
\text{bytes/step} \approx \underbrace{\text{weight bytes}}_{\text{shared by batch}} + \underbrace{B \cdot L \cdot \text{KV bytes per token}}_{\text{not shared}},
$$

在长上下文下第二项占主导：即使模型权重不变，解码的 TPOT 也会随上下文长度增长。

### 压缩 cache，直觉层面 {#shrinking-the-cache-at-intuition-level}

**1. 改架构（最稳健）。** 这些内容在[第一部分，现代 Transformer 块](../part1/04-modern-block.md)中讲过：

- **MQA**（multi-query attention，多查询注意力）：所有查询头共享同一个 K/V 头。$n_{\text{kv}}=1$。
- **GQA**（grouped-query attention，分组查询注意力）：查询头被分成 $n_{\text{kv}}$ 组，每组共享 K/V；Llama-3-70B 用 8 个 KV 头对应 64 个查询头（比 MHA 小 8 倍）。
- **MLA**（multi-head latent attention，多头潜在注意力，DeepSeek-V2/V3）：每层每个 token 只缓存一个压缩后的潜向量（外加一个很小的解耦 RoPE key），再通过学出来的上投影矩阵重建每个头的 key/value，这些上投影在推理时可以被吸收进查询/输出投影中。
- 部分层使用**滑动窗口 / 局部注意力**：这些层只保留最近 $W$ 个 token，所以它们的 cache 有上界。
- **混合模型**：把一部分 attention 层换成线性注意力或状态空间层，它们的状态大小是固定的。

**2. 量化 cache。** 把 K 和 V 存成 FP8（BF16 cache 的一半）或更低精度。FP8 KV 在各推理引擎中已被广泛支持。降到 2–4 bit 需要小心，因为 key 有**离群通道**——某些维度在不同 token 上持续偏大。KIVI 的观察是：对 key 按**通道**量化（让每个通道有自己的 scale），对 value 按 **token** 量化，就能做到 2-bit KV 而质量损失很小。3.6 章会讲量化的数学。

**3. 驱逐条目（有损）。** 只保留重要的 token：

- **StreamingLLM（attention sink，注意力汇聚点）。** 模型无论内容如何，都会把很大的注意力权重放在最开头的几个 token 上——它们像一个“汇聚点”，softmax 必须把概率质量放在某处时就倒进去。把它们驱逐掉会破坏模型。保留前约 4 个 token **加上一个最近 token 的滑动窗口**，就能在任意长的流上稳定生成（但模型无法回忆被驱逐的中间内容）。
- **H2O（heavy-hitter oracle）。** 少量 token 会获得大部分累积注意力。保留这些“重击者（heavy hitter）”加上最近的 token；在固定预算下驱逐其余的。
- **SnapKV。** 预填充之后，利用 prompt 最后几个 token 的 attention（一个“观察窗口”，通常包含用户的问题），逐头投票选出较早位置中哪些重要；保留它们（并做一些邻域池化），丢弃其余。对长 prompt 只压缩一次，之后正常解码。

驱逐方法的吸引力在于不需要重新训练，但它们**有损且依赖任务**：针对被驱逐文本的大海捞针式问题就答不出来了。2025 年的前沿模型选择了架构层面的解决方案。

## 代码 {#code}

下面是带 KV cache 的单个 attention 层，外加一个检查：缓存解码的输出与在完整序列上重算 attention 一致。注释里标出了形状；GQA 通过重复 KV 头来支持。

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)

B, d_model, n_q, n_kv = 2, 64, 8, 2          # 8 个查询头共享 2 个 KV 头 (GQA)
d_h = d_model // n_q
Wq = torch.randn(d_model, n_q * d_h) / d_model**0.5
Wk = torch.randn(d_model, n_kv * d_h) / d_model**0.5
Wv = torch.randn(d_model, n_kv * d_h) / d_model**0.5

def project(x):                                   # x: [B, S, d_model]
    S = x.shape[1]
    q = (x @ Wq).view(B, S, n_q, d_h).transpose(1, 2)   # [B, n_q,  S, d_h]
    k = (x @ Wk).view(B, S, n_kv, d_h).transpose(1, 2)  # [B, n_kv, S, d_h]
    v = (x @ Wv).view(B, S, n_kv, d_h).transpose(1, 2)
    return q, k, v

def attend(q, k, v, causal):
    rep = n_q // n_kv                                   # 为 GQA 扩展 KV 头
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

class KVCache:
    def __init__(self):
        self.k = self.v = None                            # [B, n_kv, t, d_h]
    def append(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

x = torch.randn(B, 12, d_model)                           # 8 个 prompt + 4 个“生成的”
cache = KVCache()

# 预填充：一次处理 8 个 prompt token，填充 cache。
q, k, v = project(x[:, :8])
k_all, v_all = cache.append(k, v)
outs = [attend(q, k_all, v_all, causal=True)]

# 解码：每步一个 token；只计算它的 q/k/v。
for t in range(8, 12):
    q, k, v = project(x[:, t:t + 1])                      # [B, heads, 1, d_h]
    k_all, v_all = cache.append(k, v)                     # cache 增长到 t+1
    outs.append(attend(q, k_all, v_all, causal=False))    # 1 个 query 看到所有 key

cached = torch.cat(outs, dim=2)
q, k, v = project(x)                                      # 参照：完整重算
full = attend(q, k, v, causal=True)
print("cache shape:", tuple(cache.k.shape), "max diff:", (cached - full).abs().max().item())
```

注意解码时 `causal=False`：这个唯一的新 query 就是最后一个位置，所以它可以关注每个已缓存的 key——不需要掩码。

下面是公式的一个小计算器，供后面的实例演算使用：

```python
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_elem=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_elem

GiB = 2**30
llama70 = kv_bytes_per_token(80, 8, 128)                     # GQA, BF16
print("Llama-3-70B  :", llama70 / 1024, "KiB/token")
for ctx in (8192, 131072):
    print(f"  {ctx:>6} tokens -> {ctx * llama70 / GiB:.1f} GiB per sequence")
print("70B as MHA   :", kv_bytes_per_token(80, 64, 128) / 2**20, "MiB/token")
print("70B as MQA   :", kv_bytes_per_token(80, 1, 128) / 1024, "KiB/token")
mla = (512 + 64) * 61 * 2           # DeepSeek-V3：潜向量 512 + RoPE key 64，61 层
print("DeepSeek-V3 MLA:", round(mla / 1024, 1), "KiB/token")
```

## 实例演算 {#worked-example}

**Llama-3-70B** 有 80 层、64 个查询头、**8 个 KV 头**（GQA），头维度为 128。

*每个 token（BF16）：* $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ 字节 $= 320$ KiB。

*每个序列：* 8K token（$8{,}192$）时：$8192 \times 320\ \text{KiB} = 2.5$ GiB。128K token（$131{,}072$）时：$40$ GiB——仅**一个**用户的 cache 就占了一张 80 GB H100 的一半。

*架构对比*（同样 80 层 × 头维度 128 的形状，BF16）：

| Attention | KV 头数 | 每 token 的 KV | 128K token 的序列 |
|---|---|---|---|
| MHA | 64 | 2.5 MiB | 320 GiB |
| GQA（Llama-3-70B 实际采用） | 8 | 320 KiB | 40 GiB |
| MQA | 1 | 40 KiB | 5 GiB |

对于 **MLA**，DeepSeek-V3（61 层）每层每个 token 缓存一个 512 维的潜向量加一个 64 维的 RoPE key：$(512 + 64) \times 61 \times 2 = 70{,}272$ 字节，BF16 下约 **68.6 KiB/token**——128K token 的序列约 8.6 GiB，尽管 DeepSeek-V3 有 128 个 attention 头。如果换成 DeepSeek-V3 同样 128 个头的朴素 MHA 设计（key 维度 128+64，value 维度 128），每个 token 需要 $128 \times (192 + 128) \times 61 \times 2 \approx 4.8$ MiB，约大 71 倍。

*批大小上限。* 在 8× H100 80 GB 上用张量并行部署 BF16 的 Llama-3-70B（每张 GPU 持有 1/8 的权重和 8 个 KV 头中的 1 个）。总显存 640 GB。假设为激活值、工作区和 CUDA graph 预留 10%（可用 576 GB），权重占 ≈ 140 GB。则 KV 预算 ≈ 436 GB ≈ 406 GiB：

$$
\frac{406\ \text{GiB}}{320\ \text{KiB/token}} \approx 1.33 \text{ million tokens}.
$$

这相当于 8K 上下文时约 **162 个序列**，而 128K 时只有 **10 个序列**。由于吞吐随批大小增长，长上下文推理在每 token 上要贵得多。

*每个解码步的字节数*，162 个 8K 序列：权重 140 GB，KV $\approx 436$ GB。聚合带宽 8 × 3.35 TB/s ≈ 26.8 TB/s，读权重需要 ≈ 5.2 ms，但读 KV cache 需要 ≈ 16 ms。在长上下文下，决定 TPOT 的是 KV cache，而不是权重。FP8 KV 能把这 16 ms 减半。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **GQA 是默认选择**，稠密开源模型（Llama 3、Qwen、Mistral 系列）都用它；DeepSeek-V2/V3/R1 和 Kimi K2 用 **MLA**；2025 年的一些模型把全注意力层与滑动窗口或线性注意力层混合使用，以限制 KV 的增长（gpt-oss 交替使用稠密与带状窗口 attention 层）。
- **FP8 KV cache** 是 vLLM、SGLang 和 TensorRT-LLM 中常见的生产设置；它让可缓存的 token 数翻倍。更低比特的 KV 使用得更有选择性。
- 引擎在启动时把 KV cache 作为一整块大内存池分配出来（例如 vLLM 的 `gpu_memory_utilization`），再切分成块（3.3 章）。启动日志会告诉你能放下多少 token——记得看。
- KV cache 越来越被当作可复用的资产：前缀缓存（3.3 章）以及 CPU 内存/SSD 上的分层 KV 存储（3.7 章）避免重复计算同样的预填充。
- 事后驱逐（H2O、SnapKV 等）仍主要是研究和长上下文细分场景的工具；生产系统更偏好无损技术，加上为小 cache 设计的架构。

## 常见误区 {#common-pitfalls-misconceptions}

- **忘记系数 2**（key *和* value），或者在公式里用了*查询*头数而不是 KV 头数。
- **忘记 MLA 的特殊形状。** MLA 的 cache 不是 $2 \cdot n_{\text{kv}} \cdot d_h$；而是每层每个 token 一个潜向量（加 RoPE key），没有系数 2，也没有头数。
- **以为张量并行会缩小总 KV。** TP 把 KV 头分摊到各 GPU 上，所以每张 GPU 上的 KV 变小，但总量不变。当 KV 头数少于 GPU 数时（例如 8 个 KV 头、TP=16），头必须被复制，总 KV 内存反而*增加*。
- **在延迟估算中忽略 KV 读取。** 在长上下文下，KV 带宽可能超过权重带宽（见实例演算）。
- **把驱逐当作免费的。** 它会悄悄丢掉信息；一定要在检索密集的任务上评估。
- **实现滑动窗口时丢掉 attention sink。** 一旦最开头的 token 滑出窗口，质量就会崩溃。

## 本章要点 {#key-takeaways}

- KV cache 存储每个过去 token 在每一层的 key 和 value，所以每个解码步只需处理新 token。
- 每个 token 的 $\text{KV bytes} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_h \cdot b$，再乘以 token 数和序列数。
- Llama-3-70B：BF16 下 320 KiB/token → 每个序列 8K 时 2.5 GiB，128K 时 40 GiB。
- KV 内存限制批大小（从而限制吞吐），长上下文下 KV 读取主导解码时间。
- 解决办法，按稳健程度从高到低：架构（GQA/MLA/窗口/混合）、量化（先 FP8，再更低）、驱逐（sink、H2O、SnapKV）。

## 自测 {#check-your-understanding}

??? question "Llama-3-8B 有 32 层、8 个 KV 头、头维度 128。它每个 token 的 KV cache 在 BF16 和 FP8 下分别是多少？"
    $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ 字节 = BF16 下 128 KiB；FP8 下 64 KiB。

??? question "为什么我们不需要缓存 query？"
    一个 query 只会被它自己的 token 在生成它的那一步用一次。后面的 token 关注的是更早 token 的 key 和 value，从不关注它们的 query。

??? question "在上下文长度不变的情况下把批大小翻倍：每个解码步读取的权重字节数和 KV 字节数会怎样？"
    权重字节数不变（被 batch 共享）；KV 字节数翻倍。所以批处理摊薄的是权重，而不是 KV cache。

??? question "StreamingLLM 保留 4 个 sink token + 1,020 个最近的 token。模型能回答一个关于 5,000 个 token 之前陈述的事实的问题吗？"
    不能。被驱逐 token 的 K/V 已经没了；StreamingLLM 提供的是在无限长流上的稳定*流畅度*，而不是长程*记忆*。

??? question "为什么 KIVI 对 key 按通道量化，而对 value 按 token 量化？"
    key 有离群通道（少数维度在不同 token 上持续偏大），所以按通道设 scale 能把它们隔离开。value 没有这种通道结构，而按 token 量化契合它们被组合的方式（每个输出都是 value 行的加权和）。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | 发现了 attention sink；sink + 滑动窗口可实现稳定的无限长度生成。启发了 gpt-oss 中的可学习 sink。 |
| :material-book-open-variant: [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Zhang et al., 2023 | 驱逐累积注意力很少的 KV 条目；保留 heavy hitter + 最近的 token。 |
| :material-book-open-variant: [SnapKV](https://arxiv.org/abs/2404.14469) — Li et al., 2024 | 利用 prompt 末尾的观察窗口，逐头选出重要的 prompt 位置。 |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | key 按通道、value 按 token → 损失很小的 2-bit KV cache。 |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | 多查询注意力：第一篇从架构上解决 KV 带宽问题的论文。 |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | 分组查询注意力，当今稠密模型的默认选择。 |
| :material-book-open-variant: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | 提出多头潜在注意力（MLA）和权重吸收技巧。 |
| :material-book-open-variant: [Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/) | KV cache 的大小估算及其对批大小和延迟的影响，附实算数字。 |
