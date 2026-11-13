# 3.5 投机解码（Speculative Decoding）

!!! abstract "本章你将学到"
    - 为什么解码（decode）阶段会让算力闲置，以及如何用廉价的**草稿（draft）**加上目标模型的一次**验证（verification）**前向，把这部分闲置算力变成额外的 token。
    - 接受/拒绝规则，以及输出分布与目标模型**完全一致**的**证明**。
    - 每一步期望产出的 token 数，如何随接受率（acceptance rate）$\alpha$ 与草稿长度 $k$ 变化，以及由此得到的加速比。
    - 草稿的几种选择：小模型、n-gram / prompt lookup、Medusa heads、EAGLE、MTP heads；以及**树形验证（tree verification）**。
    - 什么时候投机解码不再有效（大 batch、计算受限的场景）。

## 为什么重要 {#why-this-matters-the-problem}

3.1 章讲过，在小 batch 下，一步解码是带宽受限的：GPU 要花几毫秒把全部权重从 HBM 里读出来，而真正的计算只占其中极小一部分时间。关键在于，一次前向处理 **5 个 token** 的耗时和处理 **1 个 token** 几乎一样——两种情况下权重都只读一遍，算术强度仍然远低于拐点（ridge point）。

但解码是串行的：在知道第 $t+1$ 个 token 之前，没法算第 $t+2$ 个。投机解码靠**猜**来打破这个限制。先由一个廉价的机制提出接下来的 $k$ 个 token，大模型用一次前向把它们全部检查一遍（因为是访存受限，这次前向的耗时和一次普通解码差不多），然后保留最长的正确前缀。如果猜得大多是对的，一次昂贵的前向就能产出好几个 token。只要做得严谨，它**只改变速度，绝不改变输出分布**。这份"免费的午餐"，正是每个生产级推理引擎都支持它、输出很长的推理模型也都依赖它的原因。

## 直觉 {#intuition}

想象一位资深作者（目标模型）和一位手脚很快的初级助理（草稿）。助理先敲出接下来的几个词，作者一眼扫过，要么点头（"对，对，对——不对，这个词应该是 *这个*"），要么把第一个错词改掉。对作者来说，"读"比一个词一个词地"写"快得多。如果助理有 80% 左右的概率猜对，作者每扫一眼就能往前推进好几个词。

微妙之处在于采样。贪心解码时，"正确"就是"等于目标模型的 argmax"。但随机采样时并不存在唯一正确的 token，所以我们需要一个概率式的接受检验，使整个过程**恰好**从目标分布中采样。这个检验本质上是一种拒绝采样（rejection sampling）。

## 工作原理 {#how-it-works}

### 符号约定 {#notation}

- $p(\cdot)$：目标模型在某个位置上的下一 token 分布（已经过 temperature、top-p 等处理），定义在词表 $\mathcal{V}$ 上。
- $q(\cdot)$：草稿在同一位置上的分布。
- $k$：每一步提出的草稿 token 数（"草稿长度"或"lookahead"）。
- $\alpha$：**接受率**，即一个草稿 token 被接受的概率。

### 算法（一步） {#the-algorithm-one-step}

1. **起草。** 从当前上下文出发，让草稿自回归地采样 $k$ 个 token：$\tilde x_1 \sim q_1$，$\tilde x_2 \sim q_2(\cdot \mid \tilde x_1)$，……，$\tilde x_k$。同时记下草稿概率 $q_i(\tilde x_i)$。
2. **验证。** 把目标模型在"上下文 $+\ \tilde x_1 \dots \tilde x_k$"上**只跑一次**。因为目标模型是因果的，这一次前向就给出 $k+1$ 个分布：$p_1$（位置 1，只依赖上下文）、$p_2$（依赖 $\tilde x_1$）、……、$p_{k+1}$（依赖全部草稿）。
3. **从左到右接受/拒绝。** 对 $i = 1, \dots, k$：抽 $r \sim \text{Uniform}(0,1)$。如果 $r < \min\!\big(1, \tfrac{p_i(\tilde x_i)}{q_i(\tilde x_i)}\big)$，就接受 $\tilde x_i$ 并继续；否则**拒绝**：从**残差分布（residual distribution）**中采样一个替代 token

    $$
    p_{\text{res}}(x) = \frac{\max\big(0,\ p_i(x) - q_i(x)\big)}{\sum_{y}\max\big(0,\ p_i(y) - q_i(y)\big)},
    $$

    输出它，然后停止（丢弃 $\tilde x_{i+1},\dots,\tilde x_k$）。
4. **奖励 token（bonus token）。** 如果 $k$ 个草稿全被接受，就再从 $p_{k+1}$ 采样一个 token——它是验证那次前向顺带送的。

每一步输出 1 到 $k+1$ 个 token，开销是一次目标模型前向加 $k$ 次草稿前向。目标模型的 KV cache 需要回滚，丢掉被拒绝的位置（用 paged KV 的话，这只意味着截短序列长度）。

### 输出分布恰好是 $p$ 的证明 {#proof-that-the-output-distribution-is-exactly-p}

考虑某一个位置，草稿分布为 $q$，目标分布为 $p$。输出的 token $X$ 要么是被接受的草稿，要么是残差采样的结果。对任意 token $x$：

$$
P(X = x) = \underbrace{q(x)\cdot \min\!\Big(1, \frac{p(x)}{q(x)}\Big)}_{\text{草稿提出 } x \text{ 且被接受}} \;+\; \underbrace{P(\text{reject}) \cdot p_{\text{res}}(x)}_{\text{被拒绝，随后残差分布给出 } x}.
$$

第一项等于 $\min(q(x), p(x))$。对第二项，总接受概率为 $\beta = \sum_y \min(q(y), p(y))$，所以 $P(\text{reject}) = 1 - \beta$。利用 $\min(a,b) + \max(0, b - a) = b$ 并对 $y$ 求和：

$$
\sum_y \max\big(0, p(y) - q(y)\big) = \sum_y \big(p(y) - \min(q(y), p(y))\big) = 1 - \beta .
$$

所以残差分布的归一化常数恰好是 $1-\beta$，第二项就等于 $(1-\beta)\cdot \dfrac{\max(0, p(x) - q(x))}{1-\beta} = \max(0, p(x)-q(x))$。两项相加：

$$
P(X = x) = \min\big(q(x), p(x)\big) + \max\big(0, p(x) - q(x)\big) = p(x). \qquad \blacksquare
$$

用大白话说：当草稿"多提"了某个 token（$q > p$），我们就把多出来的那部分拒掉；当草稿"少提"了（$q < p$），残差分布恰好补上缺的那部分概率质量。逐位置地应用这个规则（每个被接受的 token 会作为条件影响下一个位置，和普通解码完全一样），整个生成序列就服从目标模型的分布。对于贪心解码（$p$ 和 $q$ 都是 one-hot），规则退化为"当且仅当草稿 token 等于目标的 argmax 时接受，否则输出目标的 argmax"。

顺带一提，某个位置上的接受概率是 $\beta = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$，也就是 1 减去总变差距离（total-variation distance）。草稿的分布越接近目标，接受率就越高。

### 每步期望 token 数 {#expected-tokens-per-step}

做一个简化假设：每个草稿 token 以概率 $\alpha$ 被独立接受。第 $i$ 个草稿要被"走到并接受"，当且仅当前 $1..i$ 个草稿全被接受，概率为 $\alpha^i$。被接受的草稿数 $A$ 满足：对 $i \le k$，$P(A \ge i) = \alpha^i$，因此 $\mathbb{E}[A] = \sum_{i=1}^{k} \alpha^i$。此外每一步还会恰好多产出一个 token（拒绝时的残差采样，或者全部接受时的奖励 token）。于是

$$
\mathbb{E}[\text{tokens per step}] = 1 + \sum_{i=1}^{k}\alpha^i = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

直观地说：草稿完美（$\alpha \to 1$）时，每步得到 $k+1$ 个 token；草稿毫无用处（$\alpha = 0$）时，每步仍然得到 1 个——投机解码每次目标模型前向产出的 token 数，永远不会比普通解码少。

**加速比。** 设一次目标模型解码步耗时 $T$，一次草稿步耗时 $cT$（即 $c$ 是草稿与目标的成本比），并假设验证 $k+1$ 个 token 的耗时约为 $T$（在访存受限时成立）。那么

$$
\text{speedup} \approx \frac{1 - \alpha^{k+1}}{(1 - \alpha)(1 + ck)} .
$$

增大 $k$，草稿开销线性增加，而多得到的 token 收益却按几何级数递减，所以存在一个最优的 $k$，通常在 3–8 之间。

### 如何选择草稿 {#choosing-a-draft}

| 草稿 | 如何提出候选 | 优点 | 缺点 |
|---|---|---|---|
| 同系列的**小模型**（例如用 1B 给 70B 起草） | 用自己的 KV cache 自回归采样 | 简单，采样下也无损 | 必须共享 tokenizer；额外占显存；$c$ 不可忽略 |
| **n-gram / prompt lookup** | 在 prompt 或已有输出中查找最近几个生成 token，提出其后面跟着的内容 | 无需训练，开销几乎为零 | 只在输出会复制输入时有用（代码修改、RAG、摘要、改写） |
| **Medusa heads** | 在目标模型最后一层隐状态上接 $k$ 个额外的 head；第 $i$ 个 head 直接预测 token $t+i$ | 只有一个模型，没有额外 KV cache | 各 head 独立预测（不以前面的草稿为条件）→ 靠后位置的接受率更低 |
| **EAGLE (1/2/3)** | 一个小型（单层）transformer，在目标模型的**隐藏特征**加 token embedding 上自回归运行 | 以较低成本获得高接受率；EAGLE-2 加入动态树，EAGLE-3 融合低/中/高层特征 | 需要针对每个目标模型训练 |
| **MTP heads**（DeepSeek-V3 风格） | 在**预训练阶段**就训练好的多 token 预测（multi-token-prediction）模块 | 随模型自带；接受率高 | 只适用于用 MTP 训练过的模型 |
| **Lookahead decoding** | 用 Jacobi 式并行迭代生成候选 n-gram，并行验证 | 不需要草稿模型 | 每步计算量更大 |

EAGLE 的核心观察是：预测下一个**特征向量**（目标模型次顶层的隐状态）比直接预测下一个 token 更容易；而且把"提前一步采样得到的 token"喂给草稿，可以消除采样究竟走了哪条分支所带来的不确定性。

### 树形验证 {#tree-verification}

不只提出一条长度为 $k$ 的猜测链，而是提出一棵**树**：比如位置 1 取 top-3 候选，每个候选再各取 top-2 作为位置 2。把树上所有节点展平成一个序列，用一个**树注意力掩码（tree attention mask）**让目标模型只跑一次：每个节点只能看到上下文和它自己的祖先（看不到兄弟节点）。一次前向就验证了所有路径，保留最长的被接受路径即可。在目标模型前向次数相同的情况下，树能提高期望接受长度，代价是每次前向要验证更多 token。

```text
草稿树：深度 1 取 top-2，每个节点下再取 top-2         展平顺序：[a, b, a1, a2, b1, b2]
                                                     树注意力掩码（1 = 可见；
             ┌─ a ──┬─ a1                            每个节点还能看到全部上下文）
 上下文 ─────┤      └─ a2                                  a  b  a1 a2 b1 b2
             └─ b ──┬─ b1                              a   1  0  0  0  0  0
                    └─ b2                              b   0  1  0  0  0  0
                                                       a1  1  0  1  0  0  0
 一次目标模型前向给全部 6 个节点打分；                  a2  1  0  0  1  0  0
 保留通过验证的最长"根到叶"路径                         b1  0  1  0  0  1  0
                                                       b2  0  1  0  0  0  1
```

位置编码也需要注意：节点 `a1` 和 `b1` 都在深度 2，所以它们拥有相同的位置索引（用于 RoPE），尽管它们在展平序列里是不同的条目。SpecInfer 提出了树形验证，并给出了采样场景下无损的多候选接受规则；Medusa 和 EAGLE-2/3 都使用草稿树（EAGLE-2 会根据草稿模型的置信度动态地塑造树的形状）。

### 什么时候投机解码反而有害 {#when-speculation-hurts}

整个论证依赖于"验证 $k+1$ 个 token 的耗时约等于一步解码"，也就是目标模型处于访存受限状态。在 batch 大小为 $B$ 时，一次验证要处理 $B(k+1)$ 个 token。3.1 章讲过，在 H100 上，线性层大约在每次前向处理几百个 token 时转为计算受限。一旦 $B(k+1)$ 越过这个点，验证开销就随 $k+1$ 近似线性增长，每个被拒绝的 token 都是白白浪费的算力，而这些算力本可以给其他用户用。在大 batch 下，投机解码可能**降低**吞吐。因此引擎会根据负载调整甚至关闭投机解码；它最适合对延迟敏感、并发为低到中等的服务，以及长的单流生成。

## 代码 {#code}

下面是在玩具"语言模型"（词表大小为 6 的随机马尔可夫链）上实现的完整投机采样，并通过实验检验：输出分布等于目标分布，实测的每步 token 数与公式估计的量级相符。

```python
import numpy as np
rng = np.random.default_rng(0)
V, k = 6, 4

def random_chain(sharpness):          # 下一 token 转移表：P[prev] 是一个分布
    logits = rng.standard_normal((V, V)) * sharpness
    P = np.exp(logits); return P / P.sum(1, keepdims=True)

P_target = random_chain(2.0)
P_draft = 0.7 * P_target + 0.3 * random_chain(2.0)     # 一个不错但不完美的草稿

def speculative_step(prev):
    """从 token `prev` 出发的一步 起草-验证。返回输出的 token 列表。"""
    drafts, q_probs, ctx = [], [], prev
    for _ in range(k):                                   # 1) 从 q 起草 k 个 token
        x = rng.choice(V, p=P_draft[ctx]); drafts.append(x); q_probs.append(P_draft[ctx]); ctx = x
    p_probs = [P_target[c] for c in [prev] + drafts]     # 2) “一次目标模型前向”：k+1 个分布
    out = []
    for i, x in enumerate(drafts):                       # 3) 从左到右接受 / 拒绝
        p, q = p_probs[i], q_probs[i]
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x); continue
        res = np.maximum(p - q, 0); res /= res.sum()     # 残差分布
        out.append(rng.choice(V, p=res)); return out
    out.append(rng.choice(V, p=p_probs[k]))              # 4) 奖励 token
    return out

# 从起始 token 0 出发生成大量长度为 3 的序列，比较第 3 个 token 的频率。
n_runs, counts, steps, toks = 40000, np.zeros(V), 0, 0
for _ in range(n_runs):
    seq = []
    while len(seq) < 3:
        new = speculative_step(seq[-1] if seq else 0); steps += 1; toks += len(new); seq += new
    counts[seq[2]] += 1
exact = np.linalg.matrix_power(P_target, 3)[0]           # 真实的 P(第 3 个 token | 起始 0)
print("speculative:", np.round(counts / n_runs, 3))
print("target     :", np.round(exact, 3))

alpha = np.mean([np.minimum(P_target[s], P_draft[s]).sum() for s in range(V)])
print(f"alpha~{alpha:.2f}  formula tokens/step={(1 - alpha**(k + 1)) / (1 - alpha):.2f}  "
      f"measured={toks / steps:.2f}")   # 接近但不相等：真实的 alpha 随上下文变化
```

两行频率在蒙特卡洛误差范围内一致，尽管草稿是另一个不同的分布。你可以试试把 `P_draft` 改成 `random_chain(2.0)`（一个毫不相关的草稿）：输出分布依然正确，只是接受率下降。

## 实例演算 {#worked-example}

**贪心解码一步的完整追踪。** 上下文："The capital of France is"。草稿提出 $k = 4$ 个 token：`" Paris"`、`"."`、`" It"`、`" is"`。目标模型在"上下文加这 4 个 token"上跑一次，返回 5 个位置上各自的 argmax：

| 位置 | 草稿 token | 目标 argmax（以之前所有内容为条件） | 结果 |
|---|---|---|---|
| 1 | `" Paris"` | `" Paris"` | 接受 |
| 2 | `"."` | `"."` | 接受 |
| 3 | `" It"` | `" The"` | 拒绝 → 输出 `" The"`，停止 |
| 4 | `" is"` | （未使用） | 丢弃 |
| 5 | — | （未使用） | 没有奖励 token |

这一步的输出是：`" Paris"`、`"."`、`" The"`——**一次目标模型前向产出 3 个 token**。目标模型为被拒绝的 `" It"` 和 `" is"` 写下的 KV 条目会被丢弃（序列长度直接截短）；`" The"` 的 KV 条目会在下一步开头把它作为输入喂进去时写入，下一步就从"…is Paris. The"继续。

**设定。** 目标：一个 70B 模型，一步解码耗时 $T$。草稿：一个 EAGLE 风格的 head，每步耗时 $c = 0.05\,T$。实测接受率 $\alpha = 0.8$。

**$k = 4$ 时每步 token 数：**

$$
\frac{1 - 0.8^{5}}{1 - 0.8} = \frac{1 - 0.32768}{0.2} = 3.36 .
$$

**加速比：** 每一步耗时 $T(1 + 4 \times 0.05) = 1.2\,T$，所以加速比 $= 3.36 / 1.2 = 2.8\times$。

**选择 $k$**（$\alpha$ 与 $c$ 同上）：

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| tokens/step | 1.80 | 2.44 | 2.95 | 3.36 | 3.69 | 3.95 | 4.16 | 4.33 | 4.57 |
| speedup | 1.71 | 2.22 | 2.57 | 2.80 | 2.95 | 3.04 | 3.08 | 3.09 | 3.05 |

收益在 $k \approx 6$–8 附近趋于平缓，之后开始下降。如果用一个独立的 1B 草稿模型（$c$ 可能在 0.1–0.2），最优的 $k$ 会向更小的值移动。

**batch 的影响。** 在 H100 上，转为计算受限的拐点是每次前向几百个 token（见 3.1 章）。batch 为 8、$k=4$ 时，验证要处理 40 个 token：仍然是访存受限，所以 2.8× 的估计大致成立。batch 为 128、$k=4$ 时，要处理 640 个 token：已经计算受限，验证耗时约为普通一步的 $640/128 = 5\times$，"加速比"变成约 $3.36 / (5 + 0.2) \approx 0.65$——变成了减速。（真实的拐点取决于模型、KV 长度和 kernel；请实测。）

## 实践中的做法（2025–2026） {#in-practice-20252026}

- vLLM、SGLang 和 TensorRT-LLM 支持草稿模型、n-gram/prompt lookup、Medusa、EAGLE（包括 EAGLE-3）以及 MTP。2025 年，EAGLE-3 成为 SGLang/vLLM 中默认的高性能方案，主流开源模型都有现成发布的草稿 head。
- **DeepSeek-V3** 报告称，其 MTP 模块复用于投机解码时，第二个 token 的接受率约为 85–90%，每秒 token 数约为原来的 1.8 倍。
- 引擎会暴露草稿长度参数（`num_speculative_tokens` 或类似名称），并在指标里报告接受率 / 平均接受长度；在生产环境中要盯住这些数字，因为接受率取决于领域（代码和结构化输出的接受率远高于创意写作）。
- 投机解码对**推理模型**在低并发时尤其有价值——一次回答可能长达数万个 token；对"apply edit"这类任务也很有用，因为 prompt lookup 能一次接受很长一段复制的文本。

## 常见误区 {#common-pitfalls-misconceptions}

- **"投机解码是近似的。"** 用上面的拒绝规则，它在分布意义上是完全无损的。（有些变体，比如 Medusa 的"typical acceptance"，会有意用精确性换速度——要清楚自己启用的是哪一种。）
- **"k 越大越好。"** 收益按几何级数递减，而草稿开销线性增长；高负载下，多验证的 token 都要消耗真金白银的算力。
- **只在 batch 为 1 时测加速比。** 要在你实际服务的整个并发范围内报告加速比；高负载下它可能变成负数。
- **采样参数不一致。** $p$ 和 $q$ 必须是*处理之后*的分布（目标模型同样应用相同的 temperature/top-p 流程）；否则无损保证就不成立。
- **逐比特一致的输出。** "分布意义上无损"并不意味着和同一个 seed 下的非投机解码生成完全相同的文本；随机数的消耗方式不同，批量验证也可能让浮点结果略有变化。

## 本章要点 {#key-takeaways}

- 解码是访存受限的，所以验证 $k+1$ 个 token 的耗时约等于一步解码；投机解码把这部分闲置算力花在了"猜测"上。
- 以概率 $\min(1, p/q)$ 接受草稿 $\tilde x$；被拒绝时，从归一化后的 $\max(0, p-q)$ 中重新采样。结果的分布恰好是 $p$。
- 每步期望 token 数 $= (1-\alpha^{k+1})/(1-\alpha)$；加速比 $\approx$ 该值除以 $(1 + ck)$。
- 草稿方案是在训练成本与接受率之间做取舍：n-gram（免费）、小模型、Medusa、EAGLE、MTP；树可以在一次前向中验证多个候选。
- 在大 batch 下，验证会变成计算受限，投机解码可能让系统变慢。

## 自测 {#check-your-understanding}

??? question "目标分布 p = (0.6, 0.3, 0.1)，草稿分布 q = (0.3, 0.6, 0.1)，对应 token (a, b, c)。接受概率是多少？残差分布是什么？"
    $\beta = \min(.6,.3) + \min(.3,.6) + \min(.1,.1) = 0.3 + 0.3 + 0.1 = 0.7$。残差 $\propto \max(0, p-q) = (0.3, 0, 0)$，也就是总是 token a。验算：$P(a) = 0.3 + 0.3 \cdot 1 = 0.6$ ✓。

??? question "α = 0.6、k = 3 时，平均每次目标模型前向能产出多少个 token？"
    $(1 - 0.6^4)/(1-0.6) = (1 - 0.1296)/0.4 = 2.18$。

??? question "为什么所有草稿都被接受时，我们能得到一个“奖励”token？"
    在"上下文 + $k$ 个草稿"上做的验证前向，同时也产出了第 $k+1$ 个位置上的目标分布，所以再从中采样一个 token 不需要任何额外开销。

??? question "为什么 Medusa heads 在第 3、4 个草稿位置上的接受率通常比 EAGLE 低？"
    Medusa 的第 $i$ 个 head 是根据位置 $t$ 的隐状态来预测 token $t+i$ 的，它看不到中间那些已起草的 token，因此无法以所选的分支为条件。EAGLE 则是自回归地起草，把每个已起草的 token 和预测出的特征都反馈回去。

??? question "你的服务白天大部分时间 batch 为 256，夜里 batch 为 8。你会如何配置投机解码？"
    负载低时开启（用中等大小的 $k$），高负载时关闭或缩小 $k$，因为那时验证是计算受限的；很多引擎支持动态调整，你也可以按负载做路由。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022 | 原始算法、无损性证明，以及本章使用的加速比分析。 |
| :material-book-open-variant: [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al. (DeepMind), 2023 | 同期独立推导出同样的规则；应用在 Chinchilla-70B 上。 |
| :material-book-open-variant: [SpecInfer: Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) — Miao et al., 2023 | 用树注意力在一次前向中验证一棵候选树；多候选的无损接受规则。 |
| :material-star: [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Cai et al., 2024 | 在目标模型上加额外 head 来预测未来 token；树形验证；无需独立的草稿模型。 |
| :material-star: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., 2024 | 特征层面的自回归起草 → 接受率大幅提高。 |
| :material-book-open-variant: [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — Li et al., 2024 | 由草稿置信度决定、随上下文变化的草稿树。 |
| :material-new-box: [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — Li et al., 2025 | 多层特征融合以及对解码过程的训练期模拟；2025 年 SGLang/vLLM 中默认的投机方法。 |
| :material-book-open-variant: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Fu et al., 2024 | 无需草稿：用 Jacobi 迭代生成 n-gram 并行验证。 |
| :material-book-open-variant: [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | 从 prompt 中取 n-gram 的最简单草稿方法，附代码。 |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 在预训练中训练 MTP 模块，并将其复用为投机草稿。 |
