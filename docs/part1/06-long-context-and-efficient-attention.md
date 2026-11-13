# 1.6 Long Context & Efficient Attention

!!! abstract "What you'll learn"
    - Why standard attention costs $O(T^2)$ compute and memory in sequence length $T$, and where each cost bites.
    - How **FlashAttention** computes *exact* attention without storing the $T \times T$ matrix, by tiling plus an **online softmax** (derived here).
    - How models trained on short sequences are extended to long ones: **position interpolation**, **NTK-aware scaling** and **YaRN**.
    - Cheaper attention patterns: **sliding windows**, local/global interleaving and **attention sinks**.
    - Sub-quadratic alternatives: **linear attention**, **state-space models (Mamba)**, the **gated delta rule**, **hybrid** models and **sparse attention** (NSA/DSA).

## Why this matters / the problem

Long context is what lets a model read a codebase, a contract, or hours of an agent's tool calls. But the attention of [1.3](03-transformer.md) compares every query with every key. For one head with sequence length $T$, the score matrix $QK^\top$ has $T^2$ entries. Doubling the context quadruples that work. At $T = 131{,}072$ (128K), one head's score matrix has $1.7 \times 10^{10}$ entries, 32 GiB in BF16, for *one head of one layer*. Even when compute is affordable, naively storing that matrix is not. This chapter covers the two families of fixes: make exact attention efficient on hardware (FlashAttention), or change attention so it is not quadratic at all (windows, linear attention, SSMs, sparse attention). It also covers how to extend the context length of a model whose positions (RoPE, [1.4](04-modern-block.md)) were trained only up to some length.

## Intuition

- **FlashAttention:** the problem with naive attention is less the arithmetic than the *traffic*: writing a giant matrix to slow GPU memory and reading it back. FlashAttention computes attention in tiles that fit in fast on-chip memory and never writes the big matrix. The trick that makes this possible is a softmax that can be computed incrementally, one block of keys at a time.
- **Context extension:** RoPE encodes position as rotation angles. A model trained to 8K tokens has never seen the angles that positions beyond 8K produce. Interpolation methods squeeze long positions back into the range of angles the model knows.
- **Cheaper patterns:** most tokens mostly need nearby context. Let most layers look only at a local window and keep a few layers with full attention for long-range retrieval.
- **Linear attention / SSMs:** instead of keeping every past key and value, keep a fixed-size *summary* (a state matrix) that is updated per token, like an RNN. Memory per token becomes constant, but the summary is lossy, so precise recall of far-away details suffers. Hybrids combine both.

## How it works

### The cost of attention

For one layer with $h$ heads of size $d_h$ (so $h d_h = d$), sequence length $T$:

- **Compute:** $QK^\top$ and $AV$ each cost $2 T^2 d$ FLOPs, so $4 T^2 d$ per layer in the forward pass (about half with causal masking exploited). The rest of the layer (projections, FFN) costs $O(T d^2)$. Attention dominates once $T$ exceeds a few times $d$.
- **Memory (training):** a naive implementation stores the $[B, h, T, T]$ attention probabilities for the backward pass.
- **Memory (inference):** the KV cache grows linearly with $T$ ([1.4](04-modern-block.md)), and each decode step must read all of it.

### FlashAttention: tiling and the online softmax

A GPU has a large but slow main memory (HBM, tens of GB at a few TB/s) and a tiny but fast on-chip SRAM per streaming multiprocessor (a few hundred KB). Naive attention writes $S = QK^\top$ to HBM, reads it back for the softmax, writes $P$, reads it again for $PV$. FlashAttention (Dao et al., 2022) splits $Q$ into row blocks and $K, V$ into column blocks, loads one tile at a time into SRAM, and accumulates the output directly. The obstacle is that softmax needs the *whole* row to normalize: $\mathrm{softmax}(s)_j = e^{s_j} / \sum_{j'} e^{s_{j'}}$. The fix is the **online softmax**.

**Deriving the online softmax.** For numerical stability softmax is always computed with the maximum subtracted: $\mathrm{softmax}(s)_j = e^{s_j - m} / \ell$ with $m = \max_j s_j$ and $\ell = \sum_j e^{s_j - m}$. Suppose we have processed a first chunk of scores and hold its statistics $m^{(1)}, \ell^{(1)}$ and an unnormalized output $o^{(1)} = \sum_{j \in \text{chunk 1}} e^{s_j - m^{(1)}} v_j$. A second chunk arrives with local max $\tilde m$. The new global max is $m^{(2)} = \max(m^{(1)}, \tilde m)$. Every old term was computed relative to $m^{(1)}$; to re-express it relative to $m^{(2)}$, multiply by $e^{m^{(1)} - m^{(2)}}$, because $e^{s_j - m^{(1)}} e^{m^{(1)} - m^{(2)}} = e^{s_j - m^{(2)}}$. So

$$
\ell^{(2)} = e^{m^{(1)} - m^{(2)}}\,\ell^{(1)} + \sum_{j \in \text{chunk 2}} e^{s_j - m^{(2)}}, \qquad
o^{(2)} = e^{m^{(1)} - m^{(2)}}\,o^{(1)} + \sum_{j \in \text{chunk 2}} e^{s_j - m^{(2)}}\, v_j .
$$

After the last chunk, the exact attention output is $o / \ell$. In words: keep a running max, a running denominator and a running weighted sum; whenever the max increases, shrink the old accumulators by the appropriate factor. This needs only $O(1)$ extra numbers per query row, regardless of $T$.

For the backward pass, FlashAttention stores only $m$ and $\ell$ per row (not the $T \times T$ probabilities) and *recomputes* the tiles of $P$ on the fly. Recomputation adds FLOPs but removes the dominant memory traffic, so it is faster overall. The result is exact attention with $O(T)$ memory. FlashAttention-2 improved parallelization across the sequence and warps; FlashAttention-3 targets Hopper GPUs (asynchronous memory copies, warp specialization, FP8). Kernel details are covered in [Part 3](../part3/index.md).

### Extending the context window

Pre-training at very long lengths is expensive, so models are trained mostly at 4K–8K tokens and then **extended** with a short continued-training phase on long documents. With RoPE, pair $i$ at position $m$ rotates by angle $m\theta_i$ with $\theta_i = b^{-2i/d_h}$ ([1.4](04-modern-block.md)). Let the trained length be $L$ and the target length $L' = sL$, where $s$ is the **scale factor**.

- **Position interpolation (PI)** (Chen et al., 2023): use position $m / s$ instead of $m$, equivalently divide every frequency by $s$: $\theta_i' = \theta_i / s$. All angles stay within the trained range. The cost: high-frequency pairs, which distinguish neighboring tokens, are also compressed by $s$, so nearby positions become harder to tell apart. A short fine-tune (Chen et al. report about 1,000 steps) recovers most quality.
- **NTK-aware scaling:** change the base instead, so the high frequencies are almost untouched and the lowest frequency is interpolated by exactly $s$. The lowest frequency is $\theta_{d_h/2-1} = b^{-(d_h-2)/d_h}$. Requiring $(b')^{-(d_h-2)/d_h} = b^{-(d_h-2)/d_h} / s$ gives

    $$
    b' = b \cdot s^{d_h/(d_h - 2)} .
    $$

    In words: increase the base by slightly more than $s$. This spreads the interpolation unevenly: none for fast pairs, full for the slowest. (The name comes from neural tangent kernel arguments about learning high frequencies.)
- **YaRN** (Peng et al., 2023) makes the split explicit, per frequency. For pair $i$, compute its wavelength $\lambda_i = 2\pi / \theta_i$ and the ratio $r_i = L / \lambda_i$ (how many full rotations it completes within the trained length). Pairs with many rotations ($r_i > \beta$) already saw all angles during training: leave them alone. Pairs with less than a fraction of a rotation ($r_i < \alpha$) must be interpolated: divide by $s$. In between, blend linearly with a ramp. The paper uses $\alpha = 1$, $\beta = 32$ for Llama models. YaRN also multiplies the attention logits by a temperature factor, $\sqrt{1/t} = 0.1 \ln s + 1$, to counter the flatter attention distributions at long range. It reaches long contexts with a small amount of continued training and is used by Qwen, DeepSeek-V3, Kimi K2 and gpt-oss.

Another, complementary lever is simply training with a large base from the start (Llama 3 uses $b = 500{,}000$), and doing a dedicated long-context training stage with gradually increasing length.

### Sliding-window and local/global attention

**Sliding-window attention (SWA)** lets each token attend only to the previous $W$ tokens. Compute per layer drops to $O(T W)$ and the KV cache per layer is capped at $W$ entries. Information still travels further than $W$: after $L$ stacked windowed layers, the receptive field is up to $L \cdot W$ tokens, like stacked convolutions. Mistral 7B used $W = 4096$.

In practice, pure SWA loses precise long-range retrieval, so models **interleave** local and global layers: Gemma 2 alternated them 1:1; Gemma 3 uses 5 local layers (window 1024) per global layer; gpt-oss alternates full attention with banded (windowed) attention of 128 tokens. Only the global layers need a full-length KV cache.

### Attention sinks

Softmax weights must sum to 1, even when a head has nothing useful to attend to. Models learn to dump that excess attention on the first token(s), which are visible to every position. Xiao et al. (2023) called these **attention sinks** and showed that a sliding-window cache that evicts the first tokens collapses, while keeping the first 4 tokens plus a recent window (StreamingLLM) keeps generation stable over very long streams. gpt-oss builds this in: each head has a **learned sink logit** that adds an extra term to the softmax denominator, letting the head put probability mass on "nothing" instead of on an arbitrary token.

### Linear attention

Write causal attention for query $t$ as $o_t = \sum_{j \le t} \mathrm{sim}(q_t, k_j)\, v_j / \sum_{j \le t} \mathrm{sim}(q_t, k_j)$, with $\mathrm{sim}(q, k) = e^{q \cdot k / \sqrt{d}}$. **Linear attention** (Katharopoulos et al., 2020) replaces the exponential with a dot product of feature maps, $\mathrm{sim}(q, k) = \phi(q)^\top \phi(k)$ with $\phi$ positive (e.g. $\mathrm{elu}(x) + 1$). Then, by associativity,

$$
o_t = \frac{\sum_{j \le t} v_j \, \phi(k_j)^\top \phi(q_t)}{\sum_{j \le t} \phi(k_j)^\top \phi(q_t)} = \frac{S_t\, \phi(q_t)}{z_t^\top \phi(q_t)}, \qquad S_t = S_{t-1} + v_t \phi(k_t)^\top, \quad z_t = z_{t-1} + \phi(k_t).
$$

In words: the whole history is compressed into a $d_v \times d_k$ matrix $S_t$ (and a vector $z_t$) that is updated by adding one outer product per token. Inference memory is constant in $T$, like an RNN, and training can still run in parallel (or in chunks). The weakness: $S_t$ is a fixed-size sum, so old information is never removed and keys interfere with each other. Recall of specific facts from far back degrades.

### State-space models and Mamba

A (discretized) **state-space model** maps an input sequence $x_t$ to outputs through a hidden state $h_t$:

$$
h_t = \bar A_t\, h_{t-1} + \bar B_t\, x_t, \qquad y_t = C_t\, h_t ,
$$

where $\bar A_t$ controls how the state decays, $\bar B_t$ how the input is written, and $C_t$ how the state is read. Classical SSMs used fixed (time-invariant) matrices, which allows fast convolution-based training but cannot decide *what* to remember based on content. **Mamba** (Gu & Dao, 2023) makes $\bar B_t$, $C_t$ and the step size (and hence the decay $\bar A_t$) functions of the input token: the model can choose to keep or forget information selectively. It trains with a hardware-aware parallel scan and decodes with $O(1)$ memory per token. **Mamba-2** (Dao & Gu, 2024) showed that a restricted form of this recurrence is equivalent to a kind of linear attention with a decay mask ("state space duality"), which allows matmul-friendly chunked training.

### The gated delta rule

Linear attention *adds* $v_t k_t^\top$ to the state. Think of $S$ as an associative memory: reading with a (unit-norm) key gives $S k$. The **delta rule** instead *corrects* the memory toward the desired value: it reads the current prediction $S_{t-1} k_t$, and writes only the error,

$$
S_t = S_{t-1} + \beta_t \big(v_t - S_{t-1} k_t\big) k_t^\top = S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,
$$

where $\beta_t \in (0, 1]$ is a learned, input-dependent writing strength. With $\beta_t = 1$, after the update $S_t k_t = v_t$ exactly: the old value stored under key $k_t$ is replaced, not piled on. **Gated DeltaNet** (Yang et al., 2024) adds a decay gate $\alpha_t \in (0, 1)$ that can quickly forget everything when context changes:

$$
S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t .
$$

In words: forget a little (gate), overwrite precisely what this key addressed (delta), then write the new association. This gives much better in-context recall than plain linear attention while keeping a constant-size state.

### Hybrid models

Linear/SSM layers give cheap long context; softmax attention gives precise retrieval. **Hybrids** stack mostly linear layers with occasional full-attention layers. MiniMax-01 interleaves lightning (linear) attention with a softmax attention layer every eighth layer; Qwen3-Next uses roughly 3 Gated DeltaNet layers per full-attention layer; Jamba and NVIDIA's Nemotron-H mix Mamba layers with attention. The KV cache only grows in the few full-attention layers.

### Sparse attention

Instead of approximating attention, **sparse attention** computes exact softmax attention over a *selected subset* of past tokens. **Native Sparse Attention (NSA)** (DeepSeek, 2025) combines three branches per query: attention over compressed block summaries (coarse view of everything), attention over the top-scoring blocks selected using those summaries (fine detail where it matters), and a sliding window (local context), mixed by learned gates. It is trained end-to-end and designed so the selected blocks are contiguous in memory, which keeps GPU kernels efficient. DeepSeek-V3.2's **DeepSeek Sparse Attention (DSA)** uses a lightweight "indexer" to score past tokens cheaply and then attends only to the top-$k$. A related idea, the **Differential Transformer**, keeps dense attention but computes it as the difference of two softmax maps, cancelling common-mode "attention noise" and improving long-context retrieval.

## Code

**FlashAttention's forward pass in 25 lines.** This is the algorithm, written in plain PyTorch (so it is not fast); it checks against naive attention.

```python
import math, torch

def naive_attention(q, k, v):                     # q, k, v: [T, d]  (single head, no mask)
    s = q @ k.T / math.sqrt(q.shape[-1])          # [T, T] materialized
    return torch.softmax(s, dim=-1) @ v

def tiled_attention(q, k, v, block=16):
    """FlashAttention-style forward: stream over K/V blocks with an online softmax.
    Never materializes the [T, T] matrix; extra state per query row is just (m, l)."""
    T, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    for i in range(0, T, block):                  # each query block (one GPU thread block)
        qi = q[i:i + block] * scale               # [b, d]
        m = torch.full((qi.shape[0], 1), -float("inf"))   # running max of scores
        l = torch.zeros(qi.shape[0], 1)                   # running sum of exp(score - m)
        acc = torch.zeros(qi.shape[0], d)                 # running unnormalized output
        for j in range(0, T, block):              # stream K/V blocks through "SRAM"
            s = qi @ k[j:j + block].T             # [b, b] scores for this tile only
            m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
            p = torch.exp(s - m_new)              # tile probabilities (unnormalized)
            c = torch.exp(m - m_new)              # rescale factor for old statistics
            l = l * c + p.sum(-1, keepdim=True)
            acc = acc * c + p @ v[j:j + block]
            m = m_new
        out[i:i + block] = acc / l                # normalize once at the end
    return out

torch.manual_seed(0)
q, k, v = (torch.randn(100, 32) for _ in range(3))
print(torch.allclose(naive_attention(q, k, v), tiled_attention(q, k, v), atol=1e-5))  # True
```

**Linear attention: parallel form = recurrent form, plus the gated delta rule.**

```python
import torch
import torch.nn.functional as F

def phi(x):                                   # positive feature map (Katharopoulos et al.)
    return F.elu(x) + 1

def linear_attention_parallel(q, k, v):       # [T, d] each; causal, training-style
    A = phi(q) @ phi(k).T                     # [T, T] similarities (no softmax)
    A = torch.tril(A)                         # causal mask
    return (A @ v) / A.sum(-1, keepdim=True)

def linear_attention_recurrent(q, k, v):      # inference-style: O(1) state per step
    d_k, d_v = q.shape[1], v.shape[1]
    S = torch.zeros(d_v, d_k)                 # state: sum_j v_j phi(k_j)^T
    z = torch.zeros(d_k)                      # normalizer: sum_j phi(k_j)
    outs = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))
        z = z + phi(k[t])
        outs.append(S @ phi(q[t]) / (z @ phi(q[t])))
    return torch.stack(outs)

def gated_delta_recurrent(q, k, v, alpha, beta):
    """S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ; o_t = S_t q_t"""
    S = torch.zeros(v.shape[1], k.shape[1])
    outs = []
    for t in range(q.shape[0]):
        kt = F.normalize(k[t], dim=0)         # unit-norm keys keep the update stable
        S = alpha[t] * (S - beta[t] * torch.outer(S @ kt, kt)) + beta[t] * torch.outer(v[t], kt)
        outs.append(S @ q[t])
    return torch.stack(outs)

torch.manual_seed(0)
T, d = 50, 16
q, k, v = (torch.randn(T, d) for _ in range(3))
print(torch.allclose(linear_attention_parallel(q, k, v),
                     linear_attention_recurrent(q, k, v), atol=1e-5))        # True
o = gated_delta_recurrent(q, k, v, alpha=torch.full((T,), 0.95), beta=torch.full((T,), 0.5))

# Delta rule as associative memory: writing the same (key -> value) twice does not accumulate.
S = torch.zeros(d, d)
key, val = F.normalize(torch.randn(d), dim=0), torch.randn(d)
for _ in range(3):
    S = S - torch.outer(S @ key, key) + torch.outer(val, key)   # beta = 1, alpha = 1
print(torch.allclose(S @ key, val, atol=1e-5))                  # True: exact recall
```

With plain linear attention, writing the same pair three times would return $3 \times$ the value; the delta rule returns it exactly.

## Worked example

**Online softmax by hand.** Scores for one query: $s = (1, 2 \mid 4)$, split into chunks of 2 and 1, with scalar values $v = (10, 20 \mid 30)$.

- Chunk 1: $m^{(1)} = 2$, $\ell^{(1)} = e^{1-2} + e^{0} = 1.3679$, $o^{(1)} = e^{-1} \cdot 10 + 1 \cdot 20 = 23.679$.
- Chunk 2: local max 4, so $m^{(2)} = 4$ and the rescale factor is $e^{2-4} = 0.1353$.
- $\ell^{(2)} = 1.3679 \times 0.1353 + e^{0} = 1.1851$; $o^{(2)} = 23.679 \times 0.1353 + 30 = 33.205$.
- Output $= 33.205 / 1.1851 = 28.02$.

Direct check: $\big(e^{-3} \cdot 10 + e^{-2} \cdot 20 + 30\big) / \big(e^{-3} + e^{-2} + 1\big) = 33.205 / 1.1851 = 28.02$. Identical.

**Memory of local/global interleaving.** Consider a hypothetical model with 62 layers, 16 KV heads of size 128, BF16 cache, and a 128K context. Per layer, each cached token costs $2 \times 16 \times 128 \times 2 = 8192$ bytes.

- All 62 layers global: $62 \times 8192 \times 131{,}072 = 62$ GiB per sequence.
- 10 global + 52 local layers with a 1024-token window: $10 \times 8192 \times 131{,}072 + 52 \times 8192 \times 1024 \approx 10.0 + 0.4 = 10.4$ GiB.

A ~6× reduction, which is the reason Gemma 3 moved from Gemma 2's 1:1 pattern to 5:1 with a short window.

**Context extension numbers.** Extending from 32K to 128K means $s = 4$. NTK-aware scaling with $d_h = 128$ and $b = 10{,}000$ gives $b' = 10{,}000 \times 4^{128/126} \approx 40{,}890$. YaRN's attention temperature factor is $0.1 \ln 4 + 1 \approx 1.139$.

## In practice (2025–2026)

- **FlashAttention-style kernels are universal** for training and prefill; nobody materializes the attention matrix. In PyTorch, `F.scaled_dot_product_attention` selects such kernels automatically.
- **Context lengths of 128K are standard** for open frontier models (Llama 3, DeepSeek-V3, Qwen3 with YaRN, gpt-oss, Kimi K2, Gemma 3), typically reached by short-context pre-training followed by staged long-context extension. Llama 3 trained its long-context stage in six increments up to 128K.
- **KV-cache reduction is the main architectural lever**: GQA/MLA ([1.4](04-modern-block.md)) plus local/global interleaving (Gemma 3, gpt-oss).
- **Hybrids are entering frontier models** (MiniMax-01, Qwen3-Next, Nemotron-H), and trainable sparse attention (NSA, DSA in DeepSeek-V3.2) is the main alternative for keeping exact softmax attention at long range.
- **Context parallelism** (splitting one sequence across GPUs, e.g. Ring Attention) is how million-token training fits in memory; see [1.8](08-training-at-scale.md).

## Common pitfalls / misconceptions

- **"FlashAttention is an approximation."** It is exact (up to floating-point reordering); it changes the memory access pattern, not the math.
- **"FlashAttention makes attention linear."** Memory becomes $O(T)$, but compute is still $O(T^2)$.
- **Claimed context ≠ usable context.** A model may accept 128K tokens and still fail to use information in the middle. Evaluate with retrieval and reasoning tasks at the lengths you need.
- **Changing RoPE scaling at inference without matching training.** Interpolation methods work best (or only) with the continued training they were designed for; settings must match the checkpoint's config.
- **Expecting linear attention to recall like softmax attention.** A fixed-size state cannot store an unbounded number of facts; that is why hybrids keep some full attention.
- **Evicting the first tokens from a sliding cache** breaks models that rely on attention sinks.

## Key takeaways

- Attention is $O(T^2)$ in compute; naive implementations are also $O(T^2)$ in memory, which FlashAttention removes via tiling and the online softmax.
- The online softmax keeps a running max, denominator and output, rescaling by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the max grows.
- Context extension rescales RoPE frequencies: uniformly (PI), via the base (NTK), or per frequency band with a temperature (YaRN).
- Local windows, interleaved with a few global layers, cut KV cache by large factors; attention sinks keep windowed caches stable.
- Linear attention, SSMs and the gated delta rule replace the growing KV cache with a fixed-size state; hybrids and trainable sparse attention combine efficiency with precise retrieval.

## Check your understanding

??? question "Why does FlashAttention speed things up even though it does extra FLOPs in the backward pass?"
    Attention is limited by memory traffic, not arithmetic. Recomputing tiles in fast SRAM is cheaper than writing and reading a $T \times T$ matrix to and from HBM.

??? question "In the online softmax, what happens if the new chunk's maximum is smaller than the running maximum?"
    $m$ stays the same, so the rescale factor $e^{m_{\text{old}} - m_{\text{new}}} = 1$ and the old accumulators are unchanged; the new terms are simply added.

??? question "A model has 32 layers, each with a sliding window of 4096. How far back can information travel to the last token?"
    Up to about $32 \times 4096 = 131{,}072$ tokens, because each layer can move information by up to one window. In practice, information degrades over many hops, which is why global layers are kept.

??? question "Why is PI worse than NTK/YaRN at short distances?"
    PI divides all frequencies by $s$, including the fast ones that distinguish neighboring tokens, so local position resolution drops by $s$. NTK/YaRN leave high frequencies nearly untouched and only interpolate the slow ones.

??? question "What does the delta rule do differently from linear attention when the same key is written twice with different values?"
    Linear attention adds both values, so reading returns their sum. The delta rule (with $\beta = 1$) replaces the first value with the second, because it writes only the difference between the target and the current read-out.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | Tiles attention so the $T \times T$ matrix never hits HBM; exact, faster, $O(T)$ memory. |
| :material-book-open-variant: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018 | The one-pass softmax normalizer that FlashAttention's online softmax builds on. |
| :material-book-open-variant: [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023 | Better work partitioning across warps/thread blocks; ~2× over v1. |
| :material-book-open-variant: [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | Hopper-specific: asynchrony (TMA, warp specialization) and FP8. |
| :material-book-open-variant: [Extending Context Window of LLMs via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Chen et al., 2023 | Position interpolation for RoPE, and why it beats extrapolation. |
| :material-star: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — Peng et al., 2023 | Per-frequency ("NTK-by-parts") RoPE rescaling plus an attention temperature; widely used. |
| :material-book-open-variant: [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023 | Shards the sequence across devices and passes K/V blocks around a ring: the basis of context parallelism. |
| :material-book-open-variant: [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023 | Sliding-window attention with a rolling KV cache in a strong small model. |
| :material-book-open-variant: [Gemma 2](https://arxiv.org/abs/2408.00118) — Gemma Team, 2024 | Interleaved local/global attention and logit soft-capping. |
| :material-star: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | Discovers attention sinks; StreamingLLM keeps initial tokens plus a window. |
| :material-book-open-variant: [Transformers are RNNs (Linear Attention)](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020 | Kernelized attention and its recurrent form, as derived above. |
| :material-star: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023 | Input-dependent SSM; $O(T)$ training, $O(1)$-per-token inference memory. |
| :material-book-open-variant: [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) — Dao & Gu, 2024 | State space duality between SSMs and linear attention; faster matmul-based algorithm. |
| :material-new-box: [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Yang et al., 2024 | Gating plus the delta rule for linear attention; used in hybrid form in Qwen3-Next. |
| :material-new-box: [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) — MiniMax, 2025 | First frontier-scale hybrid linear-attention MoE (456B), with up to 1M–4M context. |
| :material-new-box: [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek), 2025 | Hardware-aligned, trainable sparse attention; precursor of DeepSeek Sparse Attention in V3.2. |
| :material-book-open-variant: [Differential Transformer](https://arxiv.org/abs/2410.05258) — Ye et al., 2024 | Attention as the difference of two softmax maps, cancelling attention noise. |
