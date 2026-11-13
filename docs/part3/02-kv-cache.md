# 3.2 The KV Cache

!!! abstract "What you'll learn"
    - Why generating without a cache wastes quadratic work, and exactly what the **KV cache** stores.
    - Implement a decode step that appends to a KV cache and verify it matches full recomputation.
    - Derive the KV-cache size formula and apply it to Llama-3-70B (8K and 128K contexts) and to MHA vs GQA vs MQA vs MLA.
    - Explain how KV memory — not compute — caps batch size, and why long contexts make decode slower.
    - Describe at intuition level the main ways to shrink the cache: quantization, eviction (StreamingLLM, H2O, SnapKV), and architectural changes.

## Why this matters / the problem

In the [generation loop of Chapter 3.1](01-inference-fundamentals.md#code), every step re-ran the model on the whole sequence. To produce token $t+1$ we recomputed the hidden states of tokens $1 \dots t$, even though they were already computed in the previous step and **cannot change**: in a causal (decoder-only) transformer, position $i$ only attends to positions $\le i$, so its representation never depends on later tokens.

Generating $N$ tokens after an $L$-token prompt this way costs about $\sum_{t=L}^{L+N} t \approx N L + N^2/2$ token-forward-passes instead of $L + N$. For a 1,000-token prompt and 1,000 output tokens that is ~1.5 million versus ~2,000 — a 750× waste. The KV cache removes this waste, and in doing so it becomes **the single largest consumer of GPU memory in a serving system**.

## Intuition

Look at what attention needs. When the new token at position $t$ attends to the past, it needs a **query** for itself and the **keys and values of every earlier token** — in every layer and every head. Everything else about earlier tokens (their MLP activations, their queries) is never used again.

So we keep a notebook per layer: each time a token is processed, write down its key and value vectors. The next step computes Q, K, V only for the one new token, appends the new K and V to the notebook, and attends over the whole notebook. The notebook grows by one entry per layer per token: that notebook is the KV cache.

```text
step t:   new token x_t ──▶ q_t, k_t, v_t   (one row each, per layer & head)
                                   │    │
KV cache (layer ℓ):  K = [k_1 k_2 ... k_{t-1}] + k_t   ──▶ attention(q_t, K, V)
                     V = [v_1 v_2 ... v_{t-1}] + v_t
```

## How it works

### Attention recap

For one head with head dimension $d_h$, a token's hidden state $x \in \mathbb{R}^{d}$ is projected to $q = x W_Q$, $k = x W_K$, $v = x W_V$, each in $\mathbb{R}^{d_h}$. With all keys and values stacked as $K, V \in \mathbb{R}^{t \times d_h}$, the output for the newest token is

$$
o_t = \operatorname{softmax}\!\left(\frac{q_t K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{d_h}.
$$

In words: score the new query against every stored key, turn the scores into weights, and average the stored values with those weights. Only $q_t$ is new; $K$ and $V$ are the cache plus one appended row.

### Prefill and decode with a cache

- **Prefill.** Run the prompt $x_{1..L}$ through the model in one parallel pass with a causal mask. As a side effect, every layer writes $K, V \in \mathbb{R}^{L \times d_h}$ per KV head into the cache. Output: logits for position $L$ → first generated token.
- **Decode.** For each new token: embed it, and in every layer compute $q_t, k_t, v_t$ (shapes `[batch, heads, 1, d_h]`), append $k_t, v_t$ to the cache (now `[batch, kv_heads, t, d_h]`), attend, continue through the MLP. Cost per step is now proportional to the context length only inside attention (reading $K,V$); all the matrix multiplies process just one token.

### Sizing the cache

Define:

- $n_{\text{layers}}$ = number of transformer layers,
- $n_{\text{kv}}$ = number of **key/value heads** (equal to the number of query heads in classic multi-head attention, fewer with GQA, 1 with MQA),
- $d_h$ = head dimension,
- $b$ = bytes per stored element (2 for BF16/FP16, 1 for FP8/INT8),
- $L$ = number of tokens in the context, and $B$ = number of sequences in the batch.

Each token stores one key **and** one value vector of length $d_h$ per KV head per layer:

$$
\text{KV bytes per token} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_h \cdot b,
\qquad
\text{KV bytes total} = B \cdot L \cdot (\text{KV bytes per token}).
$$

In words: the cache grows **linearly in context length and in batch size**, and its per-token cost is set entirely by the attention architecture.

### Why KV memory limits batch size (and speed)

GPU memory holds the weights, a few GB of activations/workspace, and the KV cache. Whatever remains after the weights is the KV budget, and the number of sequences you can run at once is

$$
B_{\max} \approx \frac{\text{KV budget}}{L_{\text{avg}} \cdot \text{KV bytes per token}}.
$$

Chapter 3.1 showed that throughput grows almost linearly with $B$ until a few hundred sequences. If KV memory caps $B$ at, say, 20, you leave most of the GPU's compute unused. So **shrinking the KV cache directly increases throughput**.

There is a second effect. Every decode step reads each sequence's entire cache once (attention over all past tokens). Bytes read per step are therefore

$$
\text{bytes/step} \approx \underbrace{\text{weight bytes}}_{\text{shared by batch}} + \underbrace{B \cdot L \cdot \text{KV bytes per token}}_{\text{not shared}},
$$

and at long context the second term dominates: decode TPOT grows with context length even though the model weights are fixed.

### Shrinking the cache, at intuition level

**1. Change the architecture (most robust).** These are covered in [Part 1, the modern transformer block](../part1/04-modern-block.md):

- **MQA** (multi-query attention): all query heads share a single K/V head. $n_{\text{kv}}=1$.
- **GQA** (grouped-query attention): query heads are split into $n_{\text{kv}}$ groups that share K/V; Llama-3-70B uses 8 KV heads for 64 query heads (8× smaller than MHA).
- **MLA** (multi-head latent attention, DeepSeek-V2/V3): cache one compressed latent vector per token per layer (plus a small decoupled RoPE key) and reconstruct per-head keys/values from it with learned up-projections, which can be absorbed into the query/output projections at inference time.
- **Sliding-window / local attention** in some layers: those layers only keep the last $W$ tokens, so their cache is bounded.
- **Hybrid models** that replace some attention layers with linear-attention or state-space layers, whose state has fixed size.

**2. Quantize the cache.** Store K and V in FP8 (halves the BF16 cache) or lower. FP8 KV is widely supported in serving engines. Going to 2–4 bits needs care because keys have **outlier channels** — some dimensions are consistently large across tokens. KIVI's observation is to quantize keys **per channel** (so each channel gets its own scale) and values **per token**, reaching 2-bit KV with small quality loss. Chapter 3.6 explains the quantization math.

**3. Evict entries (lossy).** Keep only the tokens that matter:

- **StreamingLLM (attention sinks).** Models put large attention weight on the first few tokens regardless of content — they act as a "sink" where softmax can dump probability mass it has to put somewhere. Evicting them breaks the model. Keeping the first ~4 tokens **plus a sliding window** of recent tokens gives stable generation over arbitrarily long streams (but the model cannot recall evicted middle content).
- **H2O (heavy-hitter oracle).** A small set of tokens receives most of the cumulative attention. Keep these "heavy hitters" plus recent tokens; evict the rest under a fixed budget.
- **SnapKV.** After prefill, use the attention of the last few prompt tokens (an "observation window", which usually contains the user's question) to vote, per head, for which earlier positions are important; keep those (with some neighbourhood pooling) and drop the rest. Compresses long prompts once, then decodes normally.

Eviction methods are attractive because they need no retraining, but they are **lossy and task-dependent**: a needle-in-a-haystack question about evicted text cannot be answered. Frontier models in 2025 chose architectural fixes instead.

## Code

A single attention layer with a KV cache, plus a check that cached decoding gives the same output as recomputing attention over the full sequence. Shapes are tracked in comments; GQA is supported by repeating KV heads.

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)

B, d_model, n_q, n_kv = 2, 64, 8, 2          # 8 query heads share 2 KV heads (GQA)
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
    rep = n_q // n_kv                                   # expand KV heads for GQA
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

class KVCache:
    def __init__(self):
        self.k = self.v = None                            # [B, n_kv, t, d_h]
    def append(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

x = torch.randn(B, 12, d_model)                           # 8 prompt + 4 "generated"
cache = KVCache()

# Prefill: process 8 prompt tokens at once, fill the cache.
q, k, v = project(x[:, :8])
k_all, v_all = cache.append(k, v)
outs = [attend(q, k_all, v_all, causal=True)]

# Decode: one token per step; only its q/k/v are computed.
for t in range(8, 12):
    q, k, v = project(x[:, t:t + 1])                      # [B, heads, 1, d_h]
    k_all, v_all = cache.append(k, v)                     # cache grows to t+1
    outs.append(attend(q, k_all, v_all, causal=False))    # 1 query sees all keys

cached = torch.cat(outs, dim=2)
q, k, v = project(x)                                      # reference: full recompute
full = attend(q, k, v, causal=True)
print("cache shape:", tuple(cache.k.shape), "max diff:", (cached - full).abs().max().item())
```

Note `causal=False` in decode: the single new query is the last position, so it may attend to every cached key — no mask is needed.

A small calculator for the formula, used in the worked example below:

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
mla = (512 + 64) * 61 * 2           # DeepSeek-V3: latent 512 + RoPE key 64, 61 layers
print("DeepSeek-V3 MLA:", round(mla / 1024, 1), "KiB/token")
```

## Worked example

**Llama-3-70B** has 80 layers, 64 query heads, **8 KV heads** (GQA), head dimension 128.

*Per token (BF16):* $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes $= 320$ KiB.

*Per sequence:* at 8K tokens ($8{,}192$): $8192 \times 320\ \text{KiB} = 2.5$ GiB. At 128K tokens ($131{,}072$): $40$ GiB — half of an 80 GB H100 for **one** user's cache.

*Architecture comparison* (same 80 layers × head dim 128 shape, BF16):

| Attention | KV heads | KV per token | 128K-token sequence |
|---|---|---|---|
| MHA | 64 | 2.5 MiB | 320 GiB |
| GQA (actual Llama-3-70B) | 8 | 320 KiB | 40 GiB |
| MQA | 1 | 40 KiB | 5 GiB |

For **MLA**, DeepSeek-V3 (61 layers) caches a 512-dimensional latent plus a 64-dimensional RoPE key per token per layer: $(512 + 64) \times 61 \times 2 = 70{,}272$ bytes ≈ **68.6 KiB/token** in BF16 — about 8.6 GiB for a 128K-token sequence, despite DeepSeek-V3 having 128 attention heads. A plain MHA design with DeepSeek-V3's 128 heads (keys of dimension 128+64, values of 128) would need $128 \times (192 + 128) \times 61 \times 2 \approx 4.8$ MiB per token, about 71× more.

*Batch size limit.* Serve Llama-3-70B in BF16 on 8× H100 80 GB with tensor parallelism (each GPU holds 1/8 of the weights and 1 of the 8 KV heads). Total memory 640 GB. Suppose we leave 10% for activations, workspace and CUDA graphs (576 GB usable) and the weights take ≈ 140 GB. The KV budget is ≈ 436 GB ≈ 406 GiB:

$$
\frac{406\ \text{GiB}}{320\ \text{KiB/token}} \approx 1.33 \text{ million tokens}.
$$

That is about **162 sequences at 8K** context but only **10 sequences at 128K**. Since throughput scales with batch size, long-context serving is far more expensive per token.

*Bytes per decode step* for 162 sequences at 8K: weights 140 GB, KV $\approx 436$ GB. With 8 × 3.35 TB/s ≈ 26.8 TB/s aggregate bandwidth, reading weights takes ≈ 5.2 ms but reading the KV cache takes ≈ 16 ms. At long context the KV cache, not the weights, sets TPOT. FP8 KV would halve that 16 ms.

## In practice (2025–2026)

- **GQA is the default** for dense open models (Llama 3, Qwen, Mistral families); DeepSeek-V2/V3/R1 and Kimi K2 use **MLA**; several 2025 models mix full-attention layers with sliding-window or linear-attention layers to bound KV growth (gpt-oss alternates dense and banded-window attention layers).
- **FP8 KV cache** is a common production setting in vLLM, SGLang and TensorRT-LLM; it doubles the number of cached tokens. Lower-bit KV is used more selectively.
- Engines allocate the KV cache as a single large pool at start-up (e.g. vLLM's `gpu_memory_utilization`) and carve it into blocks (Chapter 3.3). The startup log tells you how many tokens fit — read it.
- KV caches are increasingly treated as a reusable asset: prefix caching (Chapter 3.3) and tiered KV storage in CPU memory/SSD (Chapter 3.7) avoid recomputing the same prefill.
- Post-hoc eviction (H2O, SnapKV, …) remains mostly a research and long-context-niche tool; production systems prefer lossless techniques plus architectures designed for small caches.

## Common pitfalls / misconceptions

- **Forgetting the factor 2** (keys *and* values) or using the number of *query* heads instead of KV heads in the formula.
- **Forgetting MLA's special shape.** The MLA cache is not $2 \cdot n_{\text{kv}} \cdot d_h$; it is one latent vector (plus RoPE key) per token per layer, with no factor 2 and no head count.
- **Assuming tensor parallelism shrinks total KV.** TP splits the KV heads across GPUs, so per-GPU KV shrinks, but the total is the same. With fewer KV heads than GPUs (e.g. 8 KV heads, TP=16), heads must be replicated and total KV memory *grows*.
- **Ignoring KV reads in latency estimates.** At long context, KV bandwidth can exceed weight bandwidth (see the worked example).
- **Treating eviction as free.** It silently drops information; always evaluate on retrieval-heavy tasks.
- **Dropping attention sinks** when implementing a sliding window: quality collapses once the first tokens leave the window.

## Key takeaways

- The KV cache stores each past token's keys and values in every layer, so each decode step processes only the new token.
- $\text{KV bytes} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_h \cdot b$ per token, times tokens times sequences.
- Llama-3-70B: 320 KiB/token in BF16 → 2.5 GiB at 8K, 40 GiB at 128K per sequence.
- KV memory caps batch size (hence throughput), and KV reads dominate decode time at long context.
- Fixes, from most to least robust: architecture (GQA/MLA/windowed/hybrid), quantization (FP8, then lower), eviction (sinks, H2O, SnapKV).

## Check your understanding

??? question "Llama-3-8B has 32 layers, 8 KV heads and head dimension 128. What is its KV cache per token in BF16 and in FP8?"
    $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes = 128 KiB in BF16; 64 KiB in FP8.

??? question "Why do we not need to cache queries?"
    A query is only used once, by its own token, at the step where that token is produced. Later tokens attend to earlier tokens' keys and values, never their queries.

??? question "Doubling the batch size at fixed context length: what happens to the weight bytes and the KV bytes read per decode step?"
    Weight bytes stay the same (shared by the batch); KV bytes double. So batching amortizes weights but not the KV cache.

??? question "StreamingLLM keeps 4 sink tokens + 1,020 recent tokens. Can the model answer a question about a fact stated 5,000 tokens earlier?"
    No. Evicted tokens' K/V are gone; StreamingLLM gives stable *fluency* over unbounded streams, not long-range *memory*.

??? question "Why does KIVI quantize keys per channel but values per token?"
    Keys have outlier channels (a few dimensions consistently large across tokens), so a per-channel scale isolates them. Values do not show this channel structure, and per-token quantization fits how they are combined (each output is a weighted sum of value rows).

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | Discovers attention sinks; sinks + sliding window enable stable infinite-length generation. Inspired the learned sinks in gpt-oss. |
| :material-book-open-variant: [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Zhang et al., 2023 | Evict KV entries that receive little cumulative attention; keep heavy hitters + recent tokens. |
| :material-book-open-variant: [SnapKV](https://arxiv.org/abs/2404.14469) — Li et al., 2024 | Per-head selection of important prompt positions using an observation window at the end of the prompt. |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | Keys per-channel, values per-token → 2-bit KV cache with little loss. |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | Multi-query attention: the first paper to attack KV bandwidth architecturally. |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | Grouped-query attention, the default in today's dense models. |
| :material-book-open-variant: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | Introduces multi-head latent attention (MLA) and the weight-absorption trick. |
| :material-book-open-variant: [Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/) | KV-cache sizing and its effect on batch size and latency, with worked numbers. |
