# 1.4 The Modern Transformer Block

!!! abstract "What you'll learn"
    - The five upgrades that turn the 2019 GPT block into the 2025 Llama/Qwen/DeepSeek block: **RMSNorm**, **SwiGLU**, **RoPE**, **GQA/MLA**, **QK-norm** (plus tied embeddings).
    - Why SwiGLU's hidden size is about $\tfrac{8}{3}d$ and how Llama arrives at 11,008.
    - A derivation of rotary position embeddings from 2-D rotations, and a numerical check of its relative-position property.
    - How MHA → MQA → GQA → MLA shrink the KV cache, with worked memory calculations.
    - A complete, runnable Llama-style block in PyTorch.

## Why this matters / the problem

The GPT you built in [1.3](03-transformer.md) works, but it is not what anyone trains in 2025. Between 2019 and 2024 the community converged on a set of small changes that each buy something concrete: slightly better quality per FLOP, more stable training at scale, better handling of positions and long contexts, and above all a much smaller **KV cache**, the memory that dominates the cost of serving a model. The result is often called the "Llama block", although Qwen, DeepSeek, Mistral, Gemma and gpt-oss all use variants of it. Understanding each change and its reason lets you read any model config and know what it implies.

## Intuition

| GPT-2 block (2019) | modern block (2023–2025) | what it buys |
|---|---|---|
| LayerNorm | **RMSNorm** | same stability, less compute, no mean subtraction |
| GELU MLP, $d_{ff} = 4d$ | **SwiGLU** MLP, $d_{ff} \approx \tfrac{8}{3}d$ | better loss at equal parameters and FLOPs |
| learned absolute positions | **RoPE** (rotate q and k) | relative positions, no length-limited table, extendable context |
| multi-head attention | **GQA** or **MLA** | KV cache 4–60× smaller, faster decoding |
| nothing | **QK-norm** (optional) | prevents attention logits from blowing up |
| separate input/output embeddings | tied or untied | tying saves $Vd$ parameters (mostly in small models) |

```text
x ─► RMSNorm ─► Attention (GQA or MLA, RoPE, optional QK-norm) ─► + ─►
     RMSNorm ─► FFN (SwiGLU dense, or MoE: router → top-k experts + shared expert) ─► + ─► next block
```

The MoE variant of the FFN is chapter [1.5](05-mixture-of-experts.md).

## How it works

### RMSNorm

LayerNorm ([1.3](03-transformer.md)) subtracts the mean, divides by the standard deviation, and applies a scale and shift. Zhang & Sennrich (2019) found that the re-centering is unnecessary; rescaling alone gives the stability. **RMSNorm** divides each vector by its root mean square:

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}, \qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},
$$

with a learned per-feature scale $\gamma \in \mathbb{R}^d$ and no bias. In words: shrink or stretch the vector so its typical entry has magnitude about 1, then let the model re-weight features. It saves one reduction (the mean) and one parameter vector; in practice it is computed in FP32 even when the model runs in BF16, because the sum of squares can overflow or lose precision.

### SwiGLU feed-forward

A **gated linear unit** (GLU) multiplies two linear projections elementwise, one of them passed through a nonlinearity that acts as a soft gate. Shazeer (2020) tested variants and found that the one with the **Swish** (also called **SiLU**) activation, $\mathrm{Swish}(z) = z \cdot \sigma(z)$ with $\sigma$ the logistic sigmoid, gave the best results:

$$
\mathrm{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}\big)\, W_{\text{down}},
$$

with $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$. In words: one projection decides *how much* of each hidden feature to let through, the other provides the feature, and the product is projected back. Why it works better is not fully explained theoretically (Shazeer's paper famously attributes it to "divine benevolence"); empirically it is consistently better at equal compute.

**Why $d_{ff} \approx \tfrac{8}{3}d$.** A classic GELU FFN with $d_{ff} = 4d$ has two matrices, $2 \times d \times 4d = 8d^2$ parameters. SwiGLU has three matrices, $3 d\, d_{ff}$ parameters. To keep the parameter count (and FLOPs) the same, set $3 d\, d_{ff} = 8 d^2$, so $d_{ff} = \tfrac{8}{3} d$. For Llama's $d = 4096$: $\tfrac{8}{3} \times 4096 = 10{,}922.7$, and Llama rounds up to a multiple of 256 for hardware efficiency: $43 \times 256 = 11{,}008$. Later models choose $d_{ff}$ more freely (Llama 3 8B uses 14,336 = $3.5d$).

### RoPE: rotary position embeddings

We want attention scores to depend on *where* tokens are, preferably only on their **relative** distance $n - m$ (a pattern like "look at the previous token" should work everywhere in the sequence). Su et al. (2021) achieve this by rotating queries and keys.

**Two dimensions first.** Let $R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$ be the 2-D rotation by angle $\alpha$. Put the query of the token at position $m$ through $R(m\theta)$ and the key at position $n$ through $R(n\theta)$, for some frequency $\theta$. The score becomes

$$
\big(R(m\theta) q\big)^\top \big(R(n\theta) k\big) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n - m)\theta\big)\, k,
$$

because rotation matrices are orthogonal ($R(\alpha)^\top = R(-\alpha)$) and rotations compose by adding angles. In words: rotating both vectors by their absolute positions leaves a score that depends only on their offset $n - m$. Rotation also preserves vector length, so it does not change the scale of the scores.

**Many dimensions.** Split the $d_h$-dimensional head vector into $d_h/2$ pairs and rotate pair $i$ by angle $m\theta_i$, with a different frequency per pair:

$$
\theta_i = b^{-2i/d_h}, \qquad i = 0, 1, \dots, \tfrac{d_h}{2} - 1,
$$

where $b$ is the **RoPE base** (10,000 originally). The first pairs rotate fast (angle changes by ~1 radian per token) and encode fine local position; the last pairs rotate very slowly (wavelength $2\pi/\theta_i$ up to about $2\pi b$ tokens) and encode coarse, long-range position, like the hands of a clock. The total score is a sum over pairs, each of which depends only on $n - m$, so the relative-position property holds for the full vector.

RoPE is applied to $q$ and $k$ only (not $v$), inside every attention layer, and adds no parameters. Because frequencies are a simple function of the base, it is also the hook for **context extension** methods that rescale them ([1.6](06-long-context-and-efficient-attention.md)).

!!! note "Two layouts"
    The original paper pairs adjacent dimensions $(0,1), (2,3), \dots$. The Llama/Hugging Face convention pairs dimension $i$ with $i + d_h/2$ ("rotate half"). Both are valid, but weights trained with one layout must be used with the same layout.

### From MHA to MQA, GQA and MLA: shrinking the KV cache

During generation, each new token must attend to all previous tokens. Recomputing their keys and values every step would be wasteful, so inference engines store them: the **KV cache** ([Part 3](../part3/index.md)). Its size per token is

$$
\text{KV bytes per token} = 2 \times L \times n_{kv} \times d_h \times \text{bytes per element},
$$

where the 2 counts K and V, $L$ is the number of layers, $n_{kv}$ the number of key/value heads, and $d_h$ the head size. Decoding is limited by memory bandwidth: every step reads the whole cache. A smaller cache means more concurrent users per GPU, longer contexts, and faster tokens.

- **MHA (multi-head attention):** $n_{kv} = h$, every query head has its own K and V.
- **MQA (multi-query attention, Shazeer 2019):** $n_{kv} = 1$, all query heads share a single K/V head. The cache shrinks by a factor $h$, with some quality loss and training instability.
- **GQA (grouped-query attention, Ainslie et al. 2023):** $1 < n_{kv} < h$. Query heads are split into $n_{kv}$ groups of $h / n_{kv}$; each group shares one K/V head. With $n_{kv} = 8$ quality is close to MHA, and the cache is $h/8$ times smaller. Implementation: compute $n_{kv}$ K/V heads and repeat each one $h/n_{kv}$ times before attention.
- **MLA (multi-head latent attention, DeepSeek-V2 2024):** instead of storing per-head keys and values, store one compressed **latent vector** per token and reconstruct K and V from it.

**MLA in more detail.** For a token with hidden state $x$:

1. Down-project to a latent: $c^{KV} = W^{DKV} x \in \mathbb{R}^{d_c}$, with $d_c$ much smaller than $h \cdot d_h$ (DeepSeek-V3: $d_c = 512$ vs $h \cdot d_h = 128 \times 128 = 16{,}384$).
2. Up-project to per-head keys and values: $k^C_m = W^{UK}_m c^{KV}$, $v_m = W^{UV}_m c^{KV}$ for each head $m$.
3. Only $c^{KV}$ is cached. At inference the up-projections can be **absorbed**: $q_m^\top k^C_m = q_m^\top W^{UK}_m c^{KV} = (W^{UK\top}_m q_m)^\top c^{KV}$, so the query is mapped into latent space once and attends directly to cached latents; similarly $W^{UV}$ folds into the output projection.
4. **Decoupled RoPE.** Absorption breaks if a position-dependent rotation sits between $W^{UK}$ and the query, so MLA keeps a separate small key component $k^R$ (DeepSeek-V3: 64 dimensions) that carries RoPE and is shared by all heads; it is cached alongside $c^{KV}$. Each head's query is likewise split into a no-RoPE part and a RoPE part.

The cache is then $(d_c + d_R)$ elements per token per layer, where $d_R$ is the RoPE-key size, independent of the number of heads. The DeepSeek-V2 report shows MLA matching or beating MHA quality, unlike MQA/GQA, which trade quality for memory.

### QK-norm

At scale, the attention logits $q \cdot k$ can grow without bound during training, saturating the softmax and causing loss spikes or divergence. **QK-norm** applies a normalization (RMSNorm or LayerNorm, with learned scale) to each head's query and key vectors *before* the dot product (and before RoPE). This caps the magnitude of the scores. It was used to stabilize the 22B-parameter ViT of Dehghani et al. (2023) and is now used in OLMo 2, Qwen3 and Gemma 3. Chapter [1.8](08-training-at-scale.md) discusses other instability fixes.

### Tied embeddings

The input embedding $E \in \mathbb{R}^{V \times d}$ and output projection $W_U \in \mathbb{R}^{d \times V}$ both relate tokens to $d$-dimensional vectors. **Tying** sets $W_U = E^\top$ (Press & Wolf, 2016), saving $Vd$ parameters. For a small model this is a big fraction: with $V = 151{,}936$ and $d = 1024$, $Vd \approx 156\text{M}$, a large share of a 0.6B model. For large models the saving is a small fraction and untied matrices give a bit more flexibility, so most large models (Llama 3, DeepSeek-V3, Qwen3 large variants) do not tie, while small models (Gemma, Qwen3 0.6B–4B) do.

## Code

The full Llama/Qwen-style decoder: RMSNorm, RoPE, GQA with optional QK-norm, SwiGLU, Pre-norm residuals, no biases. It uses PyTorch's fused `scaled_dot_product_attention`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):                          # normalize over the last dim
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight

def rope_cache(T, dh, base=10000.0):
    # theta_i = base^(-2i/dh), i = 0..dh/2-1 ; angle[m, i] = m * theta_i
    theta = base ** (-torch.arange(0, dh, 2).float() / dh)       # [dh/2]
    ang = torch.outer(torch.arange(T).float(), theta)            # [T, dh/2]
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    # x: [B, heads, T, dh]; "half-split" convention: pair dim i with dim i + dh/2
    x1, x2 = x.chunk(2, dim=-1)                                  # [B, heads, T, dh/2] each
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class GQAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, qk_norm=True):
        super().__init__()
        self.h, self.kv, self.dh = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.dh, bias=False)  # fewer K/V heads
        self.wv = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)
        self.q_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin):                               # x: [B, T, d]
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B, h, T, dh]
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)  # [B, kv, T, dh]
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)                     # QK-norm (per head)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # positions via rotation
        rep = self.h // self.kv                                   # query heads per KV head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, h, T, dh]
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))
```

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.attn = GQAttention(d, n_heads, n_kv_heads)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))

class LlamaStyleLM(nn.Module):
    def __init__(self, V, d=256, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=688,
                 max_len=1024, rope_base=10000.0, tie=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_kv_heads, d_ff) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if tie:
            self.head.weight = self.emb.weight          # tied embeddings
        cos, sin = rope_cache(max_len, d // n_heads, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):                             # idx: [B, T]
        T = idx.shape[1]
        x = self.emb(idx)                               # no position embedding: RoPE does it
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T])
        return self.head(self.norm(x))                  # [B, T, V]

model = LlamaStyleLM(V=1000)
logits = model(torch.randint(0, 1000, (2, 16)))
print(logits.shape, sum(p.numel() for p in model.parameters()))  # [2, 16, 1000], 3283456
```

Note $d_{ff} = 688 \approx \tfrac{8}{3} \times 256$ rounded to a multiple of 16. Swap this model into the training loop of [1.3](03-transformer.md) (it returns logits only, so compute the cross-entropy outside) to compare it with the GPT-2-style block.

**Checking the relative-position property numerically.** Using `rope_cache` and `apply_rope` from above:

```python
torch.manual_seed(0)
dh = 8
cos, sin = rope_cache(64, dh)
q, k = torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh)

def score(m, n):  # query at position m, key at position n
    qm = apply_rope(q, cos[m:m+1], sin[m:m+1])
    kn = apply_rope(k, cos[n:n+1], sin[n:n+1])
    return (qm * kn).sum().item()

print(score(5, 2), score(40, 37))   # same offset (3)  -> same score
print(score(10, 10), score(0, 0))   # same offset (0)  -> same score
```

Both pairs print equal values (up to float rounding): the score depends only on the offset.

## Worked example

**KV cache for a 70B-class model.** Llama 3 70B has $L = 80$ layers, $h = 64$ query heads of size $d_h = 128$, and uses GQA with $n_{kv} = 8$. With BF16 (2 bytes per element):

| variant | $n_{kv}$ | bytes per token: $2 \cdot 80 \cdot n_{kv} \cdot 128 \cdot 2$ | at 128K tokens (131,072) |
|---|---|---|---|
| MHA | 64 | 2,621,440 B = 2.5 MiB | 320 GiB |
| **GQA (actual)** | 8 | 327,680 B = 320 KiB | 40 GiB |
| MQA | 1 | 40,960 B = 40 KiB | 5 GiB |

With MHA, a single 128K-token conversation would need 320 GiB of cache, more than four 80 GB GPUs hold, before even counting the 140 GB of BF16 weights. GQA-8 brings it to 40 GiB.

**MLA for DeepSeek-V3.** $L = 61$, latent $d_c = 512$, decoupled RoPE key $d_R = 64$:

$$
(512 + 64) \times 61 \times 2 \text{ bytes} = 70{,}272 \text{ B} \approx 68.6 \text{ KiB per token},
$$

about 8.6 GiB at 128K tokens. If the same model used full MHA with its 128 heads (keys of 128 + 64 dims, values of 128), each token would need $128 \times (192 + 128) \times 61 \times 2 \approx 4.77$ MiB, about 71× more. MLA's cache is smaller than the GQA-8 cache of the (smaller) 70B model above.

**SwiGLU sizing check.** For $d = 4096$, a GELU FFN with $4d$ hidden has $8d^2 = 134.2\text{M}$ parameters per layer; Llama's SwiGLU with $d_{ff} = 11{,}008$ has $3 \times 4096 \times 11{,}008 = 135.3\text{M}$, the same within 1%.

## In practice (2025–2026)

- **Everyone uses** Pre-RMSNorm, SwiGLU (or a close variant), RoPE and no biases in the main linear layers. This combination was popularized by LLaMA (2023).
- **GQA** is the default for dense models: Llama 3 (8 KV heads at all sizes), Qwen3 (e.g. 64 query / 4 KV heads in Qwen3-235B-A22B), Gemma 3, gpt-oss (64 query / 8 KV heads). **MLA** is used by DeepSeek-V2/V3/R1 and Kimi K2.
- **RoPE bases** have grown to support long contexts: 10,000 (original, Llama 2) → 500,000 (Llama 3) → 1,000,000 (Qwen3, Gemma 3 global layers).
- **QK-norm** appears in OLMo 2, Qwen3 and Gemma 3. Kimi K2 uses MLA, where QK-norm is awkward (keys are never materialized during absorbed inference), and instead clips query/key weights during training (MuonClip, [1.8](08-training-at-scale.md)).
- **Some layers skip RoPE.** Llama 4 interleaves layers without positional encoding ("NoPE") with RoPE layers to help length generalization.

## Common pitfalls / misconceptions

- **Applying RoPE to values**, or after the QK dot product. It rotates q and k only, before the dot product.
- **Mixing RoPE layouts** (interleaved vs half-split) when porting weights: everything runs, the outputs are garbage.
- **Assuming GQA reduces FLOPs much.** It mainly shrinks K/V projections and the cache; the score computation still runs for every query head. The win is memory and bandwidth at inference.
- **Thinking MLA is just low-rank K/V.** The key details are weight absorption at inference and decoupled RoPE; without them, you would have to decompress the cache every step and lose most of the benefit.
- **Getting the SwiGLU hidden size from a GELU config.** Copying $d_{ff} = 4d$ into a SwiGLU layer silently increases FFN parameters by 50%.

## Key takeaways

- RMSNorm drops LayerNorm's mean-centering; SwiGLU gates the FFN and uses $d_{ff} \approx \tfrac{8}{3}d$ to keep parameters equal.
- RoPE rotates query/key pairs by position-dependent angles so scores depend only on relative position; the base sets the frequency range.
- The KV cache costs $2 L n_{kv} d_h$ elements per token; GQA cuts $n_{kv}$, MLA caches a small latent instead, and both are central to cheap long-context serving.
- QK-norm keeps attention logits bounded and stabilizes large-scale training.
- Tied embeddings matter for small models, not large ones.

## Check your understanding

??? question "A model has $L = 32$, $h = 32$, $d_h = 128$, GQA with $n_{kv} = 8$, FP16 cache. What is the KV cache per token and for a 32K context?"
    $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes = 128 KiB per token. For 32,768 tokens: $128\text{ KiB} \times 32{,}768 = 4$ GiB.

??? question "Prove in one line that RoPE scores depend only on $n - m$."
    $(R(m\theta)q)^\top(R(n\theta)k) = q^\top R(-m\theta)R(n\theta)k = q^\top R((n-m)\theta)k$, using $R(\alpha)^\top = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha + \beta)$; this holds for each 2-D pair, and the full score is a sum over pairs.

??? question "Why does increasing the RoPE base help long contexts?"
    Larger $b$ makes the slowest frequencies even slower, so their rotation angle stays small (unambiguous) over longer distances. Positions far apart remain distinguishable without angles wrapping around.

??? question "A SwiGLU model has $d = 5120$. What $d_{ff}$ matches the parameters of a $4d$ GELU FFN?"
    $\tfrac{8}{3} \times 5120 \approx 13{,}653$; rounded up to a multiple of 256 that is $54 \times 256 = 13{,}824$.

??? question "Why can't MLA simply apply standard RoPE to the reconstructed keys?"
    The key would become $R_n W^{UK} c$, and the position-dependent rotation sits between $W^{UK}$ and the query, so $W^{UK}$ can no longer be absorbed into the query projection once. Every step would require reconstructing all past keys. Decoupled RoPE puts position information in a separate small key that is cached directly.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [RMSNorm](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019 | LayerNorm without mean-centering: cheaper, equally good. Standard in Llama, Qwen, DeepSeek, Gemma. |
| :material-star: [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 | Gated FFN `(Swish(xW) ⊙ xV)W₂` beats ReLU/GELU FFNs at equal compute. The default FFN everywhere. |
| :material-star: [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al., 2021 | Encodes position by rotating query/key vectors so attention depends on relative distance. |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | All query heads share one K/V head, shrinking the KV cache by the number of heads. |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | Groups of query heads share a K/V head: the quality/memory sweet spot. Also shows how to convert MHA checkpoints to GQA. |
| :material-star: [DeepSeek-V2 (Multi-head Latent Attention, MLA)](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | Low-rank latent KV cache with decoupled RoPE; matches MHA quality with a far smaller cache. |
| :material-book-open-variant: [Scaling Vision Transformers to 22B Parameters](https://arxiv.org/abs/2302.05442) — Dehghani et al., 2023 | Among the first to use **QK-norm** to prevent attention-logit blow-ups. |
| :material-book-open-variant: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) — Press & Wolf, 2016 | The case for tying input and output embeddings. |
| :material-book-open-variant: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023 | Packaged Pre-RMSNorm + SwiGLU + RoPE into the open-model "standard block". |
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | A full dense 405B pipeline; the architecture section is a compact reference for the modern dense block (revisited in [1.9](09-frontier-model-case-studies.md)). |
