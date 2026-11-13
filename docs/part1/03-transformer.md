# 1.3 The Transformer

!!! abstract "What you'll learn"
    - How token ids become vectors (embeddings) and how positions are encoded.
    - Scaled dot-product attention, derived step by step with explicit tensor shapes, including why we divide by $\sqrt{d_k}$.
    - The causal mask, multi-head attention, the feed-forward network, residual connections and LayerNorm (Pre-LN vs Post-LN).
    - A complete minimal GPT in about 100 lines of PyTorch that you can train on a laptop.
    - How to count parameters and FLOPs ($\approx 6N$ per training token, $2N$ per inference token), worked through for a 7B-class model.

## Why this matters / the problem

Chapter [1.1](01-language-modeling.md) reduced language modeling to computing $p_\theta(x_t \mid x_{<t})$ for every position. We need a function that (a) reads a prefix of any length, (b) lets each position use information from *any* earlier position, not just a fixed window, and (c) computes all positions in parallel so training on GPUs is efficient. Recurrent networks satisfied (a) and partly (b), but process tokens one after another, so they are slow to train and struggle to carry information over long distances. The **Transformer** (Vaswani et al., 2017) satisfies all three. Every model in this course is a decoder-only descendant of it, so this chapter is the foundation for everything that follows.

## Intuition

Picture each token as a worker holding a vector, its current "understanding" of itself in context. A Transformer layer has two phases:

1. **Attention (communication).** Each token broadcasts a *key* ("here is what I contain") and a *value* ("here is what I'll hand over"), and sends out a *query* ("here is what I'm looking for"). Each token compares its query with every earlier token's key, turns the match scores into weights, and collects a weighted average of their values. "it" can look back and pull in information from "the cat".
2. **Feed-forward network (computation).** Each token then thinks on its own: a small two-layer MLP transforms its vector. Much of the model's factual knowledge is thought to be stored in these MLP weights.

Stack dozens of such layers and the vectors evolve from "this is the token *bank*" to "this is *bank* in the financial sense, the object of the verb, and the next token is probably a noun phrase". The last vector at each position is projected to logits over the vocabulary.

```text
token ids ─► embedding ─►┌───────────────────────── × L layers ─────────────────────────┐─► norm ─► unembed ─► logits
                         │ x = x + Attention(Norm(x))    (tokens talk to earlier tokens)│
                         │ x = x + MLP(Norm(x))          (each token thinks alone)      │
                         └──────────────────────────────────────────────────────────────┘
```

## How it works

Notation used throughout: batch size $B$, sequence length $T$, model (hidden) width $d$ (also written `d_model`), number of heads $h$, per-head width $d_h = d / h$, number of layers $L$, vocabulary size $V$.

### 1. Embeddings and positions

The input is a tensor of token ids `idx: [B, T]`. An **embedding matrix** $E \in \mathbb{R}^{V \times d}$ maps each id to a row: `x = E[idx]: [B, T, d]`. This is just a learnable lookup table.

Attention, as we will see, treats its inputs as an unordered set: permuting the tokens permutes the outputs in the same way. Word order must therefore be injected explicitly. The original Transformer added fixed sinusoidal vectors; GPT-2 added a **learned position embedding** $P \in \mathbb{R}^{T_{\max} \times d}$, so `x = E[idx] + P[0:T]`. Modern models instead rotate queries and keys with **RoPE** ([1.4](04-modern-block.md)), but learned positions are simplest to start with.

### 2. Scaled dot-product attention

Take one sequence, `x: [T, d]` (drop the batch dimension for now). Three learned matrices $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$ project every token:

$$
Q = x W_Q \in \mathbb{R}^{T \times d_k}, \qquad K = x W_K \in \mathbb{R}^{T \times d_k}, \qquad V = x W_V \in \mathbb{R}^{T \times d_v}.
$$

Row $i$ of $Q$ is token $i$'s query $q_i$; similarly $k_j$ and $v_j$. The similarity between token $i$'s query and token $j$'s key is the dot product $q_i \cdot k_j$. Computing all pairs at once is one matrix product:

$$
S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}, \qquad S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}.
$$

Each row of $S$ is turned into a probability distribution with a softmax over $j$, giving **attention weights** $A = \mathrm{softmax}_{\text{row}}(S)$, and each token's output is the weighted average of the value vectors:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) V \in \mathbb{R}^{T \times d_v}.
$$

In words: output row $i$ is $\sum_j A_{ij} v_j$, a mixture of all value vectors, weighted by how well token $i$'s query matches each token's key. The weights in a row are non-negative and sum to 1.

**Why divide by $\sqrt{d_k}$?** Suppose at initialization the components of $q$ and $k$ are independent with mean 0 and variance 1. Then

$$
\mathrm{Var}(q \cdot k) = \mathrm{Var}\Big(\sum_{m=1}^{d_k} q_m k_m\Big) = \sum_{m=1}^{d_k} \mathrm{Var}(q_m)\mathrm{Var}(k_m) = d_k,
$$

since each product $q_m k_m$ has mean 0 and variance $1 \cdot 1$, and variances of independent terms add. So raw scores have standard deviation $\sqrt{d_k}$, which is about 11.3 for $d_k = 128$. Softmax over values that spread so widely is nearly one-hot: one weight is ≈1, the rest ≈0, and the gradient of a saturated softmax is nearly zero, so learning stalls. Dividing by $\sqrt{d_k}$ brings the score variance back to 1 regardless of head size.

### 3. The causal mask

For language modeling, position $i$ must not see tokens $j > i$; otherwise predicting $x_{i+1}$ would be trivial (it is in the input). We set $S_{ij} = -\infty$ for $j > i$ before the softmax. Since $e^{-\infty} = 0$, those weights become exactly 0:

```text
          keys j →   1     2     3     4
query i = 1        [ s11  -inf  -inf  -inf ]
query i = 2        [ s21  s22   -inf  -inf ]
query i = 3        [ s31  s32   s33   -inf ]
query i = 4        [ s41  s42   s43   s44  ]
```

With the mask, a single forward pass over $T$ tokens produces $T$ valid next-token predictions at once, each using only its own prefix. This is what makes Transformer training parallel. A model with this mask is a **decoder-only** (causal) Transformer.

### 4. Multi-head attention

One attention pattern per layer is limiting: a token may need to look at the previous token (syntax), at the subject of the sentence (agreement), and at a matching bracket (code) all at once. **Multi-head attention** runs $h$ attention operations in parallel, each with its own projections into a smaller space of width $d_h = d/h$:

$$
\mathrm{head}_m = \mathrm{Attention}(x W_Q^{(m)}, x W_K^{(m)}, x W_V^{(m)}), \quad m = 1, \dots, h,
$$

$$
\mathrm{MHA}(x) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_O, \qquad W_O \in \mathbb{R}^{d \times d}.
$$

In practice all heads are computed together with reshapes. Shapes for the batched version:

| step | tensor | shape |
|---|---|---|
| input | `x` | `[B, T, d]` |
| fused projection | `x @ W_qkv` | `[B, T, 3d]` |
| split and reshape into heads | `q, k, v` | `[B, h, T, d_h]` each |
| scores | `q @ k.transpose(-2, -1) / sqrt(d_h)` | `[B, h, T, T]` |
| mask + softmax, then `@ v` | `out` | `[B, h, T, d_h]` |
| merge heads | `out.transpose(1, 2).reshape(B, T, d)` | `[B, T, d]` |
| output projection | `out @ W_O` | `[B, T, d]` |

The total cost is about the same as one big head of width $d$, because each head works in a $d_h = d/h$ space.

### 5. The feed-forward network (FFN / MLP)

After attention, each position independently passes through a two-layer MLP that expands to a hidden width $d_{ff}$ (classically $4d$) and projects back:

$$
\mathrm{FFN}(x) = \phi(x W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\ W_2 \in \mathbb{R}^{d_{ff} \times d},
$$

where $\phi$ is a nonlinearity (ReLU in the original, GELU in GPT-2, SwiGLU in modern models, see [1.4](04-modern-block.md)). With $d_{ff} = 4d$ the FFN holds $8d^2$ parameters, twice the attention block's $4d^2$; the FFN is where most parameters live.

### 6. Residual connections and the residual stream

Each sublayer *adds* its output to its input: $x \leftarrow x + F(x)$. Two benefits:

- **Gradient flow.** The derivative of $x + F(x)$ with respect to $x$ is $I + \partial F/\partial x$; the identity term gives gradients a direct path from the loss back to early layers, so 100-layer networks remain trainable.
- **The residual stream view.** Think of the vector $x$ at each position as a shared bus of width $d$ running from the embedding to the output. Each attention or MLP sublayer *reads* from the bus and *writes* an update back. Layers communicate by writing into subspaces that later layers read. This view (popularized by mechanistic-interpretability work) is a helpful mental model for why depth works and why the final layer can read out a prediction.

### 7. LayerNorm, and Pre-LN vs Post-LN

**Layer normalization** rescales each token's vector to zero mean and unit variance across its $d$ features, then applies a learned scale $\gamma$ and shift $\beta$:

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2,
$$

where $\odot$ is elementwise multiplication and $\epsilon$ (e.g. $10^{-5}$) avoids division by zero. It keeps activations in a stable range as they pass through many layers.

Where to put it matters:

- **Post-LN** (original Transformer): $x \leftarrow \mathrm{LN}(x + F(x))$. The normalization sits *on* the residual path, so every layer's output is renormalized. Xiong et al. (2020) showed that at initialization, gradients near the output layers are large, which is why Post-LN needs a careful learning-rate warmup and becomes unstable in deep models.
- **Pre-LN** (GPT-2 onward): $x \leftarrow x + F(\mathrm{LN}(x))$, with one final LN before the output projection. The residual path is a clean sum, gradients are well behaved, and training is stable with larger learning rates. Every modern LLM uses a Pre-norm design (usually with RMSNorm, [1.4](04-modern-block.md)).

### 8. The output layer

After the last block, a final norm and an **unembedding** matrix $W_U \in \mathbb{R}^{d \times V}$ produce `logits: [B, T, V]`, and cross-entropy against the shifted targets gives the loss from [1.1](01-language-modeling.md). Some models tie $W_U = E^\top$ to save parameters ([1.4](04-modern-block.md)).

## Code

A complete GPT in the style of GPT-2: learned positions, Pre-LN, GELU MLP. The first block defines the model, the second trains it.

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V fused
        self.proj = nn.Linear(d_model, d_model)      # W_O

    def forward(self, x):                            # x: [B, T, d]
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)       # each [B, T, d]
        # split d into h heads of size dh: [B, T, d] -> [B, h, T, dh]
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)        # [B, h, T, T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))             # no peeking ahead
        out = F.softmax(scores, dim=-1) @ v                          # [B, h, T, dh]
        out = out.transpose(1, 2).reshape(B, T, d)                   # concat heads
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.up = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = CausalSelfAttention(d_model, n_heads), MLP(d_model)

    def forward(self, x):                  # Pre-LN residual block
        x = x + self.attn(self.ln1(x))     # tokens exchange information
        x = x + self.mlp(self.ln2(x))      # each token processed independently
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)     # learned absolute positions
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # unembedding

    def forward(self, idx, targets=None):                # idx: [B, T] token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # [B, T, d]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                 # [B, T, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.pos_emb.num_embeddings:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
```

Training on characters (save both blocks in one file; it trains on its own source code so it needs no download, but any text file works):

```python
torch.manual_seed(0)
text = open(__file__).read() * 20                  # replace with any text file
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
model = GPT(vocab_size=len(chars))
print(sum(p.numel() for p in model.parameters()), "parameters")
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
B, T = 32, 64
for step in range(301):
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])          # [B, T]
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])  # shifted by one
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(step, round(loss.item(), 3))
out = model.generate(x[:1, :10], 80)
print("".join(chars[i] for i in out[0].tolist()))
```

This ~0.84M-parameter model trains in seconds on a CPU; the loss falls from about $\ln V$ to roughly 0.1–0.2 within 300 steps because it memorizes the short repeated text. On a real corpus, track a held-out loss instead. In real code, replace the explicit softmax attention with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which dispatches to fused FlashAttention-style kernels ([1.6](06-long-context-and-efficient-attention.md)).

## Worked example

**A tiny attention computation.** Three tokens, $d_k = 2$. We compute the output for token 3 (which may attend to tokens 1–3). Let $q_3 = (1, 0)$, keys $k_1 = (1, 0)$, $k_2 = (0, 1)$, $k_3 = (1, 1)$, and values $v_1 = (1, 0)$, $v_2 = (0, 2)$, $v_3 = (3, 1)$.

- Raw scores $q_3 \cdot k_j$: $(1, 0, 1)$. Scaled by $1/\sqrt{2}$: $(0.707, 0, 0.707)$.
- Softmax: $e^{0.707} = 2.028$, $e^{0} = 1$, so weights $= (2.028, 1, 2.028) / 5.056 = (0.401, 0.198, 0.401)$.
- Output: $0.401\,(1,0) + 0.198\,(0,2) + 0.401\,(3,1) = (1.604, 0.797)$.

Token 3 mostly mixes in tokens 1 and 3, whose keys align with its query.

**Parameter count for a 7B-class model.** Take the Llama 2 7B configuration: $L = 32$, $d = 4096$, $V = 32{,}000$, and a SwiGLU FFN with three $d \times d_{ff}$ matrices, $d_{ff} = 11{,}008$ (why three matrices and this odd size is explained in [1.4](04-modern-block.md)). No biases.

| component | formula | count |
|---|---|---|
| attention per layer ($W_Q, W_K, W_V, W_O$) | $4d^2$ | 67,108,864 |
| FFN per layer | $3 d\, d_{ff}$ | 135,266,304 |
| all 32 layers | $32 \times (4d^2 + 3d\,d_{ff})$ | 6,476,005,376 |
| input embedding + output layer (untied) | $2Vd$ | 262,144,000 |
| norm weights | $2dL + d$ | 266,240 |
| **total** | | **6,738,415,616 ≈ 6.74B** |

This matches the 6.74B usually quoted for "Llama 2 7B". For a GPT-style model with a $4d$ GELU MLP the per-layer count is $4d^2 + 8d^2 = 12d^2$, so $N \approx 12Ld^2$; GPT-3 175B ($L = 96$, $d = 12{,}288$) gives $12 \times 96 \times 12{,}288^2 \approx 173.9\text{B}$, plus 0.6B of embeddings.

**FLOPs per token.** A matrix–vector product with an $m \times n$ weight costs $mn$ multiply–adds = $2mn$ FLOPs. Every parameter participates in one multiply–add per token, so:

- **Forward pass:** $\approx 2N$ FLOPs per token.
- **Backward pass:** $\approx 4N$ (gradients with respect to both the activations and the weights, each costing about as much as the forward).
- **Training:** $\approx 6N$ FLOPs per token. **Inference:** $\approx 2N$ per generated token.

This ignores the attention score computation itself ($QK^\top$ and $AV$), which adds roughly $4 L T d$ forward FLOPs per token at context length $T$ (half that if the causal mask is exploited). For our 7B at $T = 4096$ that is $4 \times 32 \times 4096 \times 4096 \approx 2.1$ GFLOPs versus $2N \approx 13.5$ GFLOPs: noticeable, but $6N$ remains a good first estimate at moderate context lengths.

**Training cost.** Llama 2 7B was trained on 2T tokens: $6 \times 6.74 \times 10^9 \times 2 \times 10^{12} \approx 8.1 \times 10^{22}$ FLOPs. An A100 peaks at 312 TFLOP/s in BF16; at a realistic 40% utilization that is ~125 TFLOP/s, so the run needs $8.1 \times 10^{22} / 1.25 \times 10^{14} \approx 6.5 \times 10^8$ GPU-seconds $\approx$ 180,000 GPU-hours. The Llama 2 paper reports 184,320 A100 GPU-hours for the 7B model, so this back-of-envelope estimate is right to within a few percent.

## In practice (2025–2026)

- **Decoder-only, Pre-norm** is universal for frontier LLMs. Encoder–decoder models (T5) and encoder-only models (BERT) remain useful for embeddings and classification but not for frontier generation.
- **Sizes.** Hidden widths of 4096–8192 and 32–126 layers are typical for dense models from 7B to 405B (Llama 3 405B: 126 layers, $d = 16{,}384$, 128 heads). Head size $d_h = 128$ is by far the most common choice.
- **Biases** are mostly removed from linear layers (Llama, DeepSeek); some models keep biases in the QKV projection (Qwen2) or attention (gpt-oss).
- **The block has been upgraded but not replaced**: RMSNorm instead of LayerNorm, RoPE instead of learned positions, SwiGLU instead of GELU, GQA/MLA instead of MHA, MoE instead of dense FFNs. Chapter [1.4](04-modern-block.md) makes each upgrade.

## Common pitfalls / misconceptions

- **Forgetting the mask** gives a training loss that drops suspiciously fast to near zero (the model copies the answer) and a model that generates garbage.
- **Wrong reshape order.** `x.view(B, T, h, dh).transpose(1, 2)` is correct; `x.view(B, h, T, dh)` silently scrambles tokens and heads.
- **"Attention is all there is."** By parameter count the FFN usually dominates (two-thirds in a standard dense block). Attention moves information between positions; the FFN does most per-token computation.
- **"6N is exact."** It omits attention-score FLOPs, embeddings lookups and recomputation from activation checkpointing ([1.8](08-training-at-scale.md)); treat it as the leading-order estimate.
- **Positional information is not automatic.** Without position embeddings (or RoPE) a causal Transformer can still infer some order from the mask, but performance and length generalization suffer.

## Key takeaways

- Attention computes $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$: each token takes a weighted average of earlier tokens' values, weighted by query–key similarity.
- The $\sqrt{d_k}$ factor keeps score variance at 1 so the softmax does not saturate; the causal mask enables parallel next-token training.
- A block is `x + Attn(Norm(x))` followed by `x + FFN(Norm(x))`; Pre-norm with residual connections is what lets deep stacks train stably.
- Parameters $\approx L(4d^2 + \text{FFN}) + $ embeddings; training costs $\approx 6N$ FLOPs per token and inference $\approx 2N$.
- These formulas are good enough to estimate a 7B training run to within a few percent of the reported GPU-hours.

## Check your understanding

??? question "Why does attention without positional information treat the input as a set?"
    The score $q_i \cdot k_j$ depends only on the contents of tokens $i$ and $j$, not on their positions. If you permute the input tokens, the same scores appear in permuted order and the outputs are permuted identically. (The causal mask breaks full symmetry, but only weakly.)

??? question "With $d_k = 64$ and no scaling, what is the standard deviation of attention scores at initialization (unit-variance q and k)? What does scaling change?"
    $\sqrt{64} = 8$. Scores of order ±8 to ±16 make softmax nearly one-hot. Dividing by $\sqrt{64} = 8$ restores unit variance.

??? question "Estimate the parameter count of a GPT-style model with $L = 24$, $d = 2048$, $V = 50{,}000$ and tied embeddings."
    $12 L d^2 = 12 \times 24 \times 2048^2 \approx 1.21\text{B}$, plus $Vd \approx 0.10\text{B}$ for the shared embedding, about 1.3B total.

??? question "Why is training about 3× more expensive per token than inference?"
    The backward pass needs gradients with respect to both activations and weights, each costing about as much as the forward pass: $2N + 4N = 6N$ vs $2N$.

??? question "What goes wrong with Post-LN in very deep models, and what does Pre-LN change?"
    In Post-LN the normalization sits on the residual path, and at initialization the gradients for layers near the output are large, so training needs a long warmup and can diverge. Pre-LN keeps the residual path an identity sum, giving well-scaled gradients at every depth.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 | The Transformer: scaled dot-product attention, multi-head attention, residual connections, positional encodings. Focus on §3. |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | The decoder-only, pre-norm GPT recipe that the code in this chapter follows. |
| :material-book-open-variant: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al., 2020 | Explains why **Pre-LN** trains stably without warmup tricks; now universal. |
| :material-book-open-variant: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023 | Source of the 7B configuration and the GPU-hour figure used in the worked example. |
