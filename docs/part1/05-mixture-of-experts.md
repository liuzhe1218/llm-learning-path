# 1.5 Mixture-of-Experts

!!! abstract "What you'll learn"
    - Why sparse **Mixture-of-Experts (MoE)** layers decouple total parameters from per-token compute.
    - How a **router** picks top-$k$ experts, and the difference between softmax and sigmoid gating.
    - The **load-balancing** problem: derive the Switch Transformer auxiliary loss, understand capacity factors and token dropping, and DeepSeek-V3's auxiliary-loss-free bias balancing.
    - **Fine-grained** and **shared** experts, and **expert parallelism** with all-to-all communication.
    - How to reconstruct DeepSeek-V3's 671B total / 37B active parameters from its config, and a runnable top-$k$ MoE layer.

## Why this matters / the problem

Scaling laws ([1.7](07-scaling-laws-and-data.md)) say that bigger models reach lower loss. But in a dense Transformer every parameter is used for every token, so a model twice as big costs twice as much per token to train and to serve. Mixture-of-Experts breaks that link: it replaces the FFN with many parallel FFNs ("experts") and uses only a few of them for each token. A model can then hold, say, 671B parameters of knowledge while spending the compute of a ~37B model per token. Since 2024 nearly every frontier open model (DeepSeek-V3, Qwen3-235B-A22B, Kimi K2, gpt-oss, Llama 4) is an MoE, so this layer is now as standard as attention.

## Intuition

Picture a hospital. A dense model is one doctor who must know everything and sees every patient. An MoE is a hospital with many specialists and a triage nurse (the **router**). Each patient (token) sees the two or eight most relevant specialists, not all of them. The hospital as a whole knows far more than any one doctor, but each visit costs only a few consultations.

Two things can go wrong. First, the nurse may send everyone to the same popular specialist while others sit idle (**load imbalance**, which wastes hardware and stops idle experts from learning). Second, the specialists may work in different buildings, so sending patients around is slow (**communication cost** in expert parallelism). Most of MoE engineering is about these two problems.

```text
                 ┌─► expert 1 ─┐
 token x ─► router (scores) ─► top-k ─┤   expert 2   (skipped)
                 └─► expert 7 ─┘ ─► weighted sum g1·E1(x) + g7·E7(x)  (+ shared expert(s))
```

## How it works

### Dense vs sparse compute

Let $N_{\text{total}}$ be all parameters and $N_{\text{active}}$ the parameters actually used for one token. In a dense model they are equal. In an MoE with $E$ experts per layer, each of size $P_e$, and $k$ experts used per token, the FFN part contributes $E \cdot P_e$ to $N_{\text{total}}$ but only $k \cdot P_e$ to $N_{\text{active}}$. Training compute per token is $\approx 6 N_{\text{active}}$ ([1.3](03-transformer.md)), while memory to hold the weights scales with $N_{\text{total}}$.

### The router and top-$k$ gating

For a token with hidden state $x \in \mathbb{R}^d$, a router with weight matrix $W_r \in \mathbb{R}^{d \times E}$ computes one score per expert, $s = x W_r \in \mathbb{R}^E$. Then:

1. **Score → affinity.** Either a softmax over all experts, $p_i = e^{s_i} / \sum_j e^{s_j}$ (Switch, Mixtral, Qwen3), or an independent sigmoid per expert, $p_i = \sigma(s_i)$ (DeepSeek-V3).
2. **Select** the $k$ experts with the largest affinities: the set $\mathcal{T}(x)$.
3. **Gate values.** $g_i = p_i / \sum_{j \in \mathcal{T}(x)} p_j$ for $i \in \mathcal{T}(x)$ and 0 otherwise (renormalizing over the selected experts; some models skip this step).
4. **Combine:**

$$
y = \sum_{i \in \mathcal{T}(x)} g_i \, \mathrm{FFN}_i(x) \;+\; \sum_{j=1}^{E_s} \mathrm{FFN}^{\text{shared}}_j(x),
$$

where the second sum over $E_s$ shared experts is present only in shared-expert designs (below). In words: the output is a weighted mix of the chosen experts' outputs. Gradients flow into the router through the gate values $g_i$, so the router learns which experts help for which tokens. Top-$k$ selection itself is not differentiable; the router only learns through the gates of the experts it did pick.

**Softmax vs sigmoid.** With softmax, experts compete: raising one affinity lowers all others, and the affinities form a distribution. With sigmoid, each affinity is an independent "how relevant is this expert" in $(0, 1)$, which pairs naturally with very many experts and with the bias-based balancing below. After normalization over the selected set, both give gates summing to 1.

### Load balancing and the auxiliary loss

Left alone, routing collapses: an expert that is slightly better early on receives more tokens, gets more gradient, becomes better still, and soon a few experts take everything. This wastes capacity and, with experts spread over GPUs, makes the busiest GPU the bottleneck for everyone.

The **Switch Transformer** (Fedus et al., 2021) adds an auxiliary loss. Over a batch $\mathcal{B}$ of $n$ tokens, define for each expert $i$:

- $f_i$ = fraction of tokens dispatched to expert $i$ (with top-1 routing, $f_i = \frac{1}{n}\sum_{x \in \mathcal{B}} \mathbb{1}[\arg\max p(x) = i]$),
- $P_i$ = average router probability for expert $i$, $P_i = \frac{1}{n}\sum_{x \in \mathcal{B}} p_i(x)$.

The loss is

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \, P_i ,
$$

added to the language-modeling loss with a small coefficient $\alpha$ (Switch used $10^{-2}$).

**Why this form?** We want both $f$ and $P$ to be uniform ($1/E$ each).

- $f_i$ is a count, so it has no gradient. $P_i$ is differentiable. Treating $f$ as a constant, $\partial \mathcal{L}_{\text{aux}} / \partial P_i = \alpha E f_i$: the router probability of each expert is pushed down in proportion to how overloaded that expert is. Busy experts are discouraged most.
- At equilibrium, when the router's probabilities are consistent with its choices ($f \approx P$), the sum becomes $\sum_i P_i^2$. By the Cauchy–Schwarz inequality, $\big(\sum_i P_i\big)^2 \le E \sum_i P_i^2$, and since $\sum_i P_i = 1$ this gives $\sum_i P_i^2 \ge 1/E$, with equality exactly when $P_i = 1/E$ for all $i$. So the loss is minimized by uniform routing.
- The factor $E$ normalizes the minimum to $\alpha \cdot E \cdot E \cdot (1/E)(1/E) = \alpha$, independent of the number of experts, so the same $\alpha$ works for any $E$.

For top-$k$ routing, $f_i$ counts all $k$ assignments per token (normalized so $\sum_i f_i = 1$). The coefficient is a trade-off: too small and routing collapses; too large and the balance term fights the LM objective and hurts quality.

### Capacity factor and token dropping

Hardware wants fixed-size tensors, so classic MoE implementations give each expert a fixed buffer, its **capacity**:

$$
\text{capacity} = \Big\lceil \frac{k \cdot n}{E} \cdot \mathrm{CF} \Big\rceil,
$$

where $n$ is the number of tokens in the batch and $\mathrm{CF}$ is the **capacity factor** (e.g. 1.0–1.25). If more tokens choose an expert than its capacity, the excess tokens are **dropped**: that expert does not process them, and they pass through the layer only via the residual connection. A larger CF drops fewer tokens but wastes compute and memory on padding. Modern implementations increasingly use "dropless" kernels with variable-sized groups (grouped GEMMs) so no tokens are dropped; DeepSeek-V3 reports dropping no tokens in training or inference.

### Fine-grained and shared experts

**DeepSeekMoE** (Dai et al., 2024) made two changes that most 2025 MoEs adopted:

1. **Fine-grained experts.** Split each expert into $m$ smaller ones (hidden size divided by $m$) and route to $m$ times as many. Compute is unchanged, but the number of possible expert combinations explodes. With 16 experts and top-2 there are $\binom{16}{2} = 120$ combinations; splitting each into 4 gives 64 experts and top-8, with $\binom{64}{8} = 4{,}426{,}165{,}368$ combinations. Each token can get a much more specific mixture of knowledge.
2. **Shared experts.** One or a few experts that *every* token uses, outside the router. They capture common knowledge (basic syntax, frequent patterns) so that the routed experts do not each have to relearn it, and can specialize instead.

### Auxiliary-loss-free balancing (DeepSeek-V3)

DeepSeek-V3 replaces most of the auxiliary loss with a simple control loop. Each expert $i$ has a bias $b_i$ that is **added to its affinity only for selecting the top-$k$**:

$$
\mathcal{T}(x) = \operatorname{top-}k_i\big(p_i(x) + b_i\big), \qquad g_i = \frac{p_i(x)}{\sum_{j \in \mathcal{T}(x)} p_j(x)} \text{ for } i \in \mathcal{T}(x).
$$

The gate value uses the original affinity $p_i$, so the bias changes *which* experts are chosen but not *how much* each chosen expert contributes. After each training step, every overloaded expert has its bias decreased by a step $\gamma$ and every underloaded expert has it increased by $\gamma$ (DeepSeek-V3 reports $\gamma = 0.001$ for most of training). The bias receives no gradient; it is a feedback controller. Because no balancing term is added to the loss, the LM gradient is not distorted. DeepSeek-V3 keeps only a very small sequence-level balance loss as a safeguard against extreme imbalance within a single sequence.

### Expert parallelism and all-to-all

A 671B-parameter MoE does not fit on one GPU, and its experts are the bulk of the weights. **Expert parallelism (EP)** places different experts on different GPUs. One MoE layer then works like this:

1. Each GPU routes its own tokens.
2. **All-to-all dispatch:** every GPU sends each token's hidden state to the GPUs that hold that token's chosen experts.
3. Each GPU runs its local experts on the tokens it received (one batched matrix multiply per expert).
4. **All-to-all combine:** results are sent back to the tokens' home GPUs and combined with gate weights.

This is two all-to-all exchanges per MoE layer in the forward pass (and two more in the backward). The communication volume grows with $k$, and crossing nodes (slower interconnect than within a node) is expensive. DeepSeek-V3 therefore uses **node-limited routing**: each token's experts are restricted to at most 4 nodes. Overlapping this communication with computation is a major part of MoE systems engineering ([1.8](08-training-at-scale.md); for serving see [Part 3](../part3/index.md)).

## Code

A runnable top-$k$ MoE layer with optional shared experts and the Switch-style auxiliary loss. It loops over experts (not tokens), which is the simplest correct implementation; production kernels sort tokens by expert and use grouped matrix multiplies.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):                      # a small SwiGLU FFN
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, top_k=2, n_shared=0, aux_coef=1e-2):
        super().__init__()
        self.E, self.k, self.aux_coef = n_experts, top_k, aux_coef
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, d_ff) for _ in range(n_shared))

    def forward(self, x):                                 # x: [B, T, d]
        B, T, d = x.shape
        x = x.reshape(-1, d)                              # [N, d], N = B*T tokens
        probs = F.softmax(self.router(x), dim=-1)         # [N, E] router probabilities
        gate, idx = probs.topk(self.k, dim=-1)            # [N, k] each
        gate = gate / gate.sum(-1, keepdim=True)          # renormalize over chosen experts
        out = torch.zeros_like(x)
        for e in range(self.E):                           # loop over experts, not tokens
            tok, slot = (idx == e).nonzero(as_tuple=True) # which tokens chose expert e
            if tok.numel():
                out.index_add_(0, tok, gate[tok, slot, None] * self.experts[e](x[tok]))
        for s in self.shared:                             # shared experts see every token
            out = out + s(x)
        # Switch-style load-balancing loss: E * sum_i f_i * P_i
        f = F.one_hot(idx, self.E).float().sum(1).mean(0) / self.k   # fraction of assignments
        P = probs.mean(0)                                            # mean router prob
        aux = self.aux_coef * self.E * (f * P).sum()
        return out.view(B, T, d), aux, f

torch.manual_seed(0)
moe = MoE(d=64, d_ff=128, n_experts=8, top_k=2, n_shared=1)
y, aux, load = moe(torch.randn(4, 32, 64))
print(y.shape, round(aux.item(), 4), [round(v, 3) for v in load.tolist()])
```

**Watching collapse happen.** We train the layer to imitate a random linear map, after nudging the router toward expert 0 at initialization, once without and once with the auxiliary loss:

```python
def run(aux_coef, steps=500):
    torch.manual_seed(1)
    m = MoE(d=32, d_ff=64, n_experts=8, top_k=2, aux_coef=aux_coef)
    with torch.no_grad():
        m.router.weight[0] += 0.15           # bias the router toward expert 0 at init
    target = nn.Linear(32, 32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x = torch.randn(16, 32, 32) + 0.5
        y, aux, load = m(x)
        loss = F.mse_loss(y, target(x).detach()) + aux
        opt.zero_grad(); loss.backward(); opt.step()
    return [round(v, 2) for v in load.tolist()]

print("no aux  :", run(0.0))    # [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49, 0.0]
print("with aux:", run(1e-1))   # [0.12, 0.12, 0.13, 0.13, 0.14, 0.13, 0.12, 0.12]
```

Without the loss, two experts take all assignments and six never train (exact numbers may vary across PyTorch versions). With it, the load is close to uniform ($1/8 = 0.125$).

**Auxiliary-loss-free variant.** The DeepSeek-V3 rule as a subclass: sigmoid affinities, a non-learned bias used only for selection, and a sign-based update.

```python
class BiasBalancedMoE(MoE):
    """DeepSeek-V3-style routing: sigmoid affinities, bias used only for selection."""
    def __init__(self, *a, gamma=1e-3, **kw):
        super().__init__(*a, aux_coef=0.0, **kw)
        self.gamma = gamma
        self.register_buffer("bias", torch.zeros(self.E))

    def forward(self, x):
        B, T, d = x.shape
        x = x.reshape(-1, d)
        s = torch.sigmoid(self.router(x))                       # [N, E] affinities in (0, 1)
        _, idx = (s + self.bias).topk(self.k, dim=-1)           # bias affects WHO is chosen...
        gate = s.gather(-1, idx)                                # ...but not the gate VALUE
        gate = gate / gate.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, gate[tok, slot, None] * self.experts[e](x[tok]))
        for sh in self.shared:
            out = out + sh(x)
        load = F.one_hot(idx, self.E).float().sum((0, 1))       # tokens per expert
        if self.training:                                       # no gradient: a control loop
            self.bias += self.gamma * torch.sign(load.mean() - load)
        return out.view(B, T, d), torch.zeros(()), load / load.sum()
```

Trained the same way as above with `gamma=1e-2` (and no auxiliary term), the load ends up roughly balanced (around 0.09–0.2 per expert in our run).

## Worked example

**DeepSeek-V3: where do 671B total and 37B active come from?** The published configuration: $L = 61$ layers, $d = 7168$; the first 3 layers have a dense SwiGLU FFN with $d_{ff} = 18{,}432$; the other 58 layers are MoE with 256 routed experts plus 1 shared expert, each a SwiGLU with hidden size 2048, and 8 routed experts active per token; MLA attention with 128 heads; vocabulary 129,280 (the "128K" vocabulary), untied embeddings.

| component | calculation | total params | active params |
|---|---|---|---|
| one expert | $3 \times 7168 \times 2048$ | 44.0M | |
| MoE FFNs | $58 \times 257 \times 44.0\text{M}$ | 656.5B | $58 \times 9 \times 44.0\text{M} = 23.0\text{B}$ |
| dense FFNs | $3 \times 3 \times 7168 \times 18{,}432$ | 1.19B | 1.19B |
| MLA attention | $61 \times 187.1\text{M}$ (projections below) | 11.4B | 11.4B |
| embeddings + output | $2 \times 129{,}280 \times 7168$ | 1.85B | 1.85B |
| routers, norms | $58 \times 256 \times 7168 + \dots$ | 0.11B | 0.11B |
| **sum** | | **≈ 671.0B** | **≈ 37.6B** |

The MLA projections per layer are: query down-projection $7168 \times 1536$, query up-projection $1536 \times 128 \times (128 + 64)$, KV down-projection plus RoPE key $7168 \times (512 + 64)$, key/value up-projections $512 \times 128 \times (128 + 128)$, and output $128 \times 128 \times 7168$, totalling 187.1M. The result matches the reported 671B total and 37B activated (the report counts the separate 14B multi-token-prediction module on top, for 685B). Notice: 98% of parameters sit in routed experts, but only 9 of 257 experts per layer run per token. Training compute per token is therefore about $6 \times 37\text{B}$, roughly the cost of a 37B dense model, for a model with 18× more parameters.

**Capacity.** A batch has $n = 4096$ tokens at one MoE layer, $E = 64$ experts, $k = 2$, $\mathrm{CF} = 1.25$. Capacity $= \lceil 2 \times 4096 / 64 \times 1.25 \rceil = 160$ tokens per expert. If 200 assignments go to one expert, 40 are dropped for that expert.

## In practice (2025–2026)

| model | experts per MoE layer | active | shared | gating / balancing |
|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 0 | softmax top-2; 47B total / 13B active |
| DeepSeek-V3 | 256 | 8 | 1 | sigmoid; bias-based (aux-loss-free) + tiny sequence loss |
| Qwen3-235B-A22B | 128 | 8 | 0 | softmax top-8; global-batch balancing loss |
| Kimi K2 | 384 | 8 | 1 | DeepSeek-V3-style |
| gpt-oss-120b | 128 | 4 | 0 | top-4 |

- **Fine-grained experts are the norm**: hundreds of small experts with top-8 rather than 8 large experts with top-2.
- **Sparsity is increasing.** Kimi K2 has 1.04T total and 32B active parameters; its report argues that, at a fixed number of active parameters, more total experts keep lowering loss.
- **Balancing at the right granularity.** Qwen3 computes its balancing loss over the global batch rather than per micro-batch, which lets experts specialize by domain instead of forcing every small batch to be uniform.
- **Not every layer is MoE.** DeepSeek-V3 keeps the first 3 layers dense, Kimi K2 the first 1, because routing in the very first layers works poorly and early layers process generic features.

## Common pitfalls / misconceptions

- **"A 671B MoE is as cheap to serve as a 37B dense model."** Compute per token is, but all 671B parameters must sit in memory, and at small batch sizes each token reads different experts, so memory bandwidth and communication, not FLOPs, dominate.
- **"Experts specialize in topics."** Analyses usually find that experts specialize in token types and syntactic patterns more than in human-interpretable topics like "biology".
- **Forgetting the gates.** Summing expert outputs without the gate weights gives the router no gradient, and it never learns.
- **Over-weighting the auxiliary loss.** Perfect balance is not the goal; good LM loss with acceptable balance is. Large $\alpha$ hurts quality.
- **Comparing MoE and dense models by total parameters.** Compare by active parameters (compute) *and* total parameters (memory, knowledge capacity).

## Key takeaways

- MoE replaces the FFN with $E$ experts and a router that activates $k$ of them, decoupling total parameters (knowledge, memory) from active parameters (FLOPs).
- Without balancing, routing collapses; the Switch loss $\alpha E \sum_i f_i P_i$ pushes overloaded experts down and is minimized by uniform routing.
- DeepSeek-V3's bias-based balancing steers expert selection with a feedback loop instead of a loss term, avoiding interference with the LM objective.
- Fine-grained experts (many small) plus shared experts (always on) are the 2025 default.
- Expert parallelism needs two all-to-all exchanges per MoE layer; communication, not FLOPs, is often the bottleneck.

## Check your understanding

??? question "An MoE has 64 experts of 50M parameters each in each of 40 layers, top-4 routing, and 3B of non-expert parameters. What are total and active parameters?"
    Experts: $64 \times 50\text{M} \times 40 = 128\text{B}$; total $= 131\text{B}$. Active: $4 \times 50\text{M} \times 40 = 8\text{B}$ plus 3B, so 11B.

??? question "Why does the Switch loss use the product $f_i P_i$ rather than just $\sum_i f_i^2$?"
    $f_i$ comes from an argmax (a count), which has zero gradient. $P_i$ is differentiable, so the product lets gradients flow into the router, scaled by how overloaded each expert currently is.

??? question "In DeepSeek-V3's scheme, what would go wrong if the bias were also used in the gate value?"
    The bias would then change each expert's contribution to the output, not just the selection, so the balancing controller would directly distort the model's function and the LM loss. Keeping it out of the gate isolates balancing from the output.

??? question "Why do fine-grained experts help even though compute per token is unchanged?"
    The number of possible expert combinations per token grows enormously (120 vs over 4 billion in the DeepSeekMoE example), so each token can receive a more specific combination of knowledge, and each expert can specialize more narrowly.

??? question "With expert parallelism, why might top-8 routing be more expensive than top-2 at equal active FLOPs?"
    Each token's hidden state must be sent to every GPU holding one of its experts. Top-8 can mean up to 8 destinations per token instead of 2, increasing all-to-all traffic; node-limited routing is one mitigation.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017 | The original top-k gated MoE with load-balancing loss (in LSTMs). |
| :material-book-open-variant: [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020 | MoE Transformers at 600B scale with **expert parallelism** across devices. |
| :material-star: [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021 | Top-1 routing, capacity factor, the auxiliary loss derived above, bf16 stability tricks. The clearest intro to MoE engineering. |
| :material-book-open-variant: [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) — Zoph et al., 2022 | The router z-loss and a practical guide to MoE training instabilities and fine-tuning. |
| :material-star: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024 | 8 experts, top-2, 47B total / 13B active; the model that made open MoE mainstream. |
| :material-star: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024 | **Fine-grained experts** + **shared experts**: the design adopted by DeepSeek-V2/V3, Kimi K2 and others. |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 671B total / 37B active; **auxiliary-loss-free load balancing**, node-limited routing, and the systems work that makes it trainable. |
