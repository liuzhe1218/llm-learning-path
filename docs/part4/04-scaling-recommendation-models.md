# 4.4 Scaling Recommendation Models

!!! abstract "What you'll learn"
    - Why classic DLRMs stopped improving with size: parameters in embeddings, low hardware utilization, shallow interactions
    - What a "scaling law" means for recommenders, and what has to change in the architecture to get one
    - Generative Recommenders and the HSTU layer in depth: the sequential-transduction reformulation, pointwise aggregated attention, and the reported results
    - Wukong (stacked FMs), RankMixer (token mixing + per-token FFNs), MTGR (keeping cross features), LUM (three-step paradigm) and MARM (caching)
    - How to budget compute for a ranking model: FLOPs per candidate × candidates × QPS

## Why this matters / the problem

[Part 1](../part1/index.md) showed that LLM loss falls as a smooth power law in compute, so companies can plan: "10× more compute buys this much improvement". Recommenders did not behave this way. For years the main lever was adding features and growing embedding tables, and gains per unit of hardware kept shrinking. By the early 2020s industrial rankers had terabytes of embeddings yet dense networks of only tens of millions of parameters, and making the dense part bigger often did nothing.

Since 2024 a series of papers, most prominently Meta's **Generative Recommenders (HSTU)**, have shown that recommenders *can* scale like LLMs if the architecture spends compute in the right place. This is the "large recommendation model" wave, and it is reshaping ranking in Meta, ByteDance, Kuaishou, Meituan, Alibaba and others. This chapter explains what changed.

## Intuition

Think of a DLRM as a library with an enormous card catalogue (embedding tables) and a tiny reading room (the dense network). Adding more catalogue cards stops helping once every item already has a card; what is missing is *thinking* capacity to combine the information. But the reading room was built from small, oddly-shaped modules designed for CPUs (hand-made crosses, many small MLPs), so GPUs sit mostly idle and enlarging the room is expensive.

The scaling recipe has three parts:

1. **Put compute into the dense network**, not memory into embeddings.
2. **Use homogeneous, GPU-friendly blocks** (Transformer-like layers, big matrix multiplies) so that FLOPs are actually used (high **MFU**, model FLOPs utilization).
3. **Feed the model a rich sequence** (the user's raw action history) instead of hand-aggregated features, so there is something for the extra compute to learn.

## How it works

### Why DLRMs plateau

1. **Parameters live in embeddings.** An embedding lookup costs almost no FLOPs. A trillion-parameter DLRM may do only tens of MFLOPs per example. Parameter count is the wrong axis: what matters for quality in LLMs is compute per token, and DLRMs have very little.
2. **Low MFU.** Many small heterogeneous modules (feature crosses, gating, small MLPs per feature group) are memory-bandwidth-bound. RankMixer reports that its previous production ranker ran at **4.5% MFU**: 95% of the GPU's arithmetic was wasted.
3. **Shallow, fixed interactions.** FM/DCN layers capture low-order interactions; stacking more of them does not keep helping. And enlarging embeddings can trigger **embedding collapse** (Chapter [4.2](02-ranking-and-ctr-models.md)).
4. **Hand-engineered features cap the signal.** Aggregates like "user's CTR on this category in the last 7 days" throw away information a bigger model could use.

### What "scaling law" means for recsys

A scaling law is an empirical fit of the form

$$
\mathcal{L}(C) \approx \mathcal{L}_\infty + a\, C^{-\alpha},
$$

where $C$ is training compute (FLOPs), $\mathcal{L}$ a quality metric (e.g. loss, or negated NDCG), $\mathcal{L}_\infty$ an irreducible floor and $a, \alpha > 0$ fitted constants. In words: each multiplicative increase in compute buys a predictable improvement. For recommenders it matters for the same reason as for LLMs — capacity planning — with a twist: **inference cost per request is the binding constraint**, because the model runs over hundreds of candidates for every request, not once per generated token. Results therefore often report quality vs *inference* FLOPs per example and must hold within a fixed latency budget.

Early evidence came from ID-based sequential recommenders scaled to 0.8B parameters, showing power-law behaviour even when data-constrained. The breakthrough was showing it at industrial scale.

### Generative Recommenders and HSTU

**The reformulation.** HSTU's paper ("Actions Speak Louder than Words") recasts both ranking and retrieval as **sequential transduction** over a user's chronological stream of (item, action) events. Write the history as $(\Phi_0, a_0), (\Phi_1, a_1), \dots$ where $\Phi_i$ is the $i$-th item shown or engaged with and $a_i$ the user's action on it.

- **Retrieval** becomes "predict the next item $\Phi_{i+1}$ given the history", supervised only where the user's action on that next item was positive — exactly SASRec-style next-item prediction (Chapter [4.3](03-sequential-user-modeling.md)).
- **Ranking** interleaves items and actions into one token sequence, $\Phi_0, a_0, \Phi_1, a_1, \dots, \Phi_n$, and at each item token predicts the action that follows. The candidate item is simply the last item token; the model outputs its predicted action (click, like, watch…).

Most hand-engineered features disappear: slowly changing categorical features (followed creators, demographics) are merged into the same time series at the points where they change, and numerical aggregate features (counters, historical CTRs) are dropped on the premise that a large enough sequence model can infer them from the raw actions. The model becomes a *generative model of user behaviour*, trained like an LM on the action stream — hence "Generative Recommenders (GR)".

```text
 classic DLRM sample:   [user feats | item feats | cross feats | pooled history]  ──► MLP ──► pCTR

 GR ranking sequence:   Φ0  a0  Φ1  a1  Φ2  a2  ...  Φn   ?
                        │       │       │            │
 targets at item tokens:└─► a0  └─► a1  └─► a2       └─► predicted action on candidate Φn

 GR retrieval:          Φ0  Φ1  Φ2 ... Φn ──► next item Φn+1   (loss only where action was positive)
```

One user's whole history is one training sequence, and every position provides a label, so a single forward pass trains on many impressions at once — the same "every token is a target" efficiency that makes LM pre-training data-efficient.

**The HSTU layer.** HSTU (Hierarchical Sequential Transduction Unit) is a modified Transformer block. With input $\mathbf{X} \in \mathbb{R}^{n\times d}$ ($n$ tokens):

$$
\mathbf{U}, \mathbf{V}, \mathbf{Q}, \mathbf{K} = \text{Split}\big(\phi_1(f_1(\mathbf{X}))\big)
$$

$$
\mathbf{A}\mathbf{V} = \phi_2\big(\mathbf{Q}\mathbf{K}^\top + \mathbf{rab}^{p,t}\big)\,\mathbf{V}
$$

$$
\mathbf{Y} = f_2\big(\text{Norm}(\mathbf{A}\mathbf{V}) \odot \mathbf{U}\big)
$$

Here $f_1, f_2$ are linear projections, $\phi_1, \phi_2$ are SiLU nonlinearities, $\mathbf{rab}^{p,t}$ is a relative attention bias that depends on both position $p$ and timestamp difference $t$, and $\odot$ is element-wise multiplication. In words:

- **Pointwise aggregated attention.** Instead of softmax, each attention weight is passed through SiLU independently (with a constant normalizer such as the sequence length). Softmax forces weights to sum to 1, erasing how *many* relevant past actions there are; pointwise weights keep that intensity signal — the same reasoning as DIN's un-normalized attention (Chapter 4.2). The paper also argues pointwise attention is more robust in the non-stationary, streaming vocabulary of recommendation, where new items keep arriving.
- **Gating replaces the FFN.** $\text{Norm}(\mathbf{AV})\odot\mathbf{U}$ is a multiplicative gate, which lets the layer form feature interactions (the job DCN's crosses did) inside the attention block; there is no separate large feed-forward sublayer, making each layer cheaper.
- **Time-aware bias.** $\mathbf{rab}^{p,t}$ encodes "how long ago" as well as "how many steps ago".

**Efficiency tricks.** The paper pairs the architecture with sparsity-aware kernels for variable-length sequences, *stochastic length* (randomly subsampling long histories during training to cut cost), and an inference algorithm (M-FALCON) that scores many candidates in one forward pass: candidates are appended to the shared user history with an attention mask that lets each candidate attend to the history but not to other candidates, and the history computation is cached and reused across micro-batches — the recsys analogue of the KV cache from [Part 3](../part3/index.md).

**Reported results** (from the abstract): HSTU outperforms baselines by up to 65.8% in NDCG on synthetic and public datasets and is 5.3× to 15.2× faster than FlashAttention2-based Transformers on 8,192-length sequences. HSTU-based Generative Recommenders with 1.5 trillion parameters improved online A/B metrics by 12.4% and were deployed on multiple surfaces; and model quality **scales as a power law of training compute across three orders of magnitude, up to GPT-3/LLaMA-2 scale**. This last point is what launched the "large recommendation model" wave.

Follow-up work (e.g. Tencent's "Scaling New Frontiers") studied which HSTU components drive scaling and evaluated it on ranking tasks.

### Wukong: stacked factorization machines

Wukong (Meta) takes a different route: keep the classic feature-interaction view but make it scale. Each layer has a **factorization machine block** (all pairwise dot products among the current set of embeddings, projected back to a set of embeddings) and a **linear compression block**, with a residual connection. Because layer $l$ multiplies outputs of layer $l-1$, which already contain pairwise interactions, the interaction order grows geometrically with depth (roughly $2^l$). Making layers taller and wider scales the model smoothly. Wukong reports holding a scaling law across two orders of magnitude in model complexity, beyond 100 GFLOP per example, where earlier architectures fell short.

### RankMixer: GPU-friendly ranking backbone

RankMixer (ByteDance) replaces the zoo of CPU-era interaction modules with a homogeneous, Transformer-like backbone:

1. **Tokenization.** Group the hundreds of input features by meaning into $T$ feature tokens, each projected to dimension $D$: input `[batch, T, D]`.
2. **Multi-head token mixing** (instead of self-attention). Split each token into $H$ heads; build new token $h$ by concatenating head $h$ from every token. This is a parameter-free reshuffle that lets information cross between feature groups in $O(T)$ rather than attention's $O(T^2)$ — and with only tens of tokens, attention's flexibility is not needed.
3. **Per-token FFNs.** Unlike a Transformer, each token position has *its own* FFN parameters, because tokens are different feature subspaces (user, item, context…), not interchangeable words. A sparse-MoE variant scales this further.

Everything is large dense matmuls, so hardware is used efficiently. Reported results: **MFU rose from 4.5% to 45%**, ranking-model parameters scaled **100×** at roughly the same inference latency, and a 1B dense-parameter RankMixer serves full traffic in both recommendation and advertising scenarios without increasing serving cost, improving user active days by 0.3% and in-app usage duration by 1.08%.

### MTGR: keep the cross features

Pure GR models drop hand-crafted cross features (e.g. "user's historical CTR on this author"). Meituan found that doing so significantly degraded performance and that scaling could not compensate. **MTGR** builds on HSTU but keeps DLRM features, including cross features, as tokens; it compresses training by grouping all candidates of one user into one sample (**user-level compression**), adds **Group-Layer Normalization** (separate normalization per feature group, since tokens live in different semantic spaces) and a dynamic masking strategy to avoid information leakage. It supports 10–100× the computational complexity of the DLRM without significant cost increase and achieved **65× FLOPs per sample** for forward inference compared with the DLRM, deployed on Meituan's main traffic.

### LUM: a three-step paradigm

Alibaba's **Large User Model** argues end-to-end GR sacrifices too many practical DLRM advantages, and splits the work into three steps:

1. **Pre-train** a generative large user model on behaviour sequences (knowledge construction);
2. **Query** it — e.g. "what would this user do next in category X?" — to extract user-interest knowledge;
3. **Use** those outputs as features in the existing ranking model.

The large model runs off the critical path; the ranking model stays cheap. LUM reports performance improving as it scales to 7B parameters and gains in an online A/B test.

### MARM: trade compute for cache

Kuaishou's **MARM** starts from the observation that in recsys, parameters and data are cheap (they report over 50 billion user samples per day and models exceeding 200 billion parameters) while *FLOPs at serving time* are the expensive resource. MARM caches intermediate attention results computed for a user's history and reuses them in later requests, so a deeper sequence model can be run while paying only for the incremental part. They describe a "cache scaling law": quality improves with how much computation is cached. It is the KV-cache idea applied across requests.

### Comparing the recipes

| Approach | Where the compute goes | Keeps hand-crafted features? | Main trick |
|---|---|---|---|
| HSTU / GR (Meta) | Long sequence of interleaved item/action tokens | Mostly no | Pointwise gated attention; candidates share the history pass |
| Wukong (Meta) | Deep stack of FM interactions over feature embeddings | Yes | Interaction order grows with depth |
| RankMixer (ByteDance) | Per-token FFNs over tens of feature tokens | Yes | Parameter-free token mixing; high MFU; sparse-MoE variant |
| MTGR (Meituan) | HSTU-style sequence + DLRM features as tokens | Yes, incl. cross features | User-level compression; group layer norm |
| LUM (Alibaba) | Large generative user model offline | Yes (in the downstream DLRM) | Pre-train → query → feed features |
| MARM (Kuaishou) | Deeper sequence model with cached intermediates | — | Reuse cached attention results across requests |

## Code

A minimal HSTU-style layer and a RankMixer-style block, with shapes.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class HSTULayer(nn.Module):
    """Pointwise (SiLU) attention + gating, no softmax, no separate FFN. Causal."""
    def __init__(self, d, n_heads=2, max_len=512):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.f1 = nn.Linear(d, 4 * d)             # -> U, V, Q, K
        self.f2 = nn.Linear(d, d)
        self.norm_in, self.norm_av = nn.LayerNorm(d), nn.LayerNorm(d)
        self.rab = nn.Parameter(torch.zeros(n_heads, max_len, max_len))   # positional bias only here
    def forward(self, x):                         # x: [B, n, d]
        B, n, d = x.shape
        u, v, q, k = F.silu(self.f1(self.norm_in(x))).chunk(4, dim=-1)   # each [B, n, d]
        split = lambda t: t.view(B, n, self.h, self.dh).transpose(1, 2)  # [B, h, n, dh]
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-1, -2) + self.rab[:, :n, :n]           # [B, h, n, n]
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
        a = F.silu(scores).masked_fill(~causal, 0.0) / n                 # pointwise, not softmax
        av = (a @ v).transpose(1, 2).reshape(B, n, d)                    # [B, n, d]
        return x + self.f2(self.norm_av(av) * u)                         # gate with U, residual

class RankMixerBlock(nn.Module):
    """Parameter-free token mixing + per-token FFNs. Input [B, T, D] with D divisible by T."""
    def __init__(self, T, D, mult=4):
        super().__init__()
        self.T, self.D = T, D
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        # one FFN per token position: weights [T, D, mult*D] and [T, mult*D, D]
        self.w1 = nn.Parameter(torch.randn(T, D, mult * D) * D ** -0.5)
        self.w2 = nn.Parameter(torch.randn(T, mult * D, D) * (mult * D) ** -0.5)
    def mix(self, x):                             # H = T heads, each of size D/T
        B = x.size(0)
        heads = x.view(B, self.T, self.T, self.D // self.T)  # [B, token, head, D/T]
        return heads.transpose(1, 2).reshape(B, self.T, self.D)  # new token h = head h of all tokens
    def forward(self, x):                         # x: [B, T, D]
        x = self.norm1(x + self.mix(x))
        h = F.gelu(torch.einsum("btd,tdf->btf", x, self.w1))     # per-token weights
        return self.norm2(x + torch.einsum("btf,tfd->btd", h, self.w2))

x = torch.randn(8, 50, 64)                        # 8 users, 50 interleaved item/action tokens
print(HSTULayer(64)(x).shape)                     # torch.Size([8, 50, 64])
feats = torch.randn(8, 16, 256)                   # 16 feature tokens of width 256
print(RankMixerBlock(16, 256)(feats).shape)       # torch.Size([8, 16, 256])
```

This HSTU sketch omits the timestamp part of the relative bias and uses a dense $n\times n$ bias table; real implementations use bucketed relative position/time biases and fused variable-length kernels.

A tiny helper to redo the compute-budget arithmetic below for your own numbers:

```python
def serving_gpus(flops_user, flops_per_cand, n_cand, qps, peak=1e15, mfu=0.4):
    """GPUs needed at peak: (user part once + candidate part per candidate) per request."""
    flops_per_s = qps * (flops_user + n_cand * flops_per_cand)
    return flops_per_s / (peak * mfu)

print(serving_gpus(0, 1e9, 1000, 1e5))               # all compute per candidate  -> 250.0
print(serving_gpus(0, 1e9, 1000, 1e5, mfu=0.045))    # CPU-era MFU               -> ~2222
print(serving_gpus(0.9e9, 0.1e9, 1000, 1e5))         # 90% of FLOPs on user side -> ~25
```

## Worked example

**Compute budget for a ranking model.** Suppose:

- the ranker scores $N = 1{,}000$ candidates per request,
- peak traffic is $Q = 100{,}000$ requests per second,
- the dense model has $P = 500$M parameters used per candidate, so forward FLOPs per candidate $\approx 2P = 10^9$ (1 GFLOP; the "2 × parameters" rule from [Part 1](../part1/index.md)).

Total serving compute $= 2P \cdot N \cdot Q = 10^9 \cdot 10^3 \cdot 10^5 = 10^{17}$ FLOP/s.

A GPU with about $10^{15}$ FLOP/s of dense BF16 peak (roughly an H100-class accelerator) at 40% MFU delivers $4\times10^{14}$ FLOP/s, so we need $10^{17}/(4\times10^{14}) = 250$ GPUs — just for the ranking forward pass at peak. At the 4.5% MFU of a CPU-era design, the same model would need $\approx 2{,}200$ GPUs. **MFU is money.**

Now split the model into a **user part** computed once per request and a **candidate part** computed per candidate. If 90% of the FLOPs are in the user part (e.g. a long-sequence encoder) and only 10% (0.1 GFLOP) per candidate:

$$
\text{FLOP/s} = Q\cdot\big(0.9\times10^9 + N\cdot 0.1\times10^9\big) = 10^5\cdot(0.9\times10^9 + 10^{11}) \approx 1.01\times10^{16},
$$

about 10× less. Equivalently, for the same budget you can afford a ~10× larger user-side model. This is why HSTU's M-FALCON, MTGR's user-level compression, MARM's caching and OneRec's "encode once, decode a list" (Chapter [4.6](06-end-to-end-generative-recommenders.md)) all share user-side computation across candidates.

**Training compute.** With $10^{10}$ training examples per day (each a candidate impression), the same 1 GFLOP forward and the rule "training ≈ 3× forward", one day of training costs $3\times10^9\cdot10^{10} = 3\times10^{19}$ FLOPs, i.e. at $4\times10^{14}$ FLOP/s effective, $\approx 7.5\times10^4$ GPU-seconds $\approx 21$ GPU-hours per day of data — before counting embedding lookups and communication, which often dominate in practice.

## In practice (2025–2026)

- **Scaled ranking backbones are in production**: HSTU-based GR at Meta, RankMixer at ByteDance, MTGR at Meituan, LUM at Alibaba, MARM at Kuaishou. The common thread is high-MFU dense compute over user sequences, with user-side computation shared across candidates.
- **Two schools** coexist: pure sequence models that drop hand-crafted features (HSTU) vs hybrids that keep DLRM features and crosses as tokens (MTGR, RankMixer). Both report scaling.
- **Large models off the critical path** (LUM, SUM-style upstream user models) feed embeddings or knowledge into cheaper online rankers.
- Meta describes its **Generative Ads Recommendation Model (GEM)** as a large foundation model for ads whose knowledge is transferred to many downstream ads models (see the engineering blog in Further reading and Chapter [4.8](08-ads-systems.md)).
- The same scaling thinking extends to retrieval (HSNN) and to end-to-end generation (OneRec, Chapter 4.6).

## Common pitfalls / misconceptions

- **Counting parameters instead of FLOPs.** A "trillion-parameter" DLRM can be computationally tiny. Scaling claims should state FLOPs per example.
- **Ignoring MFU.** A model with more FLOPs but 5% utilization can be slower and more expensive than a bigger model at 45%.
- **Assuming you can drop all hand-crafted features.** Meituan reports large losses from dropping cross features; check on your data.
- **Fitting a scaling curve on too narrow a range** (less than one order of magnitude) and extrapolating.
- **Forgetting inference fan-out**: an architecture that is cheap per example can be unaffordable when multiplied by 1,000 candidates × 10^5 QPS unless user-side work is shared.

## Key takeaways

- DLRMs plateaued because parameters sat in embeddings, compute was tiny and poorly utilized, and interactions were shallow.
- Scaling requires compute-heavy, homogeneous, GPU-friendly dense blocks over rich sequences.
- HSTU reformulates recommendation as sequential transduction over (item, action) streams, uses pointwise gated attention, and showed power-law scaling with 12.4% online gains.
- Wukong, RankMixer, MTGR, LUM and MARM are complementary recipes: stacked FMs, token mixing with per-token FFNs, keep cross features, run the large model offline, cache computation.
- Budget serving as FLOPs/candidate × candidates × QPS ÷ (peak × MFU); share user-side work.

## Check your understanding

??? question "Why does a trillion-parameter DLRM not contradict the claim that DLRMs are 'small'?"
    Almost all of those parameters are embedding rows; each example touches only a few rows and the lookup costs negligible FLOPs. Compute per example — what quality scales with — remains small.

??? question "Why does HSTU use SiLU instead of softmax in attention?"
    Softmax normalizes weights to sum to one, discarding how strongly and how often the user engaged with related items. Pointwise SiLU keeps that intensity signal and, per the paper, is better suited to non-stationary streaming vocabularies.

??? question "In RankMixer, why does each token get its own FFN, unlike in a language Transformer?"
    RankMixer's tokens are different feature groups (user, item, context, …) living in different semantic spaces, not interchangeable positions in a sentence. Separate FFNs let each subspace be modelled with its own parameters, while token mixing handles cross-group interaction.

??? question "Using the worked example, how many GPUs are needed if MFU drops from 40% to 20%?"
    Effective throughput halves to $2\times10^{14}$ FLOP/s per GPU, so $10^{17}/(2\times10^{14}) = 500$ GPUs.

??? question "What is the main idea behind LUM's three-step paradigm?"
    Keep the large generative user model out of the latency-critical path: pre-train it, query it for user-interest knowledge, and feed the results into the existing ranking model as features, preserving DLRM practicalities while still benefiting from scale.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152) — Zhai et al. (Meta), ICML 2024 | The paper that started the large-recommendation-model wave: GR reformulation, HSTU, power-law scaling, +12.4% online. Code: [generative-recommenders](https://github.com/meta-recsys/generative-recommenders). |
| :material-star: [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) — Zhang et al. (Meta), 2024 | Stacked factorization machines that keep scaling across two orders of magnitude of compute. |
| :material-book-open-variant: [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351) — Zhang et al., 2023 | Empirical scaling laws for ID-based sequential recommenders up to 0.8B params, including data-constrained regimes. |
| :material-book-open-variant: [Scaling New Frontiers: Insights into Large Recommendation Models](https://arxiv.org/abs/2412.00714) — Guo et al. (Tencent), 2024 | Systematic study of HSTU scaling: which components matter, and transfer to ranking. |
| :material-new-box: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) — Zhu et al. (ByteDance), 2025 | Token mixing + per-token FFN / sparse-MoE; MFU 4.5% → 45%, 100× params at ~same latency. |
| :material-new-box: [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) — Han et al., 2025 | HSTU-style GR that keeps cross features; group-layer-norm; ~65× FLOPs per sample. |
| :material-new-box: [Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model (LUM)](https://arxiv.org/abs/2502.08309) — Yan et al. (Alibaba), 2025 | Pre-train → query → feed into DLRM; scaling to 7B. |
| :material-new-box: [MARM: Memory Augmentation and Scalable Complexity](https://arxiv.org/abs/2411.09425) — Lv et al. (Kuaishou), 2024 | Caches intermediate attention results — a "KV cache" idea across requests. |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | Practical large-ranker recipe at LinkedIn. |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | Scaling the retrieval stage beyond two-tower dot products. |
| :material-new-box: [Meta's Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) — Meta engineering blog, 2025 | Industry write-up of a large foundation model for ads ranking. |
