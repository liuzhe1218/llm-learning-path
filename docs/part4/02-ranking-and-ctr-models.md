# 4.2 Ranking and CTR Models

!!! abstract "What you'll learn"
    - How click-through-rate (CTR) prediction is set up as binary classification over sparse ID features and embedding tables
    - The lineage logistic regression → factorization machines → Wide & Deep → DCN/DCN-V2 → DLRM, including the $O(kn)$ FM trick and the DCN-V2 cross layer
    - Target attention (DIN): weighting a user's past behaviours by relevance to the candidate, with code
    - Multi-task ranking (shared-bottom, MMoE, PLE) and how several predictions are fused into one score
    - Why and how to calibrate predicted probabilities (Platt scaling, isotonic regression, downsampling correction), and what "embedding collapse" means

## Why this matters / the problem

The ranking stage (Chapter [4.1](01-recsys-fundamentals.md)) receives a few hundred to a few thousand candidates and must decide their order. It is where most modelling effort in industry has gone for the last decade, because it can afford rich features: everything about the user, everything about the item, the context, and — most importantly — *interactions* between them ("this user likes cooking videos *and* this video is a cooking video").

The core task is to predict, for each candidate, the probability of one or more actions: click, like, long watch, purchase, conversion. These probabilities must be good enough to (a) order candidates correctly and, in ads, (b) be multiplied by money (Chapter [4.8](08-ads-systems.md)), which requires them to be **calibrated**. This chapter builds the classic architectures from first principles; Chapter [4.4](04-scaling-recommendation-models.md) explains why they stopped scaling and what replaced them.

## Intuition

A ranking model's input is mostly **categorical IDs**: user ID, item ID, author ID, category, country, device, hour-of-day, and lists of IDs the user interacted with. Raw IDs have no geometry, so each ID is mapped to a learned vector (an **embedding**). The model then needs to learn which *combinations* of features predict a click. Historically, engineers hand-crafted these combinations ("user_country × item_language"). The architectures in this chapter are successive attempts to learn combinations automatically and cheaply:

- **FM:** every pair of features interacts through the dot product of their embeddings.
- **Wide & Deep:** memorize specific crossed features in a linear part, generalize with an MLP.
- **DCN:** learn explicit polynomial crosses of increasing degree, layer by layer.
- **DLRM:** dot products between all embedding pairs, then an MLP — a clean reference design.
- **DIN:** the user's history is a *set of items*; let the candidate decide which past items matter.

## How it works

### CTR prediction as binary classification

Each training example is an impression with features $\mathbf{x}$ and label $y \in \{0,1\}$ (clicked or not). The model outputs a logit $z = f(\mathbf{x})$ and probability $\hat p = \sigma(z) = 1/(1+e^{-z})$. Training minimizes the binary cross-entropy (LogLoss):

$$
\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n}\left[y_t \log \hat p_t + (1-y_t)\log(1-\hat p_t)\right].
$$

In words: push $\hat p$ up on clicked impressions and down on non-clicked ones. Minimizing LogLoss yields, at the optimum, $\hat p = P(y=1\mid\mathbf{x})$ — the model is calibrated if it is expressive enough and trained on unbiased data.

### Sparse features and embedding tables

A categorical feature with vocabulary size $V$ is one-hot encoded as $\mathbf{x} \in \{0,1\}^V$ and multiplied by an embedding matrix $\mathbf{E} \in \mathbb{R}^{V\times d}$, which is equivalent to a row lookup: `E[id]`. Multi-valued features (e.g. the last 50 clicked item IDs) are looked up and **pooled** (sum, mean or attention).

Vocabularies are enormous and open-ended (new users and items every second), so systems use the **hashing trick**: row $= \text{hash}(\text{id}) \bmod M$ for a fixed table size $M$. Collisions (two IDs sharing a row) are the price for bounded memory. Embedding tables dominate the parameter count: industrial models have embedding tables of terabytes while the dense network has far fewer parameters. This is why DLRM-style systems are **memory-bound**, use model parallelism for tables, and — as Chapter 4.4 explains — do not "scale" the way LLMs do.

### Logistic regression

$$
z = w_0 + \sum_{i=1}^{n} w_i x_i ,
$$

where $x_i$ are (mostly binary) features and $w_i$ learned weights. LR cannot model interactions unless you add crossed features by hand. It was the industry workhorse for a decade — Google's FTRL-Proximal online LR and Facebook's "GBDT leaves as features + LR" are the classic references — because it is fast, trains online, and is easy to calibrate.

### Factorization machines (FM)

FM adds all pairwise interactions, but each feature $i$ has a latent vector $\mathbf{v}_i \in \mathbb{R}^k$ and the weight of pair $(i,j)$ is $\langle \mathbf{v}_i, \mathbf{v}_j\rangle$:

$$
z = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j\rangle\, x_i x_j .
$$

In words: two features interact strongly if their latent vectors point in the same direction. Because the pair weight is factorized, the model can estimate interactions for pairs never seen together in training (if user A's vector is close to user B's, A inherits B's interactions).

**The $O(kn)$ trick.** Naively the pairwise term costs $O(kn^2)$. Write $s_f = \sum_i v_{i,f}x_i$ for each latent dimension $f$. Then

$$
\sum_{i<j}\langle \mathbf{v}_i,\mathbf{v}_j\rangle x_i x_j
= \frac{1}{2}\sum_{f=1}^{k}\left[\Big(\sum_{i=1}^n v_{i,f}x_i\Big)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2\right].
$$

??? info "Derivation"
    $\big(\sum_i a_i\big)^2 = \sum_i a_i^2 + 2\sum_{i<j} a_i a_j$. Apply this per dimension $f$ with $a_i = v_{i,f}x_i$: $\sum_{i<j} v_{i,f}v_{j,f}x_ix_j = \frac{1}{2}\big[(\sum_i v_{i,f}x_i)^2 - \sum_i v_{i,f}^2x_i^2\big]$. Summing over $f$ gives $\sum_{i<j}\langle \mathbf{v}_i,\mathbf{v}_j\rangle x_ix_j$ on the left.

In words: "square of the sum minus sum of the squares, halved" — computable in one pass over the $n$ features, $O(kn)$. With embeddings already looked up as a tensor `e: [batch, n_fields, k]`, the FM term is `0.5 * (e.sum(1)**2 - (e**2).sum(1)).sum(1)`.

### Wide & Deep

Wide & Deep (Google Play, 2016) jointly trains two parts and adds their logits:

$$
z = \underbrace{\mathbf{w}_{\text{wide}}^\top[\mathbf{x}, \phi(\mathbf{x})]}_{\text{memorization}} + \underbrace{\mathbf{w}_{\text{deep}}^\top \mathbf{a}^{(L)}}_{\text{generalization}} + b,
$$

where $\phi(\mathbf{x})$ are hand-chosen cross-product features (e.g. "installed app = X AND impression app = Y"), and $\mathbf{a}^{(L)}$ is the last hidden layer of an MLP over concatenated embeddings. The wide part memorizes specific frequent co-occurrences; the deep part generalizes to unseen combinations through embeddings. This "memorization vs generalization" vocabulary is still used today (e.g. for semantic IDs in Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)).

### Deep & Cross Network (DCN, DCN-V2)

MLPs learn multiplicative interactions only inefficiently. DCN adds a **cross network** that explicitly builds polynomial interactions. Let $\mathbf{x}_0 \in \mathbb{R}^D$ be the concatenation of all embeddings and dense features. The DCN-V2 cross layer is

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l ,
$$

with $\mathbf{W}_l \in \mathbb{R}^{D\times D}$, $\mathbf{b}_l \in\mathbb{R}^D$ and $\odot$ the element-wise product. In words: each layer multiplies the original input with a linear transform of the current representation, so layer $l$ contains feature interactions up to degree $l+1$, and the residual $+\mathbf{x}_l$ keeps lower-degree terms. (The original DCN used a rank-1 version, $\mathbf{x}_0\mathbf{x}_l^\top\mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$; V2's full matrix — often low-rank factorized as $\mathbf{U}\mathbf{V}^\top$ — is more expressive.) The cross network runs in parallel or in series with an MLP and is a strong production baseline.

### DLRM

Meta's DLRM is a reference architecture: dense features go through a bottom MLP to a $d$-dim vector; each sparse feature is an embedding lookup to $d$ dims; then **all pairwise dot products** between these $m$ vectors are computed (the FM idea, explicit), concatenated with the dense vector, and passed to a top MLP that outputs the logit. Its systems contribution matters as much as the model: embedding tables are sharded across devices (**model parallelism**) while MLPs are replicated (**data parallelism**), with an all-to-all exchange of looked-up embeddings each step.

### Target attention: DIN

A user's behaviour history $[e_1, \dots, e_T]$ (embeddings of the last $T$ items they interacted with) is the strongest signal in ranking. Sum-pooling it into one vector blurs everything: a user who watches cooking *and* football gets the same summary whether the candidate is a recipe or a match. **Deep Interest Network (DIN)** makes the pooling depend on the candidate (the "target") $e_a$:

$$
\mathbf{v}_u(a) = \sum_{j=1}^{T} \alpha(e_j, e_a)\, e_j, \qquad \alpha(e_j,e_a) = \text{MLP}\big([e_j;\ e_a;\ e_j - e_a;\ e_j \odot e_a]\big).
$$

In words: each past behaviour is weighted by a small network's estimate of its relevance to the candidate, so the user representation is re-computed per candidate. DIN deliberately does **not** softmax-normalize the weights, so a user with many relevant behaviours gets a larger vector — the *intensity* of interest is preserved. (Remember this; HSTU makes the same choice in Chapter 4.4.) This is the starting point of sequence modelling in ranking, developed in Chapter [4.3](03-sequential-user-modeling.md).

### Multi-task learning

Feeds optimize several actions at once — click, like, share, follow, watch time, skip. Training one model per task wastes compute and data; sharing helps sparse tasks (e.g. shares) borrow from dense ones (clicks).

- **Shared-bottom:** one shared network produces a representation $\mathbf{h}$, and task-specific towers $h^k(\mathbf{h})$ predict each task $k$. Simple, but tasks that conflict pull the shared layers in different directions (**negative transfer**).
- **MMoE (multi-gate mixture of experts):** $E$ expert networks $f_1..f_E$; each task $k$ has its own softmax gate $\mathbf{g}^k(\mathbf{x}) \in \mathbb{R}^E$ and sees the mixture $\mathbf{h}^k = \sum_{i=1}^E g^k_i(\mathbf{x}) f_i(\mathbf{x})$, then its tower. Tasks can share experts when they agree and use different experts when they conflict.
- **PLE (progressive layered extraction):** splits experts explicitly into shared experts and task-specific experts, stacked in several layers, to further reduce negative transfer. Treat it as "MMoE with reserved per-task capacity".

The loss is a weighted sum $\sum_k \lambda_k \mathcal{L}_k$, with BCE for binary tasks and e.g. regression or weighted-logistic losses for watch time.

### Combining objectives into one score

The ranking stage must output a single ordering. A common form is a weighted product or sum of predictions, e.g.

$$
\text{score} = \hat p_{\text{click}}^{\,\alpha}\cdot (1 + \hat p_{\text{like}})^{\beta}\cdot \widehat{\text{watch}}^{\,\gamma},
$$

where the exponents $\alpha, \beta, \gamma$ are tuned by A/B tests or by black-box search against online metrics. In ads, the score must be in money units (expected value per impression; Chapter 4.8), which is the main reason ads predictions must be calibrated. These hand-tuned fusion formulas are one of the things generative recommenders with learned reward models try to replace (Chapter [4.6](06-end-to-end-generative-recommenders.md)).

### Calibration

A model is **calibrated** if, among impressions where it predicts $\hat p \approx 0.03$, about 3% are clicked. Models drift out of calibration because of negative downsampling, distribution shift, position effects, or overfitting. A simple diagnostic is the **calibration ratio** $\sum_t \hat p_t / \sum_t y_t$ (should be ≈ 1 overall and within each bucket).

Fixes, applied on a held-out recent slice:

- **Downsampling correction.** If negatives were kept with rate $w$ (e.g. 10%), the trained model predicts $p'$ on the re-balanced data. The calibrated probability is $p = \dfrac{p'}{p' + (1-p')/w}$.
- **Platt scaling.** Fit $p = \sigma(a\cdot \text{logit}(\hat p) + b)$ with two scalars $a,b$ by minimizing LogLoss.
- **Isotonic regression.** Fit a non-decreasing step function from $\hat p$ to the empirical rate (the pool-adjacent-violators algorithm). More flexible, preserves ordering, needs more data.

Production systems often calibrate per segment (e.g. per ad type or surface) and use a small calibration layer trained online.

### Embedding collapse

Making embeddings bigger is the cheapest way to add parameters, yet it often stops helping. One explanation is **embedding collapse**: the learned embedding matrix occupies a low-dimensional subspace (most singular values near zero), so extra dimensions are wasted. Guo et al. argue feature-interaction layers have a two-sided effect — interacting with already-collapsed embeddings propagates collapse, but interaction is also needed to avoid fitting spurious features — and propose **multi-embedding**: several independent embedding sets, each with its own interaction module, which keeps embedding diversity high. You can diagnose collapse by plotting the singular values of an embedding table.

## Code

FM, a DCN-V2 cross layer and a DIN attention unit, with the FM trick checked numerically.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

def fm_pairwise(e):
    """e: [batch, n_fields, k] embeddings (already multiplied by feature values).
    Returns sum_{i<j} <e_i, e_j> per example: [batch]. O(n*k)."""
    square_of_sum = e.sum(dim=1) ** 2            # [batch, k]
    sum_of_square = (e ** 2).sum(dim=1)          # [batch, k]
    return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

e = torch.randn(4, 10, 8)
naive = torch.stack([sum((e[b, i] * e[b, j]).sum() for i in range(10) for j in range(i + 1, 10))
                     for b in range(4)])
assert torch.allclose(fm_pairwise(e), naive, atol=1e-4)

class CrossLayerV2(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l   (DCN-V2), low-rank W = U V^T."""
    def __init__(self, dim, rank=16):
        super().__init__()
        self.V = nn.Linear(dim, rank, bias=False)
        self.U = nn.Linear(rank, dim)
    def forward(self, x0, xl):                   # both [batch, dim]
        return x0 * self.U(self.V(xl)) + xl

class DINAttention(nn.Module):
    """Target attention: weight each history item by relevance to the candidate (no softmax)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, hist, target, mask):
        # hist: [B, T, d], target: [B, d], mask: [B, T] (1 = real item, 0 = padding)
        t = target[:, None, :].expand_as(hist)                       # [B, T, d]
        feats = torch.cat([hist, t, hist - t, hist * t], dim=-1)     # [B, T, 4d]
        w = self.mlp(feats).squeeze(-1) * mask                       # [B, T]
        return (w[:, :, None] * hist).sum(dim=1)                     # [B, d]

class TinyRanker(nn.Module):
    def __init__(self, n_items, n_users, d=16):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        self.user = nn.Embedding(n_users, d)
        self.din = DINAttention(d)
        self.cross = nn.ModuleList([CrossLayerV2(3 * d) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(3 * d, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, user, hist, cand):        # [B], [B, T], [B]
        h, c = self.item(hist), self.item(cand)
        interest = self.din(h, c, (hist != 0).float())
        x0 = torch.cat([self.user(user), c, interest], dim=-1)       # [B, 3d]
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(x).squeeze(-1)          # logit; train with BCEWithLogitsLoss

model = TinyRanker(n_items=1000, n_users=100)
logit = model(torch.randint(0, 100, (8,)), torch.randint(0, 1000, (8, 20)), torch.randint(1, 1000, (8,)))
print(logit.shape)                               # torch.Size([8])
```

Calibration with Platt scaling and isotonic regression (pool-adjacent-violators), on a deliberately over-confident model:

```python
import torch
torch.manual_seed(0)
n = 20000
true_p = torch.rand(n) * 0.1                      # true CTRs between 0 and 10%
y = torch.bernoulli(true_p)
pred = torch.clamp(true_p * 2.0, max=0.999)       # model over-predicts by 2x

def logit(p): return torch.log(p / (1 - p))

# Platt scaling: p_cal = sigmoid(a * logit(pred) + b)
a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([a, b], max_iter=100)
def closure():
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(a * logit(pred) + b, y)
    loss.backward(); return loss
opt.step(closure)
platt = torch.sigmoid(a * logit(pred) + b).detach()

# Isotonic regression via pool-adjacent-violators on sorted predictions
order = pred.argsort()
blocks = []                                       # each block: [sum_y, count]
for yi in y[order].tolist():
    blocks.append([yi, 1])
    while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
        s, c = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += c
iso_sorted = torch.cat([torch.full((c,), s / c) for s, c in blocks])
iso = torch.empty(n); iso[order] = iso_sorted

for name, p in [("raw", pred), ("platt", platt), ("isotonic", iso)]:
    print(f"{name:9s} calibration ratio = {p.sum() / y.sum():.3f}")
```

## Worked example

**FM trick by hand.** Three active features ($x_i = 1$) with $k=2$ embeddings $\mathbf{v}_1=(1,2)$, $\mathbf{v}_2=(0,1)$, $\mathbf{v}_3=(2,-1)$.

- Naive: $\langle v_1,v_2\rangle = 2$, $\langle v_1,v_3\rangle = 2-2 = 0$, $\langle v_2,v_3\rangle = -1$. Sum $= 1$.
- Trick: sum vector $= (3, 2)$, squared per dimension $= (9, 4)$; sum of squares per dimension $= (1+0+4,\ 4+1+1) = (5, 6)$. Difference $= (4, -2)$, total $2$, halved $= 1$. ✓

**Embedding table size.** A table with $10^9$ hashed IDs × 64 dims × 4 bytes (fp32) $= 2.56\times10^{11}$ bytes $\approx 256$ GB — for *one* feature. A model with dozens of large ID features reaches terabytes, while a dense MLP with $10^7$ parameters is only 40 MB. This asymmetry is the root of the scaling problem in Chapter 4.4.

**Downsampling correction.** Negatives were kept at rate $w = 0.1$. The model outputs $p' = 0.2$ for an impression. Corrected: $p = 0.2 / (0.2 + 0.8/0.1) = 0.2/8.2 \approx 0.0244$. Without correction the ad would be valued at 8× its true click probability.

**Calibration bucket check.** On 10,000 impressions with predictions in $[0.04, 0.05)$ (mean 0.045), 220 clicks were observed: empirical rate 0.022, calibration ratio $0.045/0.022 \approx 2.0$. Isotonic regression would map this bucket to ≈ 0.022; Platt scaling would fit a global $a,b$ that roughly halves predictions in this range.

## In practice (2025–2026)

- **DCN-V2-style crosses + target attention + MMoE/PLE multi-task heads** remain the backbone of many production rankers; LinkedIn's LiRank, for example, describes Residual DCN (DCN-V2 with attention and residuals), Transformers over history, dense gating, calibration and quantization in production.
- Large tech companies increasingly replace the zoo of CPU-era interaction modules with GPU-friendly, scalable backbones (RankMixer, Wukong, HSTU; Chapter 4.4).
- **Calibration** is a dedicated production component in ads, often per-segment and online.
- **Score fusion** of many heads is still largely hand-tuned; learned reward models and RL (Chapter 4.6) are the frontier.

## Common pitfalls / misconceptions

- **Optimizing AUC and forgetting calibration.** AUC is invariant to any monotone transform; a 2× over-prediction leaves AUC unchanged but doubles ad prices.
- **Applying softmax in DIN-style attention by habit**: it removes interest intensity.
- **Assuming bigger embeddings = better model.** Check singular-value spectra; collapsed embeddings waste memory.
- **Calibrating on stale data**: calibration must track the current distribution, so it is refreshed frequently.
- **Training MTL losses with arbitrary weights** without checking that sparse tasks are not drowned out.

## Key takeaways

- CTR prediction is binary classification over sparse IDs; embedding tables hold most parameters.
- FM, Wide & Deep, DCN and DLRM are increasingly automatic ways of learning feature interactions; FM's pairwise term costs $O(kn)$ via "square of sum minus sum of squares".
- DIN's target attention makes the user representation depend on the candidate — the seed of sequence modelling in ranking.
- Multi-task models (shared-bottom → MMoE → PLE) predict many actions, fused into one score.
- Calibration (downsampling correction, Platt, isotonic) is essential whenever predictions are used as probabilities or money.

## Check your understanding

??? question "Why does FM generalize to feature pairs never seen together in training, while a crossed-feature LR does not?"
    LR learns an independent weight per crossed feature, so an unseen pair has no weight. FM's pair weight is $\langle \mathbf{v}_i,\mathbf{v}_j\rangle$, and each $\mathbf{v}_i$ is learned from all pairs involving feature $i$, so any two features with trained vectors have a defined interaction.

??? question "What degree of feature interaction can a 3-layer DCN-V2 cross network represent?"
    Each cross layer multiplies by $\mathbf{x}_0$ once, raising the maximum polynomial degree by one. Starting from degree 1, after 3 layers the output contains interactions up to degree 4.

??? question "A model trained with negatives downsampled at 5% predicts 0.5. What is the calibrated probability?"
    $p = 0.5/(0.5 + 0.5/0.05) = 0.5/10.5 \approx 0.048$.

??? question "Why does MMoE use a separate gate per task?"
    So each task can choose its own mixture of experts. Related tasks end up sharing experts; conflicting tasks can use different ones, reducing the negative transfer of a fully shared bottom.

??? question "Why does DIN not normalize its attention weights with softmax?"
    Softmax forces weights to sum to one, which hides how many relevant behaviours the user has. Unnormalized weights let the pooled vector grow with the strength of interest, which is predictive of clicks.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | FTRL-Proximal online logistic regression and the engineering lessons of large-scale ads CTR — calibration, memory savings, monitoring. |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + logistic regression; data freshness matters most; normalized entropy as the metric. |
| :material-star: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — Cheng et al., 2016 | Linear "memorization" + DNN "generalization" jointly trained; Google Play. |
| :material-book-open-variant: [Deep & Cross Network (DCN)](https://arxiv.org/abs/1708.05123) / [DCN V2](https://arxiv.org/abs/2008.13535) — Wang et al., 2017/2020 | Explicit, efficient feature-crossing layers; V2 is a strong production baseline. |
| :material-star: [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978) — Zhou et al. (Alibaba), 2017 | Target attention: weight the user's past behaviours by relevance to the candidate ad. |
| :material-star: [DLRM: Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) — Naumov et al. (Meta), 2019 | Reference architecture and the systems view: embedding tables dominate memory, requiring model parallelism. |
| :material-book-open-variant: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) — Ma et al. (Google), KDD 2018 | The MMoE paper. |
| :material-book-open-variant: [Recommending What Video to Watch Next: A Multitask Ranking System](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/) — Zhao et al. (Google), RecSys 2019 | MMoE plus a shallow position-bias tower in YouTube's production ranker. |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | Practical large-ranker recipe: Residual DCN, Transformer over history, dense gating, calibration, quantization, explore/exploit. |
| :material-book-open-variant: [On the Embedding Collapse When Scaling up Recommendation Models](https://arxiv.org/abs/2310.04400) — Guo et al., 2023 | Why simply enlarging embeddings stops helping; proposes multi-embedding. |
