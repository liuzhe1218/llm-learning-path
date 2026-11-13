# 4.5 Semantic IDs and Generative Retrieval

!!! abstract "What you'll learn"
    - Why random item IDs are a problem at billion scale (memory, churn, cold start) and what a semantic ID is
    - Vector quantization and residual quantization from scratch, with a numeric example of residual coding
    - How an RQ-VAE is trained (losses, straight-through estimator), with a runnable minimal implementation, and how to handle ID collisions
    - TIGER: a seq2seq model that decodes the next item's semantic ID, plus beam search with trie-constrained decoding (code)
    - How semantic IDs are used inside ranking models (YouTube), and the trade-offs between generative and dense retrieval (LIGER, COBRA, GRID)

## Why this matters / the problem

Every model so far represented an item by a **random ID** with its own embedding row. That has three costs at scale:

1. **Memory.** One row per item: billions of rows, often the largest part of the model (Chapter [4.2](02-ranking-and-ctr-models.md)).
2. **Churn and cold start.** A video uploaded five minutes ago has a fresh, untrained embedding. Nothing about its content helps the model until it collects interactions. Hash collisions mix unrelated items.
3. **No shared structure.** Two near-identical items (re-uploads, the same product from two sellers) learn independently. There is no way to "spell" a new item from known parts.

LLMs avoid this by using a small vocabulary of subword tokens that combine into any word. **Semantic IDs** do the same for items: each item becomes a short tuple of discrete codes, e.g. `(17, 203, 5)`, derived from its *content* embedding, such that similar items share prefixes. With items spelled as token sequences, a Transformer can **generate** the next item token by token, exactly as an LLM generates a word — **generative retrieval**.

## Intuition

Think of postal codes. A code like `94107` is not random: the first digit names a large region, the next digits narrow it to a city and then a neighbourhood. Two nearby addresses share a long prefix. A new house gets a meaningful code immediately, because its location is known.

A semantic ID is a postal code for items in embedding space. Level 1 says roughly *what kind* of item it is (cooking videos), level 2 refines it (Italian cooking), level 3 refines further (pasta tutorials), and a final code distinguishes near-duplicates. A model that predicts "the next item will start with code 17" has already narrowed billions of items to one region, and it can do so for an item it has never seen in training, as long as the item's content places it there.

## How it works

### Vector quantization (VQ)

A **codebook** is a set of $K$ vectors $\{\mathbf{e}_1,\dots,\mathbf{e}_K\}\subset\mathbb{R}^d$. Quantizing a vector $\mathbf{z}\in\mathbb{R}^d$ means replacing it with its nearest codeword:

$$
c = \arg\min_{k} \lVert \mathbf{z} - \mathbf{e}_k\rVert^2, \qquad \hat{\mathbf{z}} = \mathbf{e}_c .
$$

In words: snap the vector to the closest of $K$ prototypes and remember only the index $c$. With $K$ codewords you can distinguish only $K$ groups, and the reconstruction error $\lVert\mathbf{z}-\hat{\mathbf{z}}\rVert$ is large unless $K$ is huge. k-means is exactly the algorithm that learns such a codebook.

### Residual quantization (RQ)

To get fine resolution with small codebooks, quantize the **error** repeatedly. Use $D$ codebooks $\mathcal{C}^{(1)},\dots,\mathcal{C}^{(D)}$, each with $K$ codewords:

$$
\mathbf{r}_0 = \mathbf{z},\qquad
c_l = \arg\min_k \lVert \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_k\rVert^2,\qquad
\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_{c_l},\qquad l = 1..D,
$$

$$
\hat{\mathbf{z}} = \sum_{l=1}^{D}\mathbf{e}^{(l)}_{c_l}, \qquad \text{semantic ID} = (c_1, c_2, \dots, c_D).
$$

In words: level 1 picks the closest coarse prototype; level 2 quantizes what level 1 got wrong; and so on. The tuple of indices is coarse-to-fine, and $D$ codebooks of size $K$ address $K^D$ cells (e.g. $256^3 \approx 1.7\times10^7$) while storing only $D\cdot K$ vectors.

### RQ-VAE

Plain RQ on raw content embeddings works, but an **RQ-VAE** learns the space being quantized. It has:

- an encoder $E$ that maps an item's content embedding $\mathbf{x}$ (e.g. from a sentence encoder over title + description, or a multimodal model) to a latent $\mathbf{z} = E(\mathbf{x})$;
- the residual quantizer above, with learnable codebooks;
- a decoder $G$ that reconstructs $\hat{\mathbf{x}} = G(\hat{\mathbf{z}})$.

The training loss is

$$
\mathcal{L} = \underbrace{\lVert \mathbf{x}-\hat{\mathbf{x}}\rVert^2}_{\text{reconstruction}}
+ \sum_{l=1}^{D}\Big(\underbrace{\lVert \text{sg}[\mathbf{r}_{l-1}] - \mathbf{e}^{(l)}_{c_l}\rVert^2}_{\text{move codeword to data}}
+ \beta\,\underbrace{\lVert \mathbf{r}_{l-1} - \text{sg}[\mathbf{e}^{(l)}_{c_l}]\rVert^2}_{\text{commit encoder to codeword}}\Big),
$$

where $\text{sg}[\cdot]$ is stop-gradient and $\beta$ (often 0.25) weights the commitment term. In words: reconstruct the input well, pull each chosen codeword toward the residual it represents, and keep the encoder from drifting away from the codewords.

The $\arg\min$ is not differentiable, so the decoder's gradient is passed to the encoder with the **straight-through estimator**: use $\mathbf{z} + \text{sg}[\hat{\mathbf{z}}-\mathbf{z}]$ as the decoder input, which equals $\hat{\mathbf{z}}$ in the forward pass but has gradient 1 with respect to $\mathbf{z}$.

Two practical issues:

- **Codebook collapse**: only a few codewords get used. Common fixes are initializing codebooks with k-means on the first batch and re-initializing unused ("dead") codes.
- **Some systems skip the VAE** and use residual k-means directly on content embeddings (sometimes with balancing so each code gets a similar number of items).

### Collisions

Two items can receive the same tuple $(c_1,\dots,c_D)$, especially near-duplicates. Since a generative model must map each ID back to items, collisions need handling:

- **Append a disambiguation token**: TIGER adds an extra final code that simply enumerates colliding items ($0, 1, 2, \dots$), making IDs unique.
- **Increase depth or codebook size**, or quantize with a collision-aware or balanced assignment.
- **Accept them** and treat a semantic ID as a small bucket of items, re-ranked by a downstream model.

### TIGER: generative retrieval with semantic IDs

**TIGER** (Google, 2023) is the reference design:

1. Build semantic IDs: content embeddings from a sentence encoder over item text → RQ-VAE → a few codes per item (plus the disambiguation code).
2. Represent a user's history as the concatenation of its items' semantic IDs: `[c1(i1) c2(i1) c3(i1) c1(i2) c2(i2) c3(i2) …]`. The vocabulary is just $D\cdot K$ code tokens (each level has its own token range) plus user tokens.
3. Train a Transformer encoder–decoder: the encoder reads the history, the decoder generates the next item's semantic ID one code at a time with cross-entropy, exactly like seq2seq translation.
4. At inference, run **beam search** to produce several high-probability IDs, map them back to items, and return them as retrieval candidates.

TIGER reports significantly outperforming then-state-of-the-art sequential recommenders on several datasets and better retrieval for items with no prior interaction history — because a new item's semantic ID comes from its content, so the model can generate it without ever having seen it in training. The idea traces back to **DSI**, which trained a seq2seq model to decode document IDs directly for search.

### Beam search and constrained decoding

Generating $D$ codes autoregressively, the model's probability of an ID is $P(c_1,\dots,c_D\mid\text{history}) = \prod_{l=1}^{D} P(c_l\mid c_{<l},\text{history})$. **Beam search** keeps the $B$ best partial sequences at each step, expands each by all possible next codes, and keeps the $B$ best by cumulative log-probability (see [Part 3](../part3/index.md) for decoding strategies in LLMs).

Not every tuple corresponds to a real item: most of the $K^D$ cells are empty. Unconstrained decoding wastes beams on **invalid IDs**. The fix is a **prefix trie** of all valid IDs: at step $l$, only codes that extend the current prefix to a prefix of some existing item are allowed (others get log-probability $-\infty$). New items are added by inserting their IDs into the trie; retired items are removed — no model retraining needed for the index to change. The same trie can also enforce business filters (region, age rating, in-stock) by building it from the eligible subset.

### Semantic IDs inside ranking models

Semantic IDs are also useful *without* generation. YouTube replaced random video-ID features in a production **ranking** model with embeddings of semantic-ID pieces. Using the content embedding directly lost memorization (popular items need their own parameters); using the full semantic ID as one feature lost generalization. The winning approach hashes **sub-pieces** of the ID — n-grams of codes, or better, pieces learned with SentencePiece (as in LLM tokenizers) — into embedding tables. Coarse pieces are shared across many similar items (generalization); long pieces are nearly unique (memorization). The paper reports improved generalization on new and long-tail items without sacrificing overall quality.

### Generative vs dense retrieval

| | Dense retrieval (two-tower + ANN) | Generative retrieval (semantic IDs + decoding) |
|---|---|---|
| Item storage | One vector per item + ANN index | Codebooks + trie of valid IDs; no per-item vector needed |
| Cold start | Needs content features in the item tower | Natural: new item's ID comes from content |
| Model–index consistency | Index rebuilt separately; can be stale | The model *is* the index (plus a trie) |
| Fine-grained memorization | Strong (per-item vectors) | Weaker; collisions blur near-duplicates |
| Inference cost | One user-tower pass + ANN lookup | $D$ decoder steps × beam width |
| Diversity control | Via the ANN query / multiple queries | Beam search naturally yields varied prefixes |

Meta's **LIGER** compared both under controlled conditions and found generative retrieval can lag dense retrieval, with cold-start among the differences; it proposes a hybrid that integrates dense retrieval into generative retrieval. Baidu's **COBRA** goes coarse-to-fine: first generate a sparse semantic ID, then generate a dense vector *conditioned* on it, trained end to end, and combines beam search with nearest-neighbour scores at inference (BeamFusion); it reports online gains on an advertising platform with over 200 million daily users.

## Code

A minimal RQ-VAE on synthetic "content embeddings" drawn from a hierarchy of clusters, so we can check that items from the same cluster share ID prefixes.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# Synthetic content embeddings: 8 topics x 8 subtopics, 4000 items, dim 32
X_DIM, N = 32, 4000
topic_c = torch.randn(8, X_DIM) * 3
sub_c = torch.randn(8, 8, X_DIM)
topic = torch.randint(0, 8, (N,)); sub = torch.randint(0, 8, (N,))
x = topic_c[topic] + sub_c[topic, sub] + 0.2 * torch.randn(N, X_DIM)

class RQVAE(nn.Module):
    def __init__(self, x_dim=32, z_dim=16, levels=3, K=16, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(), nn.Linear(64, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, x_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, z_dim) * 0.1) for _ in range(levels)])
        self.beta = beta

    def quantize(self, z):                        # z: [B, z_dim]
        residual, z_hat, codes, vq_loss = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:                 # cb: [K, z_dim]
            idx = torch.cdist(residual, cb).argmin(dim=1)          # [B] nearest codeword
            e = cb[idx]                                            # [B, z_dim]
            vq_loss = vq_loss + F.mse_loss(e, residual.detach()) \
                              + self.beta * F.mse_loss(residual, e.detach())
            z_hat = z_hat + e
            residual = residual - e.detach()      # next level quantizes what is left
            codes.append(idx)
        return z_hat, torch.stack(codes, dim=1), vq_loss          # codes: [B, levels]

    def forward(self, x):
        z = self.enc(x)
        z_hat, codes, vq_loss = self.quantize(z)
        z_st = z + (z_hat - z).detach()           # straight-through estimator
        return F.mse_loss(self.dec(z_st), x) + vq_loss, codes

model = RQVAE()
with torch.no_grad():                             # data-driven init, level by level
    r = model.enc(x[torch.randperm(N)[:1024]])
    for cb in model.codebooks:                    # each codebook starts at sampled residuals
        cb.copy_(r[torch.randperm(len(r))[:cb.shape[0]]])
        r = r - cb[torch.cdist(r, cb).argmin(1)]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(3000):
    batch = x[torch.randint(0, N, (256,))]
    loss, _ = model(batch)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    _, codes = model(x)
ids = [tuple(c) for c in codes.tolist()]
from collections import Counter
print("loss", round(loss.item(), 3), "| unique IDs", len(set(ids)), "of", N,
      "| largest collision bucket", Counter(ids).most_common(1)[0][1])
# Do items with the same topic share the first code more than random pairs do?
i, j = torch.randint(0, N, (2, 20000))
same_topic = topic[i] == topic[j]
share1 = codes[i, 0] == codes[j, 0]
print(f"P(same first code | same topic) = {share1[same_topic].float().mean():.2f}, "
      f"| different topic = {share1[~same_topic].float().mean():.2f}")
```

Items from the same topic share their first code almost always, and items from different topics almost never: the prefix carries meaning. You will also see many **collisions** (a few hundred unique IDs for 4,000 items): in this synthetic data the items inside one of the 64 subtopics differ only by small noise, so they are near-duplicates and land in the same cell. That is exactly the situation where TIGER appends a disambiguation code.

Trie-constrained beam search over semantic IDs. The "decoder" here is a stand-in function returning log-probabilities; in a real system it is the Transformer decoder conditioned on the user history.

```python
import torch
torch.manual_seed(0)
LEVELS, K, BEAM = 3, 16, 4
valid_ids = {tuple(torch.randint(0, K, (LEVELS,)).tolist()) for _ in range(200)}  # catalogue

# Build a prefix trie: dict from prefix tuple -> set of allowed next codes
trie = {}
for sid in valid_ids:
    for l in range(LEVELS):
        trie.setdefault(sid[:l], set()).add(sid[l])

W = torch.randn(LEVELS, K, K)                     # toy decoder parameters
def next_logprobs(prefix):
    """Stand-in for decoder(history, prefix) -> log P(next code). Returns [K]."""
    prev = prefix[-1] if prefix else 0
    return torch.log_softmax(W[len(prefix), prev], dim=-1)

def beam_search(constrained=True):
    beams = [((), 0.0)]                           # (prefix, cumulative log-prob)
    for l in range(LEVELS):
        candidates = []
        for prefix, score in beams:
            lp = next_logprobs(prefix)
            if constrained:                       # mask codes that lead to no real item
                mask = torch.full((K,), float("-inf"))
                mask[list(trie.get(prefix, []))] = 0.0
                lp = lp + mask
            for c in range(K):
                if lp[c] > float("-inf"):
                    candidates.append((prefix + (c,), score + lp[c].item()))
        beams = sorted(candidates, key=lambda t: -t[1])[:BEAM]
    return beams

for flag in (False, True):
    out = beam_search(flag)
    n_valid = sum(sid in valid_ids for sid, _ in out)
    print(f"constrained={flag}: {n_valid}/{BEAM} beams are real items ->", [s for s, _ in out])
```

Without the trie most beams are tuples that match no item; with it, every beam is a real item. Adding a new item is one trie insertion.

## Worked example

**Residual coding by hand.** Take a 2-D latent $\mathbf{z}=(0.8,\ 0.3)$ and three codebooks of 4 codewords each:

| Level | Codewords (index: vector) |
|---|---|
| 1 | 0: $(1,0)$, 1: $(0,1)$, 2: $(-1,0)$, 3: $(0,-1)$ |
| 2 | 0: $(0.2,0.2)$, 1: $(0.2,-0.2)$, 2: $(-0.2,0.2)$, 3: $(-0.2,-0.2)$ |
| 3 | 0: $(0.05,0)$, 1: $(0,0.05)$, 2: $(-0.05,0)$, 3: $(0,-0.05)$ |

- **Level 1.** Squared distances from $(0.8,0.3)$: to $(1,0)$: $0.04+0.09=0.13$; to $(0,1)$: $0.64+0.49=1.13$; to $(-1,0)$: $3.24+0.09=3.33$; to $(0,-1)$: $0.64+1.69=2.33$. Pick **0**. Residual $\mathbf{r}_1 = (-0.2,\ 0.3)$.
- **Level 2.** Distances from $(-0.2,0.3)$: to $(0.2,0.2)$: $0.16+0.01=0.17$; to $(0.2,-0.2)$: $0.16+0.25=0.41$; to $(-0.2,0.2)$: $0+0.01=0.01$; to $(-0.2,-0.2)$: $0.25$. Pick **2**. Residual $\mathbf{r}_2=(0,\ 0.1)$.
- **Level 3.** Distances from $(0,0.1)$: to $(0,0.05)$: $0.0025$ (smallest). Pick **1**. Residual $\mathbf{r}_3=(0,\ 0.05)$.

Semantic ID $=(0, 2, 1)$. Reconstruction $\hat{\mathbf{z}} = (1,0)+(-0.2,0.2)+(0,0.05) = (0.8,\ 0.25)$; error $\lVert\mathbf{r}_3\rVert = 0.05$, versus $0.36$ after level 1 alone. A nearby item $(0.85, 0.2)$ gets level-1 code 0 too (shared prefix), while $(-0.7, 0.4)$ starts with code 2.

**Capacity.** With 3 levels × 256 codes, there are $256^3 \approx 16.8$M cells, but the decoder's output layer has only 256 classes per step (768 code embeddings in total), versus a softmax over, say, 100M item IDs. Storing 100M items as dense 64-dim fp32 vectors needs $100\text{M}\times 64\times 4\text{ B} = 25.6$ GB; storing their semantic IDs needs 3 bytes each, 300 MB, plus the codebooks.

## In practice (2025–2026)

- Semantic IDs are now a standard building block: generative retrieval (TIGER-style), ID features in ranking (YouTube), and the item vocabulary of end-to-end generative recommenders (OneRec, Chapter [4.6](06-end-to-end-generative-recommenders.md)) and LLM-based recommenders (PLUM, LC-Rec; Chapter [4.7](07-llms-in-recommendation.md)).
- Tokenizers increasingly quantize embeddings that already mix content *and* collaborative signal (e.g. multimodal embeddings aligned with user behaviour, as in Kuaishou's QARM), rather than pure text embeddings.
- Hybrid sparse+dense designs (COBRA, LIGER) are a common answer to generative retrieval's precision gap.
- Snap's **GRID** open-source framework systematically ablates tokenizer, architecture and decoding choices and finds that many overlooked architectural components substantially change results — use it as a starting point rather than re-implementing papers from scratch.

## Common pitfalls / misconceptions

- **Codebook collapse**: check codebook utilization per level; a level using 10 of 256 codes wastes capacity.
- **Treating semantic IDs as a free lunch** for popular items: they can lose memorization; keep some item-specific capacity (longer pieces, extra tokens, or a ranking stage with per-item features).
- **Unconstrained decoding** at serving: invalid IDs waste beams and return nothing.
- **Re-training the tokenizer without re-training the recommender**: codes change meaning; tokenizer updates must be versioned and coordinated.
- **Evaluating generative retrieval only on warm items**: its main advantage is often on cold and long-tail items.

## Key takeaways

- A semantic ID spells an item as a coarse-to-fine tuple of codes from residual quantization of its content embedding; similar items share prefixes.
- RQ-VAE = encoder + residual codebooks + decoder, trained with reconstruction, codebook and commitment losses and a straight-through gradient.
- TIGER trains a seq2seq model to decode the next item's semantic ID; trie-constrained beam search guarantees valid items and makes the index editable.
- Semantic-ID pieces also improve ID features in ranking by balancing memorization and generalization.
- Generative retrieval wins on memory, cold start and model–index consistency; dense retrieval wins on fine-grained precision; hybrids combine both.

## Check your understanding

??? question "Why does residual quantization give finer resolution than a single codebook with the same total number of codewords?"
    A single codebook of $D\cdot K$ codewords distinguishes only $D\cdot K$ cells. Residual quantization with $D$ codebooks of $K$ codewords composes choices, addressing up to $K^D$ cells, because each level quantizes the error of the previous ones.

??? question "What does the straight-through estimator do in the RQ-VAE?"
    It feeds the quantized vector to the decoder in the forward pass but copies the decoder's gradient to the encoder output as if quantization were the identity, so the encoder can be trained despite the non-differentiable argmin.

??? question "Why can TIGER recommend an item that never appeared in training data?"
    Its semantic ID is computed from content, so it lies in a region of code space the model has learned about from similar items; the decoder can generate that tuple even though the exact item was never a training target.

??? question "How do you add a newly uploaded item to a generative retriever without retraining?"
    Compute its content embedding, run it through the (frozen) tokenizer to get its semantic ID, and insert the ID into the trie of valid IDs used for constrained decoding.

??? question "Give one reason generative retrieval may underperform dense retrieval on popular items."
    Popular items benefit from item-specific memorized parameters; semantic IDs share parameters across similar items and may collide, blurring fine distinctions that per-item dense vectors capture.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991) — Tay et al., 2022 | A seq2seq model memorizes a corpus and decodes document IDs directly — the origin of generative retrieval. |
| :material-book-open-variant: [Autoregressive Image Generation using Residual Quantization (RQ-VAE)](https://arxiv.org/abs/2203.01941) — Lee et al., 2022 | Residual quantization with coarse-to-fine codes — the standard semantic-ID tokenizer. |
| :material-star: [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) — Rajput et al. (Google), NeurIPS 2023 | Semantic IDs via RQ-VAE + a Transformer that decodes the next item's semantic ID. The generative retrieval reference. |
| :material-star: [Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) — Singh et al. (Google/YouTube), 2023 | Semantic-ID n-gram/SentencePiece embeddings replace video IDs in a production ranking model. |
| :material-book-open-variant: [Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER)](https://arxiv.org/abs/2411.18814) — Yang et al. (Meta), 2024 | When generative retrieval falls short of dense retrieval, and a hybrid. |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | Sparse semantic ID, then a dense vector conditioned on it; deployed in Baidu ads. |
| :material-new-box: [Generative Recommendation with Semantic IDs: A Practitioner's Handbook (GRID)](https://arxiv.org/abs/2507.22224) — Ju et al. (Snap), 2025 | Open-source framework and ablations — the best hands-on starting point ([code](https://github.com/snap-research/GRID)). |
| :material-book-open-variant: [From Matching to Generation: A Survey on Generative Information Retrieval](https://arxiv.org/abs/2404.14851) — Li et al., 2024 | Generative retrieval in search and recommendation. |
