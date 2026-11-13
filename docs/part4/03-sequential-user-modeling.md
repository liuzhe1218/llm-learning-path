# 4.3 Sequential User Modeling

!!! abstract "What you'll learn"
    - Why the user's behaviour sequence is the single most valuable signal in modern recommenders
    - SASRec: a causal Transformer over item IDs — literally a GPT whose tokens are items — with runnable code
    - BERT4Rec's masked-item objective and when bidirectional models make sense
    - How to handle lifelong sequences of 10^4–10^5 actions: search-based interest models and compression
    - Realtime vs batch user representations (TransAct + PinnerFormer) and foundation user models shared by hundreds of downstream models (Meta's SUM)

## Why this matters / the problem

In Chapter [4.2](02-ranking-and-ctr-models.md) the user was mostly a bag of IDs and a pooled history vector, with DIN letting the candidate attend over recent behaviours. But *order* matters: someone who just watched three cooking videos in a row is in a different state than someone who watched one a month ago. And *length* matters: an active user on a large platform accumulates tens of thousands of actions, which contain long-term interests that the last 50 actions miss.

Sequence modelling is also the bridge to everything generative in this part. Once you treat a user's history as a sequence of tokens and train a model to predict the next one, you are doing language modelling with items as the vocabulary — and the full toolbox of [Part 1](../part1/index.md) (Transformers, causal masks, scaling laws) and [Part 3](../part3/index.md) (KV caches) applies.

## Intuition

Read a user's history as a sentence: `[running shoes, running socks, GPS watch, energy gels, ...]`. A language model reading this would guess the next word is something like "marathon registration" or "foam roller". A sequence recommender does exactly that: it reads the history left to right and outputs a distribution over the next item.

Three practical questions shape the designs in this chapter:

1. **Direction:** predict the next item from the past (causal, like GPT) or fill in masked items from both sides (bidirectional, like BERT)?
2. **Length:** attention is quadratic in sequence length; lifelong histories need retrieval or compression.
3. **Freshness vs cost:** recomputing a big sequence model on every action is expensive; computing it once a day misses the last ten minutes.

## How it works

### Notation

A user's history is $S = (s_1, s_2, \dots, s_T)$ with $s_t \in \mathcal{I}$ the item at step $t$ (optionally with an action type, timestamp, surface). An item embedding table $\mathbf{E}\in\mathbb{R}^{|\mathcal{I}|\times d}$ maps each item to a vector. The goal of **next-item prediction** is to model $P(s_{t+1} \mid s_1,\dots,s_t)$.

### SASRec: a causal Transformer over items

SASRec (2018) is a decoder-only Transformer (see [Part 1](../part1/index.md)):

1. Input $\mathbf{X}^{(0)}_t = \mathbf{E}[s_t] + \mathbf{P}_t$, where $\mathbf{P}_t$ is a learned position embedding. Shape `[batch, T, d]`.
2. $L$ Transformer blocks with **causal** self-attention (position $t$ sees only positions $\le t$), feed-forward layers, residuals and layer norm.
3. The output at position $t$, $\mathbf{h}_t \in \mathbb{R}^d$, is the user state after $t$ actions. The score of candidate item $i$ for the next step is $r_{t,i} = \mathbf{h}_t^\top \mathbf{E}[i]$ — the input embedding table is reused as the output layer (weight tying, exactly as in many LMs).

Training uses every position as a prediction target, like LM training: at position $t$ the label is $s_{t+1}$. The original paper used a binary cross-entropy with one sampled negative per position; with modern hardware and moderately sized catalogues a full or sampled softmax cross-entropy is common and usually stronger:

$$
\mathcal{L} = -\sum_{t=1}^{T-1}\log\frac{\exp(r_{t,s_{t+1}})}{\sum_{j\in\mathcal{I}}\exp(r_{t,j})}.
$$

In words: at every step, make the actual next item more likely than all others. At serving time, $\mathbf{h}_T$ (the last position) is the user vector, which can be fed to an ANN index over item embeddings — so SASRec doubles as a retrieval model.

### BERT4Rec: masked item modelling

BERT4Rec replaces the causal mask with full bidirectional attention and trains with a **Cloze** objective: randomly replace a fraction of items in the sequence with a `[MASK]` token and predict them from both left and right context. At inference, append `[MASK]` at the end and predict it. Bidirectional context gives more training signal per sequence, but it is a mismatch with the actual task (the future is never available at serving), and later replication studies have reported that causal models trained with a full (or well-sampled) softmax can match or beat it. In industry, causal models dominate because they also support streaming updates and KV-cache-style incremental computation.

### Beyond item IDs: side information and actions

Real systems put more into each token: the action type (click, like, skip), dwell time, timestamp gaps, surface, and item content features. A common pattern is to sum several embeddings per token (item + action + time bucket), or to interleave item tokens and action tokens. Generative Recommenders (Chapter [4.4](04-scaling-recommendation-models.md)) make this interleaving the core of the formulation.

### Long sequences: retrieve, then attend

Self-attention over $T$ tokens costs $O(T^2 d)$ per layer; target attention (DIN) costs $O(Td)$ per candidate, multiplied by hundreds of candidates. Neither is affordable for $T = 10^4$–$10^5$. Two families of solutions:

**Search-based interest modelling (SIM, Alibaba).** Split the work into two stages, just like the retrieval/ranking cascade but *inside* the user history:

1. **General Search Unit (GSU):** for the current candidate, retrieve the top-$K$ relevant behaviours from the lifelong sequence cheaply — either *hard search* (keep behaviours with the same category as the candidate) or *soft search* (inner product of behaviour and candidate embeddings, using an ANN-like index).
2. **Exact Search Unit (ESU):** run full target attention (or a small Transformer) over only those $K$ behaviours (e.g. a few hundred).

Hard search is especially cheap because behaviours can be pre-indexed by category per user. SIM reports serving user sequences of length up to 54,000 in Alibaba's display advertising, with 7.1% CTR and 4.4% RPM lift online.

**Compression.** Summarize the long history into a fixed number of vectors: pooling over time windows or categories, memory networks updated incrementally as actions arrive (Alibaba's MIMN, the predecessor of SIM, handled sequences up to ~1,000), or a set of learned query tokens that cross-attend to the full history (a Perceiver-style bottleneck). Compression can be computed asynchronously, outside the request path.

### Realtime vs batch user representations

There is a tension between expressiveness and freshness:

- **Batch user embeddings** are computed by a large sequence model offline (e.g. daily) and stored. Cheap to serve, can be huge, but stale: they do not know what the user did in the last hour.
- **Realtime sequence modules** run inside the ranking model on the most recent actions at request time. Fresh, but must be small to meet latency.

Pinterest combines both. **PinnerFormer** is a batch user model trained not on "next action" but on a *dense all-action loss*: predict the set of items the user will engage with over a future window (days). That objective makes a once-a-day embedding much less stale, significantly closing the gap to realtime embeddings. **TransAct** is a Transformer over the user's most recent realtime actions, trained end to end inside the Homefeed ranking model. The production ranker uses both: TransAct for short-term intent, PinnerFormer for long-term taste. This **hybrid realtime + batch** pattern is now standard.

The resulting ranking architecture looks like this:

```text
 user's last ~100 actions (streamed, seconds old) ──► realtime Transformer (TransAct-like) ──┐
 lifelong history (10^4+ actions)  ──► search by candidate (SIM GSU) ──► target attention ───┤
 daily batch job ──► long-term user embedding (PinnerFormer-like), looked up at request ─────┼──► ranking MLP / multi-task heads
 candidate item features, context features ─────────────────────────────────────────────────┘
```

### Foundation user models: train once, use everywhere

Large platforms run hundreds of ranking models (per surface, per objective, per ad type). Each cannot afford a big sequence model. Meta's **Scaling User Modeling (SUM)** framework trains a few large *upstream* user models on massive user features and serves the resulting user embeddings as input features to hundreds of *downstream* ads ranking models. An asynchronous online serving system (SOAP) recomputes embeddings on user requests without adding latency, with mechanisms for freshness and embedding stabilization so downstream models are not disrupted when the upstream model is retrained. This is the recsys analogue of a foundation model plus many lightweight task heads.

### Connecting to LLM next-token prediction

| Language model | Sequential recommender |
|---|---|
| Token from a fixed vocabulary of ~$10^5$ | Item from a churning vocabulary of $10^6$–$10^{10}$ |
| Position embeddings | Positions **and** timestamps / time gaps |
| Next-token cross-entropy over full vocab | Next-item loss with sampled softmax (vocab too big) |
| One document = one sequence | One user = one ever-growing sequence |
| KV cache for incremental decoding | Cached user state, updated as new actions arrive |
| Scaling laws in params and data | Scaling laws appear once the backbone is compute-heavy (Chapter 4.4) |

The vocabulary row is the big difference, and it motivates **semantic IDs** (Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)): replace billions of opaque item IDs with short sequences of shared tokens, so the vocabulary looks more like an LLM's.

## Code

A minimal SASRec trained on synthetic sequences where users walk through "topics" (items within a topic tend to follow each other). Full softmax is fine at this scale.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

N_ITEMS, T, D, N_TOPICS = 500, 20, 64, 25        # item 0 is reserved for padding
topic = torch.randint(0, N_TOPICS, (N_ITEMS,)); topic[0] = -1
member = (topic[None, :] == torch.arange(N_TOPICS)[:, None]).float()   # [N_TOPICS, N_ITEMS]

def make_batch(B):
    """Each user mostly stays in a topic, jumping to a random new topic with prob 0.15."""
    seqs = torch.zeros(B, T, dtype=torch.long)
    cur = torch.randint(0, N_TOPICS, (B,))
    for t in range(T):
        jump = torch.rand(B) < 0.15
        cur = torch.where(jump, torch.randint(0, N_TOPICS, (B,)), cur)
        seqs[:, t] = torch.multinomial(member[cur], 1).squeeze(1)   # uniform item in topic
    return seqs

class SASRec(nn.Module):
    def __init__(self, n_items, d, max_len, n_layers=2, n_heads=2):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.02)                # small init: tied output layer
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, seq):                       # seq: [B, L]
        L = seq.size(1)
        x = self.item(seq) + self.pos(torch.arange(L))[None]          # [B, L, d]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)    # True = may not attend
        h = self.blocks(x, mask=causal, src_key_padding_mask=(seq == 0))
        return self.norm(h) @ self.item.weight.T  # [B, L, n_items]; tied output layer

model = SASRec(N_ITEMS, D, T)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(500):
    seq = make_batch(128)
    logits = model(seq[:, :-1])                   # position t predicts s_{t+1}
    loss = F.cross_entropy(logits.reshape(-1, N_ITEMS), seq[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    seq = make_batch(500)
    top10 = model(seq[:, :-1])[:, -1].topk(10).indices                 # [500, 10]
    hr = (top10 == seq[:, -1:]).any(1).float().mean()
    same_topic = (topic[top10] == topic[seq[:, -2]][:, None]).float().mean()
print(f"loss {loss.item():.2f}  HitRate@10 {hr:.3f}  top-10 in current topic {same_topic:.2f}")
```

With 25 topics of ~20 items each, a model that knows the current topic but not the exact item gets HitRate@10 of roughly 0.4 (≈ half the topic's items in the top 10 × ≈ 0.86 chance the user stays in the topic); a random model gets 0.02. This run reaches about 0.44, with over 90% of the top-10 in the current topic. Swap the causal mask for no mask and a `[MASK]`-token objective and you have BERT4Rec.

Two building blocks used around this model. First, the BERT4Rec masking step; second, a SIM-style "hard search" that keeps only same-category behaviours before target attention:

```python
import torch
torch.manual_seed(0)
MASK_ID = 999                                      # an extra token id reserved for [MASK]

def bert4rec_mask(seq, p=0.2):
    """Cloze masking: hide a fraction p of real items; labels are -100 (ignored) elsewhere."""
    seq = seq.clone()
    is_item = seq != 0
    masked = (torch.rand(seq.shape) < p) & is_item
    labels = torch.where(masked, seq, torch.full_like(seq, -100))
    seq[masked] = MASK_ID
    return seq, labels                             # train with cross_entropy(..., ignore_index=-100)

def sim_hard_search(hist_items, hist_cats, cand_cat, emb, K=4):
    """GSU: keep the most recent K behaviours in the candidate's category; ESU: attention."""
    idx = (hist_cats == cand_cat).nonzero().squeeze(1)[-K:]   # most recent matches
    if len(idx) == 0:
        return torch.zeros(emb.shape[1])
    h = emb[hist_items[idx]]                                   # [k, d]
    c = emb[hist_items[idx[-1]]]                               # stand-in candidate vector
    w = torch.softmax(h @ c / h.shape[1] ** 0.5, dim=0)        # ESU: attention over k items
    return (w[:, None] * h).sum(0)                             # [d]

seq = torch.tensor([[5, 17, 23, 42, 8, 0, 0]])                 # 0 = padding
print(bert4rec_mask(seq, p=0.5))
emb = torch.randn(1000, 16)
hist_items = torch.randint(1, 1000, (10_000,))                 # a 10k-long lifelong history
hist_cats = torch.randint(0, 50, (10_000,))
print(sim_hard_search(hist_items, hist_cats, cand_cat=7, emb=emb).shape)   # torch.Size([16])
```

In production the per-category lists are pre-built per user (a two-level index: user → category → behaviours), so the GSU is a lookup, not a scan, and the ESU uses the real candidate embedding.

## Worked example

**Cost of lifelong attention.** A user has $T = 10{,}000$ past actions, embedding size $d=64$, and the ranker scores $N = 500$ candidates.

- *Full causal self-attention* over the history, one layer: the $QK^\top$ and $AV$ products cost about $2\cdot 2 \cdot T^2 d = 4\cdot 10^8 \cdot 64 \approx 2.6\times10^{10}$ FLOPs (26 GFLOPs) per layer per request — before projections and MLPs. Shared across candidates, but large.
- *DIN-style target attention* with a dot-product scorer: per candidate $\approx 2Td$ for scores plus $2Td$ for the weighted sum $= 4 \cdot 10^4 \cdot 64 \approx 2.6$ MFLOPs; over 500 candidates $\approx 1.3$ GFLOPs. With DIN's MLP scorer (input $4d$, hidden 32) the score alone is $\approx 2\cdot 4d\cdot 32 = 16{,}384$ FLOPs per behaviour, i.e. $\approx 1.6\times10^8$ per candidate and $\approx 8\times10^{10}$ per request — too expensive.
- *SIM:* the GSU retrieves $K = 200$ behaviours per candidate (hard search by category is essentially a lookup). The ESU's MLP-scored attention now costs $200 \cdot 16{,}384 \approx 3.3$ MFLOPs per candidate, $\approx 1.6$ GFLOPs per request: a 50× reduction, while attending to the behaviours most relevant to each candidate.

**Freshness gap.** A batch embedding computed at 02:00 is, on average, ~12 hours stale for a request at a uniformly random time of day. Actions in the current session — often the most predictive — are invisible to it. That is why a small realtime module (TransAct-style) is paired with it.

## In practice (2025–2026)

- Every major feed and ads ranker uses some form of sequence model over user actions; target attention over recent actions plus retrieved long-term behaviours is the standard classic design.
- **Generative Recommenders (HSTU)** make the action sequence *the* input, replacing most hand-engineered user features (Chapter 4.4).
- **Foundation user models** (SUM-style upstream embeddings, LUM's pre-trained large user model, Chapter 4.4) are shared across many downstream models.
- LLM-based user encoders (HLLM, User-LLM) bring pre-trained language-model weights into user modelling (Chapter [4.7](07-llms-in-recommendation.md)).

!!! tip "Exercise"
    Extend the SASRec code with a time-gap embedding: bucket the log of the seconds since the previous action into 16 buckets and add a learned embedding per bucket to each token. Generate synthetic data where users switch topics after long gaps, and check that HitRate improves.

## Common pitfalls / misconceptions

- **Leaking the future**: forgetting the causal mask, or building sequences that include the label item, gives spectacular offline numbers and nothing online.
- **Evaluating with sampled negatives only** (rank the true item among 100 random ones) can change model rankings compared to full-catalogue evaluation; prefer full ranking when feasible.
- **Treating BERT4Rec as automatically better** because BERT beat GPT on some NLP benchmarks; for next-item prediction, causal training matches the task.
- **Ignoring time**: two actions a second apart and a month apart are not the same; encode timestamps or gaps.
- **Retraining an upstream user model without stabilization** silently shifts the input distribution of every downstream model.

## Key takeaways

- The user's action sequence is the strongest signal; modelling its order and length is where much of the modern gain comes from.
- SASRec is a GPT over item IDs: causal Transformer, next-item cross-entropy, tied embeddings.
- Lifelong sequences need a cascade inside the history: cheap search for relevant behaviours, then exact attention (SIM), or compression.
- Production systems mix fresh realtime modules with stale-but-rich batch embeddings, and share foundation user models across many downstream models.

## Check your understanding

??? question "In SASRec, why is the output layer tied to the input item embedding table?"
    The score $\mathbf{h}_t^\top\mathbf{E}[i]$ puts user states and items in the same space, halving parameters (the item table is the largest component) and letting the same embeddings serve as an ANN index for retrieval. It mirrors weight tying in language models.

??? question "Why does the GSU in SIM not need to be very accurate?"
    Like retrieval in the main cascade, its job is recall: to include the relevant behaviours among $K$ candidates. The ESU then does precise weighting. Cheap category matching is enough as long as relevant behaviours are not dropped.

??? question "What problem does PinnerFormer's dense all-action loss solve?"
    A batch user embedding is computed once per day but used for the whole day. Training it to predict only the very next action makes it brittle when stale; training it to predict all engagements over a future window makes it robust to that staleness.

??? question "List two ways sequential recommendation differs from language modelling."
    The vocabulary is orders of magnitude larger and changes daily (requiring sampled softmax or semantic IDs), and sequences carry timestamps, action types and other side information; user sequences also keep growing indefinitely rather than being fixed documents.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) — Kang & McAuley, 2018 | Causal Transformer over item sequences for next-item prediction — essentially a GPT over item IDs. |
| :material-book-open-variant: [BERT4Rec](https://arxiv.org/abs/1904.06690) — Sun et al., 2019 | Bidirectional masked-item modelling for sequential recommendation. |
| :material-book-open-variant: [Deep Interest Evolution Network for Click-Through Rate Prediction (DIEN)](https://arxiv.org/abs/1809.03672) — Zhou et al. (Alibaba), 2018 | Adds interest evolution over time to DIN with GRUs and an auxiliary loss; deployed in Taobao display ads. |
| :material-star: [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction (SIM)](https://arxiv.org/abs/2006.05639) — Pi et al. (Alibaba), 2020 | The two-stage GSU/ESU design for sequences up to 54,000 behaviours. |
| :material-book-open-variant: [TransAct](https://arxiv.org/abs/2306.00248) / [PinnerFormer](https://arxiv.org/abs/2205.04507) — Pinterest, 2022–2023 | Realtime Transformer over recent actions combined with long-term batch user embeddings — the hybrid realtime + batch pattern. |
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | SUM: a few large upstream user models produce embeddings consumed by hundreds of downstream ads models; covers online embedding freshness. |
