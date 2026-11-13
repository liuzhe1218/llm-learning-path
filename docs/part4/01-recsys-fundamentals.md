# 4.1 Recommender-System Fundamentals

!!! abstract "What you'll learn"
    - How to state the recommendation problem precisely, and the difference between explicit and implicit feedback
    - Why industrial systems use a multi-stage cascade (retrieval → pre-ranking → ranking → re-ranking), sized by latency budgets
    - How a two-tower retrieval model is trained with an in-batch sampled softmax (derived and coded) and served with ANN search (HNSW, IVF-PQ)
    - How to compute Recall@K, NDCG, AUC, GAUC and normalized entropy by hand, and why offline wins do not always become online wins
    - Why recommenders live in a feedback loop, suffer from position/exposure bias, and are trained online for (usually) one epoch

## Why this matters / the problem

A recommender answers one question millions of times per second: *given this user, right now, which few items should we show?* The catalogue may contain billions of items (every short video ever uploaded, every product in a marketplace, every active ad), the user expects a response in well under a second, and the system only ever learns from what it chose to show in the past.

Everything that makes modern recommenders look the way they do — embedding tables, cascades, two-tower models, streaming training, and the generative models of later chapters — comes from three constraints:

1. **Scale of the candidate set.** You cannot run a large neural network on a billion items per request.
2. **Latency.** The whole pipeline typically has a budget on the order of a hundred milliseconds or less end to end, and each stage gets a slice of it.
3. **Learning from your own decisions.** Labels exist only for items that were shown, so the data is biased by the previous model.

This chapter builds the classic stack from first principles. Later chapters replace pieces of it, so you need to know what each piece does.

## Intuition

Think of a large library with a librarian who has a few seconds per visitor. She cannot read every book's summary to every visitor. Instead she:

1. walks to a few shelves that match the visitor's broad interests (**retrieval**: cheap, recall-oriented, thousands of books);
2. glances at the covers and drops obviously poor fits (**pre-ranking**: a light model, hundreds left);
3. carefully considers the remaining books against everything she knows about the visitor (**ranking**: the expensive model, scores each candidate);
4. arranges the final handful so they are not all by the same author and includes the promoted titles the library is paid to feature (**re-ranking**: diversity, business rules, ad auction).

Each stage trades accuracy for speed. The cheap stages must not lose the good items (high **recall**); the expensive stage must order them well (high **precision** at the top).

## How it works

### The recommendation problem

Let $\mathcal{U}$ be the set of users and $\mathcal{I}$ the set of items. For a user $u \in \mathcal{U}$ in context $c$ (time, device, page, query if any), we want a scoring function $s(u, i, c)$ such that sorting items by $s$ puts the items the user will value most at the top. In practice "value" is approximated by predicted probabilities of observable actions, e.g. $P(\text{click} \mid u, i, c)$, expected watch time, or $P(\text{purchase} \mid u, i, c)$.

**Explicit feedback** is a direct statement of preference: a 1–5 star rating, a thumbs-up. It is clean but rare (most users never rate anything). **Implicit feedback** is inferred from behaviour: impressions, clicks, dwell time, watch completion, likes, shares, purchases, skips. It is abundant but noisy and one-sided: a click is weak positive evidence, and *not* clicking may mean "not interested" or simply "didn't see it". Almost all industrial systems learn from implicit feedback, so most models are trained as **binary or multi-task classifiers** over impressions ("was this shown item clicked / watched / bought?").

### Why a cascade: candidate sizes and latency budgets

Suppose the ranking model costs 10 MFLOPs per (user, item) pair and we have $10^9$ items. Scoring everything would be $10^{16}$ FLOPs *per request*, which is impossible at thousands of requests per second. The cascade fixes this by spending compute where it matters:

| Stage | Input size | Output size | Model | Typical goal |
|---|---|---|---|---|
| Retrieval (candidate generation) | $10^6$–$10^{10}$ | $10^3$–$10^4$ | Two-tower + ANN, item-to-item graphs, popularity, many sources merged | High recall, cheap |
| Pre-ranking | $10^3$–$10^4$ | $10^2$–$10^3$ | Small DNN or two-tower with richer features | Cheap filtering |
| Ranking | $10^2$–$10^3$ | same, scored | Heavy multi-task network with user history and cross features (Chapter [4.2](02-ranking-and-ctr-models.md)) | Accurate ordering and calibrated probabilities |
| Re-ranking | $10^2$ | $\sim 10$ | Rules, diversity models, ads auction (Chapter [4.8](08-ads-systems.md)) | Whole-page quality, business constraints |

The numbers are orders of magnitude, not a standard; every company tunes them. The key structural fact is that **retrieval cannot look at user–item feature interactions** in detail, because it must search a huge set in milliseconds. That is why retrieval models have the special "two-tower" shape below.

### Two-tower retrieval

A **two-tower** (dual-encoder) model has a user tower $f_\theta$ and an item tower $g_\phi$, each a neural network mapping features to a $d$-dimensional vector:

$$
\mathbf{u} = f_\theta(\text{user features}) \in \mathbb{R}^d, \qquad
\mathbf{v}_i = g_\phi(\text{item } i \text{ features}) \in \mathbb{R}^d, \qquad
s(u,i) = \mathbf{u}^\top \mathbf{v}_i .
$$

In words: user and item are embedded into the same space and the score is their dot product. The crucial property is that the item vectors do not depend on the user, so we can compute all $\mathbf{v}_i$ offline, put them in an index, and at request time compute only one user vector and search for its nearest neighbours.

**Training as extreme classification.** Treat "which item did the user engage with next?" as a classification over all items. The softmax probability of item $i$ is

$$
P(i \mid u) = \frac{\exp(s(u,i)/\tau)}{\sum_{j \in \mathcal{I}} \exp(s(u,j)/\tau)},
$$

where $\tau > 0$ is a temperature. For a positive pair $(u, i^+)$ the loss is $-\log P(i^+ \mid u)$. The denominator sums over all items, which is unaffordable, so we approximate it with a **sample**.

**In-batch negatives.** Take a batch of $B$ positive pairs $(u_b, i_b)$, $b = 1..B$. Compute user vectors $\mathbf{U} \in \mathbb{R}^{B \times d}$ and item vectors $\mathbf{V} \in \mathbb{R}^{B \times d}$, and the logit matrix $\mathbf{S} = \mathbf{U}\mathbf{V}^\top / \tau \in \mathbb{R}^{B \times B}$. For row $b$, the diagonal entry $S_{bb}$ is the positive and the other $B-1$ items (positives of other users) act as negatives:

$$
\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B} \log \frac{\exp(S_{bb})}{\sum_{k=1}^{B} \exp(S_{bk})}.
$$

This is just cross-entropy with label $b$ for row $b$ — the same loss as CLIP's or a retrieval-augmented LM's contrastive objective.

**The popularity bias and the logQ correction.** In-batch negatives are not uniform samples: an item appears in a batch with probability proportional to how often it is a positive, so popular items are over-represented as negatives and get pushed down too hard. Sampled-softmax theory says that if negatives are drawn from a proposal distribution $Q(j)$, an unbiased-in-the-limit estimate of the full softmax uses corrected logits

$$
S^{c}_{bk} = S_{bk} - \log Q(i_k),
$$

where $Q(i_k)$ is the probability that item $i_k$ appears in a batch (estimated with streaming frequency counts). In words: subtract the log sampling probability, so an item that is sampled very often is penalised less each time it appears. This is the "sampling-bias-corrected" two-tower recipe used at YouTube.

??? info "Why subtracting log Q works (sketch)"
    The full softmax denominator is $Z = \sum_j e^{s_j}$. If we draw negatives $j \sim Q$, then $\mathbb{E}_{j\sim Q}[e^{s_j}/Q(j)] = \sum_j e^{s_j} = Z$, i.e. importance-weighting each sampled term by $1/Q(j)$ gives an unbiased estimate of the denominator. Weighting a term by $1/Q(j)$ is the same as adding $-\log Q(j)$ to its logit, because $e^{s_j}/Q(j) = e^{s_j - \log Q(j)}$.

### Approximate nearest neighbour (ANN) search

At serving time we need the top-$K$ items by $\mathbf{u}^\top\mathbf{v}_i$ out of $N$ item vectors. Exact search costs $O(Nd)$ per request; ANN trades a little recall for orders of magnitude speed.

- **IVF (inverted file).** Cluster item vectors into $C$ clusters with k-means. At query time, compare $\mathbf{u}$ to the $C$ centroids, pick the closest `nprobe` clusters, and only score items inside them. Cost drops from $N$ to roughly $C + \texttt{nprobe}\cdot N/C$ dot products.
- **PQ (product quantization).** Compress each vector by splitting it into $m$ sub-vectors and replacing each with the index of its nearest centroid in a small per-subspace codebook (e.g. 256 centroids → 1 byte). A 128-dim float32 vector (512 bytes) becomes $m$ bytes. Dot products are computed from small lookup tables. **IVF-PQ** combines both and is the workhorse of billion-scale GPU search (Faiss).
- **HNSW (hierarchical navigable small world graphs).** Build a graph where each vector links to its near neighbours, plus sparser "express" layers on top. Search starts at the top layer, greedily walks toward the query, then descends to denser layers. It behaves like a skip list over a proximity graph and gives roughly logarithmic search cost with high recall, at the price of more memory.

Keep in mind that the ANN index is built separately from model training. When the model is updated continuously, item vectors must be re-indexed, and the index can be stale — one of the motivations for generative retrieval in Chapter [4.5](05-semantic-ids-and-generative-retrieval.md).

### Offline metrics

Let a model return a ranked list for each user. Define $\text{rel}_k \in \{0,1\}$ (or graded) as the relevance of the item at position $k$ (1-indexed).

- **Recall@K** $= \dfrac{\#\{\text{relevant items in top } K\}}{\#\{\text{relevant items}\}}$. The main retrieval metric: did we keep the good items?
- **DCG@K** $= \sum_{k=1}^{K} \dfrac{2^{\text{rel}_k}-1}{\log_2(k+1)}$, and **NDCG@K** $= \text{DCG@K} / \text{IDCG@K}$, where IDCG is the DCG of the ideal ordering. It rewards putting relevant items *near the top*; the $\log_2(k+1)$ discount says position 1 matters most. For binary relevance $2^{\text{rel}}-1 = \text{rel}$.
- **AUC** (area under the ROC curve) = the probability that a randomly chosen positive is scored higher than a randomly chosen negative. It measures ordering only and ignores calibration.
- **GAUC** (group AUC) = AUC computed per user, then averaged with weights (usually the user's number of impressions or clicks). Global AUC can be inflated by ranking *users* (active users click more), which the ranking model never needs to do — it only orders items *within* a request. GAUC removes that.
- **LogLoss** $= -\frac{1}{n}\sum_{t=1}^n [y_t \log p_t + (1-y_t)\log(1-p_t)]$ for labels $y_t \in\{0,1\}$ and predictions $p_t$. It is sensitive to calibration.
- **Normalized entropy (NE)** $= \text{LogLoss} / H(\bar p)$ where $H(\bar p) = -[\bar p\log\bar p + (1-\bar p)\log(1-\bar p)]$ is the LogLoss of always predicting the background rate $\bar p$. NE < 1 means better than the trivial predictor; lower is better. Facebook's ads paper popularised it because raw LogLoss depends on the base CTR.

### Online metrics and the offline–online gap

Offline metrics are computed on logs of what the *old* system showed. Online, the new model changes what is shown, which changes behaviour. The final arbiter is an **A/B test**: randomly split users, serve old vs new, and compare business metrics such as CTR, watch time, daily active users, retention (e.g. 7-day return), revenue per mille (RPM), and guardrails (latency, complaints, diversity). Online effects of a successful launch are often small in relative terms (fractions of a percent on time spent), yet worth a great deal at scale.

### Feedback loops, position bias and exposure bias

The training data is produced by the previous model. This creates:

- **Exposure bias:** items never shown never get labels, so the model cannot learn they are good. Popular items get more exposure, more positives, more exposure ("rich get richer").
- **Position bias:** items shown at the top get clicked more *because* they are at the top. A naive model learns "whatever was at position 1 is great". A common fix is to feed the position as a feature during training and set it to a fixed value (e.g. position 1) at serving, or to train a separate shallow "position tower" whose output is dropped at inference.
- **Exploration** (showing some items with uncertain scores) is needed to keep learning; bandit-style exploration and fresh-item boosts are standard.

### Online / streaming training and the one-epoch phenomenon

User interests, items and trends change daily, so industrial models are trained **continuously** on a stream of fresh logs and pushed to serving every few minutes to hours. Two consequences:

1. Each example is usually seen **once**. This is not just a compute choice: deep CTR models show a **one-epoch phenomenon** — when trained for a second epoch on the same data, test performance drops sharply right at the start of epoch 2. An Alibaba study attributes it to the embedding layer of very sparse ID features memorizing the training samples when combined with fast-converging optimizers (e.g. Adam).
2. Evaluation is **progressive**: train on day $t$, evaluate on day $t+1$, never shuffle across time.

## Code

A complete two-tower model with in-batch softmax and the logQ correction, trained on synthetic data where each user has a hidden "taste" cluster.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

N_USERS, N_ITEMS, N_CLUSTERS, D = 1000, 5000, 20, 32
user_cluster = torch.randint(0, N_CLUSTERS, (N_USERS,))
item_cluster = torch.randint(0, N_CLUSTERS, (N_ITEMS,))
item_pop = torch.distributions.Pareto(1.0, 1.2).sample((N_ITEMS,))  # heavy-tailed popularity

# P[c, i] ∝ popularity of item i if it belongs to cluster c, else 0   -> [N_CLUSTERS, N_ITEMS]
P = (item_cluster[None, :] == torch.arange(N_CLUSTERS)[:, None]).float() * item_pop[None, :]

def sample_batch(B):
    """Positive (user, item) pairs: each user picks a popular item from its own cluster."""
    users = torch.randint(0, N_USERS, (B,))
    items = torch.multinomial(P[user_cluster[users]], 1).squeeze(1)   # [B]
    return users, items

class Tower(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.emb = nn.Embedding(n, 64)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d))
    def forward(self, ids):                      # ids: [B]
        return F.normalize(self.mlp(self.emb(ids)), dim=-1)   # [B, d], unit norm

user_tower, item_tower = Tower(N_USERS, D), Tower(N_ITEMS, D)
opt = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=1e-2)
item_count = torch.ones(N_ITEMS)                 # streaming frequency estimate for log Q
tau, B = 0.05, 256

for step in range(1500):
    users, items = sample_batch(B)
    item_count.index_add_(0, items, torch.ones(B))
    logQ = torch.log(item_count[items] / item_count.sum())       # [B]
    U, V = user_tower(users), item_tower(items)                  # [B, d] each
    logits = U @ V.T / tau - logQ[None, :]                       # [B, B], logQ correction per column
    # mask accidental duplicates: same item as positive for two rows is not a negative
    dup = (items[:, None] == items[None, :]) & ~torch.eye(B, dtype=torch.bool)
    logits = logits.masked_fill(dup, float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(B))              # label of row b is column b
    opt.zero_grad(); loss.backward(); opt.step()

# Evaluate Recall@50 with exact (brute-force) search over all items
with torch.no_grad():
    users, items = sample_batch(1000)
    scores = user_tower(users) @ item_tower(torch.arange(N_ITEMS)).T   # [1000, N_ITEMS]
    top = scores.topk(50, dim=1).indices
    hit_cluster = (item_cluster[top] == user_cluster[users][:, None]).float().mean()
    recall = (top == items[:, None]).any(1).float().mean()
print(f"loss {loss.item():.3f}  Recall@50 {recall:.3f}  same-cluster share in top-50 {hit_cluster:.2f}")
```

Things to notice: the loss is plain `cross_entropy` over a $B\times B$ matrix; the item vectors are normalized so the dot product is a cosine; and in production the brute-force `topk` is replaced by an ANN index (e.g. Faiss IVF-PQ or HNSW) built over `item_tower(all items)`.

## Worked example

**NDCG and Recall.** A user has 3 relevant items in the test set. The model's top-5 list has relevance $[1, 0, 1, 0, 0]$ (binary).

- Recall@5 $= 2/3 \approx 0.667$.
- DCG@5 $= \frac{1}{\log_2 2} + \frac{1}{\log_2 4} = 1 + 0.5 = 1.5$.
- Ideal list puts all 3 relevant items first: IDCG@5 $= \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} = 1 + 0.631 + 0.5 = 2.131$.
- NDCG@5 $= 1.5 / 2.131 \approx 0.704$.

If the model had swapped positions 2 and 3 (relevance $[1,1,0,0,0]$), DCG $= 1 + 0.631 = 1.631$ and NDCG $\approx 0.765$: same recall, better ranking, higher NDCG.

**AUC.** Five impressions with scores and labels: $(0.9, 1), (0.8, 0), (0.7, 1), (0.4, 0), (0.2, 0)$. There are $2 \times 3 = 6$ (positive, negative) pairs. The positive at 0.9 beats all three negatives (3 pairs); the positive at 0.7 beats 0.4 and 0.2 but loses to 0.8 (2 pairs). AUC $= 5/6 \approx 0.833$.

**GAUC.** User A has 100 impressions and AUC 0.70; user B has 300 impressions and AUC 0.60. Impression-weighted GAUC $= (100\cdot 0.70 + 300 \cdot 0.60)/400 = 0.625$.

**Normalized entropy.** Background CTR $\bar p = 0.02$:
$H(0.02) = -(0.02\ln 0.02 + 0.98\ln 0.98) = 0.0782 + 0.0198 = 0.0980$ nats.
A model with LogLoss $0.0850$ has NE $= 0.0850/0.0980 \approx 0.867$, i.e. 13.3% less uncertainty than always predicting 2%.

**Latency budget.** Suppose the end-to-end budget is 150 ms: ~20 ms retrieval (several sources in parallel, each an ANN lookup), ~10 ms pre-ranking over 5,000 candidates, ~40 ms ranking over 500 candidates, ~10 ms re-ranking, and the rest for feature fetching and network. With 500 candidates and 40 ms, the ranking model must score a candidate in well under 0.1 ms of amortized time — achievable only with batching on GPU/CPU and by computing user-side features once per request rather than once per candidate. (These are illustrative numbers, not a specific company's.)

## In practice (2025–2026)

- **Retrieval is multi-source.** Production systems merge candidates from two-tower ANN, item-to-item co-engagement graphs, follow/subscription lists, trending pools and, increasingly, generative retrievers that decode semantic IDs (Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)).
- **Beyond dot products.** Two-tower models can only express a dot product between user and item. Meta's HSNN replaces it with richer learned interaction modules over a hierarchical item index while keeping cost sublinear in corpus size, and has been deployed for ads retrieval.
- **The cascade itself is under attack.** End-to-end generative recommenders (Chapter [4.6](06-end-to-end-generative-recommenders.md)) generate the final list with one model, removing the mismatch between stages that are trained separately.
- **Metrics:** NE/LogLoss and GAUC remain the standard offline gates for ranking; Recall@K for retrieval; A/B tests with retention and time-spent for launch decisions.

## Common pitfalls / misconceptions

- **"Higher AUC means better recommendations."** AUC on logged data ignores items never shown and ignores calibration; a model can gain AUC and lose online.
- **Shuffling across days** in offline evaluation leaks the future and overstates quality. Always split by time.
- **Forgetting the logQ correction** makes in-batch-trained retrievers under-recommend popular items.
- **Treating non-clicks as true negatives** without considering position and exposure.
- **Training for many epochs** as in vision/NLP: deep CTR models usually overfit sharply after one epoch.
- **Comparing models with different item-index freshness**: an ANN index built yesterday may be the real bottleneck, not the model.

## Key takeaways

- The cascade exists because the corpus is huge and latency is small: cheap high-recall stages feed expensive high-precision ones.
- Two-tower models enable retrieval because item vectors are user-independent and can be indexed for ANN search; they are trained with in-batch sampled softmax plus a logQ correction.
- Recall@K for retrieval; NDCG, AUC/GAUC and NE/LogLoss for ranking; A/B tests decide.
- Logs are produced by the previous model, so bias (position, exposure, popularity) and exploration are first-class concerns.
- Industrial models train online on a stream and usually for a single epoch.

## Check your understanding

??? question "Why can't the ranking model's architecture be used for retrieval?"
    Ranking models compute rich interactions between user and item features (cross features, target attention), so every (user, item) pair needs a full forward pass. Over $10^9$ items that is far too expensive. Two-tower models restrict the interaction to a dot product so item vectors can be precomputed and searched with ANN in sublinear time.

??? question "In an in-batch softmax with batch size 4, how many negatives does each positive get, and what is the label vector?"
    Each row has $B - 1 = 3$ negatives (the other rows' positive items). The logits are a $4\times 4$ matrix and the labels are $[0, 1, 2, 3]$: row $b$'s correct column is $b$.

??? question "Compute NDCG@3 for relevance [0, 1, 1] when the user has exactly 2 relevant items."
    DCG $= 0 + 1/\log_2 3 + 1/\log_2 4 = 0.631 + 0.5 = 1.131$. IDCG $= 1 + 0.631 = 1.631$. NDCG $= 1.131/1.631 \approx 0.693$.

??? question "Why is GAUC usually preferred over AUC for evaluating a ranking model?"
    Global AUC also rewards separating heavy users (who click a lot) from light users, which the ranker never has to do because it only orders items for one user at a time. GAUC computes AUC within each user and averages, measuring exactly the within-request ordering.

??? question "What is the one-epoch phenomenon and what does it imply for training pipelines?"
    Deep CTR models' test performance drops sharply at the beginning of a second pass over the same data, linked to the sparse ID embedding layer and fast-converging optimizers. Pipelines therefore train in a single pass over a time-ordered stream rather than iterating many epochs.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — Covington et al., RecSys 2016 | The canonical two-stage (candidate generation + ranking) deep recommender; frames retrieval as extreme multiclass classification over videos. |
| :material-star: [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) — Yi et al. (Google), RecSys 2019 | The in-batch softmax two-tower recipe with streaming frequency estimation for the logQ correction. |
| :material-book-open-variant: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2016 | The HNSW algorithm behind most graph-based ANN indexes. |
| :material-book-open-variant: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) — Johnson, Douze & Jégou, 2017 | The Faiss paper: IVF-PQ on GPUs at billion scale ([code](https://github.com/facebookresearch/faiss)). |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | Normalized entropy as the metric; data freshness matters most. |
| :material-book-open-variant: [Towards Understanding the Overfitting Phenomenon of Deep CTR Models](https://arxiv.org/abs/2209.06053) — Zhang et al. (Alibaba), 2022 | Explains the one-epoch phenomenon: CTR models overfit sharply at the start of the 2nd epoch, driven by the embedding layer. |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | Scales retrieval beyond two-tower dot products with hierarchical clustering and richer interaction modules, deployed for ads. |
