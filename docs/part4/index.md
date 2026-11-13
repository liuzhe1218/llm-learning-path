# Part 4 · LLMs & Generative Modeling in Ads Ranking and Recommendation

Every time you open a short-video app, scroll a social feed or see an ad next to a search result, a **recommender system** has picked a few items out of millions or billions in a few tens of milliseconds. These systems are some of the largest machine-learning deployments on Earth: they are trained on tens of billions of user actions per day, they decide what billions of people see, and for ad-funded companies they decide where most of the revenue comes from.

For a decade, industrial recommenders looked very different from language models. They were built from huge **ID embedding tables**, hand-designed **feature-crossing networks**, and a **cascade** of separate models that progressively narrow the candidate set. Since 2024 that has been changing quickly. The ideas you met in Parts 1–3 — Transformers over sequences, **scaling laws**, **next-token prediction**, **preference alignment with RL/DPO**, and **efficient serving with KV caches** — are being carried over, producing *generative recommenders* and *large recommendation models* that improve predictably as compute grows.

This part teaches the field from scratch. We start with how a classic recommendation and ads stack actually works (so you know what is being replaced and why), then build up to generative retrieval with semantic IDs, end-to-end generative recommenders aligned with RL, LLMs as recommenders and feature providers, and finally the economics of ads: auctions, calibration and auto-bidding.

## Prerequisites and time

- **Math:** linear algebra (dot products, matrix multiplication), probability (Bernoulli, expectation, Bayes' rule), basic calculus for gradients.
- **Code:** Python and basic PyTorch (`nn.Module`, `nn.Embedding`, an optimizer loop).
- **From this site:** the Transformer and next-token prediction from [Part 1](../part1/index.md); DPO and RL-based alignment from [Part 2](../part2/index.md); KV caching, batching and serving cost from [Part 3](../part3/index.md). Each chapter links back to the specific idea it reuses.
- **Estimated time:** 25–35 hours of reading and exercises; double that if you do all the hands-on projects.

## Chapter map

| Chapter | What you will learn |
|---|---|
| [4.1 Recommender-system fundamentals](01-recsys-fundamentals.md) | The recommendation problem, feedback types, the multi-stage cascade, two-tower retrieval with sampled softmax, ANN search, offline and online metrics, feedback loops and streaming training |
| [4.2 Ranking and CTR models](02-ranking-and-ctr-models.md) | CTR prediction as classification, embedding tables, LR → FM → Wide & Deep → DCN → DLRM, target attention (DIN), multi-task learning, score fusion, calibration, embedding collapse |
| [4.3 Sequential user modeling](03-sequential-user-modeling.md) | User behaviour sequences, SASRec ("GPT over item IDs"), BERT4Rec, lifelong sequences (search-based interest models), realtime vs batch user embeddings, foundation user models |
| [4.4 Scaling recommendation models](04-scaling-recommendation-models.md) | Why DLRMs plateau, scaling laws for recsys, HSTU / Generative Recommenders in depth, Wukong, RankMixer, MTGR, LUM, MARM, compute budgeting |
| [4.5 Semantic IDs and generative retrieval](05-semantic-ids-and-generative-retrieval.md) | Vector and residual quantization, RQ-VAE, semantic-ID collisions, TIGER, beam search with trie-constrained decoding, semantic IDs in ranking, generative vs dense retrieval |
| [4.6 End-to-end generative recommenders](06-end-to-end-generative-recommenders.md) | Replacing the cascade with one model: OneRec, reward models, iterative DPO and RL, OneRec-V2, reasoning in recsys, PLUM, open problems |
| [4.7 LLMs in recommendation](07-llms-in-recommendation.md) | LLM as recommender, as offline knowledge/feature provider, as user/item encoder; 360Brew, RecGPT; latency and cost design patterns |
| [4.8 Ads systems](08-ads-systems.md) | Ad auctions (eCPM, GSP, VCG) with worked prices, why calibration matters, conversion modeling and delayed feedback, budgets/pacing/auto-bidding, generative bidding and generative advertising |

## Mental model

The classic industrial stack is a **funnel** of increasingly expensive models, each scoring fewer candidates:

```text
             corpus: 10^6 – 10^10 items (videos, posts, products, ads)
                              │
   ┌──────────────────────────▼──────────────────────────┐
   │ Retrieval  (two-tower + ANN, many parallel sources)  │  ~10^3–10^4 out   (ch 4.1, 4.5)
   └──────────────────────────┬──────────────────────────┘
   ┌──────────────────────────▼──────────────────────────┐
   │ Pre-ranking (light model, often also two-tower)      │  ~10^2–10^3 out   (ch 4.1)
   └──────────────────────────┬──────────────────────────┘
   ┌──────────────────────────▼──────────────────────────┐
   │ Ranking (heavy multi-task CTR/CVR/watch-time model)  │  scores each      (ch 4.2, 4.3, 4.4)
   └──────────────────────────┬──────────────────────────┘
   ┌──────────────────────────▼──────────────────────────┐
   │ Re-ranking / auction (diversity, policy, ads eCPM)   │  ~10 shown        (ch 4.8)
   └──────────────────────────┬──────────────────────────┘
                              ▼
                        user's screen ──► clicks, watches, skips ──► training logs (loop back)
```

Each box is being rethought with ideas from LLMs. The map below is the backbone of the whole part:

| Classic stack | Generative / LLM stack | Where |
|---|---|---|
| Item = random **ID embedding** in huge sparse tables | Item = **semantic ID**: a short sequence of discrete tokens derived from content | 4.5 |
| Feature-crossing DNNs (FM, DCN, DIN) that stop improving when made bigger | **Sequence Transformers** over the user's action history (HSTU) and GPU-friendly backbones (RankMixer, Wukong) that follow **scaling laws** | 4.3, 4.4 |
| Separate retrieval → ranking → re-ranking models | **One generative model** that decodes the recommendation list directly (OneRec) | 4.6 |
| Pointwise CTR loss on logged clicks | Next-action prediction + **preference alignment** (reward models, DPO, RL) | 4.6 |
| Hand-engineered text/image features | **LLM / multimodal embeddings**, LLM-generated knowledge, LLM reasoning about intent | 4.7 |
| Hand-tuned auction + RL bidders | **Generative auto-bidding** (diffusion) and end-to-end generative advertising | 4.8 |

Four differences from LLMs are worth memorizing before you start, because they explain most design choices you will see:

1. **Vocabulary.** An LLM has ~10^5 tokens that never change; a recommender has 10^8–10^10 items, and millions are created and retired every day.
2. **Data.** Training data is a non-stationary *stream*. Models train online, usually see each example once, and degrade within days if not refreshed.
3. **Latency and fan-out.** A request must be answered in tens of milliseconds, and the ranking model scores hundreds to thousands of candidates per request at very high QPS.
4. **Money.** In ads, a predicted probability is multiplied by a bid and decides a price. Predictions must be **calibrated**, not just well-ordered.

## How to study this part

1. Read **4.1 → 4.2 → 4.3** in order; they are the "classic" foundation that every later chapter assumes. Do the code exercises: a two-tower model with in-batch negatives, an FM layer, a DIN attention unit, and SASRec.
2. Read **4.4** (scaling) and **4.5** (semantic IDs) next; they are independent of each other and both feed into 4.6.
3. Read **4.6** after reviewing DPO and RL from [Part 2](../part2/index.md).
4. **4.7** (LLMs in recsys) can be read any time after 4.3.
5. Read **4.8** (ads) last; it reuses calibration from 4.2 and generative modeling from 4.6.

Suggested hands-on projects (pick at least two):

- [ ] Train SASRec on MovieLens-1M or Amazon Reviews; then train a GPT-style next-item model and confirm it is the same idea (4.3).
- [ ] Build semantic IDs: embed item text with a sentence encoder, train an RQ-VAE (3 levels × 256 codes), and check that similar items share prefixes (4.5).
- [ ] Implement TIGER-style generative retrieval with trie-constrained beam search (start from [GRID](https://github.com/snap-research/GRID)); compare recall with a two-tower baseline, especially on cold-start items (4.5).
- [ ] Run Meta's open [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) (HSTU) repo and plot quality vs model size (4.4).
- [ ] KAR-style experiment: have an LLM write user-interest summaries offline, embed them, and add them as features to a DIN or DCN-V2 CTR model; measure AUC lift (4.7).
- [ ] Align a small generative recommender with DPO using click-vs-skip preference pairs (4.6).
- [ ] Simulate a GSP auction with a miscalibrated pCTR model and measure the revenue and advertiser-cost distortion (4.8).
