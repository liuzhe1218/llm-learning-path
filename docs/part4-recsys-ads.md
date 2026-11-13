# Part 4 · LLMs & Generative Modeling in Ads Ranking and Recommendation

> **Goal:** understand how industrial recommendation and ads systems work, and how the ideas from Parts 1–3 (Transformers, scaling laws, next-token prediction, RL alignment, efficient serving) are reshaping them into **generative recommenders** and **large recommendation models**.

**Legend:** :material-star: must-read · :material-book-open-variant: deep-dive · :material-new-box: 2025–2026 frontier

---

## 4.1 The big picture

A classic industrial recommender / ads system is a **multi-stage cascade**:

```text
billions of items ─► Retrieval (two-tower ANN, ~10^3–10^4) ─► Pre-ranking (~10^2–10^3)
                 ─► Ranking (heavy CTR/CVR model, multi-task) ─► Re-ranking (diversity, ads auction, policy) ─► feed
```

| Classic stack | What LLM / generative ideas change |
|---|---|
| Item = random **ID embedding** (huge sparse tables) | Item = **semantic ID** (tokens from content embeddings) — shareable, cold-start friendly |
| Feature-crossing DNNs (DCN, DIN) that don't scale well | Sequence **Transformers over user action history** that follow **scaling laws** |
| Separate retrieval → ranking models | **One generative model** that decodes the recommendation list directly (end-to-end) |
| Pointwise CTR loss | Next-action prediction + **preference alignment** (DPO/RL with reward models) |
| Text/image features hand-engineered | **LLM / multimodal embeddings** for items and users; LLM reasoning about interests |

Key differences from LLMs to keep in mind: vocabularies are *billions* of items that churn daily; data is a non-stationary stream (models train online, often one epoch); latency budgets are tens of ms per request with hundreds of candidates; and ads add **calibration** (predicted CTR × bid drives the auction) and advertiser-side objectives.

---

## 4.2 Foundations: deep recommendation & ads CTR models

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | FTRL-Proximal online logistic regression and the engineering lessons of large-scale ads CTR — calibration, memory savings, monitoring. |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + logistic regression; data freshness matters most; normalized entropy as the metric. |
| :material-star: [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — Covington et al., RecSys 2016 | The canonical two-stage (candidate generation + ranking) deep recommender; frames retrieval as extreme multiclass classification over videos. |
| :material-star: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — Cheng et al., 2016 | Linear "memorization" + DNN "generalization" jointly trained; Google Play. |
| :material-book-open-variant: [Deep & Cross Network (DCN)](https://arxiv.org/abs/1708.05123) / [DCN V2](https://arxiv.org/abs/2008.13535) — Wang et al., 2017/2020 | Explicit, efficient feature-crossing layers; V2 is a strong production baseline. |
| :material-star: [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978) — Zhou et al. (Alibaba), 2017 | **Target attention**: weight the user's past behaviors by relevance to the candidate ad. The start of user-behavior-sequence modeling in ranking. |
| :material-star: [DLRM: Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) — Naumov et al. (Meta), 2019 | Reference architecture (sparse embeddings + dense MLP + dot interactions) and the systems view: embedding tables dominate memory, requiring model parallelism. |
| :material-book-open-variant: [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) — Kang & McAuley, 2018 | Causal Transformer over item sequences for next-item prediction — essentially a GPT over item IDs. |
| :material-book-open-variant: [BERT4Rec](https://arxiv.org/abs/1904.06690) — Sun et al., 2019 | Bidirectional masked-item modeling for sequential recommendation. |
| :material-book-open-variant: [Towards Understanding the Overfitting Phenomenon of Deep CTR Models](https://arxiv.org/abs/2209.06053) — Zhang et al. (Alibaba), 2022 | Explains the "**one-epoch**" phenomenon: CTR models overfit sharply at the start of the 2nd epoch, driven by the embedding layer. |
| :material-book-open-variant: [On the Embedding Collapse When Scaling up Recommendation Models](https://arxiv.org/abs/2310.04400) — Guo et al., 2023 | Why simply enlarging embeddings stops helping: embeddings collapse to a low-dimensional subspace; proposes multi-embedding. |

---

## 4.3 Scaling laws & large recommendation models :material-new-box:

The central 2024–2026 question: *can recommenders scale like LLMs?* Traditional DLRMs mostly scale **embedding tables** (memory), not **compute**, and plateau. The answer emerging: yes, if the *dense* part is a well-designed Transformer-like network over long user sequences.

| Paper | Summary |
|---|---|
| :material-star: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152) — Zhai et al. (Meta), ICML 2024 | Reformulates ranking and retrieval as **sequential transduction** over a user's action stream (**generative recommenders, GR**). The HSTU block (pointwise aggregated attention, no softmax) handles non-stationary vocabularies; quality follows a **power-law with training compute** up to GPT-3/LLaMA-2 scale; +12.4% in online A/B tests. The paper that started the "large recommendation model" wave. |
| :material-star: [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) — Zhang et al. (Meta), 2024 | Stacked factorization machines that capture higher-order interactions and keep scaling across two orders of magnitude of compute, where DLRM-style baselines plateau. |
| :material-book-open-variant: [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351) — Zhang et al., 2023 | Empirical scaling laws for ID-based sequential recommenders (up to 0.8B params), including data-constrained regimes. |
| :material-book-open-variant: [Scaling New Frontiers: Insights into Large Recommendation Models](https://arxiv.org/abs/2412.00714) — Guo et al. (Tencent), 2024 | Systematic study of HSTU scaling: which components matter, and transfer to ranking. |
| :material-new-box: [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) — Han et al., 2025 | Brings HSTU-style GR into a production ranking model *while keeping* hand-crafted cross features; group-layer-norm and training/inference optimizations; ~65× FLOPs per sample at similar cost. |
| :material-new-box: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) — Zhu et al. (ByteDance), 2025 | Hardware-aware ranking backbone (token mixing + per-token FFN / sparse-MoE) that raises model FLOPs utilization from 4.5% to 45% and scales ranking-model parameters 100× at roughly the same inference latency; a 1B dense RankMixer serves full traffic in recommendation and ads. |
| :material-new-box: [Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model (LUM)](https://arxiv.org/abs/2502.08309) — Yan et al. (Alibaba), 2025 | Pre-train a generative large user model → query it for user-interest knowledge → feed into the traditional ranking model; scaling up to 7B params. |
| :material-new-box: [MARM: Memory Augmentation and Scalable Complexity](https://arxiv.org/abs/2411.09425) — Lv et al. (Kuaishou), 2024 | Caches intermediate attention results to make complexity scaling affordable in the latency budget — a "KV cache" idea for recsys. |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | Practical large-ranker recipe: Residual DCN, Transformer over history, dense gating, calibration, quantization, explore/exploit. |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | Scales retrieval beyond two-tower dot products with hierarchical clustering and richer interaction modules, deployed for ads. |

Useful industry read: Meta's [Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/).

---

## 4.4 Semantic IDs & generative retrieval

Instead of predicting over billions of opaque item IDs, **tokenize** each item into a short sequence of discrete codes derived from its content embedding (usually with **RQ-VAE** residual quantization). Similar items share prefixes, so a sequence model can **autoregressively decode** the next item's code, like an LLM decoding words.

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991) — Tay et al., 2022 | A seq2seq model memorizes a corpus and decodes document IDs directly — the origin of generative retrieval. |
| :material-book-open-variant: [Autoregressive Image Generation using Residual Quantization (RQ-VAE)](https://arxiv.org/abs/2203.01941) — Lee et al., 2022 | Residual quantization: each level quantizes the remaining error, giving coarse-to-fine codes — the standard semantic-ID tokenizer. |
| :material-star: [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) — Rajput et al. (Google), NeurIPS 2023 | Semantic IDs via RQ-VAE over item text embeddings + a Transformer that decodes the next item's semantic ID. Better cold-start and diversity than ID-based retrieval. **The** generative retrieval reference. |
| :material-star: [Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) — Singh et al. (Google/YouTube), 2023 | Replaces random item IDs in a production YouTube **ranking** model with semantic-ID n-gram/sentencepiece embeddings — better generalization to new/long-tail items without losing memorization. |
| :material-book-open-variant: [Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER)](https://arxiv.org/abs/2411.18814) — Yang et al. (Meta), 2024 | Analyzes when generative retrieval falls short of dense retrieval (cold-start, coverage) and combines them. |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | Generates a sparse semantic ID, then a dense vector conditioned on it — coarse-to-fine, deployed in Baidu ads. |
| :material-new-box: [Generative Recommendation with Semantic IDs: A Practitioner's Handbook (GRID)](https://arxiv.org/abs/2507.22224) — Ju et al. (Snap), 2025 | Open-source framework and careful ablations of tokenizer, architecture and decoding choices for semantic-ID recommenders — the best hands-on starting point. |

---

## 4.5 End-to-end generative recommenders :material-new-box:

2025's big step: replace the whole retrieve → rank cascade with **one generative model** that decodes the recommendation list, then **align** it with preference/RL methods borrowed from LLM post-training (Part 2).

| Paper | Summary |
|---|---|
| :material-star: [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) — Deng et al. (Kuaishou), 2025 | Encoder–decoder with sparse MoE generates a *session* of videos from semantic IDs; a reward model + **iterative DPO** aligns outputs with user preferences. Deployed on Kuaishou's main feed. |
| :material-star: [OneRec Technical Report](https://arxiv.org/abs/2506.13695) — Kuaishou, 2025 | Full end-to-end system replacing the cascade: 10× more compute, 23.7% / 28.8% MFU in training / inference, **RL with a multi-objective reward**, and lower serving cost (OPEX 10.6% of the cascade), serving 25% of Kuaishou's QPS. Read for the systems and RL details. |
| :material-new-box: [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900) — Kuaishou, 2025 | Lazy decoder-only architecture (drops the expensive encoder) for much better compute efficiency and scaling to 8B; RL using real user-feedback signals. |
| :material-new-box: [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639) — Kuaishou, 2025 | Aligns item semantic IDs with an LLM's text space so the model can **reason in natural language** before recommending — reasoning models meet recsys. |
| :material-new-box: [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784) — Google/YouTube, 2025 | Semantic-ID tokenization → continued pre-training of a Gemini-family LLM on user/item data → task-specific fine-tuning for generative retrieval on YouTube. |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | End-to-end generative **advertising**: jointly generates ad sequences and handles auction/allocation and payment, rather than a multi-stage pipeline. |
| :material-new-box: [Think Before Recommend: Latent Reasoning for Sequential Recommendation (ReaRec)](https://arxiv.org/abs/2503.22675) — Tang et al., 2025 | Test-time "thinking" via extra latent reasoning steps over the user representation — test-time compute for recsys. |

---

## 4.6 LLMs as recommenders, feature providers & user models

Three ways LLMs enter the stack: **(a)** the LLM *is* the recommender (prompting / fine-tuning), **(b)** the LLM *produces features/embeddings/knowledge* for a conventional model, **(c)** the LLM *models the user* from long histories.

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366) — Geng et al., 2022 | Casts rating, sequential rec, explanation, etc. as text-to-text tasks for one T5 model. |
| :material-book-open-variant: [Is ChatGPT a Good Recommender?](https://arxiv.org/abs/2304.10149) — Liu et al., 2023 | Zero/few-shot LLM recommendation: good at explanation and cold-start, weak at accuracy vs trained models without collaborative signal. |
| :material-star: [TALLRec](https://arxiv.org/abs/2305.00447) — Bao et al., 2023 | LoRA-tuning an LLM on a small amount of recommendation data makes it a strong few-shot recommender. |
| :material-book-open-variant: [Adapting LLMs by Integrating Collaborative Semantics (LC-Rec)](https://arxiv.org/abs/2311.09049) — Zheng et al., 2023 | Adds semantic-ID tokens to an LLM's vocabulary and aligns them with language via multiple tuning tasks. |
| :material-book-open-variant: [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841) — Li et al. (Huawei), 2023 | Contrastively aligns a text LM and a collaborative CTR model, then serves only the CTR model — LLM knowledge at zero extra latency. |
| :material-book-open-variant: [ClickPrompt](https://arxiv.org/abs/2310.09234) — Lin et al., 2023 | CTR model generates soft prompts for an LM, combining collaborative and semantic signals. |
| :material-star: [Towards Open-World Recommendation with Knowledge Augmentation from LLMs (KAR)](https://arxiv.org/abs/2306.10933) — Xi et al. (Huawei), 2023 | LLM reasons about user preferences and item facts **offline**; encoded knowledge vectors become features for any CTR model. Deployed in production — a pragmatic pattern. |
| :material-star: [HLLM: Hierarchical LLMs for Item and User Modeling](https://arxiv.org/abs/2409.12740) — Chen et al. (ByteDance), 2024 | Item LLM encodes item text → embedding; User LLM models the embedding sequence. Pre-trained LLM weights help and the approach scales to 7B. |
| :material-book-open-variant: [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) — Ning et al. (Google), 2024 | Compresses long user histories into embeddings that the LLM attends to via cross-attention / soft prompts. |
| :material-new-box: [360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://arxiv.org/abs/2501.16450) — LinkedIn, 2025 | One 150B decoder LLM with text-only interface replaces many task-specific ranking models across LinkedIn surfaces. |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | LLM as a multi-query, multi-objective retriever over a huge item set, deployed in advertising. |
| :material-new-box: [RecGPT Technical Report](https://arxiv.org/abs/2507.22879) — Alibaba (Taobao), 2025 | LLM-driven user-intent mining → item tag prediction → retrieval → explanation on Taobao's homepage, trained via a human–LLM cooperative judge loop. |
| :material-new-box: [QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739) — Luo et al., 2024 | Aligns multimodal LLM item representations with business signals and quantizes them into codes usable as ranking features. |

---

## 4.7 Ads-specific: user modeling, auctions & generative bidding

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | **SUM**: a few large upstream user models produce embeddings consumed by hundreds of downstream ads models — the "foundation user model" pattern; covers online embedding freshness. |
| :material-book-open-variant: [TransAct](https://arxiv.org/abs/2306.00248) / [PinnerFormer](https://arxiv.org/abs/2205.04507) — Pinterest, 2022–2023 | Realtime Transformer over recent actions (TransAct) combined with long-term batch user embeddings (PinnerFormer) — the hybrid realtime + batch pattern in ads/homefeed ranking. |
| :material-new-box: [AIGB: Generative Auto-bidding via Conditional Diffusion Modeling](https://arxiv.org/abs/2405.16141) — Guo et al. (Alibaba), KDD 2024 | Treats auto-bidding as generating a bid *trajectory* with a diffusion model conditioned on the target return, replacing RL bidders. |
| :material-new-box: [EGA-V2](https://arxiv.org/abs/2505.17549) — Meituan, 2025 | (See 4.5) Generative modeling of the full ad allocation + payment process. |

Other ads themes to know: **calibration** (predicted CTR must be a true probability because it multiplies the bid), **delayed conversion feedback**, **privacy-constrained signals** (fewer cross-app identifiers), and **generative ad creative** (LLM/diffusion-generated text and images, now offered by major ad platforms).

---

## 4.8 Diffusion & other generative models in recsys

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Diffusion Recommender Model (DiffRec)](https://arxiv.org/abs/2304.04971) — Wang et al., SIGIR 2023 | Denoises corrupted user–item interaction vectors to generate recommendations; latent and temporal variants for scale. |

---

## 4.9 Surveys

| Paper | Summary |
|---|---|
| :material-star: [A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860) — Wu et al., 2023 | Taxonomy of discriminative vs generative LLM-for-rec approaches. |
| :material-book-open-variant: [How Can Recommender Systems Benefit from LLMs: A Survey](https://arxiv.org/abs/2306.05817) — Lin et al., 2023 | Organizes the field by *where* (which stage) and *how* (tune or not, infer with CRM or not) LLMs are used. |
| :material-book-open-variant: [LLMs for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) — Li et al., 2023 | Focuses on generative paradigms and item ID design. |
| :material-book-open-variant: [From Matching to Generation: A Survey on Generative Information Retrieval](https://arxiv.org/abs/2404.14851) — Li et al., 2024 | Generative retrieval and generation-augmented retrieval, shared with search. |

---

## 4.10 Hands-on projects

- [ ] Train SASRec on MovieLens-1M / Amazon Reviews; then train a **GPT-style** next-item model and compare (it's the same idea).
- [ ] Build semantic IDs: embed item text with a sentence encoder → train an RQ-VAE (3 levels × 256 codes) → check that similar items share prefixes.
- [ ] Implement TIGER-style generative retrieval with beam search over semantic IDs (start from [GRID](https://github.com/snap-research/GRID)); compare recall vs a two-tower baseline, especially on cold-start items.
- [ ] Try Meta's open [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) (HSTU) repo; plot quality vs model size.
- [ ] KAR-style experiment: use an LLM offline to write user-interest summaries, embed them, add as features to a DIN/DCN-V2 CTR model; measure AUC lift.
- [ ] Align a generative recommender with DPO using click vs skip as preference pairs.

## 4.11 Self-check

??? question "Why didn't classic DLRMs follow LLM-like scaling laws?"
    Most of their parameters live in sparse embedding tables (memory, not compute), interaction layers are shallow, and embeddings collapse when enlarged. Scaling *compute* in the dense network — e.g. Transformers over long action sequences (HSTU) or efficient token-mixing backbones (RankMixer, Wukong) — is what unlocked scaling.

??? question "What problem do semantic IDs solve, and what do they cost?"
    They give items a compositional, content-based vocabulary: new items get meaningful IDs immediately, parameters are shared across similar items, and generative decoding over a small token vocabulary becomes possible. The costs are collisions/loss of fine-grained memorization for popular items and a tokenizer that must be refreshed as content shifts.

??? question "How is OneRec's training pipeline analogous to LLM post-training?"
    Pre-train a generative model on next-item prediction over logs (≈ pre-training), then optimize with a reward model and preference/RL methods (iterative DPO, RL with multi-objective rewards) to match what users and the business value (≈ RLHF).

??? question "Why is calibration crucial in ads but less so in organic recommendation?"
    In ads, eCPM = bid × pCTR (× pCVR) determines auction ranking and the price charged; a miscalibrated pCTR mis-ranks ads and over/under-charges advertisers. Organic feeds mostly need correct *ordering*.

??? question "Why is it hard to put a 7B LLM directly in a ranking path?"
    Ranking scores hundreds–thousands of candidates per request within tens of milliseconds at very high QPS. Common workarounds: precompute LLM item/user embeddings offline (KAR, HLLM item side), distill into small models (CTRL), cache user-side computation (MARM), or use a generative model that emits the final list once per request (OneRec).
