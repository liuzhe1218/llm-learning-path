# 4.6 End-to-End Generative Recommenders

!!! abstract "What you'll learn"
    - Why replacing the retrieve → rank cascade with one generative model is attractive, and what it costs
    - OneRec's architecture: user-history encoder, MoE decoder over semantic IDs, session-wise list generation
    - How reward models and iterative DPO align a recommender — the same post-training ideas as [Part 2](../part2/index.md) — with runnable code
    - What the OneRec technical report and OneRec-V2 add: RL with multi-objective rewards, high MFU, lower OPEX, lazy decoder-only design
    - Reasoning in recommendation (OneRec-Think, ReaRec), adapting pre-trained LLMs (PLUM), and the open problems

## Why this matters / the problem

The cascade of Chapter [4.1](01-recsys-fundamentals.md) works, but it has structural weaknesses:

- **Inconsistent objectives.** Retrieval, pre-ranking, ranking and re-ranking are trained separately on different data and losses. An item the ranker would love can be dropped by a retriever that never knew.
- **Computational fragmentation.** Dozens of models, feature pipelines and services, each small and with low hardware utilization. The OneRec report argues this fragmentation keeps recommenders from benefiting from AI-community breakthroughs.
- **Hand-made list assembly.** Rankers score items one at a time; the final list (diversity, ordering, mix) is assembled by rules.

LLMs suggest another way: one large model that reads the context and **generates** the output directly, then gets **aligned** to what people want with a reward model and RL. Since 2025, Kuaishou's OneRec family has shown this can replace the cascade in a large production feed. This chapter assembles the pieces from 4.3 (sequence models), 4.4 (scaling) and 4.5 (semantic IDs) into such a system.

## Intuition

Ask a skilled human curator for "ten videos for this person right now". She does not score a billion videos one by one; she reads the person's history and *writes down a list*, thinking about the whole list at once (not ten cooking videos in a row). Then, over time, she learns from feedback which lists people enjoyed.

An end-to-end generative recommender does the same:

1. **Read**: encode the user's features and behaviour history.
2. **Write**: autoregressively decode a list of items, each spelled as a semantic ID (Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)).
3. **Learn from feedback**: pre-train on logged good sessions (like LM pre-training/SFT), then improve with a reward model and preference optimization (like RLHF).

## How it works

### OneRec architecture (V1)

**Item tokenization.** Every video is represented by a semantic ID of a few codes, obtained by residual quantization of a multimodal content embedding (the OneRec papers use balanced k-means-based residual quantization so that codes are used evenly). Output vocabulary: a few thousand code tokens instead of hundreds of millions of video IDs.

**Encoder.** A Transformer encoder reads the user's behaviour sequence (and user features), producing context vectors $\mathbf{H} \in \mathbb{R}^{n\times d}$ for $n$ history positions.

**Decoder with MoE.** A Transformer decoder generates the semantic-ID tokens of the recommended videos one at a time, attending to $\mathbf{H}$ by cross-attention. Its feed-forward layers are **sparse Mixture-of-Experts**: each token is routed to a few of many expert FFNs, so total parameters grow without proportional growth in FLOPs (the same trick as MoE LLMs in [Part 1](../part1/index.md)).

**Session-wise generation.** Instead of predicting one next item, OneRec generates a whole **session**: a short list of videos for one request. The training targets are high-quality historical sessions (sessions where the user engaged well). Generating the list jointly lets the model account for context *within* the list — what came earlier — instead of relying on hand-crafted rules to combine independently scored items.

Formally, for a user context $u$ and a target session $S = (v_1,\dots,v_m)$ where item $v_j$ has semantic ID $(c_{j,1},\dots,c_{j,D})$, the pre-training loss is ordinary next-token cross-entropy over the flattened code sequence:

$$
\mathcal{L}_{\text{NTP}} = -\sum_{j=1}^{m}\sum_{l=1}^{D}\log \pi_\theta\big(c_{j,l}\mid u,\ c_{<j},\ c_{j,<l}\big),
$$

where $\pi_\theta$ is the model's next-token distribution. In words: teach the model to write out good sessions, code by code.

```text
 user features + behaviour sequence                      semantic IDs of a recommended session
 ─────────────────────────────────►  ENCODER  ──H──►  DECODER (MoE FFNs, cross-attn to H)
                                   (Transformer)        BOS c1,1 c1,2 c1,3 | c2,1 c2,2 c2,3 | ...
                                                         item 1           item 2
                                         beam search + trie of valid IDs ──► list of videos
```

### Preference alignment with a reward model and iterative DPO

Next-token training only imitates logged sessions, which were produced by the old cascade. To *improve* on them, OneRec borrows **Direct Preference Optimization** from LLM post-training (see [Part 2](../part2/index.md)). DPO needs pairs (chosen, rejected) of outputs for the same input. The recsys twist: each request is shown only *one* list, so you never observe the user's reaction to two alternatives.

OneRec's fix is a **reward model** $R(u, S)$ trained on logs to predict how much the user would enjoy session $S$ — a simulator of user feedback. **Iterative Preference Alignment** then repeats:

1. For a user context $u$, sample $N$ candidate sessions from the current model (e.g. with beam search).
2. Score them with $R$; take the highest-scoring as $S_w$ (chosen) and the lowest as $S_l$ (rejected).
3. Update the model with the DPO loss against a frozen reference model $\pi_{\text{ref}}$ (the model before this round):

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\Big[\log\frac{\pi_\theta(S_w\mid u)}{\pi_{\text{ref}}(S_w\mid u)} - \log\frac{\pi_\theta(S_l\mid u)}{\pi_{\text{ref}}(S_l\mid u)}\Big]\Big),
$$

where $\sigma$ is the logistic function, $\beta>0$ controls how far the policy may move from the reference, and $\log\pi(S\mid u)$ is the sum of token log-probabilities of the session. In words: raise the probability of the better list relative to the worse one, measured relative to where the model started.

4. Use the updated model as the new sampler and reference; repeat.

DPO training is mixed with the ordinary next-token loss so the model does not drift. The OneRec paper reports that a limited number of DPO samples is enough to align preferences and significantly improve generated results, and that deploying OneRec in Kuaishou's main scene increased watch time by 1.6%.

### The OneRec technical report: RL, MFU and cost

The follow-up technical report describes a full production system that replaces the cascade for a share of traffic. Its main claims (from the abstract):

- **Compute.** FLOPs of the recommendation model were increased **10×**, and scaling laws were identified within certain boundaries.
- **RL works here.** With the whole list generated by one policy, reinforcement learning (which was hard to apply to a cascade) becomes natural. The report optimizes the policy against a reward system that fuses multiple objectives (user engagement signals and other requirements) — conceptually the GRPO-style setup of [Part 2](../part2/index.md): sample several lists, score them, push up those with above-average reward.
- **Hardware efficiency.** Infrastructure optimizations give **23.7% and 28.8% MFU** in training and inference — close to LLM practice, versus single digits for many classic rankers (Chapter [4.4](04-scaling-recommendation-models.md)).
- **Cost.** Operating expense is only **10.6%** of the traditional pipeline, because one model replaces many services, with less communication and storage.
- **Impact.** Deployed in Kuaishou / Kuaishou Lite, handling **25% of total QPS**, improving App Stay Time by 0.54% and 1.24% respectively, with gains on 7-day lifetime.

### OneRec-V2: lazy decoder-only

OneRec-V2 found that in V1 **97.66% of compute went into encoding the context** rather than generating recommendations. V2's **lazy decoder-only** architecture removes the separate encoder: context information is consumed by the decoder layers without a full, expensive encoding pass, so compute concentrates on the generated tokens. The report states this cuts total computation by **94%** and training resources by **90%**, enabling scaling to **8B** parameters. V2 also aligns with **real user interactions** instead of relying only on a reward model, using duration-aware reward shaping and adaptive ratio clipping, and reports App Stay Time gains of 0.467% / 0.741% in A/B tests while balancing multiple objectives.

The lesson mirrors LLM history: once the paradigm works, efficiency engineering (decoder-only, where compute goes, MFU) unlocks the next scale step.

### Reasoning in recommendation

Reasoning models (test-time compute, chains of thought) are entering recsys in two forms:

- **Explicit, in-text reasoning — OneRec-Think.** Align item semantic IDs with an LLM's text space (*itemic alignment*), activate reasoning in the recommendation context (*reasoning scaffolding*), and then strengthen it with a recommendation-specific reward that respects the fact that many different items can be valid answers. The model can reason in natural language before emitting item tokens. A "Think-Ahead" architecture makes deployment feasible; the paper reports a 0.159% App Stay Time gain on Kuaishou.
- **Latent reasoning — ReaRec.** Instead of words, feed the sequence model's last hidden state back into itself for several extra steps (with special reasoning position embeddings) before scoring items — implicit multi-step "thinking" in representation space. It reports raising the performance ceiling of several sequential-recommendation backbones by roughly 30–50% in post-hoc analyses on public datasets.

### Starting from a pre-trained LLM: PLUM

Instead of training a recommender from scratch, Google/YouTube's **PLUM** adapts a pre-trained LLM: (1) tokenize items as semantic IDs; (2) **continued pre-training** of the LLM on domain data mixing user behaviour (as semantic-ID sequences) and text; (3) task-specific fine-tuning, focusing on generative retrieval — the model generates semantic IDs of recommended items from user context. PLUM reports substantial retrieval improvements over a heavily optimized production model built on large embedding tables, plus a scaling study and lessons about CPT. Chapter [4.7](07-llms-in-recommendation.md) covers other ways LLMs enter the stack.

??? note "Other generative paradigms: diffusion"
    Autoregressive decoding is not the only generative option. **DiffRec** treats a user's interaction vector as data to be generated: corrupt it with noise and train a model to denoise it, then recommend items with high values in the denoised vector. Latent and temporal variants make it practical at larger scale. Diffusion reappears in ads as a generator of bid trajectories (Chapter [4.8](08-ads-systems.md)).

### The LLM-pipeline analogy

| LLM (Parts 1–2) | End-to-end generative recommender |
|---|---|
| Tokenizer (BPE) | Item tokenizer (residual quantization → semantic IDs) |
| Pre-training on web text | Next-token training on logged high-quality sessions |
| Prompt | User features and behaviour history |
| Response | A session/list of item semantic IDs |
| Reward model from human preference labels | Reward model predicting user engagement (or real feedback, V2) |
| DPO / RLHF / GRPO | Iterative DPO (OneRec), GRPO-style RL with multi-objective rewards (tech report, V2) |
| Constrained decoding for formats | Trie-constrained decoding to valid, eligible items |
| KV cache, batching, MFU | The same serving stack (Part 3) |

## Code

A toy end-to-end loop: a tiny decoder generates 2-code "semantic IDs" for users, a fixed reward model prefers certain codes per user type, and iterative DPO with best/worst-of-$N$ sampling increases the expected reward.

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy
torch.manual_seed(0)
K, LEVELS, N_USERS, N_SAMPLES, BETA = 8, 2, 4, 8, 0.5

class TinyGenRec(nn.Module):
    """Generates a semantic ID (c1, c2) conditioned on a user id."""
    def __init__(self, d=32):
        super().__init__()
        self.user = nn.Embedding(N_USERS, d)
        self.code = nn.Embedding(K + 1, d)                    # index K = BOS
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, K)
    def logprob(self, users, ids):                            # ids: [B, LEVELS] -> [B]
        bos = torch.full((len(users), 1), K)
        inp = self.code(torch.cat([bos, ids[:, :-1]], 1))     # teacher forcing
        h, _ = self.rnn(inp, self.user(users)[None])
        lp = F.log_softmax(self.out(h), -1)                   # [B, LEVELS, K]
        return lp.gather(2, ids[..., None]).squeeze(-1).sum(1)
    @torch.no_grad()
    def sample(self, users):
        ids, h = [], self.user(users)[None]
        tok = torch.full((len(users),), K)
        for _ in range(LEVELS):
            o, h = self.rnn(self.code(tok)[:, None], h)
            tok = torch.multinomial(F.softmax(self.out(o[:, 0]), -1), 1).squeeze(1)
            ids.append(tok)
        return torch.stack(ids, 1)

def reward(users, ids):                                       # stand-in reward model
    return (ids[:, 0] == users % K).float() + 0.5 * (ids[:, 1] == (users * 3) % K).float()

policy = TinyGenRec()
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
users = torch.arange(N_USERS)
for it in range(6):                                           # alignment rounds
    ref = copy.deepcopy(policy).eval()                        # frozen reference for this round
    with torch.no_grad():
        samples = torch.stack([policy.sample(users) for _ in range(N_SAMPLES)], 1)  # [U, N, L]
        r = torch.stack([reward(users, samples[:, n]) for n in range(N_SAMPLES)], 1)
        print(f"round {it}: mean reward of samples {r.mean():.2f}")
        win, lose = samples[users, r.argmax(1)], samples[users, r.argmin(1)]
        ref_w, ref_l = ref.logprob(users, win), ref.logprob(users, lose)
    for _ in range(20):                                       # DPO updates on these pairs
        margin = (policy.logprob(users, win) - ref_w) - (policy.logprob(users, lose) - ref_l)
        loss = -F.logsigmoid(BETA * margin).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

Mean reward rises from about chance level (≈0.19) to ≈1.0 within a few rounds, even though no logged "correct answer" was ever given — only a reward model's comparisons. It then **stalls below the maximum of 1.5**: the policy has become confident about the first code but not the second, so all 8 samples look alike, the best and worst samples get the same reward, the DPO margin carries no signal, and learning stops. That is a miniature version of a real problem — alignment needs diverse samples (temperature, larger $N$, exploration) to keep producing informative pairs. Real systems also add the next-token loss on logged sessions and safeguards against reward hacking.

Group-normalized advantages and a policy-gradient loss in the GRPO style (compare with [Part 2](../part2/index.md)); `logp` would be the summed token log-probabilities of each sampled list under the current policy:

```python
import torch
rewards = torch.tensor([[0.8, 0.5, 0.2, 0.5]])        # [requests, samples per request]
adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, unbiased=False, keepdim=True) + 1e-6)
print(adv)                                              # tensor([[ 1.4142,  0.0000, -1.4142,  0.0000]])
logp = torch.tensor([[-12.0, -11.5, -10.0, -13.0]], requires_grad=True)
loss = -(adv.detach() * logp).mean()                    # raise log-prob of above-average lists
loss.backward()
print(logp.grad)                                        # negative grad for list 1, positive for list 3
```

## Worked example

**A DPO loss by hand.** $\beta = 0.1$. For a user, the chosen session has $\log\pi_\theta(S_w) = -10.0$ and $\log\pi_{\text{ref}}(S_w) = -11.0$; the rejected one has $\log\pi_\theta(S_l) = -9.0$ and $\log\pi_{\text{ref}}(S_l) = -9.5$.

- Log-ratio of chosen: $-10.0-(-11.0) = 1.0$. Log-ratio of rejected: $-9.0-(-9.5) = 0.5$.
- Margin $= \beta(1.0-0.5) = 0.05$.
- Loss $= -\log\sigma(0.05) = \log(1+e^{-0.05}) \approx 0.668$.

The gradient pushes $\log\pi_\theta(S_w)$ up and $\log\pi_\theta(S_l)$ down. Note that the rejected session is still *more* likely than the chosen one under the policy ($-9.0 > -10.0$); DPO cares about the change relative to the reference, not raw probabilities.

**Group-normalized advantages (GRPO-style RL).** Four sampled lists for one request get rewards $[0.8, 0.5, 0.2, 0.5]$. Mean $= 0.5$; deviations $[0.3, 0, -0.3, 0]$; standard deviation $= \sqrt{(0.09+0+0.09+0)/4} \approx 0.212$. Advantages $= [1.41, 0, -1.41, 0]$: the policy gradient raises the log-probability of list 1, lowers list 3, and leaves lists 2 and 4 unchanged.

**Why one model can be cheaper.** In a cascade, a request might hit several retrieval services, a pre-ranker over 5,000 candidates and a ranker over 500, each with its own feature fetching and network hops. A generative model encodes the user once and decodes, say, 10 items × 3 codes = 30 tokens with beam search. Even if each decoding step uses a large model, the work is dense matrix math on one accelerator with high MFU, instead of many memory-bound services — the mechanism behind OneRec's reported OPEX of 10.6% of the cascade.

## In practice (2025–2026)

- **Kuaishou's OneRec family** is the most complete public example of an end-to-end generative recommender in production, now on V2 (decoder-only, 8B) with reasoning variants.
- **Hybrid deployments** are more common than full replacement: generative models as one retrieval source, or as a large model whose outputs feed a classic ranker (LUM, Chapter 4.4).
- **Post-training recipes** follow LLMs: pre-train on logs, then DPO/GRPO-style RL with reward models; increasingly, rewards come from real user feedback (OneRec-V2).
- **Ads** adopt the same idea with extra machinery for bids, allocation and payment (EGA-V2, Chapter [4.8](08-ads-systems.md)).

## Common pitfalls / misconceptions

- **Imitation is not improvement.** Pure next-token training on logs reproduces the old system's biases; alignment with rewards is where gains come from.
- **Reward hacking.** A policy can exploit a learned reward model (e.g. always generating clickbait-like content the RM over-scores). Use KL/reference constraints, held-out online metrics and periodic RM retraining.
- **Ignoring constraints.** A generative model does not natively obey business rules (deduplication, policy filters, creator fairness, freshness). They must be enforced via constrained decoding, rewards or post-filters.
- **Underestimating serving engineering.** Beam search over long contexts needs LLM-style serving (batching, KV caching, quantization; [Part 3](../part3/index.md)).
- **Comparing against a weak cascade.** Gains should be measured against a strong, well-tuned production system with the same compute budget.

## Open problems

- **Business rules and controllability:** how to guarantee hard constraints (legal, safety, contractual) in generated lists.
- **Diversity and exploration:** generative models can collapse onto a narrow mode; list-level rewards and sampling strategies help but are not solved.
- **Ad insertion:** mixing organic content and auctioned ads in one generated list while keeping auction incentives intact.
- **Freshness:** semantic IDs and models must keep up with millions of new items per day without full retraining.
- **Evaluation:** offline metrics for generated lists, and reward models that remain faithful as the policy changes the data distribution.

## Key takeaways

- End-to-end generative recommenders replace the cascade with one model that reads the user and decodes a list of semantic IDs.
- OneRec = encoder + MoE decoder + session-wise generation + reward-model-guided iterative DPO; the tech report adds RL, high MFU and a 10.6% OPEX.
- OneRec-V2 moves to a lazy decoder-only architecture (94% less compute) and real-feedback RL, scaling to 8B.
- LLM post-training ideas (DPO, GRPO-style RL, reasoning) transfer directly; the recsys-specific twist is that you only observe feedback on one list per request.
- Constraints, diversity, ads and freshness remain open.

## Check your understanding

??? question "Why can't you apply DPO directly to logged recommendation data the way it is applied to human preference pairs in LLMs?"
    DPO needs a chosen and a rejected output for the same input, but each request shows the user only one list. OneRec creates pairs by sampling several candidate lists from the model and ranking them with a learned reward model that simulates user feedback.

??? question "What is the advantage of session-wise generation over generating items independently?"
    The model can condition each item on the others in the list, accounting for redundancy, ordering and mix, rather than scoring items independently and relying on hand-crafted rules to assemble the list.

??? question "OneRec-V2 found that 97.66% of compute went to encoding. Why is that a problem, and what did V2 do?"
    Compute spent re-encoding long contexts does not directly improve the generated recommendations and limits how large the generating part can be. V2's lazy decoder-only design removes the heavy encoder, reducing total computation by 94% and allowing scaling to 8B parameters.

??? question "In the worked DPO example, what happens to the loss if the policy's chosen log-ratio rises from 1.0 to 3.0?"
    Margin $= 0.1(3.0-0.5)=0.25$; loss $= \log(1+e^{-0.25}) \approx 0.576$, lower than 0.668, as expected.

??? question "Name two hard problems for deploying end-to-end generative recommenders in an ads-funded product."
    Enforcing business and policy constraints in generated lists, and inserting auctioned ads while preserving auction properties (pricing, incentive compatibility); also freshness, diversity and evaluation.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) — Deng et al. (Kuaishou), 2025 | Encoder–decoder with sparse MoE generates a session of videos; reward model + iterative DPO; deployed on Kuaishou's main feed. |
| :material-star: [OneRec Technical Report](https://arxiv.org/abs/2506.13695) — Kuaishou, 2025 | Full end-to-end system: 10× compute, 23.7% / 28.8% MFU, RL with multi-objective rewards, OPEX 10.6% of the cascade, 25% of QPS. |
| :material-new-box: [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900) — Kuaishou, 2025 | Lazy decoder-only architecture, scaling to 8B; RL with real user-feedback signals. |
| :material-new-box: [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639) — Kuaishou, 2025 | Aligns semantic IDs with an LLM's text space so the model reasons in natural language before recommending. |
| :material-new-box: [Think Before Recommend: Latent Reasoning for Sequential Recommendation (ReaRec)](https://arxiv.org/abs/2503.22675) — Tang et al., 2025 | Test-time latent reasoning steps over the user representation. |
| :material-new-box: [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784) — Google/YouTube, 2025 | Semantic IDs → continued pre-training of an LLM → fine-tuning for generative retrieval on YouTube. |
| :material-book-open-variant: [Diffusion Recommender Model (DiffRec)](https://arxiv.org/abs/2304.04971) — Wang et al., SIGIR 2023 | Denoises corrupted user–item interaction vectors to generate recommendations. |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | The end-to-end generative idea applied to advertising, including allocation and payment (see Chapter 4.8). |
