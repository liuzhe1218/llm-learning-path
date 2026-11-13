# 4.7 LLMs in Recommendation

!!! abstract "What you'll learn"
    - The three roles an LLM can play in a recommender: (a) the recommender itself, (b) an offline feature/knowledge provider, (c) a user/item encoder
    - Concrete prompt templates for LLM-as-recommender (pointwise, listwise) and how outputs are parsed and validated
    - How KAR, CTRL, ClickPrompt and QARM inject LLM knowledge into classic models at (near) zero online cost
    - How HLLM, User-LLM, LC-Rec and URM use LLMs as encoders or retrievers, and how 360Brew and RecGPT push LLMs into the core of production systems
    - Why latency and cost dominate the design, with a worked cost estimate and the standard patterns: offline precompute, caching, distillation

## Why this matters / the problem

LLMs know a lot about the world: that a Nikon Z8 is a camera body that needs lenses, that a user who reads about marathon training may want running shoes, what a movie is about. Classic recommenders know none of this; they only see IDs and co-occurrence. LLMs can also *reason* ("this user bought a crib last month, so baby clothes are likely relevant") and *explain*. They seem like the perfect recommender.

They also have three problems. First, they lack **collaborative signal**: they do not know that people who watched video 8,124,551 also watched video 2,003,117. Second, they are **expensive**: a ranking stage scores hundreds of candidates per request at 10^4–10^5 requests per second, within tens of milliseconds. Third, item catalogues are huge and change daily, while an LLM's vocabulary and knowledge are fixed.

The practical question is therefore not "LLM or classic model?" but **where in the stack** an LLM adds the most value per unit of latency and cost. This chapter organizes the field by role.

## Intuition

Picture three ways to use an expert consultant in a store:

1. **(a) Recommender:** the consultant talks to every customer and picks products. Excellent judgement, but slow and costly per customer, and she does not know which products sold together last week.
2. **(b) Knowledge provider:** the consultant writes notes overnight about every product and every regular customer; the fast in-store system reads those notes as extra features. No delay during the day.
3. **(c) Encoder:** the consultant's way of *understanding* products and customers is distilled into compact representations (embeddings) that the store's system uses.

Production systems mostly use (b) and (c), and increasingly (a) for high-value, low-QPS surfaces or once the model has been made efficient enough.

## How it works

```text
             (b) offline knowledge / features            (c) encoders
   LLM ──► text about users & items ──► vectors ─┐     LLM ──► item / user embeddings ─┐
                                                 ▼                                      ▼
   request ──► retrieval ──► ranking (classic or scaled model) ──► re-ranking ──► feed
                  ▲                       ▲
                  └──── (a) LLM as retriever / recommender (generate items or scores) ────┘
```

### (a) The LLM as recommender

**Text-to-text formulation (P5).** Express every recommendation task — rating prediction, sequential recommendation, explanation, review summarization — as a text prompt with a text answer, and train one T5 model on all of them. Users and items appear as tokens in the prompt.

**Zero/few-shot prompting.** A general chat LLM can be asked directly. The study "Is ChatGPT a Good Recommender?" evaluated ChatGPT with prompts on five recommendation scenarios on Amazon Beauty data, without fine-tuning: it did well on some tasks (and on explanation quality in human evaluation) and reached only baseline level on others. The general lesson from this period is that without collaborative signal, prompting alone rarely beats a trained recommender on accuracy, but it helps for explanation and cold-start.

**Instruction-tuning on recommendation data (TALLRec).** Turn interaction data into instruction examples ("Given the user's liked and disliked items, will they like the target item? Answer Yes or No.") and fine-tune with LoRA (see [Part 2](../part2/index.md)). TALLRec reports that this substantially improves LLM recommendation in the movie and book domains even with fewer than 100 training samples, runs on a single RTX 3090 with LLaMA-7B, and generalizes across domains. A pointwise score is the probability of "Yes":

$$
\hat p(u,i) = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})},
$$

where $z_{\text{Yes}}, z_{\text{No}}$ are the LLM's logits for the two answer tokens at the answer position. In words: the model's relative confidence in "Yes" becomes a click-like probability that can be ranked (and calibrated, Chapter [4.2](02-ranking-and-ctr-models.md)).

**Pointwise vs listwise prompts.** *Pointwise*: one prompt per (user, candidate), giving a probability — clean, but $N$ forward passes. *Listwise*: one prompt with the user history and $N$ candidates, asking for a ranked list — one pass, but output must be parsed, may omit or invent items, and is sensitive to candidate order in the prompt (LLMs have position bias, just like users).

**Adding collaborative semantics.** Pure text loses ID-level signal. **LC-Rec** adds semantic-ID tokens (Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)) to the LLM's vocabulary and aligns them with language through multiple tuning tasks, so the LLM can generate items directly and knows both content and collaborative structure. PLUM (Chapter [4.6](06-end-to-end-generative-recommenders.md)) does this at YouTube scale with continued pre-training.

**One LLM replacing many rankers (360Brew).** LinkedIn's 360Brew is a 150B-parameter decoder-only model with a **textual interface**: member profiles, behaviours and social connections are verbalized, and the task is stated in natural language. The report says this single model solves over 30 predictive tasks across LinkedIn surfaces at performance comparable to or exceeding the dedicated production models on offline metrics, without task-specific fine-tuning — models that were each built and maintained by teams over years. The pitch is less "better accuracy" than "replace a large, feature-engineered model zoo with one model".

**LLM-driven intent (RecGPT).** Alibaba's RecGPT puts an LLM at the centre of Taobao homepage recommendation: the LLM mines user intent from behaviour, predicts item tags that express that intent, tags drive retrieval, and the LLM generates explanations. Training aligns a general LLM to these tasks with reasoning-enhanced pre-alignment and self-training, guided by a human–LLM cooperative judge. It is fully deployed on Taobao, and the report describes gains for users (diversity, satisfaction), merchants and the platform.

### (b) The LLM as offline knowledge / feature provider

The most production-friendly pattern: run the LLM **offline**, store its outputs, and feed them as features to an ordinary ranker.

- **KAR (Knowledge Augmentation from LLMs).** For each user, prompt the LLM to reason about preferences ("factorization prompting" breaks it into aspects such as genre or price sensitivity); for each item, prompt for factual knowledge. A text encoder turns the generated text into vectors, and a small hybrid-expert adaptor maps them into augmented features usable by *any* CTR model. All LLM work is precomputed and stored, so online inference cost is a lookup. KAR reports online A/B gains of 7% and 1.7% on Huawei's news and music recommendation platforms.
- **CTRL.** Convert each tabular sample into text; contrastively align a language model's representation of that text with a CTR model's representation of the tabular features; then fine-tune and **serve only the CTR model**. The LLM's semantic knowledge is transferred during training, with zero extra serving latency.
- **ClickPrompt.** The CTR model produces soft prompts that condition a language model, so collaborative and semantic signals are combined in training; inference can again rely on the CTR side.
- **QARM (Kuaishou).** Multimodal LLM item embeddings are usually frozen features trained for generic objectives. QARM *aligns* them with business signals (user–item behaviour) and **quantizes** them into discrete codes, which the ranking model uses as trainable ID-like features — fixing both the objective mismatch and the "can't be updated by the ranker's gradient" problem.

??? example "A KAR-style pipeline, step by step"
    1. **Prompt per user (offline, daily or on profile change).** Give the LLM the user's recent history as text and ask, aspect by aspect, about preferences — e.g. "Which genres does this user prefer and why? How price-sensitive are they? What occasions are they shopping for?" Splitting the question into factors makes the reasoning more specific than one open question.
    2. **Prompt per item (offline, once per new item).** Ask for factual knowledge: "Describe this product's category, typical buyers, complementary products and key attributes."
    3. **Encode.** Run a text encoder over each generated text to get a vector, and store it in a key–value store keyed by user ID / item ID.
    4. **Adapt.** A small trainable adaptor (KAR uses a hybrid-expert adaptor) maps the stored vectors to augmented feature vectors, trained jointly with the CTR model.
    5. **Serve.** At request time, look up the stored vectors and feed them to the CTR model with the usual features. No LLM call happens online.

### (c) The LLM as user/item encoder

- **HLLM (ByteDance).** Two LLMs in a hierarchy: an **Item LLM** reads an item's text and outputs an item embedding; a **User LLM** reads the sequence of item embeddings from the user's history and predicts the next item's embedding. Pre-trained LLM weights help, fine-tuning helps further, and the approach scales up to 7B parameters for each component, with online A/B gains reported. Item embeddings can be precomputed, so only the User LLM runs per user (and can itself be cached).
- **User-LLM (Google).** Long user timelines are expensive as text. User-LLM pre-trains a user encoder on interaction sequences and feeds the resulting **user embeddings** to the LLM through cross-attention (with Perceiver layers to compress them), the same way vision-language models feed image embeddings. It reports inference speedups up to 78.1× over text-prompt contextualization and improvements up to 16.33% on tasks requiring deep user understanding.
- **URM (Alibaba).** Uses an LLM as a **universal retriever**: multi-query representations, matrix decomposition and probabilistic sampling let it generate a set from tens of millions of candidates for an arbitrary given objective within tens of milliseconds; it reports a 3% improvement in the core metric of an online advertising platform.

### The latency and cost problem — design patterns

| Pattern | What runs online | Examples |
|---|---|---|
| Offline precompute | Table lookup of LLM outputs (item knowledge, user summaries, embeddings) | KAR, HLLM item side, QARM codes |
| Asynchronous / cached user state | LLM runs on user events in the background; requests read the latest cache | User-side LLM embeddings refreshed per session or per day |
| Distillation / alignment then drop the LLM | Only the small CTR model | CTRL |
| One call per request, not per candidate | A single generative pass producing a list or a set | URM, generative retrieval, OneRec-style decoding |
| Low-QPS, high-value surfaces | The LLM directly | Explanations, conversational recommendation, intent mining (RecGPT) |

### How the roles combine in one system

A realistic 2025-era stack uses several roles at once:

1. **Item side, offline:** a multimodal LLM embeds every new item (text, thumbnail, audio transcript); the embedding is aligned with engagement data and quantized into semantic IDs or codes (QARM-style, Chapter [4.5](05-semantic-ids-and-generative-retrieval.md)).
2. **User side, offline/async:** an LLM summarizes long-term interests into text or an embedding, refreshed daily or per session (KAR, User-LLM-style).
3. **Retrieval, online:** one of the retrieval sources is a generative retriever over semantic IDs, possibly initialized from an LLM (URM, PLUM).
4. **Ranking, online:** a scaled sequence ranker (Chapter [4.4](04-scaling-recommendation-models.md)) consumes the LLM-derived item codes and user features as ordinary inputs.
5. **Presentation, online at low QPS:** an LLM writes explanations or powers a conversational entry point (RecGPT-style).

Each LLM call is placed where its cost is paid once per item, once per user-day, or once per request — never once per candidate.

## Code

Prompt templates for LLM-as-recommender, plus robust parsing of a listwise answer. This code is model-agnostic: plug the prompt into any chat or completion LLM.

```python
import re

def pointwise_prompt(liked, disliked, target):
    """TALLRec-style binary question; score = P('Yes') at the answer position."""
    return (
        "You are a recommendation assistant.\n"
        f"Items the user liked: {'; '.join(liked)}.\n"
        f"Items the user disliked: {'; '.join(disliked)}.\n"
        f"Will the user like the target item \"{target}\"? Answer with Yes or No.\n"
        "Answer:"
    )

def listwise_prompt(history, candidates, k=3):
    """One call ranks all candidates. Candidates get short IDs to make parsing reliable."""
    lines = [f"[{j}] {c}" for j, c in enumerate(candidates)]
    return (
        "The user recently interacted with, in order:\n"
        + "\n".join(f"- {h}" for h in history)
        + "\n\nCandidate items:\n" + "\n".join(lines)
        + f"\n\nRank the {k} candidates the user is most likely to choose next. "
        "Reply with candidate numbers only, best first, e.g. [4], [1], [7]."
    )

def parse_listwise(answer, n_candidates, k):
    """Keep valid, unique candidate indices in order; drop hallucinated or repeated ones."""
    seen, out = set(), []
    for m in re.findall(r"\[(\d+)\]", answer):
        j = int(m)
        if 0 <= j < n_candidates and j not in seen:
            seen.add(j); out.append(j)
    return out[:k]

def yes_probability(logit_yes, logit_no):
    """Two-way softmax over the answer tokens."""
    import math
    m = max(logit_yes, logit_no)
    a, b = math.exp(logit_yes - m), math.exp(logit_no - m)
    return a / (a + b)

history = ["Trail running shoes", "Hydration vest", "GPS sports watch"]
cands = ["Espresso machine", "Energy gels (12-pack)", "Foam roller", "Office chair", "Headlamp"]
print(listwise_prompt(history, cands))
fake_llm_answer = "[1], [2], [9], [2], [4]"       # contains an invalid [9] and a duplicate
print(parse_listwise(fake_llm_answer, len(cands), k=3))   # -> [1, 2, 4]
print(round(yes_probability(2.1, 0.3), 3))                # -> 0.858
print(pointwise_prompt(history[:2], ["Office chair"], "Energy gels (12-pack)"))
```

A back-of-envelope cost calculator for comparing deployment patterns:

```python
def llm_flops(params, tokens):
    return 2 * params * tokens                     # forward pass ≈ 2 FLOPs per param per token

GPU_EFFECTIVE = 1e15 * 0.4                         # ~1 PFLOP/s peak at 40% MFU
P = 7e9                                            # 7B-parameter LLM

pointwise = llm_flops(P, 1000) * 500               # 1k-token prompt, 500 candidates, per request
listwise = llm_flops(P, 1000 + 50 * 20)            # one prompt with 50 candidates x 20 tokens
item_precompute = llm_flops(P, 200) * 1e8          # 100M items x 200 tokens, one-off
user_daily = llm_flops(P, 1000) * 1e8              # 100M daily users x 1k tokens, every day

print(f"pointwise : {pointwise / GPU_EFFECTIVE:8.2f} GPU-seconds per request")
print(f"listwise  : {listwise / GPU_EFFECTIVE:8.3f} GPU-seconds per request (prefill only)")
print(f"items     : {item_precompute / GPU_EFFECTIVE / 3600:8.0f} GPU-hours, once")
print(f"users     : {user_daily / GPU_EFFECTIVE / 3600:8.0f} GPU-hours per day "
      f"(~{user_daily / GPU_EFFECTIVE / 86400:.0f} GPUs running continuously)")
```

## Worked example

Assume a 7B LLM, an accelerator with $\approx 10^{15}$ FLOP/s dense peak at 40% MFU ($4\times10^{14}$ effective), and forward cost $\approx 2 \times \text{params} \times \text{tokens}$.

**Pointwise LLM ranking online.** Prompt of 1,000 tokens (user history), 500 candidates per request:
$2 \times 7\times10^9 \times 1000 \times 500 = 7\times10^{15}$ FLOPs per request $\Rightarrow 17.5$ GPU-seconds per request. At $10^4$ QPS that is 175,000 GPUs. Not viable. (Sharing the user prefix with a prefix KV cache, [Part 3](../part3/index.md), helps a lot, but candidates still add tokens and latency.)

**Listwise, one call.** 1,000 history tokens + 50 candidates × 20 tokens = 2,000 tokens: $2\times7\times10^9\times2000 = 2.8\times10^{13}$ FLOPs $\approx 0.07$ GPU-seconds of prefill, plus decoding the answer. Possible for low-QPS surfaces, too slow and costly for a main feed at $10^5$ QPS.

**Offline precompute.** Item knowledge for $10^8$ items × 200 tokens: $2.8\times10^{20}$ FLOPs $\approx 7\times10^5$ GPU-seconds $\approx 194$ GPU-hours, once (plus new items daily). User summaries for $10^8$ daily users × 1,000 tokens: $1.4\times10^{21}$ FLOPs $\approx 972$ GPU-hours/day $\approx 40$ GPUs running continuously. Online cost: a key–value lookup and a slightly larger input to the ranker.

The orders of magnitude explain the industry pattern: **LLMs run offline or once per user, not once per candidate**, unless the model is turned into a generative retriever/recommender that produces the whole list in one pass.

## In practice (2025–2026)

- **Offline LLM features and embeddings** (KAR-style knowledge, multimodal item embeddings aligned to behaviour like QARM) are the most widely deployed pattern.
- **LLM encoders** for items are standard for cold-start; LLM-based user encoders (HLLM, User-LLM) are growing.
- **Generative retrieval with LLM backbones** (URM, PLUM) and **LLM-native rankers** (360Brew) are in or near production at large companies.
- **Intent and explanation** (RecGPT) is where LLM reasoning is most visible to users.
- Most systems mix roles: an LLM-derived item tokenizer, LLM-precomputed features, and a scaled sequence ranker.

!!! tip "Choosing a role for your system"
    Start with (b): offline LLM features are cheap to try and easy to A/B test. Move to (c) when cold-start or long-tail items matter and item text/images are informative. Consider (a) for low-QPS, high-value interactions (search assistants, conversational shopping, explanations) or when you can turn it into one-pass generation over semantic IDs.

## Common pitfalls / misconceptions

- **Expecting zero-shot LLMs to beat trained recommenders on accuracy.** Without collaborative signal they usually don't; their strengths are knowledge, cold-start and explanation.
- **Hallucinated items.** Free-text outputs can name items that do not exist; constrain generation (candidate indices, semantic-ID tries) and validate.
- **Candidate-order bias** in listwise prompts; shuffle and aggregate, or use pointwise scores for final ranking.
- **Data leakage via pre-training:** public benchmarks (e.g. movie datasets) may be in the LLM's pre-training data; evaluate on proprietary or time-split data.
- **Ignoring cost per request**: always multiply by candidates and QPS before choosing an architecture.

## Key takeaways

- LLMs enter recommenders as (a) the recommender, (b) an offline knowledge/feature provider, or (c) a user/item encoder.
- Prompting gives knowledge and explanations but lacks collaborative signal; instruction tuning (TALLRec) and semantic-ID tokens (LC-Rec, PLUM) close the gap.
- KAR, CTRL, ClickPrompt and QARM transfer LLM knowledge into classic rankers at near-zero online cost.
- HLLM, User-LLM and URM use LLMs as encoders/retrievers; 360Brew replaces a model zoo with one LLM; RecGPT puts intent mining at the centre.
- Cost math (FLOPs × candidates × QPS) forces LLMs offline, per user, or into one-pass generation.

## Check your understanding

??? question "Why is a pointwise LLM ranker so much more expensive than a listwise one?"
    Pointwise runs one forward pass per candidate, repeating the user-history prompt each time, so cost scales with (history tokens × candidates). Listwise processes the history once plus a short description per candidate in one prompt.

??? question "How does CTRL deliver LLM knowledge with zero added serving latency?"
    It aligns the CTR model's representations with a language model's during training (contrastive alignment between tabular and textual views of each sample), then fine-tunes and serves only the lightweight CTR model.

??? question "What two problems with frozen multimodal item embeddings does QARM address?"
    Representation mismatch (they were trained for generic NLP/CV objectives, not user–item behaviour) and representation unlearning (as fixed cached inputs they can't be updated by the ranker's gradients). QARM aligns them with behaviour and quantizes them into trainable codes.

??? question "Using the worked example, how many GPU-seconds per request does pointwise ranking of 100 candidates with a 500-token prompt cost?"
    $2\times7\times10^9\times500\times100 = 7\times10^{14}$ FLOPs; divided by $4\times10^{14}$ FLOP/s gives 1.75 GPU-seconds per request.

??? question "Why does HLLM split the model into an Item LLM and a User LLM?"
    Item text is encoded once per item (cacheable), and the User LLM then operates on a much shorter sequence of item embeddings rather than the full text of every history item, making user modelling with LLM weights affordable.

??? question "Where in the stack would you put an LLM to help cold-start items, and why?"
    On the item side, offline: encode each new item's content with an (aligned) LLM into an embedding or semantic ID as soon as it is uploaded. The item immediately gets a meaningful representation that retrieval and ranking can use, at a one-time cost per item and no added request latency.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366) — Geng et al., 2022 | Casts rating, sequential rec, explanation, etc. as text-to-text tasks for one T5 model. |
| :material-book-open-variant: [Is ChatGPT a Good Recommender?](https://arxiv.org/abs/2304.10149) — Liu et al., 2023 | Zero/few-shot LLM recommendation: strengths in explanation, limits on accuracy. |
| :material-star: [TALLRec](https://arxiv.org/abs/2305.00447) — Bao et al., 2023 | LoRA-tuning an LLM on a small amount of recommendation data makes it a strong few-shot recommender. |
| :material-book-open-variant: [Adapting LLMs by Integrating Collaborative Semantics (LC-Rec)](https://arxiv.org/abs/2311.09049) — Zheng et al., 2023 | Semantic-ID tokens in the LLM vocabulary, aligned with language via multiple tuning tasks. |
| :material-star: [Towards Open-World Recommendation with Knowledge Augmentation from LLMs (KAR)](https://arxiv.org/abs/2306.10933) — Xi et al. (Huawei), 2023 | Offline LLM reasoning and item facts as features for any CTR model; deployed in production. |
| :material-book-open-variant: [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841) — Li et al. (Huawei), 2023 | Contrastive alignment, then serve only the CTR model. |
| :material-book-open-variant: [ClickPrompt](https://arxiv.org/abs/2310.09234) — Lin et al., 2023 | CTR model generates soft prompts for an LM, combining collaborative and semantic signals. |
| :material-new-box: [QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739) — Luo et al., 2024 | Aligns multimodal item representations with business signals and quantizes them into ranking features. |
| :material-star: [HLLM: Hierarchical LLMs for Item and User Modeling](https://arxiv.org/abs/2409.12740) — Chen et al. (ByteDance), 2024 | Item LLM + User LLM; pre-trained weights help; scales to 7B. |
| :material-book-open-variant: [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) — Ning et al. (Google), 2024 | User embeddings fed to an LLM via cross-attention / soft prompts. |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | LLM as a multi-query, multi-objective retriever, deployed in advertising. |
| :material-new-box: [360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://arxiv.org/abs/2501.16450) — LinkedIn, 2025 | One 150B decoder LLM with a text interface replaces many task-specific rankers. |
| :material-new-box: [RecGPT Technical Report](https://arxiv.org/abs/2507.22879) — Alibaba (Taobao), 2025 | LLM-driven intent mining → tag prediction → retrieval → explanation on Taobao. |
| :material-star: [A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860) — Wu et al., 2023 | Taxonomy of discriminative vs generative LLM-for-rec approaches. |
| :material-book-open-variant: [How Can Recommender Systems Benefit from LLMs: A Survey](https://arxiv.org/abs/2306.05817) — Lin et al., 2023 | Organizes the field by where (which stage) and how LLMs are used. |
| :material-book-open-variant: [LLMs for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) — Li et al., 2023 | Generative paradigms and item ID design. |
