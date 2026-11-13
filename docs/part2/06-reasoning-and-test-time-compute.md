# 2.6 Reasoning and Test-Time Compute

!!! abstract "What you'll learn"
    - Why chain-of-thought (CoT) helps: more serial computation and a scratchpad of intermediate tokens
    - Self-consistency (majority voting), with a worked probability calculation of when it helps and when it hurts
    - Best-of-N with a verifier, pass@k, and the difference between outcome and process reward models
    - Search over reasoning (beam search, tree search) at an intuitive level
    - How "thinking" models trade test-time compute for accuracy, including budgets and s1's budget forcing

## Why this matters / the problem

So far, every method improved the model by changing its **weights** with more training compute. There is a second lever: spend more compute **at inference time** on each problem. A person solving a hard math problem does not answer instantly; they write down steps, try alternatives, check their work. Can a language model do the same, and can we make it better by letting it think longer?

Since 2024 the answer has been a clear yes. "Reasoning" or "thinking" models (OpenAI's o1 series, DeepSeek-R1, Qwen3's thinking mode, Claude with extended thinking, Gemini thinking models) generate long internal reasoning before answering, and their accuracy on math, code and science rises with the number of thinking tokens. This chapter explains *why* extra tokens help and the main ways to spend test-time compute; [Chapter 2.7](07-rl-with-verifiable-rewards.md) explains how RL trains models to use it.

## Intuition

**A transformer does a fixed amount of computation per token.** Each generated token passes through the same $n_{\text{layers}}$ layers ([Part 1](../part1/index.md)). If a question requires more sequential steps of computation than the network's depth provides, the model cannot compute the answer "in its head" in one token. But it can write intermediate results down, then read them back with attention when producing the next token. Every generated token buys another full pass through the network, and the written tokens act as **external working memory** (a scratchpad).

Try computing $37 \times 48$ instantly, versus writing: $37\times48 = 37\times50 - 37\times2 = 1850 - 74 = 1776$. Same brain, more serial steps, written memory. CoT is that for an LLM.

There are two broad ways to spend more inference compute:

```text
  SEQUENTIAL (think longer)                    PARALLEL (think wider)
  one long chain of thought                    many independent attempts, then pick
  ┌──────────────────────────────────┐         ┌──── attempt 1 → 42
  │ step 1 → step 2 → check → oops → │         ├──── attempt 2 → 42      majority vote → 42
  │ backtrack → step 2' → answer     │         ├──── attempt 3 → 17      or verifier picks best
  └──────────────────────────────────┘         └──── attempt 4 → 42
  (CoT, thinking models, revision)             (self-consistency, best-of-N, search)
```

## How it works

### Chain-of-thought prompting

Wei et al. (2022) showed that adding a few worked examples with step-by-step reasoning to the prompt (few-shot CoT) greatly improved large models on arithmetic, commonsense and symbolic reasoning, while giving little or no benefit to small models: the ability to use CoT appeared with scale. Simply appending "Let's think step by step" to a question (zero-shot CoT) also helps strong models.

Formally, instead of modeling $p(a\mid x)$ directly for answer $a$, the model generates a rationale $z$ first and then the answer:

$$
p(a\mid x) = \sum_{z} p(z\mid x)\, p(a\mid x, z).
$$

In words: the answer probability is an average over possible reasoning paths. Sampling one CoT means sampling one path $z$; this view directly motivates self-consistency, which approximates the sum by sampling several paths.

Why it helps, summarized:

1. **More serial compute** per problem (one network pass per reasoning token).
2. **Working memory**: intermediate results are stored as tokens and retrieved by attention.
3. **Decomposition**: the training data (textbooks, solutions) contains step-by-step reasoning, so the model has learned many small reliable steps, each easier than the whole jump.

CoT text is not guaranteed to be a faithful account of how the model actually reached its answer; it is generated text that *influences* the answer, which matters for interpretability and safety ([Chapter 2.9](09-agents-tool-use-and-safety.md)).

### Self-consistency (majority voting)

Wang et al. (2022): sample $N$ chain-of-thought solutions at temperature $> 0$, extract each final answer, and output the most frequent answer. No training, no verifier, just more samples.

Why does it work? If the model's per-sample probability of the correct answer is higher than that of any particular wrong answer, then as $N$ grows the most common answer converges to the correct one. Wrong reasoning paths tend to scatter across many different wrong answers, while correct paths converge on the same one.

When does it fail? If the model's *most likely* answer is wrong, voting amplifies the error (see the worked example). It also needs answers that can be compared exactly (a number, a multiple-choice letter), not free-form essays.

### Best-of-N with a verifier, and pass@k

A **verifier** is anything that scores a candidate solution: a learned reward model, a unit-test runner, a proof checker. **Best-of-N** samples $N$ candidates and returns the highest-scoring one.

The ceiling for best-of-N is **pass@N**: the probability that *at least one* of $N$ samples is correct. If each sample is independently correct with probability $p$,

$$
\text{pass@}N = 1 - (1-p)^N.
$$

In words: the chance that not all $N$ attempts fail. A perfect verifier achieves pass@N; a real one achieves less, because it sometimes prefers a wrong but convincing solution. With a *learned* verifier, increasing $N$ eventually *hurts*: more samples mean more chances to find a wrong solution that fools the verifier (the same over-optimization as in [Chapter 2.4](04-rlhf.md)). Cobbe et al. (2021) introduced GSM8K and trained verifiers to rerank sampled solutions, showing this approach can beat fine-tuning a larger model.

To *estimate* pass@k fairly from $n \ge k$ samples per problem of which $c$ are correct, use the unbiased estimator (from OpenAI's Codex paper)

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

the probability that a random subset of $k$ of the $n$ samples contains at least one correct one.

### Outcome vs process reward models

A verifier can judge at two granularities:

- **Outcome reward model (ORM)**: scores only the final answer / full solution. Training labels are cheap (is the final answer correct?), but a solution with flawed reasoning and a lucky right answer gets full credit, and the signal does not say *where* a wrong solution went wrong.
- **Process reward model (PRM)**: scores *each step* of the reasoning. A solution's score is typically the product (or minimum) of step-level probabilities of correctness. It gives denser, more precise feedback, but needs step-level labels, which are expensive.

Lightman et al. (2023) collected about 800K human step-level labels (PRM800K) on model-generated MATH solutions and found that a PRM selected correct solutions more reliably than an ORM in best-of-N reranking. Later work produced step labels automatically (e.g. by estimating each step's correctness from how often rollouts continued from that step reach the right answer).

In 2025, however, the strongest open reasoning models (DeepSeek-R1, see [Chapter 2.7](07-rl-with-verifiable-rewards.md)) were trained with **outcome-only rule-based rewards**; the R1 report lists PRMs among approaches that did not work well for them at scale, citing difficulty defining steps, label noise, and reward hacking of a learned model. PRMs remain useful for search and reranking.

### Search over reasoning

With a step-level scorer you can search rather than sample independently:

- **Beam search over steps**: keep the $B$ highest-scoring partial solutions, extend each by sampling several next steps, rescore, keep the top $B$ again.
- **Tree search** (e.g. Tree of Thoughts, Yao et al., 2023; Monte Carlo tree search variants): explore a tree of partial reasoning, using a value estimate to decide which branches to expand and backtracking from dead ends.

Search spends compute more cleverly than independent sampling on problems where early mistakes are fatal, but it is complex to implement and serve, and depends on a reliable step scorer. Thinking models largely *internalize* search: a single long chain of thought includes trying an approach, noticing a problem, and backtracking ("wait, that's wrong…").

### Scaling test-time compute

Snell et al. (2024) studied how to allocate a fixed inference budget. Key findings:

- The best strategy depends on difficulty. For easy problems, **sequential revision** (let the model revise its own answer) is efficient; for harder ones, **parallel sampling/search against a verifier** helps more.
- A **compute-optimal** strategy that adapts to estimated difficulty was much more efficient than plain best-of-N.
- In a FLOPs-matched comparison, a smaller model with well-spent test-time compute could outperform a model about 14× larger on problems where the small model already had non-trivial success rates; on the hardest problems, pre-training scale still won.

The takeaway: **test-time compute and training compute are partially exchangeable**, which is the conceptual basis for thinking models.

### Thinking models, budgets, and budget forcing

A **thinking model** is trained (mostly with RL, [Chapter 2.7](07-rl-with-verifiable-rewards.md)) to produce a long reasoning segment, often delimited by special markers such as `<think> ... </think>`, before the visible answer. Its accuracy typically increases roughly linearly with the *logarithm* of the number of thinking tokens: each doubling of thinking buys a similar improvement.

Because thinking is expensive, products expose **budgets**: a maximum number of thinking tokens or a coarse "reasoning effort" level, and some models support switching thinking on or off per request. Hybrid models trained to do both are common in 2025–2026.

**s1 and budget forcing** (Muennighoff et al., 2025) showed how simple this can be. They fine-tuned Qwen2.5-32B-Instruct on just 1,000 carefully selected questions with long reasoning traces (s1K), then controlled thinking length at inference:

- **To cap thinking**: when the budget is reached, force the end-of-thinking delimiter (and a prompt such as "Final Answer:") so the model must answer.
- **To extend thinking**: when the model tries to end its thinking early, suppress the end-of-thinking delimiter and append the word "Wait"; the model then often re-examines its reasoning and fixes mistakes.

With this, accuracy on competition math rose as the budget increased, and the s1 model reported results exceeding OpenAI's o1-preview on competition math benchmarks. The broader lesson: much of the reasoning ability is **latent** in a strong base model, and a small amount of data plus test-time control can unlock it.

### STaR: learning from your own reasoning

STaR (Zelikman et al., 2022) closes the loop between test-time reasoning and training: generate rationales for many questions; keep those that reach the correct answer; for failures, give the model the correct answer as a hint and ask it to produce a rationale ("rationalization"); fine-tune on the kept rationales; repeat. This "expert iteration" is REINFORCE with a 0/1 reward ([Chapter 2.3](03-rl-foundations-for-llms.md)) and an ancestor of RLVR.

## Code

### 1. Self-consistency and best-of-N, given sampled solutions

```python
import re
from collections import Counter

def extract_answer(text):
    """Take the last \\boxed{...} if present, else the last number in the text."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def majority_vote(samples):
    answers = [a for a in map(extract_answer, samples) if a is not None]
    if not answers:
        return None, 0.0
    ans, count = Counter(answers).most_common(1)[0]
    return ans, count / len(answers)          # answer and its vote share (a confidence signal)

def best_of_n(samples, score_fn):
    """score_fn: any verifier, e.g. a reward model or a unit-test pass rate."""
    return max(samples, key=score_fn)

samples = ["... so 37*48 = 1776. \\boxed{1776}",
           "37*48 = 37*50 - 74 = 1776",
           "37*48 = 1666, \\boxed{1666}",
           "... therefore the answer is 1776"]
print(majority_vote(samples))                # ('1776', 0.75)
```

### 2. Exact majority-vote accuracy and an unbiased pass@k estimator

```python
from math import comb

def majority_correct_prob(p, n):
    """Two-answer worst case: each sample is correct w.p. p, otherwise the SAME wrong answer.
    Majority (n odd) is correct iff more than n/2 samples are correct."""
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))

def pass_at_k(n, c, k):
    """Unbiased estimate from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

for p in (0.6, 0.4):
    print(p, [round(majority_correct_prob(p, n), 3) for n in (1, 5, 15, 41)])
print(pass_at_k(n=10, c=3, k=4))             # 0.833
```

### 3. Budget forcing with a thinking model

```python
# pip install torch transformers  (Qwen3 models use <think> ... </think>)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
END_THINK = tok.convert_tokens_to_ids("</think>")

def think_with_budget(question, min_think=200, max_think=800, max_waits=2):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    used, waits = 0, 0
    while True:
        ids = tok(text, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=max_think - used, do_sample=False,
                             eos_token_id=[END_THINK, tok.eos_token_id])
        new = out[0, ids.shape[1]:]
        used += len(new)
        ended = len(new) > 0 and new[-1].item() == END_THINK
        text += tok.decode(new[:-1] if ended else new, skip_special_tokens=False)
        if ended and used < min_think and waits < max_waits:
            text += "\nWait"                   # suppress the end of thinking: think more
            waits += 1
            continue
        break
    text += "\n</think>\n\n"                   # cap: force the end of thinking and answer
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=100, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

print(think_with_budget("How many r's are in the word 'strawberry'?"))
```

## Worked example

**When does majority voting help?** Assume the worst case for voting: each sample is correct with probability $p$, and every wrong sample gives the *same* wrong answer (if wrong answers scatter, voting does even better). With $N$ odd, the vote is correct when more than half the samples are correct, a binomial tail:

$$
P(\text{vote correct}) = \sum_{k > N/2} \binom{N}{k} p^k (1-p)^{N-k}.
$$

For $p = 0.6$ and $N = 5$:

$$
\binom{5}{3}0.6^3 0.4^2 + \binom{5}{4}0.6^4 0.4 + 0.6^5 = 0.3456 + 0.2592 + 0.0778 = 0.683.
$$

| $N$ | 1 | 5 | 15 | 41 |
|---|---|---|---|---|
| $p = 0.6$ | 0.600 | 0.683 | 0.787 | 0.903 |
| $p = 0.4$ | 0.400 | 0.317 | 0.213 | 0.097 |

Voting **amplifies whatever the model's most likely answer is**: it pushes accuracy toward 1 when the right answer is the mode, and toward 0 when a wrong answer is.

**Best-of-N ceiling.** If a hard problem has per-sample success $p = 0.2$: pass@4 $= 1 - 0.8^4 \approx 0.590$; pass@16 $= 1 - 0.8^{16} \approx 0.972$. A perfect verifier (e.g. unit tests that fully specify the task) would turn a 20% model into a 97% system with 16 samples, at 16× the inference cost. Majority voting would not help at all here if the 80% of wrong samples concentrate on one wrong answer.

## In practice (2025–2026)

- **Reasoning is trained, not just prompted.** Frontier models produce thousands to tens of thousands of thinking tokens on hard problems; this behavior comes from RL on verifiable tasks ([Chapter 2.7](07-rl-with-verifiable-rewards.md)) and from distillation of such traces ([Chapter 2.8](08-distillation-and-efficient-fine-tuning.md)).
- **Benchmarks are reported with a compute setting.** Results like "AIME accuracy with $k$ samples and majority voting" or "pass@1 averaged over 16 samples" are common; compare numbers only at matched settings.
- **Parallel + sequential.** Top systems combine long thinking with parallel attempts and aggregation for the hardest problems.
- **Overthinking is a cost.** Thinking models can spend many tokens on easy questions; budgets, effort settings, and length penalties during RL (e.g. Kimi k1.5's "long2short" methods) address this.
- **Faithfulness.** Labs study whether visible reasoning reflects the computation that produced the answer, since CoT monitoring is used as a safety tool.

## Common pitfalls / misconceptions

- **"CoT always helps."** For small models and simple lookups it can hurt or add nothing but cost.
- **"More samples always help."** Majority voting hurts when the mode is wrong; best-of-N with a learned verifier eventually over-optimizes the verifier.
- **Comparing pass@k with pass@1.** pass@k assumes an oracle picks the right sample; it is an upper bound for practical systems that must commit to one answer.
- **Estimating pass@k with $k$ samples only.** Use $n > k$ samples and the unbiased estimator; $1-(1-\hat p)^k$ with a plug-in $\hat p$ is biased.
- **Treating the CoT as a faithful explanation.** It is useful evidence, not ground truth about the model's internals.

## Key takeaways

- Each generated token is another pass through the network; CoT adds serial compute and a token scratchpad.
- Self-consistency samples many CoTs and votes; it amplifies the model's modal answer, for better or worse.
- Best-of-N is bounded by pass@N $= 1-(1-p)^N$ and by verifier quality; PRMs score steps, ORMs score outcomes.
- Test-time compute can substitute for model size on problems within reach (Snell et al.); thinking models make this the default.
- Budget forcing (s1) shows thinking length can be controlled at inference with a delimiter and a "Wait".

## Check your understanding

??? question "Why can a fixed-depth transformer solve more problems when allowed to write intermediate steps?"
    Each generated token is computed by a full forward pass, so writing $T$ tokens provides $T$ times more sequential computation, and the written intermediate results can be read back via attention, serving as working memory.

??? question "A model answers a question correctly 45% of the time and gives the same wrong answer 55% of the time. Will majority voting over 31 samples help?"
    No, it will hurt: the mode is wrong, so the vote converges toward the wrong answer as $N$ grows.

??? question "What is the difference between an ORM and a PRM, and one advantage of each?"
    An ORM scores the final outcome (cheap labels, simple); a PRM scores each reasoning step (denser, localizes errors, more reliable for reranking but needs step labels).

??? question "From 20 samples with 5 correct, estimate pass@1 and pass@2."
    pass@1 $= 1 - \binom{15}{1}/\binom{20}{1} = 1 - 15/20 = 0.25$. pass@2 $= 1 - \binom{15}{2}/\binom{20}{2} = 1 - 105/190 \approx 0.447$.

??? question "How does s1's budget forcing extend thinking?"
    When the model emits the end-of-thinking delimiter before the desired budget, the delimiter is suppressed and "Wait" is appended, prompting the model to continue reasoning (often re-checking its work).

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al., 2022 | Step-by-step exemplars make large models much better at multi-step problems. |
| :material-book-open-variant: [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022 | Sample many reasoning paths and majority-vote: the simplest test-time scaling. |
| :material-book-open-variant: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) — Cobbe et al., 2021 | Introduces GSM8K and verifier-based reranking of sampled solutions. |
| :material-book-open-variant: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — Zelikman et al., 2022 | Keep self-generated rationales that reach the right answer, fine-tune, repeat. |
| :material-star: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al. (OpenAI), 2023 | Process reward models beat outcome reward models for selecting math solutions; releases PRM800K. |
| :material-book-open-variant: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Yao et al., 2023 | Explicit tree search over reasoning steps with self-evaluation. |
| :material-star: [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) — Snell et al., 2024 | How to allocate inference compute; when it beats a larger model. |
| :material-new-box: [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) — Muennighoff et al., 2025 | 1K long-CoT examples plus budget forcing; reasoning is largely latent in strong base models. |
