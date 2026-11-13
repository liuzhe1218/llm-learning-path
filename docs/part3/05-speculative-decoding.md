# 3.5 Speculative Decoding

!!! abstract "What you'll learn"
    - Why decode leaves compute idle, and how a cheap **draft** plus a single **verification** pass by the target model turns that idle compute into extra tokens.
    - The accept/reject rule and a **proof** that the output distribution is exactly the target model's.
    - The expected number of tokens per step as a function of acceptance rate $\alpha$ and draft length $k$, and the resulting speedup.
    - Draft choices: small models, n-gram / prompt lookup, Medusa heads, EAGLE, MTP heads; and **tree verification**.
    - When speculation stops helping (large batches, compute-bound regimes).

## Why this matters / the problem

Chapter 3.1 showed that at small batch sizes a decode step is bandwidth-bound: the GPU spends a few milliseconds streaming all weights from HBM, and the arithmetic finishes in a tiny fraction of that time. Crucially, processing **5 tokens** in one forward pass costs almost the same time as processing **1 token** — the weights are read once either way, and arithmetic intensity is still far below the ridge point.

But decode is sequential: we cannot compute token $t+2$ before we know token $t+1$. Speculative decoding breaks this by **guessing**. A cheap mechanism proposes the next $k$ tokens; the large model checks all of them in one forward pass (which, being memory-bound, costs about one normal step); we keep the longest correct prefix. If the guesses are usually right we produce several tokens per expensive step. Done carefully, this changes **only speed, never the output distribution**. That "free lunch" is why every production engine supports it and why reasoning models with long outputs rely on it.

## Intuition

A senior author (the target model) and a fast junior assistant (the draft). The assistant types the next few words; the author reads them all at once and either nods ("yes, yes, yes — no, that word should be *this*") or corrects the first wrong word. Reading is much faster for the author than writing word by word. If the assistant is right about 80% of the time, the author advances several words per glance.

The subtle part is sampling. With greedy decoding "correct" just means "equals the target's argmax". With random sampling there is no single correct token, so we need a probabilistic acceptance test that makes the combined process sample **exactly** from the target distribution. That test is a form of rejection sampling.

## How it works

### Notation

- $p(\cdot)$: the target model's next-token distribution at some position (after temperature, top-p, etc.), over vocabulary $\mathcal{V}$.
- $q(\cdot)$: the draft's distribution at the same position.
- $k$: number of draft tokens proposed per step (the "draft length" or "lookahead").
- $\alpha$: the **acceptance rate**, the probability that a draft token is accepted.

### The algorithm (one step)

1. **Draft.** Starting from the current context, sample $k$ tokens autoregressively from the draft: $\tilde x_1 \sim q_1$, $\tilde x_2 \sim q_2(\cdot \mid \tilde x_1)$, …, $\tilde x_k$. Keep the draft probabilities $q_i(\tilde x_i)$.
2. **Verify.** Run the target **once** on context $+\ \tilde x_1 \dots \tilde x_k$. Because the target is causal, this one pass yields $k+1$ distributions: $p_1$ (for position 1, given only the context), $p_2$ (given $\tilde x_1$), …, $p_{k+1}$ (given all drafts).
3. **Accept/reject left to right.** For $i = 1, \dots, k$: draw $r \sim \text{Uniform}(0,1)$. If $r < \min\!\big(1, \tfrac{p_i(\tilde x_i)}{q_i(\tilde x_i)}\big)$, accept $\tilde x_i$ and continue. Otherwise **reject**: sample a replacement token from the **residual distribution**

    $$
    p_{\text{res}}(x) = \frac{\max\big(0,\ p_i(x) - q_i(x)\big)}{\sum_{y}\max\big(0,\ p_i(y) - q_i(y)\big)},
    $$

    output it, and stop (discard $\tilde x_{i+1},\dots,\tilde x_k$).
4. **Bonus token.** If all $k$ drafts were accepted, sample one more token from $p_{k+1}$ — it came for free in the verification pass.

Every step outputs between 1 and $k+1$ tokens and costs one target forward pass plus $k$ draft passes. The target's KV cache must be rolled back to drop rejected positions (with paged KV this just means truncating the sequence length).

### Proof that the output distribution is exactly $p$

Consider one position with draft $q$ and target $p$. The emitted token $X$ is either an accepted draft or a residual sample. For any token $x$:

$$
P(X = x) = \underbrace{q(x)\cdot \min\!\Big(1, \frac{p(x)}{q(x)}\Big)}_{\text{draft proposes } x \text{ and it is accepted}} \;+\; \underbrace{P(\text{reject}) \cdot p_{\text{res}}(x)}_{\text{rejected, then residual gives } x}.
$$

The first term is $\min(q(x), p(x))$. For the second, the total acceptance probability is $\beta = \sum_y \min(q(y), p(y))$, so $P(\text{reject}) = 1 - \beta$. Using $\min(a,b) + \max(0, b - a) = b$ and summing over $y$:

$$
\sum_y \max\big(0, p(y) - q(y)\big) = \sum_y \big(p(y) - \min(q(y), p(y))\big) = 1 - \beta .
$$

So the residual's normalizer is exactly $1-\beta$, and the second term equals $(1-\beta)\cdot \dfrac{\max(0, p(x) - q(x))}{1-\beta} = \max(0, p(x)-q(x))$. Adding the two terms:

$$
P(X = x) = \min\big(q(x), p(x)\big) + \max\big(0, p(x) - q(x)\big) = p(x). \qquad \blacksquare
$$

In words: whenever the draft over-proposes a token ($q > p$) we reject the excess; whenever it under-proposes ($q < p$), the residual distribution fills exactly the missing mass. Applied position by position (each accepted token conditions the next, exactly as in ordinary decoding), the whole generated sequence has the target's distribution. For greedy decoding ($p$ and $q$ one-hot), the rule reduces to "accept iff the draft token equals the target's argmax, otherwise output the target's argmax".

As a by-product, the acceptance probability at a position is $\beta = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$, one minus the total-variation distance. A draft that is close to the target in distribution has a high acceptance rate.

### Expected tokens per step

Assume (as a simplifying model) that each draft token is accepted independently with probability $\alpha$. Draft $i$ is reached and accepted only if drafts $1..i$ are all accepted, probability $\alpha^i$. The number of accepted drafts $A$ satisfies $P(A \ge i) = \alpha^i$ for $i \le k$, so $\mathbb{E}[A] = \sum_{i=1}^{k} \alpha^i$. Each step also emits exactly one extra token (the residual sample on rejection, or the bonus token if all were accepted). Hence

$$
\mathbb{E}[\text{tokens per step}] = 1 + \sum_{i=1}^{k}\alpha^i = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

In words: with perfect drafts ($\alpha \to 1$) you get $k+1$ tokens per step; with useless drafts ($\alpha = 0$) you still get 1 — speculation never produces fewer tokens per target pass than normal decoding.

**Speedup.** Let one target decode step cost $T$, one draft step cost $cT$ (so $c$ is the draft-to-target cost ratio), and assume verifying $k+1$ tokens costs about $T$ (true while memory-bound). Then

$$
\text{speedup} \approx \frac{1 - \alpha^{k+1}}{(1 - \alpha)(1 + ck)} .
$$

Increasing $k$ adds draft cost linearly but adds tokens with geometrically shrinking returns, so there is an optimal $k$, typically 3–8.

### Choosing a draft

| Draft | How it proposes | Pros | Cons |
|---|---|---|---|
| **Small model** of the same family (e.g. 1B drafting for 70B) | Autoregressive sampling with its own KV cache | Simple, lossless with sampling | Must share the tokenizer; extra memory; $c$ not negligible |
| **n-gram / prompt lookup** | Find the last few generated tokens earlier in the prompt or output, propose what followed | Zero training, near-zero cost | Only helps when the output copies input (code edits, RAG, summarization, rewriting) |
| **Medusa heads** | $k$ extra heads on the target's last hidden state; head $i$ predicts token $t+i$ directly | One model, no extra KV cache | Heads predict independently (no conditioning on earlier drafts) → lower acceptance for later positions |
| **EAGLE (1/2/3)** | A small (one-layer) transformer that runs autoregressively on the target's **hidden features** plus token embeddings | High acceptance at low cost; EAGLE-2 adds dynamic trees, EAGLE-3 fuses low/mid/high-layer features | Needs training per target model |
| **MTP heads** (DeepSeek-V3 style) | Multi-token-prediction modules trained *during pre-training* | Comes with the model; high acceptance | Only for models trained with MTP |
| **Lookahead decoding** | Jacobi-style parallel iteration generates candidate n-grams, verified in parallel | No draft model | More compute per step |

EAGLE's key observation is that predicting the next **feature vector** (the target's second-to-top-layer hidden state) is easier than predicting the next token directly, and that feeding the draft the token sampled one step ahead resolves the uncertainty of which branch the sampling took.

### Tree verification

Instead of one chain of $k$ guesses, propose a **tree**: e.g. the top-3 candidates for position 1, and for each the top-2 for position 2. Flatten all tree nodes into one sequence and run the target once with a **tree attention mask**: each node attends to the context and to its own ancestors only (not to siblings). One pass verifies every path; keep the longest accepted path. Trees raise the expected accepted length for the same number of target passes, at the cost of verifying more tokens per pass.

```text
draft tree: top-2 at depth 1, top-2 under each       flattened order: [a, b, a1, a2, b1, b2]
                                                     tree attention mask (1 = may attend;
             ┌─ a ──┬─ a1                            every node also sees the whole context)
 context ────┤      └─ a2                                  a  b  a1 a2 b1 b2
             └─ b ──┬─ b1                              a   1  0  0  0  0  0
                    └─ b2                              b   0  1  0  0  0  0
                                                       a1  1  0  1  0  0  0
 one target pass scores all 6 nodes; keep the          a2  1  0  0  1  0  0
 longest root-to-leaf path that passes verification    b1  0  1  0  0  1  0
                                                       b2  0  1  0  0  0  1
```

Positions also need care: nodes `a1` and `b1` are both at depth 2, so they receive the same position index (for RoPE) even though they are different entries in the flattened sequence. SpecInfer introduced tree-based verification with a lossless multi-candidate acceptance rule for sampling; Medusa and EAGLE-2/3 use draft trees (EAGLE-2 shapes the tree dynamically from the draft's confidence).

### When speculation hurts

The whole argument relied on "verifying $k+1$ tokens costs about one decode step", i.e. the target is memory-bound. With batch size $B$, a verification pass processes $B(k+1)$ tokens. Chapter 3.1 showed linear layers become compute-bound around a few hundred tokens per pass on an H100. Once $B(k+1)$ crosses that, verification costs grow roughly linearly with $k+1$, and every rejected token is wasted compute that other users could have used. At large batches, speculation can **reduce** throughput. Engines therefore tune or disable speculation by load, and speculation is most valuable for latency-sensitive, low-to-moderate-concurrency serving and for long single-stream generations.

## Code

A complete speculative sampling implementation on toy "language models" (random Markov chains over a vocabulary of 6 tokens), with an empirical check that the output distribution equals the target's and that the measured tokens per step match the formula's ballpark.

```python
import numpy as np
rng = np.random.default_rng(0)
V, k = 6, 4

def random_chain(sharpness):          # next-token table: P[prev] is a distribution
    logits = rng.standard_normal((V, V)) * sharpness
    P = np.exp(logits); return P / P.sum(1, keepdims=True)

P_target = random_chain(2.0)
P_draft = 0.7 * P_target + 0.3 * random_chain(2.0)     # a decent but imperfect draft

def speculative_step(prev):
    """One draft-verify step from token `prev`. Returns the list of emitted tokens."""
    drafts, q_probs, ctx = [], [], prev
    for _ in range(k):                                   # 1) draft k tokens from q
        x = rng.choice(V, p=P_draft[ctx]); drafts.append(x); q_probs.append(P_draft[ctx]); ctx = x
    p_probs = [P_target[c] for c in [prev] + drafts]     # 2) "one target pass": k+1 dists
    out = []
    for i, x in enumerate(drafts):                       # 3) accept / reject left to right
        p, q = p_probs[i], q_probs[i]
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x); continue
        res = np.maximum(p - q, 0); res /= res.sum()     # residual distribution
        out.append(rng.choice(V, p=res)); return out
    out.append(rng.choice(V, p=p_probs[k]))              # 4) bonus token
    return out

# Generate many sequences of 3 tokens from start token 0 and compare token-3 frequencies.
n_runs, counts, steps, toks = 40000, np.zeros(V), 0, 0
for _ in range(n_runs):
    seq = []
    while len(seq) < 3:
        new = speculative_step(seq[-1] if seq else 0); steps += 1; toks += len(new); seq += new
    counts[seq[2]] += 1
exact = np.linalg.matrix_power(P_target, 3)[0]           # true P(token 3 | start 0)
print("speculative:", np.round(counts / n_runs, 3))
print("target     :", np.round(exact, 3))

alpha = np.mean([np.minimum(P_target[s], P_draft[s]).sum() for s in range(V)])
print(f"alpha~{alpha:.2f}  formula tokens/step={(1 - alpha**(k + 1)) / (1 - alpha):.2f}  "
      f"measured={toks / steps:.2f}")   # close, not equal: real alpha varies by context
```

The two frequency rows agree to within Monte-Carlo noise, even though the draft is a different distribution. Try setting `P_draft = random_chain(2.0)` (an unrelated draft): the output distribution stays correct, only the acceptance rate drops.

## Worked example

**One greedy step, traced.** Context: "The capital of France is". The draft proposes $k = 4$ tokens: `" Paris"`, `"."`, `" It"`, `" is"`. The target runs once on the context plus these 4 tokens and returns its argmax at each of the 5 positions:

| Position | Draft token | Target argmax (given everything before) | Result |
|---|---|---|---|
| 1 | `" Paris"` | `" Paris"` | accept |
| 2 | `"."` | `"."` | accept |
| 3 | `" It"` | `" The"` | reject → emit `" The"`, stop |
| 4 | `" is"` | (not used) | discarded |
| 5 | — | (not used) | no bonus token |

Output of this step: `" Paris"`, `"."`, `" The"` — **3 tokens for one target pass**. The KV entries the target wrote for the rejected tokens `" It"` and `" is"` are discarded (the sequence length is simply truncated); `" The"` gets its KV entry when it is fed in at the start of the next step, which continues from "…is Paris. The".

**Setup.** Target: a 70B model where one decode step takes $T$. Draft: an EAGLE-style head whose step costs $c = 0.05\,T$. Measured acceptance $\alpha = 0.8$.

**Tokens per step for $k = 4$:**

$$
\frac{1 - 0.8^{5}}{1 - 0.8} = \frac{1 - 0.32768}{0.2} = 3.36 .
$$

**Speedup:** each step costs $T(1 + 4 \times 0.05) = 1.2\,T$, so speedup $= 3.36 / 1.2 = 2.8\times$.

**Choosing $k$** (same $\alpha$ and $c$):

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| tokens/step | 1.80 | 2.44 | 2.95 | 3.36 | 3.69 | 3.95 | 4.16 | 4.33 | 4.57 |
| speedup | 1.71 | 2.22 | 2.57 | 2.80 | 2.95 | 3.04 | 3.08 | 3.09 | 3.05 |

Gains flatten around $k \approx 6$–8 and then decline. With a separate 1B draft model ($c$ perhaps 0.1–0.2) the optimum shifts to smaller $k$.

**Batch effect.** On an H100 the crossover to compute-bound is a few hundred tokens per pass (Chapter 3.1). At batch 8 with $k=4$, verification processes 40 tokens: still memory-bound, so the 2.8× estimate holds roughly. At batch 128 with $k=4$, it processes 640 tokens: compute-bound, so verification costs about $640/128 = 5\times$ a normal step, and the "speedup" becomes about $3.36 / (5 + 0.2) \approx 0.65$ — a slowdown. (Real crossovers depend on model, KV length and kernels; measure.)

## In practice (2025–2026)

- vLLM, SGLang and TensorRT-LLM support draft models, n-gram/prompt lookup, Medusa, EAGLE (including EAGLE-3) and MTP. EAGLE-3 became the default high-performance method in SGLang/vLLM in 2025, with released draft heads for popular open models.
- **DeepSeek-V3** reports that its MTP module, reused for speculative decoding, has a second-token acceptance rate of about 85–90%, giving about 1.8× tokens per second.
- Engines expose the draft length (`num_speculative_tokens` or similar) and report acceptance rate / mean accepted length in metrics; watch those numbers in production, since acceptance depends on the domain (code and structured output accept much better than creative writing).
- Speculation is particularly valuable for **reasoning models** at low concurrency, where a single response can be tens of thousands of tokens, and for "apply edit" workloads where prompt lookup accepts long runs of copied text.

## Common pitfalls / misconceptions

- **"Speculative decoding is approximate."** With the rejection rule above it is exactly lossless in distribution. (Some variants, e.g. Medusa's "typical acceptance", deliberately trade exactness for speed — know which one you enable.)
- **"Higher k is always better."** Returns diminish geometrically while draft cost grows linearly; at high load extra verified tokens cost real compute.
- **Measuring speedup only at batch 1.** Report speedup across the concurrency range you serve; it can turn negative at high load.
- **Mismatched sampling parameters.** $p$ and $q$ must be the *processed* distributions (same temperature/top-p pipeline applied to the target); otherwise the guarantee breaks.
- **Bit-identical outputs.** Lossless "in distribution" does not mean identical text to non-speculative decoding with the same seed; random numbers are consumed differently, and batched verification can change floating-point results slightly.

## Key takeaways

- Decode is memory-bound, so verifying $k+1$ tokens costs about one decode step; speculative decoding spends that idle compute on guesses.
- Accept draft $\tilde x$ with probability $\min(1, p/q)$; on rejection resample from $\max(0, p-q)$ normalized. The result is exactly distributed as $p$.
- Expected tokens per step $= (1-\alpha^{k+1})/(1-\alpha)$; speedup $\approx$ that divided by $(1 + ck)$.
- Draft options trade training cost vs acceptance: n-gram (free), small model, Medusa, EAGLE, MTP; trees verify many candidates in one pass.
- At high batch sizes verification becomes compute-bound and speculation can slow the system down.

## Check your understanding

??? question "Target p = (0.6, 0.3, 0.1) and draft q = (0.3, 0.6, 0.1) over tokens (a, b, c). What is the acceptance probability, and what is the residual distribution?"
    $\beta = \min(.6,.3) + \min(.3,.6) + \min(.1,.1) = 0.3 + 0.3 + 0.1 = 0.7$. Residual $\propto \max(0, p-q) = (0.3, 0, 0)$, i.e. always token a. Check: $P(a) = 0.3 + 0.3 \cdot 1 = 0.6$ ✓.

??? question "With α = 0.6 and k = 3, how many tokens per target step on average?"
    $(1 - 0.6^4)/(1-0.6) = (1 - 0.1296)/0.4 = 2.18$.

??? question "Why do we get a 'bonus' token when all drafts are accepted?"
    The verification pass on context + $k$ drafts also produces the target distribution for position $k+1$, so sampling one more token from it costs nothing extra.

??? question "Why do Medusa heads usually have lower acceptance for the 3rd and 4th drafted positions than EAGLE?"
    Medusa's head $i$ predicts token $t+i$ from the hidden state at $t$ without seeing the drafted tokens in between, so it cannot condition on which branch was chosen. EAGLE drafts autoregressively, feeding back each drafted token and predicted feature.

??? question "Your service runs at batch 256 most of the day and batch 8 at night. How would you configure speculation?"
    Enable it (with a moderate $k$) when load is low and disable or shrink $k$ at high load, where verification is compute-bound; many engines support adjusting this dynamically or you can route by load.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022 | The original algorithm, its losslessness proof, and the speedup analysis used above. |
| :material-book-open-variant: [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al. (DeepMind), 2023 | Concurrent derivation of the same rule; applied to Chinchilla-70B. |
| :material-book-open-variant: [SpecInfer: Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) — Miao et al., 2023 | Verify a tree of candidates in one pass with tree attention; multi-candidate lossless acceptance. |
| :material-star: [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Cai et al., 2024 | Extra heads on the target predict future tokens; tree verification; no separate draft model. |
| :material-star: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., 2024 | Feature-level autoregressive drafting → much higher acceptance. |
| :material-book-open-variant: [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — Li et al., 2024 | Context-dependent draft trees shaped by draft confidence. |
| :material-new-box: [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — Li et al., 2025 | Multi-layer feature fusion and training-time simulation of decoding; the default speculative method in SGLang/vLLM in 2025. |
| :material-book-open-variant: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Fu et al., 2024 | Draft-free: Jacobi iteration generates n-grams verified in parallel. |
| :material-book-open-variant: [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | The simple n-gram-from-the-prompt draft method, with code. |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | MTP modules trained in pre-training and reused as a speculative draft. |
