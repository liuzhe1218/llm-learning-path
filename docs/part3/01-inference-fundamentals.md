# 3.1 Inference Fundamentals

!!! abstract "What you'll learn"
    - Write the autoregressive generation loop and explain the difference between the **prefill** and **decode** phases.
    - Know the GPU facts that matter for inference: HBM vs on-chip SRAM, tensor cores, peak FLOP/s vs memory bandwidth.
    - Use **arithmetic intensity** and the **roofline model** to show that decode is bandwidth-bound and compute the batch size at which it becomes compute-bound.
    - Define and measure TTFT, TPOT/ITL, end-to-end latency, throughput, goodput and SLOs.
    - Implement temperature, top-k, top-p, min-p and repetition penalty, and explain how constrained (structured) decoding works.

## Why this matters / the problem

A modern GPU such as the NVIDIA H100 SXM can perform roughly $10^{15}$ BF16 floating-point operations per second. An 8-billion-parameter model needs about $2 \times 8\times10^{9} = 1.6\times 10^{10}$ FLOPs to produce one token. Dividing, you might expect about 60,000 tokens per second for a single user. In reality, a single user gets on the order of a hundred or two tokens per second. Where did a factor of several hundred go?

The answer is that generating text is limited not by arithmetic but by **moving bytes from GPU memory to the compute units**. Understanding this one fact — and being able to quantify it — explains why serving engines look the way they do: why they batch requests aggressively, why they obsess over the KV cache, why quantization speeds things up, and why speculative decoding works. This chapter gives you the vocabulary and the arithmetic used throughout Part 3.

## Intuition

Think of a GPU as a giant kitchen with thousands of very fast cooks (the compute units) and one narrow door to the pantry (memory bandwidth). To make one dish (one token for one user) the cooks need *every* ingredient in the pantry — all the model's weights. If only one dish is ordered, the cooks spend almost all their time waiting at the door for ingredients and only a moment cooking. If 200 dishes that use the same ingredients are ordered together, each ingredient carried through the door is used 200 times, and the cooks finally become busy.

That is the whole story of LLM inference in one picture:

- **Prefill** (reading the prompt) is like a 2,000-dish order: all prompt tokens are processed together, so each weight is reused many times. The cooks are busy → *compute-bound*.
- **Decode** (writing the answer) produces one token per step, because token $t+1$ depends on token $t$. With a single user, each weight is fetched to make one dish → *memory-bandwidth-bound*.
- **Batching** many users' decode steps together recovers the reuse.

## How it works

### The autoregressive loop

A language model defines $p_\theta(x_{t+1} \mid x_1, \dots, x_t)$: given the tokens so far, a probability distribution over the next token from a vocabulary of size $V$. Generation repeats three steps:

1. Run the model on the current sequence to get **logits** $z \in \mathbb{R}^{V}$ for the next position.
2. Turn logits into a token with a **sampling rule** (greedy argmax, or random sampling with temperature, top-p, …).
3. Append the token and repeat, until an end-of-sequence (EOS) token or a length limit.

Naively, step 1 reprocesses the whole sequence every time. Chapter 3.2 shows that by caching each layer's keys and values (the **KV cache**), each new step only has to process the *one* new token. With a KV cache, generation splits into two phases:

| Phase | Input per sequence | What happens | Bottleneck | Latency metric |
|---|---|---|---|---|
| **Prefill** | the whole prompt, $L_{\text{in}}$ tokens, in parallel | compute keys/values for every prompt token, produce the first output token | compute (big matrix–matrix multiplies) | **TTFT** |
| **Decode** | 1 new token | read all weights + the KV cache, append one K/V entry, produce the next token | memory bandwidth (matrix–vector multiplies) | **TPOT / ITL** |

### GPU basics you actually need

A GPU has a **memory hierarchy**: a small amount of very fast memory close to the compute units and a large amount of slower memory further away. For an H100 SXM (all figures approximate, from NVIDIA's public specifications):

| Level | Size | Bandwidth / speed | What lives there |
|---|---|---|---|
| Registers | 256 KB per SM | fastest | per-thread values |
| Shared memory / L1 (**SRAM**) | up to ~228 KB usable shared memory per SM, 132 SMs | very high (tens of TB/s aggregate) | tiles of matrices a kernel is working on |
| L2 cache | ~50 MB | high | recently used data |
| **HBM** (HBM3) | 80 GB | **≈ 3.35 TB/s** | weights, KV cache, activations |

An **SM** (streaming multiprocessor) is one of the GPU's ~132 independent processors. Each SM contains **tensor cores**: hardware units that perform small matrix multiplies (e.g. a 16×16 tile) in one instruction. Almost all of the GPU's FLOP/s come from tensor cores: an H100 SXM is rated at **≈ 989 dense BF16 TFLOP/s** and **≈ 1,979 dense FP8 TFLOP/s** on tensor cores, versus only ≈ 67 TFLOP/s of ordinary FP32 arithmetic. (NVIDIA's headline numbers are often twice these, because they assume 2:4 structured sparsity; we always use the dense numbers.) For comparison, an A100 SXM 80 GB has ≈ 2.0 TB/s of HBM bandwidth and ≈ 312 dense BF16 TFLOP/s.

The key ratio:

$$
\frac{\text{peak FLOP/s}}{\text{memory bandwidth}} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35\times 10^{12}\ \text{B/s}} \approx 295\ \text{FLOP/byte} \quad (\text{H100 SXM, BF16}).
$$

In words: for the H100 to be busy computing rather than waiting, a kernel must perform about **295 floating-point operations for every byte it reads from HBM**.

### Arithmetic intensity and the roofline model

The **arithmetic intensity** $I$ of a computation is

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to/from HBM}}.
$$

The **roofline model** says the best achievable throughput of a kernel is

$$
\text{attainable FLOP/s} = \min\big(P_{\text{peak}},\; I \cdot B_{\text{mem}}\big),
$$

where $P_{\text{peak}}$ is peak compute (FLOP/s) and $B_{\text{mem}}$ is memory bandwidth (bytes/s). Plotted on log–log axes (FLOP/s vs $I$) this is a sloped line ("bandwidth roof") that meets a flat line ("compute roof") at the **ridge point** $I^{\star} = P_{\text{peak}}/B_{\text{mem}}$. Kernels with $I < I^{\star}$ are **memory-bound**: making the arithmetic faster does not help, only reading fewer bytes does. Kernels with $I > I^{\star}$ are **compute-bound**.

```text
 FLOP/s
   ^                 compute roof (989 TFLOP/s)
   |            _______________________________
   |           /
   |          /  <- ridge point I* ≈ 295 FLOP/B
   |         /
   |        /   bandwidth roof: slope = 3.35 TB/s
   |  decode (B=1, I≈1)         prefill (I≈ thousands)
   +----------------------------------------------> arithmetic intensity (log)
```

Now apply it to the workhorse of a transformer: a linear layer $Y = XW$ with weights $W \in \mathbb{R}^{d \times d}$ stored in BF16 (2 bytes each) and an input $X \in \mathbb{R}^{B \times d}$ holding $B$ token vectors.

- FLOPs: each of the $B \cdot d$ outputs is a dot product of length $d$ (one multiply + one add per term), so $2Bd^2$ FLOPs.
- Bytes: read $W$ ($2d^2$), read $X$ ($2Bd$), write $Y$ ($2Bd$): total $2d^2 + 4Bd$.

$$
I(B) = \frac{2Bd^2}{2d^2 + 4Bd} \approx B \quad \text{when } B \ll d.
$$

In words: **the arithmetic intensity of a linear layer is roughly the number of tokens processed together.** In prefill, $B$ is the number of prompt tokens (hundreds to thousands) → compute-bound. In decode with one user, $B = 1$ → about 1 FLOP per byte, roughly 300× below the H100's ridge point. With $B$ users decoding together, $I \approx B$, so decode linear layers become compute-bound around $B \approx 295$ on an H100 in BF16.

!!! note "Attention during decode does not batch"
    The argument above is for weight matrices, which are **shared** across all sequences in the batch. During decode, attention reads each sequence's **own** KV cache, which is not shared. Its intensity stays near 1–2 FLOP/byte for standard multi-head attention (or about $g$ FLOP/byte with grouped-query attention, where $g$ query heads share one KV head) regardless of batch size. This is why long contexts make decode slower even at large batch, and why Chapter 3.2 is devoted to the KV cache.

### Latency metrics

For one request with $N$ output tokens:

- **TTFT** (time to first token): from request arrival to the first output token. Includes queueing + prefill. Users perceive this as "responsiveness".
- **ITL** (inter-token latency): the gap between consecutive output tokens; a distribution, one value per token.
- **TPOT** (time per output token): the average gap for one request, $\text{TPOT} = \dfrac{\text{E2E} - \text{TTFT}}{N - 1}$. Its inverse is the per-user streaming speed ("tokens/s per user").
- **E2E latency**: total time, $\text{E2E} = \text{TTFT} + (N-1)\cdot\text{TPOT}$.
- **Throughput**: system-wide output tokens/s (or requests/s), summed over all users.
- **SLO** (service-level objective): a target such as "P99 TTFT ≤ 2 s and P99 TPOT ≤ 50 ms". Always state percentiles; averages hide the tail.
- **Goodput**: the rate of requests that **meet** the SLO (DistServe's definition). A system can have high throughput and terrible goodput if it achieves the throughput by letting latency explode.

Throughput and latency trade off: bigger batches raise throughput (more reuse of each weight read) but each step does more work, so TPOT rises; and admitting new prefills interrupts running decodes. Every scheduling technique in Chapter 3.3 is about moving along or pushing out this trade-off curve.

### Sampling: from logits to a token

Given logits $z \in \mathbb{R}^V$, the model's distribution is $p_i = \operatorname{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$. Common transformations, applied in roughly this order:

- **Repetition penalty** (from the CTRL paper): for every token $i$ already present in the context, divide $z_i$ by $\theta > 1$ if $z_i > 0$, multiply by $\theta$ if $z_i < 0$. Both make previously seen tokens less likely. (OpenAI-style APIs instead *subtract* a *frequency penalty* times the count and a *presence penalty* if the count is non-zero.)
- **Temperature** $T$: use $\operatorname{softmax}(z/T)$. $T<1$ sharpens the distribution, $T>1$ flattens it, $T \to 0$ is greedy argmax.
- **Top-k**: keep only the $k$ highest-probability tokens, renormalize.
- **Top-p (nucleus)**: sort tokens by probability and keep the smallest prefix whose cumulative probability is $\ge p$. Adapts the number of candidates to how confident the model is.
- **Min-p**: keep tokens with $p_i \ge p_{\min} \cdot \max_j p_j$. The threshold scales with the top token's probability: strict when the model is confident, permissive when it is not.

Filtering is done by setting the removed logits to $-\infty$ before the final softmax and sampling.

### Structured / constrained decoding

Applications often need output that parses: valid JSON matching a schema, a function call, a regex. **Constrained decoding** guarantees it by masking, at every step, every token that would make the output impossible to complete validly:

1. Compile the constraint (regex, JSON schema, or context-free grammar) into an automaton — a finite-state machine for regular languages, a pushdown automaton for grammars with nesting.
2. At each step, compute the set of vocabulary tokens that keep the automaton in a valid state. This is subtle because tokens are multi-character strings: the token `"},` advances the automaton by three characters at once.
3. Set the logits of disallowed tokens to $-\infty$, sample, advance the automaton.

The cost is computing the allowed-token mask over a vocabulary of 100K+ tokens every step without slowing decode. Outlines precomputes an index from FSM states to allowed tokens; XGrammar splits the vocabulary into tokens whose validity can be precomputed per grammar position and a smaller set that must be checked at runtime, and overlaps mask computation with the GPU forward pass. Masking changes the distribution (it renormalizes over valid tokens), so constrained output can differ in quality from unconstrained output; good schemas leave the model room to "think" in free-text fields.

## Code

A self-contained generation loop with a toy causal transformer (random weights — the point is the loop structure, not the text). Note how naive it is: every step re-runs the model on the entire sequence. Chapter 3.2 fixes this.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TinyLM(nn.Module):
    """One-layer causal transformer: enough to show the generation loop."""
    def __init__(self, vocab=100, d=64, heads=4):
        super().__init__()
        self.emb, self.h = nn.Embedding(vocab, d), heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):                      # ids: [batch, seq]
        x = self.emb(ids)                        # [B, S, d]
        B, S, d = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # [B, heads, S, d_head]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, d))
        x = x + self.mlp(x)
        return self.head(x)                      # logits: [B, S, vocab]

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=10, eos_id=None):
    ids = prompt_ids                             # [1, L_in]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]            # only the last position matters
        next_id = logits.argmax(-1, keepdim=True)   # greedy; see sample() below
        ids = torch.cat([ids, next_id], dim=1)   # append and repeat
        if eos_id is not None and next_id.item() == eos_id:
            break
    return ids

torch.manual_seed(0)
model = TinyLM().eval()
print(generate(model, torch.tensor([[1, 2, 3]])))
```

A sampler implementing every rule from "How it works". It operates on one sequence's logits; real engines run the same logic batched on the GPU.

```python
import torch

def sample(logits, prev_ids=(), temperature=1.0, top_k=0, top_p=1.0,
           min_p=0.0, repetition_penalty=1.0, generator=None):
    """logits: [V] float tensor. Returns a sampled token id (int)."""
    z = logits.clone().float()
    # 1) repetition penalty (CTRL-style), applied to tokens already in context
    if repetition_penalty != 1.0 and len(prev_ids) > 0:
        idx = torch.tensor(sorted(set(prev_ids)))
        zi = z[idx]
        z[idx] = torch.where(zi > 0, zi / repetition_penalty, zi * repetition_penalty)
    # 2) temperature (T -> 0 means greedy)
    if temperature == 0:
        return int(z.argmax())
    z = z / temperature
    # 3) top-k
    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]
        z[z < kth] = float("-inf")
    probs = torch.softmax(z, dim=-1)
    # 4) min-p: drop tokens below min_p * (max prob)
    if min_p > 0:
        z[probs < min_p * probs.max()] = float("-inf")
        probs = torch.softmax(z, dim=-1)
    # 5) top-p (nucleus): keep the smallest set with cumulative prob >= top_p
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = cum - sp >= top_p          # tokens whose preceding mass already >= top_p
        z[si[drop]] = float("-inf")
        probs = torch.softmax(z, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))

logits = torch.tensor([3.0, 2.5, 1.0, 0.5, -1.0])
g = torch.Generator().manual_seed(0)
print([sample(logits, prev_ids=[0], temperature=0.7, top_p=0.9,
              repetition_penalty=1.3, generator=g) for _ in range(10)])
```

## Worked example

**Question.** Serve Llama-3-8B (≈ 8.0 B parameters, BF16) on one H100 SXM. (a) What is the best possible decode speed for a single user? (b) At what batch size do the linear layers become compute-bound? (c) How long does prefill of a 2,000-token prompt take?

**(a) Batch 1 decode.** Each decode step must read every weight once: $8.0\times10^9 \times 2\ \text{B} = 16\ \text{GB}$. At ≈ 3.35 TB/s:

$$
t_{\text{step}} \ge \frac{16 \times 10^9}{3.35 \times 10^{12}} \approx 4.8\ \text{ms} \;\Rightarrow\; \le \approx 209\ \text{tokens/s}.
$$

The compute needed is only $2 \times 8\times10^9 = 1.6\times10^{10}$ FLOPs, which takes $1.6\times10^{10}/989\times10^{12} \approx 0.016$ ms. The GPU's arithmetic is idle about 99.7% of the time. Real engines reach perhaps 70–85% of this bandwidth bound, and the KV cache adds more bytes.

**(b) Crossover batch size.** With $B$ users decoding together, the bytes for weights stay ≈ 16 GB per step, while FLOPs grow to $1.6\times10^{10} \cdot B$. The step time is roughly

$$
t(B) \approx \max\left(\frac{16\times10^9}{3.35\times10^{12}},\; \frac{1.6\times10^{10}\,B}{989\times10^{12}}\right),
$$

and the two terms are equal at $B \approx 4.8\ \text{ms} \times 989\times10^{12} / 1.6\times10^{10} \approx 297$. Below ~300 concurrent sequences, adding a user is nearly free in step time (throughput grows almost linearly with $B$); above it, step time grows with $B$. In practice the crossover is reached earlier, because real kernels do not hit peak FLOP/s and because KV-cache reads (which grow with $B$ and context length) add bytes. This is why serving engines try to run **as large a batch as memory allows**.

**(c) Prefill.** A 2,000-token prompt costs about $2 \times 8\times 10^9 \times 2000 = 3.2\times10^{13}$ FLOPs (ignoring attention, which is small at this length). At peak that is 32 ms; at a realistic 40–60% of peak, roughly 55–80 ms. Prefill of 2,000 tokens costs about as much GPU time as ~15 decode steps — but it produces only one output token. This asymmetry is why mixing prefills into a decode batch has to be handled carefully (Chapter 3.3, chunked prefill).

## In practice (2025–2026)

- Every serving engine (vLLM, SGLang, TensorRT-LLM) reports TTFT, TPOT/ITL and throughput, and their benchmark tools sweep concurrency or request rate (Chapter 3.8).
- The same bandwidth argument explains three industry trends: **FP8 and FP4 weights** (fewer bytes per weight, Chapter 3.6), **MoE models** (each token reads only the active experts' weights), and **GQA/MLA attention** (fewer KV bytes per token, Chapter 3.2).
- Production SLOs are typically stated as P90/P99 TTFT and TPOT per product tier: an interactive chat product cares about TTFT and a steady streaming speed; batch/offline jobs care only about throughput and cost.
- Structured output is a standard API feature (JSON mode / JSON-schema "structured outputs"); open engines ship grammar backends such as XGrammar, Outlines and llguidance.
- Default sampling for chat models is usually temperature around 0.6–1.0 with top-p around 0.9–0.95; reasoning-model providers often publish recommended settings for their models — follow them, because these models can loop or degrade with greedy decoding.

## Common pitfalls / misconceptions

- **"Faster GPUs (more FLOP/s) make single-user decode faster."** Mostly not: single-user decode is bounded by memory bandwidth. Compare GPUs on TB/s for latency and on FLOP/s for prefill and large-batch throughput.
- **Confusing TPOT and ITL.** TPOT is a per-request average; ITL is the distribution of individual gaps. A system can have a fine mean TPOT and terrible P99 ITL because of periodic stalls (e.g. a long prefill blocking decodes).
- **Reporting throughput without latency.** Throughput can always be raised by increasing batch size until latency is unacceptable. Report goodput, or throughput at a stated SLO.
- **Using sparse TFLOP/s in roofline calculations.** Dense matmuls cannot use 2:4 sparsity throughput.
- **Applying top-p before temperature (or vice versa) without knowing which.** Different libraries use different orders; results differ. Know your engine's order.
- **Thinking constrained decoding is free.** Mask computation on large vocabularies and complex grammars can become a CPU bottleneck if not overlapped with the GPU.

## Key takeaways

- Generation = one parallel **prefill** over the prompt, then a sequential **decode** loop of one token per step.
- Arithmetic intensity of a linear layer ≈ number of tokens processed together. Decode at batch 1 is ≈ 1 FLOP/byte, far below the H100's ≈ 295 FLOP/byte ridge point → **bandwidth-bound**.
- Batch-1 decode speed ≤ bandwidth ÷ bytes read per step; batching amortizes weight reads until roughly $B \approx P_{\text{peak}}/B_{\text{mem}}$.
- Measure TTFT, TPOT/ITL, E2E and throughput **at percentiles**, and optimize goodput under an SLO.
- Sampling rules (temperature, top-k/p, min-p, penalties) and grammar masks are all logit transformations before a final softmax-and-sample.

## Check your understanding

??? question "An engine upgrades from A100 (≈2.0 TB/s, ≈312 BF16 TFLOP/s) to H100 SXM (≈3.35 TB/s, ≈989 TFLOP/s). Roughly how much faster does batch-1 decode get? Large-batch prefill?"
    Batch-1 decode is bandwidth-bound, so about 3.35/2.0 ≈ 1.7×. Prefill is compute-bound, so up to about 989/312 ≈ 3.2× (in practice somewhat less). Note that the H100's ridge point (~295 FLOP/B) is higher than the A100's (~156 FLOP/B): the newer GPU needs *larger* batches to be compute-bound.

??? question "Why is the arithmetic intensity of decode attention not improved by batching, while that of the MLP is?"
    The MLP's weight matrix is shared by every sequence in the batch, so one read of it serves $B$ tokens. In attention, each sequence attends over its *own* keys and values; there is nothing to share across sequences, so each KV byte read is used for only one query (or $g$ queries with GQA).

??? question "A request produced 201 tokens; TTFT was 0.5 s and E2E was 10.5 s. What is TPOT and the per-user streaming speed?"
    TPOT = (10.5 − 0.5)/(201 − 1) = 0.05 s = 50 ms, so about 20 tokens/s per user.

??? question "With min-p = 0.1, the top token has probability 0.6. Which tokens survive? And if the top token has probability 0.05?"
    Tokens with $p_i \ge 0.06$ survive in the first case. In the second, the threshold is 0.005, so many more tokens survive — min-p widens the candidate set exactly when the model is uncertain.

??? question "Why must a JSON constraint be enforced at the token level rather than by checking the finished string?"
    Checking afterwards would require rejecting and regenerating (unbounded cost, no guarantee). Masking per step guarantees every prefix stays completable, so the final output is valid by construction. It must be token-level because the model emits multi-character tokens, so each token must be checked against the automaton as a string.

## Further reading

| Paper / resource | Why read it |
|---|---|
| :material-star: [LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook) | Practitioner-oriented overview of every concept in Part 3, with metric definitions. |
| :material-star: [Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/) | The classic back-of-the-envelope treatment of FLOPs, bytes and latency for transformer inference. |
| :material-book-open-variant: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Pope et al., 2022 | Rigorous analysis of memory/compute/communication trade-offs for inference at scale (PaLM on TPUs). |
| :material-book-open-variant: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al., 2019 | Introduces nucleus (top-p) sampling and explains why greedy/beam decoding degenerates. |
| :material-book-open-variant: [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082) — Nguyen et al., 2024 | The min-p rule and its evaluation at high temperatures. |
| :material-book-open-variant: [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) — Keskar et al., 2019 | Source of the multiplicative repetition penalty used by many engines. |
| :material-book-open-variant: [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) — Willard & Louf, 2023 | The Outlines approach: regex/grammar → FSM → precomputed token index. |
| :material-new-box: [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100) — Dong et al., 2024 | Fast context-free-grammar decoding used by vLLM and SGLang. |
