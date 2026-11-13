# 3.8 Engines & Production Practice

!!! abstract "What you'll learn"
    - What vLLM, SGLang, TensorRT-LLM, llama.cpp, MLC-LLM and TGI are, how they are built, and when to pick each.
    - How to **benchmark** a serving system honestly: workload shapes, concurrency sweeps, TTFT/TPOT/ITL percentiles, goodput — with a working benchmark client.
    - How to do **capacity planning**: GPUs needed for a target QPS under latency SLOs, using Little's law and a measured operating point.
    - How to compute **cost per million tokens** and what drives it.
    - What changes when serving **reasoning models** and when serving **RL rollouts** (weight sync, asynchronous rollouts, determinism and batch invariance).

## Why this matters / the problem

Chapters 3.1–3.7 explained the mechanisms. In practice you rarely implement them yourself: you pick an engine, configure it, measure it, and decide how many GPUs to buy or rent. Most real-world mistakes happen here — benchmarks that do not resemble production traffic, capacity plans built on average latency, cost estimates that assume 100% utilization, or RL pipelines whose inference engine silently disagrees with the trainer. This chapter is the bridge from understanding to operating.

## Intuition

An inference engine is a **runtime** (scheduler + memory manager + kernels, Chapters 3.3–3.4) wrapped in an **API server**. Engines differ mostly in (1) which hardware they target, (2) how much scheduling overhead they impose on the CPU, (3) which models and features (quantization formats, speculative methods, parallelism modes) are ready to use, and (4) how easy they are to modify.

A benchmark is an **experiment**: the workload is the independent variable, latency percentiles and throughput are the dependent variables. A capacity plan is the experiment's result divided into your demand, plus headroom. Cost is GPU-hours divided by tokens served.

## How it works

### A tour of the engines

| Engine | What it is | Pick it when |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Python/PyTorch engine that introduced PagedAttention; continuous batching, prefix caching, chunked prefill, many quantization and speculative methods, TP/PP/EP, OpenAI-compatible server (`vllm serve`). Its V1 engine (default in 2025) separates the API front end from a busy-looping engine core to cut CPU overhead. | You want the broadest model and hardware support (NVIDIA, AMD, TPU and others via plugins), a large community, and an easy path from Hugging Face checkpoints. A safe default. |
| [SGLang](https://github.com/sgl-project/sglang) | Python engine built around RadixAttention prefix caching, a low-overhead scheduler that overlaps CPU scheduling with GPU work, fast grammar-constrained decoding, and strong large-MoE support (DeepSeek-style EP, PD disaggregation). | Prefix-heavy workloads (agents, multi-turn, few-shot), structured output, large MoE deployments, RL rollouts (widely embedded in RL frameworks). |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA's open-source LLM runtime with hand-optimized kernels, in-flight batching, FP8/FP4 on Hopper/Blackwell; often deployed behind Triton Inference Server or Dynamo. | NVIDIA-only fleets where peak performance on the newest GPUs matters and you can absorb a more involved build/config process. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | C/C++ inference with the GGUF format and many quantization types; runs on CPUs, Apple Silicon (Metal), CUDA, Vulkan and more. Ollama wraps it with model download and a local API. | Laptops, desktops, edge devices, CPU servers, single-user local use. Not designed for high-concurrency datacenter serving. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Compiler-based deployment (Apache TVM): compiles models for many backends, including phones and browsers via WebGPU. | On-device and in-browser inference, unusual hardware targets. |
| [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | Production server with a Rust router and model shards, integrated with the HF ecosystem. | You are already deep in HF tooling; check current project activity and feature coverage against vLLM/SGLang before adopting. |

Engines move fast — features that are "SGLang-only" in one quarter are in vLLM the next and vice versa. Decide with a benchmark on your model, hardware and traffic, not with a feature list.

### The configuration knobs that matter

Every engine exposes the mechanisms of Chapters 3.2–3.7 as settings. The names below are vLLM's (`vllm serve --help`); SGLang and TensorRT-LLM have equivalents under other names.

| Setting (vLLM) | Mechanism | Effect of raising it |
|---|---|---|
| `--tensor-parallel-size` | TP degree (3.7) | more aggregate bandwidth and KV room; more all-reduce overhead |
| `--max-model-len` | maximum context per request (3.2) | supports longer requests; fewer may fit concurrently |
| `--gpu-memory-utilization` | fraction of GPU memory the engine may use; what is left after weights becomes the KV pool (3.3) | more KV blocks → larger batches; less headroom for other processes |
| `--max-num-seqs` | maximum sequences per batch (3.3) | higher throughput until KV runs out; higher TPOT |
| `--max-num-batched-tokens` | chunked-prefill token budget per step (3.3) | lower TTFT for long prompts; higher ITL for running decodes |
| `--enable-prefix-caching` | hash-based prefix cache (3.3) | lower TTFT and prefill cost when prompts share prefixes |
| `--kv-cache-dtype fp8` | KV quantization (3.2, 3.6) | ~2× KV capacity; small accuracy risk |
| `--quantization` | weight/activation format, e.g. FP8 or AWQ (3.6) | less memory and bandwidth; accuracy must be checked |
| `--speculative-config` | draft method and number of speculative tokens (3.5) | lower latency at low load; can hurt at high load |

### Benchmarking correctly

**1. Define the workload shape.** Measure or assume distributions, not single values:

- input length and output length distributions (e.g. chat: ~1–2K in / ~300 out; RAG: ~8K in / ~200 out; reasoning: ~1K in / ~8K+ out);
- **prefix structure** — how much of each prompt is shared (system prompt, documents, history), which decides prefix-cache hit rates;
- **arrival process** — open-loop (requests arrive at a rate, e.g. Poisson) or closed-loop (a fixed number of concurrent users who send a new request when the last finishes).

**2. Sweep load.** Run at increasing concurrency (closed-loop) or request rate (open-loop): e.g. 1, 4, 16, 64, 128, 256. At each point record

- **TTFT** P50/P90/P99, **TPOT** P50/P99, **ITL** P99 (stalls), **E2E** P99;
- output tokens/s and requests/s;
- **goodput**: requests/s meeting all SLOs.

Plot latency percentiles against throughput. The result is a curve: latency is flat at first, then bends sharply upward as the system saturates. The usable operating point is the highest throughput whose P99 still meets the SLO.

**3. Control the variables.** Warm up the server before measuring (CUDA graph capture, caches); measure on the client (what users experience, including network and detokenization); fix output lengths if comparing engines (e.g. an `ignore_eos` option, stated explicitly) or use real outputs; use the same tokenizer and the same sampling settings; repeat runs. Engines ship tools for this — `vllm bench serve` and SGLang's `python -m sglang.bench_serving` sweep request rates or concurrency on synthetic or dataset prompts (check `--help` for the current flags).

### Capacity planning

**Little's law** relates load and latency for any stable system: the average number of requests in the system $N$ equals the arrival rate $\lambda$ times the average time each spends in it $W$,

$$
N = \lambda \cdot W .
$$

In words: if 20 requests arrive per second and each takes 14 s end to end, about 280 are in flight at any moment. This tells you the concurrency your fleet must sustain. The procedure:

1. From the benchmark, find the **operating point** of one replica: the largest concurrency $C_{\text{rep}}$ (or throughput $X_{\text{rep}}$ output tokens/s) at which P99 TTFT and TPOT meet the SLO, *on your traffic shape* (which includes the prefill load).
2. Compute the demand: output tokens/s $= \lambda \cdot \bar N_{\text{out}}$, and in-flight requests $= \lambda \cdot W$ with $W \approx \text{TTFT} + \bar N_{\text{out}} \cdot \text{TPOT}$.
3. Replicas needed $= \max\big(\text{demand tokens/s} / X_{\text{rep}},\ \text{in-flight} / C_{\text{rep}}\big)$, rounded up, plus **headroom** (often 20–50%) for peaks, failures and deploys.
4. Sanity-check KV memory per replica (Chapter 3.2) for the in-flight requests at their lengths.

### Cost per million tokens

$$
\text{cost per 1M tokens} = \frac{\text{USD per GPU-hour} \times \text{GPUs}}{\text{tokens served per hour} / 10^6} .
$$

In words: what you pay per hour divided by how many millions of tokens you actually serve per hour. Three things dominate it: the per-GPU throughput **at your SLO** (tighter latency → smaller batches → higher cost), the **utilization** of the fleet over the day (idle GPUs still cost money), and how input vs output tokens are counted (prefill tokens are much cheaper per token than decode tokens, which is why API prices for input tokens are lower — and cached input tokens lower still).

### Serving reasoning models

Reasoning models emit thousands to tens of thousands of "thinking" tokens per answer. Consequences:

- **Decode dominates.** At 30 ms TPOT, a 10,000-token trace takes 5 minutes. TPOT, not TTFT, decides user-perceived latency, and every decode optimization (FP8/FP4 weights, speculative decoding, MLA/GQA) matters more.
- **KV grows during generation.** Long outputs consume KV memory over minutes; concurrency must be planned for the *final* length, and preemption (Chapter 3.3) becomes common if you over-admit.
- **Long-tail lengths** make static capacity estimates unreliable; plan with output-length distributions, not means.
- **Product knobs:** maximum thinking budget, effort levels, and early exit trade quality for cost; engines provide parsers that separate reasoning content from the final answer in the API response.

### Serving for RL rollouts

In RL post-training (Part 2), most wall-clock time is spent **generating rollouts**. Frameworks such as [verl](https://github.com/volcengine/verl) and [slime](https://github.com/THUDM/slime) embed vLLM or SGLang as the rollout engine. Three issues are specific to this use:

- **Weight sync.** After each optimizer step, the inference engine must receive the new weights: broadcast over NCCL from trainer to inference GPUs, or, when training and inference share the same GPUs ("colocated"), by swapping memory between the two (engines offer modes to release and re-acquire GPU memory). Syncing a 70B BF16 model moves ≈ 140 GB, so sync time matters.
- **Long tails and asynchrony.** In a synchronous step the slowest rollout (the longest reasoning trace) holds up the whole batch. **Asynchronous** pipelines let generation of the next batch overlap training on the current one, and **partial rollouts** pause very long generations and resume them later. Both make the data slightly **off-policy** (generated by older weights), which the RL algorithm must tolerate or correct.
- **Train–inference mismatch and determinism.** The rollout engine computes token probabilities with different kernels (and often lower precision) than the trainer, so the "same" policy gives slightly different log-probabilities, which can destabilize RL. A root cause, analyzed in Thinking Machines' *Defeating Nondeterminism in LLM Inference*, is lack of **batch invariance**: floating-point addition is not associative, and kernels choose different reduction strategies (split-K matmuls, split-KV attention, different tile sizes) depending on batch size and sequence length. So a request's numerics depend on what else is in the batch. Batch-invariant kernels fix the reduction order regardless of batch composition, giving reproducible outputs at some performance cost.

## Code

A minimal async benchmark client for any OpenAI-compatible server (vLLM, SGLang, TensorRT-LLM via its OpenAI server, TGI). It runs a closed-loop concurrency level and reports output throughput, TTFT/TPOT/ITL percentiles and goodput. Run it at several concurrency levels to trace the latency–throughput curve.

```python
import asyncio, time, statistics
from openai import AsyncOpenAI     # pip install openai; works with any OpenAI-compatible server

async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    first, last, gaps, n_out = None, None, [], 0
    stream = await client.completions.create(
        model=model, prompt=prompt, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True})
    async for chunk in stream:
        now = time.perf_counter()
        if chunk.choices and chunk.choices[0].text:
            if first is None:
                first = now                                  # first token arrived
            else:
                gaps.append(now - last)                      # inter-token (chunk) latency
            last = now
        if chunk.usage:                                      # final chunk carries counts
            n_out = chunk.usage.completion_tokens
    e2e = time.perf_counter() - t0
    ttft = first - t0
    tpot = (e2e - ttft) / max(n_out - 1, 1)
    return dict(ttft=ttft, tpot=tpot, e2e=e2e, n_out=n_out, gaps=gaps)

def pct(xs, p):
    return statistics.quantiles(xs, n=100, method="inclusive")[p - 1]

async def run(base_url, model, prompts, concurrency, max_tokens=300,
              slo_ttft=2.0, slo_tpot=0.05):
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)                     # closed-loop: N in flight
    async def guarded(p):
        async with sem:
            return await one_request(client, model, p, max_tokens)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(p) for p in prompts))
    wall = time.perf_counter() - t0
    ttft, tpot = [r["ttft"] for r in res], [r["tpot"] for r in res]
    itl = [g for r in res for g in r["gaps"]]
    good = sum(r["ttft"] <= slo_ttft and r["tpot"] <= slo_tpot for r in res)
    print(f"concurrency={concurrency:4d}  out tok/s={sum(r['n_out'] for r in res) / wall:8.0f}  "
          f"TTFT p50/p99={pct(ttft, 50):.2f}/{pct(ttft, 99):.2f}s  "
          f"TPOT p50/p99={1000 * pct(tpot, 50):.0f}/{1000 * pct(tpot, 99):.0f}ms  "
          f"ITL p99={1000 * pct(itl, 99):.0f}ms  goodput={good / wall:.2f} req/s")

# Example sweep against a local server started with e.g. `vllm serve <model>`:
# prompts = [open("my_real_prompts.txt").read()] * 200      # use YOUR traffic shape
# for c in (1, 4, 16, 64, 128):
#     asyncio.run(run("http://localhost:8000/v1", "<model>", prompts, c))
```

Why batch invariance is a real problem, in four lines: the same numbers summed in a different order give a different float32 result, and kernels change their summation order with batch size.

```python
import numpy as np
x = np.random.default_rng(0).standard_normal(4096).astype(np.float32) * 1e3
one_pass = x.sum(dtype=np.float32)                                   # one reduction order
split_4 = sum(c.sum(dtype=np.float32) for c in np.split(x, 4))       # "split-K" with 4 chunks
print(one_pass, split_4, "equal:", one_pass == split_4)
```

## Worked example

!!! note "Illustrative numbers"
    The benchmark results and GPU price below are **hypothetical inputs** chosen to be plausible; the method is the point. Replace them with your own measurements and prices.

**Goal.** Serve Llama-3-8B (BF16) for a chat product. Demand: $\lambda = 20$ requests/s at peak; average prompt 1,000 tokens (no shared prefix), average output 300 tokens. SLO: P99 TTFT ≤ 1 s, P99 TPOT ≤ 50 ms.

**Step 1 — benchmark one replica (1× H100)** with this traffic shape. Suppose the sweep shows that at concurrency 128 the replica delivers **2,800 output tokens/s** with P99 TPOT 45 ms and P99 TTFT 0.8 s, and that at concurrency 192 P99 TPOT exceeds 50 ms. Operating point: $C_{\text{rep}} = 128$, $X_{\text{rep}} = 2{,}800$ tokens/s. (Consistency check: 128 sequences ÷ 0.045 s per token ≈ 2,850 tokens/s.)

**Step 2 — demand.**

- Output tokens/s: $20 \times 300 = 6{,}000$.
- Time in system: $W \approx 0.5\ \text{s} + 299 \times 0.045\ \text{s} \approx 14\ \text{s}$; in flight: $N = 20 \times 14 = 280$ requests.

**Step 3 — replicas.** By throughput, $6{,}000 / 2{,}800 = 2.14$; by concurrency, $280 / 128 = 2.19$. Round up to 3, and with ~30% headroom ($2.19 \times 1.3 = 2.85$) 3 replicas still suffice. **3 H100s** at peak.

**Step 4 — KV check.** 280 in flight / 3 replicas ≈ 93 sequences per GPU at up to 1,300 tokens each, at 128 KiB/token (Chapter 3.2): $93 \times 1300 \times 128\ \text{KiB} \approx 14.8$ GiB — well within the ≈ 50 GiB KV budget. Prefill load: $20 \times 1000 = 20{,}000$ prompt tokens/s, ≈ 6,700 per replica — a substantial share of each GPU's time, which is why the benchmark had to use realistic prompt lengths.

**Step 5 — cost.** Assume a price of USD 2.50 per GPU-hour. At peak, 3 GPUs cost USD 7.50 per hour and serve $6{,}000 \times 3600 = 21.6$ M output tokens/hour (plus 72 M input tokens/hour):

$$
\frac{7.50\ \text{USD/hour}}{21.6\ \text{M tokens/hour}} \approx 0.35\ \text{USD per million output tokens (at peak utilization)}.
$$

If average traffic over the day is 40% of peak but you keep 3 GPUs running, the effective cost is $0.35 / 0.4 \approx 0.87$ USD per million output tokens — utilization matters as much as kernel speed. Autoscaling (Chapter 3.7), FP8 weights (fewer bytes → larger batches at the same TPOT), and prefix caching (less prefill) are the main levers to push this down; relaxing the TPOT SLO to 80 ms would also allow bigger batches and lower cost per token.

## In practice (2025–2026)

- Most open-model deployments use **vLLM or SGLang**; NVIDIA-centric teams chasing peak performance on the newest GPUs use **TensorRT-LLM** (often via Dynamo); local and edge use is dominated by **llama.cpp/Ollama** and Apple-Silicon-specific runtimes.
- Benchmark tooling is built into the engines (`vllm bench serve`, `sglang.bench_serving`), and independent benchmark efforts publish cross-engine comparisons — read their workload definitions before trusting their rankings.
- Serving stacks expose Prometheus metrics (queue length, KV utilization, prefix-cache hit rate, speculative acceptance rate, TTFT/TPOT histograms); these are the inputs to autoscaling and SLO alerts.
- Reasoning models made **TPOT** and **long-output KV capacity** the central serving metrics, and made speculative decoding and FP8/FP4 standard.
- RL frameworks (verl, slime and others) treat the rollout engine as a first-class component; asynchronous and partial rollouts, fast weight sync and batch-invariant / deterministic inference were active engineering topics through 2025.

!!! tip "A pre-launch checklist"
    - The startup log's KV capacity (tokens or blocks) supports your planned concurrency at realistic lengths.
    - A concurrency sweep on **your** traffic shape shows the SLO is met at peak demand plus headroom.
    - Quality of the served configuration (quantization, KV dtype, speculation) was evaluated against a BF16 baseline on your tasks.
    - Dashboards and alerts exist for TTFT/TPOT percentiles, queue length, KV utilization, preemptions and prefix-cache hit rate.
    - You know the cold-start time of a new replica and have warm capacity for at least that long.

## Common pitfalls / misconceptions

- **Benchmarking with one prompt length and one output length.** Real traffic has distributions and long tails; P99 is driven by the tail.
- **Reporting max throughput without latency.** It describes a system nobody would deploy. Report throughput at an SLO, or goodput.
- **Server-side latency only.** Users experience client-side TTFT including queueing, network and detokenization.
- **Mistaking closed-loop for open-loop results.** A closed-loop benchmark slows its own arrivals when the server is slow, hiding overload; production traffic is closer to open-loop.
- **Cost estimates at 100% utilization.** Divide by real average utilization.
- **Assuming identical outputs across batch sizes or engines.** Without batch-invariant kernels, the same prompt can produce different tokens depending on concurrent load — important for evals, debugging and RL.

## Key takeaways

- Choose engines by benchmark on your model, hardware and traffic; vLLM and SGLang cover most datacenter needs, TensorRT-LLM targets peak NVIDIA performance, llama.cpp/MLC target local and edge.
- Benchmark with realistic workload shapes, sweep load, and report TTFT/TPOT/ITL percentiles, throughput and goodput.
- Capacity = demand (tokens/s and in-flight requests via Little's law) ÷ per-replica operating point at the SLO, plus headroom, checked against KV memory.
- Cost per million tokens = (USD per hour) ÷ (millions of tokens per hour); SLO tightness and fleet utilization are as important as raw speed.
- Reasoning models make decode and KV capacity dominant; RL rollouts add weight sync, asynchrony and numerical-consistency (batch invariance) requirements.

## Check your understanding

??? question "Your benchmark shows 10,000 output tokens/s at concurrency 512 with P99 TPOT of 140 ms. Your SLO is P99 TPOT ≤ 50 ms. Which number do you use for capacity planning?"
    Neither directly: find the highest concurrency whose P99 TPOT is ≤ 50 ms in the sweep and use the throughput measured there. The 10,000 tokens/s point violates the SLO.

??? question "λ = 5 requests/s, average E2E latency 40 s. How many requests are in flight on average?"
    By Little's law $N = 5 \times 40 = 200$.

??? question "A GPU costs USD 3 per hour and sustains 2,000 output tokens/s at your SLO. What is the cost per million output tokens at full utilization? At 50%?"
    $2000 \times 3600 = 7.2$ M tokens/hour → $3 / 7.2 \approx 0.42$ USD per million; at 50% utilization ≈ 0.83 USD.

??? question "Why does a closed-loop benchmark with 64 users under-report overload compared with open-loop traffic at the same average rate?"
    In closed loop each user waits for a response before sending the next request, so when the server slows the arrival rate automatically drops. Open-loop arrivals keep coming regardless, so queues (and TTFT) grow without bound once capacity is exceeded.

??? question "Why can the same prompt, sampled greedily, give different outputs when the server is busy vs idle?"
    The batch composition changes, so kernels may pick different reduction strategies (split-K, split-KV, tile sizes); floating-point non-associativity changes logits slightly, and a near-tie in the argmax can flip, after which the generations diverge. Batch-invariant kernels prevent this.

## Further reading

| Paper / resource | Why read it |
|---|---|
| :material-star: [vLLM](https://github.com/vllm-project/vllm) | De-facto open standard engine: PagedAttention, continuous batching, prefix caching, chunked prefill; V1 engine. |
| :material-star: [SGLang](https://github.com/sgl-project/sglang) | RadixAttention, fast structured output, large-scale EP and PD disaggregation; popular for RL rollouts. |
| :material-book-open-variant: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA-optimized kernels, FP8/FP4 on Hopper/Blackwell, in-flight batching. |
| :material-book-open-variant: [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | CPU/Apple Silicon/edge inference with GGUF quantization. |
| :material-book-open-variant: [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Compiler-based deployment to phones, browsers (WebGPU) and diverse GPUs. |
| :material-book-open-variant: [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | Production server integrated with the HF ecosystem. |
| :material-star: [Defeating Nondeterminism in LLM Inference (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Why inference is not batch-invariant and how to make it so; essential for RL and reproducible evals. |
| :material-book-open-variant: [verl](https://github.com/volcengine/verl) | RL post-training framework that embeds vLLM/SGLang as the rollout engine. |
| :material-book-open-variant: [slime](https://github.com/THUDM/slime) | RL framework built around SGLang rollouts and fast weight sync. |
| :material-book-open-variant: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | ~1,200-line readable engine: the best way to see all of Part 3 in real code. |
| :material-book-open-variant: [LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook) | Practical guidance on metrics, benchmarking and deployment choices. |
| :material-book-open-variant: [CMU 15-442/642 Machine Learning Systems](https://mlsyscourse.org/) | University course covering the systems side of ML, including serving. |
| :material-book-open-variant: [GPU MODE lectures](https://github.com/gpu-mode/lectures) | Community lectures on GPU programming and inference kernels. |
