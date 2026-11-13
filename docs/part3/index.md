# Part 3 · Inference Systems & Infrastructure

Training a model happens once; serving it happens billions of times. Part 3 is about the second half of a model's life: how a trained transformer is turned into a service that answers many users at once, fast, and at a cost that makes business sense. You will see that LLM serving is above all a **memory problem**. A GPU can do far more arithmetic per second than it can move bytes from memory, and generating text one token at a time keeps forcing it to move bytes. Almost every technique in this part either *reads fewer bytes* (quantization, smaller KV caches, MoE) or *does more useful work per byte read* (batching, speculative decoding, fused kernels).

We build the topic from the ground up. Chapter 3.1 writes the generation loop in a dozen lines of Python and then explains, using the roofline model, why that loop runs at a tiny fraction of the GPU's peak. Chapters 3.2–3.4 cover the three ideas that every modern engine is built on: the KV cache, paged memory with continuous batching, and IO-aware attention kernels such as FlashAttention. Chapters 3.5–3.6 cover the two biggest "free lunches" — speculative decoding and quantization — including the proofs and the error analysis behind them. Chapter 3.7 scales out to many GPUs and many machines (tensor/expert parallelism, prefill/decode disaggregation, trillion-parameter MoE serving), and Chapter 3.8 closes with how to choose an engine, benchmark it honestly, plan capacity, and compute cost per million tokens.

This matters even if you never write a CUDA kernel. Inference cost decides which models are economically deployable, how long a reasoning model may "think", how fast an RL training loop can generate rollouts, and what latency your users experience. Being able to do the back-of-the-envelope arithmetic in this part is one of the most useful skills in applied LLM work.

## Prerequisites and time

- **Math:** linear algebra (matrix multiply shapes and FLOP counts), basic probability (distributions, expectation, sampling).
- **Code:** Python and basic PyTorch / NumPy.
- **Models:** the transformer architecture from [Part 1](../part1/index.md) — attention, multi-head / grouped-query attention, the MLP block, MoE. You do not need Part 2.
- **Time:** roughly 15–25 hours to read every chapter and run the code; add 10–20 hours for the hands-on projects.

## Chapter map

| # | Chapter | One-line description |
|---|---|---|
| 3.1 | [Inference fundamentals](01-inference-fundamentals.md) | The generation loop, prefill vs decode, GPU memory hierarchy, roofline model, latency metrics, sampling and constrained decoding. |
| 3.2 | [The KV cache](02-kv-cache.md) | Why we cache keys and values, the KV size formula, how it limits batch size, and how to shrink it (GQA/MLA, quantization, eviction). |
| 3.3 | [Batching & memory management](03-batching-and-memory-management.md) | Continuous batching, PagedAttention block tables, copy-on-write, chunked prefill, prefix caching / RadixAttention, scheduling and preemption. |
| 3.4 | [Attention kernels](04-attention-kernels.md) | FlashAttention's tiling and online softmax (with a NumPy implementation), Flash-Decoding, FlashInfer, CUDA graphs, kernel fusion. |
| 3.5 | [Speculative decoding](05-speculative-decoding.md) | Draft-then-verify, the rejection rule and its losslessness proof, expected speedup, Medusa / EAGLE / MTP drafts, tree verification. |
| 3.6 | [Quantization](06-quantization.md) | Number formats bit by bit, quantization math, granularity, outliers, GPTQ / AWQ / SmoothQuant, FP8 and FP4, how to evaluate. |
| 3.7 | [Distributed & disaggregated serving](07-distributed-and-disaggregated-serving.md) | Tensor / pipeline / expert parallelism, prefill–decode disaggregation, KV-cache-centric systems, large-scale MoE serving, routing and autoscaling. |
| 3.8 | [Engines & production practice](08-engines-and-production-practice.md) | vLLM, SGLang, TensorRT-LLM, llama.cpp and friends; honest benchmarking; capacity planning; cost per token; reasoning and RL rollout serving. |

## Mental model: the life of a request

Every chapter in this part optimizes some box in the picture below. Keep it in mind as a map.

```text
 client ──HTTP/gRPC──▶ ROUTER / LOAD BALANCER ──────────────────────────────┐
  "Explain RoPE"        picks a replica (least-loaded,                       │
                        or prefix-aware: where is my KV cached?)  [3.7]      │
                                                                             ▼
                      ┌──────────────────── ONE ENGINE REPLICA ─────────────────────┐
                      │ tokenizer ─▶ waiting queue                                  │
                      │                 │                                           │
                      │                 ▼                                           │
                      │ SCHEDULER (runs every step)                       [3.3]     │
                      │   - admit requests if KV blocks are free                    │
                      │   - reuse cached prefix blocks (radix tree)                 │
                      │   - mix prefill chunks + decode tokens in one batch         │
                      │   - preempt (swap / recompute) when memory runs out         │
                      │                 │                                           │
                      │                 ▼                                           │
                      │ MODEL EXECUTOR (GPU, possibly TP/EP across GPUs)  [3.4,3.7] │
                      │   PREFILL: whole prompt in parallel ─▶ writes KV cache      │
                      │            compute-bound; sets TTFT                 [3.1]   │
                      │   DECODE:  1 new token per sequence per step                │
                      │            reads all weights + KV; memory-bound; sets TPOT  │
                      │            (+ speculative drafts [3.5], quantized [3.6])    │
                      │                 │ logits                                    │
                      │                 ▼                                           │
                      │ SAMPLER (temperature / top-p / grammar mask)      [3.1]     │
                      │                 │ token id                                  │
                      │                 ▼                                           │
                      │ DETOKENIZER ─▶ stream text chunk back to client (SSE)       │
                      │   loop until EOS / max_tokens, then free KV blocks          │
                      └─────────────────────────────────────────────────────────────┘
                      (in disaggregated serving, PREFILL and DECODE run on different
                       GPU pools and the KV cache is shipped between them    [3.7])
```

How the chapters depend on each other:

```text
3.1 fundamentals (roofline, prefill/decode)
  ├─▶ 3.2 KV cache ──▶ 3.3 paging & batching ──▶ 3.7 distributed / disaggregated
  │                    └─▶ 3.4 attention kernels
  ├─▶ 3.5 speculative decoding (uses spare compute in decode)
  └─▶ 3.6 quantization (reads fewer bytes)
                                 all ──▶ 3.8 engines, benchmarking, cost
```

## How to study this part

1. **Read 3.1 and 3.2 carefully.** The roofline argument and the KV-cache formula are used in every later chapter. Do the worked examples yourself with a calculator before reading the answers.
2. **Read 3.3 and 3.4 together.** 3.3 is the "operating system" (who runs, where memory lives); 3.4 is the "CPU instruction set" (how attention is computed on that memory). Run the block-allocator simulation and the NumPy FlashAttention and check that the outputs match.
3. **3.5 and 3.6 can be read in either order.** Both are about getting more out of each byte; both have short proofs worth working through by hand.
4. **3.7 and 3.8 are systems chapters.** Read them after you have served a model at least once.

Hands-on projects (do at least three):

- [ ] Compute the KV-cache size and maximum batch size for a model of your choice on one GPU (Chapter 3.2).
- [ ] Serve a 7–8B model with vLLM and with SGLang; sweep concurrency and plot TTFT/TPOT percentiles vs throughput (Chapter 3.8).
- [ ] Turn on prefix caching with a long shared system prompt and measure the TTFT drop (Chapter 3.3).
- [ ] Compare BF16 vs FP8 vs a 4-bit weight-only checkpoint on speed *and* accuracy on a task you care about (Chapter 3.6).
- [ ] Enable speculative decoding (n-gram or EAGLE) and measure the acceptance rate and speedup at batch 1 and at batch 64 (Chapter 3.5).
- [ ] Implement a minimal paged KV cache + continuous batching loop; [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) is a ~1,200-line reference.

Good companion courses: CMU **[15-442/642 Machine Learning Systems](https://mlsyscourse.org/)** and the **[GPU MODE](https://github.com/gpu-mode/lectures)** lecture series. Good companion overviews: the **[LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook)** and **[Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/)**.
