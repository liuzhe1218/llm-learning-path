# Part 3 · Inference Systems & Infrastructure

> **Goal:** understand why LLM serving is a *memory* problem, how modern engines (vLLM, SGLang, TensorRT-LLM) squeeze out throughput and latency, and how 2025–2026 systems serve trillion-parameter MoE reasoning models at scale.

**Legend:** :material-star: must-read · :material-book-open-variant: deep-dive · :material-new-box: 2025–2026 frontier

---

## 3.1 The big picture

Generation has two phases:

| Phase | Work | Bottleneck | Metric |
|---|---|---|---|
| **Prefill** | Process the whole prompt in parallel, build the KV cache | Compute-bound (big matmuls) | **TTFT** — time to first token |
| **Decode** | Generate one token at a time, reading all weights + KV cache each step | **Memory-bandwidth-bound** | **TPOT / ITL** — time per output token |

Back-of-the-envelope: at batch size 1, decode speed ≈ `HBM bandwidth / bytes of weights read per token`. A 70B BF16 model (~140 GB) on a GPU with ~3.35 TB/s → at most ~24 tokens/s. Almost every technique below either **reads fewer bytes** (quantization, MoE, smaller KV) or **amortizes each read over more tokens** (batching, speculative decoding).

**KV-cache size** per token = `2 × layers × kv_heads × head_dim × bytes`. For Llama-3-70B (80 layers, 8 KV heads, dim 128, BF16): ≈ 320 KB/token → a 128K-token context is ~40 GB. That is why GQA/MLA (Part 1) and KV quantization matter.

Start here: **[LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook)** and **[Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/)**.

---

## 3.2 Batching & memory management — the core of every engine

| Paper | Summary |
|---|---|
| :material-star: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022 | **Iteration-level scheduling (continuous batching)**: requests join/leave the batch every decode step instead of waiting for the slowest sequence. Multiplies throughput; now universal. |
| :material-star: [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023 | Stores KV cache in fixed-size **blocks** like OS virtual-memory pages → near-zero fragmentation, copy-on-write sharing for beam/parallel sampling. 2–4× throughput. The paper behind vLLM. |
| :material-star: [SGLang: Efficient Execution of Structured LM Programs (RadixAttention)](https://arxiv.org/abs/2312.07104) — Zheng et al., 2023 | Keeps KV caches in a **radix tree** for automatic **prefix caching** across requests (system prompts, few-shot, multi-turn, agents), plus a fast constrained-decoding engine. |
| :material-star: [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024 | **Chunked prefill**: split long prompts into chunks and piggyback them onto decode batches → no decode stalls, stable TPOT. Default in vLLM v1. |
| :material-book-open-variant: [FlashInfer: Efficient and Customizable Attention Engine for LLM Serving](https://arxiv.org/abs/2501.01005) — Ye et al., MLSys 2025 | JIT-compiled attention kernels for paged/ragged KV layouts and cascade (shared-prefix) attention; used inside vLLM and SGLang. |
| :material-book-open-variant: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) — Dao et al., 2023 (blog) | Splits the KV sequence across thread blocks during decode so long-context, small-batch decode uses the whole GPU. |

---

## 3.3 Speculative decoding

Idea: a cheap **draft** proposes *k* tokens; the big **target** model verifies them in one forward pass. Accepted tokens are "free" because decode is bandwidth-bound. Rejection sampling keeps the output distribution **exactly** the same.

| Paper | Summary |
|---|---|
| :material-star: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022 | The original algorithm and its losslessness proof; 2–3× speedups. |
| :material-book-open-variant: [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al. (DeepMind), 2023 | Concurrent derivation; applied to Chinchilla-70B. |
| :material-book-open-variant: [SpecInfer: Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) — Miao et al., 2023 | Verify a **tree** of candidate continuations in one pass with tree attention. |
| :material-star: [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Cai et al., 2024 | No separate draft model: extra heads on the target predict future tokens; tree verification. |
| :material-star: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., 2024 | Drafts at the **feature level** using the target's hidden states → much higher acceptance. |
| :material-new-box: [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — Li et al., 2025 | Multi-layer feature fusion and training-time simulation of decoding; ~3–6× speedups; the default speculative method in SGLang/vLLM in 2025. |
| :material-book-open-variant: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Fu et al., 2024 | Draft-free: Jacobi iteration generates n-grams that are verified in parallel. |

**MTP heads** trained during pre-training (DeepSeek-V3, Part 1) double as a built-in draft model — acceptance rates of ~80–90% for the second token were reported.

---

## 3.4 Quantization

| Paper | Summary |
|---|---|
| :material-book-open-variant: [LLM.int8()](https://arxiv.org/abs/2208.07339) — Dettmers et al., 2022 | Discovered **outlier features** in LLM activations; mixed-precision decomposition keeps them in FP16. |
| :material-star: [GPTQ: Accurate Post-Training Quantization for GPTs](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022 | One-shot, second-order (Hessian-based) **weight-only** quantization to 3–4 bits. |
| :material-star: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024 | Protect the ~1% salient weight channels (identified by activation magnitude) via per-channel scaling; robust 4-bit weights. Widely deployed. |
| :material-star: [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., 2022 | Migrates activation outliers into weights with a mathematically equivalent rescaling → enables **W8A8** INT8 inference. |
| :material-book-open-variant: [QServe: W4A8KV4 Quantization and System Co-design](https://arxiv.org/abs/2405.04532) — Lin et al., 2024 | Co-designs 4-bit weights / 8-bit activations / 4-bit KV with GPU kernels for real throughput gains. |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | Keys quantized per-channel, values per-token → 2-bit KV cache with little loss. |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | E4M3/E5M2 formats now native on Hopper/Blackwell; **FP8 serving is the 2024–2025 default** for large models. |
| :material-new-box: [Microscaling Data Formats for Deep Learning (MX)](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023 | Block-scaled MXFP8/MXFP6/MXFP4 formats (OCP standard). With Blackwell's FP4 tensor cores, **MXFP4/NVFP4** became practical in 2025 — gpt-oss ships its MoE weights in MXFP4. |

Rule of thumb (2025–2026): FP8 (W8A8) ≈ lossless; 4-bit weight-only (AWQ/GPTQ/MXFP4) is a good quality/size trade-off; aggressive activation/KV quantization needs evaluation on *your* tasks — reasoning models with long outputs are more sensitive.

---

## 3.5 KV-cache compression & long context

| Paper | Summary |
|---|---|
| :material-star: [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | The first few tokens absorb lots of attention ("sinks"); keeping them plus a sliding window allows stable infinite-length generation. Inspired the learned sinks in gpt-oss. |
| :material-book-open-variant: [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Zhang et al., 2023 | Evict KV entries that receive little cumulative attention. |
| :material-book-open-variant: [SnapKV](https://arxiv.org/abs/2404.14469) — Li et al., 2024 | Selects important KV positions per head using an observation window at the end of the prompt. |

Architectural fixes (GQA, MLA, sliding-window, hybrid linear attention, DeepSeek Sparse Attention — see Part 1) are generally more robust than post-hoc eviction, and are what 2025 frontier models chose.

---

## 3.6 Prefill/decode disaggregation & KV-cache-centric serving :material-new-box:

Prefill is compute-bound; decode is bandwidth-bound. Running both on the same GPUs makes them interfere. **Disaggregation** runs them on separate GPU pools and transfers the KV cache between them — the dominant large-scale architecture by 2025.

| Paper | Summary |
|---|---|
| :material-star: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024 | Separates prefill and decode instances, each with its own parallelism, to maximize **goodput** (requests meeting both TTFT and TPOT SLOs). |
| :material-book-open-variant: [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — Patel et al., ISCA 2024 | Same insight from Microsoft; even uses different hardware generations for each phase. |
| :material-star: [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Qin et al. (Moonshot AI), FAST 2025 | Production system behind Kimi: a distributed KV-cache pool across GPU/CPU/SSD with RDMA transfer; scheduling is built around KV-cache reuse. Its Transfer Engine is used by SGLang and vLLM. |
| :material-book-open-variant: [Llumnix: Dynamic Scheduling for LLM Serving](https://arxiv.org/abs/2406.03243) — Sun et al., OSDI 2024 | Live migration of requests (and their KV cache) between instances for load balancing. |

Production stacks to look at: **[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)**, **[llm-d](https://github.com/llm-d/llm-d)** (Kubernetes-native, vLLM-based), **[LMCache](https://github.com/LMCache/LMCache)** (KV-cache offloading & sharing), and SGLang's PD-disaggregation mode.

---

## 3.7 Serving large MoE models :material-new-box:

Trillion-parameter MoE models (DeepSeek-V3/R1, Kimi K2, Qwen3-235B) change the game: each token touches few parameters, but all experts must live somewhere.

- **Expert parallelism (EP)**: spread experts across many GPUs; tokens are sent to their experts with **all-to-all** communication — see DeepSeek's open-sourced [DeepEP](https://github.com/deepseek-ai/DeepEP) (all-to-all kernels) and [EPLB](https://github.com/deepseek-ai/EPLB) (expert load balancer with redundant hot experts).
- **Large-scale EP + disaggregation**: DeepSeek's [inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) describes prefill on EP32 and decode on EP144 with dual-batch overlap to hide communication. LMSYS reproduced it in SGLang on 96 H100s ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- **MLA-specific kernels**: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) for decode with latent KV.
- **Rack-scale hardware**: NVIDIA GB200/GB300 NVL72 puts 72 GPUs in one NVLink domain, making wide EP far cheaper.

---

## 3.8 Inference engines (know at least two)

| Engine | Strengths |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | De-facto open standard; PagedAttention, continuous batching, prefix caching, chunked prefill, broad model/hardware support; the v1 engine (2025) rewrote the scheduler for lower CPU overhead. |
| [SGLang](https://github.com/sgl-project/sglang) | RadixAttention prefix caching, fast structured output, strong DeepSeek/MoE support (large-scale EP, PD disaggregation); popular for RL rollouts. |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA-optimized kernels, FP8/FP4 on Hopper/Blackwell, in-flight batching. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | CPU/Apple Silicon/edge inference with GGUF quantization. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Compiler-based deployment to phones, browsers (WebGPU), and diverse GPUs. |
| [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | Production server integrated with the HF ecosystem. |

---

## 3.9 Inference for reasoning models & RL :material-new-box:

Reasoning models generate thousands of "thinking" tokens per answer, so **decode cost dominates** — every decode optimization above matters more. Related trends:

- **Test-time compute as a product knob:** thinking-budget controls, effort levels, early exit.
- **RL rollout engines:** RL training (Part 2) spends most of its time generating rollouts; frameworks like verl and slime embed vLLM/SGLang for generation and must sync updated weights to the inference engine every step. Long-tail rollouts motivate **asynchronous / partial rollouts**.
- **Train–inference mismatch:** numerical differences between training and inference kernels (e.g. in MoE routing or low precision) can destabilize RL; making inference **batch-invariant / deterministic** became a 2025 research topic (see Thinking Machines' [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).

---

## 3.10 Hands-on projects

- [ ] Compute the KV-cache size and max batch size for a model of your choice on one GPU (use the formula in 3.1).
- [ ] Serve a 7–8B model with vLLM and SGLang; benchmark TTFT/TPOT/throughput at different concurrency levels (`vllm bench serve` or SGLang's `bench_serving`).
- [ ] Turn on prefix caching with a long shared system prompt — measure the TTFT drop.
- [ ] Compare BF16 vs FP8 vs AWQ-INT4 on speed *and* accuracy (e.g. GSM8K).
- [ ] Enable speculative decoding (EAGLE-3 or n-gram) and measure acceptance rate and speedup.
- [ ] Implement a minimal paged KV cache + continuous batching loop in PyTorch (see [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) for a ~1,200-line reference).
- [ ] Courses: CMU **[15-442/642 Machine Learning Systems](https://mlsyscourse.org/)** and the GPU-programming community **[GPU MODE](https://github.com/gpu-mode/lectures)** lectures.

## 3.11 Self-check

??? question "Why is decode memory-bound but prefill compute-bound?"
    Prefill multiplies weights by many tokens at once (high arithmetic intensity). Decode multiplies weights by only one token per sequence, so each weight byte loaded from HBM is used for very few FLOPs unless the batch is large.

??? question "Why does speculative decoding not change the output distribution?"
    Draft tokens are accepted with probability min(1, p_target/p_draft), and on rejection a token is resampled from the normalized residual max(0, p_target − p_draft) — exactly reproducing target-model sampling.

??? question "When does prefill/decode disaggregation pay off, and what does it cost?"
    At scale with strict TTFT and TPOT SLOs and long prompts; it removes phase interference and allows per-phase parallelism. The cost is transferring the KV cache (needs fast interconnect such as RDMA/NVLink) and more complex scheduling.

??? question "Why is prefix caching especially valuable for agents?"
    Agent loops resend the same growing context (system prompt, tools, history) every turn; caching avoids recomputing the shared prefix, cutting TTFT and cost dramatically.
