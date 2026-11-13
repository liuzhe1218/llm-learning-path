# 3.7 Distributed & Disaggregated Serving

!!! abstract "What you'll learn"
    - When and why a model must be served across several GPUs, and the three ways to split it: **tensor**, **pipeline** and **expert** parallelism, with their communication costs.
    - Why **prefill/decode disaggregation** improves goodput, and how to estimate the cost of shipping the KV cache between pools.
    - What **KV-cache-centric** architectures and **tiered KV storage** (GPU → CPU → SSD) are, and when loading a cached prefix beats recomputing it.
    - How trillion-parameter MoE models are served in production: wide expert parallelism, dual-batch overlap, expert load balancing (DeepSeek's design).
    - How requests are routed across replicas (including **prefix-aware routing**) and how serving fleets autoscale.

## Why this matters / the problem

One H100 has 80 GB of HBM. Llama-3-70B in BF16 needs ≈ 140 GB just for weights; DeepSeek-V3 (671B total parameters) needs ≈ 671 GB even in FP8. And even when a model fits, one GPU's bandwidth (≈ 3.35 TB/s) caps decode speed (Chapter 3.1) and its leftover memory caps the KV cache and therefore the batch size (Chapter 3.2).

So large models are served on **groups of GPUs** — within a server (8 GPUs connected by NVLink), across a rack, or across many servers connected by InfiniBand/RoCE networks. How you split the model decides how much data must cross those links on every step, and links are much slower than HBM. On top of that, large deployments run **many replicas** of the model behind a router, and must decide which replica gets each request. This chapter covers both levels: splitting one model instance across GPUs, and running a fleet of instances.

## Intuition

- **Tensor parallelism** = several cooks working on *the same dish at the same time*, each chopping part of every ingredient; they must constantly hand pieces to each other (fast links needed).
- **Pipeline parallelism** = an assembly line; each station does some layers and passes the half-finished dish on. Little communication, but one dish takes as long as the whole line.
- **Expert parallelism** = specialist cooks; each order (token) is sent to the two or eight specialists it needs and the results are sent back (an all-to-all shuffle).
- **Disaggregation** = separate kitchens for prep (prefill) and plating (decode), so a giant prep order never blocks the plating line — at the cost of carrying the prepped food (the KV cache) between kitchens.

## How it works

### Interconnects (approximate figures)

| Link | Typical bandwidth | Used for |
|---|---|---|
| HBM3 on H100 SXM | ≈ 3.35 TB/s | weights, KV, activations |
| NVLink 4 (H100 SXM, within an 8-GPU server) | ≈ 900 GB/s per GPU (total, both directions) | tensor/expert parallelism |
| PCIe Gen5 ×16 | ≈ 64 GB/s per direction | GPU ↔ CPU memory |
| InfiniBand NDR (400 Gb/s per NIC) | ≈ 50 GB/s per NIC per direction | across servers; H100 servers commonly have one NIC per GPU |

The ratio between NVLink and network bandwidth (≈ an order of magnitude) explains most deployment choices: communication-heavy parallelism stays inside an NVLink domain; communication-light parallelism crosses servers. Rack-scale systems such as NVIDIA GB200/GB300 NVL72 put 72 GPUs into one NVLink domain, which is why they suit wide expert parallelism.

### Tensor parallelism (TP)

Megatron-style TP splits each layer's weight matrices across $N$ GPUs:

- **MLP.** $W_{\text{up}}$ ($d \times d_{\text{ff}}$) is split by **columns**: GPU $i$ computes $h_i = \sigma(x W_{\text{up}}^{(i)})$, a slice of the hidden activation, with no communication. $W_{\text{down}}$ ($d_{\text{ff}} \times d$) is split by **rows**: GPU $i$ computes a partial output $h_i W_{\text{down}}^{(i)} \in \mathbb{R}^{d}$, and the full output is the **sum** over GPUs — one **all-reduce**.
- **Attention.** Heads are divided among GPUs (each GPU owns $n_q/N$ query heads and $n_{\text{kv}}/N$ KV heads, and the KV cache for those heads); the output projection is row-split, followed by one all-reduce.

```text
TP=2 MLP:   x (replicated on both GPUs)
             │                              GPU 0                      GPU 1
             ├── x @ W_up[:, :d_ff/2] ─▶ h0 ─▶ act ─▶ h0 @ W_down[:d_ff/2, :] ─▶ y0 ─┐
             └── x @ W_up[:, d_ff/2:] ─▶ h1 ─▶ act ─▶ h1 @ W_down[d_ff/2:, :] ─▶ y1 ─┤
                                                                       all-reduce: y = y0 + y1
                                        (no communication until here; y is replicated again)
```

So a forward pass has **two all-reduces per layer**, each over a tensor of shape `[tokens, d_model]`. A ring all-reduce of $M$ bytes over $N$ GPUs sends and receives about $2\frac{N-1}{N}M$ bytes per GPU. Per step that is small for decode (few tokens) but frequent; each all-reduce also has a fixed latency, so TP adds a per-layer synchronization cost that grows with $N$. TP's big win is that **every GPU reads only $1/N$ of the weights**, so aggregate bandwidth — and batch-1 decode speed — scale with $N$ (until communication dominates).

### Pipeline parallelism (PP)

PP assigns contiguous blocks of layers to different GPUs (stages). Only the activations at stage boundaries move (point-to-point, `[tokens, d_model]`), so PP tolerates slow links and is used across servers. But a single request still passes through all stages sequentially, so PP does **not** reduce per-request latency; it only increases capacity. To keep all stages busy, the engine must have several micro-batches in flight; otherwise stages sit idle ("pipeline bubbles").

### Expert parallelism (EP) for MoE

In an MoE layer (see [Part 1](../part1/index.md)) a router picks the top-$k$ of $E$ experts for each token. With EP, the $E$ experts are distributed across GPUs (e.g. 256 experts on 32 GPUs = 8 experts each). Each MoE layer then needs two **all-to-all** exchanges:

1. **Dispatch:** every GPU sends each of its tokens' hidden vectors to the GPUs that host the token's chosen experts.
2. **Combine:** expert outputs are sent back to the token's home GPU and summed with the router weights.

```text
 GPU 0 tokens [t1 t2]   GPU 1 tokens [t3 t4]        experts: E0,E1 on GPU 0 · E2,E3 on GPU 1
 router (top-2):  t1→E0,E2   t2→E1,E3   t3→E0,E1   t4→E2,E3

 dispatch (all-to-all)        expert compute              combine (all-to-all)
 GPU0 ─ t1,t2 copies ─▶ GPU1   GPU0: E0{t1,t3} E1{t2,t3}   outputs return to each token's
 GPU1 ─ t3   copies ─▶ GPU0    GPU1: E2{t1,t4} E3{t2,t4}   home GPU, weighted by router
                                                           probabilities and summed
```

Per token per MoE layer, roughly $k \cdot d \cdot b$ bytes go out and come back (fewer if several chosen experts sit on the same GPU, and dispatch can be sent in FP8). The rest of the layer (attention) is typically run with **data parallelism** across the same GPUs ("attention DP"), especially for MLA models whose small KV cache would otherwise be replicated by TP. The challenge is **load imbalance**: if many tokens pick the same "hot" expert, its GPU becomes the straggler that everyone waits for at the next all-to-all.

### Prefill/decode disaggregation

Chapter 3.1: prefill is compute-bound and long; decode is bandwidth-bound and latency-sensitive. Colocating them on the same GPUs causes **interference**: a prefill batch stalls decodes (hurting TPOT), and chunked prefill (Chapter 3.3) only bounds the damage. The two phases also want **different configurations**: prefill likes high compute utilization and moderate parallelism; decode likes large batches, lots of KV memory, and parallelism that minimizes per-step latency.

**Disaggregation** runs prefill and decode on separate GPU pools:

1. A request goes to a **prefill instance**, which computes the prompt's KV cache and the first token.
2. The KV cache is **transferred** (GPU-to-GPU over NVLink or RDMA) to a **decode instance**.
3. The decode instance continues generation; its batches never contain prefills.

```text
 request ──▶ router
               │
               ▼
 PREFILL POOL  (sized for prompt tokens/s; compute-bound; e.g. TP=8)
   prefill instance: prompt ─▶ KV cache + first token
               │
               │  KV transfer: RDMA / NVLink, pipelined layer by layer
               ▼
 DECODE POOL   (sized for concurrent sequences; bandwidth-bound; big batches, e.g. wide EP)
   decode instance: 1 token/step for hundreds of sequences ─▶ stream to client
```

The two pools are sized independently (e.g. more decode GPUs for long-output workloads) to maximize **goodput** — requests per second meeting both the TTFT and TPOT SLOs, as defined by DistServe. Splitwise showed the same idea can even use different GPU generations for the two phases. The cost is the KV transfer, extra scheduling complexity, and the need for a fast network. Transfers are usually pipelined layer by layer, so a layer's KV is sent while the next layer is still being computed.

### KV-cache-centric architectures and tiered storage

Once KV caches can move between machines, they become a **shared resource**. Mooncake (the system behind Moonshot AI's Kimi) builds the whole architecture around this: a distributed KV-cache pool spanning GPU HBM, CPU DRAM and SSD across the cluster, an RDMA transfer engine, and a scheduler that places requests where their prefix KV already lives (or fetches it). Llumnix live-migrates running requests, with their KV, between instances to rebalance load. Open-source building blocks include **LMCache** (KV offloading and sharing across engine instances) and Mooncake's transfer engine, used by SGLang and vLLM.

The basic decision for tiered KV is **load vs recompute**: reusing a cached prefix from a slower tier pays off whenever

$$
\frac{\text{KV bytes of the prefix}}{\text{tier bandwidth}} < \text{time to recompute the prefix (prefill)} .
$$

Because KV bytes grow linearly with prefix length while prefill FLOPs grow at least linearly (and attention quadratically), loading usually wins for long prefixes, as the worked example shows.

### Serving large MoE models: DeepSeek's design

DeepSeek published an overview of its V3/R1 inference system that has become a reference design:

- **Disaggregated** prefill and decode.
- **Prefill:** routed experts with EP32, attention (MLA) and shared expert with DP32; a deployment unit spans 4 nodes. **Decode:** EP144 with DP144 for attention, spanning 18 nodes. Both use **redundant** copies of hot experts (32 redundant routed experts).
- **Dual-batch overlap:** the batch is split into two micro-batches so that while one is in its all-to-all communication, the other is computing — hiding communication behind computation.
- **EPLB** (expert-parallel load balancer): uses observed expert loads to decide which experts to replicate and how to pack experts onto GPUs so every GPU has similar work.
- **DeepEP**: optimized all-to-all dispatch/combine kernels (high-throughput kernels for prefill, low-latency kernels for decode).

LMSYS reproduced this style of deployment in SGLang (PD disaggregation + large-scale EP) on 96 H100 GPUs. The common lesson: at this scale, the system is designed around **communication** (all-to-all, KV transfer) as much as around computation.

### Choosing a layout

| Situation | Typical choice | Why |
|---|---|---|
| Model fits on one GPU with room for KV | 1 GPU per replica, many replicas (data parallel) | No communication at all; scale by adding replicas |
| Dense model needs 2–8 GPUs | TP within one NVLink server | Splits weights and KV; all-reduces stay on NVLink |
| Dense model larger than one server | TP inside servers × PP across servers | PP's point-to-point traffic tolerates the slower network |
| Large MoE (hundreds of experts) | EP for experts + DP (or small TP) for attention, often across servers or a rack-scale NVLink domain | Experts dominate memory; all-to-all scales better than TP for sparse layers |
| Long prompts + strict TTFT and TPOT SLOs at scale | PD disaggregation on top of the above | Removes interference, sizes each phase separately |

### Routing and load balancing across replicas

A fleet runs many independent replicas (each possibly multi-GPU). The router chooses one per request:

- **Round-robin / random:** simple; ignores load and cache state.
- **Least-loaded:** by queue length, running requests, or KV-cache utilization; good for latency.
- **Prefix-aware (cache-aware):** send a request to a replica that already holds its prefix in cache (Chapter 3.3), e.g. by hashing the first blocks of the prompt or by tracking an approximate radix tree of each replica's cache. This can multiply prefix-cache hit rates for chat and agents, but pure affinity creates hot spots, so practical routers combine cache affinity with a load threshold.
- **Session affinity:** keep a conversation on one replica — a coarse form of prefix-aware routing.

Production stacks that implement KV-aware routing include NVIDIA **Dynamo**, **llm-d** (Kubernetes-native, vLLM-based) and the SGLang router.

### Autoscaling

Traffic varies by hour and day, so fleets scale replicas up and down. Useful signals are queue length, KV-cache utilization, and observed TTFT/TPOT versus SLO (GPU utilization is a poor signal: a memory-bound decode replica can look "busy" at any load). The hard part is **cold start**: a new replica must allocate GPUs, load tens to hundreds of GB of weights, and capture CUDA graphs before serving, which takes from tens of seconds to minutes. Mitigations include keeping warm spare capacity, fast weight loading from local NVMe or peer GPUs, and in disaggregated systems scaling the prefill and decode pools independently.

## Code

A back-of-the-envelope planner for the decisions in this chapter: per-GPU memory under TP, TP all-reduce traffic per decode step, KV-transfer time for disaggregation, and load-vs-recompute for tiered KV. All hardware numbers are approximate inputs you should replace with your own measurements.

```python
GB, GiB = 1e9, 2**30

def kv_bytes_per_token(layers, kv_heads, head_dim, b=2):
    return 2 * layers * kv_heads * head_dim * b

# Llama-3-70B, BF16
P, L, KVH, HD, D = 70e9, 80, 8, 128, 8192
kv_tok = kv_bytes_per_token(L, KVH, HD)

for tp in (2, 4, 8):
    w = P * 2 / tp                                   # weight bytes per GPU
    print(f"TP={tp}: weights/GPU {w / GB:5.1f} GB, KV room on 80 GB GPU "
          f"{(72 * GB - w) / GB:5.1f} GB, batch-1 bound {1000 * P * 2 / (tp * 3.35e12):.1f} ms/token")

# TP all-reduce traffic per decode step: 2 all-reduces/layer of [batch, d_model] BF16
tp, batch = 8, 64
msg = batch * D * 2
per_gpu = 2 * L * 2 * (tp - 1) / tp * msg            # ring all-reduce: 2(N-1)/N * M each
print(f"\nTP=8, batch 64: {2 * L} all-reduces/step, {per_gpu / 1e6:.0f} MB sent per GPU "
      f"-> {1000 * per_gpu / 450e9:.2f} ms at ~450 GB/s per direction (NVLink 4)")

# Disaggregation: ship the KV cache of an 8K-token prompt
ctx = 8192
kv = ctx * kv_tok
prefill_flops = 2 * P * ctx + 2 * ctx**2 * HD * 64 * L   # weights + causal attention (64 q-heads)
prefill_s = prefill_flops / (8 * 989e12 * 0.5)       # 8 x H100 at ~50% of BF16 peak
print(f"\nKV for {ctx} tokens: {kv / GB:.2f} GB; prefill ~{prefill_s * 1000:.0f} ms on 8xH100")
for name, bw in [("1 x 400Gb/s NIC", 50e9), ("8 x 400Gb/s NICs", 400e9), ("PCIe5 from CPU", 50e9)]:
    print(f"  transfer over {name:17s}: {1000 * kv / bw:6.1f} ms")
```

## Worked example

**1. Fitting Llama-3-70B.** In BF16 the weights are ≈ 140 GB. TP=2 on H100 80 GB puts 70 GB of weights on each GPU, leaving almost nothing for KV cache — technically fits, practically useless. TP=4 leaves ≈ 37 GB per GPU for KV (assuming ~72 GB usable), TP=8 leaves ≈ 54 GB. The batch-1 bandwidth bound improves from ≈ 21 ms/token (TP=2) to ≈ 5 ms/token (TP=8), before communication overhead. Alternatively, FP8 weights (≈ 70 GB) with TP=2 gives ≈ 37 GB KV room per GPU on half the GPUs.

**2. TP communication.** At TP=8 and decode batch 64, each all-reduce message is $64 \times 8192 \times 2 = 1$ MiB; with 160 all-reduces per step, each GPU sends ≈ 294 MB, ≈ 0.65 ms at ≈ 450 GB/s per direction. That is a noticeable but not dominant fraction of a ~10–20 ms decode step; the fixed latency of 160 synchronizations matters as much as the bytes. Across servers at ≈ 50 GB/s, the same traffic would take ≈ 6 ms per step — which is why TP stays inside an NVLink domain.

**3. Disaggregation KV transfer.** An 8K-token prompt on Llama-3-70B produces $8192 \times 320\ \text{KiB} \approx 2.68$ GB of KV cache. Prefill costs ≈ $2 \times 70\times10^9 \times 8192 \approx 1.15\times10^{15}$ FLOPs for the weights plus ≈ $0.09\times10^{15}$ for attention — about **0.3 s** on 8× H100 at ~50% of BF16 peak. Transfer time:

| Path | Bandwidth | Time for 2.68 GB |
|---|---|---|
| one 400 Gb/s NIC | ≈ 50 GB/s | ≈ 54 ms |
| eight 400 Gb/s NICs (one per GPU, each GPU sends its own KV heads) | ≈ 400 GB/s | ≈ 6.7 ms |

With per-GPU NICs and layer-by-layer pipelining, the transfer adds only a few percent to TTFT. With a single slow link and no pipelining it adds ~20%. FP8 KV halves both.

**4. Load vs recompute.** The same 2.68 GB loaded from CPU memory over PCIe Gen5 (≈ 50 GB/s achieved) takes ≈ 54 ms, versus ≈ 300 ms to recompute: loading is ~5× faster, and it frees the GPUs for other prefills. For a 500-token prefix the comparison is ≈ 3 ms vs ≈ 18 ms of prefill — still a win, but small in absolute terms.

## In practice (2025–2026)

- **Within a server, TP (often TP=8) or TP×EP** is the default for models that do not fit on one GPU; **PP** is used across servers or for very large dense models.
- **Wide EP + attention DP + PD disaggregation** is the standard recipe for DeepSeek-V3/R1-class and Kimi-K2-class MoE models; SGLang, vLLM, TensorRT-LLM and Dynamo all ship support. Rack-scale NVLink systems (GB200/GB300 NVL72) make wide EP cheaper by keeping all-to-all inside one NVLink domain.
- **Open stacks:** NVIDIA Dynamo (disaggregated serving with KV-aware routing and KV block management across tiers), llm-d (Kubernetes-native, vLLM-based, with cache-aware scheduling), LMCache (KV offloading and sharing), Mooncake's transfer engine and KV store.
- **API "prompt caching"** offered by model providers is the user-visible side of KV-cache-centric serving: cached prefixes are kept (often in lower tiers) for minutes and billed at a discount.

## Common pitfalls / misconceptions

- **"More GPUs = lower latency."** TP lowers per-token latency up to a point; PP does not lower it at all; past a certain TP degree communication cancels the gain.
- **TP across servers.** Two all-reduces per layer over a ~50 GB/s network make decode slow; use PP or EP across servers instead.
- **TP degree larger than the number of KV heads.** KV heads must then be replicated, wasting KV memory (Chapter 3.2).
- **Ignoring expert load imbalance.** Average expert load looks fine while the hottest expert's GPU sets the step time; monitor per-GPU max load, not the mean.
- **Disaggregating too early.** For small models, short prompts or modest traffic, chunked prefill on colocated GPUs is simpler and nearly as good; disaggregation pays off at scale with long prompts and strict SLOs.
- **Prefix-aware routing without a load cap,** which sends every request with a popular system prompt to the same replica.

## Key takeaways

- TP splits every matrix and needs two all-reduces per layer → keep it inside an NVLink domain; it scales aggregate bandwidth for a single model instance.
- PP splits layers with cheap point-to-point transfers but does not cut latency; EP distributes MoE experts and needs all-to-all dispatch/combine plus load balancing.
- PD disaggregation removes prefill–decode interference and lets each phase be sized and parallelized for goodput; KV transfer is affordable with per-GPU RDMA and layer-wise pipelining.
- KV caches can live in tiers (HBM → DRAM → SSD → remote); loading a long prefix is usually much cheaper than recomputing it.
- Fleet-level routing should combine load balancing with cache affinity; autoscaling must account for slow cold starts.

## Check your understanding

??? question "Why does tensor parallelism need an all-reduce after the row-parallel matrix but not after the column-parallel one?"
    Column-splitting $W_{\text{up}}$ gives each GPU a complete slice of the hidden activation, which the following element-wise activation can process locally. Row-splitting $W_{\text{down}}$ gives each GPU a partial sum over its slice of the hidden dimension, and the true output is the sum of all partial sums.

??? question "An MoE layer routes each token to the top-8 of 256 experts, with d = 7168, FP8 dispatch and BF16 combine. Roughly how many bytes leave a GPU per token per MoE layer, assuming all chosen experts are on other GPUs?"
    Dispatch: $8 \times 7168 \times 1 \approx 57$ KB; combine (received back): $8 \times 7168 \times 2 \approx 115$ KB. About 57 KB out and 115 KB in per token per MoE layer — multiplied by tokens and layers, this is why EP needs fast all-to-all kernels and overlap.

??? question "Why does PP not reduce the latency of a single request?"
    The request's tokens must pass through every stage in order; splitting layers across GPUs does not let any stage start before the previous one finishes for that token. PP adds throughput by keeping different micro-batches in different stages.

??? question "For which requests is prefill/decode disaggregation least useful?"
    Short prompts with short outputs at low load: there is little interference to remove, and the KV transfer and scheduling overhead are pure cost.

??? question "Prefix-aware routing raised your cache hit rate from 20% to 70% but P99 TTFT got worse. What probably happened, and how do you fix it?"
    Popular prefixes concentrated traffic onto a few replicas, which became overloaded. Combine cache affinity with load-aware fallback (route to a less-loaded replica when the preferred one's queue exceeds a threshold), or replicate hot prefixes across replicas.

## Further reading

| Paper / resource | Why read it |
|---|---|
| :material-star: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024 | Separate prefill/decode instances with per-phase parallelism to maximize goodput. |
| :material-book-open-variant: [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — Patel et al., ISCA 2024 | The same insight from Microsoft, including different hardware for each phase. |
| :material-star: [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Qin et al. (Moonshot AI), FAST 2025 | Production KV-centric design behind Kimi: distributed KV pool over GPU/CPU/SSD with RDMA transfer. |
| :material-book-open-variant: [Llumnix: Dynamic Scheduling for LLM Serving](https://arxiv.org/abs/2406.03243) — Sun et al., OSDI 2024 | Live migration of requests and their KV cache between instances for load balancing. |
| :material-book-open-variant: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | The column/row-parallel tensor-parallelism scheme used for inference too. |
| :material-book-open-variant: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Pope et al., 2022 | Systematic analysis of partitioning strategies and their communication costs for inference. |
| :material-star: [DeepSeek-V3/R1 inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | Prefill EP32 / decode EP144, dual-batch overlap, redundant experts: the reference large-MoE design. |
| :material-book-open-variant: [DeepEP](https://github.com/deepseek-ai/DeepEP) | All-to-all dispatch/combine kernels for expert parallelism. |
| :material-book-open-variant: [EPLB](https://github.com/deepseek-ai/EPLB) | Expert-parallel load balancer with redundant hot experts. |
| :material-new-box: [Large-scale EP in SGLang (LMSYS blog)](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | Reproducing DeepSeek-style PD disaggregation + large-scale EP on 96 H100s. |
| :material-book-open-variant: [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Open-source disaggregated serving framework with KV-aware routing. |
| :material-book-open-variant: [llm-d](https://github.com/llm-d/llm-d) | Kubernetes-native distributed serving built on vLLM. |
| :material-book-open-variant: [LMCache](https://github.com/LMCache/LMCache) | KV-cache offloading and sharing across engine instances. |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | The MoE architecture (256 routed experts, top-8, MLA) these serving designs target. |
