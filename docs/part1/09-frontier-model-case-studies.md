# 1.9 Frontier Model Case Studies

!!! abstract "What you'll learn"
    - How to read a frontier model's architecture table and translate each entry into the trade-off it represents.
    - A side-by-side comparison of **Llama 3 405B, DeepSeek-V3, Qwen3-235B-A22B, gpt-oss-120b, Kimi K2 and Gemma 3 27B**, using only numbers reported in their technical reports and released model configs.
    - How to compute and compare their KV-cache footprints and training compute from those numbers.
    - What each lab's distinctive choices buy, tying together chapters 1.1–1.8.

## Why this matters / the problem

You now know every component of a modern LLM. Real models are combinations of those components chosen under constraints: a compute budget, a hardware fleet, a serving cost target, a context length goal, a team's appetite for risk. Reading technical reports with that lens is the fastest way to stay current, because new models are mostly new *combinations* and *tunings* of known ideas. This chapter practices that skill on six influential 2024–2025 models.

!!! note "Sources and conventions"
    Numbers come from each model's technical report or model card and, for exact dimensions, from the configuration files released with the weights. "—" means the figure is not reported (or we could not confirm it), not that it is zero. Context lengths are the maximum supported lengths; "K" means $2^{10}$ tokens as used by the labs (128K = 131,072).

## Intuition

Each model answers the same three questions differently:

1. **How do we buy quality?** More parameters (dense or MoE), more tokens, better data, a better optimizer.
2. **How do we keep training stable and affordable?** Precision (BF16 vs FP8), parallelism layout, balancing schemes, normalization tricks.
3. **How do we keep serving cheap?** Few active parameters (MoE), small KV cache (GQA, MLA, sliding windows), quantization-friendly design.

A dense 405B model (Llama 3) puts all its chips on (1) and stability; DeepSeek-V3 and Kimi K2 push (3) hard with MLA and very sparse MoE; gpt-oss optimizes for deployment on a single GPU; Gemma 3 targets strong quality on accelerators people actually own.

!!! tip "Try it before reading on"
    For each model, predict from the table alone:

    - roughly how much GPU memory its weights need in BF16 (2 bytes × total parameters);
    - how much compute each generated token costs ($2 \times$ active parameters);
    - whether its KV cache at 128K context fits comfortably next to the weights on one 8-GPU node.

    Then check your answers against the worked example below.

## How it works: the comparison table

| | Llama 3 405B | DeepSeek-V3 | Qwen3-235B-A22B | gpt-oss-120b | Kimi K2 | Gemma 3 27B |
|---|---|---|---|---|---|---|
| type | dense | MoE | MoE | MoE | MoE | dense |
| layers | 126 | 61 | 94 | 36 | 61 | 62 |
| $d_{\text{model}}$ | 16,384 | 7,168 | 4,096 | 2,880 | 7,168 | 5,376 |
| query heads / KV heads | 128 / 8 | 128 / — (MLA) | 64 / 4 | 64 / 8 | 64 / — (MLA) | 32 / 16 |
| head size | 128 | 128 (+64 RoPE dims for q/k) | 128 | 64 | 128 (+64 RoPE dims for q/k) | 128 |
| attention | GQA | MLA (latent 512) | GQA + QK-norm | GQA; alternating full / 128-token banded window; learned sinks | MLA (latent 512) | GQA + QK-norm; 5 local (1024 window) : 1 global |
| FFN | SwiGLU, $d_{ff}$ 53,248 | first 3 layers dense; then 256 routed + 1 shared expert, top-8, expert hidden 2,048 | 128 experts, top-8, no shared, expert hidden 1,536 | 128 experts, top-4, expert hidden 2,880 | first layer dense; then 384 routed + 1 shared, top-8, expert hidden 2,048 | GeGLU, $d_{ff}$ 21,504 |
| total / active params | 405B / 405B | 671B / 37B | 235B / 22B | ~117B / ~5.1B | 1.04T / 32B | 27B / 27B |
| vocabulary | 128K | 128K (129,280) | 151,936 | 201,088 (`o200k_harmony`) | 160K (163,840) | 262K |
| context | 128K | 128K | 32K native, 128K with YaRN | 128K (YaRN from 4K) | 128K | 128K |
| pre-training tokens | 15.6T | 14.8T | 36T | — | 15.5T | 14T |
| RoPE base | 500,000 | 10,000 (+YaRN) | 1,000,000 | 150,000 (+YaRN) | 50,000 (+YaRN) | 1M global / 10K local |
| optimizer / precision highlights | AdamW, BF16 | FP8 mixed precision; aux-loss-free balancing; MTP | global-batch balancing loss | MXFP4-quantized MoE weights at release | MuonClip | distillation from a larger teacher |

All six use Pre-norm RMSNorm, RoPE and gated FFNs. The differences are in attention (how the KV cache is shrunk), in the FFN (dense or how sparse), and in training tricks.

### A checklist for reading any model config

When a new model is released, its `config.json` and report answer most architecture questions. Go through these fields in order, and for each one ask what it costs and what it buys:

1. **`vocab_size`** → tokenizer granularity and multilingual coverage; embedding and output-layer size ([1.2](02-tokenization.md)).
2. **`num_hidden_layers`, `hidden_size`** → depth and width; with the FFN size, these give the parameter count ([1.3](03-transformer.md)).
3. **`num_attention_heads`, `num_key_value_heads`, `head_dim`** (or `kv_lora_rank`, `qk_rope_head_dim` for MLA) → attention type and KV-cache size per token ([1.4](04-modern-block.md)).
4. **`sliding_window`, layer-type lists** → local/global interleaving and how much of the cache is bounded ([1.6](06-long-context-and-efficient-attention.md)).
5. **`rope_theta`, `rope_scaling`, `max_position_embeddings`** → positional frequencies and whether context was extended with YaRN or similar ([1.6](06-long-context-and-efficient-attention.md)).
6. **`intermediate_size`, `moe_intermediate_size`, number of routed/shared experts, experts per token, first dense layers** → dense or sparse FFN, total vs active parameters ([1.5](05-mixture-of-experts.md)).
7. **Norm-related fields** (`rms_norm_eps`, QK-norm flags), **`tie_word_embeddings`**, biases → stability and small-model economies ([1.4](04-modern-block.md)).
8. **Report-only facts**: training tokens, data mix, optimizer, precision, parallelism, schedule ([1.7](07-scaling-laws-and-data.md), [1.8](08-training-at-scale.md)).

### Design choices and what they buy

| design choice | who uses it (in this chapter) | what it buys | what it costs | chapter |
|---|---|---|---|---|
| dense FFN | Llama 3 405B, Gemma 3 27B | simplicity, stable training, easy serving | compute per token grows with all parameters | [1.3](03-transformer.md) |
| fine-grained MoE | DeepSeek-V3, Qwen3, Kimi K2, gpt-oss | many parameters at small active compute | memory for all experts, all-to-all communication, balancing | [1.5](05-mixture-of-experts.md) |
| shared expert | DeepSeek-V3, Kimi K2 | common knowledge in one place; routed experts specialize | one always-on FFN per layer | [1.5](05-mixture-of-experts.md) |
| GQA with few KV heads | Llama 3, Qwen3, gpt-oss, Gemma 3 | smaller KV cache, faster decoding | small quality risk as KV heads shrink | [1.4](04-modern-block.md) |
| MLA | DeepSeek-V3, Kimi K2 | smallest full-attention cache at MHA-level quality | more complex kernels; decoupled RoPE | [1.4](04-modern-block.md) |
| local/global interleaving | gpt-oss, Gemma 3 | bounded cache and compute in most layers | weaker long-range access in local layers | [1.6](06-long-context-and-efficient-attention.md) |
| QK-norm / QK-clip | Qwen3, Gemma 3 / Kimi K2 | no attention-logit blow-ups | small overhead; QK-norm awkward with MLA | [1.8](08-training-at-scale.md) |
| large RoPE base and/or RoPE rescaling (YaRN, interpolation) | all six, in different combinations | long context from mostly short-context pre-training | needs a long-context training stage | [1.6](06-long-context-and-efficient-attention.md) |
| very large token budget | Qwen3 (36T) | better small and mid-size models | data acquisition and filtering effort | [1.7](07-scaling-laws-and-data.md) |
| FP8 training | DeepSeek-V3 | ~2× matmul throughput vs BF16 | careful scaling to avoid precision loss | [1.8](08-training-at-scale.md) |
| Muon optimizer | Kimi K2 | better compute efficiency than AdamW (per Moonlight) | newer, needs stabilization (QK-clip) | [1.8](08-training-at-scale.md) |

## Code

The script below turns the table into the two numbers that matter most for cost: KV cache per token (serving) and training FLOPs ($C \approx 6 N_{\text{active}} D$, [1.7](07-scaling-laws-and-data.md)).

```python
# KV cache per token and training compute from published configs (BF16 cache, 2 bytes/element).
MODELS = {
    # cache elements per token per full-attention layer; number of full / windowed layers; window
    "Llama 3 405B":       dict(kv_per_layer=2 * 8 * 128,  full=126, local=0,  window=0),
    "DeepSeek-V3":        dict(kv_per_layer=512 + 64,     full=61,  local=0,  window=0),   # MLA latent + RoPE key
    "Qwen3-235B-A22B":    dict(kv_per_layer=2 * 4 * 128,  full=94,  local=0,  window=0),
    "gpt-oss-120b":       dict(kv_per_layer=2 * 8 * 64,   full=18,  local=18, window=128),
    "Kimi K2":            dict(kv_per_layer=512 + 64,     full=61,  local=0,  window=0),   # MLA
    "Gemma 3 27B":        dict(kv_per_layer=2 * 16 * 128, full=10,  local=52, window=1024),
}
ACTIVE_AND_TOKENS = {  # (active params, training tokens) where both are reported
    "Llama 3 405B": (405e9, 15.6e12), "DeepSeek-V3": (37e9, 14.8e12),
    "Qwen3-235B-A22B": (22e9, 36e12), "Kimi K2": (32e9, 15.5e12), "Gemma 3 27B": (27e9, 14e12),
}

def kv_cache_bytes(cfg, context, bytes_per_elem=2):
    full = cfg["full"] * context
    local = cfg["local"] * min(context, cfg["window"])
    return (full + local) * cfg["kv_per_layer"] * bytes_per_elem

print(f"{'model':18s} {'KiB/token':>10s} {'GiB @128K':>10s} {'train FLOPs':>12s}")
for name, cfg in MODELS.items():
    per_tok = cfg["full"] * cfg["kv_per_layer"] * 2 / 1024
    at_128k = kv_cache_bytes(cfg, 131072) / 2**30
    flops = ""
    if name in ACTIVE_AND_TOKENS:
        n, d = ACTIVE_AND_TOKENS[name]
        flops = f"{6 * n * d:.1e}"                      # C ≈ 6 N_active D
    print(f"{name:18s} {per_tok:10.1f} {at_128k:10.1f} {flops:>12s}")
```

The layer split for gpt-oss (18 full + 18 banded) follows its alternating pattern; for Gemma 3 27B, its config places one global layer after every five local ones, which gives 10 global and 52 local layers out of 62.

## Worked example

Output of the script, with BF16 caches and one 128K-token sequence:

| model | KV cache per token (full-attention layers) | KV cache at 128K | training compute $6 N_{\text{active}} D$ |
|---|---|---|---|
| Llama 3 405B | 504 KiB | 63.0 GiB | $3.8 \times 10^{25}$ |
| DeepSeek-V3 | 68.6 KiB | 8.6 GiB | $3.3 \times 10^{24}$ |
| Qwen3-235B-A22B | 188 KiB | 23.5 GiB | $4.8 \times 10^{24}$ |
| gpt-oss-120b | 36 KiB | 4.5 GiB | — |
| Kimi K2 | 68.6 KiB | 8.6 GiB | $3.0 \times 10^{24}$ |
| Gemma 3 27B | 80 KiB | 10.4 GiB | $2.3 \times 10^{24}$ |

Reading the numbers:

- **Llama 3 405B** used about 11× the training compute of DeepSeek-V3 ($3.8 \times 10^{25}$ vs $3.3 \times 10^{24}$), even though DeepSeek-V3 has more total parameters, because every token passes through all 405B parameters. The $6ND$ estimate for Llama 3 matches the $3.8 \times 10^{25}$ FLOPs in its report.
- **MLA** gives DeepSeek-V3 and Kimi K2 a cache about 7× smaller per token than Llama 3 405B's GQA-8.
- **gpt-oss** has the smallest cache: half its layers are banded to 128 tokens, and its full-attention layers use 8 KV heads of size 64.
- **Gemma 3's** 5:1 local:global pattern cuts its cache from about 62 GiB (if every layer were global) to about 10 GiB.
- Per generated token, inference compute is $\approx 2 N_{\text{active}}$: 810 GFLOPs for Llama 3 405B, 74 for DeepSeek-V3, 44 for Qwen3-235B-A22B, about 10 for gpt-oss-120b.

**Reconstructing parameter counts from the configs.** The same accounting as [1.3](03-transformer.md) and [1.5](05-mixture-of-experts.md) recovers the headline numbers. (Norm weights and biases are negligible and omitted.)

*Llama 3 405B* ($L = 126$, $d = 16{,}384$, 128 query heads and 8 KV heads of size 128, SwiGLU $d_{ff} = 53{,}248$, $V = 128{,}256$, untied):

| component | calculation | parameters |
|---|---|---|
| attention per layer | $W_Q, W_O$: $2 \times 16{,}384^2$; $W_K, W_V$: $2 \times 16{,}384 \times 1024$ | 570.4M |
| FFN per layer | $3 \times 16{,}384 \times 53{,}248$ | 2,617.2M |
| 126 layers | $126 \times 3{,}187.7\text{M}$ | 401.6B |
| embeddings + output | $2 \times 128{,}256 \times 16{,}384$ | 4.2B |
| **total** | | **≈ 405.8B** |

Note how GQA makes $W_K$ and $W_V$ 16× smaller than $W_Q$, and that the FFN holds over 80% of each layer's parameters.

*Qwen3-235B-A22B* ($L = 94$, $d = 4096$, 64 query heads and 4 KV heads of size 128, 128 experts with hidden size 1,536, top-8, $V = 151{,}936$, untied):

| component | calculation | total | active |
|---|---|---|---|
| attention per layer | $2 \times 4096 \times 8192 + 2 \times 4096 \times 512$ | 71.3M | 71.3M |
| one expert | $3 \times 4096 \times 1536$ | 18.9M | |
| experts per layer | $128 \times 18.9\text{M}$ (8 active) | 2,415.9M | 151.0M |
| router per layer | $4096 \times 128$ | 0.5M | 0.5M |
| 94 layers | | 233.8B | 20.9B |
| embeddings + output | $2 \times 151{,}936 \times 4096$ | 1.2B | 1.2B |
| **total** | | **≈ 235.1B** | **≈ 22.2B** |

The name "235B-A22B" is exactly this calculation. Notice that the query projection is twice as wide as $d$ ($64 \times 128 = 8192$), a choice that decouples the attention width from the residual-stream width.

## In practice (2025–2026): what each design buys

### Llama 3 405B: the dense baseline done carefully

Meta deliberately chose a **standard dense Transformer** over MoE "to maximize training stability", and invested instead in data (15.6T tokens with a carefully tuned mix, [1.7](07-scaling-laws-and-data.md)), a 128K tokenizer ([1.2](02-tokenization.md)), a large RoPE base of 500,000 for long context ([1.4](04-modern-block.md)), and 4D parallelism on 16,384 H100s ([1.8](08-training-at-scale.md)). It also masks attention between different documents packed into the same training sequence. Its size was chosen from its own scaling-law fits (about 402B predicted compute-optimal for its budget). The report is the most complete public description of a dense frontier pipeline, including staged long-context training to 128K and annealing. Trade-off: maximal simplicity and robustness, at the cost of expensive inference ($2 \times 405\text{B}$ FLOPs and a large KV cache per token).

### DeepSeek-V3: efficiency everywhere

Almost every component is chosen for cost: **MLA** for a small KV cache ([1.4](04-modern-block.md)); **fine-grained MoE** with 256 routed + 1 shared expert, 37B active out of 671B ([1.5](05-mixture-of-experts.md)); **auxiliary-loss-free balancing** so balancing does not fight the LM loss; **node-limited routing** to bound all-to-all traffic; **FP8 training** with fine-grained scaling and the **DualPipe** schedule ([1.8](08-training-at-scale.md)). It also adds **multi-token prediction (MTP)**: an extra sequential module predicts the token after next, which densifies the training signal and can be reused at inference as a draft model for speculative decoding (the report cites an 85–90% acceptance rate for the second token; see [Part 3](../part3/index.md)). The report states 2.788M H800 GPU-hours for the full training.

### Qwen3-235B-A22B: a clean, conventional MoE

Qwen3 combines **GQA with only 4 KV heads** (a 16:1 query-to-KV ratio), **QK-norm** (and no QKV bias, unlike Qwen2), and a 128-expert, top-8 MoE **without shared experts**, balanced with a **global-batch** loss so experts can specialize by domain. Its distinctive bet is data volume: **36T tokens** for every model in the family, including staged pre-training with a reasoning-heavy stage and a long-context stage. Trade-off: a simple, well-understood design with very large data, and a GQA cache that is larger than MLA's but easier to implement and serve.

### gpt-oss-120b: designed to deploy

OpenAI's open-weight model is small in active compute (about 5.1B active of about 117B) and engineered for serving: **alternating full and 128-token banded attention** halves long-range attention cost and cache; **learned attention sinks** per head let heads attend to "nothing", which helps windowed attention ([1.6](06-long-context-and-efficient-attention.md)); **GQA with 8 KV heads**; and MoE weights released in **MXFP4** (4-bit) so the 120b model fits on a single 80 GB GPU according to its model card. It uses the `o200k_harmony` tokenizer with special tokens for its chat format. Training-token counts are not reported.

### Kimi K2: scaling sparsity and the optimizer

Kimi K2 starts from a DeepSeek-V3-like design and changes three things. It increases sparsity to **384 experts** (top-8, 1 shared), reaching about 1.04T total parameters with 32B active, motivated by its finding that more experts at fixed active compute keep lowering loss ([1.5](05-mixture-of-experts.md)). It **halves the attention heads** to 64 to cut inference cost for long agentic contexts. And it trains with **MuonClip**: the Muon optimizer plus QK-clip, which rescales query/key weights when attention logits grow too large ([1.8](08-training-at-scale.md)). The report describes 15.5T tokens of pre-training without a single loss spike, and heavy use of rephrased (synthetic) data to stretch high-quality tokens.

### Gemma 3: quality per accelerator

Gemma 3 targets models that run on one GPU or TPU. The 27B model uses **5 local layers (1024-token window) per global layer**, which keeps the KV cache small at 128K context; **QK-norm** (replacing Gemma 2's logit soft-capping); separate RoPE bases for local (10K) and global (1M) layers; a large **262K vocabulary** for multilingual coverage ([1.2](02-tokenization.md)); and **knowledge distillation** from a larger teacher during pre-training (distillation is covered in [Part 2](../part2/index.md)). It is also multimodal, with a SigLIP-based vision encoder. Trade-off: dense compute for simplicity at this size, with attention engineered aggressively for memory.

### Cross-cutting trends

- **Pre-RMSNorm + RoPE + gated FFN + GQA-or-MLA** is the universal skeleton (chapters [1.3](03-transformer.md)–[1.4](04-modern-block.md)).
- **MoE with fine-grained experts** dominates at the frontier; dense models remain common at sizes meant for single-device deployment.
- **KV-cache reduction** is pursued at every level: fewer KV heads, latent caches, local windows, sinks; and next, hybrid linear attention (Qwen3-Next, MiniMax-01) and trainable sparse attention (DeepSeek-V3.2).
- **Tokens: 15–36T**, with heavy filtering, synthetic rephrasing and annealing ([1.7](07-scaling-laws-and-data.md)).
- **Precision and optimizers** are now competitive differentiators: FP8 training (DeepSeek-V3), Muon (Kimi K2), 4-bit release formats (gpt-oss).
- **Llama 4** (2025, blog only) moved Meta to MoE as well (Scout and Maverick), with native multimodality and interleaved RoPE/NoPE layers ("iRoPE") for long context.

## Common pitfalls / misconceptions

- **Comparing MoE and dense models by total parameters.** A 671B MoE with 37B active is far cheaper to train than a 405B dense model; compare active parameters for compute and total parameters for memory.
- **Assuming a bigger vocabulary or context number means a better model.** These are trade-offs; quality depends on data and training.
- **Reading "128K context" as "128K usable".** Long-context quality varies; check needle-in-a-haystack and long-document reasoning results.
- **Ignoring precision in memory estimates.** gpt-oss's single-GPU claim depends on MXFP4 expert weights; in BF16 the ~117B weights alone would take about 234 GB, roughly three 80 GB GPUs.
- **Treating model-card numbers as interchangeable.** "128K" vocabularies are 128,000, 128,256 or 129,280 depending on special tokens; when exact numbers matter, read the config.

## Key takeaways

- Six frontier models share one skeleton and differ in attention (GQA, MLA, local/global), FFN sparsity, data volume and training tricks.
- KV cache per token ranges from about 36 KiB (gpt-oss-120b) to about 504 KiB (Llama 3 405B), which largely determines long-context serving cost.
- MoE lets DeepSeek-V3, Qwen3-235B-A22B and Kimi K2 train with roughly a tenth of Llama 3 405B's compute while holding comparable or more total parameters.
- Each distinctive choice maps to a constraint: stability (Llama 3 dense), cost (DeepSeek-V3 MLA + FP8), data (Qwen3's 36T), deployment (gpt-oss windows + MXFP4), sparsity and optimizer (Kimi K2), single-accelerator quality (Gemma 3).
- With chapters 1.1–1.8 you can compute these trade-offs yourself from any config file.

## Check your understanding

??? question "Why does Qwen3-235B-A22B have a larger KV cache per token than DeepSeek-V3, even though it has far fewer total parameters?"
    KV-cache size depends on attention design, not parameter count. Qwen3 caches 4 KV heads × 128 dims × 2 (K and V) = 1,024 elements per layer over 94 layers; DeepSeek-V3 caches a 576-element MLA latent over 61 layers.

??? question "Estimate the training compute of a hypothetical MoE with 20B active parameters trained on 25T tokens."
    $6 \times 20 \times 10^9 \times 25 \times 10^{12} = 3 \times 10^{24}$ FLOPs, about the same as DeepSeek-V3 or Kimi K2.

??? question "What problem do gpt-oss's learned attention sinks solve, and why is it especially relevant for its banded layers?"
    Softmax forces weights to sum to 1, so a head with nothing relevant to attend to must still place weight somewhere. A learned sink logit gives it a "nowhere" option. In banded layers, the usual implicit sink (the first token) falls outside the window, so an explicit sink is particularly useful.

??? question "Kimi K2 halved the number of attention heads compared with DeepSeek-V3. What does that save, given MLA's cache does not depend on the head count?"
    Attention compute during prefill and decoding (each head computes scores against the whole context) and the size of the query and output projections. For long agentic contexts, attention compute per token is a significant cost even when the cache is small.

??? question "Why might Meta have chosen a dense model for Llama 3 405B despite MoE's compute advantage?"
    The report cites training stability. Dense models avoid routing collapse, balancing losses, all-to-all communication and expert-parallel engineering, which reduces risk for a very large, months-long run.

## Further reading

| Report | What to look for |
|---|---|
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | 90+ pages covering a full dense 405B pipeline: data mix, scaling-law-driven sizing, 4D parallelism, long-context stages, post-training. The best single "how a frontier lab does it" reference. |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | MLA + fine-grained MoE + aux-loss-free balancing + MTP + FP8 + DualPipe. V3.1/V3.2 later add hybrid thinking and DeepSeek Sparse Attention. |
| :material-star: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Alibaba, 2025 | Dense and MoE families (0.6B → 235B-A22B), 36T tokens, QK-norm, unified "thinking / non-thinking" modes. Qwen3-Next adds hybrid Gated DeltaNet + very sparse MoE. |
| :material-new-box: [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) — OpenAI, 2025 | MoE, alternating dense/banded-window attention, attention sinks, MXFP4-quantized MoE weights, `o200k_harmony` tokenizer. |
| :material-new-box: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | 1T total / 32B active MoE trained with MuonClip; sparsity scaling; heavy agentic data synthesis. |
| :material-new-box: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Google DeepMind, 2025 | 5:1 local:global attention to shrink KV cache, QK-norm, distillation from a larger teacher, multimodal. |
| :material-book-open-variant: Llama 4 ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) — Meta, 2025 | MoE (Scout / Maverick), native multimodality via early fusion, interleaved RoPE/NoPE layers ("iRoPE") for long context. |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — AI2, 2025 | Fully reproducible end-to-end; best for learning since data, code and checkpoints are all released. |
| :material-book-open-variant: [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — Gloeckle et al., 2024 | Extra heads predict several future tokens: denser training signal and a free draft model for speculative decoding (used in DeepSeek-V3). |
| :material-book-open-variant: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — Sebastian Raschka | A living, illustrated comparison of recent architectures; a good companion to this chapter. |
| :material-book-open-variant: [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) — Hugging Face | A practitioner's walkthrough of the decisions behind training a modern LLM end to end. |
