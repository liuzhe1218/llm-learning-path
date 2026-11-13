# Part 1 · Pre-training & Model Architecture

> **Goal:** understand how a base LLM is built — from tokens to a trillion-parameter Mixture-of-Experts trained on tens of trillions of tokens — and be able to explain *why* every architectural choice in a 2025–2026 frontier model exists.

**Legend:** :material-star: must-read · :material-book-open-variant: deep-dive · :material-new-box: 2025–2026 frontier

---

## 1.1 The big picture

Pre-training = **next-token prediction** on a huge text corpus:

\[
\mathcal{L}(\theta) = -\sum_{t} \log p_\theta(x_t \mid x_{<t})
\]

Everything else in this part is about doing that *better per FLOP*:

| Lever | Question it answers | Section |
|---|---|---|
| Tokenization | What is a "token"? | 1.2 |
| Architecture | What function maps context → next-token distribution? | 1.3–1.6 |
| Scaling laws | How big a model, how many tokens, for a given budget? | 1.7 |
| Data | Which tokens? | 1.8 |
| Systems & optimization | How to train it stably across thousands of GPUs? | 1.9 |

A modern decoder-only block (Llama/Qwen/DeepSeek style) looks like this:

```text
x ─► RMSNorm ─► Attention (GQA or MLA, RoPE, optional QK-norm) ─► + ─►
     RMSNorm ─► FFN (SwiGLU dense, or MoE: router → top-k experts + shared expert) ─► + ─► next block
```

---

## 1.2 Tokenization

- **Byte-Pair Encoding (BPE)** merges the most frequent symbol pairs into a vocabulary; byte-level BPE (GPT-2 onward) guarantees no out-of-vocabulary tokens.
- Vocabulary sizes have grown: 32K (Llama 2) → 128K (Llama 3) → ~150K–200K+ (Qwen, gpt-oss `o200k`). Bigger vocab = shorter sequences and better multilingual compression at the cost of a larger embedding/output layer.

| Paper | Summary |
|---|---|
| :material-star: [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2015 | Introduced BPE for NLP: start from characters, repeatedly merge the most frequent pair. The basis of nearly every LLM tokenizer today. |
| :material-book-open-variant: [SentencePiece](https://arxiv.org/abs/1808.06226) — Kudo & Richardson, 2018 | Language-agnostic tokenizer that treats text as raw Unicode (whitespace included) — popular for multilingual and CJK-heavy models. |

---

## 1.3 The Transformer and the GPT recipe

| Paper | Summary |
|---|---|
| :material-star: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 | The Transformer: scaled dot-product attention, multi-head attention, residual connections, positional encodings. Every model in this guide is a descendant. Focus on §3. |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | Decoder-only LM + pre-norm + large web corpus → zero-shot task transfer. Establishes the "just scale next-token prediction" recipe. |
| :material-star: [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020 | 175B parameters; in-context (few-shot) learning emerges with scale. Read for the framing and evaluation methodology. |
| :material-book-open-variant: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al., 2020 | Explains why **Pre-LN** (normalize before the sublayer) trains stably without warmup tricks — now universal. |

---

## 1.4 The "modern block": components every 2025 model uses

| Paper | Summary |
|---|---|
| :material-star: [RMSNorm](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019 | LayerNorm without mean-centering: cheaper, equally good. Standard in Llama, Qwen, DeepSeek, Gemma. |
| :material-star: [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 | Gated FFN `(Swish(xW) ⊙ xV)W₂` beats ReLU/GELU FFNs at equal compute. The default FFN everywhere. |
| :material-star: [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al., 2021 | Encodes position by rotating query/key vectors so attention depends on *relative* distance. The default positional scheme; also the hook for context-extension methods. |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | All query heads share one K/V head → KV cache shrinks by #heads. Big inference win, small quality loss. |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | Groups of query heads share a K/V head: the quality/memory sweet spot between MHA and MQA. Used by Llama 2-70B onward, Qwen, Mistral, gpt-oss. |
| :material-book-open-variant: [Scaling Vision Transformers to 22B Parameters](https://arxiv.org/abs/2302.05442) — Dehghani et al., 2023 | Among the first to use **QK-norm** (normalizing queries/keys) to prevent attention-logit blow-ups; now common in LLMs (OLMo 2, Qwen3, Gemma 3). |
| :material-book-open-variant: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023 | Packaged Pre-RMSNorm + SwiGLU + RoPE into the open-model "standard block" and showed smaller models trained on more tokens beat bigger ones. |
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | 90+ pages covering a full dense 405B pipeline: data mix, scaling-law-driven sizing, 4D parallelism, long-context stages, post-training. The best single "how a frontier lab does it" reference. |

---

## 1.5 Mixture-of-Experts (MoE)

MoE replaces the dense FFN with many **experts**; a **router** sends each token to its top-*k*. Total parameters grow, while *active* parameters (and FLOPs) per token stay small. Since 2024 nearly all frontier open models are MoE.

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017 | Original top-k gated MoE with load-balancing loss (in LSTMs). |
| :material-book-open-variant: [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020 | MoE Transformers at 600B scale with **expert parallelism** across devices. |
| :material-star: [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021 | Top-1 routing, capacity factor, auxiliary load-balancing loss, bf16 stability tricks. The clearest intro to MoE engineering. |
| :material-star: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024 | 8 experts, top-2, 47B total / 13B active; the model that made open MoE mainstream. |
| :material-star: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024 | **Fine-grained experts** (many small experts, higher k) + **shared experts** that every token uses. The design adopted by DeepSeek-V2/V3, Qwen3-MoE, Kimi K2 and others. |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 671B total / 37B active. Introduces **auxiliary-loss-free load balancing** (per-expert bias terms), **multi-token prediction**, FP8 training, and DualPipe. Arguably the most influential open architecture report of the era. |

---

## 1.6 Attention efficiency & long context

### Smaller KV cache

| Paper | Summary |
|---|---|
| :material-star: [DeepSeek-V2 (Multi-head Latent Attention, MLA)](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | Compresses K/V into a low-rank latent vector per token, so the KV cache is far smaller than GQA while matching MHA quality. Uses "decoupled RoPE" to stay compatible with rotary positions. Used in DeepSeek-V3/R1 and Kimi K2. |

### Faster exact attention (IO-aware kernels)

| Paper | Summary |
|---|---|
| :material-star: [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | Tiles attention so the N×N matrix never hits HBM; exact, faster, O(N) memory. The key reason long-context training is affordable. |
| :material-book-open-variant: [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023 | Better work partitioning across warps/thread blocks; ~2× over v1. |
| :material-book-open-variant: [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | Hopper-specific: asynchrony (TMA, warp specialization) and FP8. |

### Context-length extension

| Paper | Summary |
|---|---|
| :material-star: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — Peng et al., 2023 | Rescales RoPE frequencies non-uniformly ("NTK-by-parts") plus an attention temperature; extends context with very little continued training. Widely used (Qwen, DeepSeek, gpt-oss). |
| :material-book-open-variant: [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023 | Shards the sequence across devices and passes K/V blocks around a ring — the basis of **context parallelism** for million-token training. |

### Sparse, linear & hybrid attention :material-new-box:

The 2025 trend: keep a few full-attention layers and make the rest cheap.

| Paper | Summary |
|---|---|
| :material-star: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023 | Input-dependent state-space model; O(N) training, O(1)-per-token inference memory. |
| :material-book-open-variant: [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) — Dao & Gu, 2024 | Shows SSMs and linear attention are duals; faster matmul-friendly algorithm. Mamba layers appear in hybrids such as Jamba, NVIDIA Nemotron-H, and others. |
| :material-new-box: [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Yang et al., 2024 | Combines gating (forgetting) with the delta rule (precise memory updates) for linear attention. Used in hybrid form (≈3 linear : 1 full attention) in Qwen3-Next. |
| :material-new-box: [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) — MiniMax, 2025 | First frontier-scale hybrid linear-attention MoE (456B), with up to 1M–4M context. |
| :material-new-box: [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek), 2025 | Hardware-aligned, trainable sparse attention (compressed + selected + sliding-window branches). Its successor, DeepSeek Sparse Attention (DSA, in DeepSeek-V3.2), uses a lightweight "indexer" to choose top-k tokens. |
| :material-book-open-variant: [Differential Transformer](https://arxiv.org/abs/2410.05258) — Ye et al., 2024 | Attention = difference of two softmax maps, cancelling attention noise; better long-context retrieval. |

Also notable: **sliding-window / local–global interleaving** (Mistral, Gemma 2/3, gpt-oss) and **attention sinks** (learned per-head sink logits in gpt-oss) — see Part 3 for StreamingLLM.

---

## 1.7 Scaling laws

| Paper | Summary |
|---|---|
| :material-star: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020 | Loss follows power laws in parameters, data and compute. Kick-started "scale is all you need". |
| :material-star: [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022 | Corrected Kaplan: params and tokens should scale roughly equally (≈20 tokens/param for compute-optimal). Modern models deliberately **over-train** far beyond this (Llama 3 8B: ~15T tokens) because inference cost depends on size, not training tokens. |
| :material-book-open-variant: [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (μP)](https://arxiv.org/abs/2203.03466) — Yang et al., 2022 | A parameterization under which optimal learning rates transfer from small to large models — tune small, train big. |
| :material-book-open-variant: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023 | Repeating data up to ~4 epochs is almost as good as fresh data; beyond that returns vanish. Relevant as high-quality web text runs out. |

---

## 1.8 Data: the real moat

| Paper | Summary |
|---|---|
| :material-star: [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — Penedo et al., 2024 | 15T-token open web corpus with fully documented filtering/dedup ablations, plus **FineWeb-Edu** (an LLM-trained "educational value" classifier). The best guide to web-data pipelines. |
| :material-star: [DataComp-LM (DCLM)](https://arxiv.org/abs/2406.11794) — Li et al., 2024 | A benchmark for data curation; shows a simple fastText quality classifier is a huge win. |
| :material-book-open-variant: [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) — Gunasekar et al., 2023 | Small models trained on high-quality and synthetic "textbook" data punch far above their size — the start of the synthetic pre-training data trend. |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — OLMo Team (AI2), 2025 | Fully open (data, code, checkpoints) recipe including **mid-training / annealing** on high-quality data at the end of pre-training — now standard practice. |

Key practices in 2025–2026 pipelines: URL/language filtering → heuristic quality filters → model-based quality scoring → fuzzy dedup (MinHash) → decontamination → domain **mixture** tuning (code, math, multilingual) → **annealing** on curated + synthetic data → long-context stage.

---

## 1.9 Training at scale: parallelism, precision, optimizers

| Paper | Summary |
|---|---|
| :material-star: [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | **Tensor parallelism**: split matmuls inside each layer across GPUs. |
| :material-star: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019 | Shard optimizer states, gradients and params across data-parallel ranks (= PyTorch **FSDP**). |
| :material-book-open-variant: [Efficient Large-Scale LM Training on GPU Clusters (PTD-P)](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021 | How to compose tensor + pipeline + data parallelism; interleaved 1F1B pipeline schedule. |
| :material-book-open-variant: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022 | **Sequence parallelism** + selective activation checkpointing. |
| :material-book-open-variant: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2017 | FP16/BF16 compute with FP32 master weights and loss scaling. FP8 (DeepSeek-V3) and FP4/MXFP4 are the current frontier. |
| :material-book-open-variant: [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2017 | The default optimizer for LLMs. |
| :material-new-box: [Muon is Scalable for LLM Training (Moonlight)](https://arxiv.org/abs/2502.16982) — Liu et al. (Moonshot AI), 2025 | Muon orthogonalizes momentum updates (Newton–Schulz); ~2× compute efficiency vs AdamW at scale. Kimi K2 trained with a stabilized variant ("MuonClip"). |
| :material-new-box: [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — Gloeckle et al., 2024 | Extra heads predict several future tokens — denser training signal and a free draft model for speculative decoding (used in DeepSeek-V3). |

Hardware-level read: **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google DeepMind) — the best free book on rooflines, sharding and TPU/GPU math.

---

## 1.10 Frontier open models to study (2025–2026) :material-new-box:

Reading the technical reports of these models is the fastest way to stay current. Look for: total vs active params, attention type, MoE config, context length, data size, optimizer, precision.

| Report | What to look for |
|---|---|
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) / V3.1 / V3.2 | MLA + fine-grained MoE + aux-loss-free balancing + MTP + FP8; V3.2 adds DeepSeek Sparse Attention. |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Alibaba, 2025 | Dense and MoE families (0.6B → 235B-A22B), 36T tokens, QK-norm, unified "thinking / non-thinking" modes. Qwen3-Next adds hybrid Gated DeltaNet + very sparse MoE. |
| [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | 1T total / 32B active MoE trained with MuonClip; heavy agentic data synthesis. |
| [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) — OpenAI, 2025 | MoE, alternating dense/banded-window attention, attention sinks, MXFP4-quantized MoE weights, `o200k_harmony` tokenizer. |
| [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Google DeepMind, 2025 | 5:1 local:global attention to shrink KV cache, QK-norm, distillation from a larger teacher, multimodal. |
| Llama 4 ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) — Meta, 2025 | MoE (Scout / Maverick), native multimodality via early fusion, interleaved RoPE/NoPE layers ("iRoPE") for long context. |
| [OLMo 2](https://arxiv.org/abs/2501.00656) — AI2 | Fully reproducible end-to-end; best for learning since everything is released. |

Useful living references: Sebastian Raschka's **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** and Hugging Face's **[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)**.

---

## 1.11 Hands-on projects

- [ ] Watch Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE); implement BPE yourself.
- [ ] Train a small GPT with [nanoGPT](https://github.com/karpathy/nanoGPT) / [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) (see how RoPE, QK-norm, Muon, ReLU² shave training time).
- [ ] Go end-to-end with [nanochat](https://github.com/karpathy/nanochat): tokenizer → pre-train → SFT → RL → chat UI for ~$100.
- [ ] Upgrade your GPT block: LayerNorm → RMSNorm, GELU → SwiGLU, learned pos-emb → RoPE, MHA → GQA. Measure loss and KV-cache size.
- [ ] Implement a toy top-2 MoE layer with a load-balancing loss; plot expert utilization.
- [ ] Read Stanford **[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)** assignments.

## 1.12 Self-check

??? question "Why does GQA/MLA matter more for inference than for training?"
    Training is compute-bound over full sequences; decoding is memory-bandwidth-bound and must read the entire KV cache each step. Shrinking K/V directly increases batch size and tokens/s at serving time.

??? question "Why do labs train 8B models on 15T+ tokens when Chinchilla says ~160B is optimal?"
    Chinchilla minimizes *training* compute for a given loss. Deployed models are served billions of times, so a smaller model trained longer has lower lifetime cost.

??? question "What problem do shared experts and aux-loss-free balancing solve?"
    Shared experts capture common knowledge so routed experts can specialize; bias-based balancing keeps experts evenly loaded without an auxiliary loss term that hurts the LM objective.

??? question "Why are hybrid (linear + full) attention models attractive?"
    Linear/SSM layers give O(1) memory per token and cheap long context, while a few full-attention layers retain precise retrieval that pure linear models struggle with.
