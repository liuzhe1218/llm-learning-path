# Part 1 · Pre-training & Model Architecture

> **Goal:** understand how a base LLM is built, from the definition of a language model to a trillion-parameter Mixture-of-Experts trained on tens of trillions of tokens, and be able to explain *why* each architectural choice in a 2025–2026 frontier model exists.

A large language model starts life as a **base model**: a neural network trained to do one thing, which is to predict the next token of text. Pre-training is that phase. It uses almost all of the compute that ever goes into a model (post-training in [Part 2](../part2/index.md) is usually a small fraction on top), and it fixes the model's architecture, its vocabulary, its context length and most of its knowledge. If you understand pre-training, every later topic (fine-tuning, RL, inference serving) becomes a story about modifying or running an object you already understand.

This part builds that understanding from the ground up. We start with probability: what it means for a model to assign probabilities to text, and why minimizing cross-entropy on next-token prediction is a sensible objective. We then turn text into integers (tokenization), build a GPT from scratch in PyTorch, and upgrade it step by step into the block used by Llama, Qwen and DeepSeek: RMSNorm, SwiGLU, RoPE, grouped-query or latent attention, and Mixture-of-Experts layers. After that we face the two walls every lab runs into: long sequences (attention is quadratic) and scale (how big a model, how much data, and how to spread training over thousands of GPUs).

The last chapter puts it all together by dissecting the published architectures of Llama 3, DeepSeek-V3, Qwen3, gpt-oss, Kimi K2 and Gemma 3. By then each row of their spec tables should read like a set of deliberate trade-offs rather than a list of magic numbers.

## Prerequisites and time

| You need | Level |
|---|---|
| Linear algebra | matrix multiplication, transposes, dot products, rotations |
| Probability | conditional probability, expectations, logarithms |
| Python / PyTorch | tensors, `nn.Module`, autograd, a training loop |

**Estimated time:** about 25–35 hours of reading and coding for the whole part (2–4 hours per chapter), plus optional projects. Chapters 3 and 4 are the core: budget the most time there.

## Chapter map

| # | Chapter | One-line summary |
|---|---|---|
| 1.1 | [Language modeling](01-language-modeling.md) | The chain rule, next-token prediction, cross-entropy and perplexity, n-grams vs neural LMs, sampling. |
| 1.2 | [Tokenization](02-tokenization.md) | How text becomes integers: BPE traced by hand, byte-level BPE, vocabulary trade-offs, quirks. |
| 1.3 | [The Transformer](03-transformer.md) | Attention derived with shapes, the causal mask, multi-head attention, a full minimal GPT, parameter and FLOP counting. |
| 1.4 | [The modern block](04-modern-block.md) | RMSNorm, SwiGLU, RoPE, MHA → MQA → GQA → MLA, QK-norm, tied embeddings: the Llama/Qwen block in code. |
| 1.5 | [Mixture-of-Experts](05-mixture-of-experts.md) | Sparse compute: routers, load balancing, shared experts, expert parallelism, total vs active parameters. |
| 1.6 | [Long context & efficient attention](06-long-context-and-efficient-attention.md) | FlashAttention, RoPE scaling (PI/NTK/YaRN), sliding windows, sinks, linear attention / SSMs, hybrids, sparse attention. |
| 1.7 | [Scaling laws & data](07-scaling-laws-and-data.md) | Power laws, $C \approx 6ND$, Kaplan vs Chinchilla, over-training, the web-data pipeline, mixtures, annealing. |
| 1.8 | [Training at scale](08-training-at-scale.md) | Memory accounting, ZeRO/FSDP, tensor/pipeline/sequence/expert parallelism, mixed precision, optimizers, instabilities. |
| 1.9 | [Frontier model case studies](09-frontier-model-case-studies.md) | Llama 3 405B, DeepSeek-V3, Qwen3-235B-A22B, gpt-oss-120b, Kimi K2, Gemma 3 compared and explained. |

## Mental model

```text
                    ┌──────────────────────────── 1.7 Scaling laws & data ───────────────────────────┐
                    │      how many parameters N, how many tokens D, which tokens, for budget C       │
                    └─────────────────────────────────────────┬───────────────────────────────────────┘
                                                              │
 raw text ──► 1.2 Tokenizer ──► token ids ──► MODEL ──► logits over vocab ──► 1.1 loss = cross-entropy
                                                │                                (next-token prediction)
                    ┌───────────────────────────┴──────────────────────────────┐
                    │ 1.3 Transformer: embeddings → N × block → unembedding    │
                    │ 1.4 modern block: RMSNorm, RoPE, GQA/MLA, SwiGLU         │
                    │ 1.5 FFN → MoE (router + experts)                         │
                    │ 1.6 attention that scales to long context                │
                    └───────────────────────────┬──────────────────────────────┘
                                                │
                    1.8 Training at scale: parallelism, precision, optimizer, stability
                                                │
                    1.9 Case studies: every choice above, in real frontier models
```

Read it top to bottom as a pipeline: data is tokenized, the model maps token ids to a probability distribution over the next token, and the loss compares that distribution with what actually came next. Chapters 1.3–1.6 are about the box in the middle; 1.7 and 1.8 are about feeding and running it at scale.

## How to study this part

1. **Read 1.1 → 1.4 in order.** Each chapter builds on the previous one. Type the code in rather than copy-pasting: a minimal GPT is about 100 lines, and writing it once removes most of the mystery.
2. **Then 1.5 and 1.6** in either order; both modify the block you built in 1.4.
3. **1.7 and 1.8** are more about systems and economics than about the network itself. They are easier to absorb once you can count parameters and FLOPs (end of 1.3).
4. **Finish with 1.9** and try to predict each model's design choices before reading the explanation.

Hands-on projects (in increasing order of effort):

- [ ] Watch Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE); implement BPE yourself (chapter 1.2 has a reference implementation).
- [ ] Train the minimal GPT from chapter 1.3 on a small text file, then train a small GPT with [nanoGPT](https://github.com/karpathy/nanoGPT). Look at the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) to see how RoPE, QK-norm and Muon shave training time.
- [ ] Upgrade your GPT block: LayerNorm → RMSNorm, GELU → SwiGLU, learned positions → RoPE, MHA → GQA. Measure loss and KV-cache size (chapter 1.4).
- [ ] Implement a toy top-2 MoE layer with a load-balancing loss and plot expert utilization (chapter 1.5).
- [ ] Go end to end with [nanochat](https://github.com/karpathy/nanochat): tokenizer → pre-train → SFT → RL → chat UI.
- [ ] Work through the assignments of Stanford **[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)**.

!!! tip "Two living references"
    Sebastian Raschka's **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** and Hugging Face's **[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)** are good companions to this part, especially to chapters 1.4 and 1.9.
