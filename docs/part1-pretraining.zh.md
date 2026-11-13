# 第 1 部分 · 预训练与模型架构

> **目标：** 理解基座 LLM 是如何构建的——从 token 到在数十万亿 token 上训练的万亿参数混合专家（Mixture-of-Experts）模型——并且能够解释 2025–2026 年前沿模型中每一个架构选择背后的原因。

**图例：** :material-star: 必读 · :material-book-open-variant: 深入阅读 · :material-new-box: 2025–2026 前沿

---

## 1.1 全局视角

预训练 = 在海量文本语料上做**下一个 token 预测**：

\[
\mathcal{L}(\theta) = -\sum_{t} \log p_\theta(x_t \mid x_{<t})
\]

本部分其余的内容，都是在回答同一个问题：如何让这件事在每 FLOP 上做得*更好*：

| 杠杆 | 它回答的问题 | 章节 |
|---|---|---|
| 分词（Tokenization） | 什么是 "token"？ | 1.2 |
| 架构 | 什么函数把上下文映射为下一个 token 的分布？ | 1.3–1.6 |
| Scaling law（缩放定律） | 给定预算，模型该多大、该训练多少 token？ | 1.7 |
| 数据 | 用哪些 token？ | 1.8 |
| 系统与优化 | 如何在数千张 GPU 上稳定地训练？ | 1.9 |

现代 decoder-only 模块（Llama/Qwen/DeepSeek 风格）长这样：

```text
x ─► RMSNorm ─► Attention (GQA or MLA, RoPE, optional QK-norm) ─► + ─►
     RMSNorm ─► FFN (SwiGLU dense, or MoE: router → top-k experts + shared expert) ─► + ─► next block
```

---

## 1.2 分词（Tokenization）

- **字节对编码（BPE）** 把最高频的符号对不断合并进词表；字节级 BPE（从 GPT-2 开始）保证不会出现词表外（OOV）token。
- 词表规模一路增长：32K（Llama 2）→ 128K（Llama 3）→ 约 150K–200K+（Qwen、gpt-oss 的 `o200k`）。词表越大，序列越短，多语言压缩率也越好，代价是 embedding / 输出层变大。

| 论文 | 简介 |
|---|---|
| :material-star: [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2015 | 把 BPE 引入 NLP：从字符出发，反复合并频率最高的符号对。几乎所有当今 LLM 分词器的基础。 |
| :material-book-open-variant: [SentencePiece](https://arxiv.org/abs/1808.06226) — Kudo & Richardson, 2018 | 与语言无关的分词器，把文本当作原始 Unicode 序列处理（包括空白符），在多语言和中日韩文本占比高的模型中很常用。 |

---

## 1.3 Transformer 与 GPT 配方

| 论文 | 简介 |
|---|---|
| :material-star: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 | Transformer 的开山之作：缩放点积注意力、多头注意力、残差连接、位置编码。本指南中的每个模型都是它的后代。重点读第 3 节。 |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | Decoder-only 语言模型 + pre-norm + 大规模网页语料 → 零样本任务迁移。确立了"把下一个 token 预测做大就行"的配方。 |
| :material-star: [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020 | 1750 亿参数；上下文学习（few-shot）随规模涌现。重点读它的问题框架和评测方法。 |
| :material-book-open-variant: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al., 2020 | 解释了为什么 **Pre-LN**（在子层之前做归一化）无需 warmup 技巧也能稳定训练——如今已成为标配。 |

---

## 1.4 "现代模块"：2025 年每个模型都在用的组件

| 论文 | 简介 |
|---|---|
| :material-star: [RMSNorm](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019 | 去掉均值中心化的 LayerNorm：更省算力，效果相当。Llama、Qwen、DeepSeek、Gemma 的标准配置。 |
| :material-star: [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 | 门控 FFN `(Swish(xW) ⊙ xV)W₂` 在同等算力下优于 ReLU/GELU FFN。如今各处的默认 FFN。 |
| :material-star: [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al., 2021 | 通过旋转 query/key 向量来编码位置，使注意力只依赖*相对*距离。默认的位置编码方案，同时也是各种上下文扩展方法的切入点。 |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | 所有 query head 共享同一个 K/V head → KV cache 缩小为原来的 1/头数。推理收益很大，质量损失很小。 |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | 多组 query head 共享一个 K/V head：在 MHA 和 MQA 之间取得质量与显存的最佳平衡点。Llama 2-70B 起的 Llama 系列、Qwen、Mistral、gpt-oss 都在用。 |
| :material-book-open-variant: [Scaling Vision Transformers to 22B Parameters](https://arxiv.org/abs/2302.05442) — Dehghani et al., 2023 | 最早采用 **QK-norm**（对 query/key 做归一化）以防止注意力 logit 爆炸的工作之一；现在在 LLM 中很常见（OLMo 2、Qwen3、Gemma 3）。 |
| :material-book-open-variant: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023 | 把 Pre-RMSNorm + SwiGLU + RoPE 打包成开源模型的"标准模块"，并证明在更多 token 上训练的小模型可以胜过更大的模型。 |
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | 90 多页，覆盖一整条 405B 稠密模型的完整流水线：数据配比、由 scaling law 驱动的规模选择、4D 并行、长上下文阶段、后训练。是了解"前沿实验室如何做模型"的最佳单篇参考。 |

---

## 1.5 混合专家（Mixture-of-Experts, MoE）

MoE 用大量**专家（expert）**取代稠密 FFN，由**路由器（router）**把每个 token 发送给 top-*k* 个专家。总参数量变大，而每个 token 的*激活*参数量（以及 FLOPs）仍然很小。2024 年以来，几乎所有前沿开源模型都是 MoE。

| 论文 | 简介 |
|---|---|
| :material-book-open-variant: [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017 | 最早的 top-k 门控 MoE，并引入负载均衡损失（基于 LSTM）。 |
| :material-book-open-variant: [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020 | 6000 亿参数规模的 MoE Transformer，提出跨设备的**专家并行（expert parallelism）**。 |
| :material-star: [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021 | Top-1 路由、capacity factor、辅助负载均衡损失、bf16 稳定性技巧。对 MoE 工程实践最清晰的入门读物。 |
| :material-star: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024 | 8 个专家，top-2，总参数 47B / 激活 13B；让开源 MoE 成为主流的模型。 |
| :material-star: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024 | **细粒度专家**（大量小专家、更大的 k）+ 每个 token 都会用到的**共享专家**。DeepSeek-V2/V3、Qwen3-MoE、Kimi K2 等模型都采用了这一设计。 |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 总参数 671B / 激活 37B。提出**无辅助损失的负载均衡**（逐专家的 bias 项）、**多 token 预测（multi-token prediction）**、FP8 训练和 DualPipe。可以说是这个时代影响力最大的开源架构报告。 |

---

## 1.6 注意力效率与长上下文

### 更小的 KV cache

| 论文 | 简介 |
|---|---|
| :material-star: [DeepSeek-V2 (Multi-head Latent Attention, MLA)](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | 把每个 token 的 K/V 压缩成一个低秩的潜向量（latent），KV cache 远小于 GQA，同时效果与 MHA 相当。通过"解耦 RoPE（decoupled RoPE）"来兼容旋转位置编码。DeepSeek-V3/R1 和 Kimi K2 都在使用。 |

### 更快的精确注意力（IO-aware kernel）

| 论文 | 简介 |
|---|---|
| :material-star: [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | 对注意力做分块（tiling），使 N×N 矩阵不必写入 HBM；结果精确、速度更快、显存 O(N)。长上下文训练之所以负担得起，这是关键原因。 |
| :material-book-open-variant: [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023 | 在 warp / thread block 之间做更好的工作划分；比 v1 快约 2 倍。 |
| :material-book-open-variant: [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | 针对 Hopper 架构：异步（TMA、warp specialization）与 FP8。 |

### 上下文长度扩展

| 论文 | 简介 |
|---|---|
| :material-star: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — Peng et al., 2023 | 对 RoPE 频率做非均匀缩放（"NTK-by-parts"），再加上注意力温度；只需很少的继续训练就能扩展上下文。应用广泛（Qwen、DeepSeek、gpt-oss）。 |
| :material-book-open-variant: [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023 | 把序列切分到多个设备上，让 K/V 块沿环传递——这是百万 token 级训练所用**上下文并行（context parallelism）**的基础。 |

### 稀疏、线性与混合注意力 :material-new-box:

2025 年的趋势：保留少数几层全注意力，把其余层做得很便宜。

| 论文 | 简介 |
|---|---|
| :material-star: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023 | 输入相关（input-dependent）的状态空间模型；训练 O(N)，推理时每个 token 的显存开销 O(1)。 |
| :material-book-open-variant: [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) — Dao & Gu, 2024 | 证明 SSM 与线性注意力互为对偶；给出更快、对矩阵乘更友好的算法。Mamba 层出现在 Jamba、NVIDIA Nemotron-H 等混合模型中。 |
| :material-new-box: [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Yang et al., 2024 | 为线性注意力把门控（遗忘）与 delta rule（精确的记忆更新）结合起来。在 Qwen3-Next 中以混合形式使用（约 3 层线性 : 1 层全注意力）。 |
| :material-new-box: [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) — MiniMax, 2025 | 首个前沿规模的混合线性注意力 MoE 模型（456B），上下文可达 1M–4M。 |
| :material-new-box: [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek), 2025 | 与硬件对齐、可训练的稀疏注意力（压缩 + 选择 + 滑动窗口三个分支）。它的后继者 DeepSeek Sparse Attention（DSA，用于 DeepSeek-V3.2）使用一个轻量的"indexer"来挑选 top-k 个 token。 |
| :material-book-open-variant: [Differential Transformer](https://arxiv.org/abs/2410.05258) — Ye et al., 2024 | 注意力 = 两个 softmax 图之差，用来抵消注意力噪声；长上下文检索效果更好。 |

另外值得一提：**滑动窗口 / 局部–全局交替**（Mistral、Gemma 2/3、gpt-oss）和 **attention sink**（gpt-oss 中逐 head 可学习的 sink logit）——StreamingLLM 见第 3 部分。

---

## 1.7 Scaling law（缩放定律）

| 论文 | 简介 |
|---|---|
| :material-star: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020 | 损失随参数量、数据量和算力呈幂律下降。开启了"规模就是一切"的时代。 |
| :material-star: [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022 | 修正了 Kaplan 的结论：参数量和 token 数应大致等比例增长（计算最优约为每个参数 20 个 token）。现代模型会刻意**过度训练（over-train）**，远超这个比例（Llama 3 8B：约 15T token），因为推理成本取决于模型大小，而不是训练 token 数。 |
| :material-book-open-variant: [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (μP)](https://arxiv.org/abs/2203.03466) — Yang et al., 2022 | 一种参数化方式，使最优学习率可以从小模型迁移到大模型——小模型调参，大模型训练。 |
| :material-book-open-variant: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023 | 数据重复最多约 4 个 epoch，效果几乎和新数据一样好；超过之后收益消失。在高质量网页文本逐渐耗尽的背景下尤为重要。 |

---

## 1.8 数据：真正的护城河

| 论文 | 简介 |
|---|---|
| :material-star: [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — Penedo et al., 2024 | 15T token 的开源网页语料，过滤 / 去重的消融实验全部公开；另有 **FineWeb-Edu**（用 LLM 训练出的"教育价值"分类器）。了解网页数据流水线的最佳指南。 |
| :material-star: [DataComp-LM (DCLM)](https://arxiv.org/abs/2406.11794) — Li et al., 2024 | 一个数据清洗（data curation）基准；表明一个简单的 fastText 质量分类器就能带来巨大收益。 |
| :material-book-open-variant: [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) — Gunasekar et al., 2023 | 在高质量数据和合成的"教科书"数据上训练的小模型，表现远超其规模——合成预训练数据这一趋势的起点。 |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — OLMo Team (AI2), 2025 | 完全开源（数据、代码、checkpoint）的训练方案，包括在预训练末期用高质量数据做的 **mid-training / annealing**——如今已是标准做法。 |

2025–2026 年数据流水线的关键实践：URL / 语言过滤 → 启发式质量过滤 → 基于模型的质量打分 → 模糊去重（MinHash）→ 去污染（decontamination）→ 领域**配比**调优（代码、数学、多语言）→ 在精选数据 + 合成数据上做 **annealing** → 长上下文阶段。

---

## 1.9 大规模训练：并行、精度与优化器

| 论文 | 简介 |
|---|---|
| :material-star: [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | **张量并行（tensor parallelism）**：把每一层内部的矩阵乘法切分到多张 GPU 上。 |
| :material-star: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019 | 在数据并行的各 rank 之间切分优化器状态、梯度和参数（即 PyTorch 的 **FSDP**）。 |
| :material-book-open-variant: [Efficient Large-Scale LM Training on GPU Clusters (PTD-P)](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021 | 如何把张量并行、流水线并行和数据并行组合起来；interleaved 1F1B 流水线调度。 |
| :material-book-open-variant: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022 | **序列并行（sequence parallelism）** + 选择性激活检查点（selective activation checkpointing）。 |
| :material-book-open-variant: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2017 | FP16/BF16 计算 + FP32 主权重 + loss scaling。FP8（DeepSeek-V3）和 FP4/MXFP4 是当前的前沿。 |
| :material-book-open-variant: [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2017 | LLM 的默认优化器。 |
| :material-new-box: [Muon is Scalable for LLM Training (Moonlight)](https://arxiv.org/abs/2502.16982) — Liu et al. (Moonshot AI), 2025 | Muon 对动量更新做正交化（Newton–Schulz 迭代）；在大规模下相比 AdamW 约有 2 倍的算力效率。Kimi K2 使用了它的稳定化变体（"MuonClip"）训练。 |
| :material-new-box: [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — Gloeckle et al., 2024 | 额外的预测头同时预测多个未来 token——训练信号更密集，还能白得一个用于投机解码的 draft 模型（DeepSeek-V3 采用）。 |

硬件层面的阅读材料：**[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)**（Google DeepMind）——讲解 roofline、分片（sharding）以及 TPU/GPU 计算的最佳免费书籍。

---

## 1.10 值得研究的前沿开源模型（2025–2026）:material-new-box:

阅读这些模型的技术报告，是保持跟进前沿最快的方式。阅读时重点关注：总参数 vs 激活参数、注意力类型、MoE 配置、上下文长度、数据量、优化器、精度。

| 报告 | 看点 |
|---|---|
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) / V3.1 / V3.2 | MLA + 细粒度 MoE + 无辅助损失均衡 + MTP + FP8；V3.2 加入了 DeepSeek Sparse Attention。 |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Alibaba, 2025 | 稠密和 MoE 两个系列（0.6B → 235B-A22B），36T token，QK-norm，统一的"思考 / 非思考"模式。Qwen3-Next 加入了混合 Gated DeltaNet + 极稀疏 MoE。 |
| [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | 总参数 1T / 激活 32B 的 MoE，用 MuonClip 训练；大量使用面向 Agent 的数据合成。 |
| [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) — OpenAI, 2025 | MoE，稠密注意力与带状窗口注意力交替，attention sink，MoE 权重采用 MXFP4 量化，`o200k_harmony` 分词器。 |
| [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Google DeepMind, 2025 | 5:1 的局部:全局注意力以缩小 KV cache，QK-norm，从更大的教师模型蒸馏，多模态。 |
| Llama 4 ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) — Meta, 2025 | MoE（Scout / Maverick），通过 early fusion 实现原生多模态，交错使用 RoPE/NoPE 层（"iRoPE"）以支持长上下文。 |
| [OLMo 2](https://arxiv.org/abs/2501.00656) — AI2 | 端到端完全可复现；因为一切都开源，最适合用来学习。 |

常更新的参考资料：Sebastian Raschka 的 **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** 和 Hugging Face 的 **[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)**。

---

## 1.11 动手项目

- [ ] 观看 Karpathy 的 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 和 [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)；自己实现一遍 BPE。
- [ ] 用 [nanoGPT](https://github.com/karpathy/nanoGPT) / [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) 训练一个小型 GPT（看看 RoPE、QK-norm、Muon、ReLU² 是如何缩短训练时间的）。
- [ ] 用 [nanochat](https://github.com/karpathy/nanochat) 走通完整流程：分词器 → 预训练 → SFT → RL → 聊天界面，成本约 100 美元。
- [ ] 升级你的 GPT 模块：LayerNorm → RMSNorm，GELU → SwiGLU，可学习位置编码 → RoPE，MHA → GQA。测量 loss 和 KV cache 大小。
- [ ] 实现一个带负载均衡损失的玩具版 top-2 MoE 层；画出专家利用率。
- [ ] 完成斯坦福 **[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)** 的作业。

## 1.12 自测

??? question "为什么 GQA/MLA 对推理的意义比对训练更大？"
    训练是在完整序列上进行的，受算力限制（compute-bound）；解码则受显存带宽限制（memory-bandwidth-bound），每一步都必须读取整个 KV cache。缩小 K/V 能直接增大 batch size，提升推理服务时的 tokens/s。

??? question "Chinchilla 认为 8B 模型的最优训练量约为 160B token，为什么各家实验室还要用 15T+ token 来训练 8B 模型？"
    Chinchilla 最小化的是达到给定 loss 所需的*训练*算力。而部署后的模型要被调用数十亿次，所以一个更小、训练更久的模型，全生命周期成本更低。

??? question "共享专家和无辅助损失均衡分别解决了什么问题？"
    共享专家负责承载通用知识，让路由专家可以专注于各自的专长；基于 bias 的均衡能让各专家负载均匀，同时避免引入会损害语言建模目标的辅助损失项。

??? question "为什么混合（线性 + 全注意力）模型很有吸引力？"
    线性 / SSM 层让每个 token 的显存开销为 O(1)，长上下文成本低；而少量全注意力层则保留了精确检索能力，这正是纯线性模型很难做到的。
