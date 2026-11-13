# 第一部分 · 预训练与模型架构

> **目标：** 理解基座大模型是如何构建的：从语言模型的定义，一直到在数十万亿 token 上训练的万亿参数混合专家（Mixture-of-Experts）模型；并且能够解释 2025–2026 年前沿模型中每一个架构选择*为什么*存在。

大语言模型的起点是**基座模型**（base model）：一个只被训练去做一件事的神经网络，即预测文本的下一个 token。预训练就是这个阶段。它消耗了一个模型整个生命周期中几乎全部的算力（[第二部分](../part2/index.md)的后训练通常只是在其之上的一小部分），并且决定了模型的架构、词表、上下文长度以及绝大部分知识。如果你理解了预训练，后面的每个主题（微调、强化学习、推理服务）就都成了对一个你已经熟悉的对象进行修改或运行的故事。

这一部分从零开始建立这种理解。我们从概率出发：模型给文本分配概率意味着什么，以及为什么在下一个 token 预测上最小化交叉熵是一个合理的目标。接着我们把文本变成整数（分词），用 PyTorch 从零搭一个 GPT，再一步步把它升级成 Llama、Qwen 和 DeepSeek 所使用的模块：RMSNorm、SwiGLU、RoPE、分组查询或潜在注意力，以及混合专家层。之后我们面对每个实验室都会撞上的两堵墙：长序列（注意力是平方复杂度）和规模（模型多大、数据多少、以及如何把训练分摊到成千上万块 GPU 上）。

最后一章把所有内容串起来，剖析 Llama 3、DeepSeek-V3、Qwen3、gpt-oss、Kimi K2 和 Gemma 3 已公开的架构。到那时，它们规格表里的每一行读起来都应该像一组经过深思熟虑的权衡，而不是一串神奇的数字。

## 先修知识与学习时间 {#prerequisites-and-time}

| 你需要 | 程度 |
|---|---|
| 线性代数 | 矩阵乘法、转置、点积、旋转 |
| 概率论 | 条件概率、期望、对数 |
| Python / PyTorch | 张量、`nn.Module`、autograd、训练循环 |

**预计时间：** 整个部分约需 25–35 小时的阅读和编码（每章 2–4 小时），外加可选的实践项目。第 3、4 章是核心，请在这里投入最多的时间。

## 章节地图 {#chapter-map}

| # | 章节 | 一句话概述 |
|---|---|---|
| 1.1 | [语言建模](01-language-modeling.md) | 链式法则、下一个 token 预测、交叉熵与困惑度、n-gram 与神经语言模型、采样。 |
| 1.2 | [分词](02-tokenization.md) | 文本如何变成整数：手工推演 BPE、字节级 BPE、词表大小的权衡、各种怪现象。 |
| 1.3 | [Transformer](03-transformer.md) | 带形状推导的注意力、因果掩码、多头注意力、完整的最小 GPT、参数量与 FLOPs 计数。 |
| 1.4 | [现代 Transformer 块](04-modern-block.md) | RMSNorm、SwiGLU、RoPE、MHA → MQA → GQA → MLA、QK-norm、权重共享嵌入：用代码实现 Llama/Qwen 的块。 |
| 1.5 | [混合专家](05-mixture-of-experts.md) | 稀疏计算：路由器、负载均衡、共享专家、专家并行、总参数量与激活参数量。 |
| 1.6 | [长上下文与高效注意力](06-long-context-and-efficient-attention.md) | FlashAttention、RoPE 缩放（PI/NTK/YaRN）、滑动窗口、注意力汇聚点、线性注意力 / SSM、混合架构、稀疏注意力。 |
| 1.7 | [缩放定律与数据](07-scaling-laws-and-data.md) | 幂律、$C \approx 6ND$、Kaplan 与 Chinchilla、过度训练、网页数据流水线、数据配比、退火。 |
| 1.8 | [大规模训练](08-training-at-scale.md) | 显存核算、ZeRO/FSDP、张量/流水线/序列/专家并行、混合精度、优化器、训练不稳定。 |
| 1.9 | [前沿模型案例研究](09-frontier-model-case-studies.md) | 对比并解释 Llama 3 405B、DeepSeek-V3、Qwen3-235B-A22B、gpt-oss-120b、Kimi K2、Gemma 3。 |

## 心智模型 {#mental-model}

```text
                    ┌──────────────────────────── 1.7 缩放定律与数据 ────────────────────────────────┐
                    │      参数量 N 多大、token 数 D 多少、用哪些 token，在预算 C 之下                  │
                    └─────────────────────────────────────────┬───────────────────────────────────────┘
                                                              │
 原始文本 ──► 1.2 分词器 ──► token id ──► 模型 ──► 词表上的 logits ──► 1.1 损失 = 交叉熵
                                                │                                （下一个 token 预测）
                    ┌───────────────────────────┴──────────────────────────────┐
                    │ 1.3 Transformer：嵌入 → N × 块 → 反嵌入                    │
                    │ 1.4 现代块：RMSNorm、RoPE、GQA/MLA、SwiGLU                 │
                    │ 1.5 FFN → MoE（路由器 + 专家）                             │
                    │ 1.6 能扩展到长上下文的注意力                                │
                    └───────────────────────────┬──────────────────────────────┘
                                                │
                    1.8 大规模训练：并行、精度、优化器、稳定性
                                                │
                    1.9 案例研究：以上每个选择，在真实前沿模型中的样子
```

请把它当作一条流水线自上而下阅读：数据被分词，模型把 token id 映射为下一个 token 的概率分布，损失则把这个分布与实际出现的下一个 token 作比较。1.3–1.6 章讲的是中间那个方框；1.7 和 1.8 讲的是如何在大规模下喂养并运行它。

## 如何学习这一部分 {#how-to-study-this-part}

1. **按顺序阅读 1.1 → 1.4。** 每章都建立在前一章之上。请亲手把代码敲一遍，而不是复制粘贴：一个最小的 GPT 大约 100 行，写一遍就能消除大部分神秘感。
2. **然后读 1.5 和 1.6**，顺序不限；两章都是在修改你在 1.4 中搭好的块。
3. **1.7 和 1.8** 更多是关于系统和经济学，而不是网络本身。当你能数清参数量和 FLOPs（1.3 末尾）之后，它们会更容易消化。
4. **最后读 1.9**，在阅读解释之前，先试着预测每个模型的设计选择。

动手项目（按投入由小到大）：

- [ ] 观看 Karpathy 的 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 和 [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)；自己实现 BPE（1.2 章有参考实现）。
- [ ] 在一个小文本文件上训练 1.3 章的最小 GPT，然后用 [nanoGPT](https://github.com/karpathy/nanoGPT) 训练一个小 GPT。看看 [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)，了解 RoPE、QK-norm 和 Muon 是如何缩短训练时间的。
- [ ] 升级你的 GPT 块：LayerNorm → RMSNorm，GELU → SwiGLU，学习式位置 → RoPE，MHA → GQA。测量损失和 KV cache 大小（1.4 章）。
- [ ] 实现一个玩具版 top-2 MoE 层，带负载均衡损失，并绘制专家利用率（1.5 章）。
- [ ] 用 [nanochat](https://github.com/karpathy/nanochat) 走完端到端流程：分词器 → 预训练 → SFT → RL → 聊天界面。
- [ ] 完成斯坦福 **[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)** 的作业。

!!! tip "两份持续更新的参考资料"
    Sebastian Raschka 的 **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** 和 Hugging Face 的 **[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)** 是这一部分很好的配套读物，尤其适合搭配 1.4 和 1.9 章。
