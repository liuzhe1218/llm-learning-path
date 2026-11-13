# LLM 学习路线

一条精心挑选、带有明确观点的学习路线，讲清现代大语言模型是如何**构建**、**对齐**和**部署**的——附原始论文链接，并简要说明每篇论文的贡献以及它至今仍然重要的原因。

!!! tip "Language / 语言"
    使用页眉中的语言切换器（🌐 图标）即可在**中文**和 **English** 之间切换。每个页面都有两种语言版本。

!!! note "划线与提问"
    在任意页面选中一段文字，点击 **Annotate**（写笔记或提问）或 **Highlight**（划线）。批注功能由 [Hypothesis](https://web.hypothes.is/) 提供：注册一次免费账号后，你的划线会固定在对应句子上，下次打开（换设备也一样）都会显示。点击页面右上边缘的箭头可展开侧栏查看全部批注。选择 **Only Me** 保存为私人笔记，选择 **Public** 则可以把问题分享给其他读者。

## 路线结构 {#how-the-path-is-organized}

| 部分 | 重点 | 学完后你将能够… |
|---|---|---|
| [**第 1 部分 · 预训练与架构**](part1/index.md) | Transformer 内部结构、分词、scaling law（缩放定律）、数据、MoE、长上下文、高效注意力、分布式训练 | 讲清楚 2025–2026 年前沿模型（如 DeepSeek-V3、Qwen3、Llama 4、gpt-oss）的每个模块，以及为什么要这样设计 |
| [**第 2 部分 · 后训练与推理**](part2/index.md) | SFT、RLHF、DPO 及其变体、基于可验证奖励的 RL（GRPO/DAPO/GSPO）、推理模型、test-time scaling、Agent/工具调用训练、PEFT | 设计一套后训练方案，把基座模型变成对话 / 推理 / Agent 模型 |
| [**第 3 部分 · 推理系统与基础设施**](part3/index.md) | KV cache、PagedAttention、continuous batching、前缀缓存、投机解码、量化、prefill/decode 分离、MoE 推理服务 | 分析推理服务的延迟 / 吞吐 / 成本，并选择合适的引擎和优化手段 |
| [**第 4 部分 · LLM 与广告推荐系统**](part4/index.md) | 工业界推荐/广告级联架构、CTR 模型、推荐系统的 scaling law（HSTU、Wukong、RankMixer）、语义 ID 与生成式召回、端到端生成式推荐（OneRec）、LLM 作为特征提供者 / 用户模型、生成式出价 | 讲清 LLM 时代的思想如何重塑排序与推荐，并能设计 LLM 增强或生成式的推荐系统 |

## 建议的学习顺序 {#suggested-order}

```text
Prerequisites ──► Part 1 (2–4 weeks) ──► Part 2 (2–3 weeks) ──► Part 3 (2–3 weeks) ──► Part 4 (2–3 weeks)
                       │                         │                       │                        │
                  build nanoGPT            fine-tune + GRPO          serve with vLLM/SGLang    semantic-ID generative recommender
```

- **前置知识：** 线性代数、概率论、PyTorch 基础、反向传播。推荐的复习资料：Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 和免费的 [Dive into Deep Learning（动手学深度学习）](https://d2l.ai/)。
- 每一部分都是一门**从零开始的课程**，按章节组织。每章按 **直觉 → 原理（数学逐步推导）→ 可运行代码 → 完整数值示例 → 2025–2026 年前沿系统的实际做法 → 常见误区 → 要点总结 → 自测题** 展开，最后是**延伸阅读**：原始论文，供你想深入时查阅。
- 论文标签：:material-star: **必读**、:material-book-open-variant: **深入阅读**、:material-new-box: **前沿**。

## 如何高效读论文 {#how-to-read-a-paper-efficiently}

1. 先读摘要，看 Figure 1 和主要结果表。
2. 问自己：*它消除了什么瓶颈？*（算力、显存、数据、稳定性、延迟……）
3. 略读方法部分，亲手推导一遍其中最关键的那个公式。
4. 检查这个想法是否经受住了时间的考验——今天的开源模型还在用吗？这里的摘要会尽量替你回答这个问题。

## 参与贡献 {#contributing}

发现了失效链接、过时的说法，或者有应该收录的论文？欢迎提 issue 或 PR。每个页面都是 `docs/` 下的一个普通 Markdown 文件（`*.md` = English，`*.zh.md` = 中文）。

*最后更新：2026 年 11 月。*
