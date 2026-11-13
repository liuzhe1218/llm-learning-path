# LLM 学习路线

一条精心挑选、带有明确观点的学习路线，讲清现代大语言模型是如何**构建**、**对齐**和**部署**的——附原始论文链接，并简要说明每篇论文的贡献以及它至今仍然重要的原因。

!!! tip "Language / 语言"
    使用页眉中的语言切换器（🌐 图标）即可在**中文**和 **English** 之间切换。每个页面都有两种语言版本。

## 路线结构

| 部分 | 重点 | 学完后你将能够… |
|---|---|---|
| [**第 1 部分 · 预训练与架构**](part1-pretraining.md) | Transformer 内部结构、分词、scaling law（缩放定律）、数据、MoE、长上下文、高效注意力、分布式训练 | 讲清楚 2025–2026 年前沿模型（如 DeepSeek-V3、Qwen3、Llama 4、gpt-oss）的每个模块，以及为什么要这样设计 |
| [**第 2 部分 · 后训练与推理**](part2-posttraining.md) | SFT、RLHF、DPO 及其变体、基于可验证奖励的 RL（GRPO/DAPO/GSPO）、推理模型、test-time scaling、Agent/工具调用训练、PEFT | 设计一套后训练方案，把基座模型变成对话 / 推理 / Agent 模型 |
| [**第 3 部分 · 推理系统与基础设施**](part3-inference.md) | KV cache、PagedAttention、continuous batching、前缀缓存、投机解码、量化、prefill/decode 分离、MoE 推理服务 | 分析推理服务的延迟 / 吞吐 / 成本，并选择合适的引擎和优化手段 |

## 建议的学习顺序

```text
Prerequisites ──► Part 1 (2–4 weeks) ──► Part 2 (2–3 weeks) ──► Part 3 (2–3 weeks)
                       │                         │                       │
                  build nanoGPT            fine-tune + GRPO          serve with vLLM/SGLang
```

- **前置知识：** 线性代数、概率论、PyTorch 基础、反向传播。推荐的复习资料：Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 和免费的 [Dive into Deep Learning（动手学深度学习）](https://d2l.ai/)。
- 每一部分的结构：**核心概念 → 必读论文（附摘要）→ 2025–2026 前沿 → 动手项目 → 自测题**。
- 论文标签：:material-star: **必读**、:material-book-open-variant: **深入阅读**、:material-new-box: **前沿**。

## 如何高效读论文

1. 先读摘要，看 Figure 1 和主要结果表。
2. 问自己：*它消除了什么瓶颈？*（算力、显存、数据、稳定性、延迟……）
3. 略读方法部分，亲手推导一遍其中最关键的那个公式。
4. 检查这个想法是否经受住了时间的考验——今天的开源模型还在用吗？这里的摘要会尽量替你回答这个问题。

## 参与贡献

发现了失效链接、过时的说法，或者有应该收录的论文？欢迎提 issue 或 PR。每个页面都是 `docs/` 下的一个普通 Markdown 文件（`*.md` = English，`*.zh.md` = 中文）。

*最后更新：2026 年 11 月。*
