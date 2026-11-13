# 第二部分 · 后训练、对齐与推理能力

> **目标：** 理解一个原始的基座模型（base model）如何一步步变成好用的对话助手、能输出长思维链的**推理模型**，以及会调用工具的**智能体（agent）**——并搞清楚真正起作用的是哪些算法（SFT、RLHF、DPO、GRPO 及其后继者）。

**图例：** :material-star: 必读 · :material-book-open-variant: 深入阅读 · :material-new-box: 2025–2026 前沿

---

## 2.1 全景图

```text
基座模型 ──► SFT / 指令微调 ──► 偏好优化（RLHF / DPO）
                          └─► 可验证奖励强化学习（数学、代码、工具）──► 推理 / 智能体模型
                                                         ▲
                       从更强的模型蒸馏 ─────────────────┘   （全程配合安全 / 对齐训练）
```

| 阶段 | 数据 | 目标函数 | 能带来什么 |
|---|---|---|---|
| **SFT** | （prompt，理想回复）对，包括长 CoT 轨迹 | 对回复计算交叉熵 | 输出格式、指令遵循，以及推理能力的"冷启动" |
| **偏好优化** | （prompt，chosen，rejected） | RLHF（奖励模型 + PPO）或 DPO 系方法 | 有用性、风格、无害性 |
| **RLVR** | 答案可校验的 prompt（单元测试、数学答案、工具调用结果） | 策略梯度（GRPO、DAPO、GSPO 等） | 推理能力、长 CoT、智能体技能 |
| **蒸馏** | 教师模型的输出 / logits | SFT 或 on-policy 知识蒸馏 | 低成本的小模型，继承大模型的能力 |

建议配合本部分阅读的免费教材：Nathan Lambert 的 **[RLHF Book](https://rlhfbook.com/)**。

---

## 2.2 指令微调（SFT）

| Paper | 摘要 |
|---|---|
| :material-star: [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — Wei et al., 2021 | 在大量以指令形式描述的任务上做微调，可以提升模型在未见任务上的零样本表现。 |
| :material-book-open-variant: [Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416) — Chung et al., 2022 | 扩大指令微调的任务数量，并加入思维链（CoT）数据。 |
| :material-book-open-variant: [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al., 2022 | 让模型自己生成指令数据来自举——是今天各类合成数据流水线的鼻祖。 |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | 仅用 1,000 条精心挑选的样本就能得到不错的对话行为，由此提出"表面对齐假说"（superficial alignment hypothesis）：SFT 主要教的是*格式*，知识来自预训练。质量远比数量重要。 |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | 完全开源的现代后训练配方：SFT → DPO → **RLVR**，数据、代码和评测一应俱全。是最值得完整复现的端到端参考。 |

---

## 2.3 RLHF：奖励模型 + PPO

| Paper | 摘要 |
|---|---|
| :material-book-open-variant: [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al., 2017 | 从人类的成对比较中学习奖励模型（reward model），再用强化学习去优化它。RLHF 的起点。 |
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | 带裁剪比率、并配合学习得到的价值函数的策略梯度方法，是经典 RLHF 所用的 RL 算法。 |
| :material-book-open-variant: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020 | 把 RLHF 用于摘要任务，首次在大规模上确立了"奖励模型 + KL 惩罚"的配方。 |
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | ChatGPT 背后的 SFT → RM → PPO 流水线。对齐后的 1.3B 模型，其输出比 175B 的 GPT-3 更受人类偏好。 |
| :material-star: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (Anthropic), 2022 | 用 AI 依据一份成文的"宪法"所做的批评来取代人类的无害性标注（RLAIF）。 |
| :material-book-open-variant: [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) — Lee et al., 2023 | 在许多任务上，AI 标注的偏好可以达到与人类标注相当的效果。 |
| :material-book-open-variant: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao et al., 2022 | 过度针对代理奖励（proxy reward）优化，最终会降低真实质量（**reward hacking（奖励投机）** / Goodhart 定律）——这正是需要 KL 惩罚的原因。 |

RLHF 的优化目标：

\[
\max_{\pi_\theta}\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
\]

---

## 2.4 直接偏好优化（离线、无需 RL）

| Paper | 摘要 |
|---|---|
| :material-star: [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 | 证明 RLHF 的目标存在闭式最优解，因此可以直接在偏好对上用一个类似分类的简单损失来训练——不需要奖励模型，也不需要采样。2023–24 年成为开源模型的默认做法。 |
| :material-book-open-variant: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024 | 只需要不成对的点赞 / 点踩标签即可训练。 |
| :material-book-open-variant: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024 | 把 SFT 和偏好损失合并成一个阶段，且无需参考模型。 |
| :material-book-open-variant: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024 | 做了长度归一化、无参考模型、带 margin 的 DPO 变体，效果好且成本低。 |

**在线 vs. 离线：** DPO 是离线方法（数据集固定）。2024–2025 年的证据表明，只要采样成本可以承受，**在线、on-policy** 的方法（PPO、GRPO、在线 / 迭代式 DPO）通常表现更好——这也是 RL 在推理任务上卷土重来的原因。

---

## 2.5 推理：思维链与测试时计算

| Paper | 摘要 |
|---|---|
| :material-star: [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al., 2022 | 给出逐步推理的示例，能让大模型在多步问题上的表现大幅提升。 |
| :material-book-open-variant: [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022 | 采样多条推理路径，对答案做多数投票。最简单的测试时扩展（test-time scaling）形式。 |
| :material-book-open-variant: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — Zelikman et al., 2022 | 保留自己生成的、最终答案正确的推理过程，微调后不断迭代——相当于推理领域的专家迭代（expert iteration）。 |
| :material-star: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al. (OpenAI), 2023 | 在挑选数学解答时，**过程奖励模型**（PRM，对每一步打分）优于结果奖励模型；同时发布了 PRM800K 数据集。 |
| :material-star: [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) — Snell et al., 2024 | 合理地使用推理阶段算力（基于验证器的搜索、串行修正）可以胜过一个大 14 倍的模型。是"思考型"模型的概念基础。 |
| :material-new-box: [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) — Muennighoff et al., 2025 | 仅用 1K 条精选长 CoT 样本做 SFT，再加上"budget forcing"（在末尾追加 "Wait"），数学能力就能达到 o1-preview 水平。说明强基座模型中推理能力很大程度上是*潜在存在*的。 |

---

## 2.6 可验证奖励强化学习（RLVR）——2025 年的革命 :material-new-box:

不再使用学习得到的奖励模型，而是采用**程序化奖励**：数学答案对不对？代码能不能通过单元测试？工具调用有没有成功？这种奖励很难被投机取巧，并且可以规模化。

| Paper | 摘要 |
|---|---|
| :material-star: [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024 | **Group Relative Policy Optimization（组相对策略优化）**：对每个 prompt 采样一组 G 个回答，用组内归一化后的奖励作为优势（advantage）——不需要价值网络，显存约为 PPO 的一半。是开源推理 RL 的主力算法。 |
| :material-star: [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | **R1-Zero**：直接在基座模型上用基于规则的奖励做纯 GRPO，模型自发学会了长 CoT、自我验证和"顿悟时刻"（aha moment）。**R1** 则加入冷启动 SFT 和多阶段 RL，并把推理能力蒸馏到小型稠密模型中。是 2025 年最重要的后训练论文（后来发表于 *Nature*）。 |
| :material-new-box: [Kimi k1.5: Scaling RL with LLMs](https://arxiv.org/abs/2501.12599) — Kimi Team, 2025 | 平行的独立发现：长上下文 RL、在线镜像下降（online mirror descent）、长度惩罚，以及用于压缩推理长度的"long2short"方法。 |
| :material-new-box: [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) — Yu et al. (ByteDance Seed), 2025 | 针对长 CoT 对 GRPO 的四项改进：**clip-higher**（解耦裁剪范围，防止熵坍塌）、**dynamic sampling**（动态采样，丢弃全对 / 全错的组）、**token 级损失**、**overlong reward shaping**（超长回复的奖励塑形）。 |
| :material-new-box: [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) — Liu et al., 2025 | 指出 GRPO 中的长度归一化和标准差归一化会使模型偏向于生成又长又错的回答，并提出了无偏的变体。 |
| :material-new-box: [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — Zheng et al. (Qwen), 2025 | 在**序列**层面而非 token 层面计算重要性比率并做裁剪；稳定性明显更好，尤其适合 MoE 模型的 RL。已用于 Qwen3。 |
| :material-new-box: [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) — MiniMax, 2025 | 裁剪的是重要性采样*权重*而不是 token 更新，这样那些罕见但重要的"反思"token 仍然能获得梯度。 |
| :material-book-open-variant: [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — Yue et al., 2025 | 在 pass@k 的 k 很大时，基座模型往往能解出和 RLVR 模型一样多的题：RL 主要是把采样*锐化*到已知的好路径上，而不是增加新能力。对过热的炒作是一个重要的补充说明。 |

对于一组 *G* 个回答中的第 *i* 个，GRPO 的优势为：

\[
\hat A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
\]

**实践经验（2025）：** 防止熵坍塌；把 prompt 过滤到模型的难度边界附近；留意回复长度；验证器必须足够鲁棒（小心格式投机！）；RL 的基础设施开销主要集中在 **rollout 生成**，所以推理引擎（第三部分）很重要——可参考开源框架 [verl](https://github.com/volcengine/verl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL](https://github.com/huggingface/trl) 和 [slime](https://github.com/THUDM/slime)。

---

## 2.7 蒸馏（Distillation）

| Paper | 摘要 |
|---|---|
| :material-book-open-variant: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015 | 让学生模型学习教师模型输出的软概率。 |
| :material-book-open-variant: [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023 | 由学生模型生成，教师模型对学生自己生成的 token 打分（类似 reverse-KL）——解决了普通 SFT 式蒸馏中训练与推理分布不一致的问题。Gemma 和 Qwen3 的小模型都用了这种方法。 |

DeepSeek-R1 表明，对强推理模型的轨迹做 **SFT**，比直接在小模型上跑 RL 更能让小模型学会推理——蒸馏是得到小型推理模型最便宜的路径。

---

## 2.8 智能体与工具调用

| Paper | 摘要 |
|---|---|
| :material-star: [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — Yao et al., 2022 | 交替进行"思考（thought）"与"行动 / 观察（action/observation）"步骤——是所有智能体循环的模板。 |
| :material-book-open-variant: [Toolformer](https://arxiv.org/abs/2302.04761) — Schick et al., 2023 | 以自监督方式学习何时、如何调用 API。 |
| :material-book-open-variant: [SWE-bench](https://arxiv.org/abs/2310.06770) — Jimenez et al., 2023 | 以真实的 GitHub issue 作为编码智能体的基准；2025–2026 年大多数智能体 RL 都以它为目标。 |
| :material-new-box: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | 大规模合成的工具使用轨迹，加上可验证奖励与自我批评（self-critique）奖励的 RL，用于培养智能体行为。 |

2025–2026 年的趋势：**智能体 RL（agentic RL）**——在真实环境（shell、浏览器、代码仓库）中进行多轮 rollout，并以结果作为奖励；[MCP](https://modelcontextprotocol.io/) 这样的协议则让工具接口标准化。

---

## 2.9 参数高效微调（PEFT）

| Paper | 摘要 |
|---|---|
| :material-star: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 | 冻结原权重，只学习一个低秩更新 ΔW = BA。checkpoint 极小、微调便宜，adapter 还可以随意切换。 |
| :material-star: [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023 | 在 4-bit（NF4）量化的基座模型上做 LoRA：一张 48 GB 显卡就能微调 65B 模型。 |
| :material-book-open-variant: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al., 2024 | 将权重的幅度与方向解耦，缩小了 LoRA 与全参数微调之间的大部分差距。 |

Thinking Machines 在 2025 年发表的博文 **[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)** 表明：只要作用于所有层并配合合适的学习率，LoRA 在 RL 以及中小规模的 SFT 上可以媲美全参数微调。

---

## 2.10 安全与对齐（精选）

| Paper | 摘要 |
|---|---|
| :material-book-open-variant: [Deliberative Alignment](https://arxiv.org/abs/2412.16339) — Guan et al. (OpenAI), 2024 | 教推理模型在回答之前，先显式地依据成文的安全规范进行推理。 |
| :material-book-open-variant: [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Hubinger et al. (Anthropic), 2024 | 被植入后门的行为可以在标准安全训练之后依然存活——这推动了可解释性和评测方面的工作。 |

---

## 2.11 动手项目

- [ ] 用 [TRL](https://github.com/huggingface/trl) 的 `SFTTrainer` + LoRA，在指令数据集上对一个 0.5–1.5B 的模型（如 Qwen）做 SFT。
- [ ] 在 [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 上训练 DPO，并与 SimPO 对比。
- [ ] 复现一个迷你版 R1-Zero：在 GSM8K / countdown 上用基于规则的奖励跑 GRPO（可用 [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) 或 verl）。画出奖励、回复长度和熵的曲线。
- [ ] 加入 DAPO 的 clip-higher 和 dynamic sampling，对比训练稳定性。
- [ ] 蒸馏：用更大的开源推理模型生成的推理轨迹对一个 1.5B 模型做 SFT，并在 AIME/MATH-500 上评测。
- [ ] 跟进 [Hugging Face Open-R1](https://github.com/huggingface/open-r1) 项目和 [Tülu 3](https://github.com/allenai/open-instruct) 的代码。

## 2.12 自测题

??? question "为什么 GRPO 不需要价值网络？"
    它的基线是同一个 prompt 下其他样本的平均奖励，由组内样本经验估计得到，而不是依赖一个学习得到的 critic。

??? question "为什么 RLVR 能让推理能力涌现，而基于人类偏好的 RLHF-PPO 却没能产生长 CoT？"
    可验证奖励既准确又难以被钻空子，所以策略可以被连续优化很多步；同时，强大的基座模型本身已经包含了这些推理行为，RL 只是把它们放大。

??? question "什么时候你会选择 DPO 而不是 GRPO？"
    当你手上有偏好数据但没有验证器（风格、有用性、安全性等场景）、算力有限，或者想要一条离线、稳定、简单的流水线时。

??? question "什么是 reward hacking（奖励投机）？请举一个 RLVR 中的例子。"
    策略利用奖励函数的缺陷，而不是真正完成任务——例如输出多个答案去匹配宽松的解析器，或者在编码环境中直接修改单元测试。
