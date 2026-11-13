# Part 2 · Post-training, Alignment & Reasoning

> **Goal:** understand how a raw base model becomes a helpful chat assistant, a long-chain-of-thought **reasoning model**, and a tool-using **agent** — and which algorithms (SFT, RLHF, DPO, GRPO and successors) do the work.

**Legend:** :material-star: must-read · :material-book-open-variant: deep-dive · :material-new-box: 2025–2026 frontier

---

## 2.1 The big picture

```text
Base model ──► SFT / instruction tuning ──► Preference optimization (RLHF / DPO)
                                        └─► RL with verifiable rewards (math, code, tools) ──► reasoning / agent model
                                                         ▲
                     distillation from a stronger model ─┘   (+ safety / alignment training throughout)
```

| Stage | Data | Objective | What it buys |
|---|---|---|---|
| **SFT** | (prompt, ideal response) pairs, incl. long CoT traces | Cross-entropy on the response | Format, instruction following, "cold start" for reasoning |
| **Preference optimization** | (prompt, chosen, rejected) | RLHF (reward model + PPO) or DPO-family | Helpfulness, style, harmlessness |
| **RLVR** | Prompts with checkable answers (unit tests, math answers, tool outcomes) | Policy gradient (GRPO, DAPO, GSPO…) | Reasoning, long CoT, agentic skills |
| **Distillation** | Teacher outputs / logits | SFT or on-policy KD | Cheap small models that inherit big-model skills |

Recommended free textbook alongside this part: Nathan Lambert's **[RLHF Book](https://rlhfbook.com/)**.

---

## 2.2 Instruction tuning (SFT)

| Paper | Summary |
|---|---|
| :material-star: [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — Wei et al., 2021 | Fine-tuning on many tasks phrased as instructions improves zero-shot performance on unseen tasks. |
| :material-book-open-variant: [Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416) — Chung et al., 2022 | Scaling #tasks and adding chain-of-thought data to instruction tuning. |
| :material-book-open-variant: [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al., 2022 | Bootstraps instruction data from the model itself — the ancestor of today's synthetic-data pipelines. |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | 1,000 carefully curated examples give strong chat behavior → "superficial alignment hypothesis": SFT mostly teaches *format*; knowledge comes from pre-training. Quality ≫ quantity. |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | Fully open modern recipe: SFT → DPO → **RLVR**, with data, code and evaluations. The best end-to-end reference to reproduce. |

---

## 2.3 RLHF: reward models + PPO

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al., 2017 | Learn a reward model from pairwise human comparisons, then optimize it with RL. The origin of RLHF. |
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | Clipped-ratio policy gradient with a learned value function. The RL algorithm of classic RLHF. |
| :material-book-open-variant: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020 | RLHF on summarization; introduces the reward-model + KL-penalty recipe at scale. |
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | The SFT → RM → PPO pipeline behind ChatGPT. A 1.3B aligned model preferred over 175B GPT-3. |
| :material-star: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (Anthropic), 2022 | Replace human harmlessness labels with AI critiques guided by a written "constitution" (RLAIF). |
| :material-book-open-variant: [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) — Lee et al., 2023 | AI-labeled preferences can match human labels for many tasks. |
| :material-book-open-variant: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao et al., 2022 | Optimizing too hard against a proxy reward eventually lowers true quality (**reward hacking** / Goodhart) — why the KL penalty exists. |

The RLHF objective:

\[
\max_{\pi_\theta}\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
\]

---

## 2.4 Direct preference optimization (offline, RL-free)

| Paper | Summary |
|---|---|
| :material-star: [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 | Shows the RLHF objective has a closed-form optimum, so you can train directly on preference pairs with a simple classification-style loss — no reward model, no sampling. Became the default for open models in 2023–24. |
| :material-book-open-variant: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024 | Works with unpaired thumbs-up/thumbs-down labels. |
| :material-book-open-variant: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024 | Merges SFT and preference loss into one stage, no reference model. |
| :material-book-open-variant: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024 | Length-normalized, reference-free DPO variant with a margin; strong and cheap. |

**Online vs offline:** DPO is offline (fixed dataset). Evidence through 2024–2025 suggests **online, on-policy** methods (PPO, GRPO, online/iterative DPO) generally win when you can afford sampling — which is why RL came back for reasoning.

---

## 2.5 Reasoning: chain-of-thought and test-time compute

| Paper | Summary |
|---|---|
| :material-star: [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al., 2022 | Showing step-by-step exemplars makes large models dramatically better at multi-step problems. |
| :material-book-open-variant: [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022 | Sample many reasoning paths, majority-vote the answer. The simplest form of test-time scaling. |
| :material-book-open-variant: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — Zelikman et al., 2022 | Keep self-generated rationales that reach the right answer, fine-tune, repeat — expert iteration for reasoning. |
| :material-star: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al. (OpenAI), 2023 | **Process reward models** (PRMs, scoring each step) beat outcome reward models for selecting math solutions; releases PRM800K. |
| :material-star: [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) — Snell et al., 2024 | Spending inference compute well (search against verifiers, sequential revision) can beat a 14× larger model. The conceptual basis of "thinking" models. |
| :material-new-box: [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) — Muennighoff et al., 2025 | Only 1K curated long-CoT examples for SFT + "budget forcing" (appending "Wait") yields o1-preview-level math. Shows reasoning is largely *latent* in strong base models. |

---

## 2.6 RL with verifiable rewards (RLVR) — the 2025 revolution :material-new-box:

Instead of a learned reward model, use **programmatic rewards**: did the math answer match? did the code pass unit tests? did the tool call succeed? This is hard to reward-hack and scales.

| Paper | Summary |
|---|---|
| :material-star: [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024 | **Group Relative Policy Optimization**: sample a group of G answers per prompt, use the group-normalized reward as the advantage — no value network, ~half the memory of PPO. The workhorse algorithm of open reasoning RL. |
| :material-star: [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | **R1-Zero**: pure GRPO on a base model with rule-based rewards spontaneously learns long CoT, self-verification and "aha moments". **R1** adds a cold-start SFT, multi-stage RL, and distills reasoning into small dense models. The most important post-training paper of 2025 (later published in *Nature*). |
| :material-new-box: [Kimi k1.5: Scaling RL with LLMs](https://arxiv.org/abs/2501.12599) — Kimi Team, 2025 | Parallel discovery: long-context RL, online mirror descent, length penalties, and "long2short" methods to compress reasoning. |
| :material-new-box: [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) — Yu et al. (ByteDance Seed), 2025 | Four fixes to GRPO for long CoT: **clip-higher** (decoupled clip range, prevents entropy collapse), **dynamic sampling** (drop all-correct/all-wrong groups), **token-level loss**, **overlong reward shaping**. |
| :material-new-box: [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) — Liu et al., 2025 | Shows GRPO's length and std normalization bias the model toward long wrong answers; proposes an unbiased variant. |
| :material-new-box: [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — Zheng et al. (Qwen), 2025 | Importance ratios and clipping at the **sequence** level rather than token level; much more stable, especially for MoE RL. Used for Qwen3. |
| :material-new-box: [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) — MiniMax, 2025 | Clips importance-sampling *weights* instead of token updates, so rare but important "reflection" tokens still get gradient. |
| :material-book-open-variant: [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — Yue et al., 2025 | At large pass@k, base models often solve as much as RLVR models: RL mainly *sharpens sampling* toward known-good paths rather than adding new capabilities. Important nuance for the hype. |

GRPO advantage for answer *i* in a group of *G*:

\[
\hat A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
\]

**Practical lessons (2025):** keep entropy from collapsing; filter prompts to the model's difficulty frontier; watch response length; verifiers must be robust (format hacking!); RL infra is dominated by **rollout generation**, so inference engines (Part 3) matter — see open frameworks [verl](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [TRL](https://github.com/huggingface/trl), and [slime](https://github.com/THUDM/slime).

---

## 2.7 Distillation

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015 | Train a student on a teacher's soft probabilities. |
| :material-book-open-variant: [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023 | Student generates; teacher scores the student's own tokens (reverse-KL style) — fixes train/inference mismatch of plain SFT distillation. Used in Gemma and Qwen3 small models. |

DeepSeek-R1 showed **SFT on a strong reasoner's traces** makes small models better reasoners than running RL on them directly — distillation is the cheapest path to small reasoning models.

---

## 2.8 Agents & tool use

| Paper | Summary |
|---|---|
| :material-star: [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — Yao et al., 2022 | Interleave "thought" and "action/observation" steps — the template of every agent loop. |
| :material-book-open-variant: [Toolformer](https://arxiv.org/abs/2302.04761) — Schick et al., 2023 | Self-supervised learning of when/how to call APIs. |
| :material-book-open-variant: [SWE-bench](https://arxiv.org/abs/2310.06770) — Jimenez et al., 2023 | Real GitHub issues as the benchmark for coding agents; the target of most agentic RL in 2025–2026. |
| :material-new-box: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | Large-scale synthetic tool-use trajectories + RL with verifiable and self-critique rewards for agentic behavior. |

2025–2026 trend: **agentic RL** — multi-turn rollouts inside real environments (shells, browsers, repos) with outcome rewards; protocols like [MCP](https://modelcontextprotocol.io/) standardize tool interfaces.

---

## 2.9 Parameter-efficient fine-tuning (PEFT)

| Paper | Summary |
|---|---|
| :material-star: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 | Freeze weights, learn a low-rank update ΔW = BA. Tiny checkpoints, cheap fine-tuning, swappable adapters. |
| :material-star: [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023 | LoRA on a 4-bit (NF4) quantized base: fine-tune a 65B model on one 48 GB GPU. |
| :material-book-open-variant: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al., 2024 | Decouples magnitude and direction; closes much of the LoRA–full-FT gap. |

Thinking Machines' 2025 post **[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)** shows LoRA matches full fine-tuning for RL and small-to-medium SFT when applied to all layers with the right learning rate.

---

## 2.10 Safety & alignment (selected)

| Paper | Summary |
|---|---|
| :material-book-open-variant: [Deliberative Alignment](https://arxiv.org/abs/2412.16339) — Guan et al. (OpenAI), 2024 | Teach reasoning models to explicitly reason over the written safety spec before answering. |
| :material-book-open-variant: [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Hubinger et al. (Anthropic), 2024 | Backdoored behaviors can survive standard safety training — motivates interpretability and evaluation work. |

---

## 2.11 Hands-on projects

- [ ] SFT a 0.5–1.5B model (e.g. Qwen) on an instruction dataset with [TRL](https://github.com/huggingface/trl) `SFTTrainer` + LoRA.
- [ ] Train DPO on [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized); compare with SimPO.
- [ ] Reproduce a mini R1-Zero: GRPO on GSM8K / countdown with a rule-based reward ([TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) or verl). Plot reward, response length and entropy.
- [ ] Add DAPO's clip-higher + dynamic sampling and compare stability.
- [ ] Distill: SFT a 1.5B model on reasoning traces from a larger open reasoner; evaluate on AIME/MATH-500.
- [ ] Follow the [Hugging Face Open-R1](https://github.com/huggingface/open-r1) project and the [Tülu 3](https://github.com/allenai/open-instruct) code.

## 2.12 Self-check

??? question "Why does GRPO not need a value network?"
    Its baseline is the mean reward of the other samples for the same prompt, estimated empirically from the group, instead of a learned critic.

??? question "Why did RLVR work for reasoning when RLHF-PPO on human preferences did not produce long CoT?"
    Verifiable rewards are accurate and hard to hack, so the policy can be optimized for many steps; and strong base models already contain the reasoning behaviors, which RL amplifies.

??? question "When would you choose DPO over GRPO?"
    When you have preference data but no verifier (style, helpfulness, safety), limited compute, or want an offline, stable, simple pipeline.

??? question "What is reward hacking? Give one example from RLVR."
    The policy exploits flaws in the reward instead of solving the task — e.g. outputting multiple answers to match a lenient parser, or editing unit tests in a coding environment.
