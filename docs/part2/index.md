# Part 2 · Post-training, Alignment & Reasoning

A pre-trained language model (the subject of [Part 1](../part1/index.md)) is an extraordinarily good *document completer*. Give it the start of a Wikipedia article and it will continue the article; give it a question and it may answer it, or it may produce three more questions, because that is what the text on the internet often looks like. **Post-training** is everything we do after pre-training to turn that raw completer into something that behaves the way we want: an assistant that follows instructions, a model that reasons carefully before answering, an agent that calls tools, and a system that declines harmful requests.

This part teaches the algorithms that do this work from first principles. We start with plain supervised fine-tuning (SFT), build up the reinforcement-learning machinery that LLM people use (policy gradients, PPO, KL penalties), and then use it to derive RLHF, DPO and GRPO. Along the way we cover how reasoning models spend extra compute at test time, why "RL with verifiable rewards" (RLVR) became the dominant recipe in 2025, how to make all of this affordable with distillation and LoRA, and how agents and safety training fit on top.

Why it matters: nearly every model you interact with is a post-trained model, and in 2025–2026 an increasing share of a frontier lab's compute and research effort goes into post-training rather than pre-training. If you understand the objective each stage optimizes, you can predict what it will and will not change about a model, and you can read any new post-training paper as "a small modification of something I already know".

## Prerequisites and time

- **Math:** linear algebra, probability (expectations, conditional probability, log-likelihood), basic calculus (gradients, the chain rule).
- **Code:** Python and basic PyTorch (tensors, `nn.Module`, autograd, an optimizer loop).
- **From Part 1:** what a decoder-only transformer is, how next-token cross-entropy training works, and what tokens and logits are ([Part 1](../part1/index.md)).
- No prior reinforcement-learning knowledge is assumed; [Chapter 2.3](03-rl-foundations-for-llms.md) builds it from scratch.

**Estimated time:** about 25–35 hours of reading and exercises, plus 10–30 hours if you do the hands-on projects (a single consumer or cloud GPU is enough for small models).

## Chapter map

| # | Chapter | One-line description |
|---|---|---|
| 2.1 | [Why post-training?](01-why-post-training.md) | Base model vs assistant, what each stage adds, chat templates, how post-training is evaluated |
| 2.2 | [Supervised fine-tuning](02-supervised-fine-tuning.md) | The SFT objective, loss masking, packing, hyperparameters, data sourcing, a full training loop |
| 2.3 | [RL foundations for LLMs](03-rl-foundations-for-llms.md) | Generation as an MDP, policy gradients, REINFORCE, baselines, PPO, GAE, the KL penalty |
| 2.4 | [RLHF](04-rlhf.md) | Preference data, the Bradley–Terry reward model, the four-model PPO loop, reward hacking, RLAIF |
| 2.5 | [Direct Preference Optimization](05-direct-preference-optimization.md) | Deriving DPO from the RLHF objective, its loss in code, β, variants, offline vs online |
| 2.6 | [Reasoning and test-time compute](06-reasoning-and-test-time-compute.md) | Chain-of-thought, self-consistency, best-of-N, process vs outcome rewards, thinking budgets |
| 2.7 | [RL with verifiable rewards](07-rl-with-verifiable-rewards.md) | Rule-based rewards, GRPO, the DeepSeek-R1 recipe, DAPO/Dr. GRPO/GSPO/CISPO fixes, RL infrastructure |
| 2.8 | [Distillation and efficient fine-tuning](08-distillation-and-efficient-fine-tuning.md) | Knowledge distillation, on-policy distillation, LoRA from scratch, QLoRA, DoRA |
| 2.9 | [Agents, tool use and safety](09-agents-tool-use-and-safety.md) | Function calling, ReAct, agentic RL, MCP, agent benchmarks, refusals and safety training |

## Mental model

The modern post-training pipeline is a sequence of stages, each optimizing a different objective on different data:

```text
                    ┌────────────────────────────────────────────────────────────────┐
                    │                 (safety / alignment data mixed in throughout)   │
                    └────────────────────────────────────────────────────────────────┘
 ┌────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌────────────────────┐
 │ Base model │──►│     SFT      │──►│ Preference tuning   │──►│  RLVR (math, code, │──► assistant /
 │ (Part 1)   │   │  ch 2.2      │   │ RLHF ch 2.4 / DPO   │   │  tools, agents)    │    reasoning /
 │ next-token │   │ cross-entropy│   │ ch 2.5              │   │  GRPO ch 2.7       │    agent model
 └────────────┘   │ on responses │   │ "which answer is    │   │ "is the answer     │        │
                  └──────────────┘   │  better?"           │   │  checkably right?" │        │
                         ▲           └─────────────────────┘   └────────────────────┘        │
                         │                       ▲                       ▲                   │
                         │                       └──── RL machinery ─────┘                   │
                         │                             (ch 2.3)                              │
                         │                                                                   ▼
                         └────────────── Distillation (ch 2.8): a big post-trained model ────┘
                                         writes data / logits for a smaller student

  Test-time compute (ch 2.6) and tool use (ch 2.9) are behaviours that the later stages train for.
```

| Stage | Data | Objective | What it buys |
|---|---|---|---|
| **SFT** | (prompt, ideal response) pairs, incl. long chain-of-thought traces | Cross-entropy on response tokens | Format, instruction following, a "cold start" for reasoning |
| **Preference optimization** | (prompt, chosen, rejected) | RLHF (reward model + PPO) or DPO-family losses | Helpfulness, style, harmlessness |
| **RLVR** | Prompts with checkable answers (unit tests, math answers, tool outcomes) | Policy gradient (GRPO, DAPO, GSPO…) | Reasoning, long CoT, agentic skills |
| **Distillation** | Teacher outputs or logits | SFT or on-policy KD | Cheap small models that inherit big-model skills |

Real recipes reorder and repeat these stages (for example DeepSeek-R1 alternates SFT and RL twice, see [Chapter 2.7](07-rl-with-verifiable-rewards.md)), but every stage is one of the boxes above.

## How to study this part

1. **Read in order through 2.5.** Chapters 2.1–2.2 are quick; 2.3 is the mathematical core of the whole part. Do not skip it: RLHF, DPO and GRPO are all a page of algebra once policy gradients and the KL-regularized objective are familiar.
2. **Chapters 2.6 and 2.7 go together.** 2.6 explains *why* more thinking tokens help; 2.7 explains *how* RL teaches a model to produce them.
3. **Chapters 2.8 and 2.9 can be read in either order** and are the most directly practical.

Suggested hands-on projects (each maps to a chapter):

- [ ] **(2.2)** SFT a 0.5–1.5B model (e.g. a small Qwen) on an instruction dataset with [TRL](https://github.com/huggingface/trl) `SFTTrainer` + LoRA. Then write the loop yourself using the code in 2.2.
- [ ] **(2.4/2.5)** Train DPO on [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized); compare with SimPO.
- [ ] **(2.7)** Reproduce a mini R1-Zero: GRPO on GSM8K or a countdown task with a rule-based reward ([TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) or [verl](https://github.com/volcengine/verl)). Plot reward, response length and entropy over training.
- [ ] **(2.7)** Add DAPO's clip-higher and dynamic sampling and compare stability.
- [ ] **(2.8)** Distill: SFT a 1.5B model on reasoning traces from a larger open reasoner; evaluate on a math benchmark.
- [ ] Follow the [Hugging Face Open-R1](https://github.com/huggingface/open-r1) project and the [Tülu 3 open-instruct](https://github.com/allenai/open-instruct) code as reference implementations.

A free textbook that pairs well with this part is Nathan Lambert's **[RLHF Book](https://rlhfbook.com/)**.
