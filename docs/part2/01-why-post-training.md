# 2.1 Why Post-training?

!!! abstract "What you'll learn"
    - Why a pre-trained base model does not behave like an assistant, with concrete examples
    - What each post-training stage (SFT, preference tuning, RLVR, distillation) adds to a model
    - How chat templates and special tokens turn a conversation into a single token sequence
    - The two competing views: "alignment is superficial" vs "RL adds real capability"
    - How post-trained models are evaluated: benchmarks, LLM-as-a-judge, and human-preference arenas

## Why this matters / the problem

In [Part 1](../part1/index.md) we trained a model to minimize next-token cross-entropy on a huge corpus of text. The result, a **base model**, has learned grammar, facts, coding idioms and a surprising amount of reasoning. But its only goal is to produce a *likely continuation of the document so far*. That is not the same as being helpful.

Suppose you type this into a base model:

```text
What is the capital of Australia?
```

Plausible continuations on the internet include:

```text
What is the capital of Australia?
What is the capital of Canada?
What is the capital of Brazil?
```

(a quiz list), or

```text
What is the capital of Australia? Many people think it is Sydney, but ...
```

(an article), or simply the correct answer. The base model has no notion that *you* asked *it* a question and expect an answer. It also has no notion of when to stop, no stable persona, no tendency to refuse dangerous requests, and no preference for honest over confidently wrong answers beyond what the training text happened to contain.

You *can* coax a base model with a carefully crafted prompt (a "few-shot" prompt with example Q/A pairs), but that is brittle. Post-training bakes the desired behavior into the weights so that every user gets it by default.

## Intuition

A useful mental picture: pre-training builds a giant library of *possible* behaviors (every writing style, every persona, every level of quality that appeared in the data). Post-training is about **selecting** and **sharpening** a small subset of those behaviors:

```text
           Base model: a distribution over "all the documents on the internet"
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ forum rants · quiz lists · textbooks · expert answers · spam · code · ...  │
   └────────────────────────────────────────────────────────────────────────────┘
                 │ SFT: "respond like this, in this format"
                 ▼
          ┌───────────────────────────────┐
          │ assistant-style answers       │  (format, persona, instruction following)
          └───────────────────────────────┘
                 │ preference tuning: "this answer is better than that one"
                 ▼
             ┌──────────────────────┐
             │ helpful, harmless,   │      (style, tone, safety, calibration)
             │ well-formatted       │
             └──────────────────────┘
                 │ RL with verifiable rewards: "this answer is *correct*"
                 ▼
               ┌─────────────────┐
               │ careful,        │          (long reasoning, self-checking, tool use)
               │ reasoning agent │
               └─────────────────┘
```

Each stage narrows the distribution further, using a stronger and more specific signal of what "good" means.

## How it works

### What each stage adds

| Stage | Training signal | Typical change in behavior | Chapter |
|---|---|---|---|
| **Supervised fine-tuning (SFT)** | Demonstrations: "for this prompt, produce exactly this response" | Answers in assistant style, follows instructions, uses the chat format, stops at the end of a turn | [2.2](02-supervised-fine-tuning.md) |
| **Preference optimization** (RLHF, DPO) | Comparisons: "response A is better than response B" | Better tone and helpfulness, fewer harmful outputs, better judgment on open-ended tasks | [2.4](04-rlhf.md), [2.5](05-direct-preference-optimization.md) |
| **RL with verifiable rewards (RLVR)** | A program checks correctness (math answer, unit tests) | Longer and more careful reasoning, self-verification, higher accuracy on checkable tasks | [2.7](07-rl-with-verifiable-rewards.md) |
| **Distillation** | A stronger teacher's outputs or probabilities | A small model inherits behaviors of a big one at low cost | [2.8](08-distillation-and-efficient-fine-tuning.md) |
| **Safety training** (mixed into all stages) | Refusal demonstrations, harmlessness preferences, written specs | Declines harmful requests without refusing benign ones | [2.9](09-agents-tool-use-and-safety.md) |

Why not just do one stage? Each signal has a different cost and a different strength:

- **Demonstrations** are expensive per example (someone must write an ideal answer) but very informative: they show the model exactly what to do.
- **Comparisons** are cheaper (judging is easier than writing) and capture subtle qualities that are hard to demonstrate, but each comparison carries roughly one bit of information.
- **Verifiable rewards** are almost free once you have a checker and cannot be fooled by style, but they only exist for tasks with checkable answers.

The classic demonstration of the full pipeline is InstructGPT (Ouyang et al., 2022): SFT, then a reward model trained on human comparisons, then PPO. Human labelers preferred outputs from a 1.3B-parameter InstructGPT model over those of the 175B-parameter GPT-3 base model. Post-training is not a cosmetic layer; it changes how useful a model is more than a 100× increase in parameters did in that setting.

### Chat templates and special tokens

A transformer consumes **one flat sequence of tokens**. A conversation, however, has structure: a system message, alternating user and assistant turns, maybe tool calls. A **chat template** is the convention that serializes this structure into a single string, using **special tokens** that never appear in ordinary text.

Here is the ChatML-style template used by the Qwen model family. The conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Australia?"},
    {"role": "assistant", "content": "The capital of Australia is Canberra."},
]
```

is rendered as:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of Australia?<|im_end|>
<|im_start|>assistant
The capital of Australia is Canberra.<|im_end|>
```

`<|im_start|>` and `<|im_end|>` ("input message start/end") are each a **single token** in the vocabulary. Different model families use different templates; the Llama 3 family, for example, uses `<|start_header_id|>role<|end_header_id|>` to open a turn and `<|eot_id|>` ("end of turn") to close it. The details differ but the ideas are the same:

1. **Role markers** tell the model who is speaking. The model learns that after `<|im_start|>assistant\n` it is *its* turn.
2. **An end-of-turn token** (`<|im_end|>`, `<|eot_id|>`) is what the model emits when it is done. The inference engine stops generating when it sees it. A base model has never seen this token used this way, which is one reason base models ramble on.
3. **At inference time** the template is rendered up to and including the assistant header (`<|im_start|>assistant\n`); this is called the **generation prompt**. The model then completes the assistant turn.

!!! warning "Use the model's own template"
    A model only behaves well with the exact template it was post-trained on. Mixing templates (e.g. formatting a Qwen prompt with Llama tokens) silently degrades quality. Always use the tokenizer's built-in template rather than hand-writing strings.

Special tokens must be added to the vocabulary *before* post-training; their embeddings are learned during SFT. Many tokenizers ship the special tokens already reserved from pre-training so that the embedding matrix does not need to grow.

### Two views of what post-training does

**View 1: the superficial alignment hypothesis.** The LIMA paper (Zhou et al., 2023) fine-tuned a 65B LLaMA model on only 1,000 carefully curated prompt–response pairs and obtained a chat model competitive with much more heavily tuned systems in their human evaluation. The authors proposed that *almost all knowledge and capability is learned during pre-training*, and alignment mostly teaches the model **which format and style** to use when interacting with users. Under this view, post-training is a thin layer that *selects* behaviors already present in the base model.

**View 2: RL adds (or at least unlocks) capability.** The 2024–2025 wave of reasoning models changed the picture. RL with verifiable rewards on math and code made models dramatically more accurate on hard problems, with emergent behaviors such as long chains of reasoning, backtracking and self-checking (DeepSeek-R1, see [Chapter 2.7](07-rl-with-verifiable-rewards.md)). That looks like more than formatting.

The truth seems to be in between, and the debate is active:

- s1 (Muennighoff et al., 2025) showed that SFT on only 1,000 long-reasoning examples plus a simple test-time trick gives strong math reasoning, suggesting reasoning is largely *latent* in strong base models.
- Yue et al. (2025) found that if you let the base model sample many times (large pass@k), it often solves as many problems as the RL-trained model, so RL mainly makes good reasoning paths *more likely* rather than creating new ones. We study this in [Chapter 2.7](07-rl-with-verifiable-rewards.md).

A safe summary: **pre-training determines what a model *can* do; post-training determines what it *reliably does by default*.** The gap between those two can be enormous, which is why post-training matters so much.

### How post-training is evaluated

Evaluation is harder after post-training because the target ("helpful") is fuzzy. Three families of methods are used together.

**1. Automatic benchmarks with ground truth.** A fixed set of questions with checkable answers:

| Capability | Example benchmarks |
|---|---|
| Knowledge | MMLU, GPQA (multiple choice, graduate-level science) |
| Math | GSM8K (grade-school word problems), MATH, AIME (competition problems) |
| Code | HumanEval, LiveCodeBench, SWE-bench (real GitHub issues) |
| Instruction following | IFEval (verifiable constraints like "answer in exactly 3 bullet points") |

These are cheap and reproducible but suffer from **contamination** (test questions leaking into training data) and **saturation** (top models all score near 100%).

**2. LLM-as-a-judge.** A strong model reads a prompt and one or two responses and outputs a score or a preference. MT-Bench and AlpacaEval work this way. Zheng et al. (2023) found that GPT-4 as a judge agreed with human experts about as often as humans agreed with each other, but also documented biases: **position bias** (preferring the first answer shown), **verbosity bias** (preferring longer answers), and **self-enhancement bias** (preferring answers in its own style). Mitigations include swapping the order and averaging, and length-controlled scoring.

**3. Human preference arenas.** Chatbot Arena (Chiang et al., 2024) shows real users two anonymous model responses side by side and asks which is better. Thousands of such votes are aggregated into a rating with a Bradley–Terry model, the same model we will use for reward models in [Chapter 2.4](04-rlhf.md).

## Code

The snippet below renders a conversation with a real tokenizer's chat template, shows the generation prompt, and generates a reply. It uses a small instruction-tuned model so it runs on a laptop CPU (it downloads ~1 GB on first run).

```python
# pip install torch transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"     # small post-trained (instruct) model
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Australia?"},
]

# 1) Render the conversation as text. add_generation_prompt=True appends the
#    "<|im_start|>assistant\n" header so the model knows it is its turn.
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

# 2) Special tokens are single vocabulary entries, not sequences of characters.
print(tok.convert_ids_to_tokens(tok("<|im_start|>user")["input_ids"]))
# -> ['<|im_start|>', 'user']

# 3) Generate. Generation stops at the end-of-turn token (eos for this model).
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
```

Try the same `text` with the *base* model `Qwen/Qwen2.5-0.5B` (no `-Instruct`): the base model often continues in odd ways, because it has never learned what to do after `<|im_start|>assistant`.

## Worked example

**Reading an arena leaderboard.** Arena ratings use the Elo scale, where a rating difference $\Delta = R_A - R_B$ maps to a predicted win probability for model A:

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{-\Delta/400}}.
$$

In words: every 400 points of rating difference multiplies the odds of winning by 10.

- $\Delta = 0$: $P = 1/(1+1) = 0.50$, a coin flip.
- $\Delta = 100$: $10^{-100/400} = 10^{-0.25} \approx 0.562$, so $P \approx 1/1.562 \approx 0.64$.
- $\Delta = 30$: $10^{-0.075} \approx 0.841$, so $P \approx 1/1.841 \approx 0.54$.

So a 30-point lead, which looks large on a leaderboard, means users prefer the leader in only about 54% of head-to-head votes. With a few thousand votes per model, the confidence interval on a rating is often ±5–10 points, so models within a few points of each other are statistically tied. This is why leaderboards publish confidence intervals, and why you should not over-read small rating gaps.

## In practice (2025–2026)

- **Pipelines are multi-stage and iterative.** Open recipes such as Tülu 3 (AI2) publish the full pipeline: SFT on a curated mix, then DPO on preference data, then RLVR on math and instruction-following prompts with checkable answers. Frontier reasoning models (DeepSeek-R1, Qwen3, Kimi) interleave several SFT and RL stages.
- **Post-training compute is growing.** Reasoning-focused RL runs are now a substantial fraction of total training compute for frontier models, whereas in the InstructGPT era post-training was a small fraction.
- **Synthetic data dominates.** Most SFT and preference data is generated or filtered by other models, with humans focusing on hard cases, quality control and specifications.
- **Chat templates have grown.** Modern templates include roles for tools and tool results, and markers for hidden reasoning (for example `<think> ... </think>` blocks in several open reasoning models).
- **Evaluation is a mix.** Labs report a panel of ground-truth benchmarks (math, code, GPQA, IFEval, SWE-bench), LLM-judge scores, and arena ratings, and increasingly build private held-out evaluations to avoid contamination.

## Common pitfalls / misconceptions

- **"Post-training teaches the model new facts."** Mostly false. Trying to inject knowledge via SFT tends to cause hallucinations: the model learns to *sound* confident about things it does not know. Knowledge comes from pre-training (or retrieval).
- **"The instruct model is just the base model with a system prompt."** No: the weights differ. A system prompt on a base model is a much weaker intervention.
- **Mismatched templates.** Training with one template and serving with another, or forgetting `add_generation_prompt=True`, is one of the most common bugs in fine-tuning projects.
- **Trusting one number.** A single benchmark or a single LLM-judge score can be gamed by length, formatting or contamination. Look at several evaluation types.

## Key takeaways

- A base model completes documents; post-training makes it respond as an assistant by default.
- SFT uses demonstrations, preference tuning uses comparisons, RLVR uses programmatic correctness checks, distillation uses a teacher; each signal has different cost and power.
- Chat templates serialize conversations into one token sequence using special tokens for roles and end-of-turn.
- "Superficial alignment" (format is learned in post-training) and "RL adds capability" are both partly right: post-training mostly changes what the model reliably does, not what it could ever do.
- Evaluate with ground-truth benchmarks, LLM-as-a-judge (beware of biases), and human-preference arenas (read confidence intervals).

## Check your understanding

??? question "Why does a base model often fail to stop generating after answering a question?"
    It was trained on documents, not turns. It has never learned that an end-of-turn token should follow a complete answer, so it keeps producing a plausible continuation (more questions, more text). SFT teaches it to emit the end-of-turn token, which the inference engine uses as a stop signal.

??? question "What is the generation prompt, and what goes wrong if you forget it?"
    It is the assistant header (e.g. `<|im_start|>assistant\n`) appended after the last user message so the model knows it must now write the assistant turn. Without it the model may continue the user's message or start a new user turn.

??? question "Why are comparisons cheaper to collect than demonstrations, and what do you lose?"
    Judging which of two answers is better is faster and easier than writing an ideal answer, and annotators can judge answers they could not write. But each comparison carries little information (roughly one bit), and it only says which is better, not what the best answer would be.

??? question "Model A has an arena rating 50 points above model B. What fraction of votes should A win?"
    $P = 1/(1+10^{-50/400}) = 1/(1+10^{-0.125}) \approx 1/(1+0.750) \approx 0.57$, so about 57%.

??? question "Name two biases of LLM-as-a-judge evaluation and a mitigation for each."
    Position bias (mitigate by evaluating both orders and averaging), verbosity bias (mitigate with length-controlled scoring or instructions to ignore length), self-enhancement bias (use a judge from a different model family).

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | The SFT → reward model → PPO pipeline behind ChatGPT; a 1.3B aligned model preferred over 175B GPT-3. |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | The "superficial alignment hypothesis": 1,000 curated examples give strong chat behavior. |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | A fully open, modern SFT → DPO → RLVR recipe with data, code and evaluations. |
| :material-book-open-variant: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023 | Introduces MT-Bench and studies the agreement and biases of LLM judges. |
| :material-book-open-variant: [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) — Chiang et al., 2024 | How pairwise human votes become a leaderboard via Bradley–Terry. |
| :material-book-open-variant: [RLHF Book](https://rlhfbook.com/) — Nathan Lambert | A free textbook on the whole post-training stack; pairs well with this part. |
