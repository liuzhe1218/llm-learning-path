# 2.2 Supervised Fine-Tuning

!!! abstract "What you'll learn"
    - The SFT objective and how it differs from pre-training only in *which tokens count*
    - How to mask prompt tokens out of the loss, for single- and multi-turn conversations, in code
    - How sequence packing works and why it needs special attention masks
    - Typical hyperparameters, and why SFT overfits after very few epochs
    - Where SFT data comes from (human, Self-Instruct, distillation) and why quality beats quantity
    - A complete, minimal SFT training loop in PyTorch + Hugging Face Transformers

## Why this matters / the problem

[Chapter 2.1](01-why-post-training.md) showed that a base model does not know it is in a conversation. The most direct fix is to *show* it thousands of conversations in the chat template and train it to produce the assistant's side. This is **supervised fine-tuning (SFT)**, also called **instruction tuning**.

SFT is the first stage of essentially every post-training pipeline. It is cheap (hours on a few GPUs for a small model), simple (it is the same next-token loss as pre-training), and it gives a reasonable assistant by itself. It is also the stage that "cold-starts" later RL: the RL stages in Chapters [2.4](04-rlhf.md) and [2.7](07-rl-with-verifiable-rewards.md) start from an SFT model because they need a policy that already produces sensible, well-formatted answers to explore from.

## Intuition

Think of the base model as a person who has read the whole library but has never worked a help desk. SFT is on-the-job training by imitation: we hand them transcripts of excellent help-desk conversations and say "when you are in the assistant's seat, answer like this".

Two details matter:

1. **We only grade the assistant's lines.** The user's messages are context. We do not want the model to learn to *write user questions*, so we exclude those tokens from the loss (**loss masking**).
2. **The model is already a good learner.** A few thousand high-quality examples shift its behavior a lot. Many passes over the same small set make it memorize them, so SFT runs are short.

## How it works

### The objective

A training example is a prompt $x$ (the system and user messages, rendered with the chat template) and a target response $y = (y_1, \dots, y_T)$ of $T$ tokens. Let $\pi_\theta$ be the language model with parameters $\theta$; $\pi_\theta(y_t \mid x, y_{<t})$ is the probability it assigns to token $y_t$ given the prompt and the previous response tokens $y_{<t} = (y_1,\dots,y_{t-1})$.

The SFT loss for one example is the negative log-likelihood of the response:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}).
$$

In words: make every token of the ideal response as likely as possible, *conditioned on* the prompt. This is exactly the pre-training cross-entropy from [Part 1](../part1/index.md), except the sum runs only over response tokens. Over a dataset $\mathcal{D}$ of pairs, we minimize the average of this loss; maximizing likelihood of demonstrations is the same as minimizing the KL divergence from the data distribution to the model, so SFT is "imitation learning" in the statistical sense.

### Loss masking

In practice we concatenate prompt and response into one sequence `input_ids` of length $L$ and run the model on all of it (the prompt tokens are needed as context). We then build a `labels` tensor of the same length that equals `input_ids` on response positions and a special value $-100$ on prompt positions. PyTorch's `cross_entropy(..., ignore_index=-100)` skips those positions both in the sum and in the averaging denominator.

Shapes: `input_ids: [batch, L]`, `labels: [batch, L]`, logits `[batch, L, V]` where $V$ is the vocabulary size. Because the model predicts the *next* token, logits at position $i$ are compared with the label at position $i+1$ (the "shift by one").

### Multi-turn formatting

A conversation with several assistant turns is one sequence. Every assistant turn (including its end-of-turn token, so the model learns to stop) is trained on; every system, user and tool message is masked. Masking the earlier assistant turns as well is possible, but training on all of them uses more signal from each conversation.

```text
<|im_start|>system\nYou are helpful.<|im_end|>\n     masked  (-100)
<|im_start|>user\nHi!<|im_end|>\n                    masked
<|im_start|>assistant\n                               masked  (the header is given at inference)
Hello! How can I help?<|im_end|>                      TRAINED
\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n         masked
<|im_start|>assistant\n                               masked
2 + 2 = 4.<|im_end|>                                  TRAINED
```

### Packing and attention masks

SFT examples vary wildly in length (10 to 10,000 tokens). If we pad every example in a batch to the longest one, most compute is spent on padding. **Packing** concatenates several short examples into one row of a fixed length (say 4,096 tokens) so almost no compute is wasted.

The danger: with an ordinary causal mask, tokens of example 2 can attend to tokens of example 1 in the same row, so the model learns from unrelated context ("cross-contamination"). The correct fix is a **block-diagonal causal mask**: token $i$ may attend to token $j$ only if $j \le i$ *and* both belong to the same example. Position ids should also restart at 0 for each example so positional encodings (e.g. RoPE) match what the model sees at inference.

```text
packed row:  [ a a a a | b b b | c c ]
mask (1 = may attend):
        a a a a b b b c c
   a    1
   a    1 1
   a    1 1 1
   a    1 1 1 1
   b            1
   b            1 1
   b            1 1 1
   c                  1
   c                  1 1
position_ids: 0 1 2 3 0 1 2 0 1
```

Efficient attention kernels (FlashAttention's "variable-length" API) implement this without materializing the mask by taking the cumulative sequence lengths `cu_seqlens = [0, 4, 7, 9]`. Some pipelines skip the block mask and accept a little contamination; it is cheaper but measurably worse on some tasks.

### Loss averaging

How you average matters once examples have different lengths:

- **Per-token mean over the batch**: sum the loss over all unmasked tokens in the batch and divide by the number of such tokens. Long responses contribute more gradient.
- **Per-sequence mean, then mean over sequences**: each example contributes equally regardless of length.

Neither is "right", but they behave differently, and the choice interacts with packing and gradient accumulation (see the worked example). We will meet the same issue again in RL ([Chapter 2.7](07-rl-with-verifiable-rewards.md)).

### Hyperparameters, and why SFT overfits fast

Typical starting points for full fine-tuning of a 1B–70B model:

| Setting | Typical range | Why |
|---|---|---|
| Learning rate | roughly $10^{-6}$ to $2\times10^{-5}$ (AdamW) | Much lower than pre-training: we want to nudge, not rewrite, the weights |
| Schedule | linear warmup (a few % of steps), then cosine or linear decay | Avoids an early shock to a well-trained network |
| Epochs | 1–3 | More epochs usually hurt held-out quality |
| Batch size | tens to a few hundred sequences | Smaller datasets → smaller batches to get enough steps |
| Weight decay | 0 to 0.1 | Minor effect |

LoRA ([Chapter 2.8](08-distillation-and-efficient-fine-tuning.md)) typically uses learning rates about 10× higher than full fine-tuning.

Why does SFT overfit after a few epochs? The base model has billions of parameters and the SFT set is small (often $10^4$–$10^6$ examples). The model can lower training loss by memorizing exact responses rather than learning the general style, and it does so quickly because it is already an excellent sequence predictor. Signs of overfitting: training loss falling towards zero, validation loss rising, and outputs that copy phrases from the training set. Note that validation *loss* is an imperfect proxy: response quality judged by humans or benchmarks sometimes keeps improving slightly after validation loss starts rising, so evaluate with generations, not only loss.

A subtler cost is **forgetting**: fine-tuning on a narrow distribution can degrade abilities that the SFT data does not exercise (e.g. a coding-only SFT set hurting multilingual ability). Mixing in diverse data is the main remedy.

### Where SFT data comes from

1. **Human-written demonstrations.** Annotators write ideal responses to real or invented prompts (InstructGPT used this). High quality, slow, expensive.
2. **Templated academic datasets.** FLAN (Wei et al., 2021) rewrote many existing NLP datasets (translation, QA, NLI…) as natural-language instructions and showed that instruction tuning on many tasks improves zero-shot performance on *unseen* tasks. Chung et al. (2022) scaled this to about 1.8K tasks and showed that including chain-of-thought examples matters.
3. **Self-Instruct (synthetic).** Wang et al. (2022) started from 175 human-written seed tasks and prompted the model itself to generate new instructions, inputs and outputs, filtering out low-quality and near-duplicate ones. This bootstrapping idea is the ancestor of today's synthetic data pipelines.
4. **Distillation from a stronger model.** Generate responses to many prompts with a strong model and SFT a smaller model on them. Most open models in 2024–2026 use this heavily, including reasoning traces from reasoning models (see [Chapter 2.8](08-distillation-and-efficient-fine-tuning.md)). Check the stronger model's license before using its outputs.

### Quality over quantity

LIMA (Zhou et al., 2023) fine-tuned a 65B model on just 1,000 carefully chosen examples and obtained a strong chat model. The lesson that has held up: **a smaller set of diverse, correct, well-formatted examples beats a large noisy set.** Practical filters include deduplication, removing responses with wrong answers (checked by a verifier or a strong model), balancing task types, and rejecting responses that refuse or hedge when they should not. Because SFT imitates *everything* in the data, including mistakes and verbal tics, a single systematic flaw in the data (e.g. every answer starts with "Certainly!") becomes a flaw in the model.

## Code

### 1. Building input ids and masked labels for a conversation

We render ChatML by hand, segment by segment, so we know exactly which tokens belong to assistant content. (With a real tokenizer you can use `apply_chat_template(..., return_assistant_tokens_mask=True)` if the template supports it; doing it by hand shows what happens.)

```python
import torch

IGNORE = -100

def build_example(messages, tok, max_len=2048):
    """messages: list of {"role", "content"}. Returns input_ids and labels (1-D LongTensors).
    Only assistant content + its end-of-turn marker is trained on."""
    input_ids, labels = [], []
    for m in messages:
        header = f"<|im_start|>{m['role']}\n"
        body = m["content"] + "<|im_end|>"
        h_ids = tok(header, add_special_tokens=False)["input_ids"]
        b_ids = tok(body, add_special_tokens=False)["input_ids"]
        nl_ids = tok("\n", add_special_tokens=False)["input_ids"]
        input_ids += h_ids + b_ids + nl_ids
        if m["role"] == "assistant":
            labels += [IGNORE] * len(h_ids) + b_ids + [IGNORE] * len(nl_ids)  # train on reply + <|im_end|>
        else:
            labels += [IGNORE] * (len(h_ids) + len(b_ids) + len(nl_ids))      # mask system/user/tool
    return torch.tensor(input_ids[:max_len]), torch.tensor(labels[:max_len])

def sft_loss(logits, labels):
    """logits: [B, L, V], labels: [B, L] with IGNORE on masked positions.
    Position i predicts token i+1, so shift by one."""
    logits = logits[:, :-1, :].contiguous()          # [B, L-1, V]
    targets = labels[:, 1:].contiguous()             # [B, L-1]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=IGNORE)                          # mean over unmasked tokens only
```

### 2. Packing with a block-diagonal causal mask

```python
def pack(examples, seq_len):
    """examples: list of (input_ids, labels). Greedily fills one row of length seq_len.
    Returns input_ids [seq_len], labels [seq_len], position_ids [seq_len], mask [seq_len, seq_len]."""
    ids, labs, pos, doc = [], [], [], []
    for d, (x, y) in enumerate(examples):
        if len(ids) + len(x) > seq_len:
            break
        ids += x.tolist(); labs += y.tolist()
        pos += list(range(len(x)))                   # positions restart per document
        doc += [d] * len(x)                          # document id per token
    pad = seq_len - len(ids)
    ids += [0] * pad; labs += [IGNORE] * pad; pos += [0] * pad; doc += [-1] * pad
    doc = torch.tensor(doc)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_doc = (doc[:, None] == doc[None, :]) & (doc[:, None] >= 0)
    mask = causal & same_doc                         # True = may attend
    return torch.tensor(ids), torch.tensor(labs), torch.tensor(pos), mask
```

Hugging Face models accept a 4-D boolean or additive mask of shape `[batch, 1, L, L]` for some attention implementations; in production you would pass `cu_seqlens` to a variable-length FlashAttention kernel instead of building an $L \times L$ matrix.

### 3. A complete minimal SFT loop

This trains a small model on a toy dataset with padding (no packing, for simplicity). Swap in a real dataset and it is a working SFT trainer.

```python
# pip install torch transformers
import torch, math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

model_id = "Qwen/Qwen2.5-0.5B"                       # a BASE model; we teach it to chat
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).train()

data = [  # in reality: thousands of curated conversations
    [{"role": "user", "content": "What is the capital of Australia?"},
     {"role": "assistant", "content": "The capital of Australia is Canberra."}],
    [{"role": "user", "content": "Give me a synonym for 'happy'."},
     {"role": "assistant", "content": "Joyful."}],
] * 64

examples = [build_example(m, tok) for m in data]

def collate(batch):
    L = max(len(x) for x, _ in batch)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(batch), L), pad_id); labs = torch.full((len(batch), L), IGNORE)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = x; labs[i, :len(y)] = y; att[i, :len(x)] = 1
    return ids, labs, att

loader = DataLoader(examples, batch_size=8, shuffle=True, collate_fn=collate)
epochs = 2
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
total = epochs * len(loader)
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(0.03 * total), num_training_steps=total)

for epoch in range(epochs):
    for step, (ids, labs, att) in enumerate(loader):
        ids, labs, att = ids.to(device), labs.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()   # [B, L, V]
        loss = sft_loss(logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)            # stability
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 4 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.3f}")
```

Notes: we pass `attention_mask` so padded positions are ignored by attention, and padded label positions are `IGNORE` so they are ignored by the loss. In real runs you would also hold out a validation set, evaluate generations periodically, and save checkpoints from each epoch.

## Worked example

**Why masking matters, in numbers.** One example has a 300-token system+user prompt and a 100-token response.

- *Without* masking, the loss averages over 400 positions; 300 of them (75%) train the model to predict the *user's* words. The gradient signal for the assistant behavior is diluted 4×, and the model learns to generate user-like text.
- *With* masking, only the 100 response tokens (plus the end-of-turn token) contribute.

**Why averaging matters, in numbers.** A batch has two examples. Example A has 10 response tokens with average per-token loss 2.0; example B has 90 response tokens with average per-token loss 1.0.

- Per-token mean (what `cross_entropy(..., ignore_index)` does on the flattened batch): $(10 \cdot 2.0 + 90 \cdot 1.0)/100 = 1.10$. Example B gets 90% of the weight.
- Per-sequence mean: $(2.0 + 1.0)/2 = 1.50$. Each example gets 50%.

Now split this batch into two micro-batches for **gradient accumulation** (one example each) and naively average the two micro-batch losses: you get $(2.0 + 1.0)/2 = 1.50$, i.e. per-sequence weighting, even though a single big batch would have given 1.10. This silent mismatch was a real bug in popular training libraries; the fix is to sum token losses across micro-batches and divide by the *total* number of unmasked tokens in the full batch.

## In practice (2025–2026)

- **Scale of SFT sets.** Open recipes use from a few thousand (LIMA, s1) to a few million examples (Tülu 3's SFT mix is on the order of a million), heavily weighted toward synthetic data from strong models, filtered for quality.
- **Reasoning SFT.** For reasoning models, SFT on long chain-of-thought traces (thousands of tokens each) is the "cold start" before RL; sequence lengths of 16K–32K+ tokens make packing and memory-efficient attention essential.
- **Libraries.** TRL's `SFTTrainer`, Axolotl, LLaMA-Factory and torchtune all implement masking ("train on completions only"), packing with proper position ids, and LoRA. Read their docs to see which masking and averaging defaults they use.
- **Chat templates in the tokenizer.** Store the template with the tokenizer (`tokenizer.chat_template`) so training and serving agree.

## Common pitfalls / misconceptions

- **Forgetting to mask the prompt** (or masking the end-of-turn token, so the model never learns to stop).
- **Off-by-one in the label shift.** If loss looks suspiciously low from step 0, check that you are not predicting the token at the same position.
- **Packing without resetting position ids or without a block mask**, causing cross-example attention.
- **Too many epochs.** Training loss near zero is a warning sign, not a success.
- **"SFT can teach new knowledge."** Fine-tuning on facts the model does not know tends to increase hallucination: it learns to produce confident answers in that style without the underlying knowledge.
- **Inconsistent templates** between SFT and inference, or a missing end-of-turn token in the training data.

## Key takeaways

- SFT is next-token cross-entropy restricted to response tokens: $-\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$.
- Mask prompt and user tokens with `-100`; train on every assistant turn including its end-of-turn token.
- Packing saves compute but needs per-document position ids and a block-diagonal causal mask.
- Use small learning rates and 1–3 epochs; SFT overfits quickly because the model is large and the data is small.
- Data quality and diversity matter more than size; synthetic and distilled data now dominate.

## Check your understanding

??? question "Why do we keep the prompt tokens in the input if we mask them in the loss?"
    The model must condition on them: the response probabilities $\pi_\theta(y_t \mid x, y_{<t})$ depend on the prompt $x$. Masking only removes them from the *targets*, not from the context.

??? question "In a packed row containing two documents, what goes wrong if you use a plain causal mask?"
    Tokens of the second document can attend to the first, so the model learns to use unrelated context, and positions continue from the first document instead of restarting, mismatching inference conditions.

??? question "Your SFT training loss is 0.05 after 6 epochs and the model repeats training answers verbatim. What happened and what do you change?"
    It overfit/memorized. Use fewer epochs (1–3), possibly a lower learning rate, more diverse data, and select checkpoints by generation quality on a held-out set.

??? question "What is the core idea of Self-Instruct?"
    Start from a small set of human-written seed tasks, prompt a language model to generate new instructions and responses, filter out low-quality and near-duplicate ones, add them to the pool, and repeat, bootstrapping a large instruction dataset with little human effort.

??? question "A batch contains a 5-token and a 95-token response with mean token losses 3.0 and 1.0. What is the per-token mean loss?"
    $(5 \cdot 3.0 + 95 \cdot 1.0)/100 = 110/100 = 1.10$.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — Wei et al., 2021 | Instruction tuning on many tasks improves zero-shot performance on unseen tasks. |
| :material-book-open-variant: [Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416) — Chung et al., 2022 | Scaling the number of tasks and adding chain-of-thought data to instruction tuning. |
| :material-book-open-variant: [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al., 2022 | Bootstrapping instruction data from the model itself; ancestor of synthetic-data pipelines. |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | 1,000 curated examples suffice for strong chat behavior; quality ≫ quantity. |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | Fully open SFT → DPO → RLVR recipe; read its SFT data-mixture section. |
