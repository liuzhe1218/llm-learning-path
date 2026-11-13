# 2.8 Distillation and Efficient Fine-Tuning

!!! abstract "What you'll learn"
    - Knowledge distillation with soft targets and temperature, and the difference between forward and reverse KL (with code)
    - Sequence-level distillation from a teacher's reasoning traces, and on-policy distillation (GKD)
    - LoRA derived: the low-rank update, rank $r$, the $\alpha/r$ scaling, which layers to adapt; a LoRA layer from scratch
    - A parameter-count and memory worked example for LoRA on a 7B model
    - QLoRA (NF4, double quantization, paged optimizers) and DoRA at an intuitive level, and when LoRA matches full fine-tuning

## Why this matters / the problem

Everything in this part so far assumed you can update all the weights of the model you care about. Two practical problems get in the way:

1. **The best behavior lives in big models**, but big models are expensive to serve. Can a small model inherit what a big model learned? That is **distillation**. DeepSeek-R1 found that fine-tuning small models on R1's reasoning traces made them better reasoners than running large-scale RL on the small models directly ([Chapter 2.7](07-rl-with-verifiable-rewards.md)).
2. **Full fine-tuning is memory-hungry.** With Adam in mixed precision, training needs about 16 bytes per parameter ([Chapter 2.4](04-rlhf.md)): over 100 GB for a 7B model before activations. **Parameter-efficient fine-tuning** (PEFT), chiefly **LoRA**, trains a tiny number of extra parameters instead, so you can fine-tune a 7B model on a single consumer GPU and keep many cheap task-specific adapters.

## Intuition

**Distillation**: a teacher's output distribution carries much more information than the single "correct" token. If the teacher says the next token is "cat" with 70%, "kitten" 20% and "dog" 5%, the student learns that "kitten" is a near-synonym and "dog" a plausible confusion: what Hinton et al. called **dark knowledge**. Training on the full distribution (soft targets) teaches more per example than training on the argmax (hard target).

**LoRA**: when you fine-tune a pre-trained model, the *change* in each weight matrix tends to be effectively low-rank: a few directions matter. So instead of learning a full $d\times d$ update, learn it as a product of two thin matrices. It is like adjusting a few knobs that each move many weights in a coordinated way, rather than turning every screw separately.

## How it works

### Knowledge distillation: soft targets and temperature

For one position, let the teacher's logits be $z^T \in \mathbb{R}^V$ and the student's $z^S \in \mathbb{R}^V$ ($V$ = vocabulary size). With a **temperature** $\tau > 0$:

$$
p^T_\tau = \operatorname{softmax}(z^T/\tau), \qquad p^S_\tau = \operatorname{softmax}(z^S/\tau).
$$

In words: dividing logits by $\tau > 1$ flattens the distributions, making small probabilities (the dark knowledge) larger and more visible to the loss; $\tau = 1$ is the ordinary softmax.

The classic distillation loss (Hinton et al., 2015) mixes a soft-target term and the usual hard-label cross-entropy:

$$
\mathcal{L}_{\text{KD}} = \alpha\,\tau^2\,\mathrm{KL}\big(p^T_\tau\,\|\,p^S_\tau\big) + (1-\alpha)\,\mathrm{CE}\big(y, p^S_1\big),
$$

where $y$ is the ground-truth token and $\alpha \in [0,1]$ weights the two. The $\tau^2$ factor compensates for the fact that gradients of the softened term scale like $1/\tau^2$, keeping the two terms' relative influence stable when you change $\tau$. For an LLM, this per-position loss is summed over response positions exactly like SFT ([Chapter 2.2](02-supervised-fine-tuning.md)). **Logit distillation requires the teacher and student to share a tokenizer** (the distributions must be over the same vocabulary).

### Forward vs reverse KL

KL divergence is asymmetric, and the direction changes what the student learns.

- **Forward KL** $\mathrm{KL}(p^T\|p^S) = \sum_v p^T(v)\log\frac{p^T(v)}{p^S(v)}$, averaged over sequences from data or the teacher. It heavily penalizes the student for putting low probability where the teacher puts high probability, so the student tries to **cover all the teacher's modes** ("mean-seeking"). A small student that cannot represent all modes spreads its probability and may produce blurry, low-quality samples.
- **Reverse KL** $\mathrm{KL}(p^S\|p^T) = \sum_v p^S(v)\log\frac{p^S(v)}{p^T(v)}$. It penalizes the student for putting probability where the teacher puts little, so the student **picks a mode and commits** ("mode-seeking"). It may drop some valid alternatives, but what it generates is high-quality by the teacher's standards.

```text
 teacher p^T:     ▁▆█▆▁      ▁▆█▆▁        (two modes)
 forward-KL student:  ▁▃▅▅▅▅▅▅▅▃▁          (covers both, mass in between)
 reverse-KL student: ▁▆█▆▁                 (commits to one mode)
```

Recall that RLHF's KL penalty, $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, is also a reverse KL ([Chapter 2.3](03-rl-foundations-for-llms.md)).

### Sequence-level distillation

Kim and Rush (2016) proposed a simpler alternative: let the teacher **generate** outputs (e.g. its beam-search or sampled answers), then train the student with plain SFT on those outputs. No logits needed, so it works across different tokenizers and with API-only teachers. This is how most "distilled" open models are made today: the R1-distilled Qwen and Llama models were SFT'd on about 800K samples generated by R1 ([Chapter 2.7](07-rl-with-verifiable-rewards.md)), and s1 used 1,000 reasoning traces from a stronger reasoning model ([Chapter 2.6](06-reasoning-and-test-time-compute.md)).

Practical points for reasoning traces: filter to traces with verified-correct final answers (rejection sampling), deduplicate, keep long contexts (traces can be tens of thousands of tokens), and pick problems that are hard for the student.

### On-policy distillation (GKD)

Both logit and sequence distillation above are **off-policy**: the student is trained on prefixes produced by the teacher (or data). At inference the student conditions on *its own* earlier tokens, which may contain mistakes the teacher never makes, a train–test mismatch called **exposure bias**. The student never learns how to recover from its own errors.

**Generalized Knowledge Distillation (GKD)** (Agarwal et al., 2023) trains on **student-generated** sequences:

1. The student samples a response $y \sim \pi_S(\cdot\mid x)$.
2. The teacher computes its next-token distribution at *every position of the student's response*.
3. Minimize a per-token divergence between teacher and student distributions on these positions (the paper considers forward KL, reverse KL and a generalized Jensen–Shannon divergence that interpolates between them), and can mix on-policy and fixed-data sequences.

With reverse KL, the per-token loss on a student sample is $\sum_v p^S(v\mid s_t)\big(\log p^S(v\mid s_t) - \log p^T(v\mid s_t)\big)$, which looks like RL with a dense per-token reward $\log p^T - \log p^S$: the teacher grades every token the student actually wrote. It gives far more signal per sample than RL's single scalar reward, and it is on-policy. The Qwen3 technical report, for example, describes an off-policy then on-policy distillation stage for its small models.

### LoRA, derived

Take one linear layer of the pre-trained model, $h = W_0 x$, with $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, input $x\in\mathbb{R}^{d_{\text{in}}}$. Full fine-tuning learns $W = W_0 + \Delta W$ with a dense $\Delta W$: $d_{\text{out}}\cdot d_{\text{in}}$ trainable numbers.

**LoRA** (Hu et al., 2021) freezes $W_0$ and constrains the update to rank at most $r$:

$$
\Delta W = \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d_{\text{out}}\times r},\quad A \in \mathbb{R}^{r\times d_{\text{in}}},\quad r \ll \min(d_{\text{in}}, d_{\text{out}}),
$$

so the forward pass becomes

$$
h = W_0 x + \frac{\alpha}{r} B (A x).
$$

In words: the input is projected down to $r$ dimensions by $A$, back up by $B$, scaled, and added to the frozen layer's output. Symbols:

- $r$, the **rank**: the number of independent directions the update can use (typical 8–64; 1–16 can suffice for RL).
- $\alpha$, the **scaling**: a constant; the update is multiplied by $\alpha/r$ so that changing $r$ does not require re-tuning the learning rate as much. A common choice is $\alpha = 2r$ or a fixed $\alpha$ like 16 or 32.
- **Initialization**: $A$ random (e.g. Gaussian or Kaiming-uniform), $B = 0$. Then $\Delta W = 0$ at the start, so training begins exactly at the pre-trained model; $B$ receives nonzero gradient because $Ax \neq 0$.

Trainable parameters per layer: $r(d_{\text{in}} + d_{\text{out}})$ instead of $d_{\text{in}}d_{\text{out}}$. For a square $4096\times4096$ layer with $r = 16$: $16\times8192 = 131{,}072$ vs $16{,}777{,}216$, a 128× reduction.

**Which layers?** The original paper adapted only the attention query and value projections. Later work, including the QLoRA paper and Thinking Machines' "LoRA Without Regret" (2025), found it is better to apply LoRA to **all linear layers**, especially the MLP (and MoE) layers; attention-only LoRA underperforms even at matched parameter count.

**Merging.** After training, $W_0 + \frac{\alpha}{r}BA$ can be computed once and stored as an ordinary dense matrix, so a merged LoRA model has **zero inference overhead**. Alternatively, keep adapters separate (a few MB to hundreds of MB each) and swap or batch many adapters over one base model.

**Why memory drops.** Gradients and Adam states are only needed for $A$ and $B$. The frozen $W_0$ needs neither, and can even be stored in low precision (QLoRA).

### QLoRA: LoRA on a 4-bit base model

QLoRA (Dettmers et al., 2023) stores the frozen base weights in 4 bits and trains bf16 LoRA adapters on top, backpropagating *through* the quantized weights (dequantized on the fly for each matmul). Three ideas:

1. **NF4 (4-bit NormalFloat).** Pre-trained weights are roughly normally distributed. NF4's 16 levels are placed at quantiles of a normal distribution, so each level is used about equally often, which is information-theoretically good for normal data. Weights are quantized in blocks of 64, each with its own scale (the block's absolute maximum).
2. **Double quantization.** Those per-block scales are themselves 32-bit numbers: $32/64 = 0.5$ extra bits per parameter. QLoRA quantizes the scales to 8 bits (in blocks of 256), cutting the overhead to about $8/64 + 32/(64\cdot256) \approx 0.127$ bits per parameter.
3. **Paged optimizers.** Memory spikes (e.g. from a long sequence) can crash training. Optimizer states are allocated in NVIDIA unified memory so they can be paged to CPU RAM when the GPU runs short, like virtual memory.

The headline result: fine-tuning a 65B model on a single 48 GB GPU, with quality matching 16-bit full fine-tuning on their benchmarks.

### DoRA: separate magnitude and direction

DoRA (Liu et al., 2024) decomposes each weight matrix into a **magnitude** vector $m$ (the norm of each column) and a **direction** (columns normalized to unit length):

$$
W' = m \odot \frac{W_0 + BA}{\lVert W_0 + BA\rVert_c},
$$

where $\lVert\cdot\rVert_c$ is the column-wise norm and $m$ is trained directly (initialized to $\lVert W_0\rVert_c$). In words: LoRA adjusts the direction of each column; a separate small vector adjusts its length. The authors observed that full fine-tuning tends to change magnitude and direction in different patterns than LoRA does, and that decoupling them closes much of the gap to full fine-tuning at a small extra cost.

### When does LoRA match full fine-tuning?

"LoRA Without Regret" (Thinking Machines, 2025) ran careful comparisons and reported:

- For **SFT on small-to-medium instruction and reasoning datasets**, LoRA performs the same as full fine-tuning.
- For **datasets that exceed LoRA's capacity** (very large, e.g. continued pre-training scale), LoRA learns less efficiently and underperforms.
- For **RL**, even very low ranks (down to rank 1) match full fine-tuning, because RL provides little information per episode (roughly a scalar reward), so little capacity is needed.
- Apply LoRA to **all layers**, especially MLPs; the optimal learning rate is about **10× higher** than for full fine-tuning; LoRA can be less tolerant of very large batch sizes.

## Code

### 1. Distillation losses (forward KL with temperature, reverse KL)

```python
import torch, torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, tau=2.0, alpha=0.5, ignore=-100):
    """Logits: [B, L, V] already aligned so position t predicts labels[:, t]; labels: [B, L].
    Forward-KL soft loss (Hinton) + hard cross-entropy, averaged over unmasked positions."""
    mask = (labels != ignore)
    s = student_logits[mask]                                  # [N, V]
    t = teacher_logits[mask].detach()
    soft = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1),
                    log_target=True, reduction="batchmean")   # KL(p_T || p_S)
    hard = F.cross_entropy(s, labels[mask])
    return alpha * tau**2 * soft + (1 - alpha) * hard

def reverse_kl_loss(student_logits, teacher_logits, mask):
    """Per-token KL(p_S || p_T) over the full vocabulary, on positions of a STUDENT-sampled
    sequence (on-policy / GKD-style). Logits: [B, L, V], mask: [B, L]."""
    log_ps = F.log_softmax(student_logits, -1)
    log_pt = F.log_softmax(teacher_logits.detach(), -1)
    kl = (log_ps.exp() * (log_ps - log_pt)).sum(-1)           # [B, L]
    return (kl * mask).sum() / mask.sum()

B, L, V = 2, 5, 11
s, t = torch.randn(B, L, V, requires_grad=True), torch.randn(B, L, V)
labels = torch.randint(0, V, (B, L)); labels[:, :2] = -100    # first 2 positions = prompt
print(kd_loss(s, t, labels).item(), reverse_kl_loss(s, t, (labels != -100).float()).item())
```

### 2. A LoRA layer from scratch

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear: h = W0 x + (alpha/r) * B(A(x))."""
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                           # freeze W0 (and bias)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B = 0  ->  Delta W = 0 at init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                     # x: [..., d_in]
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        """Fold the update into W0 for zero-overhead inference; returns a plain nn.Linear."""
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base

def add_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"), r=16, alpha=32):
    """Replace every nn.Linear whose attribute name is in `targets` (all linear layers of a
    Llama/Qwen-style block by default) and freeze everything else."""
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
    return model

# sanity checks
layer = nn.Linear(64, 32)
lora = LoRALinear(layer, r=4, alpha=8)
x = torch.randn(3, 64)
assert torch.allclose(lora(x), layer(x))                     # identical at init
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(trainable, 4 * (64 + 32))                               # 384 384
```

To use it on a Hugging Face model: `model = add_lora(AutoModelForCausalLM.from_pretrained(...))`, then pass only `[p for p in model.parameters() if p.requires_grad]` to the optimizer and train exactly as in [Chapter 2.2](02-supervised-fine-tuning.md), with a learning rate around 10× the full fine-tuning one. The `peft` library implements the same idea with many extras.

## Worked example

**LoRA parameter count and memory on a Llama-2-7B-shaped model.** Hidden size $d = 4096$, MLP intermediate size $d_{ff} = 11008$, 32 layers, about 6.7B parameters. LoRA with $r = 16$ on all seven linear layers of each block.

Per layer:

- Attention $q, k, v, o$: each $4096\to4096$: $16\times(4096+4096) = 131{,}072$; four of them: $524{,}288$.
- MLP gate and up ($4096\to11008$) and down ($11008\to4096$): each $16\times(4096+11008) = 241{,}664$; three of them: $724{,}992$.
- Total per layer: $1{,}249{,}280$.

All 32 layers: $32\times1{,}249{,}280 = 39{,}976{,}960 \approx 40$M trainable parameters, about **0.6%** of 6.7B.

Memory (weights + gradients + Adam states, ignoring activations):

| Setup | Calculation | Approx. memory |
|---|---|---|
| Full fine-tuning, mixed-precision Adam | $6.7\text{B}\times16$ bytes | ≈ 107 GB |
| LoRA, bf16 frozen base | $6.7\text{B}\times2 + 40\text{M}\times16$ | ≈ 13.4 + 0.64 ≈ 14 GB |
| QLoRA, NF4 base (≈4.127 bits/param) | $6.7\text{B}\times4.127/8 + 40\text{M}\times16$ | ≈ 3.5 + 0.64 ≈ 4.1 GB |

The adapter checkpoint itself is $40\text{M}\times2$ bytes ≈ 80 MB in bf16, versus 13.4 GB for a full copy of the model.

**Temperature in numbers.** Teacher logits $(4, 2, 0)$ for ("cat", "kitten", "dog"). At $\tau = 1$: $\operatorname{softmax} = (0.867, 0.117, 0.016)$. At $\tau = 2$: logits $(2, 1, 0)$, $\operatorname{softmax} = (0.665, 0.245, 0.090)$. The relationships between the minor options (kitten ≫ dog) become much more visible to the student at $\tau = 2$.

## In practice (2025–2026)

- **Distillation is how small models get good.** Most small open models are trained substantially on outputs of larger models: sequence-level SFT on reasoning traces is standard; logit-level and on-policy distillation are used when teacher and student share a tokenizer (e.g. within one model family).
- **On-policy distillation is gaining ground** as a cheap alternative or complement to RL for small models, since it gives a dense per-token signal on the student's own samples.
- **LoRA is the default for most fine-tuning outside frontier labs**, and is increasingly used for RL as well; multi-adapter serving lets one base model host many fine-tunes.
- **Typical LoRA settings**: all linear layers, $r$ 16–64 for SFT (lower can work for RL), $\alpha$ around $r$ to $2r$, learning rate roughly 10× full fine-tuning (often around $10^{-4}$ for 7B-scale SFT), no or small dropout.
- **QLoRA** is the go-to for fitting large models on one GPU; for fast inference, merge the adapter into a higher-precision base or serve the quantized base with the adapter separately.

## Common pitfalls / misconceptions

- **Logit distillation across tokenizers.** Distributions over different vocabularies cannot be compared directly; use sequence-level distillation instead.
- **Forgetting the $\tau^2$ factor**, so changing temperature silently changes the effective weight of the soft loss.
- **Initializing both $A$ and $B$ randomly.** The model no longer starts at the pre-trained function, and early training is unstable. One of them must be zero.
- **Using full fine-tuning learning rates with LoRA.** Too low; LoRA typically needs about 10× higher.
- **Attention-only LoRA by default.** Include the MLP layers.
- **Merging into a quantized base** without care: merging a bf16 adapter into 4-bit weights and re-quantizing loses precision; merge into a bf16 copy.
- **"Distillation copies the teacher exactly."** A student's capacity limits what it can absorb; forward-KL students may become blurry and reverse-KL students may drop modes.

## Key takeaways

- Distillation trains a student on a teacher's soft distributions (logit KD) or generated outputs (sequence KD); temperature exposes dark knowledge.
- Forward KL covers the teacher's modes; reverse KL commits to modes; on-policy distillation (GKD) evaluates the student's own samples to remove exposure bias.
- LoRA: $h = W_0x + \frac{\alpha}{r}BAx$ with $B=0$ at init; trains $r(d_{\text{in}}+d_{\text{out}})$ parameters per layer and merges for free at inference.
- QLoRA adds a 4-bit NF4 base, double quantization and paged optimizers; DoRA separates magnitude and direction.
- LoRA matches full fine-tuning for RL and small-to-medium SFT when applied to all layers with a ~10× learning rate.

## Check your understanding

??? question "Why is $B$ initialized to zero in LoRA, and why does it still receive gradients?"
    So that $\Delta W = BA = 0$ and the model starts exactly at the pre-trained weights. The gradient of the loss with respect to $B$ is proportional to $(Ax)^\top$, which is nonzero because $A$ is random, so $B$ starts learning immediately (and then $A$ does too).

??? question "How many LoRA parameters does a $5120\times5120$ layer have at rank 8?"
    $8\times(5120+5120) = 81{,}920$, versus $26{,}214{,}400$ for the full matrix.

??? question "A small student trained with forward KL on a multi-modal teacher produces bland, averaged text. What would you try?"
    Reverse KL (mode-seeking) or on-policy distillation with a reverse-KL or JSD objective, so the student commits to modes the teacher considers high-quality instead of spreading mass between them.

??? question "What is exposure bias and how does GKD address it?"
    Off-policy distillation trains on teacher/data prefixes, but at inference the student conditions on its own (possibly erroneous) tokens, which it never saw in training. GKD has the student generate sequences and has the teacher provide per-token targets on them, so training matches inference conditions.

??? question "Why can RL work with rank-1 LoRA when SFT on a large dataset cannot?"
    RL provides very little information per episode (about one scalar reward), so the total information to absorb is small; a large SFT dataset provides many bits per token and can exceed a low-rank adapter's capacity.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015 | Soft targets, temperature and "dark knowledge". |
| :material-book-open-variant: [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) — Kim & Rush, 2016 | Train the student on teacher-generated sequences; the basis of trace distillation. |
| :material-book-open-variant: [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023 | Student generates, teacher grades every token; fixes exposure bias. |
| :material-star: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 | Freeze weights, learn a low-rank update $\Delta W = BA$. |
| :material-star: [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023 | LoRA on a 4-bit NF4 base with double quantization and paged optimizers; 65B on one 48 GB GPU. |
| :material-book-open-variant: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al., 2024 | Decouples magnitude and direction; closes much of the LoRA–full-FT gap. |
| :material-new-box: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — Thinking Machines, 2025 | When LoRA matches full fine-tuning (RL, small/medium SFT), all-layer LoRA, the ~10× learning-rate rule. |
| :material-star: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | Shows distillation of reasoning traces beats RL on small models. |
