# 1.8 Training at Scale

!!! abstract "What you'll learn"
    - Exactly where GPU memory goes when training with mixed-precision Adam (~16 bytes per parameter), plus activations and **activation checkpointing**.
    - The parallelism toolbox: **data parallel**, **ZeRO 1–3 / FSDP**, **tensor parallel** (column/row split of the MLP, verified in code), **pipeline parallel** and its bubble, **sequence/context parallel**, **expert parallel**, and how they compose into 3D/4D parallelism.
    - Number formats (FP32, BF16, FP16, FP8) and how mixed precision works.
    - The optimizer stack: **AdamW**, warmup + cosine and **WSD** schedules, and **Muon** at an intuitive level.
    - Why training runs blow up (loss spikes, logit growth) and the standard fixes: **QK-norm**, **z-loss**, gradient clipping.

## Why this matters / the problem

The compute budgets of [1.7](07-scaling-laws-and-data.md) ($10^{24}$–$10^{25}$ FLOPs) are far beyond one GPU: an H100 delivers on the order of $10^{15}$ BF16 FLOP/s at peak, so $3.8 \times 10^{25}$ FLOPs would take it over a thousand years. Llama 3 405B was trained on up to 16,384 H100s. Making thousands of GPUs act as one requires splitting the model, the data and the optimizer state across devices while keeping communication from dominating, and keeping a months-long run numerically stable. A single loss spike that diverges can waste weeks of compute. This chapter is the systems and optimization side of pre-training.

## Intuition

Training a big model has three resource problems:

1. **It doesn't fit.** A 70B model needs over a terabyte for weights, gradients and optimizer state. Solution: *shard* those tensors across GPUs (ZeRO/FSDP) or split the layers themselves (tensor and pipeline parallelism).
2. **It's too slow.** Solution: process different data on different GPUs at the same time (data parallelism), and keep every GPU busy.
3. **It's fragile.** Low-precision arithmetic and huge learning rates are needed for speed but invite overflow and divergence. Solution: careful number formats, normalization, schedules and safety valves.

Each parallelism strategy cuts a different dimension of the computation, and each pays in a different kind of communication:

| strategy | what is split | communication | typical placement |
|---|---|---|---|
| data parallel (DP) | the batch | all-reduce of gradients each step | across nodes |
| ZeRO / FSDP | optimizer state, gradients, weights across DP ranks | gather weights, reduce-scatter gradients | across nodes |
| tensor parallel (TP) | each matrix multiply | all-reduce inside every layer | within a node (fast NVLink) |
| pipeline parallel (PP) | the stack of layers | activations between stages | across nodes |
| sequence / context parallel (SP/CP) | the sequence dimension | activations / K,V blocks | within or across nodes |
| expert parallel (EP) | MoE experts | all-to-all of tokens | within or across nodes |

## How it works

### Memory accounting

Let $\Psi$ be the number of parameters. With standard **mixed-precision** training and Adam, each parameter needs:

| item | format | bytes per parameter |
|---|---|---|
| weights used in forward/backward | BF16 | 2 |
| gradients | BF16 | 2 |
| master copy of weights | FP32 | 4 |
| Adam first moment $m$ | FP32 | 4 |
| Adam second moment $v$ | FP32 | 4 |
| **total** | | **16** |

So the "model state" costs $16\Psi$ bytes: 112 GB for a 7B model, 1.12 TB for 70B. An 80 GB GPU cannot even hold a 7B model's training state.

**Activations** come on top: every intermediate tensor saved in the forward pass for use in the backward pass. For a GPT-style layer with hidden size $h$, sequence length $s$ and micro-batch $b$, Korthikanti et al. (2022) estimate about $34\,sbh$ bytes per layer in 16-bit precision, plus $5\,a s^2 b$ bytes for attention scores and softmax ($a$ = number of heads) if attention is not fused. FlashAttention ([1.6](06-long-context-and-efficient-attention.md)) removes the $s^2$ term.

**Activation checkpointing** (Chen et al., 2016) saves only some activations, typically the input of each layer, and recomputes the rest during the backward pass. Full checkpointing stores about $2sbh$ bytes per layer and costs one extra forward pass, raising compute from $\approx 6N$ to $\approx 8N$ FLOPs per token. **Selective** checkpointing recomputes only the cheap-to-recompute but memory-heavy parts (such as attention softmax), getting most of the memory savings for a few percent extra compute.

### Data parallelism and ZeRO / FSDP

In **data parallelism (DP)**, each of $N_d$ GPUs holds a full replica of the model, processes a different slice of the global batch, and then gradients are averaged with an **all-reduce** so every replica takes the same step. It is simple and communication-efficient, but every GPU stores the full $16\Psi$ bytes: memory does not scale.

**ZeRO** (Rajbhandari et al., 2019) removes the redundancy: since every DP rank holds identical state, each rank only needs to *own* a $1/N_d$ shard of it.

- **Stage 1** shards the optimizer state ($12\Psi$ bytes: master weights, $m$, $v$). Each rank updates only its shard of parameters, then the updated weights are gathered.
- **Stage 2** also shards gradients: instead of all-reduce, a **reduce-scatter** gives each rank only the gradient shard it needs.
- **Stage 3** also shards the BF16 weights. Before each layer's forward (and backward), its full weights are **all-gathered** from all ranks, used, and freed.

Per-GPU memory for model state:

$$
\text{DP: } 16\Psi, \quad \text{ZeRO-1: } 4\Psi + \frac{12\Psi}{N_d}, \quad \text{ZeRO-2: } 2\Psi + \frac{14\Psi}{N_d}, \quad \text{ZeRO-3: } \frac{16\Psi}{N_d}.
$$

Stages 1 and 2 have the same communication volume as plain DP (an all-reduce is a reduce-scatter plus an all-gather); stage 3 adds weight gathers for about 1.5× DP's volume. PyTorch's **FSDP** (Fully Sharded Data Parallel) implements ZeRO-3-style sharding, with communication overlapped with computation by prefetching the next layer's weights.

### Tensor parallelism

**Tensor parallelism (TP)** (Megatron-LM, Shoeybi et al., 2019) splits individual weight matrices across GPUs. Consider the MLP $Y = \mathrm{GeLU}(XA)\,B$ with $X: [\text{tokens}, d]$, $A: [d, d_{ff}]$, $B: [d_{ff}, d]$, and $t$ GPUs.

1. Split $A$ by **columns**: $A = [A_1, \dots, A_t]$, each $[d, d_{ff}/t]$. GPU $i$ computes $H_i = \mathrm{GeLU}(X A_i)$. Because GeLU is elementwise, applying it to each column block separately is exactly the same as applying it to $XA$: no communication needed.
2. Split $B$ by **rows**: $B = [B_1; \dots; B_t]$, each $[d_{ff}/t, d]$. GPU $i$ computes $Z_i = H_i B_i$, a full-size $[\text{tokens}, d]$ partial result.
3. Since $HB = \sum_i H_i B_i$, one **all-reduce** (sum) produces $Y$ on every GPU.

So the whole MLP needs one all-reduce in the forward pass and one in the backward. Attention is split the same way along heads: each GPU computes a subset of heads (column split of $W_Q, W_K, W_V$), and the output projection $W_O$ is row-split. TP communicates inside every layer, so it is kept within a node, where GPUs are linked by NVLink, typically $t \le 8$.

**Sequence parallelism** (Korthikanti et al., 2022) complements TP: the operations between the TP regions (norms, dropout, residual adds) are replicated in plain TP; sequence parallelism instead splits them along the sequence dimension, turning each all-reduce into a reduce-scatter plus all-gather and cutting activation memory by $t$ for those parts.

### Pipeline parallelism

**Pipeline parallelism (PP)** puts consecutive groups of layers ("stages") on different GPUs: GPU 1 has layers 1–8, GPU 2 layers 9–16, and so on. Only activations at stage boundaries are communicated, which is cheap, so PP works across nodes. The problem is idling: stage 2 waits for stage 1's output. The fix is to split the batch into $m$ **micro-batches** and stream them through. With $p$ stages, at the start and end of each step some stages are idle; this **pipeline bubble** wastes a fraction

$$
\text{bubble} = \frac{p - 1}{m + p - 1}
$$

of the time for GPipe and 1F1B ("one forward, one backward") schedules. More micro-batches shrink the bubble; **interleaved** schedules (each GPU holds several non-adjacent stages) shrink it further at the cost of more communication. DeepSeek-V3's **DualPipe** overlaps the forward and backward computation of different micro-batches with MoE communication to nearly hide the bubble and the all-to-all cost.

### Context parallelism and expert parallelism

**Context parallelism (CP)** splits a single long sequence across GPUs, so activation memory per GPU is divided by the CP degree. The difficulty is attention: each token's query needs keys and values from the whole (preceding) sequence. **Ring attention** passes K/V blocks around a ring of GPUs while each GPU computes attention for its local queries block by block, using the online softmax of [1.6](06-long-context-and-efficient-attention.md). Llama 3 used CP for its long-context training stage.

**Expert parallelism (EP)** places different MoE experts on different GPUs, with all-to-all dispatch and combine of tokens ([1.5](05-mixture-of-experts.md)).

### Composing 3D/4D parallelism

Real runs nest these strategies, putting the most communication-hungry ones on the fastest links. The total GPU count is the product of the degrees: $N_{\text{GPU}} = t_{\text{TP}} \times c_{\text{CP}} \times p_{\text{PP}} \times d_{\text{DP}}$. Llama 3 405B, in its main pre-training configuration, used TP = 8 (within a node), PP = 16, and FSDP-style DP = 128 for $8 \times 16 \times 128 = 16{,}384$ GPUs, adding CP for long sequences. DeepSeek-V3 used no TP at all: 16-way PP (DualPipe), 64-way EP spanning 8 nodes, and ZeRO-1 DP, on 2,048 H800 GPUs.

A rough decision procedure: use FSDP/ZeRO DP alone while the model fits; add TP within a node when per-layer weights or activations are too big; add PP when the model still doesn't fit or DP scale is limited by batch size; add CP for long sequences; add EP for MoE.

### Mixed precision and number formats

A floating-point number has a sign, an exponent (range) and a mantissa (precision):

| format | exponent bits | mantissa bits | notes |
|---|---|---|---|
| FP32 | 8 | 23 | master weights, optimizer state, reductions |
| FP16 | 5 | 10 | small range; needs **loss scaling** to avoid gradient underflow |
| BF16 | 8 | 7 | FP32's range with low precision; the default for LLM training |
| FP8 E4M3 | 4 | 3 | more precision, less range; used for forward tensors |
| FP8 E5M2 | 5 | 2 | more range; sometimes used for gradients |

**Mixed precision** (Micikevicius et al., 2017) runs matrix multiplies in low precision (fast on tensor cores) while keeping an FP32 master copy of the weights so small updates are not lost to rounding, and performing sensitive operations (softmax, norms, loss, reductions) in FP32. FP16 needed loss scaling (multiply the loss by a large factor so small gradients don't underflow, then divide); BF16's wider exponent removes that need, which is why it became standard.

**FP8** doubles tensor-core throughput again but has very little range, so a single scale factor per tensor is not enough: one outlier forces everything else toward zero. DeepSeek-V3 trained in FP8 at scale with **fine-grained scaling** (a separate scale per $1 \times 128$ tile of activations and per $128 \times 128$ block of weights) and periodic promotion of partial sums to FP32 accumulators, keeping the loss within 0.25% of a BF16 baseline according to its report.

### Optimizers and learning-rate schedules

**AdamW** (Loshchilov & Hutter, 2017) is the default. With gradient $g_t$, learning rate $\eta$, decay rates $\beta_1, \beta_2$, weight decay $\lambda$ and small $\epsilon$:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\theta_t = \theta_{t-1} - \eta \Big( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \Big), \qquad \hat m_t = \frac{m_t}{1 - \beta_1^t},\ \hat v_t = \frac{v_t}{1 - \beta_2^t}.
$$

In words: move along a running average of the gradient ($m$), divided elementwise by a running estimate of its typical magnitude ($\sqrt{v}$), so every parameter takes steps of similar size; separately shrink weights toward zero ("decoupled" weight decay, applied outside the adaptive scaling, which is the "W" in AdamW). LLM runs commonly use $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$ and gradient clipping at norm 1.0.

**Learning-rate schedules:**

- **Warmup:** increase the learning rate linearly from 0 over the first few thousand steps. Early on, Adam's $v$ estimates are noisy and the network is far from equilibrium; a full learning rate immediately can diverge.
- **Cosine decay:** after warmup, decrease to a minimum (often 10% of peak) following half a cosine. The schedule length must be fixed in advance.
- **WSD (warmup–stable–decay)** (popularized by MiniCPM): warm up, hold the learning rate constant for most of training, then decay quickly (e.g. over the last 10–20% of steps). Any checkpoint from the stable phase can be "cooled down" into a finished model, so you can extend training or run scaling experiments without restarting, and the decay phase pairs naturally with data annealing ([1.7](07-scaling-laws-and-data.md)).

**Muon** (Keller Jordan and collaborators, 2024) treats each 2-D weight matrix as a matrix rather than a bag of numbers. It keeps a momentum $M$ of the gradients, then replaces $M$ by an approximately **orthogonalized** version: if $M = U \Sigma V^\top$ (singular value decomposition), the update is $\approx U V^\top$, i.e. all singular values set to 1. Intuition: gradients of weight matrices tend to be dominated by a few directions; orthogonalizing boosts the rare directions so the update changes the matrix "evenly" in all directions. Computing the SVD is expensive, so Muon uses a few **Newton–Schulz** iterations, a polynomial map that pushes singular values toward 1 using only matrix multiplies. Moonshot's Moonlight work showed Muon scales to large LLMs with weight decay and a per-matrix update rescaling, reporting about 2× compute efficiency over AdamW; Kimi K2 was trained with a variant, MuonClip. Muon is applied to hidden weight matrices; embeddings, output layers and norm parameters still use AdamW.

### Training instabilities and fixes

Large runs suffer **loss spikes**: the loss jumps suddenly, sometimes recovering, sometimes diverging. Common causes and fixes:

- **Attention-logit growth.** Query/key norms grow during training until $q \cdot k$ becomes huge, softmax saturates to one-hot, and gradients become erratic. Fix: **QK-norm** ([1.4](04-modern-block.md)), or Kimi K2's **QK-clip** (part of MuonClip), which rescales $W_Q$ and $W_K$ whenever the maximum attention logit of a head exceeds a threshold. Wortsman et al. (2023) reproduce this instability in small models at high learning rates and show QK-layernorm removes it.
- **Output-logit divergence.** The output logits drift to large magnitudes. Fix: **z-loss** (used in PaLM): add $10^{-4} \cdot (\log Z)^2$ to the loss, where $Z = \sum_i e^{z_i}$ is the softmax normalizer. This keeps $\log Z$ near 0, so logits stay in a well-conditioned range. ST-MoE applies the same idea to router logits.
- **Gradient explosions.** **Gradient clipping** rescales the full gradient vector $g$ to norm at most $c$ (e.g. 1.0) whenever $\lVert g \rVert > c$: $g \leftarrow g \cdot \min(1, c / \lVert g \rVert)$. This bounds the size of any single update.
- **Bad batches and recovery.** PaLM handled persistent spikes by restarting from a checkpoint about 100 steps before the spike and skipping a few hundred data batches. Monitoring gradient norms and per-layer statistics helps catch problems early.
- **Others:** lower peak learning rate, longer warmup, smaller $\beta_2$, $\epsilon$ tuning, keeping norms and softmax in FP32, and removing biases. Kimi K2 reports training on 15.5T tokens with no loss spikes using MuonClip.

## Code

**Memory, tensor parallelism and pipeline bubbles, in one script.** The tensor-parallel MLP is simulated with a list of shards and checked against the unsplit computation.

```python
import torch
import torch.nn.functional as F

def zero_memory_gb(n_params, n_gpus, stage):
    """Per-GPU bytes for weights+grads+optimizer with BF16 params/grads and FP32 Adam (16 B/param)."""
    p, g, opt = 2 * n_params, 2 * n_params, 12 * n_params   # bf16 w, bf16 grad, fp32 master+m+v
    if stage >= 1: opt /= n_gpus
    if stage >= 2: g /= n_gpus
    if stage >= 3: p /= n_gpus
    return (p + g + opt) / 1e9

for s in range(4):
    print(f"7B on 64 GPUs, ZeRO-{s}: {zero_memory_gb(7e9, 64, s):6.1f} GB per GPU")

# Tensor parallelism of an MLP  Y = GeLU(X A) B, simulated with a list of "GPUs".
torch.manual_seed(0)
d, d_ff, tp = 8, 32, 4
X = torch.randn(5, d)                       # [tokens, d], replicated on every GPU
A = torch.randn(d, d_ff)                    # first layer: split by COLUMNS
B = torch.randn(d_ff, d)                    # second layer: split by ROWS
A_shards = A.chunk(tp, dim=1)               # GPU i holds A[:, i-th slice]
B_shards = B.chunk(tp, dim=0)               # GPU i holds B[i-th slice, :]
partials = [F.gelu(X @ A_i) @ B_i for A_i, B_i in zip(A_shards, B_shards)]  # no communication
Y_tp = torch.stack(partials).sum(0)         # the single all-reduce (sum) per MLP forward
Y_ref = F.gelu(X @ A) @ B
print("tensor-parallel MLP matches:", torch.allclose(Y_tp, Y_ref, atol=1e-5))

def bubble_fraction(p, m):
    """Idle fraction of a GPipe/1F1B pipeline with p stages and m micro-batches."""
    return (p - 1) / (m + p - 1)

print(f"bubble p=16, m=64: {bubble_fraction(16, 64):.1%};  m=256: {bubble_fraction(16, 256):.1%}")
```

Output: 112.0, 29.3, 15.5 and 1.8 GB per GPU for ZeRO stages 0–3; the TP check prints `True`; bubbles of 19.0% and 5.5%. In a real system each shard lives on a different GPU and the `sum` is a `torch.distributed.all_reduce`.

**Learning-rate schedules and Muon's Newton–Schulz orthogonalization.**

```python
import math, torch

def lr_cosine(step, peak, warmup, total, min_ratio=0.1):
    if step < warmup:
        return peak * step / warmup                                   # linear warmup
    t = (step - warmup) / (total - warmup)                            # progress in [0, 1]
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

def lr_wsd(step, peak, warmup, total, decay_frac=0.2):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * step / warmup                                   # warmup
    if step < decay_start:
        return peak                                                   # stable
    return peak * (total - step) / (total - decay_start)              # linear decay to 0

for s in [0, 1000, 25000, 50000, 85000, 100000]:
    print(f"step {s:6d}: cosine {lr_cosine(s, 3e-4, 2000, 100000):.2e}   WSD {lr_wsd(s, 3e-4, 2000, 100000):.2e}")

def newton_schulz(G, steps=5):
    """Muon's orthogonalization: push all singular values of G toward ~1 (G -> ~U V^T)."""
    a, b, c = 3.4445, -4.7750, 2.0315             # quintic coefficients from Keller Jordan's Muon
    X = G / (G.norm() + 1e-7)                     # scale so singular values are <= 1
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X

torch.manual_seed(0)
G = torch.randn(64, 32) @ torch.diag(torch.logspace(0, 2, 32))   # badly conditioned "momentum"
print("singular values before:", torch.linalg.svdvals(G)[[0, 15, 31]].round(decimals=2).tolist())
print("singular values after :", torch.linalg.svdvals(newton_schulz(G))[[0, 15, 31]].round(decimals=2).tolist())
```

The singular values go from roughly (919, 74, 6) to (1.20, 0.87, 0.68): a 150:1 spread compressed to under 2:1 in five iterations of matrix multiplies. Muon deliberately trades exact orthogonality for speed; the tuned coefficients make singular values land in a band around 1 rather than converge to exactly 1.

## Worked example

**Training a 70B dense model: does it fit?**

- Model state: $16 \times 70 \times 10^9 = 1.12$ TB. On 80 GB GPUs with plain DP: impossible.
- ZeRO-3/FSDP over 64 GPUs: $1.12\,\text{TB} / 64 = 17.5$ GB per GPU for model state. ✓
- Activations for one 4096-token sequence, using the $34\,sbh$ estimate with $h = 8192$ and 80 layers: $34 \times 4096 \times 8192 \times 80 \approx 91$ GB without checkpointing. ✗ Too big.
- With full activation checkpointing ($2sbh$ per layer): $2 \times 4096 \times 8192 \times 80 \approx 5.4$ GB, plus one layer's full activations ($\approx 1.1$ GB) during recomputation. ✓ The price is $\approx 8N$ instead of $6N$ FLOPs per token.
- Alternatively, TP = 8 within each node divides most activations (and weights) by 8, avoiding full recomputation: $91/8 \approx 11$ GB with sequence parallelism. This is why large dense models combine TP with FSDP.

**For a 7B model** ($\Psi = 7 \times 10^9$, 64 GPUs): DP needs 112 GB/GPU; ZeRO-1 needs $4\Psi + 12\Psi/64 = 28 + 1.3 = 29.3$ GB; ZeRO-2 needs $2\Psi + 14\Psi/64 = 14 + 1.5 = 15.5$ GB; ZeRO-3 needs $16\Psi/64 = 1.75$ GB.

**Pipeline bubble.** 16 stages with 64 micro-batches waste $15/79 = 19\%$ of time; with 256 micro-batches, $15/271 = 5.5\%$. But 256 micro-batches per step requires a large global batch, which is one reason pipeline depth is limited.

## In practice (2025–2026)

- **Llama 3 405B**: BF16, TP 8 × PP 16 × DP (FSDP) 128 on 16,384 H100s in its main configuration, plus context parallelism for long-context training; the report also documents frequent hardware failures and automated recovery at this scale.
- **DeepSeek-V3**: FP8 mixed precision with fine-grained scaling, DualPipe pipeline schedule, 64-way EP, ZeRO-1, no TP; the report states 2.788M H800 GPU-hours for full training.
- **Kimi K2**: MuonClip optimizer, 15.5T tokens, no loss spikes reported.
- **Schedules**: WSD-style schedules (or cosine with a long constant phase) are common, because they combine naturally with annealing data and allow continued training.
- **Hyperparameters are transferred** from small proxy runs (μP or empirical scaling of learning rate and batch size with model size).
- **FP8 training** is increasingly common on Hopper/Blackwell GPUs; 4-bit formats (MXFP4, NVFP4) are mostly used for inference so far, e.g. gpt-oss's MoE weights are released in MXFP4.

## Common pitfalls / misconceptions

- **"Parameters × 2 bytes" for training memory.** That is inference weights only. Training with Adam needs ~16 bytes/parameter plus activations.
- **Using TP across nodes.** TP communicates inside every layer; over slower inter-node links it usually kills throughput.
- **Ignoring the bubble.** Deep pipelines with few micro-batches waste a large fraction of the cluster.
- **Changing the batch size without retuning the learning rate.** Optimal LR depends on batch size; a bigger batch usually allows a larger LR, up to a point.
- **Treating loss spikes as noise.** Small spikes can be precursors of divergence; track gradient norms, max attention logits and output logit magnitudes.
- **FP8 everywhere.** Even in FP8 runs, embeddings, the output layer, norms, attention softmax and the optimizer state stay in higher precision.

## Key takeaways

- Mixed-precision Adam costs ~16 bytes per parameter; activations come on top and are controlled by checkpointing.
- ZeRO/FSDP shards optimizer state, gradients and weights across data-parallel ranks, cutting per-GPU model state from $16\Psi$ to $16\Psi/N_d$.
- Tensor parallelism splits the MLP's first matrix by columns and the second by rows, needing one all-reduce per MLP; it belongs inside a node.
- Pipeline parallelism splits layers into stages; its bubble fraction is $(p-1)/(m+p-1)$. Context and expert parallelism split sequences and experts.
- BF16 is the default; FP8 needs fine-grained scaling. AdamW with warmup and cosine or WSD decay is standard; Muon orthogonalizes updates for better efficiency.
- Instabilities come from logit growth and outliers; QK-norm, z-loss and gradient clipping are the standard defenses.

## Check your understanding

??? question "How many 80 GB GPUs are needed, at minimum, just to hold the training state of a 30B model with ZeRO-3 (ignoring activations)?"
    $16 \times 30 \times 10^9 = 480$ GB; $480 / 80 = 6$ GPUs, so at least 6 (more in practice, for activations and buffers).

??? question "Why does splitting $A$ by columns and $B$ by rows avoid communication between the two matrix multiplies?"
    Column blocks of $XA$ can be computed independently, and the elementwise GeLU can be applied to each block separately. Row blocks of $B$ consume exactly those column blocks, producing partial sums of the final output, so communication is needed only once, to add the partial sums.

??? question "What does BF16 give up compared with FP16, and why is that a good trade for training?"
    Precision: 7 mantissa bits instead of 10. In exchange it has FP32's 8-bit exponent, so gradients and activations rarely overflow or underflow, which removes the need for loss scaling. Precision loss is mitigated by FP32 master weights and FP32 accumulation.

??? question "What problem does the z-loss address, and how?"
    Output logits drifting to large magnitudes, which can make the softmax numerically unstable. Penalizing $(\log Z)^2$ keeps the log-normalizer near 0, so logits stay in a moderate range without changing which token is preferred.

??? question "Why is WSD convenient for scaling-law experiments?"
    One long stable-phase run can be branched at many points, each followed by a short decay, to get fully trained models for several token budgets, instead of running a separate cosine schedule for each budget.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | **Tensor parallelism**: the column/row split of the MLP and head split of attention. |
| :material-star: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019 | Shard optimizer states, gradients and params across data-parallel ranks; the memory formulas above. |
| :material-book-open-variant: [PyTorch FSDP](https://arxiv.org/abs/2304.11277) — Zhao et al., 2023 | How ZeRO-3-style sharding is implemented and tuned in PyTorch. |
| :material-book-open-variant: [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., 2018 | Pipeline parallelism with micro-batches and the bubble. |
| :material-book-open-variant: [Efficient Large-Scale LM Training on GPU Clusters (PTD-P)](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021 | How to compose tensor + pipeline + data parallelism; interleaved 1F1B schedule. |
| :material-book-open-variant: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022 | **Sequence parallelism** + selective activation checkpointing; the activation-memory formula. |
| :material-book-open-variant: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Chen et al., 2016 | The original activation checkpointing (recomputation) paper. |
| :material-book-open-variant: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2017 | FP16/BF16 compute with FP32 master weights and loss scaling. |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | Defines E4M3 and E5M2 and shows FP8 training matches 16-bit quality. |
| :material-book-open-variant: [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2017 | The default optimizer for LLMs. |
| :material-book-open-variant: [MiniCPM](https://arxiv.org/abs/2404.06395) — Hu et al., 2024 | Introduces the WSD learning-rate schedule and uses it for scaling experiments. |
| :material-book-open-variant: [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — Keller Jordan, 2024 | The original Muon write-up: momentum + Newton–Schulz orthogonalization, with the coefficients used above. |
| :material-new-box: [Muon is Scalable for LLM Training (Moonlight)](https://arxiv.org/abs/2502.16982) — Liu et al. (Moonshot AI), 2025 | Makes Muon work at scale (weight decay, update rescaling); ~2× compute efficiency vs AdamW. |
| :material-book-open-variant: [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) — Wortsman et al., 2023 | Reproduces attention-logit growth and output-logit divergence in small models; evaluates QK-layernorm and z-loss. |
| :material-book-open-variant: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022 | z-loss, loss-spike recovery by skipping batches, and many large-scale training details. |
| :material-book-open-variant: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) — Google DeepMind | The best free book on rooflines, sharding and TPU/GPU math. |
