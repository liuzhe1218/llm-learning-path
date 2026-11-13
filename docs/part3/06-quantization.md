# 3.6 Quantization

!!! abstract "What you'll learn"
    - Read the bit layouts of FP32, BF16, FP16, FP8 (E4M3/E5M2), INT8, INT4, FP4, MXFP4 and NVFP4, and know their ranges.
    - Do symmetric and asymmetric quantization by hand and in code, and choose a **granularity** (per-tensor, per-channel, per-group, per-block).
    - Explain **weight-only** vs **weight + activation** quantization and which phase of inference each speeds up.
    - Explain the **outlier problem** and the three classic fixes: GPTQ (second-order error compensation), AWQ (activation-aware scaling) and SmoothQuant (derive its equivalence).
    - Evaluate a quantized model properly and pick a scheme for a deployment.

## Why this matters / the problem

From Chapter 3.1: batch-1 decode time ≈ bytes read ÷ bandwidth. A 70B-parameter model is ≈ 140 GB in BF16; it does not fit on one 80 GB GPU, and every token requires streaming all of it. Store the weights in 4 bits and the model is ≈ 35 GB plus a little overhead: it fits on one GPU, leaves room for KV cache, and each decode step reads ~4× fewer bytes. Quantize activations too, and the matrix multiplies themselves can run on low-precision tensor cores that offer 2× (FP8) or more the FLOP/s of BF16.

The catch is accuracy. Rounding billions of numbers to a handful of levels introduces error, and LLMs have a peculiar statistical structure — a few enormous **outlier** values — that makes naive rounding fail. This chapter builds quantization from the bit level up, then explains the techniques that make 8-bit nearly lossless and 4-bit practical.

## Intuition

Quantization is choosing a **ruler**. A real number line is replaced by a set of tick marks; every value is snapped to the nearest tick. Two choices define the ruler:

- **How many ticks** (the bit width): 8 bits → 256 ticks, 4 bits → 16 ticks.
- **Where the ticks are** (the scale, and for floats the exponent spacing): the ruler must cover the largest value you need to represent, so a single huge value stretches the ruler and leaves the small values with coarse ticks.

That last point is the outlier problem in one sentence: one value of 100 in a group of values around 1 forces the tick spacing to ~100/7 at 4 bits, and every "normal" value rounds to 0. Most of the clever methods in this chapter are ways to **keep outliers from setting the scale** for everyone else.

## How it works

### Number formats

A floating-point number has a **sign** bit, **exponent** bits (setting the power of two, i.e. the range) and **mantissa** bits (the precision within each power of two). An integer format has evenly spaced values and needs a separate scale factor.

| Format | Bits (sign/exp/mantissa) | Approx. max | Notes |
|---|---|---|---|
| FP32 | 1 / 8 / 23 | $3.4\times10^{38}$ | training master weights |
| BF16 | 1 / 8 / 7 | $3.4\times10^{38}$ | FP32's range, ~3 significant digits; default for weights/activations |
| FP16 | 1 / 5 / 10 | 65,504 | more precision, much less range than BF16 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | more precision; used for weights and activations |
| FP8 E5M2 | 1 / 5 / 2 | 57,344 | more range; used mainly for gradients in training |
| INT8 | 8-bit integer | −128 … 127 | needs a scale |
| INT4 | 4-bit integer | −8 … 7 | needs a scale |
| FP4 E2M1 | 1 / 2 / 1 | 6 | representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| MXFP4 | FP4 E2M1 elements, blocks of 32 share an 8-bit power-of-two scale (E8M0) | — | OCP Microscaling standard; ≈ 4.25 bits/element |
| NVFP4 | FP4 E2M1 elements, blocks of 16 share an FP8 E4M3 scale, plus a per-tensor FP32 scale | — | NVIDIA Blackwell format; ≈ 4.5 bits/element |

```text
BF16      s eeeeeeee mmmmmmm            FP8 E4M3  s eeee mmm
FP16      s eeeee mmmmmmmmmm            FP8 E5M2  s eeeee mm
FP32      s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm      FP4 E2M1  s ee m
MXFP4 block: [ shared scale E8M0 (8 bits) ] [ 32 x FP4 E2M1 ]
NVFP4 block: [ shared scale FP8 E4M3 ]      [ 16 x FP4 E2M1 ]   (+ one FP32 scale per tensor)
```

Low-bit floats have non-uniform spacing (dense near zero, sparse for large values), which matches the bell-shaped distribution of weights better than uniform integers do. **Block-scaled** formats (MX, NVFP4) attach a scale to every small block, so an outlier only stretches the ruler for its own 16–32 neighbours.

### Symmetric and asymmetric quantization

To quantize a real tensor $x$ to $b$-bit integers:

**Symmetric** (zero maps to zero): with $q_{\max} = 2^{b-1}-1$,

$$
s = \frac{\max |x|}{q_{\max}}, \qquad q = \operatorname{clamp}\!\Big(\operatorname{round}\big(\tfrac{x}{s}\big), -q_{\max}, q_{\max}\Big), \qquad \hat x = s\,q .
$$

In words: pick the step size $s$ so the largest magnitude lands on the last tick, divide, round, and store the integer $q$; dequantize by multiplying back.

**Asymmetric** (zero-point): use the full unsigned range $[0, 2^b - 1]$ for data that is not centred at zero,

$$
s = \frac{\max x - \min x}{2^b - 1}, \qquad z = \operatorname{round}\!\Big(\frac{-\min x}{s}\Big), \qquad q = \operatorname{clamp}\!\Big(\operatorname{round}\big(\tfrac{x}{s}\big) + z,\ 0,\ 2^b - 1\Big), \qquad \hat x = s\,(q - z).
$$

In words: the integer $z$ ("zero-point") represents real 0; the ticks span exactly $[\min x, \max x]$ instead of $[-\max|x|, \max|x|]$, so no ticks are wasted on a side the data does not use.

The rounding error per element is at most $s/2$ (when not clipped), so everything that **shrinks $s$** — smaller ranges, finer granularity — reduces error.

### Granularity

Which elements share one scale?

- **Per-tensor:** one scale for the whole matrix. Cheapest, least accurate.
- **Per-channel:** one scale per output row of a weight matrix (or per token for activations).
- **Per-group:** one scale per group of $G$ consecutive weights along the input dimension (commonly $G = 128$ for INT4 weight-only methods). Overhead: a 16-bit scale per 128 weights adds $16/128 = 0.125$ bits per weight.
- **Per-block:** one scale per small block (32 for MXFP4, 16 for NVFP4), or 2-D blocks such as the $128\times128$ weight blocks used by DeepSeek-V3's FP8 recipe.

Finer granularity isolates outliers and lowers error, at the cost of storing more scales and more complex kernels.

### Weight-only vs weight + activation

- **Weight-only** (e.g. W4A16: 4-bit weights, 16-bit activations). Weights are stored in 4 bits and **dequantized to BF16 inside the matmul kernel**. Bytes read drop ~4×, so **memory-bound decode speeds up**; the arithmetic still runs in BF16, so **compute-bound prefill does not** (dequantization can even make it slightly slower). Best for low-batch, latency-oriented serving and for fitting big models on fewer GPUs.
- **Weight + activation** (W8A8 in INT8 or FP8, W4A8, W4A4/FP4). Both operands are low precision and the matmul runs on low-precision tensor cores (on an H100 SXM, dense FP8 ≈ 1,979 TFLOP/s vs ≈ 989 for BF16). This speeds up **compute-bound** work as well — prefill and large-batch decode — but activations are harder to quantize because they are computed on the fly and contain outliers.

### The outlier problem

LLM.int8() found that beyond a certain model scale (around 6.7B parameters in their study), transformer activations develop **systematic outlier features**: a small number of hidden dimensions whose values are up to orders of magnitude larger than the rest, consistently across tokens and layers. Quantizing such an activation per-tensor (or per-token) to INT8 lets those dimensions set the scale and wipes out the precision of everything else. Their fix was mixed precision: keep the outlier dimensions in FP16 and quantize the rest. Weights are better behaved, but some weight channels are far more **important** than others because they multiply those large activations.

### GPTQ: compensate the error you just made

GPTQ quantizes one linear layer at a time to minimize the error in its **output** on calibration data $X$ (a few hundred sequences):

$$
\min_{\hat W} \big\| W X - \hat W X \big\|_2^2 .
$$

Quantizing every weight to its nearest grid point independently ignores that errors in different weights can cancel. GPTQ quantizes the input columns of $W$ one at a time and, after quantizing column $j$ (incurring error $e_j = w_j - \operatorname{quant}(w_j)$), **adjusts all not-yet-quantized columns** to compensate:

$$
W_{:,\,F} \;\leftarrow\; W_{:,\,F} - \frac{e_j}{[H^{-1}]_{jj}} \, [H^{-1}]_{j,\,F}, \qquad H = 2 X X^\top ,
$$

where $F$ is the set of remaining columns and $H$ is the Hessian of the layer's squared error with respect to its weights. In words: input features that are correlated with feature $j$ in the data can absorb its rounding error, and $H^{-1}$ says exactly how much each should move so the layer's output changes least. This "second-order" (curvature-aware) error compensation comes from the Optimal Brain Surgeon line of pruning work; GPTQ's contribution was making it fast enough (column order shared across rows, lazy batched updates, a Cholesky formulation) to quantize 175B-parameter models to 3–4 bits in a few GPU-hours.

### AWQ: protect the salient channels by scaling

AWQ observes that a small fraction of weight **input channels** matter far more than the rest, and that they can be found from **activation** magnitudes (channels that multiply large activations). Rather than keeping them in high precision (mixed formats are hardware-unfriendly), AWQ scales them. For an input channel with scale $s > 1$:

$$
y = \sum_j w_j x_j = \sum_j (s_j w_j)\Big(\frac{x_j}{s_j}\Big) \;\;\to\;\; \sum_j \operatorname{quant}(s_j w_j)\,\frac{x_j}{s_j}.
$$

Scaling one weight up by $s$ usually does not change the group's maximum, so the step size and the absolute rounding error of $s_j w_j$ stay about the same; dividing by $s_j$ afterwards shrinks that channel's error by about $1/s_j$. AWQ chooses $s = s_X^{\,\alpha}$ (with $s_X$ the average activation magnitude per channel) and grid-searches $\alpha \in [0,1]$ per layer to minimize output error on calibration data. The $1/s$ factor is folded into the preceding operation (e.g. a normalization layer's weights), so inference cost is unchanged. No backpropagation or weight reconstruction is involved, which makes AWQ fast and less prone to overfitting the calibration set.

### SmoothQuant: move the difficulty from activations to weights

For W8A8, activations with outlier channels are hard to quantize while weights are easy. SmoothQuant moves "difficulty" across the multiplication. For $Y = XW$ with $X \in \mathbb{R}^{T \times C_{\text{in}}}$ and $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$, and any positive vector $s \in \mathbb{R}^{C_{\text{in}}}$,

$$
Y = XW = \big(X \operatorname{diag}(s)^{-1}\big)\big(\operatorname{diag}(s)\, W\big) = \hat X \hat W ,
$$

because $\operatorname{diag}(s)^{-1}\operatorname{diag}(s) = I$. This is **exactly** equivalent in real arithmetic: input channel $j$ of $X$ is divided by $s_j$ and row $j$ of $W$ is multiplied by $s_j$. SmoothQuant chooses

$$
s_j = \frac{\max_t |X_{tj}|^{\alpha}}{\max_k |W_{jk}|^{1-\alpha}} .
$$

Then the new channel maxima are $\max|\hat X_{:,j}| = \max|X_{:,j}|^{1-\alpha} \cdot \max|W_{j,:}|^{1-\alpha}$ and $\max|\hat W_{j,:}| = \max|X_{:,j}|^{\alpha}\cdot\max|W_{j,:}|^{\alpha}$. With $\alpha = 0.5$ both equal $\sqrt{\max|X_{:,j}|\cdot\max|W_{j,:}|}$: activation and weight ranges are balanced, so neither is dominated by outliers. In words: an outlier activation channel is shrunk, and the matching weight row is enlarged to compensate; since weights were easy to quantize, they can afford it. The $1/s$ is folded offline into the previous LayerNorm/RMSNorm weights, so there is no runtime cost.

### KV-cache quantization

The KV cache (Chapter 3.2) is activation data, written at runtime. FP8 KV with a per-tensor (or per-head) scale is widely used and close to lossless for most models. Lower bits need finer granularity matched to the data: KIVI quantizes **keys per channel** (keys have outlier channels) and **values per token**, reaching 2 bits; QServe co-designs W4A8 with a 4-bit KV cache and the GPU kernels needed to actually realize a throughput gain.

### Evaluating a quantized model

1. **Same harness, same prompts, same sampling** for the BF16 baseline and the quantized model.
2. **Perplexity** on held-out text is a quick, sensitive smoke test, but small perplexity changes can hide large task regressions.
3. **Downstream tasks** relevant to your product: knowledge (e.g. MMLU-style), math (GSM8K-style), code, long-context retrieval, instruction following, multilingual.
4. **Long generations**: reasoning models produce thousands of tokens and small per-token errors compound; evaluate with multiple seeds and report variance.
5. **Distribution-level checks**: KL divergence between baseline and quantized next-token distributions, or the fraction of answers that "flip" between correct and incorrect.
6. **Speed and memory** on the target hardware and batch sizes, since a format without native kernel support can be slower than BF16.

## Code

Symmetric/asymmetric quantization, three granularities, and a SmoothQuant demonstration on activations with outlier channels. PyTorch on CPU; quantization is simulated by quantize-then-dequantize ("fake quantization"), which is how accuracy is studied before writing kernels.

```python
import torch
torch.manual_seed(0)

def quant_sym(x, bits, dim=None):
    """Symmetric quantization. dim=None: per-tensor; dim=k: one scale per slice along k."""
    qmax = 2 ** (bits - 1) - 1                            # 127 for INT8, 7 for INT4
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    s = amax.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / s), -qmax, qmax)      # integers stored in memory
    return q * s                                          # dequantized value

def quant_asym(x, bits):
    """Asymmetric (zero-point) quantization of a whole tensor to [0, 2^bits - 1]."""
    qmax = 2 ** bits - 1
    s = (x.max() - x.min()).clamp(min=1e-8) / qmax
    z = torch.round(-x.min() / s)                         # integer that represents 0.0
    q = torch.clamp(torch.round(x / s) + z, 0, qmax)
    return (q - z) * s

def quant_group(w, bits, group=128):
    """Per-group symmetric weight quantization along the input dim (GPTQ/AWQ style)."""
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // group, group)
    return quant_sym(wg, bits, dim=-1).reshape(out_f, in_f)

w = torch.tensor([0.12, -0.5, 0.31, 0.9, -0.05, 0.0, 0.44, -0.27])
for name, wq in [("sym INT4 ", quant_sym(w, 4)), ("asym INT4", quant_asym(w, 4))]:
    print(name, [round(v, 3) for v in wq.tolist()], "MSE", f"{((w - wq) ** 2).mean():.5f}")

# Granularity matters: a weight matrix whose rows have very different scales.
W = torch.randn(256, 1024) * torch.logspace(-2, 0, 256)[:, None]
for name, Wq in [("per-tensor", quant_sym(W, 4)), ("per-channel", quant_sym(W, 4, dim=1)),
                 ("per-group128", quant_group(W, 4))]:
    print(f"INT4 {name:13s} rel. error {((W - Wq).norm() / W.norm()):.3f}")

# SmoothQuant: activations with outlier channels, W8A8 with and without smoothing.
X = torch.randn(64, 1024); X[:, :8] *= 50                 # 8 outlier input channels
W = torch.randn(1024, 1024) / 32
Y = X @ W
def w8a8(X, W):                                           # per-token acts, per-out-channel weights
    return quant_sym(X, 8, dim=1) @ quant_sym(W, 8, dim=0)
alpha = 0.5
s = X.abs().amax(0) ** alpha / W.abs().amax(1) ** (1 - alpha)     # one scale per input channel
Xs, Ws = X / s, W * s[:, None]                            # (X diag(s)^-1)(diag(s) W) == XW
print("smoothing is exact:", torch.allclose(Xs @ Ws, Y, rtol=1e-3, atol=1e-2))
for name, Yq in [("W8A8 plain   ", w8a8(X, W)), ("W8A8 smoothed", w8a8(Xs, Ws))]:
    print(name, f"rel. error {((Y - Yq).norm() / Y.norm()):.4f}")
```

On this toy data, per-group INT4 has roughly a third of the error of per-tensor INT4, and smoothing cuts the W8A8 output error by about 4×.

## Worked example

**By hand: 8 weights to INT4.** $w = [0.12, -0.50, 0.31, 0.90, -0.05, 0.00, 0.44, -0.27]$.

*Symmetric.* $q_{\max} = 7$, $s = 0.90 / 7 = 0.1286$.
$w/s = [0.93, -3.89, 2.41, 7.00, -0.39, 0, 3.42, -2.10]$ → $q = [1, -4, 2, 7, 0, 0, 3, -2]$ → $\hat w = [0.129, -0.514, 0.257, 0.900, 0, 0, 0.386, -0.257]$. Mean squared error ≈ $1.09\times10^{-3}$. Note the range $[-0.9, 0.9]$ is covered but the data only reaches $-0.5$ on the negative side: about a quarter of the 15 levels are wasted.

*Asymmetric.* $s = (0.90 - (-0.50))/15 = 0.0933$, $z = \operatorname{round}(0.50/0.0933) = \operatorname{round}(5.36) = 5$.
$q = \operatorname{round}(w/s) + 5 = [6, 0, 8, 15, 4, 5, 10, 2]$ → $\hat w = s(q - 5) = [0.093, -0.467, 0.280, 0.933, -0.093, 0, 0.467, -0.280]$. MSE ≈ $0.82\times10^{-3}$, about 25% lower, because all 16 levels are used.

**Model sizes (weights only).** A 70B-parameter model:

| Scheme | Bits/weight (incl. scales) | Size |
|---|---|---|
| BF16 | 16 | ≈ 140 GB |
| FP8 (per-channel scales) | ≈ 8 | ≈ 70 GB |
| INT4, group 128, FP16 scale | 4 + 0.125 ≈ 4.13 | ≈ 36 GB |
| MXFP4 (32-block, 8-bit scale) | 4 + 8/32 = 4.25 | ≈ 37 GB |
| NVFP4 (16-block, 8-bit scale) | 4 + 8/16 = 4.5 | ≈ 39 GB |

With 4-bit weights the model fits on one 80 GB H100 with ~40 GB left for KV cache, and the batch-1 bandwidth bound improves from 140/3.35 ≈ 42 ms per token (which also would not fit) to ≈ 36/3.35 ≈ 11 ms per token.

## In practice (2025–2026)

- **FP8 (W8A8) is the default** for serving large models on Hopper and Blackwell: near-lossless when done with per-channel or block scales, and it speeds up both prefill and decode. DeepSeek-V3 was even *trained* in FP8 with fine-grained block scaling, and its released weights are FP8.
- **4-bit weight-only** (AWQ, GPTQ, and GGUF k-quants in llama.cpp) is the standard way to run models on a single GPU, consumer hardware or Apple Silicon.
- **FP4** became practical with Blackwell's FP4 tensor cores: OpenAI's gpt-oss models ship their MoE weights in MXFP4, and NVIDIA promotes NVFP4 checkpoints for TensorRT-LLM and other engines.
- **MoE** models are especially good targets: expert weights dominate the parameter count and memory, while attention and shared layers can stay at higher precision.
- **KV cache in FP8** is a routine option; lower-bit KV is used selectively and evaluated per model.
- Rule of thumb: FP8 ≈ lossless; 4-bit weight-only is a good quality/size trade-off; aggressive activation or KV quantization needs evaluation on *your* tasks — reasoning models with long outputs are more sensitive.

## Common pitfalls / misconceptions

- **"4-bit makes everything 4× faster."** Weight-only quantization speeds memory-bound decode; prefill and large-batch decode stay compute-bound in BF16 unless activations are quantized too.
- **Judging quality only by perplexity.** Use task benchmarks, especially long-generation and long-context tasks.
- **Calibration data mismatch.** GPTQ/AWQ/SmoothQuant use calibration samples; calibrating on generic web text and serving code or Chinese can hurt. Use data resembling your traffic.
- **Unsupported formats.** A checkpoint in a format your GPU or engine has no fast kernel for will be dequantized on the fly and can be slower than BF16 (e.g. FP8 compute requires Hopper/Ada or newer; FP4 tensor cores require Blackwell).
- **Quantizing everything uniformly.** Embeddings, the LM head, normalization, router (gate) weights and attention are often kept at higher precision because they are small or sensitive.
- **Forgetting the scales in size estimates.** Group/block scales add 0.125–0.5 bits per weight.

## Key takeaways

- A quantizer is a scale (and optionally a zero-point) plus rounding; the error is at most half a step, so anything that shrinks the step (finer granularity, fewer outliers) helps.
- Block-scaled formats (MXFP4, NVFP4) and per-group INT4 confine outliers to small blocks.
- Weight-only quantization speeds memory-bound decode; W8A8/FP8/FP4 compute speeds compute-bound work too.
- GPTQ compensates rounding errors using second-order (Hessian) information; AWQ scales salient channels found from activations; SmoothQuant rescales $X$ and $W$ by an exactly equivalent diagonal transform to balance their ranges.
- Always evaluate on your own tasks, with long generations, against a BF16 baseline on the same harness.

## Check your understanding

??? question "Symmetric INT8, max|x| = 2.54. What is the step size, and which integer represents x = 1.0?"
    $s = 2.54/127 = 0.02$; $q = \operatorname{round}(1.0/0.02) = 50$.

??? question "Why does a W4A16 model not speed up prefill?"
    Prefill is compute-bound, and W4A16 still performs the matrix multiplies in 16-bit after dequantizing the weights; fewer bytes read do not help a compute-bound kernel.

??? question "In SmoothQuant with α = 0.5, an input channel has max|X| = 100 and the matching weight row has max|W| = 0.01. What is s, and what are the new maxima?"
    $s = 100^{0.5}/0.01^{0.5} = 10/0.1 = 100$. New max|X̂| = 100/100 = 1, new max|Ŵ| = 0.01 × 100 = 1. Both equal $\sqrt{100 \times 0.01} = 1$.

??? question "Why does AWQ scaling a salient weight by s > 1 reduce its effective error by about 1/s?"
    The rounding step is set by the group's maximum, which usually does not change, so the absolute rounding error of $s\cdot w$ is about the same as that of $w$; dividing the activation by $s$ (equivalently, the dequantized product) shrinks that error's contribution to the output by $1/s$.

??? question "How many bits per element does an INT4 scheme with group size 64 and an FP16 scale plus an INT4 zero-point use?"
    $4 + (16 + 4)/64 = 4.3125$ bits.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al., 2022 | Discovered outlier features in LLM activations; mixed-precision decomposition. |
| :material-star: [GPTQ: Accurate Post-Training Quantization for GPTs](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022 | One-shot, Hessian-based weight-only quantization to 3–4 bits. |
| :material-star: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024 | Protect salient channels via activation-aware scaling; robust 4-bit weights, widely deployed. |
| :material-star: [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., 2022 | Equivalent rescaling migrates activation outliers into weights → W8A8. |
| :material-book-open-variant: [QServe: W4A8KV4 Quantization and System Co-design](https://arxiv.org/abs/2405.04532) — Lin et al., 2024 | 4-bit weights, 8-bit activations, 4-bit KV co-designed with GPU kernels for real throughput. |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | Keys per-channel, values per-token → 2-bit KV cache. |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | Defines E4M3/E5M2, now native on Hopper/Blackwell; FP8 serving is the 2024–2025 default for large models. |
| :material-new-box: [Microscaling Data Formats for Deep Learning (MX)](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023 | Block-scaled MXFP8/MXFP6/MXFP4 (OCP standard); the basis for FP4 serving on Blackwell. |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | Fine-grained (block-wise) FP8 recipe used at frontier scale. |
