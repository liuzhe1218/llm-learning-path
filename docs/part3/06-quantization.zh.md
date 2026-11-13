# 3.6 量化（Quantization）

!!! abstract "本章你将学到"
    - 读懂 FP32、BF16、FP16、FP8（E4M3/E5M2）、INT8、INT4、FP4、MXFP4 和 NVFP4 的比特布局，并知道它们的取值范围。
    - 手算并用代码实现对称与非对称量化，并选择合适的**粒度（granularity）**（per-tensor、per-channel、per-group、per-block）。
    - 说清**仅权重（weight-only）**量化与**权重 + 激活（weight + activation）**量化的区别，以及它们分别加速推理的哪个阶段。
    - 解释**离群值（outlier）问题**以及三种经典解法：GPTQ（二阶误差补偿）、AWQ（激活感知缩放）和 SmoothQuant（推导其等价性）。
    - 正确地评估量化后的模型，并为一次部署选择合适的方案。

## 为什么重要 {#why-this-matters-the-problem}

回顾 3.1 章：batch 为 1 的解码耗时 ≈ 读取的字节数 ÷ 带宽。一个 70B 参数的模型在 BF16 下约 140 GB，放不进一张 80 GB 的 GPU，而且每生成一个 token 都要把这些权重完整地读一遍。把权重存成 4 bit，模型约 35 GB 再加一点额外开销：它能放进一张 GPU，还给 KV cache 留出空间，每步解码读取的字节数也少了约 4 倍。如果激活也一起量化，矩阵乘本身就可以跑在低精度的 tensor core 上，其 FLOP/s 是 BF16 的 2 倍（FP8）甚至更高。

代价是精度。把数十亿个数四舍五入到寥寥几个取值上会引入误差，而大语言模型有一种特殊的统计结构——少数几个极大的**离群值（outlier）**——会让朴素的取整失效。本章先从比特层面把量化搭起来，再讲让 8 bit 几乎无损、让 4 bit 变得实用的各种技术。

## 直觉 {#intuition}

量化就是选一把**尺子**。把连续的实数轴换成一组刻度，每个值都被吸附到最近的刻度上。两个选择决定了这把尺子：

- **有多少刻度**（位宽）：8 bit → 256 个刻度，4 bit → 16 个刻度。
- **刻度在哪里**（scale，对浮点数来说还有指数的间隔）：尺子必须覆盖你需要表示的最大值，所以一个巨大的值会把尺子拉长，让小数值只能落在很粗的刻度上。

最后这一点，一句话就是离群值问题：在一组大小约为 1 的数里混进一个 100，会让 4 bit 下的刻度间隔变成约 100/7，于是每个"正常"值都被取整成 0。本章大多数巧妙的方法，本质上都是在想办法**不让离群值替所有其他数决定 scale**。

## 工作原理 {#how-it-works}

### 数值格式 {#number-formats}

浮点数有一个**符号**位、若干**指数**位（决定 2 的幂次，也就是取值范围）和若干**尾数**位（决定每个 2 的幂区间内的精度）。整数格式的取值是均匀分布的，需要另外配一个 scale 因子。

| 格式 | 位数（符号/指数/尾数） | 近似最大值 | 说明 |
|---|---|---|---|
| FP32 | 1 / 8 / 23 | $3.4\times10^{38}$ | 训练时的 master weights |
| BF16 | 1 / 8 / 7 | $3.4\times10^{38}$ | 与 FP32 相同的范围，约 3 位有效数字；权重/激活的默认格式 |
| FP16 | 1 / 5 / 10 | 65,504 | 精度更高，但范围比 BF16 小得多 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | 精度更高；用于权重和激活 |
| FP8 E5M2 | 1 / 5 / 2 | 57,344 | 范围更大；主要用于训练中的梯度 |
| INT8 | 8 位整数 | −128 … 127 | 需要 scale |
| INT4 | 4 位整数 | −8 … 7 | 需要 scale |
| FP4 E2M1 | 1 / 2 / 1 | 6 | 可表示的幅值：0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| MXFP4 | FP4 E2M1 元素，每 32 个一块，共享一个 8 位 2 的幂 scale（E8M0） | — | OCP Microscaling 标准；≈ 4.25 bit/元素 |
| NVFP4 | FP4 E2M1 元素，每 16 个一块，共享一个 FP8 E4M3 scale，外加每个 tensor 一个 FP32 scale | — | NVIDIA Blackwell 格式；≈ 4.5 bit/元素 |

```text
BF16      s eeeeeeee mmmmmmm            FP8 E4M3  s eeee mmm
FP16      s eeeee mmmmmmmmmm            FP8 E5M2  s eeeee mm
FP32      s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm      FP4 E2M1  s ee m
MXFP4 块: [ 共享 scale E8M0（8 位） ] [ 32 x FP4 E2M1 ]
NVFP4 块: [ 共享 scale FP8 E4M3 ]      [ 16 x FP4 E2M1 ]   （外加每个 tensor 一个 FP32 scale）
```

低位宽浮点数的间隔是不均匀的（靠近 0 处密，数值大处稀疏），这比均匀的整数更贴合权重那种钟形分布。**分块缩放（block-scaled）**格式（MX、NVFP4）给每个小块各配一个 scale，所以一个离群值只会拉长它自己所在的那 16–32 个邻居的尺子。

### 对称与非对称量化 {#symmetric-and-asymmetric-quantization}

把一个实数 tensor $x$ 量化成 $b$ 位整数：

**对称（symmetric）**（0 映射到 0）：令 $q_{\max} = 2^{b-1}-1$，

$$
s = \frac{\max |x|}{q_{\max}}, \qquad q = \operatorname{clamp}\!\Big(\operatorname{round}\big(\tfrac{x}{s}\big), -q_{\max}, q_{\max}\Big), \qquad \hat x = s\,q .
$$

用大白话说：选一个步长 $s$，让绝对值最大的数正好落在最后一个刻度上；除以 $s$、取整，存下整数 $q$；反量化时再乘回来。

**非对称（asymmetric）**（带零点 zero-point）：对不以 0 为中心的数据，使用完整的无符号范围 $[0, 2^b - 1]$，

$$
s = \frac{\max x - \min x}{2^b - 1}, \qquad z = \operatorname{round}\!\Big(\frac{-\min x}{s}\Big), \qquad q = \operatorname{clamp}\!\Big(\operatorname{round}\big(\tfrac{x}{s}\big) + z,\ 0,\ 2^b - 1\Big), \qquad \hat x = s\,(q - z).
$$

用大白话说：整数 $z$（"零点"）代表实数 0；刻度恰好铺满 $[\min x, \max x]$，而不是 $[-\max|x|, \max|x|]$，所以数据用不到的那一侧不会浪费刻度。

每个元素的取整误差至多为 $s/2$（不发生截断时），所以任何能**缩小 $s$** 的做法——更小的取值范围、更细的粒度——都能减小误差。

### 粒度 {#granularity}

哪些元素共用同一个 scale？

- **Per-tensor：** 整个矩阵共用一个 scale。开销最低，精度最差。
- **Per-channel：** 权重矩阵每个输出行一个 scale（对激活来说则是每个 token 一个）。
- **Per-group：** 沿输入维度，每 $G$ 个连续的权重一个 scale（INT4 仅权重方法中常用 $G = 128$）。额外开销：每 128 个权重配一个 16 位 scale，相当于每个权重多 $16/128 = 0.125$ bit。
- **Per-block：** 每个小块一个 scale（MXFP4 是 32，NVFP4 是 16），或者二维的块，比如 DeepSeek-V3 的 FP8 方案中使用的 $128\times128$ 权重块。

粒度越细，越能隔离离群值、降低误差，代价是要存更多 scale，kernel 也更复杂。

### 仅权重 vs 权重 + 激活 {#weight-only-vs-weight-activation}

- **仅权重**（例如 W4A16：4 bit 权重，16 bit 激活）。权重以 4 bit 存储，并在矩阵乘 kernel **内部反量化成 BF16**。读取的字节数下降约 4 倍，所以**访存受限的解码会变快**；但算术运算仍然在 BF16 下进行，所以**计算受限的预填充（prefill）不会变快**（反量化甚至可能让它略微变慢）。最适合低 batch、重视延迟的服务，以及用更少的 GPU 装下大模型。
- **权重 + 激活**（INT8 或 FP8 的 W8A8，以及 W4A8、W4A4/FP4）。两个操作数都是低精度，矩阵乘跑在低精度 tensor core 上（H100 SXM 上稠密 FP8 ≈ 1,979 TFLOP/s，而 BF16 ≈ 989）。这也能加速**计算受限**的工作——预填充和大 batch 解码——但激活更难量化，因为它们是即时算出来的，而且含有离群值。

### 离群值问题 {#the-outlier-problem}

LLM.int8() 发现，超过某个模型规模（他们的研究中约为 6.7B 参数）之后，transformer 的激活会出现**系统性的离群特征**：少数几个隐藏维度的值比其余维度大几个数量级，并且在各个 token 和各层之间保持一致。对这样的激活做 per-tensor（或 per-token）INT8 量化，会让这些维度来决定 scale，把其他所有维度的精度都抹掉。他们的解法是混合精度：离群维度保留 FP16，其余部分量化。权重的分布要规整一些，但有些权重通道比其他通道**重要**得多，因为它们要去乘那些很大的激活。

### GPTQ：补偿你刚刚造成的误差 {#gptq-compensate-the-error-you-just-made}

GPTQ 逐层量化每个线性层，目标是在校准数据 $X$（几百条序列）上，让该层**输出**的误差最小：

$$
\min_{\hat W} \big\| W X - \hat W X \big\|_2^2 .
$$

把每个权重独立地量化到最近的格点，忽略了一个事实：不同权重上的误差可以互相抵消。GPTQ 逐列量化 $W$ 的输入列，在量化完第 $j$ 列（产生误差 $e_j = w_j - \operatorname{quant}(w_j)$）之后，**调整所有尚未量化的列**来补偿：

$$
W_{:,\,F} \;\leftarrow\; W_{:,\,F} - \frac{e_j}{[H^{-1}]_{jj}} \, [H^{-1}]_{j,\,F}, \qquad H = 2 X X^\top ,
$$

其中 $F$ 是剩余列的集合，$H$ 是该层平方误差对其权重的 Hessian。用大白话说：在数据中和特征 $j$ 相关的那些输入特征，可以吸收它的取整误差，而 $H^{-1}$ 精确地告诉我们每一列该移动多少，才能让该层的输出变化最小。这种"二阶"（考虑曲率的）误差补偿源自剪枝领域的 Optimal Brain Surgeon 一系工作；GPTQ 的贡献在于把它做得足够快（各行共用列顺序、延迟的批量更新、Cholesky 分解形式），从而能在几个 GPU 小时内把 175B 参数的模型量化到 3–4 bit。

### AWQ：通过缩放保护显著通道 {#awq-protect-the-salient-channels-by-scaling}

AWQ 观察到，一小部分权重的**输入通道**比其余通道重要得多，而且可以从**激活**的幅值（也就是去乘大激活的那些通道）找出它们。AWQ 没有把它们保留成高精度（混合格式对硬件不友好），而是对它们做缩放。对于一个缩放系数为 $s > 1$ 的输入通道：

$$
y = \sum_j w_j x_j = \sum_j (s_j w_j)\Big(\frac{x_j}{s_j}\Big) \;\;\to\;\; \sum_j \operatorname{quant}(s_j w_j)\,\frac{x_j}{s_j}.
$$

把某个权重放大 $s$ 倍，通常不会改变该组的最大值，所以步长以及 $s_j w_j$ 的绝对取整误差大致保持不变；之后再除以 $s_j$，该通道的误差就缩小约 $1/s_j$。AWQ 选择 $s = s_X^{\,\alpha}$（$s_X$ 是每个通道的平均激活幅值），并对每层在 $\alpha \in [0,1]$ 上做网格搜索，使校准数据上的输出误差最小。$1/s$ 因子会被合并进前一个操作（例如归一化层的权重），所以推理开销不变。整个过程不涉及反向传播或权重重建，因此 AWQ 速度快，也不容易对校准集过拟合。

### SmoothQuant：把难度从激活转移到权重 {#smoothquant-move-the-difficulty-from-activations-to-weights}

对 W8A8 来说，带有离群通道的激活很难量化，而权重很容易。SmoothQuant 把"难度"在乘法的两侧之间转移。对 $Y = XW$，其中 $X \in \mathbb{R}^{T \times C_{\text{in}}}$，$W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$，以及任意正向量 $s \in \mathbb{R}^{C_{\text{in}}}$，

$$
Y = XW = \big(X \operatorname{diag}(s)^{-1}\big)\big(\operatorname{diag}(s)\, W\big) = \hat X \hat W ,
$$

因为 $\operatorname{diag}(s)^{-1}\operatorname{diag}(s) = I$。这在实数运算下是**完全**等价的：$X$ 的第 $j$ 个输入通道除以 $s_j$，$W$ 的第 $j$ 行乘以 $s_j$。SmoothQuant 选取

$$
s_j = \frac{\max_t |X_{tj}|^{\alpha}}{\max_k |W_{jk}|^{1-\alpha}} .
$$

于是新的通道最大值为 $\max|\hat X_{:,j}| = \max|X_{:,j}|^{1-\alpha} \cdot \max|W_{j,:}|^{1-\alpha}$ 以及 $\max|\hat W_{j,:}| = \max|X_{:,j}|^{\alpha}\cdot\max|W_{j,:}|^{\alpha}$。当 $\alpha = 0.5$ 时，两者都等于 $\sqrt{\max|X_{:,j}|\cdot\max|W_{j,:}|}$：激活和权重的取值范围被均衡了，谁也不会被离群值主导。用大白话说：离群的激活通道被缩小，对应的权重行被放大来补偿；既然权重本来就容易量化，它们承受得起这点放大。$1/s$ 会离线合并进前一层 LayerNorm/RMSNorm 的权重里，所以没有运行时开销。

### KV cache 量化 {#kv-cache-quantization}

KV cache（见 3.2 章）是运行时写入的激活数据。带有 per-tensor（或 per-head）scale 的 FP8 KV 被广泛使用，对大多数模型来说接近无损。更低的位宽需要与数据特点匹配的更细粒度：KIVI 对 **key 按通道（per-channel）**量化（key 存在离群通道），对 **value 按 token（per-token）**量化，做到了 2 bit；QServe 则把 W4A8 与 4 bit KV cache 以及真正兑现吞吐提升所需的 GPU kernel 一起协同设计。

### 评估量化后的模型 {#evaluating-a-quantized-model}

1. BF16 基线和量化模型使用**相同的评测框架、相同的 prompt、相同的采样设置**。
2. 在留出文本上的**困惑度（perplexity）**是一个快速、灵敏的冒烟测试，但困惑度的微小变化可能掩盖任务上的大幅退化。
3. 与你产品相关的**下游任务**：知识（如 MMLU 类）、数学（如 GSM8K 类）、代码、长上下文检索、指令遵循、多语言。
4. **长生成**：推理模型会产出数千个 token，每个 token 上的小误差会累积；用多个随机种子评估并报告方差。
5. **分布层面的检查**：基线与量化模型的下一 token 分布之间的 KL 散度，或者答案在正确与错误之间"翻转"的比例。
6. 在目标硬件和 batch 大小上测**速度与显存**，因为没有原生 kernel 支持的格式可能比 BF16 还慢。

## 代码 {#code}

对称/非对称量化、三种粒度，以及在带有离群通道的激活上演示 SmoothQuant。PyTorch 在 CPU 上运行；量化通过先量化再反量化来模拟（"伪量化 fake quantization"），在写 kernel 之前研究精度时就是这样做的。

```python
import torch
torch.manual_seed(0)

def quant_sym(x, bits, dim=None):
    """对称量化。dim=None：per-tensor；dim=k：沿第 k 维每个切片一个 scale。"""
    qmax = 2 ** (bits - 1) - 1                            # INT8 为 127，INT4 为 7
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    s = amax.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / s), -qmax, qmax)      # 实际存进内存的整数
    return q * s                                          # 反量化后的值

def quant_asym(x, bits):
    """对整个 tensor 做非对称（零点）量化，映射到 [0, 2^bits - 1]。"""
    qmax = 2 ** bits - 1
    s = (x.max() - x.min()).clamp(min=1e-8) / qmax
    z = torch.round(-x.min() / s)                         # 代表 0.0 的整数
    q = torch.clamp(torch.round(x / s) + z, 0, qmax)
    return (q - z) * s

def quant_group(w, bits, group=128):
    """沿输入维度做 per-group 对称权重量化（GPTQ/AWQ 风格）。"""
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // group, group)
    return quant_sym(wg, bits, dim=-1).reshape(out_f, in_f)

w = torch.tensor([0.12, -0.5, 0.31, 0.9, -0.05, 0.0, 0.44, -0.27])
for name, wq in [("sym INT4 ", quant_sym(w, 4)), ("asym INT4", quant_asym(w, 4))]:
    print(name, [round(v, 3) for v in wq.tolist()], "MSE", f"{((w - wq) ** 2).mean():.5f}")

# 粒度很重要：一个各行尺度差异很大的权重矩阵。
W = torch.randn(256, 1024) * torch.logspace(-2, 0, 256)[:, None]
for name, Wq in [("per-tensor", quant_sym(W, 4)), ("per-channel", quant_sym(W, 4, dim=1)),
                 ("per-group128", quant_group(W, 4))]:
    print(f"INT4 {name:13s} rel. error {((W - Wq).norm() / W.norm()):.3f}")

# SmoothQuant：带有离群通道的激活，对比平滑前后的 W8A8。
X = torch.randn(64, 1024); X[:, :8] *= 50                 # 8 个离群的输入通道
W = torch.randn(1024, 1024) / 32
Y = X @ W
def w8a8(X, W):                                           # 激活 per-token，权重 per-输出通道
    return quant_sym(X, 8, dim=1) @ quant_sym(W, 8, dim=0)
alpha = 0.5
s = X.abs().amax(0) ** alpha / W.abs().amax(1) ** (1 - alpha)     # 每个输入通道一个 scale
Xs, Ws = X / s, W * s[:, None]                            # (X diag(s)^-1)(diag(s) W) == XW
print("smoothing is exact:", torch.allclose(Xs @ Ws, Y, rtol=1e-3, atol=1e-2))
for name, Yq in [("W8A8 plain   ", w8a8(X, W)), ("W8A8 smoothed", w8a8(Xs, Ws))]:
    print(name, f"rel. error {((Y - Yq).norm() / Y.norm()):.4f}")
```

在这份玩具数据上，per-group INT4 的误差约为 per-tensor INT4 的三分之一，而平滑处理使 W8A8 的输出误差降低约 4 倍。

## 实例演算 {#worked-example}

**手算：把 8 个权重量化到 INT4。** $w = [0.12, -0.50, 0.31, 0.90, -0.05, 0.00, 0.44, -0.27]$。

*对称。* $q_{\max} = 7$，$s = 0.90 / 7 = 0.1286$。
$w/s = [0.93, -3.89, 2.41, 7.00, -0.39, 0, 3.42, -2.10]$ → $q = [1, -4, 2, 7, 0, 0, 3, -2]$ → $\hat w = [0.129, -0.514, 0.257, 0.900, 0, 0, 0.386, -0.257]$。均方误差约为 $1.09\times10^{-3}$。注意取值范围 $[-0.9, 0.9]$ 被完整覆盖，但数据在负侧只到 $-0.5$：15 个量化级中大约四分之一被浪费了。

*非对称。* $s = (0.90 - (-0.50))/15 = 0.0933$，$z = \operatorname{round}(0.50/0.0933) = \operatorname{round}(5.36) = 5$。
$q = \operatorname{round}(w/s) + 5 = [6, 0, 8, 15, 4, 5, 10, 2]$ → $\hat w = s(q - 5) = [0.093, -0.467, 0.280, 0.933, -0.093, 0, 0.467, -0.280]$。均方误差约为 $0.82\times10^{-3}$，低了约 25%，因为全部 16 个量化级都用上了。

**模型大小（仅权重）。** 一个 70B 参数的模型：

| 方案 | 每个权重的位数（含 scale） | 大小 |
|---|---|---|
| BF16 | 16 | ≈ 140 GB |
| FP8（per-channel scale） | ≈ 8 | ≈ 70 GB |
| INT4，group 128，FP16 scale | 4 + 0.125 ≈ 4.13 | ≈ 36 GB |
| MXFP4（32 一块，8 位 scale） | 4 + 8/32 = 4.25 | ≈ 37 GB |
| NVFP4（16 一块，8 位 scale） | 4 + 8/16 = 4.5 | ≈ 39 GB |

用 4 bit 权重，模型能放进一张 80 GB 的 H100，还剩约 40 GB 给 KV cache；batch 为 1 时的带宽上界，从 140/3.35 ≈ 42 ms 每 token（而且这样也放不下）改善到 ≈ 36/3.35 ≈ 11 ms 每 token。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **FP8（W8A8）是默认方案**，用于在 Hopper 和 Blackwell 上服务大模型：配合 per-channel 或分块 scale 时近乎无损，并且同时加速预填充和解码。DeepSeek-V3 甚至是直接用细粒度分块缩放的 FP8 *训练*的，其发布的权重就是 FP8。
- **4 bit 仅权重**（AWQ、GPTQ，以及 llama.cpp 中的 GGUF k-quants）是在单张 GPU、消费级硬件或 Apple Silicon 上运行模型的标准方式。
- **FP4** 随着 Blackwell 的 FP4 tensor core 而变得实用：OpenAI 的 gpt-oss 模型以 MXFP4 发布其 MoE 权重，NVIDIA 则为 TensorRT-LLM 及其他引擎主推 NVFP4 checkpoint。
- **MoE** 模型尤其适合量化：专家权重占据了大部分参数量和显存，而注意力层和共享层可以保持较高精度。
- **FP8 KV cache** 是一个常规选项；更低位宽的 KV 只会有选择地使用，并且要按模型逐一评估。
- 经验法则：FP8 ≈ 无损；4 bit 仅权重是质量与体积之间的好折中；激进的激活或 KV 量化需要在*你自己的*任务上评估——输出很长的推理模型更敏感。

## 常见误区 {#common-pitfalls-misconceptions}

- **"4 bit 让一切快 4 倍。"** 仅权重量化加速的是访存受限的解码；除非激活也被量化，否则预填充和大 batch 解码仍然是 BF16 下的计算受限。
- **只用困惑度判断质量。** 要用任务基准，尤其是长生成和长上下文任务。
- **校准数据不匹配。** GPTQ/AWQ/SmoothQuant 都要用校准样本；用通用网页文本校准，却拿去服务代码或中文，可能会有损失。要用接近你线上流量的数据。
- **不被支持的格式。** 如果 checkpoint 的格式在你的 GPU 或引擎上没有快速 kernel，就会被即时反量化，可能比 BF16 还慢（例如 FP8 计算需要 Hopper/Ada 或更新的架构；FP4 tensor core 需要 Blackwell）。
- **对所有部分一刀切地量化。** embedding、LM head、归一化层、路由器（gate）权重和注意力层常常保持较高精度，因为它们体积小或者很敏感。
- **估算大小时忘了 scale。** group/block 的 scale 会让每个权重多出 0.125–0.5 bit。

## 本章要点 {#key-takeaways}

- 量化器就是一个 scale（以及可选的零点）加上取整；误差至多是半个步长，所以任何缩小步长的办法（更细的粒度、更少的离群值）都有帮助。
- 分块缩放格式（MXFP4、NVFP4）和 per-group INT4 把离群值限制在小块之内。
- 仅权重量化加速访存受限的解码；W8A8/FP8/FP4 的计算加速也能覆盖计算受限的工作。
- GPTQ 用二阶（Hessian）信息补偿取整误差；AWQ 缩放由激活找出的显著通道；SmoothQuant 通过一个严格等价的对角变换对 $X$ 和 $W$ 重新缩放，以均衡二者的取值范围。
- 始终在你自己的任务上评估，包含长生成，并和 BF16 基线在同一个评测框架下对比。

## 自测 {#check-your-understanding}

??? question "对称 INT8，max|x| = 2.54。步长是多少？哪个整数代表 x = 1.0？"
    $s = 2.54/127 = 0.02$；$q = \operatorname{round}(1.0/0.02) = 50$。

??? question "为什么 W4A16 模型不能加速预填充？"
    预填充是计算受限的，而 W4A16 在反量化权重之后仍然用 16 位做矩阵乘；读取的字节数变少，对计算受限的 kernel 没有帮助。

??? question "SmoothQuant 中 α = 0.5，某个输入通道的 max|X| = 100，对应权重行的 max|W| = 0.01。s 是多少？新的最大值是多少？"
    $s = 100^{0.5}/0.01^{0.5} = 10/0.1 = 100$。新的 max|X̂| = 100/100 = 1，新的 max|Ŵ| = 0.01 × 100 = 1。两者都等于 $\sqrt{100 \times 0.01} = 1$。

??? question "为什么 AWQ 把显著权重放大 s > 1 倍，就能把它的有效误差降低约 1/s？"
    取整步长由该组的最大值决定，而这个最大值通常不变，所以 $s\cdot w$ 的绝对取整误差和 $w$ 的差不多；再把激活除以 $s$（等价于对反量化后的乘积做同样处理），这个误差对输出的贡献就缩小了 $1/s$。

??? question "一个 INT4 方案，group 大小为 64，带一个 FP16 scale 和一个 INT4 零点，每个元素用多少 bit？"
    $4 + (16 + 4)/64 = 4.3125$ bit。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al., 2022 | 发现了 LLM 激活中的离群特征；混合精度分解。 |
| :material-star: [GPTQ: Accurate Post-Training Quantization for GPTs](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022 | 一次性（one-shot）、基于 Hessian 的仅权重 3–4 bit 量化。 |
| :material-star: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024 | 通过激活感知缩放保护显著通道；稳健的 4 bit 权重，被广泛部署。 |
| :material-star: [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., 2022 | 等价的重新缩放把激活离群值迁移到权重里 → W8A8。 |
| :material-book-open-variant: [QServe: W4A8KV4 Quantization and System Co-design](https://arxiv.org/abs/2405.04532) — Lin et al., 2024 | 4 bit 权重、8 bit 激活、4 bit KV，与 GPU kernel 协同设计，换来真实的吞吐提升。 |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | key 按通道、value 按 token → 2 bit KV cache。 |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | 定义了 E4M3/E5M2，现已在 Hopper/Blackwell 上原生支持；FP8 服务是 2024–2025 年大模型的默认方案。 |
| :material-new-box: [Microscaling Data Formats for Deep Learning (MX)](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023 | 分块缩放的 MXFP8/MXFP6/MXFP4（OCP 标准）；Blackwell 上 FP4 服务的基础。 |
| :material-book-open-variant: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 在前沿规模上使用的细粒度（分块）FP8 方案。 |
