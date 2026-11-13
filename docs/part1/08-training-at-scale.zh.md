# 1.8 大规模训练

!!! abstract "本章你将学到"
    - 用混合精度 Adam 训练时，GPU 显存究竟花在哪里（每个参数约 16 字节），以及激活值和**激活检查点（activation checkpointing）**。
    - 并行工具箱：**数据并行**、**ZeRO 1–3 / FSDP**、**张量并行**（MLP 的列/行切分，附代码验证）、**流水线并行**及其气泡、**序列/上下文并行**、**专家并行**，以及它们如何组合成 3D/4D 并行。
    - 数值格式（FP32、BF16、FP16、FP8）以及混合精度的工作方式。
    - 优化器栈：**AdamW**、学习率预热加余弦衰减与 **WSD** 调度，以及直观层面的 **Muon**。
    - 训练运行为什么会“炸”（损失尖峰、logit 增长），以及标准的修复方法：**QK-norm**、**z-loss**、梯度裁剪。

## 为什么重要 {#why-this-matters-the-problem}

[1.7](07-scaling-laws-and-data.md) 中的算力预算（$10^{24}$–$10^{25}$ FLOPs）远远超出单块 GPU 的能力：一块 H100 的 BF16 峰值约为 $10^{15}$ FLOP/s 量级，所以 $3.8 \times 10^{25}$ FLOPs 要让它跑上一千多年。Llama 3 405B 用了多达 16,384 块 H100 来训练。要让成千上万块 GPU 像一台机器那样工作，就必须把模型、数据和优化器状态拆分到各个设备上，同时避免通信成为主导，还要让持续数月的训练在数值上保持稳定。一次发散的损失尖峰，就可能浪费数周的算力。本章讲的就是预训练的系统与优化这一面。

## 直觉 {#intuition}

训练一个大模型有三个资源问题：

1. **放不下。** 一个 70B 的模型，仅权重、梯度和优化器状态就需要超过 1 TB。解决办法：把这些张量*分片*到多块 GPU 上（ZeRO/FSDP），或者把层本身拆开（张量并行和流水线并行）。
2. **太慢。** 解决办法：让不同的 GPU 同时处理不同的数据（数据并行），并让每块 GPU 都保持忙碌。
3. **很脆弱。** 为了速度，需要低精度运算和很大的学习率，但它们容易引起溢出和发散。解决办法：精心选择数值格式、归一化、调度和安全阀。

每种并行策略切分的是计算的不同维度，付出的通信代价也各不相同：

| 策略 | 切分的对象 | 通信 | 典型放置位置 |
|---|---|---|---|
| 数据并行（DP） | batch | 每步对梯度做 all-reduce | 跨节点 |
| ZeRO / FSDP | 优化器状态、梯度、权重（跨 DP rank） | gather 权重，reduce-scatter 梯度 | 跨节点 |
| 张量并行（TP） | 每个矩阵乘法 | 每层内部做 all-reduce | 节点内（高速 NVLink） |
| 流水线并行（PP） | 层的堆叠 | 阶段之间传激活值 | 跨节点 |
| 序列 / 上下文并行（SP/CP） | 序列维度 | 激活值 / K,V 块 | 节点内或跨节点 |
| 专家并行（EP） | MoE 专家 | token 的 all-to-all | 节点内或跨节点 |

## 工作原理 {#how-it-works}

### 显存核算 {#memory-accounting}

设 $\Psi$ 为参数个数。使用标准的**混合精度**训练加 Adam 时，每个参数需要：

| 项目 | 格式 | 每参数字节数 |
|---|---|---|
| 前向/反向用的权重 | BF16 | 2 |
| 梯度 | BF16 | 2 |
| 权重的主副本（master copy） | FP32 | 4 |
| Adam 一阶矩 $m$ | FP32 | 4 |
| Adam 二阶矩 $v$ | FP32 | 4 |
| **合计** | | **16** |

所以“模型状态”占 $16\Psi$ 字节：7B 模型是 112 GB，70B 是 1.12 TB。一块 80 GB 的 GPU 连一个 7B 模型的训练状态都装不下。

**激活值**是额外的开销：前向传播中为反向传播保存的每一个中间张量。对于隐藏维度为 $h$、序列长度为 $s$、micro-batch 为 $b$ 的 GPT 风格层，Korthikanti et al.（2022）估计，在 16 位精度下每层约 $34\,sbh$ 字节，如果注意力没有融合，还要加上用于注意力分数和 softmax 的 $5\,a s^2 b$ 字节（$a$ = 头数）。FlashAttention（见 [1.6](06-long-context-and-efficient-attention.md)）去掉了 $s^2$ 这一项。

**激活检查点**（Chen et al., 2016）只保存一部分激活值，通常是每层的输入，其余的在反向传播时重新计算。完整检查点每层大约存储 $2sbh$ 字节，代价是多一次前向传播，使每个 token 的计算量从约 $6N$ 升到约 $8N$ FLOPs。**选择性**检查点只重新计算那些重算便宜、但占显存多的部分（如注意力 softmax），用多出几个百分点的计算量，换取大部分的显存节省。

### 数据并行与 ZeRO / FSDP {#data-parallelism-and-zero-fsdp}

在**数据并行（DP）**中，$N_d$ 块 GPU 中的每一块都持有模型的完整副本，处理全局 batch 的不同切片，然后通过 **all-reduce** 对梯度求平均，使每个副本迈出相同的一步。它简单、通信高效，但每块 GPU 都要存完整的 $16\Psi$ 字节：显存无法随规模扩展。

**ZeRO**（Rajbhandari et al., 2019）消除了这种冗余：既然每个 DP rank 持有的状态完全相同，那么每个 rank 只需要*拥有*其中 $1/N_d$ 的分片。

- **阶段 1** 对优化器状态（$12\Psi$ 字节：主权重、$m$、$v$）分片。每个 rank 只更新自己那一片参数，然后把更新后的权重 gather 起来。
- **阶段 2** 同时对梯度分片：用 **reduce-scatter** 代替 all-reduce，让每个 rank 只拿到自己需要的那一片梯度。
- **阶段 3** 同时对 BF16 权重分片。在每一层前向（和反向）之前，它的完整权重会从所有 rank **all-gather** 过来，用完即释放。

每块 GPU 上模型状态的显存：

$$
\text{DP: } 16\Psi, \quad \text{ZeRO-1: } 4\Psi + \frac{12\Psi}{N_d}, \quad \text{ZeRO-2: } 2\Psi + \frac{14\Psi}{N_d}, \quad \text{ZeRO-3: } \frac{16\Psi}{N_d}.
$$

阶段 1 和 2 的通信量与普通 DP 相同（一次 all-reduce 就等于一次 reduce-scatter 加一次 all-gather）；阶段 3 增加了权重的 gather，通信量约为 DP 的 1.5 倍。PyTorch 的 **FSDP**（Fully Sharded Data Parallel）实现的是 ZeRO-3 风格的分片，并通过预取下一层的权重，让通信与计算重叠。

### 张量并行 {#tensor-parallelism}

**张量并行（TP）**（Megatron-LM, Shoeybi et al., 2019）把单个权重矩阵拆分到多块 GPU 上。考虑 MLP $Y = \mathrm{GeLU}(XA)\,B$，其中 $X: [\text{tokens}, d]$，$A: [d, d_{ff}]$，$B: [d_{ff}, d]$，共 $t$ 块 GPU。

1. 把 $A$ 按**列**切分：$A = [A_1, \dots, A_t]$，每块为 $[d, d_{ff}/t]$。GPU $i$ 计算 $H_i = \mathrm{GeLU}(X A_i)$。因为 GeLU 是逐元素的，对每个列块分别施加它，与对 $XA$ 整体施加完全相同：无需通信。
2. 把 $B$ 按**行**切分：$B = [B_1; \dots; B_t]$，每块为 $[d_{ff}/t, d]$。GPU $i$ 计算 $Z_i = H_i B_i$，得到一个完整大小为 $[\text{tokens}, d]$ 的部分结果。
3. 由于 $HB = \sum_i H_i B_i$，一次 **all-reduce**（求和）就能在每块 GPU 上得到 $Y$。

所以整个 MLP 在前向传播中只需一次 all-reduce，反向传播中也只需一次。注意力也是沿头维度这样切分的：每块 GPU 计算一部分头（对 $W_Q, W_K, W_V$ 做列切分），输出投影 $W_O$ 则做行切分。TP 在每一层内部都要通信，所以要把它限制在节点内（那里的 GPU 通过 NVLink 相连），通常 $t \le 8$。

**序列并行**（Korthikanti et al., 2022）是对 TP 的补充：在普通 TP 中，TP 区域之间的操作（归一化、dropout、残差相加）是各卡复制的；序列并行则沿序列维度把它们切开，把每次 all-reduce 变成一次 reduce-scatter 加一次 all-gather，并把这些部分的激活值显存削减为 $1/t$。

### 流水线并行 {#pipeline-parallelism}

**流水线并行（PP）**把连续的若干层（“阶段”）放在不同的 GPU 上：GPU 1 放第 1–8 层，GPU 2 放第 9–16 层，依此类推。只需要传递阶段边界上的激活值，开销很小，所以 PP 可以跨节点使用。问题在于空闲：阶段 2 要等阶段 1 的输出。解决办法是把 batch 切成 $m$ 个 **micro-batch**，让它们流水线式地流过去。有 $p$ 个阶段时，每一步的开头和结尾总有一些阶段处于空闲；这种**流水线气泡（pipeline bubble）**，对 GPipe 和 1F1B（“一次前向，一次反向”）调度，会浪费的时间比例为

$$
\text{bubble} = \frac{p - 1}{m + p - 1}.
$$

micro-batch 越多，气泡越小；**交错式（interleaved）**调度（每块 GPU 持有多个不相邻的阶段）能把它缩得更小，代价是通信更多。DeepSeek-V3 的 **DualPipe** 让不同 micro-batch 的前向与反向计算同 MoE 通信重叠，几乎隐藏了气泡和 all-to-all 的开销。

### 上下文并行与专家并行 {#context-parallelism-and-expert-parallelism}

**上下文并行（CP）**把单个长序列拆分到多块 GPU 上，因此每块 GPU 的激活值显存被 CP 度数除。难点在注意力：每个 token 的 query 需要整个（之前的）序列的 key 和 value。**Ring attention** 让 K/V 块沿着由 GPU 组成的环传递，同时每块 GPU 逐块地为自己本地的 query 计算注意力，用的是 [1.6](06-long-context-and-efficient-attention.md) 中的在线 softmax。Llama 3 在其长上下文训练阶段使用了 CP。

**专家并行（EP）**把不同的 MoE 专家放在不同的 GPU 上，并对 token 做 all-to-all 的分发与合并（见 [1.5](05-mixture-of-experts.md)）。

### 组合成 3D/4D 并行 {#composing-3d4d-parallelism}

真实的训练运行会把这些策略嵌套起来，把通信最密集的放在最快的链路上。GPU 总数是各并行度的乘积：$N_{\text{GPU}} = t_{\text{TP}} \times c_{\text{CP}} \times p_{\text{PP}} \times d_{\text{DP}}$。Llama 3 405B 在主要预训练配置中，使用 TP = 8（节点内）、PP = 16，以及 FSDP 风格的 DP = 128，即 $8 \times 16 \times 128 = 16{,}384$ 块 GPU，并在长序列时再加上 CP。DeepSeek-V3 完全没有用 TP：在 2,048 块 H800 GPU 上，使用 16 路 PP（DualPipe）、跨 8 个节点的 64 路 EP，以及 ZeRO-1 DP。

一个粗略的决策流程：只要模型放得下，就单独使用 FSDP/ZeRO DP；当单层权重或激活值太大时，在节点内加 TP；当模型仍然放不下，或者 DP 的规模受 batch 大小限制时，加 PP；对长序列加 CP；对 MoE 加 EP。

### 混合精度与数值格式 {#mixed-precision-and-number-formats}

一个浮点数由符号位、指数（决定范围）和尾数（决定精度）组成：

| 格式 | 指数位 | 尾数位 | 备注 |
|---|---|---|---|
| FP32 | 8 | 23 | 主权重、优化器状态、归约 |
| FP16 | 5 | 10 | 范围小；需要**损失缩放（loss scaling）**来避免梯度下溢 |
| BF16 | 8 | 7 | FP32 的范围、较低的精度；LLM 训练的默认格式 |
| FP8 E4M3 | 4 | 3 | 精度更高、范围更小；用于前向张量 |
| FP8 E5M2 | 5 | 2 | 范围更大；有时用于梯度 |

**混合精度**（Micikevicius et al., 2017）在低精度下运行矩阵乘法（在 tensor core 上很快），同时保留一份 FP32 的权重主副本，使微小的更新不会因舍入而丢失，并让敏感的运算（softmax、归一化、损失、归约）用 FP32 进行。FP16 需要损失缩放（把损失乘以一个很大的系数，使很小的梯度不会下溢，之后再除回去）；BF16 更宽的指数位使这一步不再必要，这就是它成为标准的原因。

**FP8** 把 tensor core 的吞吐量再翻了一倍，但范围非常小，所以每个张量只用一个缩放因子是不够的：一个离群值就会把其他所有数据压向零。DeepSeek-V3 用**细粒度缩放**（对每个 $1 \times 128$ 的激活 tile 和每个 $128 \times 128$ 的权重块使用单独的缩放因子），并周期性地把部分和提升到 FP32 累加器中，在大规模上实现了 FP8 训练；据其报告，损失与 BF16 基线的差距在 0.25% 以内。

### 优化器与学习率调度 {#optimizers-and-learning-rate-schedules}

**AdamW**（Loshchilov & Hutter, 2017）是默认选择。设梯度为 $g_t$，学习率为 $\eta$，衰减率为 $\beta_1, \beta_2$，权重衰减为 $\lambda$，以及一个小量 $\epsilon$：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\theta_t = \theta_{t-1} - \eta \Big( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \Big), \qquad \hat m_t = \frac{m_t}{1 - \beta_1^t},\ \hat v_t = \frac{v_t}{1 - \beta_2^t}.
$$

用大白话说：沿着梯度的滑动平均（$m$）的方向前进，再逐元素除以对其典型幅度的滑动估计（$\sqrt{v}$），使每个参数迈出大小相近的步子；另外单独把权重向零收缩（“解耦”的权重衰减，它作用在自适应缩放之外，这正是 AdamW 中的“W”）。LLM 训练常用 $\beta_1 = 0.9$，$\beta_2 = 0.95$，$\lambda = 0.1$，以及范数为 1.0 的梯度裁剪。

**学习率调度：**

- **预热（warmup）：** 在最初的几千步里，把学习率从 0 线性增大。早期 Adam 的 $v$ 估计噪声很大，网络也远离平衡；一上来就用满学习率可能会发散。
- **余弦衰减：** 预热之后，沿着半个余弦曲线降到最小值（通常是峰值的 10%）。调度的长度必须提前确定。
- **WSD（warmup–stable–decay，预热–稳定–衰减）**（由 MiniCPM 推广）：预热后，在训练的大部分时间里保持学习率恒定，然后快速衰减（例如在最后 10–20% 的步数里）。稳定阶段的任何一个检查点都可以被“冷却”成一个完成的模型，所以你可以延长训练或做缩放定律实验而无需重新开始，而且衰减阶段与数据退火（见 [1.7](07-scaling-laws-and-data.md)）天然契合。

**Muon**（Keller Jordan 及合作者，2024）把每个二维权重矩阵当作矩阵，而不是一堆数。它维护梯度的动量 $M$，然后把 $M$ 换成一个近似**正交化**的版本：若 $M = U \Sigma V^\top$（奇异值分解），则更新量约为 $U V^\top$，也就是把所有奇异值都设为 1。直觉：权重矩阵的梯度往往被少数几个方向主导；正交化会放大那些稀有的方向，使更新在所有方向上“均匀地”改变矩阵。计算 SVD 代价很高，所以 Muon 使用几次 **Newton–Schulz** 迭代，这是一个只用矩阵乘法就能把奇异值推向 1 的多项式映射。Moonshot 的 Moonlight 工作表明，借助权重衰减和按矩阵缩放的更新量，Muon 可以扩展到大型 LLM，报告相对 AdamW 约有 2× 的算力效率；Kimi K2 就是用它的一个变体 MuonClip 训练的。Muon 用于隐藏层的权重矩阵；嵌入层、输出层和归一化参数仍然使用 AdamW。

### 训练不稳定及其修复 {#training-instabilities-and-fixes}

大规模训练会出现**损失尖峰（loss spike）**：损失突然跳高，有时恢复，有时发散。常见原因与修复方法：

- **注意力 logit 增长。** query/key 的范数在训练中不断增长，直到 $q \cdot k$ 变得极大，softmax 饱和成 one-hot，梯度变得不稳定。修复：**QK-norm**（见 [1.4](04-modern-block.md)），或者 Kimi K2 的 **QK-clip**（MuonClip 的一部分），当某个头的最大注意力 logit 超过阈值时，对 $W_Q$ 和 $W_K$ 重新缩放。Wortsman et al.（2023）在小模型上用高学习率复现了这种不稳定，并表明 QK-layernorm 可以消除它。
- **输出 logit 发散。** 输出 logit 漂移到很大的量级。修复：**z-loss**（PaLM 中使用）：在损失上加 $10^{-4} \cdot (\log Z)^2$，其中 $Z = \sum_i e^{z_i}$ 是 softmax 的归一化项。这使 $\log Z$ 保持在 0 附近，从而让 logit 处在条件良好的范围内。ST-MoE 把同样的想法用在了路由器的 logit 上。
- **梯度爆炸。** **梯度裁剪**在 $\lVert g \rVert > c$ 时，把整个梯度向量 $g$ 缩放到范数至多为 $c$（例如 1.0）：$g \leftarrow g \cdot \min(1, c / \lVert g \rVert)$。这限制了任何单次更新的大小。
- **坏 batch 与恢复。** PaLM 对持续出现的尖峰的处理办法是：从尖峰前约 100 步的检查点重新开始，并跳过几百个数据 batch。监控梯度范数和各层的统计量有助于及早发现问题。
- **其他：** 降低峰值学习率、延长预热、减小 $\beta_2$、调整 $\epsilon$、让归一化和 softmax 保持 FP32，以及去掉偏置。Kimi K2 报告称，使用 MuonClip 在 15.5T token 上训练，没有出现损失尖峰。

## 代码 {#code}

**一个脚本搞定显存、张量并行和流水线气泡。** 张量并行的 MLP 用一个分片列表来模拟，并与未切分的计算做对照检查。

```python
import torch
import torch.nn.functional as F

def zero_memory_gb(n_params, n_gpus, stage):
    """BF16 参数/梯度加 FP32 Adam（16 B/参数）时，每块 GPU 上权重+梯度+优化器的字节数。"""
    p, g, opt = 2 * n_params, 2 * n_params, 12 * n_params   # bf16 权重、bf16 梯度、fp32 主权重+m+v
    if stage >= 1: opt /= n_gpus
    if stage >= 2: g /= n_gpus
    if stage >= 3: p /= n_gpus
    return (p + g + opt) / 1e9

for s in range(4):
    print(f"7B on 64 GPUs, ZeRO-{s}: {zero_memory_gb(7e9, 64, s):6.1f} GB per GPU")

# MLP  Y = GeLU(X A) B 的张量并行，用一个“GPU”列表来模拟。
torch.manual_seed(0)
d, d_ff, tp = 8, 32, 4
X = torch.randn(5, d)                       # [tokens, d]，在每块 GPU 上都有一份副本
A = torch.randn(d, d_ff)                    # 第一层：按列切分
B = torch.randn(d_ff, d)                    # 第二层：按行切分
A_shards = A.chunk(tp, dim=1)               # GPU i 持有 A[:, 第 i 片]
B_shards = B.chunk(tp, dim=0)               # GPU i 持有 B[第 i 片, :]
partials = [F.gelu(X @ A_i) @ B_i for A_i, B_i in zip(A_shards, B_shards)]  # 无需通信
Y_tp = torch.stack(partials).sum(0)         # 每次 MLP 前向只需一次 all-reduce（求和）
Y_ref = F.gelu(X @ A) @ B
print("tensor-parallel MLP matches:", torch.allclose(Y_tp, Y_ref, atol=1e-5))

def bubble_fraction(p, m):
    """p 个阶段、m 个 micro-batch 的 GPipe/1F1B 流水线的空闲比例。"""
    return (p - 1) / (m + p - 1)

print(f"bubble p=16, m=64: {bubble_fraction(16, 64):.1%};  m=256: {bubble_fraction(16, 256):.1%}")
```

输出：ZeRO 阶段 0–3 每块 GPU 分别为 112.0、29.3、15.5 和 1.8 GB；TP 检查打印 `True`；气泡分别为 19.0% 和 5.5%。在真实系统中，每个分片位于不同的 GPU 上，而 `sum` 是一次 `torch.distributed.all_reduce`。

**学习率调度与 Muon 的 Newton–Schulz 正交化。**

```python
import math, torch

def lr_cosine(step, peak, warmup, total, min_ratio=0.1):
    if step < warmup:
        return peak * step / warmup                                   # 线性预热
    t = (step - warmup) / (total - warmup)                            # 进度，取值在 [0, 1]
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

def lr_wsd(step, peak, warmup, total, decay_frac=0.2):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * step / warmup                                   # 预热
    if step < decay_start:
        return peak                                                   # 稳定
    return peak * (total - step) / (total - decay_start)              # 线性衰减到 0

for s in [0, 1000, 25000, 50000, 85000, 100000]:
    print(f"step {s:6d}: cosine {lr_cosine(s, 3e-4, 2000, 100000):.2e}   WSD {lr_wsd(s, 3e-4, 2000, 100000):.2e}")

def newton_schulz(G, steps=5):
    """Muon 的正交化：把 G 的所有奇异值推向约 1（G -> ~U V^T）。"""
    a, b, c = 3.4445, -4.7750, 2.0315             # 来自 Keller Jordan 的 Muon 的五次多项式系数
    X = G / (G.norm() + 1e-7)                     # 缩放使奇异值 <= 1
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X

torch.manual_seed(0)
G = torch.randn(64, 32) @ torch.diag(torch.logspace(0, 2, 32))   # 条件数很差的“动量”
print("singular values before:", torch.linalg.svdvals(G)[[0, 15, 31]].round(decimals=2).tolist())
print("singular values after :", torch.linalg.svdvals(newton_schulz(G))[[0, 15, 31]].round(decimals=2).tolist())
```

奇异值从大约 (919, 74, 6) 变成了 (1.20, 0.87, 0.68)：150:1 的分布范围，仅用五次矩阵乘法迭代就压缩到了 2:1 以内。Muon 有意用精确的正交性换取速度；经过调优的系数使奇异值落在 1 附近的一个带内，而不是精确地收敛到 1。

## 实例演算 {#worked-example}

**训练一个 70B 稠密模型：放得下吗？**

- 模型状态：$16 \times 70 \times 10^9 = 1.12$ TB。在 80 GB 的 GPU 上用普通 DP：不可能。
- 在 64 块 GPU 上用 ZeRO-3/FSDP：模型状态为 $1.12\,\text{TB} / 64 = 17.5$ GB/GPU。✓
- 一条 4096-token 序列的激活值，用 $34\,sbh$ 估计，$h = 8192$、80 层：不做检查点时为 $34 \times 4096 \times 8192 \times 80 \approx 91$ GB。✗ 太大。
- 使用完整的激活检查点（每层 $2sbh$）：$2 \times 4096 \times 8192 \times 80 \approx 5.4$ GB，再加上重计算时某一层的完整激活值（$\approx 1.1$ GB）。✓ 代价是每个 token 的计算量从 $6N$ 变为约 $8N$ FLOPs。
- 另一种办法是，在每个节点内用 TP = 8，把大部分激活值（和权重）除以 8，从而避免完整的重计算：配合序列并行，$91/8 \approx 11$ GB。这就是大型稠密模型把 TP 与 FSDP 结合使用的原因。

**对于 7B 模型**（$\Psi = 7 \times 10^9$，64 块 GPU）：DP 需要 112 GB/GPU；ZeRO-1 需要 $4\Psi + 12\Psi/64 = 28 + 1.3 = 29.3$ GB；ZeRO-2 需要 $2\Psi + 14\Psi/64 = 14 + 1.5 = 15.5$ GB；ZeRO-3 需要 $16\Psi/64 = 1.75$ GB。

**流水线气泡。** 16 个阶段、64 个 micro-batch，浪费 $15/79 = 19\%$ 的时间；用 256 个 micro-batch 则为 $15/271 = 5.5\%$。但每步用 256 个 micro-batch 需要很大的全局 batch，这是流水线深度受限的原因之一。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **Llama 3 405B**：BF16，主要配置为在 16,384 块 H100 上使用 TP 8 × PP 16 × DP（FSDP）128，长上下文训练时再加上上下文并行；报告还记录了这种规模下频繁的硬件故障以及自动恢复。
- **DeepSeek-V3**：带细粒度缩放的 FP8 混合精度、DualPipe 流水线调度、64 路 EP、ZeRO-1、没有 TP；报告称完整训练共耗费 2.788M 个 H800 GPU 小时。
- **Kimi K2**：MuonClip 优化器，15.5T token，据报告没有损失尖峰。
- **调度**：WSD 风格的调度（或带一段很长的恒定阶段的余弦调度）很常见，因为它们能与退火数据自然结合，并允许继续训练。
- **超参数从小型代理运行迁移而来**（μP，或按模型大小对学习率和 batch 大小做经验性缩放）。
- **FP8 训练**在 Hopper/Blackwell GPU 上越来越常见；4 比特格式（MXFP4、NVFP4）到目前为止主要用于推理，例如 gpt-oss 的 MoE 权重就是以 MXFP4 发布的。

## 常见误区 {#common-pitfalls-misconceptions}

- **用“参数 × 2 字节”估算训练显存。** 那只是推理时的权重。用 Adam 训练需要约 16 字节/参数，外加激活值。
- **跨节点使用 TP。** TP 在每一层内部都要通信；在较慢的节点间链路上，通常会彻底拖垮吞吐。
- **忽略气泡。** 流水线很深而 micro-batch 很少，会浪费集群的很大一部分算力。
- **改变 batch 大小却不重新调学习率。** 最优学习率取决于 batch 大小；更大的 batch 通常允许更大的学习率，但有上限。
- **把损失尖峰当作噪声。** 小的尖峰可能是发散的前兆；要跟踪梯度范数、最大注意力 logit 和输出 logit 的量级。
- **“处处 FP8”。** 即使在 FP8 训练中，嵌入层、输出层、归一化层、注意力 softmax 和优化器状态仍然保持更高的精度。

## 本章要点 {#key-takeaways}

- 混合精度 Adam 每个参数花约 16 字节；激活值是额外的开销，靠检查点来控制。
- ZeRO/FSDP 把优化器状态、梯度和权重在数据并行 rank 之间分片，把每块 GPU 的模型状态从 $16\Psi$ 降到 $16\Psi/N_d$。
- 张量并行把 MLP 的第一个矩阵按列切、第二个按行切，每个 MLP 只需一次 all-reduce；它应该放在节点内部。
- 流水线并行把层切成阶段；其气泡比例为 $(p-1)/(m+p-1)$。上下文并行和专家并行则分别切分序列和专家。
- BF16 是默认格式；FP8 需要细粒度缩放。带预热以及余弦或 WSD 衰减的 AdamW 是标准做法；Muon 通过对更新做正交化来获得更高的效率。
- 不稳定性来自 logit 增长和离群值；QK-norm、z-loss 和梯度裁剪是标准的防御手段。

## 自测 {#check-your-understanding}

??? question "在使用 ZeRO-3 的情况下，仅仅容纳一个 30B 模型的训练状态（忽略激活值），最少需要多少块 80 GB 的 GPU？"
    $16 \times 30 \times 10^9 = 480$ GB；$480 / 80 = 6$ 块 GPU，所以至少 6 块（实际上还需要更多，用于激活值和缓冲区）。

??? question "为什么把 $A$ 按列切、把 $B$ 按行切，就能避免两个矩阵乘法之间的通信？"
    $XA$ 的各列块可以独立计算，逐元素的 GeLU 也可以对每个块分别施加。$B$ 的行块恰好消费这些列块，产生最终输出的部分和，所以只需要通信一次，用来把部分和加起来。

??? question "与 FP16 相比，BF16 放弃了什么？为什么这对训练是一笔划算的交易？"
    精度：只有 7 个尾数位，而不是 10 个。作为交换，它拥有 FP32 的 8 位指数，所以梯度和激活值很少上溢或下溢，这就不再需要损失缩放。精度的损失由 FP32 主权重和 FP32 累加来缓解。

??? question "z-loss 解决什么问题，又是如何解决的？"
    解决输出 logit 漂移到很大量级的问题，这会让 softmax 数值不稳定。惩罚 $(\log Z)^2$ 会使对数归一化项保持在 0 附近，于是 logit 保持在适中的范围内，同时不改变模型偏好哪个 token。

??? question "为什么 WSD 对缩放定律实验很方便？"
    一次很长的稳定阶段运行，可以在许多点上分叉，每个分叉后接一小段衰减，就能得到针对多种 token 预算的完整训练好的模型，而不必为每种预算单独跑一次余弦调度。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019 | **张量并行**：MLP 的列/行切分和注意力的按头切分。 |
| :material-star: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019 | 在数据并行 rank 之间对优化器状态、梯度和参数分片；上文的显存公式。 |
| :material-book-open-variant: [PyTorch FSDP](https://arxiv.org/abs/2304.11277) — Zhao et al., 2023 | ZeRO-3 风格的分片在 PyTorch 中如何实现与调优。 |
| :material-book-open-variant: [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., 2018 | 带 micro-batch 的流水线并行以及气泡。 |
| :material-book-open-variant: [Efficient Large-Scale LM Training on GPU Clusters (PTD-P)](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021 | 如何把张量并行 + 流水线并行 + 数据并行组合起来；交错式 1F1B 调度。 |
| :material-book-open-variant: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022 | **序列并行** + 选择性激活检查点；激活值显存公式。 |
| :material-book-open-variant: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Chen et al., 2016 | 最初的激活检查点（重计算）论文。 |
| :material-book-open-variant: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2017 | FP16/BF16 计算配合 FP32 主权重和损失缩放。 |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | 定义了 E4M3 和 E5M2，并表明 FP8 训练能达到 16 位的质量。 |
| :material-book-open-variant: [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2017 | LLM 的默认优化器。 |
| :material-book-open-variant: [MiniCPM](https://arxiv.org/abs/2404.06395) — Hu et al., 2024 | 提出 WSD 学习率调度，并把它用于缩放实验。 |
| :material-book-open-variant: [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — Keller Jordan, 2024 | Muon 的原始文章：动量 + Newton–Schulz 正交化，以及上文所用的系数。 |
| :material-new-box: [Muon is Scalable for LLM Training (Moonlight)](https://arxiv.org/abs/2502.16982) — Liu et al. (Moonshot AI), 2025 | 让 Muon 能在大规模上工作（权重衰减、更新量重缩放）；相对 AdamW 约有 2× 的算力效率。 |
| :material-book-open-variant: [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) — Wortsman et al., 2023 | 在小模型中复现注意力 logit 增长和输出 logit 发散；评估了 QK-layernorm 和 z-loss。 |
| :material-book-open-variant: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022 | z-loss、通过跳过 batch 从损失尖峰中恢复，以及许多大规模训练的细节。 |
| :material-book-open-variant: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) — Google DeepMind | 关于 roofline、分片以及 TPU/GPU 数学的最佳免费书籍。 |
