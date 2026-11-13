# 2.8 蒸馏与高效微调

!!! abstract "本章你将学到"
    - 知识蒸馏（knowledge distillation）：软目标与温度，以及前向 KL 与反向 KL 的区别（附代码）
    - 基于教师推理轨迹的序列级蒸馏，以及在线策略蒸馏（on-policy distillation，GKD）
    - 推导低秩适配（LoRA）：低秩更新、秩 $r$、$\alpha/r$ 缩放、该适配哪些层；从零实现一个 LoRA 层
    - 7B 模型上 LoRA 的参数量与显存实例演算
    - 直观理解 QLoRA（NF4、双重量化、分页优化器）和 DoRA，以及 LoRA 何时能与全量微调持平

## 为什么重要 {#why-this-matters-the-problem}

本部分到目前为止都假设你可以更新目标模型的全部权重。现实中有两个问题挡在前面：

1. **最好的行为存在于大模型里**，但大模型部署起来很贵。小模型能不能继承大模型学到的东西？这就是**蒸馏**。DeepSeek-R1 发现，在 R1 的推理轨迹上微调小模型，得到的推理能力比直接在小模型上做大规模 RL 更强（[第 2.7 章](07-rl-with-verifiable-rewards.md)）。
2. **全量微调非常吃显存。** 使用混合精度 Adam，训练每个参数大约需要 16 字节（[第 2.4 章](04-rlhf.md)）：7B 模型在算激活之前就超过 100 GB。**参数高效微调**（PEFT），主要是 **LoRA**，只训练极少量额外参数，因此你可以在一块消费级 GPU 上微调 7B 模型，并保留许多廉价的任务专用适配器。

## 直觉 {#intuition}

**蒸馏**：教师的输出分布比单一的"正确" token 包含多得多的信息。如果教师认为下一个 token 是"cat"的概率为 70%、"kitten"为 20%、"dog"为 5%，学生就会学到"kitten"是近义词，"dog"是一个说得通的混淆项：这就是 Hinton 等人所说的**暗知识（dark knowledge）**。在完整分布（软目标）上训练，比在 argmax（硬目标）上训练，每个样本教得更多。

**LoRA**：微调一个预训练模型时，每个权重矩阵的*变化量*往往是有效低秩的：只有少数几个方向起作用。所以与其学习完整的 $d\times d$ 更新，不如把它学成两个细长矩阵的乘积。这就像调几个旋钮，每个旋钮协同地带动许多权重，而不是逐个拧每一颗螺丝。

## 工作原理 {#how-it-works}

### 知识蒸馏：软目标与温度 {#knowledge-distillation-soft-targets-and-temperature}

对于某一个位置，设教师的 logits 为 $z^T \in \mathbb{R}^V$，学生的为 $z^S \in \mathbb{R}^V$（$V$ 为词表大小）。引入**温度** $\tau > 0$：

$$
p^T_\tau = \operatorname{softmax}(z^T/\tau), \qquad p^S_\tau = \operatorname{softmax}(z^S/\tau).
$$

用大白话说：把 logits 除以 $\tau > 1$ 会让分布变平，使小概率（暗知识）变大，更容易被损失看到；$\tau = 1$ 就是普通的 softmax。

经典的蒸馏损失（Hinton et al., 2015）把软目标项与通常的硬标签交叉熵混合起来：

$$
\mathcal{L}_{\text{KD}} = \alpha\,\tau^2\,\mathrm{KL}\big(p^T_\tau\,\|\,p^S_\tau\big) + (1-\alpha)\,\mathrm{CE}\big(y, p^S_1\big),
$$

其中 $y$ 是真实 token，$\alpha \in [0,1]$ 用来平衡两项。$\tau^2$ 这个因子用来补偿软化后的项其梯度大约按 $1/\tau^2$ 缩放这一事实，使得改变 $\tau$ 时两项的相对影响保持稳定。对 LLM 来说，这个逐位置的损失在回答的各个位置上求和，方式与 SFT 完全相同（[第 2.2 章](02-supervised-fine-tuning.md)）。**Logit 蒸馏要求教师和学生共用同一个分词器**（两个分布必须定义在同一个词表上）。

### 前向 KL 与反向 KL {#forward-vs-reverse-kl}

KL 散度是不对称的，方向不同，学生学到的东西也不同。

- **前向 KL** $\mathrm{KL}(p^T\|p^S) = \sum_v p^T(v)\log\frac{p^T(v)}{p^S(v)}$，在数据或教师产生的序列上取平均。当学生在教师给高概率的地方给了低概率，会受到重罚，所以学生会设法**覆盖教师的所有模态**（"均值寻求"）。一个无法表示所有模态的小学生会把概率摊开，可能生成模糊、低质量的样本。
- **反向 KL** $\mathrm{KL}(p^S\|p^T) = \sum_v p^S(v)\log\frac{p^S(v)}{p^T(v)}$。当学生在教师给低概率的地方给了概率，会受到惩罚，所以学生会**选定一个模态并坚持它**（"模态寻求"）。它可能丢掉一些合理的替代答案，但生成出来的内容以教师的标准看质量很高。

```text
 教师 p^T:           ▁▆█▆▁      ▁▆█▆▁        （两个模态）
 前向 KL 的学生:         ▁▃▅▅▅▅▅▅▅▃▁          （两个都覆盖，概率质量落在中间）
 反向 KL 的学生:     ▁▆█▆▁                    （只坚持一个模态）
```

回想一下，RLHF 的 KL 惩罚 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ 也是反向 KL（[第 2.3 章](03-rl-foundations-for-llms.md)）。

### 序列级蒸馏 {#sequence-level-distillation}

Kim 和 Rush（2016）提出了一个更简单的替代方案：让教师**生成**输出（例如它的束搜索或采样得到的答案），然后用普通 SFT 在这些输出上训练学生。不需要 logits，所以可以跨不同分词器使用，也适用于只能通过 API 访问的教师。如今大多数"蒸馏"的开源模型都是这样做出来的：经 R1 蒸馏的 Qwen 和 Llama 模型，就是在 R1 生成的约 80 万条样本上做 SFT 得到的（[第 2.7 章](07-rl-with-verifiable-rewards.md)），而 s1 用的是来自更强推理模型的 1,000 条推理轨迹（[第 2.6 章](06-reasoning-and-test-time-compute.md)）。

针对推理轨迹的实用要点：只保留最终答案经过验证为正确的轨迹（拒绝采样），去重，保持长上下文（轨迹可能有数万 token），并挑选对学生来说困难的问题。

### 在线策略蒸馏（GKD） {#on-policy-distillation-gkd}

上面的 logit 蒸馏和序列蒸馏都是**离策略**的：学生是在教师（或数据）产生的前缀上训练的。而在推理时，学生以*自己*之前生成的 token 为条件，其中可能含有教师永远不会犯的错误，这种训练与测试之间的不匹配叫做**暴露偏差（exposure bias）**。学生从来没学过如何从自己的错误中恢复。

**广义知识蒸馏（Generalized Knowledge Distillation，GKD）**（Agarwal et al., 2023）在**学生生成的**序列上训练：

1. 学生采样一个回答 $y \sim \pi_S(\cdot\mid x)$。
2. 教师在*学生回答的每一个位置*上计算自己的下一个 token 分布。
3. 在这些位置上最小化教师分布与学生分布之间的逐 token 散度（论文考虑了前向 KL、反向 KL，以及在两者之间插值的广义 Jensen–Shannon 散度），并且可以混合同策略序列和固定数据序列。

用反向 KL 时，学生样本上的逐 token 损失是 $\sum_v p^S(v\mid s_t)\big(\log p^S(v\mid s_t) - \log p^T(v\mid s_t)\big)$，这看起来就像带稠密逐 token 奖励 $\log p^T - \log p^S$ 的 RL：教师给学生实际写出的每一个 token 打分。它在每个样本上提供的信号远多于 RL 那一个标量奖励，而且是同策略的。例如，Qwen3 技术报告就描述了它为小模型设计的"先离策略、后在线策略"的蒸馏阶段。

### 推导 LoRA {#lora-derived}

取预训练模型中的一个线性层 $h = W_0 x$，其中 $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$，输入 $x\in\mathbb{R}^{d_{\text{in}}}$。全量微调学习 $W = W_0 + \Delta W$，其中 $\Delta W$ 是稠密的：$d_{\text{out}}\cdot d_{\text{in}}$ 个可训练的数。

**LoRA**（Hu et al., 2021）冻结 $W_0$，并把更新约束为秩至多为 $r$：

$$
\Delta W = \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d_{\text{out}}\times r},\quad A \in \mathbb{R}^{r\times d_{\text{in}}},\quad r \ll \min(d_{\text{in}}, d_{\text{out}}),
$$

于是前向传播变为

$$
h = W_0 x + \frac{\alpha}{r} B (A x).
$$

用大白话说：输入先被 $A$ 投影到 $r$ 维，再由 $B$ 投回来，缩放后加到冻结层的输出上。各符号的含义：

- $r$，即**秩**：更新能使用的相互独立方向的数量（典型值 8–64；RL 中 1–16 就可能足够）。
- $\alpha$，即**缩放**：一个常数；更新会乘以 $\alpha/r$，这样改变 $r$ 时不必那么频繁地重新调学习率。常见的选择是 $\alpha = 2r$，或固定的 $\alpha$（如 16 或 32）。
- **初始化**：$A$ 随机初始化（例如高斯或 Kaiming 均匀分布），$B = 0$。这样一开始 $\Delta W = 0$，训练恰好从预训练模型起步；$B$ 能收到非零梯度，因为 $Ax \neq 0$。

每层的可训练参数：$r(d_{\text{in}} + d_{\text{out}})$，而不是 $d_{\text{in}}d_{\text{out}}$。对于一个 $4096\times4096$ 的方阵层，$r = 16$：$16\times8192 = 131{,}072$ 对 $16{,}777{,}216$，减少了 128 倍。

**适配哪些层？** 原论文只适配了注意力的 query 和 value 投影。后来的工作，包括 QLoRA 论文和 Thinking Machines 的"LoRA Without Regret"（2025），发现把 LoRA 用在**所有线性层**上更好，尤其是 MLP（和 MoE）层；即使参数量相同，只适配注意力的 LoRA 表现也更差。

**合并。** 训练结束后，可以一次性算出 $W_0 + \frac{\alpha}{r}BA$ 并作为普通稠密矩阵存储，所以合并后的 LoRA 模型**推理时没有额外开销**。或者，也可以让适配器保持独立（每个几 MB 到几百 MB），在同一个基座模型上切换或批量使用许多适配器。

**为什么显存下降。** 只有 $A$ 和 $B$ 需要梯度和 Adam 状态。冻结的 $W_0$ 两者都不需要，甚至可以用低精度存储（QLoRA）。

### QLoRA：在 4 比特基座模型上做 LoRA {#qlora-lora-on-a-4-bit-base-model}

QLoRA（Dettmers et al., 2023）把冻结的基座权重存成 4 比特，并在其上训练 bf16 的 LoRA 适配器，反向传播*穿过*量化后的权重（每次矩阵乘法时即时反量化）。包含三个想法：

1. **NF4（4 比特 NormalFloat）。** 预训练权重大致服从正态分布。NF4 的 16 个量化级别放置在正态分布的分位数处，因此每个级别被使用的频率大致相同，这在信息论上对正态数据是最优的。权重以 64 个为一块进行量化，每块有自己的缩放因子（该块的绝对值最大值）。
2. **双重量化。** 这些逐块的缩放因子本身是 32 位的数：每个参数额外多出 $32/64 = 0.5$ 比特。QLoRA 把缩放因子量化为 8 位（以 256 个为一块），把开销降到每个参数约 $8/64 + 32/(64\cdot256) \approx 0.127$ 比特。
3. **分页优化器。** 显存峰值（例如来自很长的序列）可能让训练崩溃。优化器状态被分配在 NVIDIA 统一内存中，因此当 GPU 显存不足时可以换页到 CPU 内存，就像虚拟内存一样。

最醒目的结果：在单块 48 GB GPU 上微调 65B 模型，质量与他们基准上的 16 位全量微调相当。

### DoRA：把幅度和方向分开 {#dora-separate-magnitude-and-direction}

DoRA（Liu et al., 2024）把每个权重矩阵分解成一个**幅度**向量 $m$（每一列的范数）和**方向**（归一化为单位长度的各列）：

$$
W' = m \odot \frac{W_0 + BA}{\lVert W_0 + BA\rVert_c},
$$

其中 $\lVert\cdot\rVert_c$ 是按列计算的范数，$m$ 直接训练（初始化为 $\lVert W_0\rVert_c$）。用大白话说：LoRA 调整每一列的方向；一个单独的小向量调整它的长度。作者观察到，全量微调改变幅度和方向的模式与 LoRA 不同，而把两者解耦能以很小的额外代价弥合与全量微调之间的大部分差距。

### LoRA 何时能与全量微调持平？ {#when-does-lora-match-full-fine-tuning}

"LoRA Without Regret"（Thinking Machines, 2025）做了细致的对比，并报告：

- 对于**小到中等规模指令和推理数据集上的 SFT**，LoRA 与全量微调表现相同。
- 对于**超出 LoRA 容量的数据集**（非常大，例如持续预训练的规模），LoRA 学得效率较低，表现不如全量微调。
- 对于 **RL**，即使秩非常低（低至秩 1）也能与全量微调持平，因为 RL 每个回合提供的信息很少（大致就是一个标量奖励），所以需要的容量很小。
- 把 LoRA 用在**所有层**上，尤其是 MLP；最优学习率大约比全量微调**高 10 倍**；LoRA 对特别大的批大小可能没那么耐受。

## 代码 {#code}

### 1. 蒸馏损失（带温度的前向 KL、反向 KL） {#1-distillation-losses-forward-kl-with-temperature-reverse-kl}

```python
import torch, torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, tau=2.0, alpha=0.5, ignore=-100):
    """Logits: [B, L, V]，已对齐，使位置 t 预测 labels[:, t]；labels: [B, L]。
    前向 KL 软损失 (Hinton) + 硬交叉熵，在未被掩码的位置上取平均。"""
    mask = (labels != ignore)
    s = student_logits[mask]                                  # [N, V]
    t = teacher_logits[mask].detach()
    soft = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1),
                    log_target=True, reduction="batchmean")   # KL(p_T || p_S)
    hard = F.cross_entropy(s, labels[mask])
    return alpha * tau**2 * soft + (1 - alpha) * hard

def reverse_kl_loss(student_logits, teacher_logits, mask):
    """在学生采样序列的各位置上，对整个词表计算逐 token 的 KL(p_S || p_T)
    （同策略 / GKD 风格）。Logits: [B, L, V]，mask: [B, L]。"""
    log_ps = F.log_softmax(student_logits, -1)
    log_pt = F.log_softmax(teacher_logits.detach(), -1)
    kl = (log_ps.exp() * (log_ps - log_pt)).sum(-1)           # [B, L]
    return (kl * mask).sum() / mask.sum()

B, L, V = 2, 5, 11
s, t = torch.randn(B, L, V, requires_grad=True), torch.randn(B, L, V)
labels = torch.randint(0, V, (B, L)); labels[:, :2] = -100    # 前 2 个位置 = prompt
print(kd_loss(s, t, labels).item(), reverse_kl_loss(s, t, (labels != -100).float()).item())
```

### 2. 从零实现一个 LoRA 层 {#2-a-lora-layer-from-scratch}

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear：h = W0 x + (alpha/r) * B(A(x))。"""
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                           # 冻结 W0（以及偏置）
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B = 0  ->  初始时 Delta W = 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                     # x: [..., d_in]
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        """把更新并入 W0，实现零开销推理；返回一个普通的 nn.Linear。"""
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base

def add_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"), r=16, alpha=32):
    """替换所有属性名在 `targets` 中的 nn.Linear（默认是 Llama/Qwen 风格模块中的所有
    线性层），并冻结其余部分。"""
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
    return model

# 自检
layer = nn.Linear(64, 32)
lora = LoRALinear(layer, r=4, alpha=8)
x = torch.randn(3, 64)
assert torch.allclose(lora(x), layer(x))                     # 初始化时完全一致
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(trainable, 4 * (64 + 32))                               # 384 384
```

要在 Hugging Face 模型上使用：`model = add_lora(AutoModelForCausalLM.from_pretrained(...))`，然后只把 `[p for p in model.parameters() if p.requires_grad]` 传给优化器，并完全按[第 2.2 章](02-supervised-fine-tuning.md)的方式训练，学习率取全量微调的约 10 倍。`peft` 库实现的是同一个想法，并带有许多额外功能。

## 实例演算 {#worked-example}

**Llama-2-7B 形状模型上的 LoRA 参数量与显存。** 隐藏维度 $d = 4096$，MLP 中间维度 $d_{ff} = 11008$，32 层，约 6.7B 参数。在每个模块的全部七个线性层上使用 $r = 16$ 的 LoRA。

每一层：

- 注意力 $q, k, v, o$：每个都是 $4096\to4096$：$16\times(4096+4096) = 131{,}072$；四个共 $524{,}288$。
- MLP 的 gate 和 up（$4096\to11008$）以及 down（$11008\to4096$）：每个 $16\times(4096+11008) = 241{,}664$；三个共 $724{,}992$。
- 每层合计：$1{,}249{,}280$。

全部 32 层：$32\times1{,}249{,}280 = 39{,}976{,}960 \approx 40$M 可训练参数，约占 6.7B 的 **0.6%**。

显存（权重 + 梯度 + Adam 状态，不含激活）：

| 配置 | 计算 | 大致显存 |
|---|---|---|
| 全量微调，混合精度 Adam | $6.7\text{B}\times16$ 字节 | ≈ 107 GB |
| LoRA，bf16 冻结基座 | $6.7\text{B}\times2 + 40\text{M}\times16$ | ≈ 13.4 + 0.64 ≈ 14 GB |
| QLoRA，NF4 基座（≈4.127 比特/参数） | $6.7\text{B}\times4.127/8 + 40\text{M}\times16$ | ≈ 3.5 + 0.64 ≈ 4.1 GB |

适配器检查点本身是 $40\text{M}\times2$ 字节 ≈ 80 MB（bf16），而完整模型的一份拷贝是 13.4 GB。

**数字上看温度。** 教师对（"cat"、"kitten"、"dog"）的 logits 为 $(4, 2, 0)$。$\tau = 1$ 时：$\operatorname{softmax} = (0.867, 0.117, 0.016)$。$\tau = 2$ 时：logits 变为 $(2, 1, 0)$，$\operatorname{softmax} = (0.665, 0.245, 0.090)$。次要选项之间的关系（kitten ≫ dog）在 $\tau = 2$ 时对学生来说清晰得多。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **蒸馏是小模型变强的方式。** 大多数小型开源模型在很大程度上是用更大模型的输出训练出来的：在推理轨迹上做序列级 SFT 是标准做法；当教师与学生共用分词器时（例如同一模型家族内），会使用 logit 级和在线策略蒸馏。
- **在线策略蒸馏越来越受欢迎**，对小模型而言，它是 RL 的一个廉价替代或补充，因为它在学生自己的样本上提供稠密的逐 token 信号。
- **LoRA 是前沿实验室之外大多数微调的默认选择**，并且越来越多地用于 RL；多适配器部署让一个基座模型可以同时承载许多微调版本。
- **典型的 LoRA 设置**：所有线性层，SFT 时 $r$ 取 16–64（RL 可以更低），$\alpha$ 取 $r$ 到 $2r$ 左右，学习率约为全量微调的 10 倍（7B 规模的 SFT 常在 $10^{-4}$ 左右），不用或只用很小的 dropout。
- **QLoRA** 是在单块 GPU 上装下大模型的首选；为了快速推理，可以把适配器合并进更高精度的基座，或者单独用适配器来部署量化后的基座。

## 常见误区 {#common-pitfalls-misconceptions}

- **跨分词器做 logit 蒸馏。** 定义在不同词表上的分布无法直接比较；请改用序列级蒸馏。
- **忘记 $\tau^2$ 因子**，导致改变温度时悄悄改变了软损失的有效权重。
- **$A$ 和 $B$ 都随机初始化。** 模型不再从预训练模型的函数起步，早期训练会不稳定。两者之中必须有一个为零。
- **对 LoRA 使用全量微调的学习率。** 太低；LoRA 通常需要高出约 10 倍。
- **默认只对注意力做 LoRA。** 请把 MLP 层也包括进来。
- **不小心地合并进量化基座。** 把 bf16 适配器合并进 4 比特权重再重新量化会损失精度；请合并到一份 bf16 的拷贝里。
- **"蒸馏会完全复制教师。"** 学生的容量限制了它能吸收多少；前向 KL 的学生可能变得模糊，反向 KL 的学生可能丢掉一些模态。

## 本章要点 {#key-takeaways}

- 蒸馏在教师的软分布（logit 蒸馏）或教师生成的输出（序列蒸馏）上训练学生；温度可以暴露暗知识。
- 前向 KL 覆盖教师的各个模态；反向 KL 坚持选定的模态；在线策略蒸馏（GKD）在学生自己的样本上做评估，以消除暴露偏差。
- LoRA：$h = W_0x + \frac{\alpha}{r}BAx$，初始时 $B=0$；每层训练 $r(d_{\text{in}}+d_{\text{out}})$ 个参数，推理时可以免费合并。
- QLoRA 加入了 4 比特 NF4 基座、双重量化和分页优化器；DoRA 把幅度与方向分开。
- 当应用于所有层并使用约 10 倍的学习率时，LoRA 在 RL 以及小到中等规模的 SFT 上能与全量微调持平。

## 自测 {#check-your-understanding}

??? question "为什么 LoRA 中 $B$ 初始化为零，它为什么仍然能收到梯度？"
    这样 $\Delta W = BA = 0$，模型恰好从预训练权重开始。损失对 $B$ 的梯度正比于 $(Ax)^\top$，由于 $A$ 是随机的，它不为零，所以 $B$ 马上就开始学习（随后 $A$ 也开始学习）。

??? question "一个 $5120\times5120$ 的层在秩为 8 时有多少个 LoRA 参数？"
    $8\times(5120+5120) = 81{,}920$，而完整矩阵有 $26{,}214{,}400$ 个。

??? question "一个小学生模型在多模态的教师上用前向 KL 训练，产生的文本平淡、像被平均过。你会尝试什么？"
    反向 KL（模态寻求），或使用反向 KL 或 JSD 目标的在线策略蒸馏，让学生坚持选定教师认为高质量的模态，而不是在它们之间摊开概率质量。

??? question "什么是暴露偏差，GKD 如何解决它？"
    离策略蒸馏在教师/数据的前缀上训练，但推理时学生是以自己（可能有错）的 token 为条件的，这是它在训练中从未见过的。GKD 让学生生成序列，并让教师在这些序列上给出逐 token 的目标，使训练与推理条件一致。

??? question "为什么 RL 用秩 1 的 LoRA 就行，而大数据集上的 SFT 不行？"
    RL 每个回合提供的信息非常少（大约一个标量奖励），所以需要吸收的总信息量很小；而大型 SFT 数据集每个 token 都提供许多比特的信息，可能超出低秩适配器的容量。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015 | 软目标、温度与"暗知识"。 |
| :material-book-open-variant: [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) — Kim & Rush, 2016 | 在教师生成的序列上训练学生；轨迹蒸馏的基础。 |
| :material-book-open-variant: [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023 | 学生生成，教师给每个 token 打分；修复暴露偏差。 |
| :material-star: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 | 冻结权重，学习一个低秩更新 $\Delta W = BA$。 |
| :material-star: [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023 | 在 4 比特 NF4 基座上做 LoRA，带双重量化和分页优化器；单块 48 GB GPU 上跑 65B。 |
| :material-book-open-variant: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al., 2024 | 解耦幅度与方向；弥合了 LoRA 与全量微调之间的大部分差距。 |
| :material-new-box: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — Thinking Machines, 2025 | LoRA 何时能与全量微调持平（RL、小/中等规模 SFT）、全层 LoRA、约 10 倍学习率法则。 |
| :material-star: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | 表明在小模型上，推理轨迹的蒸馏胜过 RL。 |
