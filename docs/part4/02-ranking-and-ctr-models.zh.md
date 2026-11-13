# 4.2 精排与 CTR 模型

!!! abstract "本章你将学到"
    - 点击率（CTR）预估如何被建模为基于稀疏 ID 特征和 embedding 表的二分类问题
    - 从逻辑回归 → 因子分解机 → Wide & Deep → DCN/DCN-V2 → DLRM 的演进脉络，包括 FM 的 $O(kn)$ 技巧和 DCN-V2 的交叉层
    - 目标注意力（DIN）：按与候选的相关性给用户的历史行为加权，附代码
    - 多任务精排（shared-bottom、MMoE、PLE），以及如何把多个预测融合成一个分数
    - 为什么以及如何校准预测概率（Platt scaling、isotonic regression、降采样修正），以及"embedding collapse"是什么意思

## 为什么重要 {#why-this-matters-the-problem}

精排阶段（第 [4.1](01-recsys-fundamentals.md) 章）接收几百到几千个候选，并要决定它们的顺序。过去十年，工业界大部分建模精力都花在这里，因为它负担得起丰富的特征：关于用户的一切、关于物品的一切、上下文，以及——最重要的——它们之间的*交互*（"这个用户喜欢做菜视频，*并且*这个视频是做菜视频"）。

核心任务是为每个候选预测一个或多个行为的概率：点击、点赞、长时间观看、购买、转化。这些概率必须足够好，以便 (a) 把候选排对顺序，以及在广告中 (b) 与金额相乘（第 [4.8](08-ads-systems.md) 章），这就要求它们是**校准**过的。本章从第一性原理出发搭建经典架构；第 [4.4](04-scaling-recommendation-models.md) 章会解释它们为什么不再 scale，以及后来是什么取代了它们。

## 直觉 {#intuition}

精排模型的输入主要是**类别型 ID**：用户 ID、物品 ID、作者 ID、品类、国家、设备、一天中的小时，以及用户互动过的 ID 列表。原始 ID 没有几何结构，所以每个 ID 都被映射成一个学出来的向量（**embedding**）。随后模型需要学会哪些特征*组合*能预测点击。历史上，工程师是手工构造这些组合（"user_country × item_language"）。本章的这些架构，就是一次次让组合自动、低成本地被学出来的尝试：

- **FM：** 每一对特征通过各自 embedding 的点积发生交互。
- **Wide & Deep：** 在线性部分里记忆特定的交叉特征，用 MLP 做泛化。
- **DCN：** 逐层学习次数不断升高的显式多项式交叉。
- **DLRM：** 对所有 embedding 两两做点积，再接 MLP——一个干净的参考设计。
- **DIN：** 用户历史是一个*物品集合*；让候选来决定哪些过去的物品重要。

## 工作原理 {#how-it-works}

### 作为二分类的 CTR 预估 {#ctr-prediction-as-binary-classification}

每个训练样本是一次曝光，带有特征 $\mathbf{x}$ 和标签 $y \in \{0,1\}$（点击与否）。模型输出 logit $z = f(\mathbf{x})$ 和概率 $\hat p = \sigma(z) = 1/(1+e^{-z})$。训练最小化二元交叉熵（LogLoss）：

$$
\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n}\left[y_t \log \hat p_t + (1-y_t)\log(1-\hat p_t)\right].
$$

用话来说：在被点击的曝光上把 $\hat p$ 推高，在未点击的曝光上把它压低。最小化 LogLoss 的最优解满足 $\hat p = P(y=1\mid\mathbf{x})$——只要模型表达能力足够、训练数据无偏，模型就是校准的。

### 稀疏特征与 embedding 表 {#sparse-features-and-embedding-tables}

词表大小为 $V$ 的类别特征被编码为 one-hot 向量 $\mathbf{x} \in \{0,1\}^V$，再乘以 embedding 矩阵 $\mathbf{E} \in \mathbb{R}^{V\times d}$，这等价于一次行查找：`E[id]`。多值特征（例如最近点击的 50 个物品 ID）会被逐个查找并做**池化**（求和、均值或注意力）。

词表既巨大又开放（每秒都有新用户和新物品），所以系统使用**哈希技巧**：行号 $= \text{hash}(\text{id}) \bmod M$，其中 $M$ 是固定的表大小。冲突（两个 ID 共用一行）是为了限定内存所付出的代价。Embedding 表主导了参数量：工业模型的 embedding 表有 TB 级，而稠密网络的参数要少得多。这就是 DLRM 类系统是**内存受限**的、要对表做模型并行，并且——正如第 4.4 章所解释的——无法像 LLM 那样"scale"的原因。

### 逻辑回归 {#logistic-regression}

$$
z = w_0 + \sum_{i=1}^{n} w_i x_i ,
$$

其中 $x_i$ 是（大多为二值的）特征，$w_i$ 是学到的权重。LR 无法建模交互，除非你手工加入交叉特征。它曾是工业界十多年的主力——Google 的 FTRL-Proximal 在线 LR 和 Facebook 的"GBDT 叶子节点作特征 + LR"是经典参考——因为它速度快、可以在线训练、也容易校准。

### 因子分解机（FM） {#factorization-machines-fm}

FM 加入了所有的两两交互，但每个特征 $i$ 有一个隐向量 $\mathbf{v}_i \in \mathbb{R}^k$，特征对 $(i,j)$ 的权重是 $\langle \mathbf{v}_i, \mathbf{v}_j\rangle$：

$$
z = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j\rangle\, x_i x_j .
$$

用话来说：如果两个特征的隐向量方向一致，它们就有很强的交互。因为特征对的权重是被分解出来的，所以模型能估计训练中从未同时出现过的特征对的交互（如果用户 A 的向量与用户 B 的接近，A 就继承了 B 的交互）。

**$O(kn)$ 技巧。** 朴素地计算两两交互项，开销是 $O(kn^2)$。对每个隐维度 $f$，记 $s_f = \sum_i v_{i,f}x_i$。则

$$
\sum_{i<j}\langle \mathbf{v}_i,\mathbf{v}_j\rangle x_i x_j
= \frac{1}{2}\sum_{f=1}^{k}\left[\Big(\sum_{i=1}^n v_{i,f}x_i\Big)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2\right].
$$

??? info "推导"
    $\big(\sum_i a_i\big)^2 = \sum_i a_i^2 + 2\sum_{i<j} a_i a_j$。对每个维度 $f$ 取 $a_i = v_{i,f}x_i$ 并应用该式：$\sum_{i<j} v_{i,f}v_{j,f}x_ix_j = \frac{1}{2}\big[(\sum_i v_{i,f}x_i)^2 - \sum_i v_{i,f}^2x_i^2\big]$。对 $f$ 求和，左边就是 $\sum_{i<j}\langle \mathbf{v}_i,\mathbf{v}_j\rangle x_ix_j$。

用话来说："和的平方减去平方的和，再除以二"——只需对 $n$ 个特征遍历一遍，$O(kn)$。当 embedding 已经被查好并堆成张量 `e: [batch, n_fields, k]` 时，FM 项就是 `0.5 * (e.sum(1)**2 - (e**2).sum(1)).sum(1)`。

### Wide & Deep {#wide-deep}

Wide & Deep（Google Play，2016）联合训练两部分，并把它们的 logit 相加：

$$
z = \underbrace{\mathbf{w}_{\text{wide}}^\top[\mathbf{x}, \phi(\mathbf{x})]}_{\text{memorization}} + \underbrace{\mathbf{w}_{\text{deep}}^\top \mathbf{a}^{(L)}}_{\text{generalization}} + b,
$$

其中 $\phi(\mathbf{x})$ 是人工选定的交叉乘积特征（例如"已安装应用 = X 且 曝光应用 = Y"），$\mathbf{a}^{(L)}$ 是作用在拼接 embedding 上的 MLP 的最后一个隐藏层。wide 部分记忆特定的高频共现；deep 部分借助 embedding 泛化到没见过的组合。这套"记忆 vs 泛化"的说法至今仍在使用（例如用于第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章中的语义 ID）。

### Deep & Cross Network（DCN、DCN-V2） {#deep-cross-network-dcn-dcn-v2}

MLP 学习乘性交互的效率很低。DCN 加入了一个**交叉网络**，显式地构造多项式交互。设 $\mathbf{x}_0 \in \mathbb{R}^D$ 是所有 embedding 与稠密特征的拼接。DCN-V2 的交叉层是

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l ,
$$

其中 $\mathbf{W}_l \in \mathbb{R}^{D\times D}$，$\mathbf{b}_l \in\mathbb{R}^D$，$\odot$ 是逐元素乘积。用话来说：每一层把原始输入与当前表示的一个线性变换相乘，所以第 $l$ 层包含最高 $l+1$ 次的特征交互，而残差项 $+\mathbf{x}_l$ 保留了低次项。（最初的 DCN 使用秩为 1 的版本，$\mathbf{x}_0\mathbf{x}_l^\top\mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$；V2 的满矩阵——通常再分解成低秩形式 $\mathbf{U}\mathbf{V}^\top$——表达力更强。）交叉网络可以与 MLP 并行或串行，是一个很强的生产基线。

### DLRM {#dlrm}

Meta 的 DLRM 是一个参考架构：稠密特征经过底部 MLP 变成 $d$ 维向量；每个稀疏特征做一次 embedding 查找得到 $d$ 维向量；然后对这 $m$ 个向量计算**所有两两点积**（即显式的 FM 思想），与稠密向量拼接，再送入顶部 MLP 输出 logit。它在系统上的贡献与模型本身同样重要：embedding 表在多个设备间切分（**模型并行**），而 MLP 则被复制（**数据并行**），每一步都要对查到的 embedding 做一次 all-to-all 交换。

### 目标注意力：DIN {#target-attention-din}

用户的行为历史 $[e_1, \dots, e_T]$（最近 $T$ 个互动过的物品的 embedding）是精排中最强的信号。把它直接求和池化成一个向量会模糊一切：一个既看做菜又看足球的用户，无论候选是一份菜谱还是一场比赛，得到的摘要都是一样的。**Deep Interest Network（DIN）**让池化依赖于候选（"目标"）$e_a$：

$$
\mathbf{v}_u(a) = \sum_{j=1}^{T} \alpha(e_j, e_a)\, e_j, \qquad \alpha(e_j,e_a) = \text{MLP}\big([e_j;\ e_a;\ e_j - e_a;\ e_j \odot e_a]\big).
$$

用话来说：每个过去的行为都按一个小网络估计的它与候选的相关性来加权，所以用户表示是针对每个候选重新计算的。DIN 有意**不**对权重做 softmax 归一化，所以相关行为很多的用户会得到更大的向量——兴趣的*强度*被保留了下来。（记住这一点；HSTU 在第 4.4 章做了同样的选择。）这是精排中序列建模的起点，将在第 [4.3](03-sequential-user-modeling.md) 章中展开。

### 多任务学习 {#multi-task-learning}

信息流会同时优化多个行为——点击、点赞、分享、关注、观看时长、跳过。为每个任务单独训练一个模型既浪费算力又浪费数据；共享则能帮助稀疏任务（例如分享）从稠密任务（例如点击）中借力。

- **Shared-bottom：** 一个共享网络产生表示 $\mathbf{h}$，各任务专属的塔 $h^k(\mathbf{h})$ 预测每个任务 $k$。结构简单，但相互冲突的任务会把共享层往不同方向拉扯（**负迁移**）。
- **MMoE（multi-gate mixture of experts）：** 有 $E$ 个专家网络 $f_1..f_E$；每个任务 $k$ 有自己的 softmax 门 $\mathbf{g}^k(\mathbf{x}) \in \mathbb{R}^E$，看到的是混合结果 $\mathbf{h}^k = \sum_{i=1}^E g^k_i(\mathbf{x}) f_i(\mathbf{x})$，再送入各自的塔。任务一致时可以共享专家，冲突时可以使用不同的专家。
- **PLE（progressive layered extraction）：** 把专家显式地分成共享专家和任务专属专家，叠成多层，以进一步减少负迁移。可以把它理解为"为每个任务预留了容量的 MMoE"。

损失是加权和 $\sum_k \lambda_k \mathcal{L}_k$，二值任务用 BCE，观看时长则用例如回归或加权 logistic 损失。

### 把多个目标合成一个分数 {#combining-objectives-into-one-score}

精排阶段必须输出唯一的排序。常见的形式是预测值的加权乘积或加权和，例如

$$
\text{score} = \hat p_{\text{click}}^{\,\alpha}\cdot (1 + \hat p_{\text{like}})^{\beta}\cdot \widehat{\text{watch}}^{\,\gamma},
$$

其中指数 $\alpha, \beta, \gamma$ 通过 A/B 实验或针对线上指标的黑盒搜索来调。在广告中，分数必须是金额单位（每次曝光的期望价值；第 4.8 章），这也是广告预测必须校准的主要原因。这些手工调参的融合公式，正是带有学习型 reward model 的生成式推荐想要取代的东西之一（第 [4.6](06-end-to-end-generative-recommenders.md) 章）。

### 校准 {#calibration}

如果在模型预测 $\hat p \approx 0.03$ 的曝光中大约有 3% 被点击，就说这个模型是**校准**（calibrated）的。模型会因为负样本降采样、分布漂移、位置效应或过拟合而偏离校准。一个简单的诊断指标是**校准比** $\sum_t \hat p_t / \sum_t y_t$（整体上以及在每个分桶内都应当 ≈ 1）。

修正方法（在保留的最近一段数据上进行）：

- **降采样修正。** 如果负样本以比例 $w$（例如 10%）保留，那么训练出的模型在重新平衡后的数据上预测的是 $p'$。校准后的概率为 $p = \dfrac{p'}{p' + (1-p')/w}$。
- **Platt scaling。** 拟合 $p = \sigma(a\cdot \text{logit}(\hat p) + b)$，通过最小化 LogLoss 求两个标量 $a,b$。
- **Isotonic regression。** 拟合一个从 $\hat p$ 到经验率的非递减阶梯函数（pool-adjacent-violators 算法）。更灵活，保序，但需要更多数据。

生产系统通常按分段（例如按广告类型或展示位）分别校准，并使用一个在线训练的小型校准层。

### Embedding collapse {#embedding-collapse}

把 embedding 做大是增加参数最便宜的办法，但它往往到某个点就不再有帮助。一种解释是 **embedding collapse**：学到的 embedding 矩阵只占据一个低维子空间（大多数奇异值接近于零），所以多出来的维度被浪费了。Guo 等人认为，特征交互层有两面性——与已经坍缩的 embedding 做交互会传播坍缩，但交互又是避免拟合伪特征所必需的——并提出了 **multi-embedding**：使用若干套独立的 embedding，每套配各自的交互模块，从而保持 embedding 的多样性。你可以通过画出 embedding 表的奇异值来诊断坍缩。

## 代码 {#code}

FM、DCN-V2 交叉层和 DIN 注意力单元，其中 FM 技巧做了数值验证。

```python
import torch, torch.nn as nn
torch.manual_seed(0)

def fm_pairwise(e):
    """e: [batch, n_fields, k]，embedding（已乘以特征值）。
    返回每个样本的 sum_{i<j} <e_i, e_j>：[batch]。复杂度 O(n*k)。"""
    square_of_sum = e.sum(dim=1) ** 2            # [batch, k]
    sum_of_square = (e ** 2).sum(dim=1)          # [batch, k]
    return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

e = torch.randn(4, 10, 8)
naive = torch.stack([sum((e[b, i] * e[b, j]).sum() for i in range(10) for j in range(i + 1, 10))
                     for b in range(4)])
assert torch.allclose(fm_pairwise(e), naive, atol=1e-4)

class CrossLayerV2(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l   (DCN-V2)，低秩 W = U V^T。"""
    def __init__(self, dim, rank=16):
        super().__init__()
        self.V = nn.Linear(dim, rank, bias=False)
        self.U = nn.Linear(rank, dim)
    def forward(self, x0, xl):                   # 均为 [batch, dim]
        return x0 * self.U(self.V(xl)) + xl

class DINAttention(nn.Module):
    """目标注意力：按与候选的相关性给每个历史物品加权（不做 softmax）。"""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, hist, target, mask):
        # hist: [B, T, d], target: [B, d], mask: [B, T] (1 = 真实物品, 0 = 填充)
        t = target[:, None, :].expand_as(hist)                       # [B, T, d]
        feats = torch.cat([hist, t, hist - t, hist * t], dim=-1)     # [B, T, 4d]
        w = self.mlp(feats).squeeze(-1) * mask                       # [B, T]
        return (w[:, :, None] * hist).sum(dim=1)                     # [B, d]

class TinyRanker(nn.Module):
    def __init__(self, n_items, n_users, d=16):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        self.user = nn.Embedding(n_users, d)
        self.din = DINAttention(d)
        self.cross = nn.ModuleList([CrossLayerV2(3 * d) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(3 * d, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, user, hist, cand):        # [B], [B, T], [B]
        h, c = self.item(hist), self.item(cand)
        interest = self.din(h, c, (hist != 0).float())
        x0 = torch.cat([self.user(user), c, interest], dim=-1)       # [B, 3d]
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(x).squeeze(-1)          # logit；用 BCEWithLogitsLoss 训练

model = TinyRanker(n_items=1000, n_users=100)
logit = model(torch.randint(0, 100, (8,)), torch.randint(0, 1000, (8, 20)), torch.randint(1, 1000, (8,)))
print(logit.shape)                               # torch.Size([8])
```

在一个刻意设成过度自信的模型上，用 Platt scaling 和 isotonic regression（pool-adjacent-violators）做校准：

```python
import torch
torch.manual_seed(0)
n = 20000
true_p = torch.rand(n) * 0.1                      # 真实 CTR 在 0 到 10% 之间
y = torch.bernoulli(true_p)
pred = torch.clamp(true_p * 2.0, max=0.999)       # 模型高估了 2 倍

def logit(p): return torch.log(p / (1 - p))

# Platt scaling: p_cal = sigmoid(a * logit(pred) + b)
a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([a, b], max_iter=100)
def closure():
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(a * logit(pred) + b, y)
    loss.backward(); return loss
opt.step(closure)
platt = torch.sigmoid(a * logit(pred) + b).detach()

# 对排好序的预测值用 pool-adjacent-violators 做 isotonic regression
order = pred.argsort()
blocks = []                                       # 每个 block：[sum_y, count]
for yi in y[order].tolist():
    blocks.append([yi, 1])
    while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
        s, c = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += c
iso_sorted = torch.cat([torch.full((c,), s / c) for s, c in blocks])
iso = torch.empty(n); iso[order] = iso_sorted

for name, p in [("raw", pred), ("platt", platt), ("isotonic", iso)]:
    print(f"{name:9s} calibration ratio = {p.sum() / y.sum():.3f}")
```

## 实例演算 {#worked-example}

**手算 FM 技巧。** 三个激活的特征（$x_i = 1$），$k=2$ 的 embedding 为 $\mathbf{v}_1=(1,2)$、$\mathbf{v}_2=(0,1)$、$\mathbf{v}_3=(2,-1)$。

- 朴素算法：$\langle v_1,v_2\rangle = 2$，$\langle v_1,v_3\rangle = 2-2 = 0$，$\langle v_2,v_3\rangle = -1$。和 $= 1$。
- 技巧：求和向量 $= (3, 2)$，逐维平方 $= (9, 4)$；各维平方和 $= (1+0+4,\ 4+1+1) = (5, 6)$。差 $= (4, -2)$，求和得 $2$，再除以 2 得 $1$。✓

**Embedding 表大小。** 一张有 $10^9$ 个哈希 ID × 64 维 × 4 字节（fp32）的表 $= 2.56\times10^{11}$ 字节 $\approx 256$ GB——这还只是*一个*特征。一个有几十个大 ID 特征的模型会达到 TB 级，而一个有 $10^7$ 个参数的稠密 MLP 只有 40 MB。这种不对称正是第 4.4 章中 scaling 问题的根源。

**降采样修正。** 负样本以比例 $w = 0.1$ 保留。模型对某次曝光输出 $p' = 0.2$。修正后：$p = 0.2 / (0.2 + 0.8/0.1) = 0.2/8.2 \approx 0.0244$。如果不修正，这条广告会被估价为其真实点击概率的 8 倍。

**校准分桶检查。** 在 10,000 次预测落在 $[0.04, 0.05)$（均值 0.045）的曝光中，观察到 220 次点击：经验率 0.022，校准比 $0.045/0.022 \approx 2.0$。Isotonic regression 会把这个桶映射到约 0.022；Platt scaling 会拟合一个全局的 $a,b$，大致把这一区间内的预测减半。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **DCN-V2 式交叉 + 目标注意力 + MMoE/PLE 多任务头**仍然是许多生产精排模型的骨干；例如 LinkedIn 的 LiRank 就描述了在生产中使用 Residual DCN（带注意力和残差的 DCN-V2）、对历史做 Transformer、稠密门控、校准和量化。
- 大型科技公司正越来越多地用对 GPU 友好、可扩展的骨干（RankMixer、Wukong、HSTU；第 4.4 章）取代 CPU 时代那一堆交互模块。
- **校准**在广告中是一个专门的生产组件，通常按分段、在线地进行。
- 多个头的**分数融合**目前基本仍靠手工调参；学习型 reward model 与 RL（第 4.6 章）是前沿方向。

## 常见误区 {#common-pitfalls-misconceptions}

- **只优化 AUC 而忘了校准。** AUC 对任何单调变换都不变；高估 2 倍不会改变 AUC，却会让广告价格翻倍。
- **习惯性地在 DIN 式注意力里加 softmax**：它会抹掉兴趣强度。
- **以为 embedding 越大模型越好。** 请检查奇异值谱；坍缩的 embedding 只会浪费内存。
- **用过期数据做校准**：校准必须跟随当前分布，所以要频繁刷新。
- **用任意权重训练多任务损失**，却不检查稀疏任务是否被淹没。

## 本章要点 {#key-takeaways}

- CTR 预估是针对稀疏 ID 的二分类；embedding 表占据了大部分参数。
- FM、Wide & Deep、DCN 和 DLRM 是越来越自动化的特征交互学习方式；FM 的两两交互项通过"和的平方减平方的和"只需 $O(kn)$。
- DIN 的目标注意力让用户表示依赖于候选——这是精排中序列建模的种子。
- 多任务模型（shared-bottom → MMoE → PLE）预测多种行为，再融合为一个分数。
- 只要预测值被当作概率或金额使用，校准（降采样修正、Platt、isotonic）就是必不可少的。

## 自测 {#check-your-understanding}

??? question "为什么 FM 能泛化到训练中从未同时出现过的特征对，而交叉特征的 LR 做不到？"
    LR 为每个交叉特征学一个独立的权重，所以没见过的特征对没有权重。FM 的特征对权重是 $\langle \mathbf{v}_i,\mathbf{v}_j\rangle$，而每个 $\mathbf{v}_i$ 是从所有涉及特征 $i$ 的特征对中学出来的，所以任意两个向量已训练好的特征，其交互都有定义。

??? question "3 层 DCN-V2 交叉网络最多能表示多少次的特征交互？"
    每个交叉层与 $\mathbf{x}_0$ 相乘一次，使最高多项式次数加一。从 1 次开始，经过 3 层后，输出包含最高 4 次的交互。

??? question "一个以 5% 的比例对负样本降采样训练出来的模型预测 0.5。校准后的概率是多少？"
    $p = 0.5/(0.5 + 0.5/0.05) = 0.5/10.5 \approx 0.048$。

??? question "为什么 MMoE 要给每个任务设一个单独的门？"
    这样每个任务都能选择自己的专家混合方式。相关的任务最终共享专家；冲突的任务可以使用不同的专家，从而减轻完全共享底层所带来的负迁移。

??? question "为什么 DIN 不用 softmax 对注意力权重做归一化？"
    Softmax 强制权重之和为 1，这会掩盖用户有多少相关行为。不归一化的权重让池化向量随兴趣强度增长，而这一点对点击具有预测力。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | FTRL-Proximal 在线逻辑回归，以及大规模广告 CTR 的工程经验——校准、节省内存、监控。 |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + 逻辑回归；数据新鲜度最重要；以 normalized entropy 为指标。 |
| :material-star: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — Cheng et al., 2016 | 线性"记忆"与 DNN"泛化"联合训练；Google Play。 |
| :material-book-open-variant: [Deep & Cross Network (DCN)](https://arxiv.org/abs/1708.05123) / [DCN V2](https://arxiv.org/abs/2008.13535) — Wang et al., 2017/2020 | 显式且高效的特征交叉层；V2 是一个很强的生产基线。 |
| :material-star: [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978) — Zhou et al. (Alibaba), 2017 | 目标注意力：按与候选广告的相关性给用户过去的行为加权。 |
| :material-star: [DLRM: Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) — Naumov et al. (Meta), 2019 | 参考架构及系统视角：embedding 表主导内存，需要模型并行。 |
| :material-book-open-variant: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) — Ma et al. (Google), KDD 2018 | MMoE 论文。 |
| :material-book-open-variant: [Recommending What Video to Watch Next: A Multitask Ranking System](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/) — Zhao et al. (Google), RecSys 2019 | YouTube 生产精排中的 MMoE 加浅层位置偏差塔。 |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | 实用的大型精排模型方案：Residual DCN、对历史做 Transformer、稠密门控、校准、量化、探索/利用。 |
| :material-book-open-variant: [On the Embedding Collapse When Scaling up Recommendation Models](https://arxiv.org/abs/2310.04400) — Guo et al., 2023 | 为什么单纯放大 embedding 不再有帮助；提出 multi-embedding。 |
