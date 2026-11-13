# 1.5 混合专家（Mixture-of-Experts）

!!! abstract "本章你将学到"
    - 为什么稀疏的**混合专家（MoE）**层能让总参数量与每个 token 的计算量脱钩。
    - **路由器（router）**如何选出 top-$k$ 个专家，以及 softmax 门控与 sigmoid 门控的区别。
    - **负载均衡**问题：推导 Switch Transformer 的辅助损失，理解容量因子与 token 丢弃，以及 DeepSeek-V3 的无辅助损失（auxiliary-loss-free）偏置均衡。
    - **细粒度专家**与**共享专家**，以及基于 all-to-all 通信的**专家并行（EP）**。
    - 如何从配置出发还原 DeepSeek-V3 的 671B 总参数 / 37B 激活参数，并写出一个可运行的 top-$k$ MoE 层。

## 为什么重要 {#why-this-matters-the-problem}

缩放定律（scaling law，见 [1.7](07-scaling-laws-and-data.md)）告诉我们，模型越大，损失越低。但在稠密（dense）Transformer 中，每个 token 都要用到全部参数，所以模型大一倍，训练和推理的单 token 成本也大一倍。混合专家打破了这种绑定：它把 FFN 换成许多并行的 FFN（“专家”），每个 token 只用其中的几个。这样一个模型可以容纳 671B 参数的知识，而每个 token 只花约 37B 模型的计算量。2024 年以来，几乎所有前沿开源模型（DeepSeek-V3、Qwen3-235B-A22B、Kimi K2、gpt-oss、Llama 4）都是 MoE，这一层如今已经和注意力一样标准。

## 直觉 {#intuition}

想象一家医院。稠密模型就像一位什么都得懂、每个病人都要亲自看的医生。MoE 则是一家有许多专科医生和一位分诊护士（**路由器**）的医院。每位病人（token）只会见到最相关的两位或八位专家，而不是全部专家。整家医院掌握的知识远多于任何一位医生，但每次就诊只花几次会诊的成本。

有两件事可能出问题。第一，护士可能把所有人都派给同一位热门专家，其他专家闲着没事干（**负载不均衡**：既浪费硬件，又让闲置的专家得不到训练）。第二，专家们可能分散在不同的楼里，来回转送病人很慢（专家并行中的**通信开销**）。MoE 的大部分工程工作，都是围绕这两个问题展开的。

```text
                 ┌─► 专家 1 ─┐
 token x ─► 路由器（打分）─► top-k ─┤   专家 2   （跳过）
                 └─► 专家 7 ─┘ ─► 加权求和 g1·E1(x) + g7·E7(x)  (+ 共享专家)
```

## 工作原理 {#how-it-works}

### 稠密计算与稀疏计算 {#dense-vs-sparse-compute}

设 $N_{\text{total}}$ 为全部参数量，$N_{\text{active}}$ 为处理单个 token 时实际用到的参数量。稠密模型中两者相等。在每层有 $E$ 个专家、每个专家大小为 $P_e$、每个 token 使用 $k$ 个专家的 MoE 中，FFN 部分对 $N_{\text{total}}$ 的贡献是 $E \cdot P_e$，但对 $N_{\text{active}}$ 只贡献 $k \cdot P_e$。每个 token 的训练计算量约为 $6 N_{\text{active}}$（见 [1.3](03-transformer.md)），而存放权重所需的内存则随 $N_{\text{total}}$ 增长。

### 路由器与 top-$k$ 门控 {#the-router-and-top-k-gating}

对于隐状态为 $x \in \mathbb{R}^d$ 的 token，权重矩阵为 $W_r \in \mathbb{R}^{d \times E}$ 的路由器为每个专家算出一个分数：$s = x W_r \in \mathbb{R}^E$。然后：

1. **分数 → 亲和度。** 要么对所有专家做 softmax，$p_i = e^{s_i} / \sum_j e^{s_j}$（Switch、Mixtral、Qwen3），要么对每个专家独立做 sigmoid，$p_i = \sigma(s_i)$（DeepSeek-V3）。
2. **选择**亲和度最大的 $k$ 个专家，得到集合 $\mathcal{T}(x)$。
3. **门控值。** 对 $i \in \mathcal{T}(x)$，$g_i = p_i / \sum_{j \in \mathcal{T}(x)} p_j$，其余为 0（在被选中的专家上重新归一化；有些模型会省略这一步）。
4. **合并：**

$$
y = \sum_{i \in \mathcal{T}(x)} g_i \, \mathrm{FFN}_i(x) \;+\; \sum_{j=1}^{E_s} \mathrm{FFN}^{\text{shared}}_j(x),
$$

其中对 $E_s$ 个共享专家的第二个求和只出现在带共享专家的设计中（见下文）。用大白话说：输出是被选中专家输出的加权混合。梯度通过门控值 $g_i$ 流回路由器，所以路由器能学到哪些专家对哪些 token 有用。top-$k$ 选择本身不可微；路由器只能通过它选中的那些专家的门控值来学习。

**softmax 与 sigmoid。** 用 softmax 时，专家之间是竞争关系：提高一个专家的亲和度会降低其他所有专家的亲和度，亲和度构成一个概率分布。用 sigmoid 时，每个亲和度都是一个独立的“该专家有多相关”的值，取值在 $(0, 1)$ 内，这和数量非常多的专家、以及下面要讲的基于偏置的均衡方法天然契合。对选中的集合做归一化之后，两种方式得到的门控值之和都为 1。

### 负载均衡与辅助损失 {#load-balancing-and-the-auxiliary-loss}

如果放任不管，路由会发生坍缩：早期稍好一点的专家会收到更多 token、得到更多梯度、变得更好，很快就只剩几个专家包揽一切。这既浪费容量，又因为专家分散在多块 GPU 上，让最忙的那块 GPU 成为所有人的瓶颈。

**Switch Transformer**（Fedus et al., 2021）加入了一个辅助损失。对一个含 $n$ 个 token 的批次 $\mathcal{B}$，为每个专家 $i$ 定义：

- $f_i$ = 被分发到专家 $i$ 的 token 比例（top-1 路由时，$f_i = \frac{1}{n}\sum_{x \in \mathcal{B}} \mathbb{1}[\arg\max p(x) = i]$），
- $P_i$ = 专家 $i$ 的平均路由概率，$P_i = \frac{1}{n}\sum_{x \in \mathcal{B}} p_i(x)$。

损失为

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \, P_i ,
$$

以一个很小的系数 $\alpha$ 加到语言建模损失上（Switch 用的是 $10^{-2}$）。

**为什么是这个形式？** 我们希望 $f$ 和 $P$ 都是均匀的（各为 $1/E$）。

- $f_i$ 是一个计数，没有梯度；$P_i$ 是可微的。把 $f$ 当作常数，则 $\partial \mathcal{L}_{\text{aux}} / \partial P_i = \alpha E f_i$：每个专家的路由概率被压低的力度，与该专家的过载程度成正比。越忙的专家，被压得越狠。
- 在均衡点，当路由器的概率与它的选择一致（$f \approx P$）时，求和变成 $\sum_i P_i^2$。由柯西–施瓦茨不等式，$\big(\sum_i P_i\big)^2 \le E \sum_i P_i^2$，又因为 $\sum_i P_i = 1$，得到 $\sum_i P_i^2 \ge 1/E$，当且仅当对所有 $i$ 都有 $P_i = 1/E$ 时取等号。所以均匀路由使损失最小。
- 系数 $E$ 把最小值归一化为 $\alpha \cdot E \cdot E \cdot (1/E)(1/E) = \alpha$，与专家数量无关，因此同一个 $\alpha$ 对任意 $E$ 都适用。

对于 top-$k$ 路由，$f_i$ 统计每个 token 的全部 $k$ 次分配（归一化使得 $\sum_i f_i = 1$）。这个系数是一种权衡：太小，路由会坍缩；太大，均衡项会和语言建模目标打架，损害质量。

### 容量因子与 token 丢弃 {#capacity-factor-and-token-dropping}

硬件喜欢固定大小的张量，所以经典的 MoE 实现给每个专家一块固定大小的缓冲区，即它的**容量**：

$$
\text{capacity} = \Big\lceil \frac{k \cdot n}{E} \cdot \mathrm{CF} \Big\rceil,
$$

其中 $n$ 是批次中的 token 数，$\mathrm{CF}$ 是**容量因子**（例如 1.0–1.25）。如果选择某个专家的 token 数超过了它的容量，多出来的 token 就会被**丢弃**：该专家不处理它们，它们只能通过残差连接穿过这一层。CF 越大，丢弃的 token 越少，但会把计算和内存浪费在填充（padding）上。现代实现越来越多地采用“无丢弃”（dropless）的 kernel，使用可变大小的分组（grouped GEMM），从而不丢任何 token；DeepSeek-V3 报告称，其训练和推理中都不丢弃 token。

### 细粒度专家与共享专家 {#fine-grained-and-shared-experts}

**DeepSeekMoE**（Dai et al., 2024）做了两项改动，2025 年的大多数 MoE 都采用了它们：

1. **细粒度专家。** 把每个专家拆成 $m$ 个更小的专家（隐藏维度除以 $m$），并路由到 $m$ 倍数量的专家。计算量不变，但可能的专家组合数会爆炸式增长。16 个专家取 top-2 时有 $\binom{16}{2} = 120$ 种组合；把每个专家拆成 4 个后，就有 64 个专家、取 top-8，组合数为 $\binom{64}{8} = 4{,}426{,}165{,}368$。每个 token 可以得到一个更有针对性的知识混合。
2. **共享专家。** 设置一个或几个*每个* token 都会用到的专家，放在路由器之外。它们负责承载通用知识（基础句法、常见模式），这样各个路由专家就不必各自重复学一遍，可以专注于专业化。

### 无辅助损失的均衡（DeepSeek-V3） {#auxiliary-loss-free-balancing-deepseek-v3}

DeepSeek-V3 用一个简单的控制回路取代了大部分辅助损失。每个专家 $i$ 有一个偏置 $b_i$，它**只在选择 top-$k$ 时**加到亲和度上：

$$
\mathcal{T}(x) = \operatorname{top-}k_i\big(p_i(x) + b_i\big), \qquad g_i = \frac{p_i(x)}{\sum_{j \in \mathcal{T}(x)} p_j(x)} \text{ for } i \in \mathcal{T}(x).
$$

门控值用的是原始亲和度 $p_i$，所以偏置改变的是*选谁*，而不是被选中的专家*贡献多少*。每个训练步之后，每个过载的专家的偏置减小一个步长 $\gamma$，每个欠载的专家的偏置增大 $\gamma$（DeepSeek-V3 报告在训练的大部分阶段取 $\gamma = 0.001$）。偏置不接收梯度，它是一个反馈控制器。由于损失里没有加入任何均衡项，语言建模的梯度不会被扭曲。DeepSeek-V3 只保留了一个非常小的序列级均衡损失，作为防止单个序列内部出现极端不均衡的保险。

### 专家并行与 all-to-all {#expert-parallelism-and-all-to-all}

一个 671B 参数的 MoE 放不进单块 GPU，而且它的大部分权重都在专家里。**专家并行（EP）**把不同的专家放在不同的 GPU 上。一个 MoE 层的工作流程如下：

1. 每块 GPU 为自己的 token 做路由。
2. **all-to-all 分发（dispatch）：** 每块 GPU 把每个 token 的隐状态发送给持有该 token 所选专家的那些 GPU。
3. 每块 GPU 在收到的 token 上运行本地的专家（每个专家一次批量矩阵乘法）。
4. **all-to-all 合并（combine）：** 结果被送回 token 所在的原 GPU，并按门控权重合并。

也就是说，前向传播中每个 MoE 层有两次 all-to-all 交换（反向传播中还有两次）。通信量随 $k$ 增长，而跨节点通信（互联速度比节点内慢）代价很高。因此 DeepSeek-V3 采用了**节点受限路由（node-limited routing）**：每个 token 的专家最多分布在 4 个节点上。让这部分通信与计算重叠，是 MoE 系统工程的重要内容（见 [1.8](08-training-at-scale.md)；推理部署见 [第 3 部分](../part3/index.md)）。

## 代码 {#code}

下面是一个可运行的 top-$k$ MoE 层，带可选的共享专家和 Switch 风格的辅助损失。它是对专家做循环（而不是对 token 循环），这是最简单且正确的实现；生产级 kernel 会按专家对 token 排序，再用分组矩阵乘法。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):                      # 一个小型 SwiGLU FFN
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, top_k=2, n_shared=0, aux_coef=1e-2):
        super().__init__()
        self.E, self.k, self.aux_coef = n_experts, top_k, aux_coef
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, d_ff) for _ in range(n_shared))

    def forward(self, x):                                 # x: [B, T, d]
        B, T, d = x.shape
        x = x.reshape(-1, d)                              # [N, d]，N = B*T 个 token
        probs = F.softmax(self.router(x), dim=-1)         # [N, E] 路由概率
        gate, idx = probs.topk(self.k, dim=-1)            # 各为 [N, k]
        gate = gate / gate.sum(-1, keepdim=True)          # 在被选中的专家上重新归一化
        out = torch.zeros_like(x)
        for e in range(self.E):                           # 对专家循环，而不是对 token 循环
            tok, slot = (idx == e).nonzero(as_tuple=True) # 哪些 token 选了专家 e
            if tok.numel():
                out.index_add_(0, tok, gate[tok, slot, None] * self.experts[e](x[tok]))
        for s in self.shared:                             # 共享专家处理每一个 token
            out = out + s(x)
        # Switch 风格的负载均衡损失：E * sum_i f_i * P_i
        f = F.one_hot(idx, self.E).float().sum(1).mean(0) / self.k   # 分配比例
        P = probs.mean(0)                                            # 平均路由概率
        aux = self.aux_coef * self.E * (f * P).sum()
        return out.view(B, T, d), aux, f

torch.manual_seed(0)
moe = MoE(d=64, d_ff=128, n_experts=8, top_k=2, n_shared=1)
y, aux, load = moe(torch.randn(4, 32, 64))
print(y.shape, round(aux.item(), 4), [round(v, 3) for v in load.tolist()])
```

**亲眼看看坍缩是怎么发生的。** 我们训练这一层去拟合一个随机线性映射，并在初始化时把路由器稍微推向专家 0，分别在不加和加辅助损失两种情况下运行：

```python
def run(aux_coef, steps=500):
    torch.manual_seed(1)
    m = MoE(d=32, d_ff=64, n_experts=8, top_k=2, aux_coef=aux_coef)
    with torch.no_grad():
        m.router.weight[0] += 0.15           # 初始化时让路由器偏向专家 0
    target = nn.Linear(32, 32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x = torch.randn(16, 32, 32) + 0.5
        y, aux, load = m(x)
        loss = F.mse_loss(y, target(x).detach()) + aux
        opt.zero_grad(); loss.backward(); opt.step()
    return [round(v, 2) for v in load.tolist()]

print("no aux  :", run(0.0))    # [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49, 0.0]
print("with aux:", run(1e-1))   # [0.12, 0.12, 0.13, 0.13, 0.14, 0.13, 0.12, 0.12]
```

没有这个损失时，两个专家包揽了全部分配，另外六个从未得到训练（具体数值在不同 PyTorch 版本间可能略有差异）。加上它之后，负载接近均匀（$1/8 = 0.125$）。

**无辅助损失的变体。** 把 DeepSeek-V3 的规则写成一个子类：sigmoid 亲和度、仅用于选择的不可学习偏置，以及基于符号函数的更新。

```python
class BiasBalancedMoE(MoE):
    """DeepSeek-V3 风格的路由：sigmoid 亲和度，偏置只用于选择。"""
    def __init__(self, *a, gamma=1e-3, **kw):
        super().__init__(*a, aux_coef=0.0, **kw)
        self.gamma = gamma
        self.register_buffer("bias", torch.zeros(self.E))

    def forward(self, x):
        B, T, d = x.shape
        x = x.reshape(-1, d)
        s = torch.sigmoid(self.router(x))                       # [N, E] 亲和度，取值在 (0, 1)
        _, idx = (s + self.bias).topk(self.k, dim=-1)           # 偏置影响“选谁”……
        gate = s.gather(-1, idx)                                # ……但不影响门控“数值”
        gate = gate / gate.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, gate[tok, slot, None] * self.experts[e](x[tok]))
        for sh in self.shared:
            out = out + sh(x)
        load = F.one_hot(idx, self.E).float().sum((0, 1))       # 每个专家收到的 token 数
        if self.training:                                       # 无梯度：这是一个控制回路
            self.bias += self.gamma * torch.sign(load.mean() - load)
        return out.view(B, T, d), torch.zeros(()), load / load.sum()
```

按上面同样的方式训练，取 `gamma=1e-2`（且不加辅助项），负载最终大致均衡（在我们的运行中，每个专家约在 0.09–0.2 之间）。

## 实例演算 {#worked-example}

**DeepSeek-V3：671B 总参数和 37B 激活参数是怎么来的？** 公开的配置如下：$L = 61$ 层，$d = 7168$；前 3 层是稠密的 SwiGLU FFN，$d_{ff} = 18{,}432$；其余 58 层是 MoE，含 256 个路由专家加 1 个共享专家，每个都是隐藏维度为 2048 的 SwiGLU，每个 token 激活 8 个路由专家；MLA 注意力，128 个头；词表 129,280（即所谓“128K”词表），输入输出嵌入不共享。

| 组成部分 | 计算 | 总参数 | 激活参数 |
|---|---|---|---|
| 一个专家 | $3 \times 7168 \times 2048$ | 44.0M | |
| MoE FFN | $58 \times 257 \times 44.0\text{M}$ | 656.5B | $58 \times 9 \times 44.0\text{M} = 23.0\text{B}$ |
| 稠密 FFN | $3 \times 3 \times 7168 \times 18{,}432$ | 1.19B | 1.19B |
| MLA 注意力 | $61 \times 187.1\text{M}$（投影见下） | 11.4B | 11.4B |
| 嵌入 + 输出层 | $2 \times 129{,}280 \times 7168$ | 1.85B | 1.85B |
| 路由器、归一化层 | $58 \times 256 \times 7168 + \dots$ | 0.11B | 0.11B |
| **合计** | | **≈ 671.0B** | **≈ 37.6B** |

每层 MLA 的投影为：query 下投影 $7168 \times 1536$，query 上投影 $1536 \times 128 \times (128 + 64)$，KV 下投影加 RoPE key $7168 \times (512 + 64)$，key/value 上投影 $512 \times 128 \times (128 + 128)$，输出投影 $128 \times 128 \times 7168$，合计 187.1M。结果与报告中的 671B 总参数、37B 激活参数吻合（报告里另有一个 14B 的多 token 预测（MTP）模块，算上它共 685B）。请注意：98% 的参数都在路由专家里，但每层 257 个专家中，每个 token 只会运行 9 个。因此每个 token 的训练计算量约为 $6 \times 37\text{B}$，大致相当于一个 37B 稠密模型的成本，却拥有 18 倍的参数量。

**容量。** 在某个 MoE 层，一个批次有 $n = 4096$ 个 token，$E = 64$ 个专家，$k = 2$，$\mathrm{CF} = 1.25$。容量 $= \lceil 2 \times 4096 / 64 \times 1.25 \rceil = 160$ 个 token/专家。如果有 200 次分配都指向同一个专家，则该专家有 40 个 token 被丢弃。

## 实践中的做法（2025–2026） {#in-practice-20252026}

| 模型 | 每个 MoE 层的专家数 | 激活数 | 共享专家 | 门控 / 均衡方式 |
|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 0 | softmax top-2；总 47B / 激活 13B |
| DeepSeek-V3 | 256 | 8 | 1 | sigmoid；基于偏置（无辅助损失）+ 极小的序列级损失 |
| Qwen3-235B-A22B | 128 | 8 | 0 | softmax top-8；全局批次（global-batch）均衡损失 |
| Kimi K2 | 384 | 8 | 1 | DeepSeek-V3 风格 |
| gpt-oss-120b | 128 | 4 | 0 | top-4 |

- **细粒度专家已成常态**：用数百个小专家配 top-8，而不是 8 个大专家配 top-2。
- **稀疏度在不断提高。** Kimi K2 总参数 1.04T、激活参数 32B；其报告指出，在激活参数固定的前提下，专家总数越多，损失还会继续下降。
- **在合适的粒度上做均衡。** Qwen3 在全局批次上而不是每个 micro-batch 上计算均衡损失，这让专家可以按领域专业化，而不是强行要求每个小批次都均匀。
- **并非每一层都是 MoE。** DeepSeek-V3 前 3 层保持稠密，Kimi K2 则是第 1 层，因为最前面几层的路由效果不好，而且早期层处理的是通用特征。

## 常见误区 {#common-pitfalls-misconceptions}

- **“671B 的 MoE 和 37B 的稠密模型一样便宜。”** 单 token 的计算量确实如此，但全部 671B 参数都得放在内存里；而且在小 batch 下，每个 token 读取的专家各不相同，所以起主导作用的是内存带宽和通信，而不是 FLOPs。
- **“专家按主题分工。”** 分析通常发现，专家更多是按 token 类型和句法模式来专业化的，而不是“生物学”这类人类可解释的主题。
- **忘了门控值。** 如果把专家输出直接求和而不乘门控权重，路由器就得不到梯度，永远学不会。
- **过度加大辅助损失权重。** 目标不是完美均衡，而是在可接受的均衡下获得好的语言建模损失。$\alpha$ 过大会损害质量。
- **用总参数量来比较 MoE 和稠密模型。** 应当同时比较激活参数量（计算）*和*总参数量（内存、知识容量）。

## 本章要点 {#key-takeaways}

- MoE 把 FFN 替换成 $E$ 个专家，加上一个每次激活其中 $k$ 个的路由器，使总参数量（知识、内存）与激活参数量（FLOPs）脱钩。
- 没有均衡机制，路由就会坍缩；Switch 损失 $\alpha E \sum_i f_i P_i$ 会压低过载的专家，并在均匀路由时取得最小值。
- DeepSeek-V3 基于偏置的均衡用反馈回路而非损失项来引导专家选择，避免干扰语言建模目标。
- 细粒度专家（许多小专家）加共享专家（始终开启）是 2025 年的默认配方。
- 专家并行每个 MoE 层需要两次 all-to-all 交换；瓶颈往往是通信而不是 FLOPs。

## 自测 {#check-your-understanding}

??? question "一个 MoE 有 40 层，每层 64 个专家，每个专家 50M 参数，采用 top-4 路由，非专家参数为 3B。总参数量和激活参数量各是多少？"
    专家：$64 \times 50\text{M} \times 40 = 128\text{B}$；总参数 $= 131\text{B}$。激活参数：$4 \times 50\text{M} \times 40 = 8\text{B}$，再加 3B，共 11B。

??? question "为什么 Switch 损失用乘积 $f_i P_i$，而不是直接用 $\sum_i f_i^2$？"
    $f_i$ 来自 argmax（一个计数），梯度为零。$P_i$ 是可微的，所以用乘积能让梯度流入路由器，并按每个专家当前的过载程度缩放。

??? question "在 DeepSeek-V3 的方案中，如果偏置也被用到门控值里，会出什么问题？"
    那样偏置就不只是改变选择，还会改变每个专家对输出的贡献，于是均衡控制器会直接扭曲模型的函数以及语言建模损失。把偏置排除在门控之外，就把均衡与输出隔离开了。

??? question "为什么单 token 计算量不变，细粒度专家仍然有帮助？"
    每个 token 可能的专家组合数急剧增长（在 DeepSeekMoE 的例子中，从 120 增长到超过 40 亿），所以每个 token 可以得到更有针对性的知识组合，每个专家也可以专业化得更精细。

??? question "使用专家并行时，为什么在激活 FLOPs 相同的情况下，top-8 路由可能比 top-2 更贵？"
    每个 token 的隐状态都必须发送给持有其某个专家的每一块 GPU。top-8 意味着每个 token 最多有 8 个目的地，而不是 2 个，all-to-all 的流量因此增加；节点受限路由是缓解办法之一。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017 | 最早的 top-k 门控 MoE，带负载均衡损失（基于 LSTM）。 |
| :material-book-open-variant: [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020 | 600B 规模的 MoE Transformer，跨设备的**专家并行**。 |
| :material-star: [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021 | top-1 路由、容量因子、上文推导的辅助损失，以及 bf16 稳定性技巧。MoE 工程最清晰的入门读物。 |
| :material-book-open-variant: [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) — Zoph et al., 2022 | 路由器 z-loss，以及关于 MoE 训练不稳定性和微调的实用指南。 |
| :material-star: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024 | 8 个专家、top-2，总 47B / 激活 13B；让开源 MoE 成为主流的模型。 |
| :material-star: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024 | **细粒度专家** + **共享专家**：DeepSeek-V2/V3、Kimi K2 等采用的设计。 |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | 总 671B / 激活 37B；**无辅助损失的负载均衡**、节点受限路由，以及让它得以训练的系统工程。 |
