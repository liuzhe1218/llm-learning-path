# 4.4 推荐模型的 Scaling

!!! abstract "本章你将学到"
    - 为什么经典 DLRM 不再随规模变大而提升：参数都在 embedding 里、硬件利用率低、交互太浅
    - 推荐系统里的"scaling law"是什么意思，以及要得到它，架构上必须改变什么
    - 深入理解 Generative Recommenders 与 HSTU 层：序列转导（sequential transduction）的重新表述、pointwise 聚合注意力，以及论文报告的结果
    - Wukong（堆叠 FM）、RankMixer（token mixing + 逐 token FFN）、MTGR（保留交叉特征）、LUM（三步范式）和 MARM（缓存）
    - 如何为精排模型做算力预算：每个候选的 FLOPs × 候选数 × QPS

## 为什么重要 {#why-this-matters-the-problem}

[第 1 部分](../part1/index.md)讲过，LLM 的损失随算力呈平滑的幂律下降，所以公司可以做规划："多 10 倍算力能换来这么多提升"。推荐系统却不是这样。多年来，主要的杠杆是增加特征和扩大 embedding 表，而单位硬件带来的收益越来越小。到 2020 年代初，工业精排模型有着 TB 级的 embedding，稠密网络却只有几千万参数，而把稠密部分做大往往毫无作用。

2024 年以来，一系列论文——最突出的是 Meta 的 **Generative Recommenders（HSTU）**——表明只要架构把算力花在正确的地方，推荐模型*是可以*像 LLM 一样 scale 的。这就是"大推荐模型"浪潮，它正在重塑 Meta、字节跳动、快手、美团、阿里巴巴等公司的精排。本章讲的就是到底改变了什么。

## 直觉 {#intuition}

可以把 DLRM 想象成一座图书馆：它有一个巨大的卡片目录（embedding 表），却只有一间很小的阅览室（稠密网络）。当每个物品都已经有了卡片，再多加卡片就不再有帮助；缺的是把信息组合起来的*思考*能力。可这间阅览室是用为 CPU 设计的、形状古怪的小模块搭起来的（手工交叉、大量小 MLP），所以 GPU 大多闲置，扩建阅览室也很昂贵。

scaling 的配方有三部分：

1. **把算力放进稠密网络**，而不是把内存放进 embedding。
2. **使用同质的、对 GPU 友好的模块**（类 Transformer 层、大矩阵乘法），让 FLOPs 真正被用起来（高 **MFU**，即 model FLOPs utilization）。
3. **给模型喂丰富的序列**（用户的原始行为历史），而不是手工聚合的特征，这样多出来的算力才有东西可学。

## 工作原理 {#how-it-works}

### 为什么 DLRM 会遇到瓶颈 {#why-dlrms-plateau}

1. **参数都在 embedding 里。** 一次 embedding 查找几乎不消耗 FLOPs。一个万亿参数的 DLRM 每个样本可能只做几十 MFLOPs。参数量是错误的坐标轴：在 LLM 里决定质量的是每个 token 的算力，而 DLRM 的算力非常少。
2. **MFU 低。** 许多小而异构的模块（特征交叉、门控、每个特征组一个小 MLP）受限于内存带宽。RankMixer 报告称它之前的生产精排模型的 **MFU 仅为 4.5%**：GPU 95% 的算力被浪费了。
3. **交互浅且固定。** FM/DCN 层只能捕捉低阶交互；堆更多层也不会持续有帮助。而且放大 embedding 可能触发 **embedding collapse**（第 [4.2](02-ranking-and-ctr-models.md) 章）。
4. **人工设计的特征限制了信号。** 像"用户过去 7 天在这个品类上的 CTR"这样的聚合特征，丢掉了更大模型本可利用的信息。

### 推荐系统里的"scaling law"指什么 {#what-scaling-law-means-for-recsys}

scaling law 是如下形式的经验拟合：

$$
\mathcal{L}(C) \approx \mathcal{L}_\infty + a\, C^{-\alpha},
$$

其中 $C$ 是训练算力（FLOPs），$\mathcal{L}$ 是质量指标（例如损失，或取负的 NDCG），$\mathcal{L}_\infty$ 是不可约的下限，$a, \alpha > 0$ 是拟合出来的常数。用话来说：算力每乘上一个倍数，就能换来可预测的提升。它对推荐系统重要的原因与 LLM 一样——容量规划——但有一点不同：**每次请求的推理成本是硬约束**，因为模型要对每次请求的数百个候选运行，而不是像生成那样每个 token 只运行一次。因此，结果往往以质量对*推理* FLOPs/样本来报告，而且必须满足固定的延迟预算。

早期的证据来自规模扩展到 0.8B 参数的基于 ID 的序列推荐模型，它们即使在数据受限的情况下也表现出幂律行为。突破在于在工业规模上证明了这一点。

### Generative Recommenders 与 HSTU {#generative-recommenders-and-hstu}

**重新表述。** HSTU 的论文（"Actions Speak Louder than Words"）把精排和召回都重新表述为对用户按时间排列的（物品，行为）事件流的**序列转导（sequential transduction）**。把历史写成 $(\Phi_0, a_0), (\Phi_1, a_1), \dots$，其中 $\Phi_i$ 是第 $i$ 个被展示或被互动的物品，$a_i$ 是用户对它的行为。

- **召回**变成"给定历史，预测下一个物品 $\Phi_{i+1}$"，只在用户对下一个物品的行为为正向的位置施加监督——这正是 SASRec 式的下一物品预测（第 [4.3](03-sequential-user-modeling.md) 章）。
- **精排**把物品与行为交错成同一个 token 序列 $\Phi_0, a_0, \Phi_1, a_1, \dots, \Phi_n$，并在每个物品 token 处预测紧随其后的行为。候选物品就是最后一个物品 token；模型输出它的预测行为（点击、点赞、观看……）。

大多数人工设计的特征都消失了：变化缓慢的类别特征（关注的创作者、人口统计信息）在发生变化的时间点被并入同一条时间序列，而数值聚合特征（计数器、历史 CTR）则被丢弃，前提是足够大的序列模型可以从原始行为中推断出它们。模型变成了一个*用户行为的生成模型*，像 LM 一样在行为流上训练——这就是"Generative Recommenders（GR）"这个名字的由来。

```text
 经典 DLRM 样本：      [用户特征 | 物品特征 | 交叉特征 | 池化历史]  ──► MLP ──► pCTR

 GR 精排序列：          Φ0  a0  Φ1  a1  Φ2  a2  ...  Φn   ?
                        │       │       │            │
 物品 token 处的目标：  └─► a0  └─► a1  └─► a2       └─► 对候选 Φn 的预测行为

 GR 召回：              Φ0  Φ1  Φ2 ... Φn ──► 下一个物品 Φn+1   （仅在行为为正向时计算损失）
```

一个用户的整个历史就是一条训练序列，每个位置都提供标签，所以一次前向传播就能在许多条曝光上同时训练——这与让 LM 预训练数据高效的"每个 token 都是目标"的效率是同一个道理。

**HSTU 层。** HSTU（Hierarchical Sequential Transduction Unit）是一个经过修改的 Transformer 块。设输入 $\mathbf{X} \in \mathbb{R}^{n\times d}$（$n$ 个 token）：

$$
\mathbf{U}, \mathbf{V}, \mathbf{Q}, \mathbf{K} = \text{Split}\big(\phi_1(f_1(\mathbf{X}))\big)
$$

$$
\mathbf{A}\mathbf{V} = \phi_2\big(\mathbf{Q}\mathbf{K}^\top + \mathbf{rab}^{p,t}\big)\,\mathbf{V}
$$

$$
\mathbf{Y} = f_2\big(\text{Norm}(\mathbf{A}\mathbf{V}) \odot \mathbf{U}\big)
$$

其中 $f_1, f_2$ 是线性投影，$\phi_1, \phi_2$ 是 SiLU 非线性，$\mathbf{rab}^{p,t}$ 是同时依赖位置 $p$ 与时间戳差 $t$ 的相对注意力偏置，$\odot$ 是逐元素乘法。用话来说：

- **Pointwise 聚合注意力。** 不用 softmax，而是让每个注意力权重各自独立地通过 SiLU（并除以一个常数归一化因子，例如序列长度）。Softmax 强制权重之和为 1，抹掉了相关的过去行为有*多少*；pointwise 权重则保留了这一强度信号——与 DIN 不做归一化的注意力是同样的道理（第 4.2 章）。论文还论证，在推荐这种不断有新物品到来的非平稳、流式词表中，pointwise 注意力更为稳健。
- **门控取代 FFN。** $\text{Norm}(\mathbf{AV})\odot\mathbf{U}$ 是一个乘性门，让这一层能够在注意力块内部形成特征交互（也就是 DCN 的交叉所做的事）；没有单独的大型前馈子层，因此每一层更便宜。
- **时间感知的偏置。** $\mathbf{rab}^{p,t}$ 既编码"多久之前"，也编码"多少步之前"。

**效率技巧。** 论文把架构与面向变长序列的稀疏感知 kernel、*随机长度*（训练时随机对长历史做子采样以降低成本），以及一个能在一次前向传播中为许多候选打分的推理算法（M-FALCON）配合使用：候选被追加到共享的用户历史之后，并用注意力掩码使每个候选能看到历史但看不到其他候选，历史部分的计算被缓存并在各个 micro-batch 间复用——这就是推荐领域里对应于[第 3 部分](../part3/index.md) KV cache 的做法。

**报告的结果**（来自摘要）：HSTU 在合成数据集和公开数据集上的 NDCG 比基线高出至多 65.8%，在长度为 8,192 的序列上比基于 FlashAttention2 的 Transformer 快 5.3 倍到 15.2 倍。基于 HSTU 的、有 1.5 万亿参数的 Generative Recommenders 把线上 A/B 指标提升了 12.4%，并已部署在多个展示位上；并且模型质量**随训练算力呈幂律 scaling，跨越三个数量级，直至 GPT-3/LLaMA-2 的规模**。最后这一点引发了"大推荐模型"浪潮。

后续工作（例如腾讯的"Scaling New Frontiers"）研究了 HSTU 的哪些组件驱动了 scaling，并在精排任务上做了评估。

### Wukong：堆叠的因子分解机 {#wukong-stacked-factorization-machines}

Wukong（Meta）走了另一条路：保留经典的特征交互视角，但让它能够 scale。每一层有一个**因子分解机块**（对当前这组 embedding 做所有两两点积，再投影回一组 embedding）和一个**线性压缩块**，并带残差连接。由于第 $l$ 层是把第 $l-1$ 层的输出相乘，而后者已经包含了两两交互，交互阶数会随深度呈几何级数增长（大约 $2^l$）。把层做得更高更宽，模型就能平滑地 scale。Wukong 报告称，它在模型复杂度跨越两个数量级、超过 100 GFLOP/样本的范围内仍保持 scaling law，而此前的架构做不到。

### RankMixer：对 GPU 友好的精排骨干 {#rankmixer-gpu-friendly-ranking-backbone}

RankMixer（字节跳动）用同质的、类 Transformer 的骨干取代了 CPU 时代那一堆交互模块：

1. **Tokenization。** 把数百个输入特征按含义分组为 $T$ 个特征 token，每个投影到维度 $D$：输入为 `[batch, T, D]`。
2. **多头 token mixing**（取代自注意力）。把每个 token 拆成 $H$ 个头；通过把每个 token 的第 $h$ 个头拼接起来，构造新的第 $h$ 个 token。这是一次无参数的重排，让信息能以 $O(T)$ 的代价在特征组之间流动，而不是注意力的 $O(T^2)$——并且 token 只有几十个，也用不着注意力的灵活性。
3. **逐 token 的 FFN。** 与 Transformer 不同，每个 token 位置都有*自己的* FFN 参数，因为各 token 是不同的特征子空间（用户、物品、上下文……），而不是可以互换的词。稀疏 MoE 变体可以进一步扩大规模。

全部是大型稠密矩阵乘法，所以硬件被高效利用。报告的结果：**MFU 从 4.5% 升到 45%**，精排模型参数量扩大 **100 倍**而推理延迟大致不变，一个 1B 稠密参数的 RankMixer 在推荐和广告场景中承接全量流量而不增加服务成本，使用户活跃天数提升 0.3%，App 内使用时长提升 1.08%。

### MTGR：保留交叉特征 {#mtgr-keep-the-cross-features}

纯 GR 模型会去掉手工设计的交叉特征（例如"用户对这个作者的历史 CTR"）。美团发现，这样做会显著损害效果，而 scaling 也无法弥补。**MTGR** 建立在 HSTU 之上，但保留了 DLRM 的特征（包括交叉特征）并将其作为 token；它通过把同一用户的所有候选合并成一个样本来压缩训练（**用户级压缩**），加入了 **Group-Layer Normalization**（对每个特征组分别归一化，因为各 token 处在不同的语义空间），以及一种动态掩码策略来避免信息泄露。它支持 DLRM 10–100 倍的计算复杂度而不显著增加成本，前向推理的每样本 FLOPs 达到 DLRM 的 **65 倍**，并已部署在美团的主流量上。

### LUM：三步范式 {#lum-a-three-step-paradigm}

阿里巴巴的 **Large User Model** 认为端到端 GR 牺牲了 DLRM 太多的实际优势，于是把工作拆成三步：

1. 在行为序列上**预训练**一个生成式大型用户模型（知识构建）；
2. 对它做**查询**——例如"这个用户接下来会在品类 X 中做什么？"——来提取用户兴趣知识；
3. 把这些输出作为特征**用于**现有的精排模型。

大模型运行在关键链路之外；精排模型依然廉价。LUM 报告称，随着规模扩大到 7B 参数，性能持续提升，并在线上 A/B 实验中取得收益。

### MARM：用缓存换算力 {#marm-trade-compute-for-cache}

快手的 **MARM** 从这样的观察出发：在推荐系统里，参数和数据是廉价的（他们报告每天有超过 500 亿条用户样本，模型超过 2,000 亿参数），而*上线时的 FLOPs* 才是昂贵的资源。MARM 缓存针对用户历史计算出的中间注意力结果，并在之后的请求中复用，这样就可以运行更深的序列模型，而只为增量部分付费。他们描述了一种"缓存 scaling law"：缓存的计算越多，质量越好。这就是把 KV cache 的思想应用到跨请求的场景。

### 各种方案对比 {#comparing-the-recipes}

| 方法 | 算力花在哪里 | 是否保留手工特征？ | 主要技巧 |
|---|---|---|---|
| HSTU / GR (Meta) | 物品/行为 token 交错的长序列 | 基本不保留 | pointwise 门控注意力；候选共享历史的前向计算 |
| Wukong (Meta) | 作用于特征 embedding 的深层 FM 交互堆叠 | 保留 | 交互阶数随深度增长 |
| RankMixer (字节跳动) | 作用于几十个特征 token 的逐 token FFN | 保留 | 无参数 token mixing；高 MFU；稀疏 MoE 变体 |
| MTGR (美团) | HSTU 式序列 + 作为 token 的 DLRM 特征 | 保留，含交叉特征 | 用户级压缩；group layer norm |
| LUM (阿里巴巴) | 离线运行的大型生成式用户模型 | 保留（在下游 DLRM 中） | 预训练 → 查询 → 作为特征输入 |
| MARM (快手) | 带缓存中间结果的更深序列模型 | — | 跨请求复用缓存的注意力结果 |

## 代码 {#code}

一个最小化的 HSTU 式层和一个 RankMixer 式块，附形状标注。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class HSTULayer(nn.Module):
    """Pointwise (SiLU) 注意力 + 门控，无 softmax，无独立 FFN。因果。"""
    def __init__(self, d, n_heads=2, max_len=512):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.f1 = nn.Linear(d, 4 * d)             # -> U, V, Q, K
        self.f2 = nn.Linear(d, d)
        self.norm_in, self.norm_av = nn.LayerNorm(d), nn.LayerNorm(d)
        self.rab = nn.Parameter(torch.zeros(n_heads, max_len, max_len))   # 位置偏置只在这里
    def forward(self, x):                         # x: [B, n, d]
        B, n, d = x.shape
        u, v, q, k = F.silu(self.f1(self.norm_in(x))).chunk(4, dim=-1)   # 各为 [B, n, d]
        split = lambda t: t.view(B, n, self.h, self.dh).transpose(1, 2)  # [B, h, n, dh]
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-1, -2) + self.rab[:, :n, :n]           # [B, h, n, n]
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
        a = F.silu(scores).masked_fill(~causal, 0.0) / n                 # pointwise，而非 softmax
        av = (a @ v).transpose(1, 2).reshape(B, n, d)                    # [B, n, d]
        return x + self.f2(self.norm_av(av) * u)                         # 用 U 做门控，加残差

class RankMixerBlock(nn.Module):
    """无参数 token mixing + 逐 token FFN。输入 [B, T, D]，D 能被 T 整除。"""
    def __init__(self, T, D, mult=4):
        super().__init__()
        self.T, self.D = T, D
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        # 每个 token 位置一个 FFN：权重 [T, D, mult*D] 和 [T, mult*D, D]
        self.w1 = nn.Parameter(torch.randn(T, D, mult * D) * D ** -0.5)
        self.w2 = nn.Parameter(torch.randn(T, mult * D, D) * (mult * D) ** -0.5)
    def mix(self, x):                             # H = T 个头，每个大小为 D/T
        B = x.size(0)
        heads = x.view(B, self.T, self.T, self.D // self.T)  # [B, token, head, D/T]
        return heads.transpose(1, 2).reshape(B, self.T, self.D)  # 新 token h = 所有 token 的第 h 个头
    def forward(self, x):                         # x: [B, T, D]
        x = self.norm1(x + self.mix(x))
        h = F.gelu(torch.einsum("btd,tdf->btf", x, self.w1))     # 逐 token 的权重
        return self.norm2(x + torch.einsum("btf,tfd->btd", h, self.w2))

x = torch.randn(8, 50, 64)                        # 8 个用户，50 个交错的物品/行为 token
print(HSTULayer(64)(x).shape)                     # torch.Size([8, 50, 64])
feats = torch.randn(8, 16, 256)                   # 16 个宽度为 256 的特征 token
print(RankMixerBlock(16, 256)(feats).shape)       # torch.Size([8, 16, 256])
```

这个 HSTU 示意代码省略了相对偏置中的时间戳部分，并使用了稠密的 $n\times n$ 偏置表；真实实现使用分桶的相对位置/时间偏置以及融合的变长 kernel。

下面是一个小工具函数，可以用你自己的数字重做下面的算力预算计算：

```python
def serving_gpus(flops_user, flops_per_cand, n_cand, qps, peak=1e15, mfu=0.4):
    """峰值时所需 GPU 数：每次请求 (用户部分算一次 + 候选部分对每个候选各算一次)。"""
    flops_per_s = qps * (flops_user + n_cand * flops_per_cand)
    return flops_per_s / (peak * mfu)

print(serving_gpus(0, 1e9, 1000, 1e5))               # 全部算力都在每个候选上  -> 250.0
print(serving_gpus(0, 1e9, 1000, 1e5, mfu=0.045))    # CPU 时代的 MFU          -> ~2222
print(serving_gpus(0.9e9, 0.1e9, 1000, 1e5))         # 90% 的 FLOPs 在用户侧   -> ~25
```

## 实例演算 {#worked-example}

**精排模型的算力预算。** 假设：

- 精排对每次请求的 $N = 1{,}000$ 个候选打分，
- 峰值流量为 $Q = 100{,}000$ 次请求/秒，
- 稠密模型对每个候选使用 $P = 500$M 参数，所以每个候选的前向 FLOPs $\approx 2P = 10^9$（1 GFLOP；即[第 1 部分](../part1/index.md)中的"2 × 参数量"规则）。

总服务算力 $= 2P \cdot N \cdot Q = 10^9 \cdot 10^3 \cdot 10^5 = 10^{17}$ FLOP/s。

一块稠密 BF16 峰值约 $10^{15}$ FLOP/s 的 GPU（大致相当于 H100 级别的加速卡），在 40% MFU 下可提供 $4\times10^{14}$ FLOP/s，所以我们需要 $10^{17}/(4\times10^{14}) = 250$ 块 GPU——仅仅是峰值时精排的前向传播。在 CPU 时代设计的 4.5% MFU 下，同样的模型大约需要 $2{,}200$ 块 GPU。**MFU 就是钱。**

现在把模型拆成每次请求只算一次的**用户部分**，以及对每个候选都要算的**候选部分**。如果 90% 的 FLOPs 在用户部分（例如一个长序列编码器），每个候选只有 10%（0.1 GFLOP）：

$$
\text{FLOP/s} = Q\cdot\big(0.9\times10^9 + N\cdot 0.1\times10^9\big) = 10^5\cdot(0.9\times10^9 + 10^{11}) \approx 1.01\times10^{16},
$$

约少了 10 倍。同样地，在相同预算下，你可以负担一个大约 10 倍大的用户侧模型。这就是为什么 HSTU 的 M-FALCON、MTGR 的用户级压缩、MARM 的缓存，以及 OneRec 的"编码一次，解码一个列表"（第 [4.6](06-end-to-end-generative-recommenders.md) 章）都在候选之间共享用户侧计算。

**训练算力。** 每天有 $10^{10}$ 个训练样本（每个是一次候选曝光），同样的 1 GFLOP 前向，加上"训练 ≈ 3 倍前向"的规则，一天的训练成本是 $3\times10^9\cdot10^{10} = 3\times10^{19}$ FLOPs，按 $4\times10^{14}$ FLOP/s 的有效算力计，约 $7.5\times10^4$ GPU 秒 $\approx 21$ GPU 小时/天数据——这还没算 embedding 查找和通信，而它们在实践中往往占主导。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **规模化的精排骨干已经上线**：Meta 的基于 HSTU 的 GR、字节跳动的 RankMixer、美团的 MTGR、阿里巴巴的 LUM、快手的 MARM。共同的主线是：在用户序列上做高 MFU 的稠密计算，并在候选之间共享用户侧计算。
- **两个流派并存**：丢弃手工特征的纯序列模型（HSTU），与保留 DLRM 特征和交叉并将其作为 token 的混合模型（MTGR、RankMixer）。两者都报告了 scaling。
- **大模型放在关键链路之外**（LUM、SUM 式上游用户模型），把 embedding 或知识输入给更便宜的在线精排模型。
- Meta 把它的 **Generative Ads Recommendation Model（GEM）** 描述为一个面向广告的大型基础模型，其知识被迁移到许多下游广告模型（参见延伸阅读中的工程博客与第 [4.8](08-ads-systems.md) 章）。
- 同样的 scaling 思路也延伸到召回（HSNN）和端到端生成（OneRec，第 4.6 章）。

## 常见误区 {#common-pitfalls-misconceptions}

- **数参数量而不是数 FLOPs。** "万亿参数"的 DLRM 在计算上可能很小。scaling 的结论应当注明每个样本的 FLOPs。
- **忽视 MFU。** 一个 FLOPs 更多但利用率只有 5% 的模型，可能比一个更大但利用率 45% 的模型更慢、更贵。
- **以为可以去掉所有手工特征。** 美团报告称去掉交叉特征损失很大；请在你自己的数据上检验。
- **在过窄的范围上拟合 scaling 曲线**（不到一个数量级）并外推。
- **忘记推理时的扇出**：一个每个样本很便宜的架构，在乘上 1,000 个候选 × 10^5 QPS 之后可能负担不起，除非共享用户侧的工作。

## 本章要点 {#key-takeaways}

- DLRM 遇到瓶颈，是因为参数在 embedding 里、算力很少且利用率低、交互很浅。
- scaling 需要在丰富序列之上的、算力密集、同质、对 GPU 友好的稠密模块。
- HSTU 把推荐重新表述为对（物品，行为）流的序列转导，使用 pointwise 门控注意力，展示了幂律 scaling 以及 12.4% 的线上收益。
- Wukong、RankMixer、MTGR、LUM 和 MARM 是互补的方案：堆叠 FM、带逐 token FFN 的 token mixing、保留交叉特征、离线运行大模型、缓存计算。
- 服务预算 = 每个候选的 FLOPs × 候选数 × QPS ÷ (峰值 × MFU)；要共享用户侧的工作。

## 自测 {#check-your-understanding}

??? question "为什么万亿参数的 DLRM 并不与"DLRM 很小"这一说法矛盾？"
    这些参数几乎全是 embedding 行；每个样本只触及其中少数几行，查找的 FLOPs 可以忽略不计。每个样本的算力——也就是质量随之 scale 的那个量——仍然很小。

??? question "为什么 HSTU 在注意力里用 SiLU 而不是 softmax？"
    Softmax 把权重归一化到和为 1，丢掉了用户对相关物品互动的强度和频次。Pointwise 的 SiLU 保留了这一强度信号，并且按论文所说，更适合非平稳的流式词表。

??? question "在 RankMixer 中，为什么每个 token 都有自己的 FFN，这与语言 Transformer 不同？"
    RankMixer 的 token 是不同的特征组（用户、物品、上下文……），处在不同的语义空间，而不是句子中可互换的位置。独立的 FFN 让每个子空间都能用自己的参数建模，而 token mixing 负责跨组的交互。

??? question "利用例题，如果 MFU 从 40% 降到 20%，需要多少块 GPU？"
    每块 GPU 的有效吞吐减半，变为 $2\times10^{14}$ FLOP/s，所以 $10^{17}/(2\times10^{14}) = 500$ 块 GPU。

??? question "LUM 的三步范式背后的主要思想是什么？"
    让大型生成式用户模型远离对延迟敏感的关键链路：先预训练它，再通过查询它来获取用户兴趣知识，并把结果作为特征输入现有的精排模型，在保留 DLRM 实用性的同时仍能受益于规模。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152) — Zhai et al. (Meta), ICML 2024 | 开启大推荐模型浪潮的论文：GR 重新表述、HSTU、幂律 scaling、线上 +12.4%。代码：[generative-recommenders](https://github.com/meta-recsys/generative-recommenders)。 |
| :material-star: [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) — Zhang et al. (Meta), 2024 | 堆叠的因子分解机，在跨越两个数量级的算力范围内持续 scale。 |
| :material-book-open-variant: [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351) — Zhang et al., 2023 | 针对基于 ID 的序列推荐模型、至多 0.8B 参数的经验 scaling law，包括数据受限的情形。 |
| :material-book-open-variant: [Scaling New Frontiers: Insights into Large Recommendation Models](https://arxiv.org/abs/2412.00714) — Guo et al. (Tencent), 2024 | 对 HSTU scaling 的系统研究：哪些组件重要，以及向精排的迁移。 |
| :material-new-box: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) — Zhu et al. (ByteDance), 2025 | Token mixing + 逐 token FFN / 稀疏 MoE；MFU 4.5% → 45%，参数量 100 倍而延迟几乎不变。 |
| :material-new-box: [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) — Han et al., 2025 | 保留交叉特征的 HSTU 式 GR；group-layer-norm；每样本约 65 倍 FLOPs。 |
| :material-new-box: [Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model (LUM)](https://arxiv.org/abs/2502.08309) — Yan et al. (Alibaba), 2025 | 预训练 → 查询 → 输入 DLRM；scale 到 7B。 |
| :material-new-box: [MARM: Memory Augmentation and Scalable Complexity](https://arxiv.org/abs/2411.09425) — Lv et al. (Kuaishou), 2024 | 缓存中间注意力结果——跨请求的"KV cache"思想。 |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | LinkedIn 实用的大型精排模型方案。 |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | 把召回阶段扩展到双塔点积之外。 |
| :material-new-box: [Meta's Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) — Meta engineering blog, 2025 | 一个面向广告精排的大型基础模型的业界介绍文章。 |
