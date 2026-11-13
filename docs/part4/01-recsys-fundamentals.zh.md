# 4.1 推荐系统基础

!!! abstract "本章你将学到"
    - 如何精确地表述推荐问题，以及显式反馈与隐式反馈的区别
    - 工业系统为什么采用多阶段级联（召回 → 粗排 → 精排 → 重排），以及各阶段如何由延迟预算决定规模
    - 双塔召回模型如何用 in-batch sampled softmax 训练（含推导与代码），又如何用 ANN 检索（HNSW、IVF-PQ）上线服务
    - 如何手算 Recall@K、NDCG、AUC、GAUC 和 normalized entropy，以及为什么离线提升不一定能变成线上提升
    - 为什么推荐系统处在反馈回路中，会受到位置偏差/曝光偏差的影响，并且（通常）以单个 epoch 做在线训练

## 为什么重要 {#why-this-matters-the-problem}

推荐系统每秒要回答数百万次同一个问题：*给定这个用户，此刻应该展示哪几个物品？* 候选库里可能有数十亿个物品（每一个上传过的短视频、电商平台的每一件商品、每一条在投广告），用户期待在远不到一秒的时间内得到响应，而系统只能从自己过去选择展示的内容里学习。

现代推荐系统之所以长成现在这个样子——embedding 表、级联、双塔模型、流式训练，以及后面章节要讲的生成式模型——都源自三个约束：

1. **候选集规模。** 你不可能对每次请求都在十亿个物品上跑一个大神经网络。
2. **延迟。** 整条链路的端到端预算通常在一百毫秒量级甚至更低，每个阶段只能分到其中一小段。
3. **从自己的决策中学习。** 只有被展示过的物品才有标签，所以数据天然带有上一版模型的偏差。

本章从第一性原理出发搭建经典架构。后面的章节会替换其中的某些部分，所以你需要先清楚每个部分是做什么的。

## 直觉 {#intuition}

想象一座大图书馆，馆员对每位读者只有几秒钟。她不可能给每位读者把每本书的简介都读一遍。于是她会：

1. 走到几个符合读者大致兴趣的书架前（**召回，retrieval**：便宜、以召回率为目标，留下几千本书）；
2. 扫一眼封面，剔除明显不合适的（**粗排，pre-ranking**：轻量模型，剩下几百本）；
3. 结合她对读者的全部了解，仔细斟酌剩下的书（**精排，ranking**：昂贵的模型，给每个候选打分）；
4. 把最后挑出的几本排好顺序，避免全是同一位作者的书，并放进图书馆收了推广费的书目（**重排，re-ranking**：多样性、业务规则、广告竞价）。

每个阶段都在用精度换速度。便宜的阶段不能丢掉好物品（高**召回率**）；昂贵的阶段要把它们排好（头部要有高**精确率**）。

## 工作原理 {#how-it-works}

### 推荐问题 {#the-recommendation-problem}

设 $\mathcal{U}$ 为用户集合，$\mathcal{I}$ 为物品集合。对于上下文 $c$（时间、设备、页面、如有的话还包括 query）下的用户 $u \in \mathcal{U}$，我们想要一个打分函数 $s(u, i, c)$，使得按 $s$ 排序后，用户最看重的物品排在最前面。实践中，"价值"用可观测行为的预测概率来近似，例如 $P(\text{click} \mid u, i, c)$、期望观看时长，或者 $P(\text{purchase} \mid u, i, c)$。

**显式反馈**是对偏好的直接表达：1–5 星评分、点赞。它干净但稀少（大多数用户从不打分）。**隐式反馈**则由行为推断而来：曝光、点击、停留时长、完播、点赞、分享、购买、跳过。它量大，但有噪声且是单边的：点击是弱正向证据，而*没有*点击既可能是"不感兴趣"，也可能只是"没看到"。几乎所有工业系统都从隐式反馈中学习，所以大多数模型都被训练成针对曝光的**二分类或多任务分类器**（"这个被展示的物品有没有被点击/观看/购买？"）。

### 为什么要用级联：候选规模与延迟预算 {#why-a-cascade-candidate-sizes-and-latency-budgets}

假设精排模型对每个（用户，物品）对需要 10 MFLOPs，而我们有 $10^9$ 个物品。对所有物品打分，*每次请求*就是 $10^{16}$ FLOPs，在每秒数千次请求的情况下根本不可能。级联通过把算力花在刀刃上来解决这个问题：

| 阶段 | 输入规模 | 输出规模 | 模型 | 典型目标 |
|---|---|---|---|---|
| 召回（候选生成） | $10^6$–$10^{10}$ | $10^3$–$10^4$ | 双塔 + ANN、item-to-item 图、热门榜，多路合并 | 高召回、低成本 |
| 粗排 | $10^3$–$10^4$ | $10^2$–$10^3$ | 小型 DNN，或带更丰富特征的双塔 | 低成本过滤 |
| 精排 | $10^2$–$10^3$ | 同左，逐个打分 | 带用户历史与交叉特征的重型多任务网络（第 [4.2](02-ranking-and-ctr-models.md) 章） | 精准排序、校准的概率 |
| 重排 | $10^2$ | $\sim 10$ | 规则、多样性模型、广告竞价（第 [4.8](08-ads-systems.md) 章） | 整页质量、业务约束 |

这些数字只是数量级，不是标准；每家公司都会各自调整。关键的结构性事实是：**召回无法细致地建模用户–物品之间的特征交互**，因为它必须在毫秒内搜索一个巨大的集合。这就是召回模型要采用下面这种特殊的"双塔"结构的原因。

### 双塔召回 {#two-tower-retrieval}

**双塔**（two-tower，也叫 dual-encoder）模型有一个用户塔 $f_\theta$ 和一个物品塔 $g_\phi$，各自是一个把特征映射成 $d$ 维向量的神经网络：

$$
\mathbf{u} = f_\theta(\text{user features}) \in \mathbb{R}^d, \qquad
\mathbf{v}_i = g_\phi(\text{item } i \text{ features}) \in \mathbb{R}^d, \qquad
s(u,i) = \mathbf{u}^\top \mathbf{v}_i .
$$

用话来说：用户和物品被嵌入到同一个空间，分数就是两者的点积。关键性质在于物品向量不依赖于用户，所以我们可以离线算好所有 $\mathbf{v}_i$ 并放进索引，请求到来时只需算一个用户向量，再去找它的最近邻。

**把训练看作极端分类。** 把"用户接下来会与哪个物品发生互动？"看成对所有物品的分类问题。物品 $i$ 的 softmax 概率为

$$
P(i \mid u) = \frac{\exp(s(u,i)/\tau)}{\sum_{j \in \mathcal{I}} \exp(s(u,j)/\tau)},
$$

其中 $\tau > 0$ 是温度。对一个正样本对 $(u, i^+)$，损失为 $-\log P(i^+ \mid u)$。分母要对所有物品求和，代价太高，所以我们用**采样**来近似它。

**In-batch 负样本。** 取一个包含 $B$ 个正样本对 $(u_b, i_b)$（$b = 1..B$）的 batch。计算用户向量 $\mathbf{U} \in \mathbb{R}^{B \times d}$ 和物品向量 $\mathbf{V} \in \mathbb{R}^{B \times d}$，以及 logit 矩阵 $\mathbf{S} = \mathbf{U}\mathbf{V}^\top / \tau \in \mathbb{R}^{B \times B}$。对第 $b$ 行，对角元 $S_{bb}$ 是正样本，其余 $B-1$ 个物品（别的用户的正样本）充当负样本：

$$
\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B} \log \frac{\exp(S_{bb})}{\sum_{k=1}^{B} \exp(S_{bk})}.
$$

这其实就是第 $b$ 行标签为 $b$ 的交叉熵——和 CLIP 或检索增强语言模型所用的对比学习目标是同一个损失。

**热门偏差与 logQ 修正。** In-batch 负样本并不是均匀采样：一个物品出现在 batch 里的概率与它作为正样本的频率成正比，所以热门物品作为负样本被过度代表，会被压得过狠。采样 softmax 的理论指出：若负样本取自提议分布 $Q(j)$，则对完整 softmax 的"极限无偏"估计需要使用修正后的 logit

$$
S^{c}_{bk} = S_{bk} - \log Q(i_k),
$$

其中 $Q(i_k)$ 是物品 $i_k$ 出现在 batch 中的概率（用流式频次计数来估计）。用话来说：减去对数采样概率，这样被频繁采到的物品每次出现时受到的惩罚就更小。这就是 YouTube 使用的"采样偏差修正"双塔方案。

??? info "为什么减去 log Q 是对的（简要推导）"
    完整 softmax 的分母是 $Z = \sum_j e^{s_j}$。如果我们从 $j \sim Q$ 中抽取负样本，那么 $\mathbb{E}_{j\sim Q}[e^{s_j}/Q(j)] = \sum_j e^{s_j} = Z$，也就是说，用 $1/Q(j)$ 对每个采样项做重要性加权，就能得到分母的无偏估计。用 $1/Q(j)$ 给一项加权，等价于在它的 logit 上加 $-\log Q(j)$，因为 $e^{s_j}/Q(j) = e^{s_j - \log Q(j)}$。

### 近似最近邻（ANN）检索 {#approximate-nearest-neighbour-ann-search}

上线服务时，我们需要从 $N$ 个物品向量中找出 $\mathbf{u}^\top\mathbf{v}_i$ 最大的前 $K$ 个。精确检索每次请求要 $O(Nd)$；ANN 用少量召回率换来数量级的提速。

- **IVF（倒排文件）。** 用 k-means 把物品向量聚成 $C$ 个簇。查询时，把 $\mathbf{u}$ 与 $C$ 个聚类中心比较，选出最近的 `nprobe` 个簇，只对其中的物品打分。开销从 $N$ 降到大约 $C + \texttt{nprobe}\cdot N/C$ 次点积。
- **PQ（乘积量化）。** 把每个向量切成 $m$ 个子向量，并用每个子空间中一个小码本里最近中心的编号来替换各子向量（例如 256 个中心 → 1 字节）。一个 128 维 float32 向量（512 字节）就变成 $m$ 字节。点积通过小型查找表计算。**IVF-PQ** 把两者结合起来，是十亿级 GPU 检索的主力方案（Faiss）。
- **HNSW（层次化可导航小世界图）。** 构建一张图，每个向量连向它的近邻，上面再叠加更稀疏的"快速通道"层。搜索从最顶层出发，贪心地朝查询点走，然后下降到更稠密的层。它的行为类似于建立在近邻图上的跳表，以更多内存为代价，得到大约对数级的搜索开销和较高的召回率。

要记住，ANN 索引是独立于模型训练构建的。当模型持续更新时，物品向量必须重新建索引，而索引可能是过期的——这也是第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章中生成式召回的动机之一。

### 离线指标 {#offline-metrics}

设模型对每个用户返回一个排序列表。定义 $\text{rel}_k \in \{0,1\}$（或分级）为第 $k$ 位（从 1 开始计）物品的相关性。

- **Recall@K** $= \dfrac{\#\{\text{relevant items in top } K\}}{\#\{\text{relevant items}\}}$。主要的召回指标：我们有没有把好物品留下来？
- **DCG@K** $= \sum_{k=1}^{K} \dfrac{2^{\text{rel}_k}-1}{\log_2(k+1)}$，**NDCG@K** $= \text{DCG@K} / \text{IDCG@K}$，其中 IDCG 是理想排序下的 DCG。它奖励把相关物品放在*靠前的位置*；$\log_2(k+1)$ 这个折扣意味着第 1 位最重要。对二值相关性，$2^{\text{rel}}-1 = \text{rel}$。
- **AUC**（ROC 曲线下面积）= 随机抽一个正样本，其得分高于随机抽一个负样本的概率。它只度量排序，忽略校准。
- **GAUC**（group AUC）= 按用户分别计算 AUC，再加权平均（权重通常是该用户的曝光数或点击数）。全局 AUC 可能因为对*用户*排序而虚高（活跃用户点击更多），而精排模型从不需要做这件事——它只需要在*一次请求内部*给物品排序。GAUC 去掉了这一干扰。
- **LogLoss** $= -\frac{1}{n}\sum_{t=1}^n [y_t \log p_t + (1-y_t)\log(1-p_t)]$，其中标签 $y_t \in\{0,1\}$，预测值为 $p_t$。它对校准敏感。
- **Normalized entropy（NE）** $= \text{LogLoss} / H(\bar p)$，其中 $H(\bar p) = -[\bar p\log\bar p + (1-\bar p)\log(1-\bar p)]$ 是始终预测背景率 $\bar p$ 时的 LogLoss。NE < 1 表示优于平凡预测器；越低越好。Facebook 的广告论文让它流行起来，因为原始 LogLoss 取决于基础 CTR。

### 在线指标与离线–在线差距 {#online-metrics-and-the-offlineonline-gap}

离线指标是基于*旧*系统展示内容的日志算出来的。上线后，新模型会改变展示内容，进而改变用户行为。最终的裁判是 **A/B 实验**：把用户随机分流，分别提供旧模型和新模型的服务，再比较业务指标，如 CTR、观看时长、日活、留存（例如 7 日回访）、千次展示收入（RPM），以及护栏指标（延迟、投诉、多样性）。一次成功上线带来的线上收益，相对值往往很小（时长上的零点几个百分点），但在大规模下价值巨大。

### 反馈回路、位置偏差与曝光偏差 {#feedback-loops-position-bias-and-exposure-bias}

训练数据由上一版模型产生。这带来：

- **曝光偏差：** 从未被展示的物品永远得不到标签，所以模型无法学到它们其实不错。热门物品获得更多曝光、更多正样本，再带来更多曝光（"富者愈富"）。
- **位置偏差：** 展示在顶部的物品被点击得更多，*正是因为*它们在顶部。朴素的模型会学到"第 1 位放的东西就是好东西"。常见的修正方法是：训练时把位置作为特征输入，而在上线时把它固定为某个值（例如第 1 位）；或者训练一个单独的浅层"位置塔"，推理时丢弃它的输出。
- **探索**（展示一些得分不确定的物品）是持续学习所必需的；bandit 式探索和新内容加权都是标配做法。

### 在线 / 流式训练与单 epoch 现象 {#online-streaming-training-and-the-one-epoch-phenomenon}

用户兴趣、物品和热点每天都在变，所以工业模型在新鲜日志流上**持续训练**，每隔几分钟到几小时就推送到线上。由此产生两个后果：

1. 每个样本通常只被看**一次**。这不仅仅是算力上的选择：深度 CTR 模型存在**单 epoch 现象（one-epoch phenomenon）**——在同一份数据上训练第二个 epoch 时，测试效果会在 epoch 2 一开始就急剧下降。阿里的一项研究把它归因于：非常稀疏的 ID 特征的 embedding 层，在配合收敛很快的优化器（如 Adam）时，会记住训练样本。
2. 评估是**渐进式**的：用第 $t$ 天训练，在第 $t+1$ 天评估，绝不跨时间打乱。

## 代码 {#code}

一个完整的双塔模型，带 in-batch softmax 和 logQ 修正，在合成数据上训练，其中每个用户有一个隐藏的"口味"簇。

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

N_USERS, N_ITEMS, N_CLUSTERS, D = 1000, 5000, 20, 32
user_cluster = torch.randint(0, N_CLUSTERS, (N_USERS,))
item_cluster = torch.randint(0, N_CLUSTERS, (N_ITEMS,))
item_pop = torch.distributions.Pareto(1.0, 1.2).sample((N_ITEMS,))  # 重尾分布的热门度

# P[c, i] ∝ 若物品 i 属于簇 c，则为其热门度，否则为 0   -> [N_CLUSTERS, N_ITEMS]
P = (item_cluster[None, :] == torch.arange(N_CLUSTERS)[:, None]).float() * item_pop[None, :]

def sample_batch(B):
    """正样本 (用户, 物品) 对：每个用户从自己所属簇中挑一个热门物品。"""
    users = torch.randint(0, N_USERS, (B,))
    items = torch.multinomial(P[user_cluster[users]], 1).squeeze(1)   # [B]
    return users, items

class Tower(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.emb = nn.Embedding(n, 64)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d))
    def forward(self, ids):                      # ids: [B]
        return F.normalize(self.mlp(self.emb(ids)), dim=-1)   # [B, d]，单位范数

user_tower, item_tower = Tower(N_USERS, D), Tower(N_ITEMS, D)
opt = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=1e-2)
item_count = torch.ones(N_ITEMS)                 # 用于 log Q 的流式频次估计
tau, B = 0.05, 256

for step in range(1500):
    users, items = sample_batch(B)
    item_count.index_add_(0, items, torch.ones(B))
    logQ = torch.log(item_count[items] / item_count.sum())       # [B]
    U, V = user_tower(users), item_tower(items)                  # 各为 [B, d]
    logits = U @ V.T / tau - logQ[None, :]                       # [B, B]，按列做 logQ 修正
    # 屏蔽意外重复：同一物品同时是两行的正样本时，不应被当作负样本
    dup = (items[:, None] == items[None, :]) & ~torch.eye(B, dtype=torch.bool)
    logits = logits.masked_fill(dup, float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(B))              # 第 b 行的标签是第 b 列
    opt.zero_grad(); loss.backward(); opt.step()

# 对所有物品做精确（暴力）检索，评估 Recall@50
with torch.no_grad():
    users, items = sample_batch(1000)
    scores = user_tower(users) @ item_tower(torch.arange(N_ITEMS)).T   # [1000, N_ITEMS]
    top = scores.topk(50, dim=1).indices
    hit_cluster = (item_cluster[top] == user_cluster[users][:, None]).float().mean()
    recall = (top == items[:, None]).any(1).float().mean()
print(f"loss {loss.item():.3f}  Recall@50 {recall:.3f}  top-50 中同簇占比 {hit_cluster:.2f}")
```

值得注意的几点：损失就是对 $B\times B$ 矩阵做普通的 `cross_entropy`；物品向量做了归一化，所以点积就是余弦相似度；在生产环境中，暴力 `topk` 会被一个 ANN 索引（例如 Faiss IVF-PQ 或 HNSW）取代，该索引建立在 `item_tower(all items)` 之上。

## 实例演算 {#worked-example}

**NDCG 与 Recall。** 某用户在测试集中有 3 个相关物品。模型的 top-5 列表的相关性为 $[1, 0, 1, 0, 0]$（二值）。

- Recall@5 $= 2/3 \approx 0.667$。
- DCG@5 $= \frac{1}{\log_2 2} + \frac{1}{\log_2 4} = 1 + 0.5 = 1.5$。
- 理想列表把 3 个相关物品全放在最前面：IDCG@5 $= \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} = 1 + 0.631 + 0.5 = 2.131$。
- NDCG@5 $= 1.5 / 2.131 \approx 0.704$。

如果模型把第 2 位和第 3 位互换（相关性 $[1,1,0,0,0]$），则 DCG $= 1 + 0.631 = 1.631$，NDCG $\approx 0.765$：召回率相同，排序更好，NDCG 更高。

**AUC。** 五条曝光，得分与标签为 $(0.9, 1), (0.8, 0), (0.7, 1), (0.4, 0), (0.2, 0)$。共有 $2 \times 3 = 6$ 个（正，负）样本对。得分 0.9 的正样本胜过全部三个负样本（3 对）；得分 0.7 的正样本胜过 0.4 和 0.2，但输给 0.8（2 对）。AUC $= 5/6 \approx 0.833$。

**GAUC。** 用户 A 有 100 条曝光，AUC 为 0.70；用户 B 有 300 条曝光，AUC 为 0.60。按曝光数加权的 GAUC $= (100\cdot 0.70 + 300 \cdot 0.60)/400 = 0.625$。

**Normalized entropy。** 背景 CTR $\bar p = 0.02$：
$H(0.02) = -(0.02\ln 0.02 + 0.98\ln 0.98) = 0.0782 + 0.0198 = 0.0980$ nats。
一个 LogLoss 为 $0.0850$ 的模型，其 NE $= 0.0850/0.0980 \approx 0.867$，即相比始终预测 2%，不确定性降低了 13.3%。

**延迟预算。** 假设端到端预算为 150 ms：召回约 20 ms（多路并行，每路是一次 ANN 查询），在 5,000 个候选上做粗排约 10 ms，在 500 个候选上做精排约 40 ms，重排约 10 ms，其余留给特征获取和网络开销。对 500 个候选、40 ms 的预算，精排模型给每个候选打分的摊销时间必须远低于 0.1 ms——这只有靠在 GPU/CPU 上做 batching，并且每次请求只计算一次用户侧特征（而不是每个候选算一次）才能做到。（这些是示意性数字，并非某家公司的实际数据。）

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **召回是多路的。** 生产系统会合并来自双塔 ANN、item-to-item 共现图、关注/订阅列表、热门池的候选，并且越来越多地加入对语义 ID 做解码的生成式召回器（第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章）。
- **超越点积。** 双塔模型只能表达用户与物品之间的点积。Meta 的 HSNN 在层次化物品索引之上，用更丰富的可学习交互模块取而代之，同时保持开销随候选池规模呈次线性增长，并已部署用于广告召回。
- **级联本身正受到挑战。** 端到端生成式推荐（第 [4.6](06-end-to-end-generative-recommenders.md) 章）用一个模型直接生成最终列表，消除了各自独立训练的阶段之间的错配。
- **指标：** NE/LogLoss 和 GAUC 仍然是精排的标准离线关卡；召回看 Recall@K；上线决策则靠带留存与使用时长指标的 A/B 实验。

## 常见误区 {#common-pitfalls-misconceptions}

- **"AUC 越高，推荐越好。"** 基于日志数据的 AUC 忽略了从未展示过的物品，也忽略校准；模型可能 AUC 上升而线上变差。
- 离线评估时**跨天打乱数据**会泄露未来信息，高估模型质量。一定要按时间划分。
- **忘记 logQ 修正**会使基于 in-batch 训练的召回器对热门物品推荐不足。
- **在不考虑位置和曝光的情况下把未点击当作真正的负样本。**
- **像视觉/NLP 那样训练多个 epoch**：深度 CTR 模型通常在第一个 epoch 之后就会严重过拟合。
- **比较物品索引新鲜度不同的模型**：真正的瓶颈可能是昨天构建的 ANN 索引，而不是模型。

## 本章要点 {#key-takeaways}

- 之所以有级联，是因为候选池巨大而延迟很小：便宜的高召回阶段为昂贵的高精度阶段供给候选。
- 双塔模型之所以能做召回，是因为物品向量与用户无关，可以建 ANN 索引；它们用 in-batch sampled softmax 加 logQ 修正来训练。
- 召回看 Recall@K；精排看 NDCG、AUC/GAUC 和 NE/LogLoss；最终由 A/B 实验裁决。
- 日志由上一版模型产生，因此偏差（位置、曝光、热门度）和探索是一等公民级别的问题。
- 工业模型在数据流上做在线训练，通常只训练一个 epoch。

## 自测 {#check-your-understanding}

??? question "为什么精排模型的结构不能用于召回？"
    精排模型会计算用户与物品特征之间丰富的交互（交叉特征、目标注意力），所以每个（用户，物品）对都需要完整的一次前向传播。在 $10^9$ 个物品上，这代价太高。双塔模型把交互限制为点积，因此物品向量可以预先算好，并通过 ANN 在次线性时间内检索。

??? question "在 batch 大小为 4 的 in-batch softmax 中，每个正样本有多少个负样本？标签向量是什么？"
    每一行有 $B - 1 = 3$ 个负样本（其他行的正样本物品）。logit 是一个 $4\times 4$ 矩阵，标签是 $[0, 1, 2, 3]$：第 $b$ 行的正确列是 $b$。

??? question "对相关性 [0, 1, 1]，且用户恰好有 2 个相关物品，计算 NDCG@3。"
    DCG $= 0 + 1/\log_2 3 + 1/\log_2 4 = 0.631 + 0.5 = 1.131$。IDCG $= 1 + 0.631 = 1.631$。NDCG $= 1.131/1.631 \approx 0.693$。

??? question "为什么评估精排模型时通常更偏好 GAUC 而不是 AUC？"
    全局 AUC 还会奖励"区分重度用户（点击很多）和轻度用户"的能力，而精排模型根本不需要做这件事，因为它每次只为一个用户给物品排序。GAUC 在每个用户内部计算 AUC 再取平均，度量的恰好是一次请求内部的排序质量。

??? question "什么是单 epoch 现象？它对训练流水线意味着什么？"
    深度 CTR 模型在对同一数据进行第二遍训练的一开始，测试效果就急剧下降，这与稀疏 ID 的 embedding 层和收敛很快的优化器有关。因此流水线是对按时间排序的数据流做单遍训练，而不是反复迭代多个 epoch。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — Covington et al., RecSys 2016 | 经典的两阶段（候选生成 + 排序）深度推荐系统；把召回表述为对视频的极端多分类问题。 |
| :material-star: [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) — Yi et al. (Google), RecSys 2019 | 带流式频次估计来做 logQ 修正的 in-batch softmax 双塔方案。 |
| :material-book-open-variant: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2016 | 大多数基于图的 ANN 索引背后的 HNSW 算法。 |
| :material-book-open-variant: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) — Johnson, Douze & Jégou, 2017 | Faiss 论文：十亿规模下 GPU 上的 IVF-PQ（[代码](https://github.com/facebookresearch/faiss)）。 |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | 以 normalized entropy 为指标；数据新鲜度最重要。 |
| :material-book-open-variant: [Towards Understanding the Overfitting Phenomenon of Deep CTR Models](https://arxiv.org/abs/2209.06053) — Zhang et al. (Alibaba), 2022 | 解释了单 epoch 现象：CTR 模型在第 2 个 epoch 一开始就严重过拟合，主因是 embedding 层。 |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | 通过层次聚类和更丰富的交互模块，把召回扩展到双塔点积之外，已部署用于广告。 |
