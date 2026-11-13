# 4.5 语义 ID 与生成式召回

!!! abstract "本章你将学到"
    - 为什么随机物品 ID 在十亿量级会出问题（内存、更新频繁、冷启动），以及什么是语义 ID（semantic ID）
    - 从零讲起的向量量化与残差量化（residual quantization），附一个残差编码的数值例子
    - RQ-VAE 如何训练（损失函数、直通估计器），附可运行的最小实现，以及如何处理 ID 冲突
    - TIGER：一个解码下一个物品语义 ID 的 seq2seq 模型，以及束搜索（beam search）与基于前缀树（trie）的约束解码（含代码）
    - 语义 ID 如何用在排序模型里（YouTube），以及生成式召回与稠密召回的取舍（LIGER、COBRA、GRID）

## 为什么重要 {#why-this-matters-the-problem}

此前所有模型都用**随机 ID** 表示物品，每个物品在 embedding 表里占一行。规模上去之后，这有三个代价：

1. **内存。** 每个物品一行：数十亿行，往往是模型里最大的一部分（见第 [4.2](02-ranking-and-ctr-models.md) 章）。
2. **更新频繁与冷启动。** 五分钟前刚上传的视频，embedding 是全新的、没训练过的。在积累到交互之前，它的内容对模型毫无帮助。哈希冲突还会把不相关的物品混在一起。
3. **没有共享结构。** 两个几乎一样的物品（重复上传的视频、两个卖家的同款商品）各学各的。没有办法用已知的“零件”去“拼写”一个新物品。

LLM 的做法是用一个小的子词（subword）词表，任何单词都能由子词组合出来。**语义 ID** 对物品做的是同一件事：每个物品变成一个由离散码组成的短元组，例如 `(17, 203, 5)`，它由物品的*内容* embedding 导出，并且相似物品共享前缀。物品被“拼写”成 token 序列之后，Transformer 就可以像 LLM 生成单词那样，一个 token 一个 token 地**生成**下一个物品——这就是**生成式召回**（generative retrieval）。

## 直觉 {#intuition}

想想邮政编码。像 `94107` 这样的编码并不是随机的：第一位指向一个大区域，后面几位依次缩小到城市、街区。相邻的两个地址共享很长的前缀。新盖的房子一出生就有一个有意义的编码，因为它的位置是已知的。

语义 ID 就是物品在 embedding 空间里的邮政编码。第 1 层大致说明它是*哪一类*物品（美食视频），第 2 层细化（意大利菜），第 3 层继续细化（意面教程），最后一个码用来区分近似重复的物品。模型预测“下一个物品以码 17 开头”，就已经把几十亿物品缩小到了一个区域；而且即使是训练中从没见过的物品，只要它的内容把它放在这个区域里，模型也能做到这一点。

## 工作原理 {#how-it-works}

### 向量量化（VQ） {#vector-quantization-vq}

**码本**（codebook）是一组 $K$ 个向量 $\{\mathbf{e}_1,\dots,\mathbf{e}_K\}\subset\mathbb{R}^d$。对向量 $\mathbf{z}\in\mathbb{R}^d$ 做量化，就是用离它最近的码字替换它：

$$
c = \arg\min_{k} \lVert \mathbf{z} - \mathbf{e}_k\rVert^2, \qquad \hat{\mathbf{z}} = \mathbf{e}_c .
$$

通俗地说：把向量吸附到 $K$ 个原型中最近的一个，只记下下标 $c$。$K$ 个码字只能区分 $K$ 个组，除非 $K$ 非常大，否则重构误差 $\lVert\mathbf{z}-\hat{\mathbf{z}}\rVert$ 会很大。k-means 正是学习这种码本的算法。

### 残差量化（RQ） {#residual-quantization-rq}

想用小码本获得精细的分辨率，就**反复量化误差**。使用 $D$ 个码本 $\mathcal{C}^{(1)},\dots,\mathcal{C}^{(D)}$，每个有 $K$ 个码字：

$$
\mathbf{r}_0 = \mathbf{z},\qquad
c_l = \arg\min_k \lVert \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_k\rVert^2,\qquad
\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}^{(l)}_{c_l},\qquad l = 1..D,
$$

$$
\hat{\mathbf{z}} = \sum_{l=1}^{D}\mathbf{e}^{(l)}_{c_l}, \qquad \text{semantic ID} = (c_1, c_2, \dots, c_D).
$$

通俗地说：第 1 层选最接近的粗粒度原型；第 2 层量化第 1 层没量化好的部分；依此类推。下标元组是从粗到细的；$D$ 个大小为 $K$ 的码本可以寻址 $K^D$ 个格子（例如 $256^3 \approx 1.7\times10^7$），却只需存 $D\cdot K$ 个向量。

### RQ-VAE {#rq-vae}

直接对原始内容 embedding 做 RQ 也能用，但 **RQ-VAE** 会把被量化的空间也一并学出来。它包含：

- 一个编码器 $E$，把物品的内容 embedding $\mathbf{x}$（例如由句子编码器对标题 + 描述编码得到，或来自多模态模型）映射成隐向量 $\mathbf{z} = E(\mathbf{x})$；
- 上面介绍的残差量化器，码本可学习；
- 一个解码器 $G$，重构 $\hat{\mathbf{x}} = G(\hat{\mathbf{z}})$。

训练损失为

$$
\mathcal{L} = \underbrace{\lVert \mathbf{x}-\hat{\mathbf{x}}\rVert^2}_{\text{reconstruction}}
+ \sum_{l=1}^{D}\Big(\underbrace{\lVert \text{sg}[\mathbf{r}_{l-1}] - \mathbf{e}^{(l)}_{c_l}\rVert^2}_{\text{move codeword to data}}
+ \beta\,\underbrace{\lVert \mathbf{r}_{l-1} - \text{sg}[\mathbf{e}^{(l)}_{c_l}]\rVert^2}_{\text{commit encoder to codeword}}\Big),
$$

其中 $\text{sg}[\cdot]$ 是停止梯度（stop-gradient），$\beta$（通常取 0.25）是承诺项（commitment）的权重。通俗地说：把输入重构好；把被选中的码字拉向它所代表的残差；同时不让编码器漂离码字。

$\arg\min$ 不可微，所以用**直通估计器**（straight-through estimator）把解码器的梯度传给编码器：把 $\mathbf{z} + \text{sg}[\hat{\mathbf{z}}-\mathbf{z}]$ 作为解码器输入，它在前向传播中等于 $\hat{\mathbf{z}}$，而对 $\mathbf{z}$ 的梯度为 1。

两个实践中的问题：

- **码本坍塌（codebook collapse）**：只有少数几个码字被用到。常见的补救办法是用第一个 batch 上的 k-means 初始化码本，并重新初始化没被用到的（“死”）码。
- **有些系统干脆跳过 VAE**，直接对内容 embedding 做残差 k-means（有时还加上均衡约束，让每个码对应的物品数量差不多）。

### 冲突 {#collisions}

两个物品（尤其是近似重复的物品）可能得到同一个元组 $(c_1,\dots,c_D)$。生成式模型必须能把每个 ID 映射回物品，所以冲突需要处理：

- **追加一个消歧 token**：TIGER 在最后加一个额外的码，只用来给冲突的物品编号（$0, 1, 2, \dots$），让 ID 唯一。
- **增加层数或码本大小**，或者使用考虑冲突的、均衡的分配方式来量化。
- **接受冲突**，把一个语义 ID 当作一个装有少量物品的小桶，由下游模型再做排序。

### TIGER：基于语义 ID 的生成式召回 {#tiger-generative-retrieval-with-semantic-ids}

**TIGER**（Google，2023）是这一方向的参考设计：

1. 构建语义 ID：用句子编码器对物品文本编码得到内容 embedding → RQ-VAE → 每个物品得到几个码（再加上消歧码）。
2. 把用户历史表示为其各物品语义 ID 的拼接：`[c1(i1) c2(i1) c3(i1) c1(i2) c2(i2) c3(i2) …]`。词表就是 $D\cdot K$ 个码 token（每一层有各自的 token 区间）加上用户 token。
3. 训练一个 Transformer 编码器–解码器：编码器读取历史，解码器用交叉熵逐个码地生成下一个物品的语义 ID，和 seq2seq 机器翻译完全一样。
4. 推理时用**束搜索**（beam search）得到若干高概率的 ID，映射回物品，作为召回候选返回。

TIGER 报告称，在多个数据集上显著超过了当时最先进的序列推荐模型，并且对没有任何历史交互的物品召回更好——因为新物品的语义 ID 来自内容，模型即使在训练中从没见过它，也能把它生成出来。这个思路可以追溯到 **DSI**，它训练了一个 seq2seq 模型在搜索场景里直接解码文档 ID。

### 束搜索与约束解码 {#beam-search-and-constrained-decoding}

自回归地生成 $D$ 个码时，模型给出一个 ID 的概率为 $P(c_1,\dots,c_D\mid\text{history}) = \prod_{l=1}^{D} P(c_l\mid c_{<l},\text{history})$。**束搜索**每一步保留 $B$ 个最优的部分序列，把每个序列用所有可能的下一个码扩展，再按累计对数概率保留最好的 $B$ 个（LLM 中的解码策略见[第 3 部分](../part3/index.md)）。

并不是每个元组都对应真实物品：$K^D$ 个格子里大部分是空的。不加约束的解码会把束浪费在**无效 ID** 上。办法是为所有有效 ID 建一棵**前缀树**（prefix trie）：在第 $l$ 步，只允许那些能把当前前缀扩展成某个现有物品前缀的码（其余码的对数概率置为 $-\infty$）。新物品通过把它的 ID 插入前缀树加入；下线的物品则删除——索引变化不需要重训模型。同一棵前缀树还可以强制执行业务过滤（地区、年龄分级、有货），只要用符合条件的子集来建树即可。

### 排序模型里的语义 ID {#semantic-ids-inside-ranking-models}

语义 ID 即使*不做生成*也有用。YouTube 在一个线上**排序**模型里，把随机的视频 ID 特征换成了语义 ID 片段的 embedding。直接使用内容 embedding 会丢掉记忆能力（热门物品需要自己专属的参数）；把完整语义 ID 当作一个特征又会丢掉泛化能力。效果最好的做法是把 ID 的**子片段**哈希进 embedding 表——可以是码的 n-gram，更好的是用 SentencePiece（和 LLM 的分词器一样）学出来的片段。粗粒度片段被大量相似物品共享（泛化）；长片段几乎是唯一的（记忆）。论文报告称，在不牺牲整体质量的前提下，对新物品和长尾物品的泛化有所提升。

### 生成式召回 vs 稠密召回 {#generative-vs-dense-retrieval}

| | 稠密召回（双塔 + ANN） | 生成式召回（语义 ID + 解码） |
|---|---|---|
| 物品存储 | 每个物品一个向量 + ANN 索引 | 码本 + 有效 ID 的前缀树；不需要逐物品向量 |
| 冷启动 | 物品塔需要内容特征 | 天然支持：新物品的 ID 来自内容 |
| 模型与索引的一致性 | 索引单独重建，可能过期 | 模型*本身就是*索引（外加一棵前缀树） |
| 细粒度记忆 | 强（逐物品向量） | 较弱；冲突会模糊近似重复的物品 |
| 推理开销 | 一次用户塔前向 + ANN 查询 | $D$ 步解码 × 束宽 |
| 多样性控制 | 通过 ANN 查询 / 多路查询 | 束搜索天然产生不同前缀 |

Meta 的 **LIGER** 在受控条件下对比了两者，发现生成式召回可能落后于稠密召回，冷启动是差异之一；它提出了一种把稠密召回融入生成式召回的混合方案。百度的 **COBRA** 走的是由粗到细的路线：先生成稀疏的语义 ID，再以它为*条件*生成稠密向量，端到端训练，并在推理时把束搜索和最近邻得分结合起来（BeamFusion）；它报告了在日活超过 2 亿的广告平台上的线上收益。

## 代码 {#code}

下面是一个最小的 RQ-VAE，用来自层级聚类的合成“内容 embedding”做实验，这样可以检查同一簇的物品是否共享 ID 前缀。

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# 合成内容 embedding：8 个主题 x 8 个子主题，4000 个物品，维度 32
X_DIM, N = 32, 4000
topic_c = torch.randn(8, X_DIM) * 3
sub_c = torch.randn(8, 8, X_DIM)
topic = torch.randint(0, 8, (N,)); sub = torch.randint(0, 8, (N,))
x = topic_c[topic] + sub_c[topic, sub] + 0.2 * torch.randn(N, X_DIM)

class RQVAE(nn.Module):
    def __init__(self, x_dim=32, z_dim=16, levels=3, K=16, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(), nn.Linear(64, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, x_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, z_dim) * 0.1) for _ in range(levels)])
        self.beta = beta

    def quantize(self, z):                        # z: [B, z_dim]
        residual, z_hat, codes, vq_loss = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:                 # cb: [K, z_dim]
            idx = torch.cdist(residual, cb).argmin(dim=1)          # [B] 最近的码字
            e = cb[idx]                                            # [B, z_dim]
            vq_loss = vq_loss + F.mse_loss(e, residual.detach()) \
                              + self.beta * F.mse_loss(residual, e.detach())
            z_hat = z_hat + e
            residual = residual - e.detach()      # 下一层量化剩下的部分
            codes.append(idx)
        return z_hat, torch.stack(codes, dim=1), vq_loss          # codes: [B, levels]

    def forward(self, x):
        z = self.enc(x)
        z_hat, codes, vq_loss = self.quantize(z)
        z_st = z + (z_hat - z).detach()           # 直通估计器
        return F.mse_loss(self.dec(z_st), x) + vq_loss, codes

model = RQVAE()
with torch.no_grad():                             # 数据驱动初始化，逐层进行
    r = model.enc(x[torch.randperm(N)[:1024]])
    for cb in model.codebooks:                    # 每个码本用采样得到的残差初始化
        cb.copy_(r[torch.randperm(len(r))[:cb.shape[0]]])
        r = r - cb[torch.cdist(r, cb).argmin(1)]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(3000):
    batch = x[torch.randint(0, N, (256,))]
    loss, _ = model(batch)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    _, codes = model(x)
ids = [tuple(c) for c in codes.tolist()]
from collections import Counter
print("loss", round(loss.item(), 3), "| unique IDs", len(set(ids)), "of", N,
      "| largest collision bucket", Counter(ids).most_common(1)[0][1])
# 同一主题的物品，是否比随机物品对更常共享第一个码？
i, j = torch.randint(0, N, (2, 20000))
same_topic = topic[i] == topic[j]
share1 = codes[i, 0] == codes[j, 0]
print(f"P(same first code | same topic) = {share1[same_topic].float().mean():.2f}, "
      f"| different topic = {share1[~same_topic].float().mean():.2f}")
```

同一主题的物品几乎总是共享第一个码，不同主题的物品几乎从不共享：前缀是带语义的。你还会看到大量**冲突**（4,000 个物品只有几百个唯一 ID）：在这份合成数据里，同一个子主题（共 64 个）内的物品只相差很小的噪声，属于近似重复，会落进同一个格子。这正是 TIGER 要追加消歧码的场景。

下面是在语义 ID 上做前缀树约束的束搜索。这里的“解码器”是一个返回对数概率的替身函数；真实系统里它是以用户历史为条件的 Transformer 解码器。

```python
import torch
torch.manual_seed(0)
LEVELS, K, BEAM = 3, 16, 4
valid_ids = {tuple(torch.randint(0, K, (LEVELS,)).tolist()) for _ in range(200)}  # 物品库

# 构建前缀树：前缀元组 -> 允许的下一个码的集合
trie = {}
for sid in valid_ids:
    for l in range(LEVELS):
        trie.setdefault(sid[:l], set()).add(sid[l])

W = torch.randn(LEVELS, K, K)                     # 玩具解码器的参数
def next_logprobs(prefix):
    """decoder(history, prefix) -> log P(next code) 的替身函数。返回 [K]。"""
    prev = prefix[-1] if prefix else 0
    return torch.log_softmax(W[len(prefix), prev], dim=-1)

def beam_search(constrained=True):
    beams = [((), 0.0)]                           # (前缀, 累计对数概率)
    for l in range(LEVELS):
        candidates = []
        for prefix, score in beams:
            lp = next_logprobs(prefix)
            if constrained:                       # 屏蔽不通向任何真实物品的码
                mask = torch.full((K,), float("-inf"))
                mask[list(trie.get(prefix, []))] = 0.0
                lp = lp + mask
            for c in range(K):
                if lp[c] > float("-inf"):
                    candidates.append((prefix + (c,), score + lp[c].item()))
        beams = sorted(candidates, key=lambda t: -t[1])[:BEAM]
    return beams

for flag in (False, True):
    out = beam_search(flag)
    n_valid = sum(sid in valid_ids for sid, _ in out)
    print(f"constrained={flag}: {n_valid}/{BEAM} beams are real items ->", [s for s, _ in out])
```

不用前缀树时，大多数束是不对应任何物品的元组；用了之后，每个束都是真实物品。新增一个物品只需要一次前缀树插入。

## 实例演算 {#worked-example}

**手算残差编码。** 取一个二维隐向量 $\mathbf{z}=(0.8,\ 0.3)$，三个码本，每个 4 个码字：

| 层 | 码字（下标：向量） |
|---|---|
| 1 | 0: $(1,0)$, 1: $(0,1)$, 2: $(-1,0)$, 3: $(0,-1)$ |
| 2 | 0: $(0.2,0.2)$, 1: $(0.2,-0.2)$, 2: $(-0.2,0.2)$, 3: $(-0.2,-0.2)$ |
| 3 | 0: $(0.05,0)$, 1: $(0,0.05)$, 2: $(-0.05,0)$, 3: $(0,-0.05)$ |

- **第 1 层。** 从 $(0.8,0.3)$ 出发的平方距离：到 $(1,0)$：$0.04+0.09=0.13$；到 $(0,1)$：$0.64+0.49=1.13$；到 $(-1,0)$：$3.24+0.09=3.33$；到 $(0,-1)$：$0.64+1.69=2.33$。选 **0**。残差 $\mathbf{r}_1 = (-0.2,\ 0.3)$。
- **第 2 层。** 从 $(-0.2,0.3)$ 出发的距离：到 $(0.2,0.2)$：$0.16+0.01=0.17$；到 $(0.2,-0.2)$：$0.16+0.25=0.41$；到 $(-0.2,0.2)$：$0+0.01=0.01$；到 $(-0.2,-0.2)$：$0.25$。选 **2**。残差 $\mathbf{r}_2=(0,\ 0.1)$。
- **第 3 层。** 从 $(0,0.1)$ 出发的距离：到 $(0,0.05)$：$0.0025$（最小）。选 **1**。残差 $\mathbf{r}_3=(0,\ 0.05)$。

语义 ID $=(0, 2, 1)$。重构 $\hat{\mathbf{z}} = (1,0)+(-0.2,0.2)+(0,0.05) = (0.8,\ 0.25)$；误差 $\lVert\mathbf{r}_3\rVert = 0.05$，而只用第 1 层时误差是 $0.36$。相近的物品 $(0.85, 0.2)$ 的第 1 层码同样是 0（共享前缀），而 $(-0.7, 0.4)$ 则以码 2 开头。

**容量。** 3 层 × 每层 256 个码，共有 $256^3 \approx 16.8$M 个格子，但解码器的输出层每步只有 256 个类别（总共 768 个码 embedding），而不是对比如 1 亿个物品 ID 做 softmax。把 1 亿个物品存成 64 维 fp32 稠密向量需要 $100\text{M}\times 64\times 4\text{ B} = 25.6$ GB；存它们的语义 ID，每个只要 3 字节，共 300 MB，再加上码本。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- 语义 ID 如今已是标准构件：生成式召回（TIGER 式）、排序中的 ID 特征（YouTube），以及端到端生成式推荐（OneRec，第 [4.6](06-end-to-end-generative-recommenders.md) 章）和基于 LLM 的推荐（PLUM、LC-Rec；第 [4.7](07-llms-in-recommendation.md) 章）的物品词表。
- 分词器（tokenizer）越来越多地量化那些已经融合了内容*和*协同信号的 embedding（例如与用户行为对齐的多模态 embedding，如快手的 QARM），而不是纯文本 embedding。
- 稀疏 + 稠密的混合设计（COBRA、LIGER）是弥补生成式召回精度差距的常见答案。
- Snap 的 **GRID** 开源框架系统地对分词器、模型架构和解码方式做了消融，发现许多被忽视的架构组件会显著改变结果——把它当作起点，而不是从头复现论文。

## 常见误区 {#common-pitfalls-misconceptions}

- **码本坍塌**：要按层检查码本利用率；某一层 256 个码只用到 10 个，就是浪费容量。
- **把语义 ID 当成热门物品的“免费午餐”**：它们可能丢失记忆能力；要保留一些物品专属的容量（更长的片段、额外 token，或带逐物品特征的排序阶段）。
- **线上服务时不加约束解码**：无效 ID 浪费束，最后什么都召回不到。
- **重训了分词器却没重训推荐模型**：码的含义变了；分词器的更新必须做版本管理并协调发布。
- **只在热门（warm）物品上评估生成式召回**：它的主要优势往往体现在冷启动和长尾物品上。

## 本章要点 {#key-takeaways}

- 语义 ID 把物品“拼写”成对其内容 embedding 做残差量化得到的、由粗到细的码元组；相似物品共享前缀。
- RQ-VAE = 编码器 + 残差码本 + 解码器，用重构损失、码本损失和承诺损失训练，并用直通梯度回传。
- TIGER 训练一个 seq2seq 模型来解码下一个物品的语义 ID；基于前缀树的约束束搜索保证产出的都是有效物品，并让索引可编辑。
- 语义 ID 片段还能通过平衡记忆与泛化，改善排序中的 ID 特征。
- 生成式召回在内存、冷启动以及模型与索引一致性上占优；稠密召回在细粒度精度上占优；混合方案兼得两者。

## 自测 {#check-your-understanding}

??? question "为什么残差量化比总码字数相同的单一码本分辨率更细？"
    一个含 $D\cdot K$ 个码字的单一码本只能区分 $D\cdot K$ 个格子。含 $D$ 个、每个 $K$ 个码字的残差量化把各层的选择组合起来，最多可寻址 $K^D$ 个格子，因为每一层都在量化前面各层留下的误差。

??? question "直通估计器在 RQ-VAE 中起什么作用？"
    前向传播时，它把量化后的向量送入解码器；反向传播时，把解码器的梯度当作量化是恒等映射一样，直接拷贝给编码器的输出，这样尽管 argmin 不可微，编码器依然能被训练。

??? question "为什么 TIGER 能推荐训练数据里从没出现过的物品？"
    它的语义 ID 是由内容算出来的，位于码空间中一个模型已经通过相似物品学到过的区域；即使这个具体物品从没作为训练目标出现过，解码器也能生成该元组。

??? question "如何在不重训的前提下把新上传的物品加入生成式召回？"
    计算它的内容 embedding，经过（冻结的）分词器得到语义 ID，再把这个 ID 插入约束解码所用的有效 ID 前缀树即可。

??? question "给出一个生成式召回在热门物品上可能不如稠密召回的原因。"
    热门物品受益于物品专属的、被记住的参数；语义 ID 在相似物品之间共享参数，还可能发生冲突，会模糊逐物品稠密向量所能捕捉的细微差别。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991) — Tay et al., 2022 | seq2seq 模型记住整个语料并直接解码文档 ID——生成式召回的源头。 |
| :material-book-open-variant: [Autoregressive Image Generation using Residual Quantization (RQ-VAE)](https://arxiv.org/abs/2203.01941) — Lee et al., 2022 | 由粗到细码的残差量化——标准的语义 ID 分词器。 |
| :material-star: [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) — Rajput et al. (Google), NeurIPS 2023 | 用 RQ-VAE 构造语义 ID，再用 Transformer 解码下一个物品的语义 ID。生成式召回的参考实现。 |
| :material-star: [Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) — Singh et al. (Google/YouTube), 2023 | 用语义 ID 的 n-gram / SentencePiece embedding 取代线上排序模型里的视频 ID。 |
| :material-book-open-variant: [Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER)](https://arxiv.org/abs/2411.18814) — Yang et al. (Meta), 2024 | 生成式召回何时不如稠密召回，以及一种混合方案。 |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | 先生成稀疏语义 ID，再以它为条件生成稠密向量；已在百度广告中上线。 |
| :material-new-box: [Generative Recommendation with Semantic IDs: A Practitioner's Handbook (GRID)](https://arxiv.org/abs/2507.22224) — Ju et al. (Snap), 2025 | 开源框架与消融实验——最好的动手起点（[代码](https://github.com/snap-research/GRID)）。 |
| :material-book-open-variant: [From Matching to Generation: A Survey on Generative Information Retrieval](https://arxiv.org/abs/2404.14851) — Li et al., 2024 | 搜索与推荐中的生成式召回综述。 |
