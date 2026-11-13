# 4.3 用户行为序列建模

!!! abstract "本章你将学到"
    - 为什么用户行为序列是现代推荐系统中最有价值的单一信号
    - SASRec：作用于 item ID 的因果 Transformer——本质上就是 token 为物品的 GPT，附可运行代码
    - BERT4Rec 的掩码物品目标，以及双向模型何时有意义
    - 如何处理长达 10^4–10^5 次行为的终身序列：基于检索的兴趣模型与压缩
    - 实时 vs 离线用户表示（TransAct + PinnerFormer），以及被数百个下游模型共享的基础用户模型（Meta 的 SUM）

## 为什么重要 {#why-this-matters-the-problem}

在第 [4.2](02-ranking-and-ctr-models.md) 章中，用户主要是一袋 ID 加一个池化后的历史向量，DIN 让候选可以对近期行为做注意力。但*顺序*很重要：刚连着看了三个做菜视频的人，与一个月前才看过一个的人，处于不同的状态。*长度*也很重要：大型平台上的活跃用户会积累数万次行为，其中蕴含着最近 50 次行为所遗漏的长期兴趣。

序列建模也是通往本部分所有生成式内容的桥梁。一旦你把用户历史当作 token 序列，并训练模型去预测下一个 token，你做的就是以物品为词表的语言建模——[第 1 部分](../part1/index.md)的整套工具箱（Transformer、因果掩码、scaling law）和[第 3 部分](../part3/index.md)的整套工具箱（KV cache）就都能用上了。

## 直觉 {#intuition}

把用户的历史读成一句话：`[跑鞋, 跑步袜, GPS 手表, 能量胶, ...]`。语言模型读到这里，会猜下一个词大概是"马拉松报名"或"泡沫轴"。序列推荐模型做的正是这件事：从左到右读历史，并输出下一个物品的分布。

三个实际问题决定了本章各种设计：

1. **方向：** 是从过去预测下一个物品（因果，像 GPT），还是从两侧填补被掩码的物品（双向，像 BERT）？
2. **长度：** 注意力对序列长度是平方复杂度；终身历史需要检索或压缩。
3. **新鲜度 vs 成本：** 每次行为都重算一个大序列模型太贵；一天只算一次又会错过最近十分钟。

## 工作原理 {#how-it-works}

### 记号 {#notation}

用户的历史是 $S = (s_1, s_2, \dots, s_T)$，其中 $s_t \in \mathcal{I}$ 是第 $t$ 步的物品（可选地带有行为类型、时间戳、展示位）。物品 embedding 表 $\mathbf{E}\in\mathbb{R}^{|\mathcal{I}|\times d}$ 把每个物品映射成一个向量。**下一物品预测**的目标是建模 $P(s_{t+1} \mid s_1,\dots,s_t)$。

### SASRec：作用于物品的因果 Transformer {#sasrec-a-causal-transformer-over-items}

SASRec（2018）是一个 decoder-only 的 Transformer（参见[第 1 部分](../part1/index.md)）：

1. 输入 $\mathbf{X}^{(0)}_t = \mathbf{E}[s_t] + \mathbf{P}_t$，其中 $\mathbf{P}_t$ 是学出来的位置 embedding。形状为 `[batch, T, d]`。
2. $L$ 个带**因果**自注意力（位置 $t$ 只能看到 $\le t$ 的位置）的 Transformer 块，加上前馈层、残差和 layer norm。
3. 位置 $t$ 的输出 $\mathbf{h}_t \in \mathbb{R}^d$ 就是经过 $t$ 次行为之后的用户状态。候选物品 $i$ 对下一步的得分是 $r_{t,i} = \mathbf{h}_t^\top \mathbf{E}[i]$——输入的 embedding 表被复用为输出层（权重绑定，与许多语言模型完全一样）。

训练时每个位置都是预测目标，和语言模型训练一样：在位置 $t$，标签是 $s_{t+1}$。原论文使用的是每个位置采样一个负样本的二元交叉熵；在现代硬件和规模适中的物品库下，完整的或采样的 softmax 交叉熵很常见，并且通常更强：

$$
\mathcal{L} = -\sum_{t=1}^{T-1}\log\frac{\exp(r_{t,s_{t+1}})}{\sum_{j\in\mathcal{I}}\exp(r_{t,j})}.
$$

用话来说：在每一步，让真实的下一个物品比所有其他物品更有可能。上线时，$\mathbf{h}_T$（最后一个位置）就是用户向量，可以直接送入物品 embedding 上的 ANN 索引——所以 SASRec 同时也是一个召回模型。

### BERT4Rec：掩码物品建模 {#bert4rec-masked-item-modelling}

BERT4Rec 把因果掩码换成完全的双向注意力，并用 **Cloze** 目标训练：随机把序列中一部分物品替换为 `[MASK]` token，再根据左右两侧的上下文去预测它们。推理时，在末尾追加 `[MASK]` 并预测它。双向上下文使每条序列提供更多训练信号，但它与真实任务不匹配（上线时永远拿不到未来），而且后续的复现研究报告称，用完整（或采样良好的）softmax 训练的因果模型可以追平甚至超过它。工业界以因果模型为主，因为它们还支持流式更新和类 KV cache 的增量计算。

### 超越 item ID：边信息与行为类型 {#beyond-item-ids-side-information-and-actions}

真实系统会往每个 token 里塞更多东西：行为类型（点击、点赞、跳过）、停留时长、时间间隔、展示位，以及物品内容特征。常见做法是为每个 token 把多个 embedding 相加（物品 + 行为 + 时间分桶），或者把物品 token 与行为 token 交错排列。Generative Recommenders（第 [4.4](04-scaling-recommendation-models.md) 章）把这种交错排列作为其建模方式的核心。

### 长序列：先检索，再做注意力 {#long-sequences-retrieve-then-attend}

对 $T$ 个 token 做自注意力，每层开销是 $O(T^2 d)$；目标注意力（DIN）每个候选的开销是 $O(Td)$，还要乘以数百个候选。当 $T = 10^4$–$10^5$ 时，两者都负担不起。有两类解决方案：

**基于检索的兴趣建模（SIM，阿里巴巴）。** 把工作拆成两个阶段，就像召回/精排级联，只是发生在用户历史*内部*：

1. **General Search Unit（GSU）：** 针对当前候选，从终身序列中低成本地检索出 top-$K$ 个相关行为——要么是*硬搜索*（保留与候选同品类的行为），要么是*软搜索*（用行为与候选 embedding 的内积，借助类似 ANN 的索引）。
2. **Exact Search Unit（ESU）：** 只对这 $K$ 个行为（例如几百个）做完整的目标注意力（或小型 Transformer）。

硬搜索尤其便宜，因为可以按品类为每个用户预先给行为建索引。SIM 报告称，在阿里的展示广告中它能服务长达 54,000 的用户序列，线上 CTR 提升 7.1%，RPM 提升 4.4%。

**压缩。** 把长历史概括为固定数量的向量：按时间窗口或品类做池化；随着行为到来增量更新的记忆网络（阿里的 MIMN，SIM 的前身，处理的序列最长约 1,000）；或者一组可学习的 query token 对完整历史做 cross-attention（Perceiver 式的瓶颈）。压缩可以在请求链路之外异步完成。

### 实时 vs 离线用户表示 {#realtime-vs-batch-user-representations}

表达能力和新鲜度之间存在张力：

- **离线用户 embedding** 由一个大型序列模型离线计算（例如每天一次）并存储。上线便宜，可以做得很大，但会过期：它不知道用户在过去一小时里做了什么。
- **实时序列模块**在请求时，在精排模型内部对最近的行为运行。足够新鲜，但为满足延迟必须很小。

Pinterest 把两者结合起来。**PinnerFormer** 是一个离线用户模型，它不是以"下一个行为"为目标训练的，而是用一个*稠密的全行为损失*：预测用户在未来一个时间窗口（若干天）内会互动的物品集合。这个目标让每天一次的 embedding 不那么容易过期，显著缩小了与实时 embedding 的差距。**TransAct** 是作用于用户最近实时行为的 Transformer，在 Homefeed 精排模型内部做端到端训练。生产中的精排模型同时使用两者：TransAct 负责短期意图，PinnerFormer 负责长期口味。这种**实时 + 离线混合**的模式如今已是标配。

最终的精排架构如下所示：

```text
 用户最近约 100 次行为（流式，延迟几秒）──► 实时 Transformer（类 TransAct）──────────────┐
 终身历史（10^4+ 次行为）──► 按候选检索（SIM 的 GSU）──► 目标注意力 ─────────────────────┤
 每日离线任务 ──► 长期用户 embedding（类 PinnerFormer），请求时查表 ───────────────────────┼──► 精排 MLP / 多任务头
 候选物品特征、上下文特征 ────────────────────────────────────────────────────────────────┘
```

### 基础用户模型：训练一次，处处使用 {#foundation-user-models-train-once-use-everywhere}

大型平台会运行数百个精排模型（每个展示位、每个目标、每种广告类型各一个）。它们每一个都负担不起一个大型序列模型。Meta 的 **Scaling User Modeling（SUM）** 框架训练少数几个大型*上游*用户模型，基于海量用户特征，并把得到的用户 embedding 作为输入特征提供给数百个*下游*广告精排模型。一个异步在线服务系统（SOAP）在用户请求时重新计算 embedding，而不增加延迟，并配有新鲜度和 embedding 稳定化机制，使得上游模型重训时下游模型不会受到扰动。这就是推荐领域里"基础模型 + 许多轻量任务头"的对应物。

### 与 LLM 的 next-token prediction 的联系 {#connecting-to-llm-next-token-prediction}

| 语言模型 | 序列推荐模型 |
|---|---|
| token 来自约 $10^5$ 大小的固定词表 | 物品来自 $10^6$–$10^{10}$ 大小、不断更替的词表 |
| 位置 embedding | 位置**以及**时间戳 / 时间间隔 |
| 对完整词表的 next-token 交叉熵 | 采用 sampled softmax 的下一物品损失（词表太大） |
| 一篇文档 = 一条序列 | 一个用户 = 一条不断增长的序列 |
| 用 KV cache 做增量解码 | 缓存的用户状态，随新行为到来而更新 |
| 参数量与数据量上的 scaling law | 骨干网络变得算力密集之后才出现 scaling law（第 4.4 章） |

词表这一行是最大的差异，也引出了**语义 ID**（第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章）：用共享 token 构成的短序列取代数十亿个不透明的物品 ID，让词表看起来更像 LLM 的词表。

## 代码 {#code}

一个最小化的 SASRec，在合成序列上训练，其中用户在"主题"之间游走（同一主题内的物品倾向于前后相继）。在这个规模下用完整 softmax 就可以。

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

N_ITEMS, T, D, N_TOPICS = 500, 20, 64, 25        # 物品 0 保留作填充
topic = torch.randint(0, N_TOPICS, (N_ITEMS,)); topic[0] = -1
member = (topic[None, :] == torch.arange(N_TOPICS)[:, None]).float()   # [N_TOPICS, N_ITEMS]

def make_batch(B):
    """每个用户大多停留在一个主题里，以 0.15 的概率跳到随机的新主题。"""
    seqs = torch.zeros(B, T, dtype=torch.long)
    cur = torch.randint(0, N_TOPICS, (B,))
    for t in range(T):
        jump = torch.rand(B) < 0.15
        cur = torch.where(jump, torch.randint(0, N_TOPICS, (B,)), cur)
        seqs[:, t] = torch.multinomial(member[cur], 1).squeeze(1)   # 在主题内均匀选一个物品
    return seqs

class SASRec(nn.Module):
    def __init__(self, n_items, d, max_len, n_layers=2, n_heads=2):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.02)                # 小初始化：输出层与之绑定
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, seq):                       # seq: [B, L]
        L = seq.size(1)
        x = self.item(seq) + self.pos(torch.arange(L))[None]          # [B, L, d]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)    # True = 不允许关注
        h = self.blocks(x, mask=causal, src_key_padding_mask=(seq == 0))
        return self.norm(h) @ self.item.weight.T  # [B, L, n_items]；权重绑定的输出层

model = SASRec(N_ITEMS, D, T)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(500):
    seq = make_batch(128)
    logits = model(seq[:, :-1])                   # 位置 t 预测 s_{t+1}
    loss = F.cross_entropy(logits.reshape(-1, N_ITEMS), seq[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    seq = make_batch(500)
    top10 = model(seq[:, :-1])[:, -1].topk(10).indices                 # [500, 10]
    hr = (top10 == seq[:, -1:]).any(1).float().mean()
    same_topic = (topic[top10] == topic[seq[:, -2]][:, None]).float().mean()
print(f"loss {loss.item():.2f}  HitRate@10 {hr:.3f}  top-10 中属于当前主题的占比 {same_topic:.2f}")
```

在 25 个主题、每个主题约 20 个物品的设定下，一个知道当前主题但不知道具体物品的模型，HitRate@10 大约为 0.4（≈ top 10 中占一半主题物品 × 用户留在该主题的约 0.86 概率）；随机模型只有 0.02。这次运行达到约 0.44，top-10 中超过 90% 属于当前主题。把因果掩码换成不加掩码并改用 `[MASK]` token 目标，你就得到了 BERT4Rec。

围绕这个模型使用的两个构件。第一个是 BERT4Rec 的掩码步骤；第二个是 SIM 式的"硬搜索"，在做目标注意力之前只保留同品类的行为：

```python
import torch
torch.manual_seed(0)
MASK_ID = 999                                      # 为 [MASK] 额外保留的 token id

def bert4rec_mask(seq, p=0.2):
    """Cloze 掩码：隐藏比例为 p 的真实物品；其余位置的标签为 -100（被忽略）。"""
    seq = seq.clone()
    is_item = seq != 0
    masked = (torch.rand(seq.shape) < p) & is_item
    labels = torch.where(masked, seq, torch.full_like(seq, -100))
    seq[masked] = MASK_ID
    return seq, labels                             # 用 cross_entropy(..., ignore_index=-100) 训练

def sim_hard_search(hist_items, hist_cats, cand_cat, emb, K=4):
    """GSU：保留候选所属品类下最近的 K 个行为；ESU：注意力。"""
    idx = (hist_cats == cand_cat).nonzero().squeeze(1)[-K:]   # 最近的若干匹配项
    if len(idx) == 0:
        return torch.zeros(emb.shape[1])
    h = emb[hist_items[idx]]                                   # [k, d]
    c = emb[hist_items[idx[-1]]]                               # 候选向量的替身
    w = torch.softmax(h @ c / h.shape[1] ** 0.5, dim=0)        # ESU：对 k 个物品做注意力
    return (w[:, None] * h).sum(0)                             # [d]

seq = torch.tensor([[5, 17, 23, 42, 8, 0, 0]])                 # 0 = 填充
print(bert4rec_mask(seq, p=0.5))
emb = torch.randn(1000, 16)
hist_items = torch.randint(1, 1000, (10_000,))                 # 一条长度为 1 万的终身历史
hist_cats = torch.randint(0, 50, (10_000,))
print(sim_hard_search(hist_items, hist_cats, cand_cat=7, emb=emb).shape)   # torch.Size([16])
```

在生产中，每个用户的分品类列表是预先建好的（两级索引：用户 → 品类 → 行为），所以 GSU 是一次查表而不是扫描，ESU 使用的则是真实的候选 embedding。

## 实例演算 {#worked-example}

**终身注意力的开销。** 一个用户有 $T = 10{,}000$ 次过往行为，embedding 维度 $d=64$，精排模型为 $N = 500$ 个候选打分。

- *对历史做完整的因果自注意力*，单层：$QK^\top$ 和 $AV$ 两个乘积约需 $2\cdot 2 \cdot T^2 d = 4\cdot 10^8 \cdot 64 \approx 2.6\times10^{10}$ FLOPs（26 GFLOPs），每层每次请求——这还没算投影和 MLP。各候选之间可以共享，但开销依然很大。
- *点积打分的 DIN 式目标注意力*：每个候选，打分约 $2Td$，加权求和约 $2Td$，合计 $4 \cdot 10^4 \cdot 64 \approx 2.6$ MFLOPs；对 500 个候选约 $1.3$ GFLOPs。若使用 DIN 的 MLP 打分器（输入 $4d$，隐层 32），仅打分一项就是每个行为约 $2\cdot 4d\cdot 32 = 16{,}384$ FLOPs，即每个候选约 $1.6\times10^8$，每次请求约 $8\times10^{10}$——太贵了。
- *SIM：* GSU 为每个候选检索 $K = 200$ 个行为（按品类做硬搜索本质上就是一次查表）。ESU 的 MLP 打分注意力现在每个候选的开销是 $200 \cdot 16{,}384 \approx 3.3$ MFLOPs，每次请求约 $1.6$ GFLOPs：降低了 50 倍，同时关注的仍是与每个候选最相关的行为。

**新鲜度差距。** 对于一天中任意随机时刻的请求，凌晨 02:00 算出的离线 embedding 平均大约已经过期 12 小时。当前会话中的行为——往往是最具预测力的——它完全看不到。这就是要给它配上一个小型实时模块（TransAct 式）的原因。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- 每一个主要的信息流和广告精排模型都使用某种针对用户行为的序列模型；对近期行为做目标注意力，再加上检索得到的长期行为，是标准的经典设计。
- **Generative Recommenders（HSTU）**把行为序列作为*唯一*输入，取代了大部分人工设计的用户特征（第 4.4 章）。
- **基础用户模型**（SUM 式上游 embedding、LUM 预训练的大型用户模型，第 4.4 章）被许多下游模型共享。
- 基于 LLM 的用户编码器（HLLM、User-LLM）把预训练语言模型的权重引入用户建模（第 [4.7](07-llms-in-recommendation.md) 章）。

!!! tip "练习"
    给 SASRec 代码加上时间间隔 embedding：把距上一次行为的秒数取对数后分成 16 个桶，为每个 token 加上其所在桶的可学习 embedding。生成合成数据，让用户在长时间间隔之后切换主题，并检查 HitRate 是否提升。

## 常见误区 {#common-pitfalls-misconceptions}

- **泄露未来**：忘记因果掩码，或构造的序列包含了标签物品，会带来漂亮的离线数字，线上却毫无作用。
- **只用采样负样本做评估**（让真实物品与 100 个随机物品比排名）可能使模型排名与全量物品库评估的结果不同；条件允许时优先使用全量排序。
- **因为 BERT 在某些 NLP 基准上击败了 GPT，就认为 BERT4Rec 理所当然更好**；对下一物品预测而言，因果训练与任务相匹配。
- **忽略时间**：相隔一秒的两次行为和相隔一个月的两次行为并不相同；要对时间戳或时间间隔编码。
- **不做稳定化就重训上游用户模型**，会悄悄改变所有下游模型的输入分布。

## 本章要点 {#key-takeaways}

- 用户行为序列是最强的信号；对其顺序和长度建模，是许多现代收益的来源。
- SASRec 就是 item ID 上的 GPT：因果 Transformer、下一物品交叉熵、绑定 embedding。
- 终身序列需要在历史内部再套一层级联：先用低成本搜索找出相关行为，再做精确注意力（SIM），或者做压缩。
- 生产系统把新鲜的实时模块与过期但信息丰富的离线 embedding 结合使用，并在众多下游模型之间共享基础用户模型。

## 自测 {#check-your-understanding}

??? question "在 SASRec 中，为什么输出层要与输入的物品 embedding 表绑定？"
    得分 $\mathbf{h}_t^\top\mathbf{E}[i]$ 把用户状态和物品放在同一个空间里，使参数量减半（物品表是最大的组成部分），并让同一份 embedding 既能当作召回用的 ANN 索引。这与语言模型中的权重绑定如出一辙。

??? question "为什么 SIM 中的 GSU 不需要很精确？"
    与主级联中的召回一样，它的任务是召回率：让相关行为包含在 $K$ 个候选之中。之后由 ESU 做精确的加权。只要相关行为没有被丢掉，廉价的品类匹配就足够了。

??? question "PinnerFormer 的稠密全行为损失解决了什么问题？"
    离线用户 embedding 每天只算一次，却要用一整天。只训练它预测紧接着的下一个行为，会使它在过期后变得脆弱；训练它预测未来一个时间窗口内的所有互动，则使它对这种过期更加鲁棒。

??? question "列举序列推荐与语言建模的两点不同。"
    词表要大几个数量级且每天变化（需要 sampled softmax 或语义 ID）；序列带有时间戳、行为类型及其他边信息；此外用户序列会无限增长，而不是固定的文档。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) — Kang & McAuley, 2018 | 对物品序列做下一物品预测的因果 Transformer——本质上就是 item ID 上的 GPT。 |
| :material-book-open-variant: [BERT4Rec](https://arxiv.org/abs/1904.06690) — Sun et al., 2019 | 用于序列推荐的双向掩码物品建模。 |
| :material-book-open-variant: [Deep Interest Evolution Network for Click-Through Rate Prediction (DIEN)](https://arxiv.org/abs/1809.03672) — Zhou et al. (Alibaba), 2018 | 用 GRU 和辅助损失在 DIN 的基础上加入随时间演化的兴趣；已部署于淘宝展示广告。 |
| :material-star: [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction (SIM)](https://arxiv.org/abs/2006.05639) — Pi et al. (Alibaba), 2020 | 两阶段 GSU/ESU 设计，支持长达 54,000 个行为的序列。 |
| :material-book-open-variant: [TransAct](https://arxiv.org/abs/2306.00248) / [PinnerFormer](https://arxiv.org/abs/2205.04507) — Pinterest, 2022–2023 | 对近期行为的实时 Transformer，与长期离线用户 embedding 相结合——实时 + 离线混合模式。 |
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | SUM：少数几个大型上游用户模型产出 embedding，供数百个下游广告模型使用；涵盖在线 embedding 的新鲜度。 |
