# 4.6 端到端生成式推荐

!!! abstract "本章你将学到"
    - 为什么用一个生成式模型取代“召回 → 排序”级联很有吸引力，以及它的代价
    - OneRec 的架构：用户历史编码器、基于语义 ID 的 MoE 解码器、按 session 生成列表
    - 奖励模型与迭代式 DPO 如何对推荐模型做偏好对齐——和[第 2 部分](../part2/index.md)是同一套后训练思路——附可运行代码
    - OneRec 技术报告与 OneRec-V2 新增了什么：多目标奖励的 RL、高 MFU、更低的 OPEX、lazy decoder-only 设计
    - 推荐中的推理（OneRec-Think、ReaRec）、适配预训练 LLM（PLUM），以及尚未解决的问题

## 为什么重要 {#why-this-matters-the-problem}

第 [4.1](01-recsys-fundamentals.md) 章的级联架构是可用的，但有结构性弱点：

- **目标不一致。** 召回、粗排、精排、重排分别在不同的数据和损失上训练。精排很喜欢的物品，可能被一个对此毫不知情的召回环节丢掉。
- **计算碎片化。** 几十个模型、特征管线和服务，每个都很小，硬件利用率低。OneRec 报告认为，这种碎片化让推荐系统享受不到 AI 社区的技术突破。
- **人工拼装列表。** 精排一次只给一个物品打分；最终列表（多样性、顺序、混排）靠规则拼出来。

LLM 提示了另一条路：用一个大模型读取上下文并直接**生成**输出，再用奖励模型和 RL 把它**对齐**到用户真正想要的东西上。2025 年以来，快手的 OneRec 系列已经证明，这可以在大规模线上信息流中取代级联。本章把 4.3（序列模型）、4.4（规模化）和 4.5（语义 ID）的各个部件组装成这样一个系统。

## 直觉 {#intuition}

请一位经验丰富的人工策展人“现在给这个人挑十个视频”。她不会把十亿个视频逐个打分；她会读这个人的历史，然后*写下一份列表*，整份列表一起考虑（不会连着放十个做菜视频）。再随着时间推移，她从反馈中学到哪些列表大家爱看。

端到端生成式推荐模型做的是同样的事：

1. **读**：对用户特征和行为历史做编码。
2. **写**：自回归地解码出一个物品列表，每个物品都拼写成语义 ID（见第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章）。
3. **从反馈中学习**：先在日志里的优质 session 上预训练（类似 LM 预训练 / SFT），再用奖励模型和偏好优化继续提升（类似 RLHF）。

## 工作原理 {#how-it-works}

### OneRec 架构（V1） {#onerec-architecture-v1}

**物品 tokenize。** 每个视频由几个码组成的语义 ID 表示，来自对多模态内容 embedding 做残差量化（OneRec 的论文使用基于均衡 k-means 的残差量化，让各个码被均匀使用）。输出词表：几千个码 token，而不是几亿个视频 ID。

**编码器。** Transformer 编码器读取用户行为序列（以及用户特征），为 $n$ 个历史位置产生上下文向量 $\mathbf{H} \in \mathbb{R}^{n\times d}$。

**带 MoE 的解码器。** Transformer 解码器逐个生成所推荐视频的语义 ID token，并通过交叉注意力关注 $\mathbf{H}$。它的前馈层是**稀疏混合专家**（Mixture-of-Experts）：每个 token 被路由到众多专家 FFN 中的少数几个，因此总参数量增长时 FLOPs 不会成比例增长（和[第 1 部分](../part1/index.md)里 MoE LLM 用的是同一个技巧）。

**按 session 生成。** OneRec 不是预测单个下一个物品，而是生成一整个 **session**：针对一次请求的一小列视频。训练目标是高质量的历史 session（用户互动良好的 session）。联合生成整份列表，使模型能考虑列表*内部*的上下文——前面出现过什么——而不必依赖手写规则去拼装各自独立打分的物品。

形式化地说，对于用户上下文 $u$ 和目标 session $S = (v_1,\dots,v_m)$，其中物品 $v_j$ 的语义 ID 为 $(c_{j,1},\dots,c_{j,D})$，预训练损失就是对展平后的码序列做普通的下一 token 交叉熵：

$$
\mathcal{L}_{\text{NTP}} = -\sum_{j=1}^{m}\sum_{l=1}^{D}\log \pi_\theta\big(c_{j,l}\mid u,\ c_{<j},\ c_{j,<l}\big),
$$

其中 $\pi_\theta$ 是模型的下一 token 分布。通俗地说：教模型一个码一个码地把好的 session 写出来。

```text
 用户特征 + 行为序列                                     所推荐 session 的语义 ID
 ─────────────────────────────────►  编码器  ──H──►  解码器（MoE FFN，对 H 做交叉注意力）
                                   (Transformer)        BOS c1,1 c1,2 c1,3 | c2,1 c2,2 c2,3 | ...
                                                         物品 1           物品 2
                                         束搜索 + 有效 ID 前缀树 ──► 视频列表
```

### 用奖励模型和迭代式 DPO 做偏好对齐 {#preference-alignment-with-a-reward-model-and-iterative-dpo}

下一 token 训练只是在模仿日志里的 session，而这些 session 是旧的级联系统产生的。为了*超越*它们，OneRec 借用了 LLM 后训练里的**直接偏好优化**（Direct Preference Optimization，见[第 2 部分](../part2/index.md)）。DPO 需要针对同一输入的（chosen, rejected）输出对。推荐场景的特殊之处在于：每次请求只展示*一份*列表，所以你永远观察不到用户对两个备选方案的反应。

OneRec 的解法是训练一个**奖励模型** $R(u, S)$，用日志学习预测用户会有多喜欢 session $S$——相当于用户反馈的模拟器。然后**迭代式偏好对齐**（Iterative Preference Alignment）反复执行：

1. 对一个用户上下文 $u$，从当前模型采样 $N$ 个候选 session（例如用束搜索）。
2. 用 $R$ 给它们打分；取得分最高的作为 $S_w$（chosen），得分最低的作为 $S_l$（rejected）。
3. 针对一个冻结的参考模型 $\pi_{\text{ref}}$（本轮开始前的模型），用 DPO 损失更新模型：

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\Big[\log\frac{\pi_\theta(S_w\mid u)}{\pi_{\text{ref}}(S_w\mid u)} - \log\frac{\pi_\theta(S_l\mid u)}{\pi_{\text{ref}}(S_l\mid u)}\Big]\Big),
$$

其中 $\sigma$ 是 logistic 函数，$\beta>0$ 控制策略可以偏离参考模型多远，$\log\pi(S\mid u)$ 是该 session 各 token 对数概率之和。通俗地说：相对于模型起点，提高好列表相对于差列表的概率。

4. 把更新后的模型作为新的采样器和参考模型，重复上述过程。

DPO 训练会与普通的下一 token 损失混合使用，避免模型漂移。OneRec 论文报告称，少量 DPO 样本就足以对齐偏好并显著改善生成结果，并且在快手主场景部署 OneRec 使观看时长提升了 1.6%。

### OneRec 技术报告：RL、MFU 与成本 {#the-onerec-technical-report-rl-mfu-and-cost}

后续的技术报告描述了一个完整的线上系统，在一部分流量上取代了级联。它的主要结论（来自摘要）：

- **算力。** 推荐模型的 FLOPs 提升了 **10 倍**，并在一定范围内发现了 scaling law。
- **RL 在这里行得通。** 整份列表由同一个策略生成，强化学习（很难用在级联上）就变得很自然。报告针对一个融合多目标（用户互动信号和其他要求）的奖励体系来优化策略——概念上就是[第 2 部分](../part2/index.md)里 GRPO 式的设定：采样多份列表、打分，推高奖励高于平均的那些。
- **硬件效率。** 基础设施优化使训练和推理的 MFU 分别达到 **23.7% 和 28.8%**——接近 LLM 的水平，而许多传统精排模型只有个位数（见第 [4.4](04-scaling-recommendation-models.md) 章）。
- **成本。** 运营开销（OPEX）只有传统管线的 **10.6%**，因为一个模型取代了许多服务，通信和存储都更少。
- **效果。** 部署在快手 / 快手极速版，承接了 **25% 的总 QPS**，App 停留时长分别提升 0.54% 和 1.24%，7 日生命周期（LT7）也有提升。

### OneRec-V2：lazy decoder-only {#onerec-v2-lazy-decoder-only}

OneRec-V2 发现，V1 里 **97.66% 的算力花在对上下文做编码**上，而不是生成推荐。V2 的 **lazy decoder-only** 架构去掉了独立的编码器：上下文信息由解码器各层直接消化，不需要一次完整而昂贵的编码，于是算力集中在生成的 token 上。报告称这使总计算量降低了 **94%**、训练资源降低了 **90%**，从而能够扩展到 **8B** 参数。V2 还改为与**真实用户交互**对齐，而不是只依赖奖励模型，使用了时长感知的奖励塑形（duration-aware reward shaping）和自适应比率裁剪（adaptive ratio clipping），并报告了 A/B 实验中 App 停留时长 0.467% / 0.741% 的提升，同时兼顾多个目标。

这个教训和 LLM 的历史如出一辙：范式跑通之后，效率工程（decoder-only、算力花在哪、MFU）会打开下一级规模。

### 推荐中的推理 {#reasoning-in-recommendation}

推理模型（测试时算力、思维链）正以两种形式进入推荐系统：

- **显式的文本推理——OneRec-Think。** 把物品语义 ID 与 LLM 的文本空间对齐（*itemic alignment*），在推荐场景中激活推理能力（*reasoning scaffolding*），再用一个推荐专用的奖励加强它——这个奖励考虑到了“很多不同的物品都可以是有效答案”这一事实。模型可以先用自然语言推理，再输出物品 token。一个“Think-Ahead”架构让部署变得可行；论文报告在快手上带来了 0.159% 的 App 停留时长提升。
- **隐式推理——ReaRec。** 不用文字，而是在给物品打分之前，把序列模型最后一层的隐状态再喂回模型自身，多走几步（配上特殊的推理位置 embedding）——也就是在表示空间里做隐式的多步“思考”。它报告称，在公开数据集的事后分析中，可把若干序列推荐骨干模型的性能上限提高约 30–50%。

### 从预训练 LLM 出发：PLUM {#starting-from-a-pre-trained-llm-plum}

与其从零训练推荐模型，Google/YouTube 的 **PLUM** 选择适配一个预训练 LLM：(1) 把物品 tokenize 成语义 ID；(2) 在混合了用户行为（语义 ID 序列）和文本的领域数据上对 LLM 做**继续预训练**（continued pre-training）；(3) 针对具体任务微调，重点是生成式召回——模型根据用户上下文生成所推荐物品的语义 ID。PLUM 报告称，相比一个基于大型 embedding 表、经过重度优化的线上模型，召回有显著提升，并给出了 scaling 研究以及关于 CPT 的经验教训。第 [4.7](07-llms-in-recommendation.md) 章介绍 LLM 进入推荐系统的其他方式。

??? note "其他生成式范式：扩散"
    自回归解码并不是唯一的生成式选择。**DiffRec** 把用户的交互向量当作要生成的数据：给它加噪声破坏，再训练模型去噪，然后推荐去噪后向量中取值高的物品。隐空间和时序变体让它能在更大规模上实用。扩散模型在广告中再次出现，用来生成出价轨迹（见第 [4.8](08-ads-systems.md) 章）。

### 与 LLM 流水线的类比 {#the-llm-pipeline-analogy}

| LLM（第 1–2 部分） | 端到端生成式推荐 |
|---|---|
| 分词器（BPE） | 物品分词器（残差量化 → 语义 ID） |
| 在网页文本上预训练 | 在日志中的优质 session 上做下一 token 训练 |
| Prompt | 用户特征与行为历史 |
| 回复 | 一个由物品语义 ID 组成的 session / 列表 |
| 由人类偏好标注训练的奖励模型 | 预测用户互动的奖励模型（或 V2 里的真实反馈） |
| DPO / RLHF / GRPO | 迭代式 DPO（OneRec）、多目标奖励的 GRPO 式 RL（技术报告、V2） |
| 为格式服务的约束解码 | 基于前缀树的约束解码，只生成有效且合规的物品 |
| KV cache、batching、MFU | 同一套推理服务栈（第 3 部分） |

## 代码 {#code}

一个玩具级的端到端循环：一个很小的解码器为用户生成 2 个码的“语义 ID”，一个固定的奖励模型对每类用户偏好特定的码，再用基于 best/worst-of-$N$ 采样的迭代式 DPO 来提高期望奖励。

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy
torch.manual_seed(0)
K, LEVELS, N_USERS, N_SAMPLES, BETA = 8, 2, 4, 8, 0.5

class TinyGenRec(nn.Module):
    """以用户 id 为条件，生成语义 ID (c1, c2)。"""
    def __init__(self, d=32):
        super().__init__()
        self.user = nn.Embedding(N_USERS, d)
        self.code = nn.Embedding(K + 1, d)                    # 下标 K = BOS
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, K)
    def logprob(self, users, ids):                            # ids: [B, LEVELS] -> [B]
        bos = torch.full((len(users), 1), K)
        inp = self.code(torch.cat([bos, ids[:, :-1]], 1))     # teacher forcing
        h, _ = self.rnn(inp, self.user(users)[None])
        lp = F.log_softmax(self.out(h), -1)                   # [B, LEVELS, K]
        return lp.gather(2, ids[..., None]).squeeze(-1).sum(1)
    @torch.no_grad()
    def sample(self, users):
        ids, h = [], self.user(users)[None]
        tok = torch.full((len(users),), K)
        for _ in range(LEVELS):
            o, h = self.rnn(self.code(tok)[:, None], h)
            tok = torch.multinomial(F.softmax(self.out(o[:, 0]), -1), 1).squeeze(1)
            ids.append(tok)
        return torch.stack(ids, 1)

def reward(users, ids):                                       # 奖励模型的替身
    return (ids[:, 0] == users % K).float() + 0.5 * (ids[:, 1] == (users * 3) % K).float()

policy = TinyGenRec()
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
users = torch.arange(N_USERS)
for it in range(6):                                           # 对齐轮次
    ref = copy.deepcopy(policy).eval()                        # 本轮冻结的参考模型
    with torch.no_grad():
        samples = torch.stack([policy.sample(users) for _ in range(N_SAMPLES)], 1)  # [U, N, L]
        r = torch.stack([reward(users, samples[:, n]) for n in range(N_SAMPLES)], 1)
        print(f"round {it}: mean reward of samples {r.mean():.2f}")
        win, lose = samples[users, r.argmax(1)], samples[users, r.argmin(1)]
        ref_w, ref_l = ref.logprob(users, win), ref.logprob(users, lose)
    for _ in range(20):                                       # 在这些样本对上做 DPO 更新
        margin = (policy.logprob(users, win) - ref_w) - (policy.logprob(users, lose) - ref_l)
        loss = -F.logsigmoid(BETA * margin).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

平均奖励在几轮内就从接近随机水平（≈0.19）升到 ≈1.0，尽管我们从未给出任何日志里的“正确答案”——只有奖励模型给出的比较。随后它**停在了最大值 1.5 以下**：策略对第一个码已经很有把握，对第二个码却没有，于是 8 个样本看起来都差不多，最好和最差的样本奖励相同，DPO 的 margin 不带任何信号，学习就停了。这是一个真实问题的缩影——对齐需要多样的样本（温度、更大的 $N$、探索）才能持续产生有信息量的样本对。真实系统还会在日志 session 上加入下一 token 损失，并加上防止奖励作弊（reward hacking）的保护措施。

GRPO 风格的组内归一化优势与策略梯度损失（可与[第 2 部分](../part2/index.md)对照）；`logp` 应是当前策略下每个采样列表的 token 对数概率之和：

```python
import torch
rewards = torch.tensor([[0.8, 0.5, 0.2, 0.5]])        # [请求数, 每个请求的样本数]
adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, unbiased=False, keepdim=True) + 1e-6)
print(adv)                                              # tensor([[ 1.4142,  0.0000, -1.4142,  0.0000]])
logp = torch.tensor([[-12.0, -11.5, -10.0, -13.0]], requires_grad=True)
loss = -(adv.detach() * logp).mean()                    # 提高高于平均水平的列表的对数概率
loss.backward()
print(logp.grad)                                        # 列表 1 的梯度为负，列表 3 的为正
```

## 实例演算 {#worked-example}

**手算 DPO 损失。** $\beta = 0.1$。对某个用户，chosen session 有 $\log\pi_\theta(S_w) = -10.0$、$\log\pi_{\text{ref}}(S_w) = -11.0$；rejected session 有 $\log\pi_\theta(S_l) = -9.0$、$\log\pi_{\text{ref}}(S_l) = -9.5$。

- chosen 的对数比：$-10.0-(-11.0) = 1.0$。rejected 的对数比：$-9.0-(-9.5) = 0.5$。
- Margin $= \beta(1.0-0.5) = 0.05$。
- 损失 $= -\log\sigma(0.05) = \log(1+e^{-0.05}) \approx 0.668$。

梯度会推高 $\log\pi_\theta(S_w)$、压低 $\log\pi_\theta(S_l)$。注意，在当前策略下 rejected session 仍然比 chosen 的概率*更高*（$-9.0 > -10.0$）；DPO 关心的是相对于参考模型的变化，而不是原始概率。

**组内归一化优势（GRPO 式 RL）。** 针对同一请求采样的四份列表，奖励分别为 $[0.8, 0.5, 0.2, 0.5]$。均值 $= 0.5$；偏差 $[0.3, 0, -0.3, 0]$；标准差 $= \sqrt{(0.09+0+0.09+0)/4} \approx 0.212$。优势 $= [1.41, 0, -1.41, 0]$：策略梯度会提高列表 1 的对数概率、降低列表 3 的，列表 2 和 4 保持不变。

**为什么一个模型可以更便宜。** 在级联架构里，一次请求可能要经过多个召回服务、一个处理 5,000 个候选的粗排和一个处理 500 个候选的精排，每一环都有各自的特征拉取和网络跳转。生成式模型只对用户编码一次，然后用束搜索解码，比如 10 个物品 × 3 个码 = 30 个 token。即使每个解码步用的是大模型，做的也是在单个加速器上的稠密矩阵运算，MFU 很高，而不是许多受内存带宽限制的服务——这就是 OneRec 报告的 OPEX 只有级联的 10.6% 背后的机制。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **快手的 OneRec 系列**是目前公开资料中最完整的线上端到端生成式推荐案例，现已发展到 V2（decoder-only，8B），并有带推理的变体。
- **混合部署**比完全替换更常见：生成式模型作为一路召回源，或者作为一个大模型，其输出再喂给经典精排模型（LUM，见 4.4 章）。
- **后训练配方**沿用 LLM 的做法：先在日志上预训练，再用奖励模型做 DPO / GRPO 式 RL；奖励越来越多地来自真实用户反馈（OneRec-V2）。
- **广告**采用同样的思路，并额外加入出价、分配与付费的机制（EGA-V2，见第 [4.8](08-ads-systems.md) 章）。

## 常见误区 {#common-pitfalls-misconceptions}

- **模仿不等于提升。** 在日志上做纯下一 token 训练只会复现旧系统的偏差；收益来自用奖励做对齐。
- **奖励作弊。** 策略可能钻学出来的奖励模型的空子（例如总是生成奖励模型打分偏高的标题党式内容）。要使用 KL / 参考模型约束、留出的线上指标，并定期重训奖励模型。
- **忽视约束。** 生成式模型天然不会遵守业务规则（去重、政策过滤、创作者公平、新鲜度）。这些必须通过约束解码、奖励或后置过滤来保证。
- **低估线上服务的工程量。** 长上下文上的束搜索需要 LLM 式的推理服务（batching、KV 缓存、量化；见[第 3 部分](../part3/index.md)）。
- **和一个弱的级联系统比。** 收益应当与一个强大、调优充分、算力预算相同的线上系统比较。

## 尚未解决的问题 {#open-problems}

- **业务规则与可控性：** 如何在生成的列表里保证硬约束（法律、安全、合同）。
- **多样性与探索：** 生成式模型可能坍缩到狭窄的模式上；列表级奖励和采样策略有帮助，但问题远未解决。
- **广告插入：** 在同一份生成列表里混排自然内容和竞价广告，同时保持竞价机制的激励相容。
- **新鲜度：** 语义 ID 和模型必须跟上每天数百万的新物品，又不能每次都完整重训。
- **评估：** 生成列表的离线指标，以及在策略改变数据分布之后仍然可信的奖励模型。

## 本章要点 {#key-takeaways}

- 端到端生成式推荐用一个模型取代级联：读取用户，解码出一个语义 ID 列表。
- OneRec = 编码器 + MoE 解码器 + 按 session 生成 + 奖励模型引导的迭代式 DPO；技术报告新增了 RL、高 MFU 和 10.6% 的 OPEX。
- OneRec-V2 转向 lazy decoder-only 架构（计算量减少 94%）和基于真实反馈的 RL，规模扩展到 8B。
- LLM 后训练的思路（DPO、GRPO 式 RL、推理）可以直接迁移；推荐场景特有的难点是每次请求你只能观察到一份列表的反馈。
- 约束、多样性、广告和新鲜度仍是开放问题。

## 自测 {#check-your-understanding}

??? question "为什么不能像 LLM 里处理人类偏好对那样，把 DPO 直接用在日志里的推荐数据上？"
    DPO 需要针对同一输入的一个 chosen 和一个 rejected 输出，但每次请求只给用户展示一份列表。OneRec 的做法是让模型采样出多个候选列表，再用一个模拟用户反馈的奖励模型给它们排序，从而构造出样本对。

??? question "按 session 生成相比独立生成各个物品有什么好处？"
    模型可以让每个物品以列表中的其他物品为条件，考虑冗余、顺序和混排，而不是独立给物品打分、再靠手写规则拼装列表。

??? question "OneRec-V2 发现 97.66% 的算力花在编码上。这为什么是个问题，V2 是怎么做的？"
    花在反复编码长上下文上的算力并不能直接改善生成的推荐，还限制了负责生成的部分能做多大。V2 的 lazy decoder-only 设计去掉了笨重的编码器，使总计算量降低 94%，并支持扩展到 8B 参数。

??? question "在 DPO 手算例子里，如果策略对 chosen 的对数比从 1.0 升到 3.0，损失会怎样变化？"
    Margin $= 0.1(3.0-0.5)=0.25$；损失 $= \log(1+e^{-0.25}) \approx 0.576$，低于 0.668，符合预期。

??? question "说出在靠广告变现的产品里部署端到端生成式推荐的两个难题。"
    在生成的列表里落实业务和政策约束，以及在插入竞价广告的同时保持竞价机制的性质（定价、激励相容）；此外还有新鲜度、多样性和评估。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) — Deng et al. (Kuaishou), 2025 | 带稀疏 MoE 的编码器–解码器生成一个视频 session；奖励模型 + 迭代式 DPO；已部署在快手主信息流。 |
| :material-star: [OneRec Technical Report](https://arxiv.org/abs/2506.13695) — Kuaishou, 2025 | 完整的端到端系统：10 倍算力，23.7% / 28.8% MFU，多目标奖励的 RL，OPEX 为级联的 10.6%，承接 25% 的 QPS。 |
| :material-new-box: [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900) — Kuaishou, 2025 | lazy decoder-only 架构，扩展到 8B；用真实用户反馈信号做 RL。 |
| :material-new-box: [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639) — Kuaishou, 2025 | 把语义 ID 与 LLM 的文本空间对齐，让模型在推荐之前先用自然语言推理。 |
| :material-new-box: [Think Before Recommend: Latent Reasoning for Sequential Recommendation (ReaRec)](https://arxiv.org/abs/2503.22675) — Tang et al., 2025 | 在用户表示上做测试时的隐式推理步骤。 |
| :material-new-box: [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784) — Google/YouTube, 2025 | 语义 ID → LLM 继续预训练 → 在 YouTube 上微调做生成式召回。 |
| :material-book-open-variant: [Diffusion Recommender Model (DiffRec)](https://arxiv.org/abs/2304.04971) — Wang et al., SIGIR 2023 | 对被破坏的用户–物品交互向量去噪，从而生成推荐。 |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | 把端到端生成的思路用于广告，包括分配与付费（见 4.8 章）。 |
