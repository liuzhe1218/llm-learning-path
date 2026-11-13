# 4.8 广告系统

!!! abstract "本章你将学到"
    - 广告排序与自然内容推荐有何不同：三方市场、以金额衡量的预测、广告主目标
    - 从零讲起的广告竞价（auction）：eCPM $=$ 出价 × pCTR（× pCVR），第二价格与广义第二价格（GSP）定价（附例子演算），以及为什么真实报价（truthfulness）很重要（VCG 直觉）
    - 为什么校准（calibration）在广告里至关重要，以及转化建模如何处理样本选择偏差（ESMM）和延迟反馈
    - 预算、预算平滑（pacing）与自动出价（auto-bidding，目标 CPA / ROAS），作为控制与优化问题，附一个仿真
    - 生成式前沿：基于扩散的自动出价（AIGB）、端到端生成式广告（EGA-V2）、生成式广告创意、受隐私约束的建模，以及大型广告基础模型（Meta GEM）

## 为什么重要 {#why-this-matters-the-problem}

广告养活了大多数大型消费互联网平台。广告系统是一个有三方而不是两方的推荐系统：

- **用户**想要相关、不烦人的内容。
- **广告主**想要以自己设定的价格换来效果（点击、安装、购买），并且花的是真金白银的预算。
- **平台**想要长期收入，而这取决于让用户和广告主都满意。

排序的那套机制和第 [4.1](01-recsys-fundamentals.md)–[4.4](04-scaling-recommendation-models.md) 章里是一样的：召回、多任务模型精排、序列特征。新的地方在于，预测要在**竞价**中**乘上金额**，由竞价决定展示哪些广告，*以及广告主付多少钱*。预测误差不再只是排序变差一点，而是一笔定错价的交易。而且广告主越来越多地把出价交给平台的**自动出价**算法，这就把整个系统变成了一个庞大的在线优化问题。

## 直觉 {#intuition}

想象一个每秒都在变的广告牌位置，拍卖给最看重它的人。广告主看重的不是一次*曝光*，而是一个*效果*。如果广告主 A 愿意为每次点击付 \$2，而她的广告被这位用户点击的概率是 5%，那么这次曝光对她的价值就是 $0.05 \times \$2 = \$0.10$。按这个每次曝光的期望价值——**eCPM**（effective cost per mille，每千次曝光的有效成本）——给广告排序，能选出创造价值最多的广告，而平台的任务就是*准确地*估计点击概率，因为它直接决定了这个价值。

接下来是定价：如果直接按出价向赢家收费，赢家就会压低出价、玩各种花样。**第二价格**这类拍卖只向赢家收取获胜所需的最低价，这让诚实出价成为一个合理的策略。

## 工作原理 {#how-it-works}

### 从预测到期望价值 {#from-prediction-to-expected-value}

对于在上下文 $c$ 中展示给用户 $u$ 的广告 $a$：

- $\text{pCTR}_a = P(\text{click}\mid u,a,c)$，来自精排模型；
- $\text{pCVR}_a = P(\text{conversion}\mid \text{click}, u, a, c)$，用于以转化为优化目标的广告；
- $b_a$ = 广告主的出价，可以按点击（CPC）、按转化（CPA）或按千次曝光（CPM）表示。

用统一货币衡量的每次曝光期望价值为

$$
\text{eCPM}_a = 1000 \times \begin{cases}
b_a^{\text{CPM}}/1000 & \text{CPM bid}\\
b_a^{\text{CPC}}\cdot \text{pCTR}_a & \text{CPC bid}\\
b_a^{\text{CPA}}\cdot \text{pCTR}_a\cdot\text{pCVR}_a & \text{CPA bid}
\end{cases}
$$

通俗地说：用模型给出的概率把每个出价都换算成“每次曝光的期望金额”，这样出价类型不同的广告就能在同一场竞价里竞争。真实系统还会加上一个**质量 / 用户体验项**（例如预测的负反馈、相关性、落地页质量），避免高出价买来糟糕的用户体验：

$$
\text{rank score}_a = \text{eCPM}_a + \lambda\cdot \text{quality}_a ,
$$

其中 $\lambda$ 由平台调节。广告还要和自然内容争夺信息流位置，这就需要把自然内容的互动价值和广告价值换算成可比较的单位——这也是绝对校准很重要的另一个原因。

### 竞价：第一价格、第二价格、GSP、VCG {#auctions-first-price-second-price-gsp-vcg}

**单个广告位。** 按 eCPM 排序；最高者获胜。

- *第一价格*：赢家支付自己的出价。竞价者有动机把出价压到真实价值以下，而最优的压价幅度取决于对竞争对手的猜测——不稳定且复杂。
- *第二价格（Vickrey）*：赢家支付仍能获胜的最低出价，也就是由第二名决定。按真实价值出价是一个**占优策略**：出价过高只会冒着付出超过物品价值的风险；出价过低只会冒着输掉一场你本来愿意按该价格拿下的竞价的风险。这个性质称为**真实性**（truthfulness）或**激励相容**（incentive compatibility）。

**多个广告位：GSP。** 搜索广告和信息流广告有多个曝光度不同的位置。把点击概率建模为可分离的：$P(\text{click on } a \text{ in slot } s) = \theta_s \cdot q_a$，其中 $\theta_s$ 是位置 $s$ 的位置效应（随 $s$ 递减），$q_a$ 是广告自身的 pCTR。按 $b_a q_a$ 给广告排序。在**广义第二价格**（generalized second-price，GSP）竞价里，位置 $s$ 的广告每次点击支付的是能保住其位置的最低出价：

$$
p_s = \frac{b_{s+1}\, q_{s+1}}{q_s},
$$

其中下标 $s+1$ 指排在紧接着下面的那条广告。通俗地说：你付的钱刚好让你的 eCPM 与下一条广告的 eCPM 持平。

**VCG。** 在多个广告位时 GSP *并不*满足真实性：广告主有时可以通过把出价压到真实价值以下，拿到一个更便宜、更靠后的位置而获益。**Vickrey–Clarke–Groves** 机制是满足真实性的：每个赢家支付她给其他人造成的**外部性**——如果她不在，其他人总共能得到的价值，减去她在场时他们得到的价值。VCG 在理论上很干净，但向广告主解释起来更难；GSP 在历史上是搜索广告的行业标准，而随着自动出价（平台代广告主出价）的出现，设计上的取舍又发生了变化。例子演算里会把两者都算一遍。

### 为什么校准至关重要 {#why-calibration-is-critical}

在自然内容排序里，对分数做任何单调变换都不会改变排序。但在广告里：

- **出价类型混合。** CPC 广告用 pCTR，CPA 广告用 pCTR × pCVR，CPM 广告两者都不用。如果 pCVR 高估了 2 倍，每个 CPA 广告的 eCPM 相对于 CPC 和 CPM 广告都翻倍，竞价就被扭曲了。
- **效果目标按曝光计费。** 当平台按曝光收费，而广告主按转化出价时（自动出价很常见），收费是由预测概率算出来的。高估意味着广告主为每个实际转化多付了钱；低估则意味着平台少收了钱、也少投放了。
- **广告 vs 自然内容与底价。** 决定到底要不要展示广告，要与自然内容或底价（reserve price）比较，比较的是绝对价值。

所以广告系统会按人群分段跟踪校准情况（见第 [4.2](02-ranking-and-ctr-models.md) 章），并维护在线校准层。

### 转化建模 {#conversion-modeling}

**样本选择偏差与 ESMM。** CVR 模型 $P(\text{conv}\mid\text{click})$ 天然只在被点击的曝光上训练，却要用来给*所有*候选曝光打分——这是分布不匹配，而且点击很稀疏。**全空间多任务模型**（Entire Space Multi-task Model，ESMM）利用“曝光 → 点击 → 转化”这条漏斗：

$$
\underbrace{P(\text{click}, \text{conv}\mid x)}_{\text{pCTCVR}} = \underbrace{P(\text{click}\mid x)}_{\text{pCTR}}\times\underbrace{P(\text{conv}\mid \text{click}, x)}_{\text{pCVR}} .
$$

两个塔分别输出 pCTR 和 pCVR（共享 embedding）；模型在**所有曝光**上用两个损失做监督——pCTR 用点击标签，pCTCVR 用“点击且转化”标签。pCVR 从不直接在有偏的被点击子集上训练；它是作为“让乘积对得上”的那个因子被隐式学出来的。共享 embedding 让稀疏的 CVR 任务可以借用稠密的 CTR 任务。

**延迟反馈。** 转化可能在点击之后几小时甚至几天才到。在流式训练（见 4.1 章）里，一条曝光必须在它的转化发生之前就被打上标签——这就是*假负例*。可选做法：

1. 先等一个归因窗口再训练：标签正确，但模型陈旧。
2. 立即训练，之后再修正：先把样本作为负例插入，等转化到达时再作为正例重新插入，并用重要性权重修正由此带来的偏差。
3. 显式建模延迟：设 $p(x)$ 是最终转化概率，$F(e\mid x)$ 是转化在经过时间 $e$ 之内到达的概率，那么 $P(\text{observed conversion by } e) = p(x)\,F(e\mid x)$，两者联合学习。

### 预算、预算平滑与自动出价 {#budgets-pacing-and-auto-bidding}

广告主很少手动逐场竞价出价。他们设定一个**预算**和一个**目标**，由平台代为出价：

- **目标 CPA（tCPA）：**“在平均每次转化成本 ≤ $T$ 的前提下，拿到尽可能多的转化”。
- **目标 ROAS：**“收入 / 花费 ≥ $R$”。
- **预算内最大化投放：** 花掉每日预算 $B$，使转化最大化。

**把出价看成优化问题。** 考虑曝光 $i = 1..n$，价值为 $v_i$（例如 $T\cdot\text{pCVR}_i$ 这种期望转化价值），价格为 $c_i$（市场价，广告主在第二价格竞价中赢下后要付的钱）。选择要赢下哪些，$x_i\in\{0,1\}$：

$$
\max_{x}\ \sum_i x_i v_i \quad \text{s.t.}\quad \sum_i x_i c_i \le B .
$$

对预算约束引入拉格朗日乘子 $\lambda \ge 0$ 进行松弛：最大化 $\sum_i x_i (v_i - \lambda c_i)$。当且仅当 $v_i > \lambda c_i$，即 $c_i < v_i/\lambda$ 时赢下曝光 $i$。在第二价格竞价里，价格低于你的出价时你恰好会赢，所以最优策略是

$$
b_i = \frac{v_i}{\lambda}.
$$

通俗地说：用价值除以一个数来出价，这个数就是预算的“影子价格”。$\lambda$ 大（预算紧）→ 出价低，只买便宜且高价值的曝光；$\lambda$ 小 → 出价高。整个问题归结为找到合适的 $\lambda$——而它取决于你看不到的未来流量。

**把预算平滑看成反馈控制。** 在一天内运行一个控制器：把实际花费与目标花费曲线（例如与预期流量成正比）比较，并调整 $\lambda$——花得过快就调大，花得不够就调小。典型做法是 PI(D) 控制器或乘性更新。tCPA 和 ROAS 约束以同样的方式引入更多乘子。

**基于学习的出价器。** 由于流量非平稳且各约束相互影响，平台会在历史竞价日志上用 RL 训练出价策略。

### 生成式自动出价（AIGB） {#generative-auto-bidding-aigb}

RL 出价器把出价建模为马尔可夫决策过程，在长时间跨度和有噪声的环境下比较吃力。**AIGB**（阿里）把自动出价重新表述为对*整条出价轨迹的生成式建模*：它的 DiffBid 模型是一个条件**扩散模型**，以期望回报和约束为条件生成轨迹，避免了误差在各时间步之间传播。它报告了在阿里广告平台的线上 A/B 实验中 GMV +2.81%、ROI +3.36%。它呼应了离线 RL 中“把决策当作条件生成”的思路。

### 端到端生成式广告（EGA-V2） {#end-to-end-generative-advertising-ega-v2}

正如 OneRec 取代了自然内容的级联（见第 [4.6](06-end-to-end-generative-recommenders.md) 章），**EGA-V2**（美团）为广告提出了一个统一的生成式模型，联合处理用户兴趣、POI（兴趣点）与创意生成、广告分配以及付费。它使用层级 tokenization 和多 token 预测、感知排列的奖励模型和 token 级出价，而且——这是广告特有的部分——用可微的事后遗憾最小化（ex-post regret minimization）把分配与**付费**解耦，目标是在 POI 层面达到*近似激励相容*（即竞价那一节讲的真实性）。它的评估是离线的，但它很好地勾勒出“生成式模型里的竞价”必须解决哪些问题。

### 生成式广告创意 {#generative-ad-creatives}

各大广告平台现在都提供由 LLM 和扩散模型生成的广告文案、背景、图片变体和视频。这把创意选择变成了一个带有生成式素材库的推荐问题：生成许多变体，按用户人群预测效果，投放最好的——同时落实品牌安全和广告主审核。

### 隐私与更少信号下的建模 {#privacy-and-modeling-with-fewer-signals}

平台和监管的变化（例如 iOS 上 Apple 的 App Tracking Transparency，要求跨 App 追踪必须获得用户许可）减少了广告系统可用的用户级转化数据。对建模的影响：

- 转化标签变成**聚合的、延迟的或带噪的**（例如由保护隐私的归因 API 上报），因此模型要校准到聚合计数，而不是个体标签。
- 更依赖**第一方的平台内信号**（用户在平台自身上的行为）、**上下文信号**，以及广告主的**服务端转化回传**。
- **建模转化**：预测无法被直接观测到的转化，这抬高了对校准和验证的要求。

### 大型广告基础模型 {#large-ads-foundation-models}

Meta 的 **GEM**（Generative Ads Recommendation Model）被描述为一个 LLM 规模的广告基础模型，在数千块 GPU 上用广告和自然内容的互动数据训练。它的架构对非序列特征扩展了 Wukong 式的堆叠因子分解机（见第 [4.4](04-scaling-recommendation-models.md) 章），并用交错的序列层和交叉特征层来建模长行为序列。GEM 的关键作用是**知识迁移**：它通过直接迁移和层级迁移（蒸馏、表示学习、参数共享）来教数百个下游垂直广告模型。Meta 的工程博客报告称，上线后在 2025 年第二季度，Instagram 的广告转化提升了 5%，Facebook Feed 提升了 3%。这就是 SUM 模式（见第 [4.3](03-sequential-user-modeling.md) 章）被推到基础模型的规模。

## 代码 {#code}

多广告位可分离竞价下的 GSP 与 VCG 价格，以及 pCTR 校准失准的影响。

```python
def run_auction(ads, slot_ctr, pctr_override=None):
    """ads: dict name -> (bid_per_click, true_ctr)。slot_ctr: 位置效应 theta_s。
    按 bid * 预测 CTR 排序。返回带 GSP 和 VCG 价格的分配结果。"""
    pred = {a: (pctr_override or {}).get(a, q) for a, (b, q) in ads.items()}
    order = sorted(ads, key=lambda a: ads[a][0] * pred[a], reverse=True)
    S = len(slot_ctr)
    ecpm = lambda a: ads[a][0] * pred[a]                  # 每单位位置 CTR 的预测价值

    def welfare(ranking):                                 # 某个广告位分配的预测价值
        return sum(slot_ctr[s] * ecpm(a) for s, a in enumerate(ranking[:S]))

    rows = []
    for s, a in enumerate(order[:S]):
        nxt = order[s + 1] if s + 1 < len(order) else None
        gsp = ecpm(nxt) / pred[a] if nxt else 0.0         # 每次点击的价格
        others_without = welfare([x for x in order if x != a])
        others_with = welfare(order) - slot_ctr[s] * ecpm(a)
        vcg = (others_without - others_with) / (slot_ctr[s] * pred[a])
        true_clicks = slot_ctr[s] * ads[a][1]             # 每次曝光的实际期望点击数
        rows.append((s + 1, a, round(gsp, 2), round(vcg, 2), true_clicks))
    return rows

ads = {"A": (2.00, 0.05), "B": (4.00, 0.02), "C": (1.00, 0.06), "D": (3.00, 0.01)}
theta = [1.0, 0.5]
for label, override in [("calibrated", None), ("B's pCTR 2x too high", {"B": 0.04})]:
    rows = run_auction(ads, theta, override)
    revenue = sum(gsp * clicks for _, _, gsp, _, clicks in rows)
    value = sum(clicks * ads[a][0] for _, a, _, _, clicks in rows)
    print(label)
    for slot, a, gsp, vcg, _ in rows:
        print(f"  slot {slot}: ad {a}  GSP ${gsp:.2f}/click   VCG ${vcg:.2f}/click")
    print(f"  GSP revenue per impression ${revenue:.3f}, advertiser value created ${value:.3f}")
```

用单个乘子 $\lambda$ 做预算平滑：出价 $= v/\lambda$，每小时朝目标花费曲线调整一次。

```python
import random
B, T_CPA, HOURS, PER_HOUR = 1000.0, 50.0, 24, 10000         # 预算 $1000，目标 CPA $50
traffic = [0.5 + 0.5 * (8 <= h <= 22) for h in range(HOURS)]   # 白天流量更大
target_curve = [sum(traffic[:h + 1]) / sum(traffic) * B for h in range(HOURS)]

def simulate(pacing):
    random.seed(0)
    lam, spend, conv, out_at = 1.0, 0.0, 0.0, None
    for h in range(HOURS):
        for _ in range(int(PER_HOUR * traffic[h])):
            pcvr = random.betavariate(1, 400)                   # 该曝光的转化概率
            price = random.lognormvariate(-4.0, 0.8)            # 竞争对手的第二价格
            value = T_CPA * pcvr                                # 按目标 CPA 计算的期望价值
            if value / lam > price and spend + price <= B:      # 第二价格：出价高于价格就赢
                spend += price; conv += pcvr
        if out_at is None and spend > 0.99 * B:
            out_at = h
        if pacing:                                              # 乘性控制器
            err = (spend - target_curve[h]) / B                 # 领先于计划 -> err > 0
            lam *= 2.0 ** (8 * err)                             # 花多了 -> 调大 lambda
    return spend, conv, out_at

for pacing in (False, True):
    s, c, h = simulate(pacing)
    print(f"pacing={pacing}: spend ${s:.0f}, budget exhausted at hour {h}, "
          f"expected conversions {c:.0f}, CPA ${s / c:.2f}")
```

不做平滑时（$\lambda = 1$，出价 = 价值），广告主一开始几乎什么都赢，预算在第 9 小时左右耗尽，错过了晚上的流量。做了平滑，$\lambda$ 上升，出价器变得挑剔（只要价值与价格之比高的曝光），花费贴着目标曲线走，在这个仿真里同样的 \$1,000 大约多买到 45% 的期望转化。

## 实例演算 {#worked-example}

**一场 GSP 竞价。** 两个广告位，位置效应 $\theta_1 = 1.0$，$\theta_2 = 0.5$。四个 CPC 广告主：

| 广告 | 每次点击出价 | pCTR $q$ | $b\cdot q$（每次曝光） | eCPM |
|---|---|---|---|---|
| A | \$2.00 | 0.05 | \$0.10 | \$100 |
| B | \$4.00 | 0.02 | \$0.08 | \$80 |
| C | \$1.00 | 0.06 | \$0.06 | \$60 |
| D | \$3.00 | 0.01 | \$0.03 | \$30 |

- **排序**：A（位置 1），B（位置 2）。B 的出价最高，但排第二；A 获胜是因为她被点击的可能性更大。
- **GSP 价格**（每次点击）：A 付 $0.08/0.05 = \$1.60$（低于她 \$2.00 的出价）；B 付 $0.06/0.02 = \$3.00$。
- **每次曝光的期望收入**：A：$\theta_1 q_A p_A = 1.0\times0.05\times1.60 = \$0.080$；B：$0.5\times0.02\times3.00 = \$0.030$。合计 \$0.110，即每千次曝光 \$110。

**VCG 价格**（相同出价，把出价当作真实价值）：

- *没有 A 时*：B 得位置 1（价值 $1.0\times0.08 = 0.08$），C 得位置 2（$0.5\times0.06 = 0.03$）：其他人的总福利为 0.11。*有 A 时*：其他人只剩 B 在位置 2：0.04。A 的支付 $= 0.11 - 0.04 = \$0.07$ 每次曝光 $= 0.07/0.05 = \$1.40$ 每次点击。
- *没有 B 时*：A 位置 1（0.10），C 位置 2（0.03）：0.13。*有 B 时*：其他人得到 A 的 0.10。B 支付 $\$0.03$ 每次曝光 $= 0.03/(0.5\times0.02) = \$3.00$ 每次点击。
- VCG 收入 $= \$0.100$ 每次曝光，GSP 为 \$0.110。在这些（真实的）出价下，GSP 向头部位置多收了钱；差额就是 A 在 GSP 中有动机压低出价的原因。

**校准失准。** 假设模型把 B 的 pCTR 高估为 0.04（真实值 0.02）。B 每次曝光的预测价值变成 $4.00\times0.04 = 0.16$，于是 B 拿到位置 1，A 掉到位置 2。

- B 付 $0.10/0.04 = \$2.50$ 每次点击；A 付 $0.06/0.05 = \$1.20$。
- 每次曝光的实际收入：B $1.0\times0.02\times2.50 = 0.050$；A $0.5\times0.05\times1.20 = 0.030$；合计 \$0.080，而不是 \$0.110（−27%）。
- 实际创造的价值：B $1.0\times0.02\times4 = 0.08$ + A $0.5\times0.05\times2 = 0.05$ = 0.13，而不是 0.14：对所有人来说都是更差的分配。

注意，如果*所有* pCTR 都被高估 2 倍，排序和 GSP 的每次点击价格都不会变（它们只取决于比值）。伤害来自各广告之间的*相对*校准失准，以及绝对校准失准——只要预测值要与其他单位的出价（CPM、CPA 计费、自然内容、底价）比较。

**自动出价。** 一个 tCPA 为 \$50 的广告主看到一次曝光，其 pCTR × pCVR $= 0.002$。价值 $v = 50\times0.002 = \$0.10$ 每次曝光。预算乘子 $\lambda = 1.25$ 时，出价为 $0.10/1.25 = \$0.08$；如果第二价格是 \$0.06，广告主获胜并支付 \$0.06。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **自动出价是默认选项**：大平台上大多数广告主的花费都通过平台代运行的出价，朝 tCPA / ROAS / 预算目标投放，用基于乘子的控制加上学习得到的策略来求解；生成式出价器（AIGB）是前沿方向。
- **排序模型就是第 4.3–4.4 章的大型推荐模型栈**（序列模型、规模化骨干），上面再加按人群分段的校准层。
- **广告基础模型**（Meta 的 GEM、SUM 式用户模型）把知识迁移给数百个下游模型。
- **广告中的生成式召回**（百度的 COBRA、阿里的 URM）和**端到端生成式广告**（EGA-V2）正在兴起。
- **生成式创意**已成为广告主的标准工具；**隐私变化**推动建模转向聚合的、平台内的和建模出来的信号。

## 常见误区 {#common-pitfalls-misconceptions}

- **“AUC 越高 = 收入越高。”** 收入取决于校准和竞价机制；AUC 两者都不管。
- **只在点击上训练 CVR**，却给所有曝光打分（样本选择偏差）。
- **在转化延迟的情况下，把新曝光不加修正地当作负例。**
- **以为 GSP 是真实的。** 只有单个广告位时才是（那时它就是第二价格），一般情况下并不是。
- **忽视预算平滑**：价值模型再完美，没有预算控制，就会早上把预算花完，错过晚间流量。
- **只优化短期收入**：广告负载和广告质量会影响用户留存，而留存决定长期收入。

## 本章要点 {#key-takeaways}

- 广告按每次曝光的期望价值排序：eCPM $=$ 出价 × pCTR（× pCVR），再加质量项。
- 第二价格式竞价收取最低获胜价；GSP 把它推广到多个广告位；VCG 是基于外部性的、满足真实性的多广告位机制。
- 校准必不可少，因为概率要乘上金额，并且要在不同出价类型之间比较。
- 转化建模必须处理样本选择偏差（ESMM）和延迟反馈。
- 自动出价归结为按 $v/\lambda$ 出价并控制 $\lambda$；生成式出价器、端到端生成式广告和广告基础模型是前沿。

## 自测 {#check-your-understanding}

??? question "为什么在单广告位第二价格竞价中，真实出价是占优策略？"
    你付的价格由别人决定，而不是由你的出价决定；你的出价只决定你能不能赢。按真实价值出价，意味着价格低于你的价值时你恰好会赢。出价更高只会多赢下一些你付得比价值还多的竞价；出价更低只会丢掉本来有利可图的竞价。

??? question "在 GSP 例子演算里，如果 C 的出价升到 \$1.50，A 每次点击付多少？"
    C 每次曝光的价值变成 $1.50\times0.06=0.09$，于是排序为 A（0.10）、C（0.09）、B（0.08）。A 每次点击付 $0.09/0.05=\$1.80$，B 被挤出这两个位置。

??? question "ESMM 如何避免在有偏的“只有被点击”子集上训练 pCVR？"
    它把 pCTCVR 写成 pCTR × pCVR，在所有曝光上，用点击标签监督 pCTR，用“点击且转化”标签监督 pCTCVR。pCVR 作为“让乘积对得上”的那个因子被隐式学出来，用到的是整个曝光空间。

??? question "在预算平滑里，控制器调大 λ 时出价会怎样，什么时候应该调大？"
    出价 $v/\lambda$ 降低，广告主赢下的曝光变少、变便宜。当实际花费领先于目标花费曲线时，控制器就应调大 λ。

??? question "为什么所有 pCTR 统一高估 2 倍，GSP 的排序不变，却仍可能造成伤害？"
    排序和每次点击的 GSP 价格只取决于 CPC 广告之间预测 CTR 的比值。但只要预测值要与 CPM 出价比较、要换算成按效果出价的每次曝光费用、要与自然内容或底价比较，或要进入自动出价器的价值估计，绝对值就很重要。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | 大规模线上广告 CTR：FTRL-Proximal、校准、节省内存、监控。 |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + LR、归一化熵、数据新鲜度，以及负采样后的重新校准公式。 |
| :material-star: [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (ESMM)](https://arxiv.org/abs/1804.07931) — Ma et al. (Alibaba), 2018 | 通过在整个曝光空间上学习，解决 CVR 建模中的样本选择偏差。 |
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | 在数百个广告模型之间共享的基础用户模型。 |
| :material-new-box: [AIGB: Generative Auto-bidding via Conditional Diffusion Modeling](https://arxiv.org/abs/2405.16141) — Guo et al. (Alibaba), KDD 2024 | 把自动出价看作对出价轨迹的条件扩散，取代 RL 出价器。 |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | 对完整的广告分配 + 付费流程做生成式建模。 |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | 部署在广告平台上的生成式召回。 |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | 部署在广告中的基于 LLM 的召回。 |
| :material-new-box: [Meta's Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) — Meta engineering blog, 2025 | 一个 LLM 规模的广告基础模型，以及它的知识如何迁移到下游模型。 |
