# 4.7 推荐中的 LLM

!!! abstract "本章你将学到"
    - LLM 在推荐系统里可以扮演的三种角色：(a) 直接当推荐模型，(b) 离线的特征 / 知识提供者，(c) 用户 / 物品编码器
    - 把 LLM 当推荐模型时的具体 prompt 模板（pointwise、listwise），以及如何解析和校验输出
    - KAR、CTRL、ClickPrompt 和 QARM 如何以（接近）零在线成本把 LLM 的知识注入传统模型
    - HLLM、User-LLM、LC-Rec 和 URM 如何把 LLM 用作编码器或召回器，以及 360Brew 和 RecGPT 如何把 LLM 推进到线上系统的核心
    - 为什么延迟和成本决定了设计，附一个成本估算例子和标准做法：离线预计算、缓存、蒸馏

## 为什么重要 {#why-this-matters-the-problem}

LLM 对世界懂得很多：Nikon Z8 是一台需要配镜头的相机机身，看马拉松训练文章的用户可能想买跑鞋，一部电影讲的是什么。传统推荐模型对这些一无所知，它们只看得到 ID 和共现。LLM 还能*推理*（“这位用户上个月买了婴儿床，所以婴儿服装很可能相关”）和*解释*。看起来它们是完美的推荐模型。

但它们有三个问题。第一，缺少**协同信号**：它们不知道看过视频 8,124,551 的人也看过视频 2,003,117。第二，**很贵**：排序阶段每次请求要给几百个候选打分，QPS 在 10^4–10^5，延迟要控制在几十毫秒内。第三，物品库巨大且每天都在变，而 LLM 的词表和知识是固定的。

所以实际的问题并不是“用 LLM 还是用传统模型”，而是 LLM 放在系统的**哪一环**，才能在单位延迟和成本上带来最大价值。本章按角色来梳理这个领域。

## 直觉 {#intuition}

想想在一家商店里使用专家顾问的三种方式：

1. **(a) 推荐模型：** 顾问和每位顾客交谈并挑选商品。判断力极好，但对每位顾客都又慢又贵，而且她不知道上周哪些商品一起卖得好。
2. **(b) 知识提供者：** 顾问在夜里为每件商品和每位常客写笔记；店里快速运转的系统把这些笔记当作额外特征来读。白天没有任何延迟。
3. **(c) 编码器：** 顾问*理解*商品和顾客的方式被蒸馏成紧凑的表示（embedding），供店里的系统使用。

线上系统主要使用 (b) 和 (c)；对于高价值、低 QPS 的场景，或者模型被做得足够高效之后，(a) 的用法也越来越多。

## 工作原理 {#how-it-works}

```text
             (b) 离线知识 / 特征                          (c) 编码器
   LLM ──► 关于用户和物品的文本 ──► 向量 ─┐     LLM ──► 物品 / 用户 embedding ─┐
                                          ▼                                    ▼
   请求 ──► 召回 ──► 排序（传统或规模化模型） ──► 重排 ──► 信息流
               ▲                       ▲
               └──── (a) LLM 作为召回器 / 推荐模型（生成物品或分数） ────┘
```

### (a) LLM 作为推荐模型 {#a-the-llm-as-recommender}

**文本到文本的建模方式（P5）。** 把每个推荐任务——评分预测、序列推荐、解释、评论摘要——都表达成一个文本 prompt 加一个文本答案，用一个 T5 模型在所有任务上训练。用户和物品以 token 的形式出现在 prompt 中。

**零样本 / 少样本 prompting。** 可以直接去问一个通用的对话 LLM。“Is ChatGPT a Good Recommender?” 这项研究在 Amazon Beauty 数据的五个推荐场景上，不经微调、只用 prompt 评估了 ChatGPT：它在一些任务上表现不错（在人工评估中解释质量也不错），在另一些任务上只达到基线水平。这一时期得出的总体教训是：没有协同信号，单靠 prompting 在准确率上很难击败训练过的推荐模型，但它对解释和冷启动有帮助。

**在推荐数据上做指令微调（TALLRec）。** 把交互数据转成指令样本（“给定用户喜欢和不喜欢的物品，他会喜欢目标物品吗？回答 Yes 或 No。”），再用 LoRA 微调（见[第 2 部分](../part2/index.md)）。TALLRec 报告称，即使训练样本不足 100 条，这也能显著提升 LLM 在电影和图书领域的推荐效果，用 LLaMA-7B 在单张 RTX 3090 上即可运行，并且能跨领域泛化。pointwise 分数就是 “Yes” 的概率：

$$
\hat p(u,i) = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})},
$$

其中 $z_{\text{Yes}}, z_{\text{No}}$ 是 LLM 在答案位置上这两个答案 token 的 logit。通俗地说：模型对 “Yes” 的相对置信度，变成了一个类似点击率的概率，可以用来排序（也可以做校准，见第 [4.2](02-ranking-and-ctr-models.md) 章）。

**Pointwise 与 listwise prompt。** *Pointwise*：每个（用户，候选）一个 prompt，输出一个概率——干净，但需要 $N$ 次前向。*Listwise*：一个 prompt 里放用户历史和 $N$ 个候选，要求输出排序后的列表——只要一次前向，但输出需要解析，可能漏掉或编造物品，而且对 prompt 里候选的顺序敏感（LLM 和用户一样有位置偏差）。

**加入协同语义。** 纯文本会丢失 ID 层面的信号。**LC-Rec** 把语义 ID token（见第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章）加入 LLM 的词表，并通过多个微调任务把它们与语言对齐，让 LLM 能直接生成物品，同时了解内容结构和协同结构。PLUM（第 [4.6](06-end-to-end-generative-recommenders.md) 章）则通过继续预训练在 YouTube 的规模上做到了这一点。

**一个 LLM 取代众多排序模型（360Brew）。** LinkedIn 的 360Brew 是一个 150B 参数的 decoder-only 模型，带**文本接口**：会员资料、行为和社交关系被转写成文字，任务用自然语言描述。报告称，这一个模型在 LinkedIn 各个场景中解决了 30 多个预测任务，在离线指标上与专门的线上模型相当或更好，并且不需要针对任务微调——而那些模型各自由团队花了多年构建和维护。它的卖点与其说是“更高的准确率”，不如说是“用一个模型取代庞大的、靠特征工程堆出来的模型动物园”。

**LLM 驱动的意图（RecGPT）。** 阿里的 RecGPT 把 LLM 放在淘宝首页推荐的中心：LLM 从行为中挖掘用户意图，预测表达该意图的物品标签，标签驱动召回，LLM 再生成解释。训练时通过增强推理的预对齐和自训练，把通用 LLM 对齐到这些任务上，并由人机协作的评判器来指导。它已在淘宝全量上线，报告描述了对用户（多样性、满意度）、商家和平台的收益。

### (b) LLM 作为离线知识 / 特征提供者 {#b-the-llm-as-offline-knowledge-feature-provider}

最适合落地的模式：**离线**运行 LLM，存下它的输出，作为特征喂给普通的排序模型。

- **KAR（Knowledge Augmentation from LLMs）。** 对每个用户，让 LLM 推理其偏好（“因子分解式 prompting”把问题拆成题材、价格敏感度等方面）；对每个物品，让 LLM 给出事实性知识。文本编码器把生成的文本变成向量，再由一个小型的混合专家适配器（hybrid-expert adaptor）把它们映射成可用于*任何* CTR 模型的增强特征。所有 LLM 的工作都是预先算好并存起来的，所以在线推理成本就是一次查表。KAR 报告称，在华为的新闻和音乐推荐平台上，线上 A/B 实验分别带来 7% 和 1.7% 的提升。
- **CTRL。** 把每个表格样本转成文本；用对比学习把语言模型对该文本的表示与 CTR 模型对表格特征的表示对齐；然后微调，并且**只部署 CTR 模型**。LLM 的语义知识在训练阶段迁移过来，线上不增加任何延迟。
- **ClickPrompt。** CTR 模型产生 soft prompt 来调节语言模型，从而在训练中把协同信号和语义信号结合起来；推理时同样可以只依赖 CTR 一侧。
- **QARM（快手）。** 多模态 LLM 的物品 embedding 通常是为通用目标训练出来的冻结特征。QARM 把它们与业务信号（用户–物品行为）*对齐*，再**量化**成离散码，排序模型把这些码当作可训练的类 ID 特征使用——同时解决了目标不匹配和“无法被排序模型的梯度更新”这两个问题。

??? example "KAR 式流水线，一步步来"
    1. **按用户 prompt（离线，每天一次或在画像变化时）。** 把用户近期历史以文本形式给 LLM，逐方面询问偏好——例如“这位用户偏好哪些题材，为什么？他们对价格有多敏感？他们在为什么场合购物？”把问题拆成多个因子，比一个开放式问题能让推理更具体。
    2. **按物品 prompt（离线，每个新物品一次）。** 询问事实性知识：“描述这个商品的类目、典型买家、互补商品和关键属性。”
    3. **编码。** 用文本编码器对每段生成的文本编码成向量，存入以用户 ID / 物品 ID 为 key 的 key–value 存储。
    4. **适配。** 一个小型可训练的适配器（KAR 用的是混合专家适配器）把存好的向量映射成增强特征向量，与 CTR 模型联合训练。
    5. **服务。** 请求到来时，查出存好的向量，与常规特征一起喂给 CTR 模型。线上不会调用 LLM。

### (c) LLM 作为用户 / 物品编码器 {#c-the-llm-as-useritem-encoder}

- **HLLM（字节跳动）。** 层级结构的两个 LLM：**Item LLM** 读取物品文本并输出物品 embedding；**User LLM** 读取用户历史里的物品 embedding 序列，预测下一个物品的 embedding。预训练的 LLM 权重有帮助，微调帮助更大，而且这种方法每个组件都可以扩展到 7B 参数，并报告了线上 A/B 收益。物品 embedding 可以预先算好，所以每个用户只需运行 User LLM（它的结果本身也可以缓存）。
- **User-LLM（Google）。** 长的用户时间线以文本形式表示代价很高。User-LLM 在交互序列上预训练一个用户编码器，把得到的**用户 embedding** 通过交叉注意力（用 Perceiver 层压缩）送入 LLM，和视觉语言模型输入图像 embedding 的方式一样。它报告称，相比文本 prompt 的上下文方式，推理加速最高可达 78.1 倍，在需要深度理解用户的任务上提升最高达 16.33%。
- **URM（阿里）。** 把 LLM 用作**通用召回器**：多查询表示、矩阵分解和概率采样使它能在几十毫秒内，针对任意给定目标，从几千万候选中生成一个集合；报告称在一个线上广告平台的核心指标上提升了 3%。

### 延迟与成本问题——设计模式 {#the-latency-and-cost-problem-design-patterns}

| 模式 | 线上跑什么 | 例子 |
|---|---|---|
| 离线预计算 | 对 LLM 输出（物品知识、用户摘要、embedding）做查表 | KAR、HLLM 的物品侧、QARM 的码 |
| 异步 / 缓存的用户状态 | LLM 在后台针对用户事件运行；请求读取最新缓存 | 按 session 或按天刷新的用户侧 LLM embedding |
| 蒸馏 / 对齐之后丢掉 LLM | 只有小的 CTR 模型 | CTRL |
| 每次请求一次调用，而不是每个候选一次 | 一次生成式前向，产出一个列表或集合 | URM、生成式召回、OneRec 式解码 |
| 低 QPS、高价值场景 | 直接用 LLM | 解释、对话式推荐、意图挖掘（RecGPT） |

### 各个角色如何在一个系统里组合 {#how-the-roles-combine-in-one-system}

一个 2025 年的现实系统会同时用到几种角色：

1. **物品侧，离线：** 多模态 LLM 为每个新物品（文本、缩略图、音频转写）生成 embedding；该 embedding 与互动数据对齐，并量化成语义 ID 或码（QARM 式，见第 [4.5](05-semantic-ids-and-generative-retrieval.md) 章）。
2. **用户侧，离线 / 异步：** LLM 把长期兴趣总结成文本或 embedding，每天或每个 session 刷新（KAR、User-LLM 式）。
3. **召回，在线：** 召回源之一是基于语义 ID 的生成式召回器，可能由 LLM 初始化（URM、PLUM）。
4. **排序，在线：** 规模化的序列排序模型（见第 [4.4](04-scaling-recommendation-models.md) 章）把 LLM 导出的物品码和用户特征当作普通输入来使用。
5. **展示，在线低 QPS：** LLM 撰写解释，或支撑一个对话式入口（RecGPT 式）。

每一次 LLM 调用都被放在这样的位置：成本只在每个物品一次、每个用户每天一次或每次请求一次时支付——绝不是每个候选一次。

## 代码 {#code}

LLM 作为推荐模型的 prompt 模板，以及对 listwise 答案的稳健解析。这段代码与具体模型无关：可以把 prompt 接到任何对话或补全类 LLM 上。（prompt 本身是发给 LLM 的输入，因此保持英文原样。）

```python
import re

def pointwise_prompt(liked, disliked, target):
    """TALLRec 式二元问题；分数 = 答案位置上 P('Yes')。"""
    return (
        "You are a recommendation assistant.\n"
        f"Items the user liked: {'; '.join(liked)}.\n"
        f"Items the user disliked: {'; '.join(disliked)}.\n"
        f"Will the user like the target item \"{target}\"? Answer with Yes or No.\n"
        "Answer:"
    )

def listwise_prompt(history, candidates, k=3):
    """一次调用给所有候选排序。候选用短 ID 标记，保证解析可靠。"""
    lines = [f"[{j}] {c}" for j, c in enumerate(candidates)]
    return (
        "The user recently interacted with, in order:\n"
        + "\n".join(f"- {h}" for h in history)
        + "\n\nCandidate items:\n" + "\n".join(lines)
        + f"\n\nRank the {k} candidates the user is most likely to choose next. "
        "Reply with candidate numbers only, best first, e.g. [4], [1], [7]."
    )

def parse_listwise(answer, n_candidates, k):
    """按顺序保留有效且不重复的候选下标；丢弃幻觉出来的或重复的。"""
    seen, out = set(), []
    for m in re.findall(r"\[(\d+)\]", answer):
        j = int(m)
        if 0 <= j < n_candidates and j not in seen:
            seen.add(j); out.append(j)
    return out[:k]

def yes_probability(logit_yes, logit_no):
    """对两个答案 token 做二路 softmax。"""
    import math
    m = max(logit_yes, logit_no)
    a, b = math.exp(logit_yes - m), math.exp(logit_no - m)
    return a / (a + b)

history = ["Trail running shoes", "Hydration vest", "GPS sports watch"]
cands = ["Espresso machine", "Energy gels (12-pack)", "Foam roller", "Office chair", "Headlamp"]
print(listwise_prompt(history, cands))
fake_llm_answer = "[1], [2], [9], [2], [4]"       # 包含一个无效的 [9] 和一个重复项
print(parse_listwise(fake_llm_answer, len(cands), k=3))   # -> [1, 2, 4]
print(round(yes_probability(2.1, 0.3), 3))                # -> 0.858
print(pointwise_prompt(history[:2], ["Office chair"], "Energy gels (12-pack)"))
```

一个用来比较各种部署模式的粗略成本计算器：

```python
def llm_flops(params, tokens):
    return 2 * params * tokens                     # 前向 ≈ 每参数每 token 2 FLOPs

GPU_EFFECTIVE = 1e15 * 0.4                         # 峰值约 1 PFLOP/s，MFU 40%
P = 7e9                                            # 7B 参数的 LLM

pointwise = llm_flops(P, 1000) * 500               # 1k token 的 prompt，每次请求 500 个候选
listwise = llm_flops(P, 1000 + 50 * 20)            # 一个 prompt，50 个候选 x 每个 20 token
item_precompute = llm_flops(P, 200) * 1e8          # 1 亿物品 x 200 token，一次性
user_daily = llm_flops(P, 1000) * 1e8              # 1 亿日活用户 x 1k token，每天一次

print(f"pointwise : {pointwise / GPU_EFFECTIVE:8.2f} GPU-seconds per request")
print(f"listwise  : {listwise / GPU_EFFECTIVE:8.3f} GPU-seconds per request (prefill only)")
print(f"items     : {item_precompute / GPU_EFFECTIVE / 3600:8.0f} GPU-hours, once")
print(f"users     : {user_daily / GPU_EFFECTIVE / 3600:8.0f} GPU-hours per day "
      f"(~{user_daily / GPU_EFFECTIVE / 86400:.0f} GPUs running continuously)")
```

## 实例演算 {#worked-example}

假设一个 7B 的 LLM，一块稠密峰值约 $10^{15}$ FLOP/s、MFU 为 40%（有效 $4\times10^{14}$）的加速器，前向开销约为 $2 \times \text{params} \times \text{tokens}$。

**在线 pointwise LLM 排序。** 1,000 token 的 prompt（用户历史），每次请求 500 个候选：
$2 \times 7\times10^9 \times 1000 \times 500 = 7\times10^{15}$ FLOPs 每次请求 $\Rightarrow$ 每次请求 17.5 GPU 秒。在 $10^4$ QPS 下需要 175,000 块 GPU。不可行。（用前缀 KV cache 共享用户前缀，见[第 3 部分](../part3/index.md)，能帮上很大忙，但候选仍然会增加 token 数和延迟。）

**Listwise，一次调用。** 1,000 个历史 token + 50 个候选 × 20 token = 2,000 token：$2\times7\times10^9\times2000 = 2.8\times10^{13}$ FLOPs $\approx 0.07$ GPU 秒的 prefill，外加解码答案。对低 QPS 场景可行，但对 $10^5$ QPS 的主信息流来说太慢太贵。

**离线预计算。** $10^8$ 个物品 × 200 token 的物品知识：$2.8\times10^{20}$ FLOPs $\approx 7\times10^5$ GPU 秒 $\approx 194$ GPU 小时，一次性（外加每天的新物品）。$10^8$ 个日活用户 × 1,000 token 的用户摘要：$1.4\times10^{21}$ FLOPs $\approx 972$ GPU 小时/天 $\approx$ 持续运行 40 块 GPU。在线成本：一次 key–value 查询，以及排序模型的输入稍微变大一点。

这些数量级解释了业界的做法：**LLM 离线运行，或者每个用户运行一次，而不是每个候选一次**，除非把模型变成一次前向就产出整份列表的生成式召回 / 推荐模型。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **离线 LLM 特征和 embedding**（KAR 式知识、像 QARM 那样与行为对齐的多模态物品 embedding）是部署最广泛的模式。
- 用 **LLM 编码物品**已是冷启动的标准做法；基于 LLM 的用户编码器（HLLM、User-LLM）也在增多。
- **以 LLM 为骨干的生成式召回**（URM、PLUM）和 **LLM 原生排序模型**（360Brew）已在大公司上线或接近上线。
- **意图与解释**（RecGPT）是 LLM 推理最能被用户感知到的地方。
- 大多数系统混用多种角色：LLM 导出的物品分词器、LLM 预计算的特征，以及一个规模化的序列排序模型。

!!! tip "为你的系统选择角色"
    从 (b) 开始：离线 LLM 特征试起来成本低，也容易做 A/B 实验。当冷启动或长尾物品很重要、且物品文本 / 图像信息丰富时，转向 (c)。对于低 QPS、高价值的交互（搜索助手、对话式购物、解释），或者你能把它变成基于语义 ID 的一次前向生成时，再考虑 (a)。

## 常见误区 {#common-pitfalls-misconceptions}

- **期望零样本 LLM 在准确率上击败训练好的推荐模型。** 没有协同信号，它们通常做不到；它们的长处是知识、冷启动和解释。
- **幻觉出来的物品。** 自由文本输出可能点名根本不存在的物品；要约束生成（候选下标、语义 ID 前缀树）并做校验。
- listwise prompt 里的**候选顺序偏差**；可以打乱后聚合，或者最终排序用 pointwise 分数。
- **预训练带来的数据泄漏：** 公开基准（例如电影数据集）可能已经出现在 LLM 的预训练数据里；要在私有数据或按时间切分的数据上评估。
- **忽略每次请求的成本**：选择架构之前，一定要乘上候选数和 QPS。

## 本章要点 {#key-takeaways}

- LLM 以 (a) 推荐模型、(b) 离线知识 / 特征提供者、(c) 用户 / 物品编码器 三种身份进入推荐系统。
- Prompting 能给出知识和解释，但缺少协同信号；指令微调（TALLRec）和语义 ID token（LC-Rec、PLUM）弥补了这一差距。
- KAR、CTRL、ClickPrompt 和 QARM 以接近零的在线成本，把 LLM 的知识迁移到传统排序模型里。
- HLLM、User-LLM 和 URM 把 LLM 用作编码器 / 召回器；360Brew 用一个 LLM 取代模型动物园；RecGPT 把意图挖掘放在核心。
- 成本算术（FLOPs × 候选数 × QPS）迫使 LLM 只能离线运行、按用户运行，或者变成一次前向的生成。

## 自测 {#check-your-understanding}

??? question "为什么 pointwise LLM 排序比 listwise 贵那么多？"
    Pointwise 对每个候选做一次前向，每次都重复处理用户历史 prompt，所以成本与（历史 token 数 × 候选数）成正比。Listwise 在一个 prompt 里只处理一次历史，外加每个候选的一小段描述。

??? question "CTRL 是如何做到传递 LLM 知识而线上服务延迟为零的？"
    它在训练时把 CTR 模型的表示与语言模型的表示对齐（对每个样本的表格视图和文本视图做对比对齐），然后微调，并且只部署轻量的 CTR 模型。

??? question "QARM 解决了冻结的多模态物品 embedding 的哪两个问题？"
    表示不匹配（它们是为通用 NLP / CV 目标训练的，而不是为用户–物品行为训练的）和表示无法更新（作为固定的缓存输入，它们无法被排序模型的梯度更新）。QARM 把它们与行为对齐，并量化成可训练的码。

??? question "利用例子演算，对 100 个候选、500 token prompt 做 pointwise 排序，每次请求要花多少 GPU 秒？"
    $2\times7\times10^9\times500\times100 = 7\times10^{14}$ FLOPs；除以 $4\times10^{14}$ FLOP/s，得到每次请求 1.75 GPU 秒。

??? question "为什么 HLLM 把模型拆成 Item LLM 和 User LLM？"
    物品文本每个物品只编码一次（可缓存），User LLM 随后处理的是长度短得多的物品 embedding 序列，而不是每个历史物品的完整文本，这使得带 LLM 权重的用户建模负担得起。

??? question "为了帮助冷启动物品，你会把 LLM 放在系统的哪一环，为什么？"
    放在物品侧、离线：一旦新物品上传，就用（对齐过的）LLM 把它的内容编码成 embedding 或语义 ID。物品立刻就有了召回和排序可以使用的有意义的表示，每个物品只付一次成本，也不增加请求延迟。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366) — Geng et al., 2022 | 把评分、序列推荐、解释等都转成文本到文本任务，用一个 T5 模型处理。 |
| :material-book-open-variant: [Is ChatGPT a Good Recommender?](https://arxiv.org/abs/2304.10149) — Liu et al., 2023 | 零样本 / 少样本 LLM 推荐：解释上的长处，准确率上的局限。 |
| :material-star: [TALLRec](https://arxiv.org/abs/2305.00447) — Bao et al., 2023 | 用少量推荐数据对 LLM 做 LoRA 微调，就能成为强大的少样本推荐模型。 |
| :material-book-open-variant: [Adapting LLMs by Integrating Collaborative Semantics (LC-Rec)](https://arxiv.org/abs/2311.09049) — Zheng et al., 2023 | 把语义 ID token 加入 LLM 词表，通过多个微调任务与语言对齐。 |
| :material-star: [Towards Open-World Recommendation with Knowledge Augmentation from LLMs (KAR)](https://arxiv.org/abs/2306.10933) — Xi et al. (Huawei), 2023 | 把离线 LLM 推理和物品事实作为特征提供给任何 CTR 模型；已上线。 |
| :material-book-open-variant: [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841) — Li et al. (Huawei), 2023 | 先做对比对齐，再只部署 CTR 模型。 |
| :material-book-open-variant: [ClickPrompt](https://arxiv.org/abs/2310.09234) — Lin et al., 2023 | CTR 模型为语言模型生成 soft prompt，结合协同信号和语义信号。 |
| :material-new-box: [QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739) — Luo et al., 2024 | 把多模态物品表示与业务信号对齐，并量化成排序特征。 |
| :material-star: [HLLM: Hierarchical LLMs for Item and User Modeling](https://arxiv.org/abs/2409.12740) — Chen et al. (ByteDance), 2024 | Item LLM + User LLM；预训练权重有帮助；可扩展到 7B。 |
| :material-book-open-variant: [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) — Ning et al. (Google), 2024 | 通过交叉注意力 / soft prompt 把用户 embedding 送入 LLM。 |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | LLM 作为多查询、多目标的召回器，已在广告中上线。 |
| :material-new-box: [360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://arxiv.org/abs/2501.16450) — LinkedIn, 2025 | 一个带文本接口的 150B decoder LLM 取代许多任务专用的排序模型。 |
| :material-new-box: [RecGPT Technical Report](https://arxiv.org/abs/2507.22879) — Alibaba (Taobao), 2025 | 淘宝上由 LLM 驱动的意图挖掘 → 标签预测 → 召回 → 解释。 |
| :material-star: [A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860) — Wu et al., 2023 | 判别式与生成式 LLM 推荐方法的分类体系。 |
| :material-book-open-variant: [How Can Recommender Systems Benefit from LLMs: A Survey](https://arxiv.org/abs/2306.05817) — Lin et al., 2023 | 按 LLM 用在哪个阶段、如何使用来梳理这个领域。 |
| :material-book-open-variant: [LLMs for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) — Li et al., 2023 | 生成式范式与物品 ID 设计。 |
