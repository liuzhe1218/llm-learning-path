# 第四部分 · LLM 与生成式建模在广告排序与推荐系统中的应用

> **目标：** 理解工业级推荐系统与广告系统的运作方式，以及第 1–3 部分的思想（Transformer、scaling law、next-token prediction、RL 偏好对齐、高效推理服务）如何正在把它们重塑为**生成式推荐（generative recommender）**和**大推荐模型（large recommendation model）**。

**图例：** :material-star: 必读 · :material-book-open-variant: 深入阅读 · :material-new-box: 2025–2026 前沿

---

## 4.1 全局视角

经典的工业级推荐 / 广告系统是一个**多阶段级联（cascade）**架构：

```text
数十亿物品 ─► 召回（双塔 ANN，~10^3–10^4） ─► 粗排（~10^2–10^3）
          ─► 精排（重型 CTR/CVR 模型，多任务） ─► 重排（多样性、广告竞价、策略） ─► 信息流
```

| 经典技术栈 | LLM / 生成式思路带来的变化 |
|---|---|
| 物品 = 随机的 **ID embedding**（巨大的稀疏表） | 物品 = **语义 ID（semantic ID）**（由内容 embedding 生成的 token）——可共享，对冷启动友好 |
| 特征交叉 DNN（DCN、DIN），扩展性不佳 | 基于**用户行为序列**的 Transformer，遵循 **scaling law** |
| 召回、排序各自独立建模 | **一个生成式模型**直接解码出推荐列表（端到端） |
| Pointwise CTR loss | 下一行为预测 + **偏好对齐**（DPO / 基于奖励模型的 RL） |
| 文本 / 图像特征靠人工设计 | 用 **LLM / 多模态 embedding** 表示物品和用户；用 LLM 推理用户兴趣 |

与 LLM 相比，有几点差异需要牢记：词表规模是*数十亿*级物品，且每天都在更替；数据是非平稳的流式数据（模型在线训练，通常只训一个 epoch）；单次请求的延迟预算只有几十毫秒，却要处理数百个候选；广告场景还要额外考虑**校准（calibration）**（预估 CTR × 出价决定竞价结果）以及广告主侧的目标。

---

## 4.2 基础：深度推荐与广告 CTR 模型

| 论文 | 摘要 |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | FTRL-Proximal 在线逻辑回归，以及大规模广告 CTR 的工程经验——校准、节省内存、监控。 |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + 逻辑回归；数据新鲜度最重要；用归一化熵（normalized entropy）作为评估指标。 |
| :material-star: [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — Covington et al., RecSys 2016 | 经典的两阶段（候选生成 + 排序）深度推荐系统；把召回建模为对视频的超大规模多分类问题。 |
| :material-star: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — Cheng et al., 2016 | 线性部分负责“记忆”，DNN 负责“泛化”，联合训练；应用于 Google Play。 |
| :material-book-open-variant: [Deep & Cross Network (DCN)](https://arxiv.org/abs/1708.05123) / [DCN V2](https://arxiv.org/abs/2008.13535) — Wang et al., 2017/2020 | 显式且高效的特征交叉层；V2 是很强的线上基线。 |
| :material-star: [Deep Interest Network (DIN)](https://arxiv.org/abs/1706.06978) — Zhou et al. (Alibaba), 2017 | **Target attention**：按与候选广告的相关性对用户历史行为加权。排序中用户行为序列建模的开端。 |
| :material-star: [DLRM: Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) — Naumov et al. (Meta), 2019 | 参考架构（稀疏 embedding + 稠密 MLP + 点积交互），以及系统层面的视角：embedding 表占据主要内存，必须做模型并行。 |
| :material-book-open-variant: [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) — Kang & McAuley, 2018 | 用因果 Transformer 对物品序列做下一物品预测——本质上就是一个跑在物品 ID 上的 GPT。 |
| :material-book-open-variant: [BERT4Rec](https://arxiv.org/abs/1904.06690) — Sun et al., 2019 | 序列推荐中的双向掩码物品建模。 |
| :material-book-open-variant: [Towards Understanding the Overfitting Phenomenon of Deep CTR Models](https://arxiv.org/abs/2209.06053) — Zhang et al. (Alibaba), 2022 | 解释“**one-epoch**”现象：CTR 模型在第 2 个 epoch 开始时会急剧过拟合，根源在 embedding 层。 |
| :material-book-open-variant: [On the Embedding Collapse When Scaling up Recommendation Models](https://arxiv.org/abs/2310.04400) — Guo et al., 2023 | 为什么单纯增大 embedding 不再有收益：embedding 会坍缩到低维子空间；提出 multi-embedding。 |

---

## 4.3 Scaling law 与大推荐模型 :material-new-box:

2024–2026 年的核心问题是：*推荐模型能否像 LLM 那样 scale？* 传统 DLRM 主要扩的是 **embedding 表**（内存），而不是**算力**，因此很快遇到瓶颈。目前浮现的答案是：可以，前提是*稠密*部分采用设计良好的、类 Transformer 的网络，并作用于较长的用户行为序列。

| 论文 | 摘要 |
|---|---|
| :material-star: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152) — Zhai et al. (Meta), ICML 2024 | 把排序和召回重新表述为对用户行为流的**序列转导（sequential transduction）**（**生成式推荐，GR**）。HSTU block（pointwise 聚合注意力，无 softmax）可以应对非平稳词表；效果随训练算力呈**幂律**增长，直至 GPT-3/LLaMA-2 量级；线上 A/B 实验提升 +12.4%。这篇论文开启了“大推荐模型”浪潮。 |
| :material-star: [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) — Zhang et al. (Meta), 2024 | 堆叠因子分解机（FM）以捕捉高阶交互，在两个数量级的算力范围内持续 scale，而 DLRM 类基线早已饱和。 |
| :material-book-open-variant: [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351) — Zhang et al., 2023 | 针对基于 ID 的序列推荐模型的经验性 scaling law（最大 0.8B 参数），包含数据受限场景。 |
| :material-book-open-variant: [Scaling New Frontiers: Insights into Large Recommendation Models](https://arxiv.org/abs/2412.00714) — Guo et al. (Tencent), 2024 | 对 HSTU scaling 的系统性研究：哪些组件起作用，以及向排序任务迁移的效果。 |
| :material-new-box: [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654) — Han et al., 2025 | 把 HSTU 风格的 GR 引入线上精排模型，同时*保留*人工设计的交叉特征；group-layer-norm 以及训练 / 推理优化；在成本相近的情况下单样本 FLOPs 约 65×。 |
| :material-new-box: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) — Zhu et al. (ByteDance), 2025 | 面向硬件的精排主干（token mixing + per-token FFN / 稀疏 MoE），把模型 FLOPs 利用率（MFU）从 4.5% 提升到 45%，在推理延迟基本不变的情况下将精排模型参数量扩大 100×；1B 稠密 RankMixer 已在推荐和广告中全量上线。 |
| :material-new-box: [Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model (LUM)](https://arxiv.org/abs/2502.08309) — Yan et al. (Alibaba), 2025 | 先预训练生成式大用户模型 → 再查询它获取用户兴趣知识 → 最后输入传统精排模型；可 scale 到 7B 参数。 |
| :material-new-box: [MARM: Memory Augmentation and Scalable Complexity](https://arxiv.org/abs/2411.09425) — Lv et al. (Kuaishou), 2024 | 缓存注意力的中间结果，使复杂度扩展在延迟预算内可承受——推荐系统版的“KV cache”思路。 |
| :material-book-open-variant: [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) — Borisyuk et al., 2024 | 实用的大型精排模型方案：Residual DCN、基于历史的 Transformer、稠密门控、校准、量化、探索 / 利用（explore/exploit）。 |
| :material-book-open-variant: [Hierarchical Structured Neural Network (HSNN)](https://arxiv.org/abs/2408.06653) — Meta, 2024 | 用层次聚类和更丰富的交互模块，把召回从双塔点积扩展到更强的结构，已在广告中上线。 |

值得一读的业界文章：Meta 的 [Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)。

---

## 4.4 语义 ID 与生成式召回

不再在数十亿个不透明的物品 ID 上做预测，而是把每个物品**token 化**为一小段离散码序列，这些码来自物品的内容 embedding（通常采用 **RQ-VAE** 残差量化）。相似物品共享前缀，因此序列模型可以像 LLM 解码单词一样，**自回归地解码**下一个物品的码。

| 论文 | 摘要 |
|---|---|
| :material-book-open-variant: [Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991) — Tay et al., 2022 | seq2seq 模型记住整个语料，并直接解码文档 ID——生成式检索的源头。 |
| :material-book-open-variant: [Autoregressive Image Generation using Residual Quantization (RQ-VAE)](https://arxiv.org/abs/2203.01941) — Lee et al., 2022 | 残差量化：每一层量化剩余误差，得到由粗到细的码——语义 ID 的标准 tokenizer。 |
| :material-star: [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) — Rajput et al. (Google), NeurIPS 2023 | 对物品文本 embedding 做 RQ-VAE 得到语义 ID，再用 Transformer 解码下一个物品的语义 ID。相比基于 ID 的召回，冷启动和多样性更好。生成式召回的**标杆**论文。 |
| :material-star: [Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) — Singh et al. (Google/YouTube), 2023 | 在 YouTube 线上**精排**模型中，用语义 ID 的 n-gram / sentencepiece embedding 取代随机物品 ID——对新物品 / 长尾物品泛化更好，同时不损失记忆能力。 |
| :material-book-open-variant: [Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER)](https://arxiv.org/abs/2411.18814) — Yang et al. (Meta), 2024 | 分析生成式召回在哪些方面不如稠密召回（冷启动、覆盖率），并将两者结合。 |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | 先生成稀疏的语义 ID，再生成以其为条件的稠密向量——由粗到细，已在百度广告中上线。 |
| :material-new-box: [Generative Recommendation with Semantic IDs: A Practitioner's Handbook (GRID)](https://arxiv.org/abs/2507.22224) — Ju et al. (Snap), 2025 | 开源框架，并对语义 ID 推荐系统中 tokenizer、模型架构与解码方式的选择做了细致的消融——最好的上手起点。 |

---

## 4.5 端到端生成式推荐 :material-new-box:

2025 年的重要进展：用**一个生成式模型**解码出推荐列表，取代整个召回 → 排序级联，再借用 LLM 后训练（第 2 部分）中的偏好 / RL 方法对其进行**对齐**。

| 论文 | 摘要 |
|---|---|
| :material-star: [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) — Deng et al. (Kuaishou), 2025 | 带稀疏 MoE 的 encoder–decoder 基于语义 ID 生成一个视频*会话（session）*；奖励模型 + **迭代 DPO** 使输出与用户偏好对齐。已在快手主信息流上线。 |
| :material-star: [OneRec Technical Report](https://arxiv.org/abs/2506.13695) — Kuaishou, 2025 | 完整的端到端系统，取代原有级联：算力提升 10×，训练 / 推理 MFU 分别为 23.7% / 28.8%，采用**多目标奖励的 RL**，服务成本更低（OPEX 仅为级联方案的 10.6%），承接快手 25% 的 QPS。系统与 RL 细节值得细读。 |
| :material-new-box: [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900) — Kuaishou, 2025 | Lazy decoder-only 架构（去掉开销很大的 encoder），算力效率大幅提升，并可 scale 到 8B；使用真实用户反馈信号做 RL。 |
| :material-new-box: [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639) — Kuaishou, 2025 | 把物品语义 ID 与 LLM 的文本空间对齐，使模型在推荐之前能够**用自然语言推理**——推理模型遇上推荐系统。 |
| :material-new-box: [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784) — Google/YouTube, 2025 | 语义 ID tokenization → 在用户 / 物品数据上对 Gemini 系列 LLM 做继续预训练 → 针对具体任务微调，用于 YouTube 的生成式召回。 |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | 端到端生成式**广告**：联合生成广告序列，并处理竞价 / 分配与计费，而不是走多阶段流水线。 |
| :material-new-box: [Think Before Recommend: Latent Reasoning for Sequential Recommendation (ReaRec)](https://arxiv.org/abs/2503.22675) — Tang et al., 2025 | 通过在用户表示上增加隐式推理步骤实现测试时“思考”——推荐系统里的 test-time compute。 |

---

## 4.6 LLM 作为推荐器、特征提供者与用户模型

LLM 进入推荐系统有三种方式：**(a)** LLM *本身就是*推荐器（prompting / 微调），**(b)** LLM 为传统模型*产出特征 / embedding / 知识*，**(c)** LLM 基于长历史*对用户建模*。

| 论文 | 摘要 |
|---|---|
| :material-book-open-variant: [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366) — Geng et al., 2022 | 把评分预测、序列推荐、解释生成等任务统一成 text-to-text 任务，交给同一个 T5 模型。 |
| :material-book-open-variant: [Is ChatGPT a Good Recommender?](https://arxiv.org/abs/2304.10149) — Liu et al., 2023 | 零样本 / 少样本 LLM 推荐：擅长解释和冷启动，但没有协同信号时，精度不如训练过的模型。 |
| :material-star: [TALLRec](https://arxiv.org/abs/2305.00447) — Bao et al., 2023 | 用少量推荐数据对 LLM 做 LoRA 微调，即可得到很强的少样本推荐器。 |
| :material-book-open-variant: [Adapting LLMs by Integrating Collaborative Semantics (LC-Rec)](https://arxiv.org/abs/2311.09049) — Zheng et al., 2023 | 把语义 ID token 加入 LLM 词表，并通过多个微调任务使其与语言对齐。 |
| :material-book-open-variant: [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841) — Li et al. (Huawei), 2023 | 用对比学习对齐文本语言模型与协同 CTR 模型，线上只部署 CTR 模型——不增加任何延迟就获得 LLM 知识。 |
| :material-book-open-variant: [ClickPrompt](https://arxiv.org/abs/2310.09234) — Lin et al., 2023 | 由 CTR 模型为语言模型生成 soft prompt，融合协同信号与语义信号。 |
| :material-star: [Towards Open-World Recommendation with Knowledge Augmentation from LLMs (KAR)](https://arxiv.org/abs/2306.10933) — Xi et al. (Huawei), 2023 | LLM **离线**推理用户偏好和物品事实；编码后的知识向量作为特征供任意 CTR 模型使用。已在线上落地——一种务实的模式。 |
| :material-star: [HLLM: Hierarchical LLMs for Item and User Modeling](https://arxiv.org/abs/2409.12740) — Chen et al. (ByteDance), 2024 | Item LLM 把物品文本编码为 embedding；User LLM 对 embedding 序列建模。预训练 LLM 权重有帮助，且该方法可 scale 到 7B。 |
| :material-book-open-variant: [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) — Ning et al. (Google), 2024 | 把较长的用户历史压缩为 embedding，LLM 通过 cross-attention / soft prompt 来使用它们。 |
| :material-new-box: [360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://arxiv.org/abs/2501.16450) — LinkedIn, 2025 | 一个 150B 的 decoder LLM，仅用文本接口，就取代了 LinkedIn 各个场景中大量针对具体任务的排序模型。 |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | 让 LLM 作为超大物品集上的多查询、多目标召回器，已在广告中上线。 |
| :material-new-box: [RecGPT Technical Report](https://arxiv.org/abs/2507.22879) — Alibaba (Taobao), 2025 | 在淘宝首页上，由 LLM 驱动的用户意图挖掘 → 物品标签预测 → 召回 → 解释生成，并通过人与 LLM 协作的评判闭环进行训练。 |
| :material-new-box: [QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739) — Luo et al., 2024 | 把多模态 LLM 的物品表示与业务信号对齐，并量化为可用作精排特征的码。 |

---

## 4.7 广告专题：用户建模、竞价与生成式出价

| 论文 | 摘要 |
|---|---|
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | **SUM**：少数几个大型上游用户模型产出 embedding，供数百个下游广告模型使用——即“基础用户模型”范式；涵盖在线 embedding 的新鲜度问题。 |
| :material-book-open-variant: [TransAct](https://arxiv.org/abs/2306.00248) / [PinnerFormer](https://arxiv.org/abs/2205.04507) — Pinterest, 2022–2023 | 对近期行为做实时 Transformer 建模（TransAct），再结合长期批量用户 embedding（PinnerFormer）——广告 / 首页信息流排序中实时 + 批量的混合模式。 |
| :material-new-box: [AIGB: Generative Auto-bidding via Conditional Diffusion Modeling](https://arxiv.org/abs/2405.16141) — Guo et al. (Alibaba), KDD 2024 | 把自动出价（auto-bidding）视为：以目标回报为条件，用扩散模型生成一条出价*轨迹*，取代基于 RL 的出价器。 |
| :material-new-box: [EGA-V2](https://arxiv.org/abs/2505.17549) — Meituan, 2025 | （见 4.5）对完整的广告分配 + 计费过程做生成式建模。 |

广告方面还需要了解的主题：**校准（calibration）**（预估 CTR 必须是真实概率，因为它要与出价相乘）、**转化延迟反馈**、**隐私受限信号**（跨 App 标识符减少）、以及**生成式广告创意**（由 LLM / 扩散模型生成文案和图片，主流广告平台均已提供）。

---

## 4.8 推荐中的扩散模型及其他生成式模型

| 论文 | 摘要 |
|---|---|
| :material-book-open-variant: [Diffusion Recommender Model (DiffRec)](https://arxiv.org/abs/2304.04971) — Wang et al., SIGIR 2023 | 对被破坏的用户–物品交互向量去噪来生成推荐；提供 latent 和 temporal 变体以支持大规模场景。 |

---

## 4.9 综述

| 论文 | 摘要 |
|---|---|
| :material-star: [A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860) — Wu et al., 2023 | 对判别式与生成式两类 LLM 推荐方法的分类梳理。 |
| :material-book-open-variant: [How Can Recommender Systems Benefit from LLMs: A Survey](https://arxiv.org/abs/2306.05817) — Lin et al., 2023 | 按 LLM 用在*哪里*（哪个阶段）以及*怎么用*（是否微调、推理时是否配合传统推荐模型 CRM）来组织该领域。 |
| :material-book-open-variant: [LLMs for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) — Li et al., 2023 | 聚焦生成式范式与物品 ID 设计。 |
| :material-book-open-variant: [From Matching to Generation: A Survey on Generative Information Retrieval](https://arxiv.org/abs/2404.14851) — Li et al., 2024 | 生成式检索与生成增强检索，与搜索领域共通。 |

---

## 4.10 动手实践项目

- [ ] 在 MovieLens-1M / Amazon Reviews 上训练 SASRec；再训练一个 **GPT 风格**的下一物品预测模型并做对比（二者思路相同）。
- [ ] 构建语义 ID：用句子编码器对物品文本做 embedding → 训练 RQ-VAE（3 层 × 每层 256 个码）→ 检查相似物品是否共享前缀。
- [ ] 实现 TIGER 风格的生成式召回，在语义 ID 上做 beam search（可从 [GRID](https://github.com/snap-research/GRID) 入手）；与双塔基线比较召回率，尤其关注冷启动物品。
- [ ] 试用 Meta 开源的 [generative-recommenders](https://github.com/meta-recsys/generative-recommenders)（HSTU）仓库；绘制效果随模型规模变化的曲线。
- [ ] KAR 风格实验：离线用 LLM 撰写用户兴趣摘要，做 embedding 后作为特征加入 DIN/DCN-V2 CTR 模型；衡量 AUC 提升。
- [ ] 以点击 vs 跳过作为偏好对，用 DPO 对齐一个生成式推荐模型。

## 4.11 自测

??? question "为什么经典 DLRM 没有呈现出类似 LLM 的 scaling law？"
    它们的大部分参数在稀疏 embedding 表里（吃的是内存而非算力），交互层很浅，而且 embedding 一放大就会坍缩。真正打开 scaling 的是扩展稠密网络的*算力*——例如作用于长行为序列的 Transformer（HSTU），或高效的 token-mixing 主干（RankMixer、Wukong）。

??? question "语义 ID 解决了什么问题，代价又是什么？"
    它为物品提供了一套可组合的、基于内容的词表：新物品一上线就有有意义的 ID，相似物品之间共享参数，并且可以在较小的 token 词表上做生成式解码。代价是：热门物品会出现碰撞、损失细粒度的记忆能力，并且当内容分布漂移时 tokenizer 需要定期刷新。

??? question "OneRec 的训练流程与 LLM 后训练有何类比？"
    先在日志上以下一物品预测预训练生成式模型（≈ 预训练），再借助奖励模型和偏好 / RL 方法（迭代 DPO、多目标奖励的 RL）进行优化，使其符合用户和业务所看重的目标（≈ RLHF）。

??? question "为什么校准在广告中至关重要，而在自然推荐中没那么重要？"
    广告中，eCPM = 出价 × pCTR（× pCVR）决定竞价排序和实际扣费；pCTR 校准不准会导致广告排序错误，并使广告主被多收或少收费用。自然信息流多数情况下只需要*排序*正确。

??? question "为什么很难把 7B 的 LLM 直接放进排序链路？"
    精排要在几十毫秒内、在极高的 QPS 下为每个请求给数百到数千个候选打分。常见的变通办法：离线预计算 LLM 的物品 / 用户 embedding（KAR、HLLM 的物品侧）、蒸馏到小模型（CTRL）、缓存用户侧计算（MARM），或者使用每个请求只输出一次最终列表的生成式模型（OneRec）。
