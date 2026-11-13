# 1.7 缩放定律与数据

!!! abstract "本章你将学到"
    - 什么是**幂律**，以及为什么 LLM 的损失随参数量、数据量和算力都遵循幂律。
    - 算力恒等式 $C \approx 6ND$，以及如何用它为一次训练运行定规模。
    - **Kaplan 与 Chinchilla** 的分歧，配一个算力最优配置的实例演算，以及为什么现代模型都被刻意**过训练（over-training）**。
    - 网页数据流水线的各个阶段：文本抽取、语言识别、启发式过滤、基于模型的质量分类器、**MinHash** 去重（dedup，本章会讲解并实现）、去污染。
    - 数据配比、退火（annealing）/ 中期训练、合成数据，以及重复使用数据的局限。

## 为什么重要 {#why-this-matters-the-problem}

训练一个前沿模型要花数千万美元的算力，而且你只有一次机会。开工之前，实验室必须决定模型应该有多少参数 $N$、应该看多少 token $D$。模型太大而数据太少，或者反过来，都会浪费预算中的很大一部分。**缩放定律（scaling law）**通过对许多小而便宜的实验拟合简单的曲线，再外推到大规模运行，来回答这个问题。问题的另一半是*用哪些* token：两个 $N$ 和 $D$ 相同的模型，质量也可能因为数据不同而天差地别。到了 2025 年，数据整理被普遍认为是架构相近的各实验室之间的主要竞争优势。

## 直觉 {#intuition}

- **幂律。** 每多 10 倍算力，（可约）损失就会按大致固定的*比例*下降。在双对数坐标图上，损失对算力是一条直线。正是这种规律性，让人们能凭百万美元级别的实验来规划十亿美元级别的训练。
- **平衡 N 和 D。** 一个看了很少数据的大模型是欠训练的；一个看无穷无尽数据的小模型则会饱和。在固定的算力预算下，存在一个最佳点。Chinchilla 发现 $N$ 和 $D$ 应该同步增长，大约每个参数对应 20 个 token。
- **但训练并不是唯一的成本。** 模型只训练一次，却要被服务数十亿次。服务成本取决于 $N$，而不取决于 $D$。所以，往往值得训练一个*更小*的模型、喂给它比算力最优时*多得多*的数据。
- **数据质量是一个乘数。** 过滤掉垃圾和重复内容，意味着每个训练 token 能教到更多东西，这等效于把整条缩放曲线整体下移。

## 工作原理 {#how-it-works}

### 幂律 {#power-laws}

**幂律**是 $y = a x^{-b}$，其中 $a, b > 0$ 为常数。取对数后，$\log y = \log a - b \log x$：在双对数坐标上是一条斜率为 $-b$ 的直线。Kaplan et al.（2020）发现，当其他资源不构成瓶颈时，测试损失 $L$ 对每种资源都遵循幂律：

$$
L(N) \approx \Big(\frac{N_c}{N}\Big)^{\alpha_N}, \qquad L(D) \approx \Big(\frac{D_c}{D}\Big)^{\alpha_D}, \qquad L(C) \approx \Big(\frac{C_c}{C}\Big)^{\alpha_C},
$$

拟合出的指数约为 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$（常数 $N_c, D_c, C_c$ 取决于分词器和数据）。用大白话说：模型大小翻倍，损失乘以 $2^{-0.076} \approx 0.95$，即降低 5%，而且只要数据和算力跟得上，每翻一倍都能再得到同样的 5%。这一趋势在超过六个数量级的范围内都成立。

### 算力恒等式 $C \approx 6ND$ {#the-compute-identity-c-approx-6nd}

由 [1.3](03-transformer.md)，训练每个 token 约花 $6N$ FLOPs，所以一次用 $N$ 个参数、在 $D$ 个 token 上的训练，成本为

$$
C \approx 6 N D \ \text{FLOPs}.
$$

这一个等式把三个量联系在一起：固定其中任意两个，第三个就确定了。合理性检验：Llama 3 405B 在 15.6T token 上训练，得到 $6 \times 405 \times 10^9 \times 15.6 \times 10^{12} \approx 3.8 \times 10^{25}$ FLOPs，与 Llama 3 报告中的数字一致。

### Kaplan 与 Chinchilla {#kaplan-vs-chinchilla}

给定预算 $C$，应该如何在 $N$ 和 $D$ 之间分配？

- **Kaplan et al.（2020）**得出的结论是，模型大小应当比数据增长快得多：$N_{\text{opt}} \propto C^{0.73}$，$D_{\text{opt}} \propto C^{0.27}$。依此，GPT-3（175B）只在 300B token 上训练，每个参数还不到 2 个 token。
- **Hoffmann et al.（2022），“Chinchilla”**训练了 400 多个模型，发现 $N_{\text{opt}} \propto C^{0.5}$，$D_{\text{opt}} \propto C^{0.5}$：参数量和 token 数应当**同等**缩放，大约**每个参数 20 个 token**。造成分歧的一个主要原因是，Kaplan 的实验所用的学习率调度，其长度与每次运行的 token 数不匹配，这低估了训练更久的小模型的潜力。为了证明这一点，Chinchilla（70B，1.4T token）在算力预算相近的情况下，表现优于 Gopher（280B，300B token）。

Chinchilla 还拟合了一个参数化的损失曲面：

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
$$

其中 $E$ 是不可约损失（文本本身的熵，任何模型都无法超越），第二项是模型有限的惩罚，第三项是数据有限的惩罚。论文的拟合结果是 $E = 1.69$，$A = 406.4$，$B = 410.7$，$\alpha = 0.34$，$\beta = 0.28$。用大白话说：损失会趋近一个下限，而参数太少和 token 太少是分别要付出代价的。（Besiroglu et al. 在 2024 年的复现发现，这组具体的拟合与论文中另外两种估计方法并不一致；他们修正后的拟合与约 20 token/参数的经验法则吻合。做规划时请用经验法则，不要用这些常数。）

**从经验法则推导最优分配。** 如果 $D = rN$，$r \approx 20$，代入 $C = 6ND$：$C = 6rN^2$，于是

$$
N_{\text{opt}} = \sqrt{\frac{C}{6r}} = \sqrt{\frac{C}{120}}, \qquad D_{\text{opt}} = 20\, N_{\text{opt}}.
$$

### 为推理经济性而过训练 {#over-training-for-inference-economics}

Chinchilla 最优的含义是*在给定训练预算下损失最低*。但全生命周期的成本是训练加推理，而推理每生成一个 token 都要花 $2N$ FLOPs，并且永远如此。如果你要服务数万亿个 token，那么一个在多得多的数据上训练的较小模型，用稍高一点的训练开销，就能达到同样的损失，而服务成本要小得多（Sardana et al., 2023 对此做了形式化）。损失在超过每参数 20 个 token 之后仍在改善，只是每 FLOP 的改善速度变慢了。所以现代模型都被刻意**过训练**：Llama 3 8B 看了约 15T token（约每参数 1,900 个 token），Qwen3 的稠密模型则在 36T token 上训练。

### 数据流水线 {#the-data-pipeline}

大多数预训练 token 来自网页，通常取自 **Common Crawl**——一个自 2008 年起收集的、包含数十亿网页的公共存档。原始爬取数据大多不能直接用：导航菜单、垃圾内容、模板文字、机器生成的文本、近似重复的页面。典型的流水线（FineWeb、DCLM、RefinedWeb 和 Dolma 都公开了各自的流水线）如下：

```text
WARC（原始 HTML）→ 文本抽取 → URL 过滤 → 语言识别 → 启发式过滤
  → 基于模型的质量过滤 → 去重（精确 + MinHash）→ PII 移除
  → 去污染 → 领域配比 → 分词
```

1. **抽取。** 把 HTML 转成干净的文本，去掉菜单、广告和模板文字。直接对原始 HTML（WARC 文件）使用 `trafilatura` 这类工具，得到的文本明显优于 Common Crawl 预先抽取的文本（WET 文件）；FineWeb 报告了明显的质量差异。
2. **URL 过滤。** 使用成人、垃圾和恶意软件域名的黑名单。
3. **语言识别。** 用一个快速的分类器（例如 fastText 的语言识别模型）给每篇文档的语言打上标签并给出置信度；对目标语言，保留高于阈值的文档。
4. **启发式过滤。** 来自 C4 和 Gopher 的廉价规则：最小和最大词数、平均词长、以标点结尾的行所占比例、符号与词的比值、重复行或 n-gram 的比例、是否出现 “lorem ipsum” 或 “javascript must be enabled”。每条规则都能去掉一部分垃圾。
5. **基于模型的质量分类器。** 训练一个分类器来识别“好”文本，并保留得分最高的文档。DCLM 训练了一个简单的 fastText 分类器，以类指令和解释性文本（OpenHermes 数据和 ELI5 子版块）作为正例，以随机网页文本作为负例；它优于更复杂的过滤器。FineWeb-Edu 让 Llama-3-70B-Instruct 按 0–5 分的教育价值标准给约 450,000 个页面打分，用这些评分训练一个小分类器，并保留得分至少为 3 的页面，在知识和推理基准上取得了显著提升。
6. **去重。** 精确重复可以通过对文档（或行/段落）做哈希找出。**近似重复**（同一篇文章配不同的广告，或者一个只改了一个词的模板）需要模糊匹配，通常使用**带局部敏感哈希的 MinHash**，下文会解释。Lee et al.（2021）表明，去重可以减少记忆并提高训练效率。
7. **PII 移除。** 屏蔽电子邮件、IP 地址以及类似的个人信息。
8. **去污染。** 移除与评测基准重叠的训练文档（例如与测试题共享一个长 n-gram；GPT-3 使用 13-gram 重叠），使基准分数衡量的是泛化而不是记忆。

### MinHash 去重详解 {#minhash-deduplication-explained}

把每篇文档表示为它的**shingle**集合，例如所有 5 个词的序列。两篇文档 $A$ 和 $B$ 的相似度是它们的 **Jaccard 相似度** $J(A, B) = |A \cap B| / |A \cup B|$。对数十亿篇文档的所有文档对都计算 $J$ 是不可能的，所以我们用两个技巧：

**MinHash 估计 Jaccard。** 选一个随机哈希函数 $h$，计算 $\min_{s \in A} h(s)$，即 $A$ 的 shingle 上哈希值的最小值。关键事实：

$$
\Pr\Big[\min_{s \in A} h(s) = \min_{s \in B} h(s)\Big] = J(A, B).
$$

为什么：在随机哈希下，$A \cup B$ 中的每个 shingle 拥有最小哈希值的可能性相同。当且仅当整体最小的那个 shingle 同时在两个集合中，即位于 $A \cap B$ 内时，两个最小值才相等，这件事发生的概率是 $|A \cap B| / |A \cup B|$。所以，用 $K$ 个独立的哈希函数，每篇文档得到一个由 $K$ 个数组成的**签名**，两个签名中相同位置所占的比例就是 $J$ 的估计值。

**LSH 无需比较所有文档对就能找出候选。** 把每个签名分成 $b$ 个**带（band）**，每个带含 $r$ 个值（$K = br$）。每个带对应一次哈希分桶，以该带的 $r$ 个值作为键，把文档放进桶里。如果两篇文档至少在一个带中落进了同一个桶，它们就成为候选对。对于 Jaccard 为 $J$ 的一对文档，某一个带匹配的概率是 $J^r$，所以

$$
\Pr[\text{candidate}] = 1 - \big(1 - J^r\big)^{b}.
$$

这是一条关于 $J$ 的 S 形曲线：对不相似的文档对接近 0，对相似的接近 1，阈值由 $b$ 和 $r$ 决定。FineWeb 使用 112 个哈希函数，分成 $b = 14$ 个带、每带 $r = 8$，目标是找出相似度至少约 75% 的文档。只有共享同一个桶的文档才会被比较，所以成本大致与文档数成线性关系。

### 数据配比 {#data-mixtures}

最终的语料是多种来源的混合：网页文本、代码、数学、科学论文、书籍、多语言文本。各部分的比例会强烈影响能力，最佳配比是通过小型代理模型和缩放定律风格的实验凭经验找出来的。Llama 3 报告给出的最终预训练配比大致为：50% 通用知识，25% 数学和推理 token，17% 代码，8% 多语言 token。配比在训练过程中也常常调整，在接近结束时提高高质量来源的权重。

### 退火与中期训练 {#annealing-and-mid-training}

在预训练接近尾声、学习率衰减到零的时候，模型对它所看到的数据格外敏感。实验室利用这一点，设置一个**退火（annealing）**（或称**中期训练 mid-training**）阶段：训练的最后一段使用精心整理的混合数据，其中富含高质量网页文本、数学、代码、类指令数据和合成数据。OLMo 2 记录了一个专门的中期训练阶段；Llama 3 在最后用高质量数据做了退火，还利用对少量候选数据集做退火，作为衡量该数据集价值的廉价方法。长上下文阶段（见 [1.6](06-long-context-and-efficient-attention.md)）通常也是后期阶段的一部分。

### 合成数据 {#synthetic-data}

强大的现有模型可以生成或改写训练数据：教科书式的讲解和习题（phi-1 的“教科书质量”数据）、把网页改写成更干净的风格（Maini et al., 2024, “Rephrasing the Web”），以及经过验证的数学和代码题解。合成数据现在已是预训练配比的标准组成部分，尤其是在退火阶段。风险包括：多样性下降、继承生成器的错误和偏见，以及生成器若见过测试集则会带来基准污染。

### 重复数据 {#repeated-data}

高质量文本是有限的。Muennighoff et al.（2023）在重复数据上训练模型，发现直到约 4 个 epoch，重复的 token 几乎与新鲜 token 一样有价值；超过这个数，收益迅速递减，最终多出来的 epoch 几乎没有增益。这就是为什么实验室更努力地去提取更多独一无二的 token（更好的文本抽取、更多语言、PDF、合成改写），而不是对同样的数据反复循环许多遍。

## 代码 {#code}

**算力最优的规模配置与幂律拟合。**

```python
import math, random

def chinchilla_optimal(C, tokens_per_param=20):
    """在 C = 6*N*D 且 D = r*N 下的算力最优 (N, D)  =>  N = sqrt(C / (6r))。"""
    N = math.sqrt(C / (6 * tokens_per_param))
    return N, tokens_per_param * N

def fit_power_law(xs, ys):
    """通过双对数空间中的一条直线，对 y = a * x^(-b) 做最小二乘拟合。"""
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((u - mx) * (v - my) for u, v in zip(lx, ly)) / sum((u - mx) ** 2 for u in lx)
    return math.exp(my - slope * mx), -slope            # a, b

for C in [1e21, 1e23, 1e24, 3.8e25]:
    N, D = chinchilla_optimal(C)
    print(f"C={C:.1e}: N={N/1e9:7.1f}B  D={D/1e12:6.2f}T")

# 合成“实验”：loss = 30 * N^-0.076，带 2% 噪声；恢复出指数。
random.seed(0)
Ns = [1e7, 3e7, 1e8, 3e8, 1e9, 3e9]
losses = [30 * N ** -0.076 * (1 + random.uniform(-0.02, 0.02)) for N in Ns]
a, b = fit_power_law(Ns, losses)
print(f"fitted: loss = {a:.1f} * N^-{b:.3f}; extrapolated loss at 1e11 params: {a * 1e11 ** -b:.3f}")
```

真实的缩放定律拟合做法与此相同，只是实验设计更加讲究：每个点都是该规模下最好的一次运行（学习率调度经过调优），并且通过留出最大的几次运行来检验拟合。请注意，仅仅 2% 的噪声就会把拟合出的指数从 0.076 推到约 0.079，而当你向外推 30 倍时，这个差别影响极大。

**MinHash + LSH 近似重复检测。**

```python
import random, zlib

def shingles(text, n=5):
    """文档的词 n-gram（“shingle”）集合。"""
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

P = (1 << 61) - 1                                  # 用于通用哈希的大素数
random.seed(1)
HASHES = [(random.randrange(1, P), random.randrange(P)) for _ in range(112)]

def minhash(sh):
    """签名：对每个哈希函数，取所有 shingle 上的最小哈希值。"""
    xs = [zlib.crc32(s.encode()) for s in sh]
    return [min((a * x + b) % P for x in xs) for a, b in HASHES]

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)

def lsh_candidates(sigs, bands=14, rows=8):
    """在任一带中共享全部 `rows` 个值的文档成为候选重复对。"""
    buckets, pairs = {}, set()
    for doc_id, sig in sigs.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            for other in buckets.setdefault(key, []):
                pairs.add((other, doc_id))
            buckets[key].append(doc_id)
    return pairs

base = ("pre-training data is collected from common crawl snapshots then extracted filtered "
        "deduplicated and mixed with code math books and papers before the final annealing "
        "phase where the learning rate decays and only the highest quality documents are used")
docs = {
    "a": base,
    "b": base.replace("final", "last"),             # 近似重复（改了一个词）
    "c": "language models are trained on trillions of tokens of filtered web text and code",
}
sh = {k: shingles(v) for k, v in docs.items()}
sigs = {k: minhash(s) for k, s in sh.items()}
est = sum(x == y for x, y in zip(sigs["a"], sigs["b"])) / len(HASHES)
print(f"true Jaccard(a,b)={jaccard(sh['a'], sh['b']):.2f}  MinHash estimate={est:.2f}")
print("candidate duplicate pairs:", lsh_candidates(sigs))
```

这段代码会打印出真实 Jaccard 为 0.74，估计值为 0.72，并把 (a, b) 标记为候选对。改动一个词会改变所有包含它的 5-gram shingle（最多 5 个），这就是这些短文档的相似度是 0.74 而不是接近 1 的原因。换别的随机种子试试：对于 $J = 0.74$，公式给出 $1 - (1 - 0.74^8)^{14} \approx 0.73$，所以大约每四个种子里就有一个会*漏掉*这一对。LSH 在设计上就是概率性的。

## 实例演算 {#worked-example}

**为 $C = 10^{24}$ FLOPs 的训练运行定规模。** 使用 Chinchilla 规则：

- $N_{\text{opt}} = \sqrt{10^{24} / 120} = \sqrt{8.33 \times 10^{21}} \approx 9.1 \times 10^{10}$，即约 **91B 参数**。
- $D_{\text{opt}} = 20 \times 91\text{B} \approx$ **1.8T token**。
- 验算：$6 \times 9.1 \times 10^{10} \times 1.83 \times 10^{12} \approx 1.0 \times 10^{24}$。✓

**Llama 3 8B 被过训练了多少？** $C = 6 \times 8 \times 10^9 \times 15 \times 10^{12} = 7.2 \times 10^{23}$ FLOPs。对这个预算，Chinchilla 最优的模型会有 $\sqrt{7.2 \times 10^{23} / 120} \approx 77\text{B}$ 个参数，并看 1.55T 个 token。而 Meta 则把一个约 77B 算力最优模型的算力，花在了一个 8B 模型上。训练出来的 8B 模型比 77B 模型要差一些，但每生成一个 token，成本是约 $2 \times 8\text{B}$ 而不是 $2 \times 77\text{B}$ FLOPs，便宜了近 10 倍，而且它能放进单块消费级 GPU。

**Llama 3 自己的缩放定律。** Llama 3 报告在自己的数据配比上拟合了缩放定律，并预测对于 $3.8 \times 10^{25}$ FLOPs，算力最优的模型约有 402B 参数，在 16.55T token 上训练，约每参数 41 个 token，而不是 20 个；最终选定的 405B/15.6T 与这一预测很接近。算力最优的比例取决于数据和设置，这就是实验室要重新拟合它的原因。

**MinHash LSH 阈值**，取 $b = 14$，$r = 8$：当 $J = 0.5$ 时，$\Pr[\text{candidate}] = 1 - (1 - 0.5^8)^{14} \approx 0.05$；$J = 0.75$ 时约为 $0.77$；$J = 0.8$ 时约为 $0.92$；$J = 0.9$ 时约为 $0.9996$。曲线在 0.7–0.8 附近急剧上升，这正是预期的阈值。

## 实践中的做法（2025–2026） {#in-practice-20252026}

| 模型 | 参数量（总） | 预训练 token 数（报告值） |
|---|---|---|
| Llama 3 405B | 405B 稠密 | 15.6T |
| DeepSeek-V3 | 671B MoE（37B 激活） | 14.8T |
| Kimi K2 | 1.04T MoE（32B 激活） | 15.5T |
| Qwen3（所有规模） | 0.6B–235B | ~36T |
| Gemma 3 27B | 27B 稠密 | 14T |

- **15–36T 的 token 量已是标配**，对其中大多数规模来说，都远超 Chinchilla 最优。对 MoE 模型，“每参数 token 数”在算力方面应使用*激活*参数来算，在容量方面则应使用总参数来算。
- **公开、有文档的流水线**（FineWeb/FineWeb-Edu、DCLM、Dolma、OLMo 2 的配比）已经让配方公开；剩下的秘密主要在数据配比、合成数据和退火的选择上。
- **基于模型的过滤已无处不在**，通常是用现有的 LLM 给一个样本打标签，再用一个小分类器扩展到全量。
- **超参数迁移**：实验室在小模型上调好学习率和初始化，再借助 μP（Yang et al., 2022）这类参数化方式把它们迁移过去，使缩放定律实验和最终的训练运行共享最优设置。
- **数据墙**：随着高质量网页文本越来越被用尽，改写、合成生成、多语言扩展，以及在最好的子集上做多 epoch 训练，是主要的抓手。

## 常见误区 {#common-pitfalls-misconceptions}

- **“Chinchilla 说永远用 20 token/参数。”** 它说的是，在它的设置下，那是*训练算力最优*的比例。如果推理很重要（几乎总是如此），就应该把更小的模型训练得更久。
- **把有噪声的拟合外推得太远。** 指数上的一点小误差，在 100 倍之外会变成很大的误差。请留出你最大的几次运行来验证拟合。
- **跨分词器或跨数据配比比较损失**来评估缩放定律。常数（有时还有指数）是特定于设置的。
- **去重过于激进。** 把每个近似重复都删掉，可能会丢掉有价值的重复内容（许可证、常见代码惯用法），并且对某些领域过度过滤；FineWeb 发现，按每个快照分别去重，比跨所有快照去重效果更好。
- **跳过去污染**会抬高基准成绩，并误导你自己的开发决策。
- **以为质量分类器是中立的。** 它们编码了“好”的标注所带有的偏见；过重的过滤会减少多样性、方言覆盖或领域覆盖。

## 本章要点 {#key-takeaways}

- 损失随参数量、数据量和算力按幂律下降；小实验可以预测大规模训练。
- $C \approx 6ND$；Chinchilla 算力最优的分配约为每参数 20 个 token，给出 $N_{\text{opt}} \approx \sqrt{C/120}$。
- 推理成本使过训练变得合理：现代模型每个参数要看数百到数千个 token。
- 网页数据流水线是 抽取 → 语言识别 → 启发式过滤 → 基于模型的质量过滤 → （MinHash）去重 → 去污染 → 配比。
- MinHash 之所以能估计 Jaccard 相似度，是因为最小哈希值碰撞的概率恰为 $J$；LSH 分带把它变成一个廉价的近似重复过滤器。
- 当新鲜的高质量数据不够用时，后期在精选数据和合成数据上退火，以及最多约 4 个 epoch 的重复，是主要的工具。

## 自测 {#check-your-understanding}

??? question "你有 $6 \times 10^{23}$ FLOPs。Chinchilla 最优的 N 和 D 是多少？"
    $N = \sqrt{6 \times 10^{23} / 120} = \sqrt{5 \times 10^{21}} \approx 70.7\text{B}$，$D \approx 1.41\text{T}$。这基本上就是 Chinchilla 本身（70B，1.4T）。

??? question "Kaplan et al. 为什么会得出模型应当比数据增长更快的结论？"
    除其他差异之外，他们的运行所用的学习率调度与每次运行的长度不匹配，这让小模型训练更久看起来没有实际那么有用。Chinchilla 为每次运行调好了调度，发现 $N$ 和 $D$ 应该同等缩放。

??? question "为什么两个 MinHash 值相等的概率等于 Jaccard 相似度？"
    在随机哈希下，$A \cup B$ 中的每个元素拥有最小哈希值的可能性相同。当且仅当那个元素同时在两个集合中，两个最小值才相等，其概率为 $|A \cap B| / |A \cup B|$。

??? question "取 $b = 20$ 个带、每带 $r = 5$，当 $J = 0.6$ 时，候选概率是多少？"
    $0.6^5 = 0.0778$；$1 - (1 - 0.0778)^{20} = 1 - 0.9222^{20} \approx 1 - 0.198 = 0.80$。

??? question "为什么一个实验室明明知道，在 20T token 上训练 7B 模型比达到同样损失的算力最优模型花更多算力，还是这样做？"
    因为服务成本随 $N$ 增长。更小的模型每个 token 更便宜、更快，能放在更便宜的硬件上，并且会被使用数十亿次，所以多花的那一次性训练成本会被成倍地赚回来。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020 | 损失随参数量、数据量和算力遵循幂律。开启了“规模就是一切”的时代。 |
| :material-star: [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022 | 参数量和 token 数应当大致同等缩放（算力最优约为 20 token/参数）。 |
| :material-book-open-variant: [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) — Besiroglu et al., 2024 | 重新拟合 Chinchilla 的参数化损失，发现发表的拟合与另外两种方法不一致；关于如何拟合缩放定律的有益一课。 |
| :material-book-open-variant: [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) — Sardana et al., 2023 | 把推理成本加进缩放定律：为什么过训练较小的模型是经济的。 |
| :material-book-open-variant: [Tensor Programs V: Zero-Shot Hyperparameter Transfer (μP)](https://arxiv.org/abs/2203.03466) — Yang et al., 2022 | 一种参数化方式，使最优学习率能从小模型迁移到大模型：小模型调参，大模型训练。 |
| :material-book-open-variant: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023 | 重复数据最多约 4 个 epoch，几乎与新鲜数据一样好；超过之后收益消失。 |
| :material-star: [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — Penedo et al., 2024 | 15T token 的开放网页语料，带有完整记录的过滤/去重消融实验，外加 FineWeb-Edu。网页数据流水线的最佳指南。 |
| :material-star: [DataComp-LM (DCLM)](https://arxiv.org/abs/2406.11794) — Li et al., 2024 | 数据整理的基准；表明一个简单的 fastText 质量分类器就是巨大的收益。 |
| :material-book-open-variant: [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Penedo et al., 2023 | 表明仅靠过滤和去重得当的网页数据，就能媲美精选语料。 |
| :material-book-open-variant: [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159) — Soldaini et al., 2024 | 一个有完整记录的开放预训练语料及工具包。 |
| :material-book-open-variant: [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — Lee et al., 2021 | 去除精确重复和近似重复，可以减少记忆并提高效率。 |
| :material-book-open-variant: [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) — Gunasekar et al., 2023 | 在高质量和合成的“教科书”数据上训练的小模型，表现远超它的规模。 |
| :material-book-open-variant: [Rephrasing the Web](https://arxiv.org/abs/2401.16380) — Maini et al., 2024 | 用 LLM 把网页文档改写成更干净的风格，作为预训练数据的来源。 |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — OLMo Team (AI2), 2025 | 完全开放的配方，包括预训练末期在高质量数据上的**中期训练 / 退火**。 |
