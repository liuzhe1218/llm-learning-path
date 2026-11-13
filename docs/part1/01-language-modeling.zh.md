# 1.1 语言建模

!!! abstract "本章你将学到"
    - 语言模型*究竟是什么*，用形式化的话说：token 序列上的一个概率分布。
    - 概率的链式法则如何把它变成**下一个 token 预测**。
    - 如何手算**交叉熵**和**困惑度**，以及它们的数值意味着什么。
    - 为什么 n-gram 模型会撞墙，神经语言模型又是如何越过这堵墙的。
    - 如何从模型中生成文本：贪心解码、温度、top-k 和 top-p 采样。

## 为什么重要 {#why-this-matters-the-problem}

本课程里你会遇到的每一个大模型，从 100 行的玩具 GPT 到万亿参数的 MoE，训练目标都是同一个：*给定到目前为止的文本，预测下一个 token*。预训练（整个第一部分）本质上就是在最小化一个数：在数万亿 token 上这个预测的平均交叉熵。在谈 Transformer、分词器或 GPU 集群之前，我们需要先精确地弄清楚：这个数是什么、为什么最小化它是个好主意，以及一个每次只预测一个 token 的模型，是怎么写出整篇文章的。

## 直觉 {#intuition}

可以把语言模型想象成一个读书极多的自动补全。给定"The capital of France is"，它不会只输出一个答案，而是为**每一个可能的下一个 token 输出一个概率**：也许 " Paris" 是 0.8，" a" 是 0.03，" located" 是 0.01，词表里其余约 100,000 个 token 各分到微乎其微的一点。

生成文本时，你从这个分布里挑一个 token，把它接到末尾，再问一次。**训练**模型时，你给它看真实文本，衡量它对每个实际出现的下一个 token 有多"意外"：给正确答案分配 0.8 很好，0.001 就很糟。训练信号就是"对真实文本少一点意外"。

为什么这会产生类似智能的东西？因为要想对物理教科书、法律合同、Python 代码或一段对话的下一个 token 不感到意外，模型必须隐式地学会语法、事实、算术、代码语义，以及一些关于人如何推理的知识。预测是一个通用任务：几乎任何会在文本中留下痕迹的技能，都会在某处帮助降低损失。

## 工作原理 {#how-it-works}

### 语言模型是序列上的分布 {#a-language-model-is-a-distribution-over-sequences}

设**词表**（vocabulary）$\mathcal{V}$ 是一个有限的 token 集合（token 从哪里来，见 [1.2](02-tokenization.md) 章），大小为 $V = |\mathcal{V}|$。一段文本是一个序列 $x = (x_1, x_2, \dots, x_T)$，其中每个 $x_t \in \mathcal{V}$。**语言模型**为每一个这样的序列分配一个概率 $p(x_1, \dots, x_T)$，这些概率在所有序列上求和为 1（用一个结束符 token 使长度有限）。

直接对联合分布建模是没有希望的：当 $V = 50{,}000$、$T = 100$ 时，可能的序列有 $50{,}000^{100}$ 个。诀窍是把它分解。

### 链式法则把它变成下一个 token 预测 {#the-chain-rule-turns-it-into-next-token-prediction}

**概率的链式法则**说，任何联合分布都可以写成条件分布的乘积：

$$
p(x_1, \dots, x_T) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2) \cdots = \prod_{t=1}^{T} p(x_t \mid x_{<t}),
$$

其中 $x_{<t} = (x_1, \dots, x_{t-1})$ 是位置 $t$ 之前的**上下文**（或前缀）。用话说：整段文本的概率等于第一个 token 的概率，乘以给定第一个 token 时第二个 token 的概率，依此类推。这是一个恒等式，不是近似。所以，一个能对任意前缀输出 $p(x_t \mid x_{<t})$ 的模型，*就是*一个完整的语言模型。这种从左到右的方式叫**自回归**（autoregressive，每个输出会被反馈回来作为输入）。

神经语言模型用参数 $\theta$ 来实现它：网络 $f_\theta$ 读入前缀，输出一个 **logits** 向量 $z \in \mathbb{R}^{V}$（每个 token 一个实数），再由 **softmax** 把它转成概率：

$$
p_\theta(x_t = i \mid x_{<t}) = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

softmax 先对每个 logit 取指数（使其为正），再除以总和（使它们加起来为 1）。logit 越大，概率越大。

### 训练目标：最大似然 = 交叉熵 {#the-training-objective-maximum-likelihood-cross-entropy}

给定训练语料，我们选择 $\theta$ 让真实文本的概率尽可能大（**最大似然**）。许多小概率相乘会下溢，所以我们取对数，把乘积变成求和，再翻转符号以便做最小化：

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

这就是每个 token 的**平均负对数似然（NLL）**。它也叫**交叉熵损失**，因为在每个位置上，它等于"真实"的 one-hot 分布 $q$（所有概率质量都在实际出现的那个 token 上）与模型分布 $p$ 之间的交叉熵 $H(q, p) = -\sum_i q_i \log p_i$：当 $q$ 是 one-hot 时，求和退化为 $-\log p_{\text{correct}}$。

值得牢记的几条性质：

- 如果模型给正确 token 的概率是 1，这个位置的损失就是 $-\log 1 = 0$。它永远不会是负数。
- 如果它给正确 token 的概率 $\varepsilon \to 0$，损失会趋向 $+\infty$。自信的错误会被重罚。
- 一个什么都不知道、在 $V$ 个 token 上均匀预测的模型，损失为 $\log V$。当 $V = 50{,}000$ 时，这是 $\ln 50{,}000 \approx 10.82$ nats。每条大模型训练曲线都从这个值附近开始。

!!! note "Nats 与 bits"
    使用自然对数时单位是 **nats**；使用 $\log_2$ 时是 **bits**。把 nats 除以 $\ln 2 \approx 0.693$ 就得到 bits。PyTorch 的 `F.cross_entropy` 使用自然对数。

在 Transformer 中，全部 $T$ 个位置在一次前向传播里并行预测（[1.3](03-transformer.md) 章的因果掩码保证位置 $t$ 偷看不到 $x_t$ 及之后的内容）。因此在长度为 $T$ 的序列上训练，一次就得到 $T$ 个训练样本。这种并行性是 Transformer 取代循环网络的原因之一。

### 困惑度 {#perplexity}

**困惑度（perplexity，PPL）** 是平均 NLL 的指数：

$$
\mathrm{PPL} = \exp\!\Big(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\Big) = \Big(\prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})\Big)^{-1/T}.
$$

用话说：困惑度是模型给每个正确 token 的概率的几何平均的倒数。它有一个很直观的解释：**有效分支因子**。困惑度为 20，意味着模型平均而言的不确定程度，就像在 20 个 token 中均匀地选一个。均匀模型的困惑度恰好是 $V$。

!!! warning "只有分词器相同，困惑度才可比"
    损失和困惑度都是*按 token* 计算的。token 更长的分词器，每句话的预测次数更少、但每次更难。要比较使用不同分词器的模型，需要用与分词器无关的量来归一化，比如每字节比特数（bits per byte，总比特数除以文本的 UTF-8 字节数）。

### 从 n-gram 到神经语言模型 {#from-n-grams-to-neural-language-models}

在神经网络之前，标准的语言模型是 **n-gram 模型**。它做了一个**马尔可夫假设**：下一个 token 只依赖前面 $n-1$ 个 token，

$$
p(x_t \mid x_{<t}) \approx p(x_t \mid x_{t-n+1}, \dots, x_{t-1}),
$$

并通过计数来估计每个条件概率：$p(w \mid u) = \mathrm{count}(u, w) / \mathrm{count}(u)$，其中 $u$ 是 $(n-1)$ 个 token 的上下文。**二元（bigram）**模型（$n = 2$）只以前一个 token 为条件。

n-gram 撞上了两堵墙：

1. **稀疏性。** 表有 $V^n$ 项。当 $V = 50{,}000$ 时，三元（trigram）表有 $1.25 \times 10^{14}$ 个格子；大多数上下文从未出现过，所以计数为零，未见过的事件概率为 0（损失无穷大）。平滑技巧（加一平滑、Kneser–Ney、回退）可以打补丁，但无法根治。
2. **无法在相似上下文之间泛化。** 见过"the cat sat on the mat"，并不会让 n-gram 模型学到任何关于"the dog sat on the rug"的东西，因为"cat"和"dog"只是两个互不相关的下标。

**神经语言模型**（Bengio 等，2003）解决了这两个问题。每个 token 有一个学出来的**嵌入向量**（embedding），一个 $\mathbb{R}^d$ 中的稠密向量；相似的 token 会得到相似的向量，所以关于"cat"的证据可以迁移到"dog"。神经网络把上下文的嵌入映射成 logits，所以参数量随 $d$ 和网络规模增长，而不是随 $V^n$ 增长。随后循环网络（RNN、LSTM）去掉了固定窗口，Transformer（[1.3](03-transformer.md)）则让计算在各位置上并行，并且能直接关注任何更早的 token。

### 生成文本：解码策略 {#generating-text-decoding-strategies}

推理时，我们反复计算 $p_\theta(\cdot \mid x_{<t})$ 并选择一个 token。主要的选择有：

- **贪心解码：** 总是取 $\arg\max_i p_i$。确定性的，适合简短的事实性回答，但在长文本中容易陷入乏味、重复的循环。
- **采样：** 从分布中抽取 $x_t \sim p_\theta(\cdot \mid x_{<t})$。多样，但成千上万个各自概率很低的 token 合起来在长尾里占了不少质量，所以采样偶尔会选到胡言乱语，把后面的文本带偏。
- **温度 $\tau$：** 在 softmax 之前把 logits 除以 $\tau > 0$，$p_i \propto e^{z_i/\tau}$。$\tau < 1$ 会让分布更尖锐（更自信，$\tau \to 0$ 时趋近贪心）；$\tau > 1$ 会让分布更平坦（更随机，$\tau \to \infty$ 时趋于均匀）。
- **Top-k：** 只保留概率最高的 $k$ 个 token，重新归一化，再采样。
- **Top-p（核）采样**（Holtzman 等，2019）：保留概率之和至少为 $p$ 的最小 token 集合，重新归一化，再采样。与 top-k 不同，候选集是自适应的：模型自信时它很小，许多续写都合理时它很大。

实践中这些方法是依次叠加使用的（先温度，再 top-k 和/或 top-p）。推理引擎和更高级的解码方法（投机解码、受约束解码）在[第三部分](../part3/index.md)中介绍。

### 为什么"只预测下一个 token"就能得到通用能力 {#why-just-predict-the-next-token-yields-general-capabilities}

交叉熵目标是*恰当的*（proper）：当且仅当模型分布等于真实数据分布时，它的期望值才最小。所以损失越低的模型，按定义就是对文本生成过程越好的模型；而对互联网规模的文本来说，这个过程包括人们解方程、写代码、翻译和回答问题。几点观察让这件事更具体：

- **预测需要理解。** 要预测"17 × 23 = "的下一个 token，模型必须算出 391（或者把它背下来）。要预测一个证明的下一行，它必须跟上论证。
- **任务就是文本。** GPT-2 表明，一个只在网页文本上训练的模型，只要用合适的格式提示（"TL;DR:"），就能做翻译或摘要，因为这类模式在网上自然出现。GPT-3（175B 参数）展示了**上下文学习**（in-context learning）：在提示里给几个例子，足够大的模型就会在新输入上延续这个模式，无需更新权重。
- **损失随规模可预测地改善。** [1.7](07-scaling-laws-and-data.md) 章会说明，损失随参数量、数据量和算力呈平滑的幂律下降，这正是建造越来越大的模型的依据。

基座模型仍然不是助手：它只会续写文本，不会遵循指令，也不会拒绝有害请求。把它变成助手是后训练的工作（[第二部分](../part2/index.md)）。

## 代码 {#code}

一个完整的二元语言模型，用梯度下降和交叉熵训练，外加一个支持温度、top-k 和 top-p 的采样器。训练循环和 GPT 完全一样，只是 $f_\theta$ 简单得多（一张 logits 查找表）。

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
text = "the cat sat on the mat. the dog sat on the rug. the cat ran. "
chars = sorted(set(text))                       # 字符级词表
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in text])    # [T]

# 二元模型：下一个 token 的 logits 只取决于当前 token。
# W[i] 存放"当前 token 为 i 时下一个 token"的 V 个 logits。
W = torch.zeros(V, V, requires_grad=True)
opt = torch.optim.Adam([W], lr=0.1)

x, y = data[:-1], data[1:]                      # 输入和下一个 token 的目标
for step in range(300):
    logits = W[x]                               # [T-1, V]
    loss = F.cross_entropy(logits, y)           # 以 nats 为单位的平均 NLL
    opt.zero_grad(); loss.backward(); opt.step()
print(f"uniform loss = {torch.log(torch.tensor(V)).item():.3f}, "
      f"trained loss = {loss.item():.3f}, perplexity = {loss.exp().item():.2f}")

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None:                       # 把低于第 k 大值的全部屏蔽
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:                       # 核采样：质量 >= top_p 的最小集合
        sorted_p, idx = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        keep = cum - sorted_p < top_p           # 若某 token *之前*的累积质量 < top_p，就保留它
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(start="t", n=60, **kw):
    out = [stoi[start]]
    for _ in range(n):
        out.append(sample_next(W[out[-1]], **kw))
    return "".join(itos[i] for i in out)

print(generate(temperature=0.7, top_p=0.9))
```

运行后会打印出约 0.75 nats 的训练损失（均匀基线是 $\ln V \approx 2.71$），以及类似 "the t the sathe cathe og. rug." 的字符串。训练文本的片段会出现，但超过一两个字符就没有连贯性了，这正是一个只看得到前一个字符的模型所能达到的极限。[1.3](03-transformer.md) 章会把这张查找表换成能看到整个前缀的 Transformer。

## 实例演算 {#worked-example}

**手算交叉熵和困惑度。** 一个模型读入三个 token 的文本 "the cat sat"，并给正确 token 分配如下概率：

| 位置 | 上下文 | 正确 token | $p_\theta$ | $-\ln p_\theta$ |
|---|---|---|---|---|
| 1 | （开头） | the | 0.05 | 2.996 |
| 2 | the | cat | 0.01 | 4.605 |
| 3 | the cat | sat | 0.20 | 1.609 |

- 总 NLL $= 2.996 + 4.605 + 1.609 = 9.210$ nats。
- 平均损失 $= 9.210 / 3 = 3.070$ nats 每 token $= 3.070 / 0.693 \approx 4.43$ bits 每 token。
- 困惑度 $= e^{3.070} \approx 21.5$。用另一个公式验算：$(0.05 \times 0.01 \times 0.20)^{-1/3} = (10^{-4})^{-1/3} \approx 21.5$。这个模型"困惑得就像在约 21.5 个 token 中均匀选择"。
- 在 $V = 50{,}000$ 个 token 上的均匀模型，得分是 $\ln 50{,}000 \approx 10.82$ nats，困惑度为 50,000。

**温度与 top-p。** 假设在一个 3 个 token 的词表上，logits 为 $z = (2.0, 1.0, 0.1)$。

| 温度 | 缩放后的 logits | 概率 |
|---|---|---|
| $\tau = 1$ | (2.0, 1.0, 0.1) | (0.659, 0.242, 0.099) |
| $\tau = 0.5$ | (4.0, 2.0, 0.2) | (0.864, 0.117, 0.019) |
| $\tau = 2$ | (1.0, 0.5, 0.05) | (0.502, 0.304, 0.194) |

对 $\tau=1$：$e^{2} = 7.389$，$e^{1} = 2.718$，$e^{0.1} = 1.105$，总和 $11.212$，所以 $p = (0.659, 0.242, 0.099)$。取 top-p $= 0.9$，按降序排列：累积质量先是 $0.659$，然后是 $0.901 \ge 0.9$，所以我们保留前两个 token 并重新归一化为 $(0.731, 0.269, 0)$。取 top-k $= 1$ 就回到了贪心解码。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **目标没有变过。** 前沿基座模型（Llama 3、DeepSeek-V3、Qwen3、Kimi K2）都是在因果 Transformer 上用下一个 token 的交叉熵训练的。新增的部分都是辅助性的：DeepSeek-V3 加了一个**多 token 预测**损失，同时预测更远处的 token（[1.9](09-frontier-model-case-studies.md) 章），MoE 模型则加入负载均衡项（[1.5](05-mixture-of-experts.md)）。
- **损失数值。** 训练曲线从 $\ln V$ 附近开始（对 128K–262K 的词表约为 11.8–12.5 nats），在最初几千步里急剧下降，之后在训练的其余阶段缓慢下降。最终数值很大程度上取决于分词器和数据配比，所以永远不要直接比较不同实验室的原始损失。
- **词表很大**（Llama 3 和 DeepSeek-V3 为 128K，Qwen3 约 150K，gpt-oss 约 200K，Gemma 3 约 262K），所以在 $V$ 上做 softmax 本身就是一笔可观的开销；见 [1.2](02-tokenization.md)。
- **解码默认值：** 聊天模型通常把约 0.6–1.0 的温度与约 0.9–0.95 的 top-p 结合使用；模型卡片常常会给出推荐值。推理（reasoning）模型通常是采样而不是贪心解码，因为贪心解码在很长的输出里可能陷入重复循环。

## 常见误区 {#common-pitfalls-misconceptions}

- **"跨不同分词器，困惑度 10 就比困惑度 12 好。"** 不一定；请按字节或按词归一化。
- **把训练数据上的损失与泛化混为一谈。** 模型可以把小语料背到损失接近零（我们的二元玩具模型做不到，但大网络可以）。一定要跟踪留出集损失。
- **以为温度 0 是一种特殊模式。** 它是贪心解码的极限；实现里要对它做特殊处理，因为除以零没有定义。
- **以为模型"知道"它采样出的答案。** 采样是从分布里抽取；同一个提示可能产生不同答案，也可能选中一个低概率的错误 token。
- **目标错位一位（off-by-one）。** 输入是 $x_{1:T-1}$，目标是 $x_{2:T}$。忘了这个偏移，就会训练出一个只会复制输入的模型。

## 本章要点 {#key-takeaways}

- 语言模型是 token 序列上的概率分布；链式法则把它变成下一个 token 条件概率的乘积。
- 训练最小化平均负对数似然（交叉熵）；均匀模型得分是 $\log V$，完美模型是 0。
- 困惑度 $= e^{\text{loss}}$ 是有效分支因子，且只有分词器相同时才可比。
- 神经语言模型胜过 n-gram，靠的是通过嵌入共享统计强度，而不是对精确上下文计数。
- 生成就是反复采样；温度、top-k 和 top-p 在多样性与连贯性之间做权衡。
- 因为几乎每种技能都会在文本中留下痕迹，所以更好的下一个 token 预测意味着更强的通用能力，而且这种提升随规模可预测地发生。

## 自测 {#check-your-understanding}

??? question "某模型在测试集上的平均损失是每 token 2.3 nats。它的困惑度是多少？以 bits 计的损失又是多少？"
    困惑度 $= e^{2.3} \approx 9.97$，即约 10。以 bits 计：$2.3 / \ln 2 \approx 3.32$ bits 每 token。

??? question "为什么链式法则分解不是近似，而 n-gram 模型是？"
    链式法则 $p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$ 对任何分布都是精确的恒等式。n-gram 模型额外*假设* $p(x_t \mid x_{<t})$ 只依赖最后 $n-1$ 个 token，这会丢掉信息。

??? question "当温度趋于 0 和趋于无穷时，采样会怎样？"
    当 $\tau \to 0$ 时，最大的 logit 占主导，采样变成贪心（argmax）。当 $\tau \to \infty$ 时，所有缩放后的 logits 都趋近 0，分布在词表上变得均匀。

??? question "为什么 top-p 比 top-k 适应性更好？"
    Top-k 总是恰好保留 $k$ 个 token。当模型很自信（某个 token 概率为 0.97）时，$k = 50$ 仍会放进 49 个不太可能的 token；当许多 token 都合理时，$k$ 又可能太少。Top-p 则保留覆盖概率质量 $p$ 所需的任意数量的 token，所以候选集会随着模型的不确定性收缩或扩张。

??? question "一个刚初始化、词表有 128,000 个 token 的模型，报告的初始损失是 11.76。这是 bug 吗？"
    不是。$\ln 128{,}000 \approx 11.76$，恰好是均匀预测的损失。输出 logits 很小的随机初始化会给出几乎均匀的预测。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Bengio et al., 2003 | 第一个带有学习式词嵌入的神经语言模型；它针对 n-gram 稀疏性的论证，至今仍是神经语言模型最清晰的动机。 |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | 仅解码器语言模型 + pre-norm + 大规模网页语料 → 零样本任务迁移。确立了"把下一个 token 预测放大"的配方。 |
| :material-star: [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020 | 175B 参数；上下文（少样本）学习随规模涌现。重点读它的问题框架和评测方法。 |
| :material-book-open-variant: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al., 2019 | 为什么贪心和纯采样都会产生糟糕的文本，以及核（top-p）采样的提出。 |
