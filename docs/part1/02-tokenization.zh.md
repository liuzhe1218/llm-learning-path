# 1.2 分词

!!! abstract "本章你将学到"
    - 为什么大模型处理的是**子词 token**，而不是字符或单词。
    - **字节对编码（Byte-Pair Encoding，BPE）**的训练算法如何工作，并在一个玩具语料上逐次合并地推演一遍。
    - **字节级 BPE** 增加了什么，以及编码和解码如何进行。
    - 词表大小如何在序列长度、嵌入大小和多语言覆盖之间做权衡。
    - 分词器的怪现象（数字、空白、中日韩文本）以及它们如何泄露到模型行为中。

## 为什么重要 {#why-this-matters-the-problem}

神经网络吃进去的是数字，不是文本。[1.1](01-language-modeling.md) 章假设了一个由 token 组成的词表 $\mathcal{V}$；本章要决定这些 token 到底是什么。这个选择比看上去重要得多：

- 它决定了一段文本需要**多少个预测步**。注意力的开销随序列长度增长（平方级，见 [1.6](06-long-context-and-efficient-attention.md)），生成时间则随输出 token 数线性增长。
- 它决定了**嵌入层和输出层**的大小（各 $V \times d$ 个参数）。
- 它决定了模型能"看到"什么。如果模型收到的"2025"是一个不透明的 token，它就得间接地去学数字的各位；如果它看到的是 "2"、"0"、"2"、"5"，就可以逐位做算术。
- 它决定了不同语言之间的公平性：同一句话在一种语言里消耗的 token 数，可能是另一种语言的好几倍。

分词器在模型之前训练一次，并在模型的整个生命周期里保持冻结。选错了代价很高。

## 直觉 {#intuition}

有三种显而易见的粒度，每一种失败的方式都不同：

| 单位 | 词表 | 序列长度 | 问题 |
|---|---|---|---|
| 字符 / 字节 | 很小（256 个字节） | 很长 | 模型必须花费容量和算力把字母拼成单词 |
| 单词 | 巨大、开放 | 很短 | 任何没见过的词（"ChatGPT"、拼写错误、新名字）都是词表外（out-of-vocabulary）的；词形变化（"run"、"running"）无法共享 |
| **子词** | 自己选定（32K–262K） | 中等 | 大家都在用的折中方案 |

子词分词器把高频字符串（"the"、" because"、"ing"）保留为单个 token，把罕见词拆成可复用的片段（"tokenization" → "token" + "ization"）。不会有任何东西是词表外的，因为最坏的情况下，一个词也可以用小片段逐个拼出来。BPE 用一个简单的压缩思想来找到这些片段：**反复把最高频的相邻符号对粘在一起**。

## 工作原理 {#how-it-works}

### BPE 训练 {#bpe-training}

在 Sennrich 等人（2015）把它用于 NLP 之前，字节对编码是一种数据压缩算法。训练的输入是一个语料和目标词表大小 $V$：

1. **预分词（pre-tokenize）**：把文本切成块（大致是单词，标点和空格由规则处理）。合并永远不会跨越块的边界。统计每个块出现的次数。
2. **初始化**词表为所有基础符号（字符，或字节级 BPE 中的 256 个字节值），并把每个块写成基础符号的序列。
3. **统计**所有块中每一对相邻符号，按块的频次加权。
4. 把最高频的符号对 $(a, b)$ **合并**成一个新符号 $ab$：把它加入词表，记录规则"$a\ b \to ab$"，并改写每一处出现。
5. **重复**第 3–4 步，直到词表有 $V$ 个符号。

输出是词表加上一个**有序的合并规则列表**。顺序很重要：编码时按学到的顺序重放这些合并。

### 编码与解码 {#encoding-and-decoding}

**编码**新文本：先预分词，把每个块变成基础符号，然后反复应用块中任何相邻对所匹配的*最早学到的*合并规则，直到没有规则可用。再把每个最终符号映射为它的整数 id。

**解码**：把 id 映射回它们的字符串（或字节串）并拼接起来。解码只是一次查表；编码才是唯一不平凡的方向。

### 字节级 BPE {#byte-level-bpe}

如果基础符号是 Unicode 字符，基础词表就很大（Unicode 分配了约 150,000 个字符），而且一个没见过的字符（一个 emoji、一个罕见的汉字）依然会打破"永远不会词表外"的承诺。GPT-2 引入了**字节级 BPE**：先把文本编码成 UTF-8 字节，这样基础词表恰好就是 256 个字节值，然后在字节上运行 BPE。任何语言的任何字符串，甚至是非法文本，都能被表示。常见的多字节序列（一个汉字是 3 个 UTF-8 字节，"é" 是 2 个）只要足够高频就会被合并。

代价是：单个字节自身不一定是合法的 UTF-8，所以模型可能输出字节无法干净解码的 token 序列。解码器通过缓冲字节或替换非法序列来处理。

### 预分词 {#pre-tokenization}

在原始文本上直接运行 BPE，会很乐意学到像 "dog." 或 "the cat" 这样横跨标点和词边界的合并，把词表浪费在这些组合上。因此 GPT 风格的分词器先用正则表达式把文本切成块，BPE 只在块内部合并。GPT-2 的模式大致切分为：

- 可选的前导空格后接字母（" hello"、"world"），
- 可选的前导空格后接数字（后来的分词器把数字串限制为 1–3 个字符），
- 可选的前导空格后接标点（" ..."、"!"），
- 连续的空白，
- 常见的英语缩写（"'s"、"'t"、"'re"）。

所以 `"Hello world, it's 2025!"` 在任何合并之前会变成 `["Hello", " world", ",", " it", "'s", " 2025", "!"]`（按 GPT-2 的规则；更新的模式会把 " 2025" 进一步拆开）。注意空格属于*后面*的那个词；这就是 " world" 和 "world" 是不同 token 的原因。

### 特殊 token {#special-tokens}

除了学出来的 token，每个词表都会保留一些**特殊 token**，它们永远不来自 BPE：一个文本结束标记（如 `<|endoftext|>`），在预训练时放在文档之间；后来还有聊天角色、工具调用或图像的标记。它们在预分词之前就被匹配，并直接映射到保留的 id。用户键入字面字符串 "<|endoftext|>" 时，绝不能注入真正的控制 token，所以生产环境的分词器把特殊 token 的解析作为显式的可选项。

### 其他子词算法 {#other-subword-algorithms}

- **WordPiece**（BERT）类似 BPE，但选择的是最能提高训练数据似然的合并，而不是纯粹频次最高的对。
- **Unigram LM**（Kudo，2018）是自顶向下的：从一个很大的候选词表开始，剪掉那些移除后对 unigram 语言模型似然伤害最小的片段。它可以为同一个词给出带概率的多种切分，从而实现"子词正则化"（subword regularization）。
- **SentencePiece** 是一个库，不是一个算法：它实现了 BPE 和 Unigram，并把输入当作原始字符流（空格变成一个可见符号"▁"），所以不需要特定语言的预分词器。Llama 1 和 2 使用 32K 词表的 SentencePiece BPE。

### 词表大小的权衡 {#vocabulary-size-trade-offs}

设 $V$ 为词表大小，$d$ 为模型的隐藏维度。

- **序列长度。** $V$ 越大，合并越多、越长，所以同样的文本会变成更少的 token。token 更少意味着注意力计算更少、上下文窗口里装得下更多文本、生成更快。
- **参数量。** 输入嵌入矩阵是 $V \times d$，输出（反嵌入）矩阵是 $d \times V$，除非它们被共享（[1.4](04-modern-block.md)）。对 Llama 3 8B（$V = 128{,}256$，$d = 4096$），每个矩阵有 $128{,}256 \times 4096 \approx 525\text{M}$ 个参数，所以两者加起来约 1.05B，占模型 8B 的一部分。
- **输出层的计算。** 每个位置都要计算 $V$ 个 logits：输出投影每个 token 需要 $2dV$ FLOPs。对小模型来说，这占总计算量的很大一部分。
- **罕见 token 训练不足。** 每个 token 的嵌入只有在该 token 出现时才会得到梯度。$V$ 很大时，许多 token 很罕见，它们的嵌入学得很差。
- **多语言性。** 主要在英文上训练的词表，分给其他文字的合并很少，这些文字就会被切成很多 token。更大的、在多语言数据上训练的词表是主要的解决办法；这也是词表从 32K（Llama 2）增长到 128K（Llama 3）、约 150K（Qwen）、约 200K（gpt-oss）和约 262K（Gemma 3）的原因之一。

## 代码 {#code}

一个完整、可运行的字节级 BPE 分词器：训练、编码和解码，约 60 行。它沿用了 GPT-2 风格分词器的结构（正则预分词，然后在 UTF-8 字节上做 BPE），只是正则做了简化。

```python
import re
from collections import Counter

# 简化的 GPT-2 风格预分词器：可选前导空格 + 字母，
# 1-3 位一组的数字，标点串，空白。
PAT = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+")

def pair_counts(chunks):
    counts = Counter()
    for ids, freq in chunks.items():
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += freq
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return tuple(out)

class ByteBPE:
    def train(self, text, vocab_size):
        assert vocab_size >= 256
        # 1) 预分词，2) 每个块 -> UTF-8 字节值的元组，并带上其计数
        chunks = Counter(tuple(c.encode("utf-8")) for c in PAT.findall(text))
        self.merges = {}                                   # (a, b) -> 新 id，按学到的顺序
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes
        for new_id in range(256, vocab_size):
            counts = pair_counts(chunks)
            if not counts:
                break
            best = max(counts, key=counts.get)             # 最高频的相邻对
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunks = Counter({merge(ids, best, new_id): f for ids, f in chunks.items()})
        return self

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            # 应用块中出现的、最早学到的合并
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text):
        out = []
        for chunk in PAT.findall(text):
            out += self._encode_chunk(tuple(chunk.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

corpus = "low lower lowest newest widest " * 20 + "the newest model is the lowest risk. 你好世界 2025"
tok = ByteBPE().train(corpus, vocab_size=280)
ids = tok.encode("the lowest newest 你好")
print(len("the lowest newest 你好".encode()), "bytes ->", len(ids), "tokens:", ids)
print([tok.vocab[i] for i in ids])
assert tok.decode(ids) == "the lowest newest 你好"
```

输出：24 个字节变成 12 个 token。" lowest" 和 " newest" 各是一个 token（它们在训练文本中很高频），"the" 仍是三个字节（它太罕见，在只有 24 次合并的预算下没能争到合并），每个汉字仍是它的 3 个原始 UTF-8 字节，因为语料里的中文太少。最后这一点就是多语言问题的缩影。

!!! tip "让它更快"
    这个实现在每次合并之后都会重新统计所有对，复杂度是 $O(\text{merges} \times \text{corpus})$。生产级的训练器（Hugging Face `tokenizers`、SentencePiece）会增量更新对的计数，并用 Rust 或 C++ 编写。推理时的编码可以用 OpenAI 的 `tiktoken`，它是一个快速的 BPE 编码器。

## 实例演算 {#worked-example}

我们在 Sennrich 等人的经典玩具语料上训练 BPE：单词 "low" ×5、"lower" ×2、"newest" ×6、"widest" ×3。我们在字符级工作，并追加一个词尾标记 "\_"，这样 "est\_"（词尾）就能与词中的 "est" 区分开。平局时先看到的对优先。

**初始状态**（每个词拆成字符，并带上它的计数）：

```text
l o w _          ×5
l o w e r _      ×2
n e w e s t _    ×6
w i d e s t _    ×3
```

**对的计数。** "e s" 出现在 newest（6）和 widest（3）中，共 9；同样 "s t" = 9，"t \_" = 9。"w e" = 2（lower）+ 6（newest）= 8。"l o" = "o w" = 5 + 2 = 7。赢家是 "e s"，计数 9。

| 合并序号 | 合并的对 | 计数 | 合并后的语料 |
|---|---|---|---|
| 1 | e + s → es | 9 | `n e w es t _`, `w i d es t _` |
| 2 | es + t → est | 9 | `n e w est _`, `w i d est _` |
| 3 | est + \_ → est\_ | 9 | `n e w est_`, `w i d est_` |
| 4 | l + o → lo | 7 | `lo w _`, `lo w e r _` |
| 5 | lo + w → low | 7 | `low _`, `low e r _` |
| 6 | n + e → ne | 6 | `ne w est_` |
| 7 | ne + w → new | 6 | `new est_` |
| 8 | new + est\_ → newest\_ | 6 | `newest_` |

经过 8 次合并，词表就是基础字符加上 {es, est, est\_, lo, low, ne, new, newest\_}。算法没有任何语言学知识，纯粹依靠频次就发现了后缀 "est" 和词干 "low"。

**编码一个没见过的词 "lowest"。** 从 `l o w e s t _` 开始，按顺序重放合并：

1. 合并 1（e s）：`l o w es t _`
2. 合并 2（es t）：`l o w est _`
3. 合并 3（est \_）：`l o w est_`
4. 合并 4（l o）：`lo w est_`
5. 合并 5（lo w）：`low est_`
6. 合并 6–8 不适用。

结果：**[low, est\_]**，一个训练中从未见过的词被编码成两个 token，且由有意义的片段组成。

**词表大小的代价。** 取一个小模型，$d = 1024$，总共约 0.6B 参数，使用 Qwen 量级的词表 $V = 151{,}936$。

- 嵌入矩阵：$151{,}936 \times 1024 \approx 156\text{M}$ 个参数，约占模型的四分之一。共享输入和输出嵌入（[1.4](04-modern-block.md)）可以避免付两次。
- 输出投影的计算：$2dV \approx 311\text{M}$ FLOPs 每 token，而整个前向传播大约是 $2N \approx 1.2$ GFLOPs：约四分之一的计算量花在了产生 logits 上。
- 对于 $d = 8192$ 的 70B 模型，同样的词表代价是 $2 \times 8192 \times 151{,}936 \approx 2.5$ GFLOPs 每 token，不到它约 $\approx 140$ GFLOPs 前向传播的 2%。

所以大词表对大模型很便宜，对小模型很贵，这也是为什么一个系列里最小的模型常常共用大词表、但同时共享嵌入。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **字节级 BPE 占主导。** 从 GPT-2 起，Llama 3（128K 词表，建立在 OpenAI 的 `tiktoken` 词表之上，并为其他语言增加了额外 token）、Qwen（约 150K）、DeepSeek-V3（128K）和 gpt-oss（`o200k_harmony`，约 200K）都使用字节级 BPE。Gemma 使用约 262K token 的 SentencePiece 词表。
- **更大的词表是值得的。** Llama 3 报告说，与 Llama 2 的分词器相比，它的新分词器把英文的压缩率从每 token 3.17 个字符提高到 3.94 个，意味着模型在相同算力下读到更多文本。
- **数字处理是一个刻意的选择。** Llama 1/2 把数字拆成单个数字。GPT-4 时代的分词器（以及 Llama 3 的）把数字串切成最多 3 位一块。无论哪种，目的都是避免任意的多位数变成单个不透明的 token。
- **特殊 token**，如 `<|begin_of_text|>`、`<|end_of_text|>` 和聊天角色标记，是在 BPE 之外加入词表的。由它们构成的聊天模板属于后训练的议题（[第二部分](../part2/index.md)）。
- **无分词器模型**直接在字节上操作，并通过学习得到的动态分组来处理，这是一个活跃的研究方向，但还没有哪个前沿模型取代了子词分词。

## 常见误区 {#common-pitfalls-misconceptions}

- **前导空格是 token 的一部分。** 在 GPT 风格的分词器里，" hello" 和 "hello" 是不同的 token。以空格结尾的提示会迫使模型接着生成一个不带前导空格的不常见 token，输出质量就会下降。
- **数字的分词不一致。** 在 GPT-2 的分词器里，"1234" 和 "1235" 的切分方式可能不同，这让算术更难学。数字符（"how many r's in strawberry?"）之所以难，也是同样的原因：模型看到的是 token，不是字母。
- **代码中的空白。** 早期的分词器每个缩进空格消耗一个 token；现代的分词器有专门表示连续空格和制表符的 token。
- **中日韩及其他文字成本更高。** 在字节级 BPE 下，一个没被合并的汉字要花 3 个 token。Petrov 等人（2023）测量了同样内容在不同语言间 token 数的巨大差异，这会影响成本、延迟和有效上下文长度。
- **故障 token（glitch tokens）。** 一个在分词器训练数据中很高频、但在模型训练数据中很罕见的 token（例如从爬取的论坛里来的用户名），它的嵌入几乎没被训练过，用它来提示会产生怪异的输出。请用能代表模型语料的数据来训练分词器。
- **换分词器就意味着重新训练。** token id 已经固化在嵌入矩阵里；你不能在训练好的模型上直接换分词器，至少得重新学习嵌入。

## 本章要点 {#key-takeaways}

- 子词分词在词表大小和序列长度之间取得平衡，并且永远不会产生词表外 token。
- BPE 训练 = 反复合并最高频的相邻对；编码则按顺序重放学到的合并。
- 字节级 BPE 从 256 个字节值出发，因此能编码任何语言的任何字符串。
- 更大的词表缩短序列并有利于多语言文本，代价是更大的嵌入/输出矩阵和很少被训练到的 token。
- 分词器的怪现象（空格、数字、文字）会直接表现为模型的怪现象；许多"大模型数不清字母"的失败其实是分词造成的假象。

## 自测 {#check-your-understanding}

??? question "为什么 BPE 需要一个有序的合并列表，而不只是最终的词表？"
    利用词表中的 token，同一个字符串可以有许多种切分方式。按学到的顺序重放合并，能得到一个与模型训练时所见一致的确定性切分。（在词表上做贪心最长匹配会给出不同的、不一致的切分。）

??? question "使用字节级 BPE，任何输入字符串都可能产生未知 token 吗？"
    不会。每个字符串都是字节序列，而全部 256 个字节值都在基础词表里，所以最坏的情况下，字符串会被逐字节编码。

??? question "若输入和输出嵌入不共享，把词表从 64K 翻倍到 128K、$d = 4096$，会增加多少参数？"
    每个矩阵增加 $64{,}000 \times 4096 \approx 262\text{M}$（为取整数，用 64K = 64,000），所以两者加起来增加约 524M 个参数。值不值得取决于序列缩短了多少。

??? question "在实例演算中，为什么我们要加词尾标记 '\_'？"
    它让 BPE 能区分词尾的后缀（"newest" 中的 "est\_"）和词中的同样字母。字节级 BPE 用不同的方式达到类似效果：预分词把前导空格附在词上，所以词首的 " est" 与词中的 "est" 不同。

??? question "某个模型不擅长逐字母反转单词。这是推理能力的失败吗？"
    往往在很大程度上是分词造成的假象：模型收到的是多字符的 token，必须靠记忆才知道每个 token 是怎么拼写的。在提示里把单词拆成单个字符（例如用空格隔开）通常会大有帮助。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2015 | 把 BPE 引入 NLP：从字符出发，反复合并最高频的对。如今几乎所有大模型分词器的基础；本章的玩具语料就来自这里。 |
| :material-book-open-variant: [SentencePiece](https://arxiv.org/abs/1808.06226) — Kudo & Richardson, 2018 | 与语言无关的分词器，把文本当作原始 Unicode（包括空白），在多语言和中日韩占比高的模型中很流行。 |
| :material-book-open-variant: [Subword Regularization](https://arxiv.org/abs/1804.10959) — Kudo, 2018 | Unigram LM 分词器，以及在训练中对多种切分进行采样。 |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | 第 2.2 节介绍了字节级 BPE，并解释了预分词规则。 |
| :material-book-open-variant: [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) — Petrov et al., 2023 | 测量了同样内容的 token 数在不同语言之间的差异，以及由此带来的成本/延迟后果。 |
