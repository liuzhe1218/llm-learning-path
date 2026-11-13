# 2.6 推理与测试时计算

!!! abstract "本章你将学到"
    - 为什么思维链（chain-of-thought，CoT）有效：更多的串行计算，以及由中间 token 构成的草稿纸
    - 自洽性（self-consistency，即多数投票），并通过一个概率计算实例看它何时有帮助、何时有害
    - 带验证器的 best-of-N、pass@k，以及结果奖励模型与过程奖励模型的区别
    - 对推理过程做搜索（束搜索、树搜索）的直观理解
    - "思考"型模型如何用测试时计算换取准确率，包括预算和 s1 的预算强制（budget forcing）

## 为什么重要 {#why-this-matters-the-problem}

到目前为止，我们提升模型的方法都是通过投入更多训练算力来改变它的**权重**。其实还有第二个杠杆：在**推理时**为每个问题花更多算力。一个人解难题时不会立刻给出答案，而是写下步骤、尝试不同思路、检查自己的结果。语言模型能不能也这样做？让它想得更久，能不能让它变得更好？

2024 年以来，答案是明确的"能"。"推理"或"思考"型模型（OpenAI 的 o1 系列、DeepSeek-R1、Qwen3 的思考模式、带扩展思考的 Claude、Gemini 的思考模型）在给出答案前会先生成很长的内部推理，它们在数学、代码和科学问题上的准确率会随思考 token 数的增加而上升。本章解释*为什么*多出来的 token 有帮助，以及花费测试时计算（test-time compute）的主要方式；[第 2.7 章](07-rl-with-verifiable-rewards.md)则解释如何用 RL 训练模型学会使用它。

## 直觉 {#intuition}

**Transformer 为每个 token 做的计算量是固定的。** 每个生成的 token 都要经过同样的 $n_{\text{layers}}$ 层（[第 1 部分](../part1/index.md)）。如果一个问题需要的串行计算步数超过网络深度所能提供的，模型就无法在一个 token 里"心算"出答案。但它可以把中间结果写下来，在生成下一个 token 时再用注意力把它们读回来。每生成一个 token，就多买到一次完整的网络前向计算，而写下来的 token 就充当了**外部工作记忆**（草稿纸）。

试试直接心算 $37 \times 48$，再对比写出来：$37\times48 = 37\times50 - 37\times2 = 1850 - 74 = 1776$。同一个大脑，更多的串行步骤，加上书面记忆。CoT 之于 LLM 就是这么回事。

花费更多推理算力大致有两种方式：

```text
  串行（想得更久）                              并行（想得更广）
  一条很长的思维链                              多次独立尝试，再挑选
  ┌──────────────────────────────────┐         ┌──── 尝试 1 → 42
  │ 步骤 1 → 步骤 2 → 检查 → 不对 → │         ├──── 尝试 2 → 42      多数投票 → 42
  │ 回退 → 步骤 2' → 答案            │         ├──── 尝试 3 → 17      或由验证器挑最好的
  └──────────────────────────────────┘         └──── 尝试 4 → 42
  （CoT、思考型模型、自我修订）                 （自洽性、best-of-N、搜索）
```

## 工作原理 {#how-it-works}

### 思维链提示 {#chain-of-thought-prompting}

Wei 等人（2022）表明，在提示中加入几个带逐步推理的示例（少样本 CoT），能显著提升大模型在算术、常识和符号推理上的表现，而对小模型几乎没有或完全没有帮助：使用 CoT 的能力随规模涌现。只需在问题后面加上一句"Let's think step by step"（零样本 CoT），对强模型也有帮助。

形式化地说，模型不再直接建模答案 $a$ 的 $p(a\mid x)$，而是先生成一段推理依据 $z$，再给出答案：

$$
p(a\mid x) = \sum_{z} p(z\mid x)\, p(a\mid x, z).
$$

用大白话说：答案的概率是对所有可能推理路径的平均。采样一条 CoT，就是采样一条路径 $z$；这个视角直接引出了自洽性，它通过采样多条路径来近似这个求和。

为什么有效，总结如下：

1. 每个问题得到**更多串行计算**（每个推理 token 对应一次网络前向）。
2. **工作记忆**：中间结果以 token 的形式存储，并通过注意力取回。
3. **分解**：训练数据（教科书、解题过程）里本来就包含逐步推理，所以模型学到了许多小而可靠的步骤，每一步都比一口气跳到结论容易。

CoT 文本并不保证如实反映模型到底是怎么得出答案的；它只是会*影响*答案的生成文本，这对可解释性和安全性都很重要（[第 2.9 章](09-agents-tool-use-and-safety.md)）。

### 自洽性（多数投票） {#self-consistency-majority-voting}

Wang 等人（2022）：在温度 $> 0$ 下采样 $N$ 条思维链解答，提取每条的最终答案，输出出现次数最多的那个。不需要训练，也不需要验证器，只是多采样。

为什么有效？如果模型对正确答案的单次采样概率高于任何一个具体的错误答案，那么随着 $N$ 增大，最常见的答案会收敛到正确答案。错误的推理路径往往分散到许多不同的错误答案上，而正确的路径会汇聚到同一个答案。

什么时候会失败？如果模型*最可能*给出的答案是错的，投票就会放大这个错误（见实例演算）。它还要求答案可以被精确比较（一个数字、一个选择题字母），而不是自由形式的长文。

### 带验证器的 best-of-N 与 pass@k {#best-of-n-with-a-verifier-and-passk}

**验证器**是任何能给候选解答打分的东西：学习得到的奖励模型、单元测试运行器、证明检查器。**Best-of-N** 采样 $N$ 个候选，返回得分最高的那个。

best-of-N 的上限是 **pass@N**：$N$ 个样本中*至少有一个*正确的概率。如果每个样本独立地以概率 $p$ 正确，则

$$
\text{pass@}N = 1 - (1-p)^N.
$$

用大白话说：$N$ 次尝试不全部失败的概率。完美的验证器能达到 pass@N；真实的验证器达不到，因为它有时会偏爱一个错误但看起来很有说服力的解答。对于*学习得到的*验证器，增大 $N$ 最终反而*有害*：样本越多，就越有可能碰到一个能骗过验证器的错误解答（这与[第 2.4 章](04-rlhf.md)中的过度优化是同一回事）。Cobbe 等人（2021）提出了 GSM8K，并训练验证器对采样得到的解答重排序，表明这种做法可以胜过微调一个更大的模型。

要从每个问题的 $n \ge k$ 个样本（其中 $c$ 个正确）中公平地*估计* pass@k，使用无偏估计量（来自 OpenAI 的 Codex 论文）

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

即从 $n$ 个样本中随机取 $k$ 个，其中至少包含一个正确样本的概率。

### 结果奖励模型与过程奖励模型 {#outcome-vs-process-reward-models}

验证器可以在两种粒度上做判断：

- **结果奖励模型（outcome reward model，ORM）**：只给最终答案/完整解答打分。训练标签很便宜（最终答案对不对？），但一个推理有缺陷、碰巧答案正确的解答也能得到满分，而且这个信号并不会告诉你错误的解答是在*哪里*出的错。
- **过程奖励模型（process reward model，PRM）**：给推理的*每一步*打分。一条解答的得分通常是各步正确概率的乘积（或最小值）。它能提供更稠密、更精确的反馈，但需要步骤级标签，成本很高。

Lightman 等人（2023）在模型生成的 MATH 解答上收集了约 80 万条人工步骤级标签（PRM800K），发现在 best-of-N 重排序中，PRM 比 ORM 更可靠地挑出正确解答。后续工作则自动生成步骤标签（例如，根据从某一步继续展开的 rollout 有多大比例能得到正确答案，来估计该步的正确性）。

不过在 2025 年，最强的开源推理模型（DeepSeek-R1，见[第 2.7 章](07-rl-with-verifiable-rewards.md)）是用**仅基于结果的规则奖励**训练的；R1 报告把 PRM 列为在大规模下对他们效果不佳的方法之一，原因包括难以定义"步骤"、标签噪声，以及对学习得到的模型进行奖励投机。PRM 在搜索和重排序中仍然有用。

### 对推理过程做搜索 {#search-over-reasoning}

有了步骤级打分器，就可以搜索，而不只是独立采样：

- **对步骤做束搜索**：保留得分最高的 $B$ 个部分解答，对每个解答采样若干个下一步来扩展，重新打分，再保留前 $B$ 个。
- **树搜索**（例如 Tree of Thoughts，Yao et al., 2023；蒙特卡洛树搜索的各种变体）：探索一棵由部分推理组成的树，用价值估计决定扩展哪些分支，并从死胡同里回退。

在早期错误就致命的问题上，搜索比独立采样更聪明地使用算力，但它实现和部署都很复杂，还依赖可靠的步骤打分器。思考型模型在很大程度上把搜索*内化*了：一条长思维链里就包括尝试某种方法、发现问题、回退（"等等，这不对……"）。

### 扩展测试时计算 {#scaling-test-time-compute}

Snell 等人（2024）研究了如何分配固定的推理预算，主要发现：

- 最佳策略取决于难度。对简单问题，**串行修订**（让模型修改自己的答案）很高效；对更难的问题，**针对验证器的并行采样/搜索**帮助更大。
- 能根据估计难度自适应调整的**计算最优**策略，比朴素的 best-of-N 高效得多。
- 在 FLOPs 相当的比较中，对于小模型本来就有一定成功率的问题，用好测试时计算的较小模型可以胜过大约大 14 倍的模型；而在最难的问题上，预训练规模仍然胜出。

结论：**测试时计算和训练算力在一定程度上可以互换**，这就是思考型模型的概念基础。

### 思考型模型、预算与预算强制 {#thinking-models-budgets-and-budget-forcing}

**思考型模型**（主要用 RL 训练，[第 2.7 章](07-rl-with-verifiable-rewards.md)）会在可见答案之前生成一长段推理，通常用 `<think> ... </think>` 之类的特殊标记界定。它的准确率通常大致随思考 token 数的*对数*线性增长：思考量每翻一倍，带来的提升相近。

因为思考很昂贵，产品会提供**预算**：思考 token 的最大数量或一个粗粒度的"推理力度"档位，有些模型还支持按请求开关思考。2025–2026 年，同时训练了这两种模式的混合模型很常见。

**s1 与预算强制**（Muennighoff et al., 2025）表明这件事可以做得多简单。他们仅用 1,000 道精心挑选、附带长推理轨迹的题目（s1K）微调 Qwen2.5-32B-Instruct，然后在推理时控制思考长度：

- **限制思考**：达到预算时，强制插入思考结束分隔符（以及类似"Final Answer:"的提示），让模型必须作答。
- **延长思考**：当模型想提前结束思考时，抑制思考结束分隔符，并追加一个词"Wait"；模型随后常常会重新审视自己的推理并改正错误。

这样，竞赛数学的准确率随预算增加而上升，s1 模型报告的结果在竞赛数学基准上超过了 OpenAI 的 o1-preview。更广泛的启示是：许多推理能力是**潜藏**在强基座模型里的，少量数据加上测试时控制就能把它解锁。

### STaR：从自己的推理中学习 {#star-learning-from-your-own-reasoning}

STaR（Zelikman et al., 2022）把测试时推理和训练闭合成一个环：为许多问题生成推理依据；保留那些得出正确答案的；对失败的，把正确答案作为提示给模型，让它写出推理依据（"合理化"）；在保留下来的推理依据上微调；重复。这种"专家迭代"就是带 0/1 奖励的 REINFORCE（[第 2.3 章](03-rl-foundations-for-llms.md)），也是 RLVR 的前身。

## 代码 {#code}

### 1. 给定采样得到的解答，做自洽性与 best-of-N {#1-self-consistency-and-best-of-n-given-sampled-solutions}

```python
import re
from collections import Counter

def extract_answer(text):
    """若有则取最后一个 \\boxed{...}，否则取文本中最后一个数字。"""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def majority_vote(samples):
    answers = [a for a in map(extract_answer, samples) if a is not None]
    if not answers:
        return None, 0.0
    ans, count = Counter(answers).most_common(1)[0]
    return ans, count / len(answers)          # 答案及其得票占比（一种置信度信号）

def best_of_n(samples, score_fn):
    """score_fn: 任意验证器，例如奖励模型或单元测试通过率。"""
    return max(samples, key=score_fn)

samples = ["... so 37*48 = 1776. \\boxed{1776}",
           "37*48 = 37*50 - 74 = 1776",
           "37*48 = 1666, \\boxed{1666}",
           "... therefore the answer is 1776"]
print(majority_vote(samples))                # ('1776', 0.75)
```

### 2. 精确的多数投票准确率与无偏 pass@k 估计量 {#2-exact-majority-vote-accuracy-and-an-unbiased-passk-estimator}

```python
from math import comb

def majority_correct_prob(p, n):
    """两种答案的最坏情形：每个样本以概率 p 正确，否则给出同一个错误答案。
    多数票（n 为奇数）正确，当且仅当正确样本多于 n/2。"""
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))

def pass_at_k(n, c, k):
    """由 n 个样本（其中 c 个正确）得到的无偏估计。"""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

for p in (0.6, 0.4):
    print(p, [round(majority_correct_prob(p, n), 3) for n in (1, 5, 15, 41)])
print(pass_at_k(n=10, c=3, k=4))             # 0.833
```

### 3. 在思考型模型上做预算强制 {#3-budget-forcing-with-a-thinking-model}

```python
# pip install torch transformers  (Qwen3 模型使用 <think> ... </think>)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
END_THINK = tok.convert_tokens_to_ids("</think>")

def think_with_budget(question, min_think=200, max_think=800, max_waits=2):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    used, waits = 0, 0
    while True:
        ids = tok(text, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=max_think - used, do_sample=False,
                             eos_token_id=[END_THINK, tok.eos_token_id])
        new = out[0, ids.shape[1]:]
        used += len(new)
        ended = len(new) > 0 and new[-1].item() == END_THINK
        text += tok.decode(new[:-1] if ended else new, skip_special_tokens=False)
        if ended and used < min_think and waits < max_waits:
            text += "\nWait"                   # 抑制思考结束：让模型继续想
            waits += 1
            continue
        break
    text += "\n</think>\n\n"                   # 封顶：强制结束思考并作答
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=100, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

print(think_with_budget("How many r's are in the word 'strawberry'?"))
```

## 实例演算 {#worked-example}

**多数投票何时有帮助？** 假设对投票最不利的情形：每个样本以概率 $p$ 正确，且每个错误样本给出的都是*同一个*错误答案（如果错误答案是分散的，投票的效果会更好）。设 $N$ 为奇数，当正确样本超过一半时投票结果正确，这是一个二项分布的尾部：

$$
P(\text{vote correct}) = \sum_{k > N/2} \binom{N}{k} p^k (1-p)^{N-k}.
$$

当 $p = 0.6$、$N = 5$ 时：

$$
\binom{5}{3}0.6^3 0.4^2 + \binom{5}{4}0.6^4 0.4 + 0.6^5 = 0.3456 + 0.2592 + 0.0778 = 0.683.
$$

| $N$ | 1 | 5 | 15 | 41 |
|---|---|---|---|---|
| $p = 0.6$ | 0.600 | 0.683 | 0.787 | 0.903 |
| $p = 0.4$ | 0.400 | 0.317 | 0.213 | 0.097 |

投票会**放大模型最可能给出的那个答案**：当正确答案是众数时，它把准确率推向 1；当某个错误答案是众数时，则推向 0。

**best-of-N 的上限。** 如果一道难题的单次成功率为 $p = 0.2$：pass@4 $= 1 - 0.8^4 \approx 0.590$；pass@16 $= 1 - 0.8^{16} \approx 0.972$。一个完美的验证器（例如完整规定了任务的单元测试），可以用 16 个样本把一个 20% 的模型变成 97% 的系统，代价是推理成本扩大 16 倍。如果那 80% 的错误样本都集中在同一个错误答案上，多数投票在这里完全帮不上忙。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **推理是训练出来的，而不只是提示出来的。** 前沿模型在难题上会生成数千到数万个思考 token；这种行为来自在可验证任务上的 RL（[第 2.7 章](07-rl-with-verifiable-rewards.md)）以及对这类轨迹的蒸馏（[第 2.8 章](08-distillation-and-efficient-fine-tuning.md)）。
- **基准成绩要附带计算设置。** 像"用 $k$ 个样本加多数投票的 AIME 准确率"或"16 个样本平均的 pass@1"这样的结果很常见；只有在设置一致时才能比较数字。
- **并行 + 串行。** 顶尖系统在最难的问题上，会把长思考与并行尝试、结果聚合结合起来。
- **过度思考是一种成本。** 思考型模型可能在简单问题上耗费大量 token；预算、力度档位，以及 RL 期间的长度惩罚（例如 Kimi k1.5 的"long2short"方法）可以缓解。
- **忠实性。** 各实验室在研究可见推理是否反映了产生答案的真实计算，因为 CoT 监控被用作一种安全工具。

## 常见误区 {#common-pitfalls-misconceptions}

- **"CoT 总是有帮助。"** 对小模型和简单查表式问题，它可能有害，或者除了增加成本外毫无作用。
- **"样本越多越好。"** 当众数是错的时，多数投票反而有害；带学习得到的验证器的 best-of-N 最终会对验证器过度优化。
- **把 pass@k 与 pass@1 直接比较。** pass@k 假定有一个先知来挑出正确样本；对于必须只给出一个答案的实际系统，它只是一个上界。
- **只用 $k$ 个样本来估计 pass@k。** 请用 $n > k$ 个样本和无偏估计量；用代入式 $\hat p$ 算 $1-(1-\hat p)^k$ 是有偏的。
- **把 CoT 当作忠实的解释。** 它是有用的证据，而不是关于模型内部机制的事实。

## 本章要点 {#key-takeaways}

- 每生成一个 token，就是又一次经过整个网络；CoT 增加了串行计算和一块由 token 构成的草稿纸。
- 自洽性采样许多条 CoT 并投票；它放大的是模型的众数答案，无论好坏。
- Best-of-N 受限于 pass@N $= 1-(1-p)^N$ 和验证器质量；PRM 给步骤打分，ORM 给结果打分。
- 测试时计算可以在力所能及的问题上替代模型规模（Snell et al.）；思考型模型把这变成了默认做法。
- 预算强制（s1）表明，只需一个分隔符和一个"Wait"，就能在推理时控制思考长度。

## 自测 {#check-your-understanding}

??? question "为什么固定深度的 Transformer 在允许写出中间步骤时能解决更多问题？"
    每个生成的 token 都由一次完整的前向传播算出，所以写下 $T$ 个 token 就提供了 $T$ 倍的串行计算量，而且写出来的中间结果可以通过注意力读回，起到工作记忆的作用。

??? question "某模型 45% 的时候答对，55% 的时候给出同一个错误答案。对 31 个样本做多数投票有帮助吗？"
    没有，反而有害：众数是错的，所以随着 $N$ 增大，投票结果会收敛到错误答案。

??? question "ORM 与 PRM 有什么区别？各说出一个优点。"
    ORM 给最终结果打分（标签便宜、简单）；PRM 给每个推理步骤打分（更稠密、能定位错误、用于重排序更可靠，但需要步骤标签）。

??? question "20 个样本中有 5 个正确，估计 pass@1 和 pass@2。"
    pass@1 $= 1 - \binom{15}{1}/\binom{20}{1} = 1 - 15/20 = 0.25$。pass@2 $= 1 - \binom{15}{2}/\binom{20}{2} = 1 - 105/190 \approx 0.447$。

??? question "s1 的预算强制是如何延长思考的？"
    当模型在达到期望预算之前就输出思考结束分隔符时，抑制该分隔符并追加"Wait"，促使模型继续推理（通常会重新检查自己的工作）。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al., 2022 | 逐步推理的示例让大模型在多步问题上好得多。 |
| :material-book-open-variant: [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022 | 采样多条推理路径再多数投票：最简单的测试时扩展。 |
| :material-book-open-variant: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) — Cobbe et al., 2021 | 提出 GSM8K，以及基于验证器对采样解答重排序的方法。 |
| :material-book-open-variant: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — Zelikman et al., 2022 | 保留自己生成的、得出正确答案的推理依据，微调，再重复。 |
| :material-star: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al. (OpenAI), 2023 | 在挑选数学解答上，过程奖励模型胜过结果奖励模型；发布了 PRM800K。 |
| :material-book-open-variant: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Yao et al., 2023 | 对推理步骤做显式树搜索，并配合自我评估。 |
| :material-star: [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) — Snell et al., 2024 | 如何分配推理算力；何时它能胜过更大的模型。 |
| :material-new-box: [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) — Muennighoff et al., 2025 | 1K 条长 CoT 样本加预算强制；推理能力在很大程度上潜藏于强基座模型之中。 |
