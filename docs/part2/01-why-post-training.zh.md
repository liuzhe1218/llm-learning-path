# 2.1 为什么要后训练？

!!! abstract "本章你将学到"
    - 为什么预训练得到的基座模型不像助手，并通过具体例子说明
    - 每个后训练阶段（SFT、偏好调优、RLVR、蒸馏）分别为模型带来什么
    - 对话模板和特殊 token 如何把一段对话变成单个 token 序列
    - 两种相互竞争的观点："对齐只是表面功夫" vs "RL 带来了真正的能力"
    - 如何评估后训练模型：基准测试、LLM-as-a-judge，以及人类偏好竞技场

## 为什么重要 {#why-this-matters-the-problem}

在[第一部分](../part1/index.md)中，我们训练模型在海量文本语料上最小化下一个 token 的交叉熵。得到的**基座模型（base model）**已经学会了语法、事实、编程惯用法，以及相当多的推理能力。但它唯一的目标是生成*到目前为止这篇文档的一个可能续写*，这和"有帮助"不是一回事。

假设你向基座模型输入：

```text
What is the capital of Australia?
```

互联网上合理的续写包括：

```text
What is the capital of Australia?
What is the capital of Canada?
What is the capital of Brazil?
```

（一份测验题列表），或者

```text
What is the capital of Australia? Many people think it is Sydney, but ...
```

（一篇文章），或者干脆就是正确答案。基座模型并没有"*你*在向*它*提问、并期待一个回答"这样的概念。它也不知道何时该停下，没有稳定的人设，没有拒绝危险请求的倾向，除了训练文本里恰好有的那些，它也不偏好诚实而非自信地胡说。

你*可以*用精心设计的提示词来引导基座模型（带示例问答对的"少样本"提示），但这很脆弱。后训练则把期望的行为固化进权重里，让每个用户默认就能得到它。

## 直觉 {#intuition}

一个有用的心智图景：预训练构建了一座巨大的*可能*行为的图书馆（数据中出现过的每一种写作风格、每一种人设、每一档质量）。后训练做的是对其中一小部分行为进行**选择**和**打磨**：

```text
           基座模型：对"互联网上所有文档"的一个分布
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 论坛吐槽 · 测验列表 · 教科书 · 专家回答 · 垃圾信息 · 代码 · ...            │
   └────────────────────────────────────────────────────────────────────────────┘
                 │ SFT："像这样、按这个格式回答"
                 ▼
          ┌───────────────────────────────┐
          │ 助手风格的回答                │  （格式、人设、指令遵循）
          └───────────────────────────────┘
                 │ 偏好调优："这个回答比那个更好"
                 ▼
             ┌──────────────────────┐
             │ 有用、无害、         │      （风格、语气、安全、校准）
             │ 格式良好             │
             └──────────────────────┘
                 │ 可验证奖励的强化学习："这个回答是*正确*的"
                 ▼
               ┌─────────────────┐
               │ 严谨的          │          （长推理、自我检查、工具使用）
               │ 推理智能体      │
               └─────────────────┘
```

每个阶段都用更强、更具体的"什么算好"的信号，把分布进一步收窄。

## 工作原理 {#how-it-works}

### 各阶段分别带来什么 {#what-each-stage-adds}

| 阶段 | 训练信号 | 典型的行为变化 | 章节 |
|---|---|---|---|
| **监督微调（SFT）** | 示范："对这个提示，就生成这样的回答" | 以助手风格回答、遵循指令、使用对话格式、在一轮结束时停下 | [2.2](02-supervised-fine-tuning.md) |
| **偏好优化**（RLHF、DPO） | 比较："回答 A 比回答 B 更好" | 更好的语气和有用性、更少的有害输出、对开放式任务有更好的判断 | [2.4](04-rlhf.md)、[2.5](05-direct-preference-optimization.md) |
| **可验证奖励的强化学习（RLVR）** | 由程序检查正确性（数学答案、单元测试） | 更长、更仔细的推理，自我验证，在可检验任务上准确率更高 | [2.7](07-rl-with-verifiable-rewards.md) |
| **蒸馏** | 更强的教师模型的输出或概率 | 小模型以低成本继承大模型的行为 | [2.8](08-distillation-and-efficient-fine-tuning.md) |
| **安全训练**（混入所有阶段） | 拒答示范、无害性偏好、书面规范 | 拒绝有害请求，但不拒绝无害请求 | [2.9](09-agents-tool-use-and-safety.md) |

为什么不只做一个阶段？因为每种信号的成本和强度都不同：

- **示范**每条的成本很高（得有人写出理想回答），但信息量非常大：它直接告诉模型该怎么做。
- **比较**更便宜（评判比写作容易），能捕捉难以示范的微妙品质，但每次比较大约只携带一个比特的信息。
- **可验证奖励**一旦有了检查器几乎是免费的，而且不会被风格蒙骗，但只存在于答案可检验的任务上。

完整流水线的经典示范是 InstructGPT（Ouyang et al., 2022）：先 SFT，再用人类比较数据训练奖励模型，最后用 PPO。在人类标注员看来，1.3B 参数的 InstructGPT 的输出优于 175B 参数的 GPT-3 基座模型。后训练不是表面的装饰层；在那个设定下，它对模型有用程度的改变超过了参数量增加 100 倍所带来的改变。

### 对话模板与特殊 token {#chat-templates-and-special-tokens}

Transformer 消费的是**一条扁平的 token 序列**。然而对话是有结构的：一条系统消息、交替出现的用户和助手轮次，也许还有工具调用。**对话模板（chat template）**就是把这种结构序列化成单个字符串的约定，它使用普通文本中从不出现的**特殊 token**。

下面是 Qwen 模型家族使用的 ChatML 风格模板。这段对话

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Australia?"},
    {"role": "assistant", "content": "The capital of Australia is Canberra."},
]
```

会被渲染为：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of Australia?<|im_end|>
<|im_start|>assistant
The capital of Australia is Canberra.<|im_end|>
```

`<|im_start|>` 和 `<|im_end|>`（"input message start/end"，输入消息的开始/结束）在词表中各自是**单个 token**。不同模型家族使用不同的模板；例如 Llama 3 家族用 `<|start_header_id|>role<|end_header_id|>` 开启一个轮次，用 `<|eot_id|>`（"end of turn"，轮次结束）关闭它。细节不同，但思想是一样的：

1. **角色标记**告诉模型是谁在说话。模型会学到：在 `<|im_start|>assistant\n` 之后，轮到*它*说话了。
2. **轮次结束 token**（`<|im_end|>`、`<|eot_id|>`）是模型说完后输出的标记。推理引擎一看到它就停止生成。基座模型从未见过这个 token 被这样使用，这是基座模型喋喋不休的原因之一。
3. **推理时**，模板渲染到助手头部（`<|im_start|>assistant\n`）为止（包含它）；这称为**生成提示（generation prompt）**。然后由模型补全助手这一轮。

!!! warning "使用模型自己的模板"
    模型只有在使用它后训练时所用的那一套确切模板时才表现良好。混用模板（例如用 Llama 的 token 来格式化 Qwen 的提示）会悄无声息地降低质量。请始终使用 tokenizer 内置的模板，而不要手写字符串。

特殊 token 必须在后训练*之前*加入词表；它们的嵌入在 SFT 期间学习。许多 tokenizer 从预训练时就已经预留了特殊 token，这样嵌入矩阵就不需要扩容。

### 关于后训练作用的两种观点 {#two-views-of-what-post-training-does}

**观点一：表面对齐假说（superficial alignment hypothesis）。**LIMA 论文（Zhou et al., 2023）仅用 1,000 条精心挑选的提示–回答对微调了一个 65B 的 LLaMA 模型，在他们的人工评估中得到了一个可与经过重得多的调优的系统相竞争的聊天模型。作者提出：*几乎所有的知识和能力都是在预训练中习得的*，对齐主要是教会模型在与用户交互时使用**哪种格式和风格**。按这种观点，后训练只是一层薄薄的外壳，*选择*基座模型中本来就有的行为。

**观点二：RL 增加了（或至少解锁了）能力。**2024–2025 年的推理模型浪潮改变了这幅图景。在数学和代码上做可验证奖励的强化学习，使模型在难题上的准确率大幅提升，并涌现出长推理链、回溯和自我检查等行为（DeepSeek-R1，见[第 2.7 章](07-rl-with-verifiable-rewards.md)）。这看起来不止是格式问题。

真相似乎介于两者之间，这场争论仍在继续：

- s1（Muennighoff et al., 2025）表明，仅用 1,000 条长推理样本做 SFT，再加一个简单的测试时技巧，就能获得很强的数学推理能力，这说明推理能力在强基座模型中很大程度上是*潜在*的。
- Yue et al.（2025）发现，如果让基座模型多次采样（较大的 pass@k），它往往能解出和 RL 训练后的模型一样多的题目，所以 RL 主要是让好的推理路径*更有可能出现*，而不是创造出新的路径。我们会在[第 2.7 章](07-rl-with-verifiable-rewards.md)中研究这一点。

一个稳妥的总结：**预训练决定了模型*能*做什么；后训练决定了模型默认*可靠地*做什么。**这两者之间的差距可能极大，这正是后训练如此重要的原因。

### 如何评估后训练 {#how-post-training-is-evaluated}

后训练之后的评估更难，因为目标（"有用"）是模糊的。人们会同时使用三类方法。

**1. 带标准答案的自动基准。**一组固定的、答案可检验的问题：

| 能力 | 基准示例 |
|---|---|
| 知识 | MMLU、GPQA（研究生水平科学选择题） |
| 数学 | GSM8K（小学应用题）、MATH、AIME（竞赛题） |
| 代码 | HumanEval、LiveCodeBench、SWE-bench（真实的 GitHub issue） |
| 指令遵循 | IFEval（可验证的约束，如"恰好用 3 个要点回答"） |

这些基准便宜且可复现，但存在**污染**（测试题泄漏进训练数据）和**饱和**（顶尖模型得分都接近 100%）的问题。

**2. LLM-as-a-judge。**由一个强模型阅读提示和一两个回答，输出分数或偏好。MT-Bench 和 AlpacaEval 就是这样工作的。Zheng et al.（2023）发现，GPT-4 作为评判者与人类专家的一致程度，大致和人类之间的一致程度相当，但也记录了一些偏差：**位置偏差**（偏好先展示的回答）、**冗长偏差**（偏好更长的回答）和**自我增强偏差**（偏好自己风格的回答）。缓解办法包括交换顺序后取平均，以及长度控制的评分。

**3. 人类偏好竞技场。**Chatbot Arena（Chiang et al., 2024）向真实用户并排展示两个匿名模型的回答，询问哪个更好。成千上万的投票通过 Bradley–Terry 模型聚合成一个评分，这和我们将在[第 2.4 章](04-rlhf.md)中用于奖励模型的模型是同一个。

## 代码 {#code}

下面的代码片段用一个真实 tokenizer 的对话模板渲染一段对话，展示生成提示，并生成一条回复。它使用一个小型的指令微调模型，因此能在笔记本 CPU 上运行（首次运行会下载约 1 GB）。

```python
# pip install torch transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"     # 小型的后训练（instruct）模型
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Australia?"},
]

# 1) 把对话渲染成文本。add_generation_prompt=True 会追加
#    "<|im_start|>assistant\n" 头部，让模型知道轮到它说话了。
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

# 2) 特殊 token 是词表中的单个条目，而不是字符序列。
print(tok.convert_ids_to_tokens(tok("<|im_start|>user")["input_ids"]))
# -> ['<|im_start|>', 'user']

# 3) 生成。遇到轮次结束 token（该模型的 eos）时停止。
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
```

用同样的 `text` 去试*基座*模型 `Qwen/Qwen2.5-0.5B`（不带 `-Instruct`）：基座模型常常会以奇怪的方式续写，因为它从未学过在 `<|im_start|>assistant` 之后该做什么。

## 实例演算 {#worked-example}

**读懂竞技场排行榜。**竞技场评分采用 Elo 刻度，评分差 $\Delta = R_A - R_B$ 对应模型 A 的预测胜率：

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{-\Delta/400}}.
$$

用话说：评分每相差 400 分，获胜的赔率就乘以 10。

- $\Delta = 0$：$P = 1/(1+1) = 0.50$，相当于抛硬币。
- $\Delta = 100$：$10^{-100/400} = 10^{-0.25} \approx 0.562$，所以 $P \approx 1/1.562 \approx 0.64$。
- $\Delta = 30$：$10^{-0.075} \approx 0.841$，所以 $P \approx 1/1.841 \approx 0.54$。

也就是说，在排行榜上看起来很大的 30 分领先，只意味着在一对一的投票中，用户偏好领先者的比例大约只有 54%。每个模型只有几千张票时，评分的置信区间通常是 ±5–10 分，所以相差几分以内的模型在统计上是打平的。这就是排行榜要公布置信区间的原因，也是你不该过度解读小评分差距的原因。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **流水线是多阶段且迭代的。**像 Tülu 3（AI2）这样的开放配方公开了完整的流水线：先在精选混合数据上做 SFT，再在偏好数据上做 DPO，然后在带可检验答案的数学和指令遵循提示上做 RLVR。前沿推理模型（DeepSeek-R1、Qwen3、Kimi）会交错进行多个 SFT 和 RL 阶段。
- **后训练算力在增长。**以推理为目标的 RL 训练如今已占前沿模型总训练算力的相当大一部分，而在 InstructGPT 时代，后训练只占很小的一部分。
- **合成数据占主导。**大部分 SFT 和偏好数据由其他模型生成或过滤，人类则专注于困难样例、质量控制和规范制定。
- **对话模板越来越丰富。**现代模板包含工具和工具结果的角色，以及隐藏推理的标记（例如若干开源推理模型中的 `<think> ... </think>` 块）。
- **评估是多种方法的组合。**各实验室报告一组带标准答案的基准（数学、代码、GPQA、IFEval、SWE-bench）、LLM 评判得分和竞技场评分，并越来越多地构建私有的留出评测以避免污染。

## 常见误区 {#common-pitfalls-misconceptions}

- **"后训练教会模型新事实。"**基本是错的。试图通过 SFT 注入知识往往会导致幻觉：模型学会的是对自己不知道的事情*听起来*很自信。知识来自预训练（或检索）。
- **"instruct 模型就是加了系统提示的基座模型。"**不对：权重是不同的。在基座模型上加系统提示，干预力度弱得多。
- **模板不匹配。**训练时用一种模板、部署时用另一种，或者忘记 `add_generation_prompt=True`，是微调项目中最常见的 bug 之一。
- **只信一个数字。**单个基准或单个 LLM 评判得分可能被长度、格式或污染所操纵。要综合看多种评估类型。

## 本章要点 {#key-takeaways}

- 基座模型补全文档；后训练让它默认以助手的方式作答。
- SFT 用示范，偏好调优用比较，RLVR 用程序化的正确性检查，蒸馏用教师模型；每种信号的成本和威力不同。
- 对话模板用表示角色和轮次结束的特殊 token，把对话序列化成单个 token 序列。
- "表面对齐"（格式是在后训练中学到的）和"RL 增加能力"都有一定道理：后训练主要改变的是模型可靠地做什么，而不是它可能做什么。
- 用带标准答案的基准、LLM-as-a-judge（当心偏差）和人类偏好竞技场（看置信区间）来评估。

## 自测 {#check-your-understanding}

??? question "为什么基座模型在回答完问题后常常停不下来？"
    它是在文档而非对话轮次上训练的。它从未学到完整回答之后应该跟一个轮次结束 token，所以会继续生成看似合理的续写（更多问题、更多文字）。SFT 教会它输出轮次结束 token，推理引擎把这个 token 当作停止信号。

??? question "什么是生成提示？如果忘了加会出什么问题？"
    它是追加在最后一条用户消息之后的助手头部（例如 `<|im_start|>assistant\n`），让模型知道现在该写助手这一轮了。没有它，模型可能会接着写用户的消息，或者开启新的用户轮次。

??? question "为什么比较比示范更便宜？又损失了什么？"
    判断两个回答哪个更好，比写出一个理想回答更快更容易，标注员也能判断自己写不出来的回答。但每次比较携带的信息很少（大约一个比特），而且它只说明哪个更好，并不说明最佳回答应该是什么。

??? question "模型 A 的竞技场评分比模型 B 高 50 分。A 应该赢得多大比例的投票？"
    $P = 1/(1+10^{-50/400}) = 1/(1+10^{-0.125}) \approx 1/(1+0.750) \approx 0.57$，约 57%。

??? question "说出 LLM-as-a-judge 评估的两种偏差，并各给出一种缓解办法。"
    位置偏差（缓解：两种顺序都评一遍再取平均）；冗长偏差（缓解：使用长度控制的评分，或指示评判者忽略长度）；自我增强偏差（缓解：使用来自不同模型家族的评判者）。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | ChatGPT 背后的 SFT → 奖励模型 → PPO 流水线；1.3B 的对齐模型被偏好于 175B 的 GPT-3。 |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | "表面对齐假说"：1,000 条精选样本就能带来很强的对话行为。 |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | 完全开放的现代 SFT → DPO → RLVR 配方，附带数据、代码和评测。 |
| :material-book-open-variant: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023 | 提出 MT-Bench，并研究 LLM 评判者的一致性与偏差。 |
| :material-book-open-variant: [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) — Chiang et al., 2024 | 成对的人类投票如何通过 Bradley–Terry 变成排行榜。 |
| :material-book-open-variant: [RLHF Book](https://rlhfbook.com/) — Nathan Lambert | 一本涵盖整个后训练技术栈的免费教材；与本部分很搭。 |
