# 2.2 监督微调

!!! abstract "本章你将学到"
    - SFT 的目标函数，以及它与预训练的区别仅在于*哪些 token 计入损失*
    - 如何用代码在单轮和多轮对话中把提示 token 从损失中掩掉
    - 序列打包（packing）如何工作，以及为什么需要特殊的注意力掩码
    - 典型超参数，以及为什么 SFT 在很少的 epoch 后就会过拟合
    - SFT 数据从哪里来（人工、Self-Instruct、蒸馏），以及为什么质量胜过数量
    - 用 PyTorch + Hugging Face Transformers 写的一个完整、最小化的 SFT 训练循环

## 为什么重要 {#why-this-matters-the-problem}

[第 2.1 章](01-why-post-training.md)说明了基座模型并不知道自己身处对话之中。最直接的补救办法是：把成千上万段使用对话模板的对话*展示*给它，训练它生成助手那一侧的内容。这就是**监督微调（supervised fine-tuning，SFT）**，也叫**指令微调（instruction tuning）**。

SFT 是几乎所有后训练流水线的第一阶段。它很便宜（小模型在几块 GPU 上几小时即可）、很简单（和预训练是同一个下一个 token 损失），而且单靠它就能得到一个像样的助手。它也是为后续 RL"冷启动"的阶段：第 [2.4](04-rlhf.md) 章和第 [2.7](07-rl-with-verifiable-rewards.md) 章中的 RL 阶段都从 SFT 模型出发，因为它们需要一个已经能给出合理、格式良好的回答的策略，才能在此基础上探索。

## 直觉 {#intuition}

把基座模型想象成一个读遍了整座图书馆、却从未坐过客服台的人。SFT 就是通过模仿来岗位培训：我们把优秀客服对话的记录交给他，并说"当你坐在助手的位置上时，就像这样回答"。

有两个细节很重要：

1. **我们只给助手的台词打分。**用户的消息只是上下文。我们不希望模型学会*写用户的问题*，所以把这些 token 排除在损失之外（**损失掩码**）。
2. **模型本来就是个好学生。**几千条高质量样本就能大幅改变它的行为。对同一小批数据反复过很多遍会让它把数据背下来，所以 SFT 的训练都很短。

## 工作原理 {#how-it-works}

### 目标函数 {#the-objective}

一个训练样本包括一个提示 $x$（用对话模板渲染的系统消息和用户消息）和一个由 $T$ 个 token 组成的目标回答 $y = (y_1, \dots, y_T)$。设 $\pi_\theta$ 为参数为 $\theta$ 的语言模型；$\pi_\theta(y_t \mid x, y_{<t})$ 是在给定提示和之前的回答 token $y_{<t} = (y_1,\dots,y_{t-1})$ 的条件下，模型赋给 token $y_t$ 的概率。

单个样本的 SFT 损失是回答的负对数似然：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}).
$$

用话说：在*给定*提示的条件下，让理想回答的每个 token 的概率尽可能大。这正是[第一部分](../part1/index.md)中的预训练交叉熵，只不过求和只覆盖回答 token。在由若干（提示，回答）对组成的数据集 $\mathcal{D}$ 上，我们最小化这个损失的平均值；最大化示范数据的似然，等价于最小化数据分布到模型的 KL 散度，所以 SFT 在统计意义上就是"模仿学习"。

### 损失掩码 {#loss-masking}

实践中，我们把提示和回答拼成一条长度为 $L$ 的序列 `input_ids`，让模型在整条序列上运行（提示 token 需要作为上下文）。然后构造一个等长的 `labels` 张量：在回答位置上等于 `input_ids`，在提示位置上取特殊值 $-100$。PyTorch 的 `cross_entropy(..., ignore_index=-100)` 会在求和以及平均的分母中都跳过这些位置。

形状：`input_ids: [batch, L]`，`labels: [batch, L]`，logits 为 `[batch, L, V]`，其中 $V$ 是词表大小。因为模型预测的是*下一个* token，所以位置 $i$ 的 logits 要与位置 $i+1$ 的标签比较（"错位一格"）。

### 多轮格式 {#multi-turn-formatting}

包含多个助手轮次的对话是一条序列。每个助手轮次（包括它的轮次结束 token，这样模型才学会停下）都参与训练；每条系统、用户和工具消息都被掩掉。也可以把前面的助手轮次一并掩掉，但对所有助手轮次都训练能从每段对话中榨取更多信号。

```text
<|im_start|>system\nYou are helpful.<|im_end|>\n     已掩码 (-100)
<|im_start|>user\nHi!<|im_end|>\n                    已掩码
<|im_start|>assistant\n                               已掩码（推理时头部由系统给出）
Hello! How can I help?<|im_end|>                      参与训练
\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n         已掩码
<|im_start|>assistant\n                               已掩码
2 + 2 = 4.<|im_end|>                                  参与训练
```

### 序列打包与注意力掩码 {#packing-and-attention-masks}

SFT 样本的长度差异极大（10 到 10,000 个 token）。如果把一个 batch 内的所有样本都 pad 到最长的那个，大部分算力都浪费在填充上。**序列打包（packing）**把若干条短样本拼接到一个固定长度（比如 4,096 个 token）的行里，几乎不浪费算力。

危险在于：如果用普通的因果掩码，样本 2 的 token 可以注意到同一行中样本 1 的 token，于是模型会从不相关的上下文中学习（"交叉污染"）。正确的做法是使用**块对角因果掩码**：token $i$ 只有在 $j \le i$ *且*两者属于同一样本时，才可以注意到 token $j$。位置 id 也应该在每个样本处从 0 重新开始，这样位置编码（例如 RoPE）才与模型在推理时看到的一致。

```text
打包后的一行:  [ a a a a | b b b | c c ]
掩码 (1 = 可以注意):
        a a a a b b b c c
   a    1
   a    1 1
   a    1 1 1
   a    1 1 1 1
   b            1
   b            1 1
   b            1 1 1
   c                  1
   c                  1 1
position_ids: 0 1 2 3 0 1 2 0 1
```

高效的注意力 kernel（FlashAttention 的"变长"接口）无需显式构造掩码，只需传入累计序列长度 `cu_seqlens = [0, 4, 7, 9]` 即可实现。有些流水线会省掉块掩码，容忍少量污染；这样更便宜，但在某些任务上效果明显更差。

### 损失的平均方式 {#loss-averaging}

一旦样本长度不同，怎么平均就变得重要了：

- **batch 内按 token 取平均**：把 batch 中所有未被掩码的 token 的损失求和，再除以这类 token 的数量。长回答贡献更多梯度。
- **先按序列取平均，再对序列取平均**：无论长度如何，每个样本的贡献相同。

两者都没有"对错"之分，但行为不同，而且这个选择会与打包以及梯度累积相互影响（见实例演算）。我们在 RL 中还会再遇到同样的问题（[第 2.7 章](07-rl-with-verifiable-rewards.md)）。

### 超参数，以及 SFT 为何很快过拟合 {#hyperparameters-and-why-sft-overfits-fast}

对 1B–70B 模型做全参数微调时，典型的起点如下：

| 设置 | 典型范围 | 原因 |
|---|---|---|
| 学习率 | 大约 $10^{-6}$ 到 $2\times10^{-5}$（AdamW） | 远低于预训练：我们想轻推权重，而不是改写它们 |
| 调度 | 线性 warmup（占总步数的百分之几），然后余弦或线性衰减 | 避免对训练良好的网络造成早期冲击 |
| Epoch 数 | 1–3 | 更多 epoch 通常会损害留出集上的质量 |
| Batch 大小 | 几十到几百条序列 | 数据集越小，batch 越小，以保证足够的步数 |
| 权重衰减 | 0 到 0.1 | 影响很小 |

LoRA（[第 2.8 章](08-distillation-and-efficient-fine-tuning.md)）通常使用比全参数微调高约 10 倍的学习率。

为什么 SFT 在几个 epoch 之后就会过拟合？基座模型有数十亿参数，而 SFT 数据集很小（通常是 $10^4$–$10^6$ 条样本）。模型可以通过背下确切的回答，而不是学习通用的风格，来降低训练损失；而且由于它本来就是极好的序列预测器，这个过程很快。过拟合的迹象有：训练损失趋近于零、验证损失上升、输出中照抄训练集里的短语。注意验证*损失*只是一个不完美的代理指标：有时在验证损失开始上升之后，人类或基准所评判的回答质量仍会略有提升，所以要用生成结果来评估，而不能只看损失。

一个更隐蔽的代价是**遗忘**：在狭窄的分布上微调，可能会削弱 SFT 数据没有涉及的能力（例如只含代码的 SFT 数据集损害了多语言能力）。主要的补救办法是混入多样化的数据。

### SFT 数据从哪里来 {#where-sft-data-comes-from}

1. **人工撰写的示范。**标注员为真实或虚构的提示写出理想回答（InstructGPT 就是这样做的）。质量高，但慢且贵。
2. **模板化的学术数据集。**FLAN（Wei et al., 2021）把许多现有的 NLP 数据集（翻译、问答、NLI……）改写成自然语言指令，并表明在众多任务上做指令微调能提升在*未见*任务上的零样本表现。Chung et al.（2022）把规模扩大到约 1.8K 个任务，并表明加入思维链样本很重要。
3. **Self-Instruct（合成）。**Wang et al.（2022）从 175 个人工编写的种子任务出发，提示模型自己生成新的指令、输入和输出，并过滤掉低质量和近似重复的样本。这种自举（bootstrapping）思路是当今合成数据流水线的鼻祖。
4. **从更强的模型蒸馏。**用一个强模型为大量提示生成回答，再用它们对较小的模型做 SFT。2024–2026 年的大多数开源模型都大量使用这种方法，包括来自推理模型的推理轨迹（见[第 2.8 章](08-distillation-and-efficient-fine-tuning.md)）。使用更强模型的输出之前，请先查看其许可协议。

### 质量胜过数量 {#quality-over-quantity}

LIMA（Zhou et al., 2023）仅用 1,000 条精心挑选的样本微调了一个 65B 模型，就得到了一个很强的聊天模型。经受住了检验的教训是：**一小批多样、正确、格式良好的样本，胜过一大批嘈杂的样本。**实用的过滤手段包括去重、剔除答案错误的回答（由验证器或强模型检查）、平衡任务类型，以及拒绝那些不该拒答或不该含糊其辞时却这样做的回答。由于 SFT 会模仿数据中的*一切*，包括错误和口头禅，数据中的单个系统性缺陷（例如每个回答都以"当然可以！"开头）就会变成模型的缺陷。

## 代码 {#code}

### 1. 为一段对话构造 input ids 和带掩码的 labels {#1-building-input-ids-and-masked-labels-for-a-conversation}

我们逐段手工渲染 ChatML，这样就能确切知道哪些 token 属于助手内容。（如果用真实的 tokenizer，而且模板支持，可以使用 `apply_chat_template(..., return_assistant_tokens_mask=True)`；手工实现则能展示其中到底发生了什么。）

```python
import torch

IGNORE = -100

def build_example(messages, tok, max_len=2048):
    """messages: list of {"role", "content"}. Returns input_ids and labels (1-D LongTensors).
    Only assistant content + its end-of-turn marker is trained on."""
    # 中文说明：messages 为消息列表；返回 input_ids 与 labels（一维 LongTensor）。
    # 只有助手内容及其轮次结束标记参与训练。
    input_ids, labels = [], []
    for m in messages:
        header = f"<|im_start|>{m['role']}\n"
        body = m["content"] + "<|im_end|>"
        h_ids = tok(header, add_special_tokens=False)["input_ids"]
        b_ids = tok(body, add_special_tokens=False)["input_ids"]
        nl_ids = tok("\n", add_special_tokens=False)["input_ids"]
        input_ids += h_ids + b_ids + nl_ids
        if m["role"] == "assistant":
            labels += [IGNORE] * len(h_ids) + b_ids + [IGNORE] * len(nl_ids)  # 训练回复 + <|im_end|>
        else:
            labels += [IGNORE] * (len(h_ids) + len(b_ids) + len(nl_ids))      # 掩掉 system/user/tool
    return torch.tensor(input_ids[:max_len]), torch.tensor(labels[:max_len])

def sft_loss(logits, labels):
    """logits: [B, L, V], labels: [B, L] with IGNORE on masked positions.
    Position i predicts token i+1, so shift by one."""
    # 中文说明：被掩码的位置标签为 IGNORE；位置 i 预测 token i+1，所以要错位一格。
    logits = logits[:, :-1, :].contiguous()          # [B, L-1, V]
    targets = labels[:, 1:].contiguous()             # [B, L-1]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=IGNORE)                          # 只对未掩码的 token 取平均
```

### 2. 使用块对角因果掩码的打包 {#2-packing-with-a-block-diagonal-causal-mask}

```python
def pack(examples, seq_len):
    """examples: list of (input_ids, labels). Greedily fills one row of length seq_len.
    Returns input_ids [seq_len], labels [seq_len], position_ids [seq_len], mask [seq_len, seq_len]."""
    # 中文说明：贪心地填满长度为 seq_len 的一行。
    ids, labs, pos, doc = [], [], [], []
    for d, (x, y) in enumerate(examples):
        if len(ids) + len(x) > seq_len:
            break
        ids += x.tolist(); labs += y.tolist()
        pos += list(range(len(x)))                   # 每个文档的位置从 0 重新开始
        doc += [d] * len(x)                          # 每个 token 所属的文档 id
    pad = seq_len - len(ids)
    ids += [0] * pad; labs += [IGNORE] * pad; pos += [0] * pad; doc += [-1] * pad
    doc = torch.tensor(doc)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_doc = (doc[:, None] == doc[None, :]) & (doc[:, None] >= 0)
    mask = causal & same_doc                         # True = 可以注意
    return torch.tensor(ids), torch.tensor(labs), torch.tensor(pos), mask
```

对于某些注意力实现，Hugging Face 模型接受形状为 `[batch, 1, L, L]` 的 4 维布尔或加性掩码；在生产环境中，你应该把 `cu_seqlens` 传给变长 FlashAttention kernel，而不是构造 $L \times L$ 的矩阵。

### 3. 一个完整的最小 SFT 循环 {#3-a-complete-minimal-sft-loop}

下面在一个玩具数据集上训练一个小模型，使用 padding（为简单起见不打包）。换成真实数据集，它就是一个可用的 SFT 训练器。

```python
# pip install torch transformers
import torch, math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

model_id = "Qwen/Qwen2.5-0.5B"                       # 一个基座模型；我们教它聊天
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).train()

data = [  # 实际中：成千上万条精选对话
    [{"role": "user", "content": "What is the capital of Australia?"},
     {"role": "assistant", "content": "The capital of Australia is Canberra."}],
    [{"role": "user", "content": "Give me a synonym for 'happy'."},
     {"role": "assistant", "content": "Joyful."}],
] * 64

examples = [build_example(m, tok) for m in data]

def collate(batch):
    L = max(len(x) for x, _ in batch)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(batch), L), pad_id); labs = torch.full((len(batch), L), IGNORE)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = x; labs[i, :len(y)] = y; att[i, :len(x)] = 1
    return ids, labs, att

loader = DataLoader(examples, batch_size=8, shuffle=True, collate_fn=collate)
epochs = 2
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
total = epochs * len(loader)
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(0.03 * total), num_training_steps=total)

for epoch in range(epochs):
    for step, (ids, labs, att) in enumerate(loader):
        ids, labs, att = ids.to(device), labs.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()   # [B, L, V]
        loss = sft_loss(logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)            # 稳定训练
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 4 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.3f}")
```

说明：我们传入 `attention_mask`，这样注意力会忽略填充位置；填充的标签位置是 `IGNORE`，所以损失也会忽略它们。在真实训练中，你还应该留出一个验证集，定期评估生成结果，并保存每个 epoch 的检查点。

## 实例演算 {#worked-example}

**用数字看掩码为什么重要。**一个样本有 300 个 token 的系统+用户提示和 100 个 token 的回答。

- *不*做掩码时，损失在 400 个位置上取平均；其中 300 个（75%）是在训练模型预测*用户*说的话。助手行为的梯度信号被稀释了 4 倍，而且模型会学着生成像用户一样的文本。
- *做*掩码时，只有 100 个回答 token（加上轮次结束 token）参与贡献。

**用数字看平均方式为什么重要。**一个 batch 中有两个样本。样本 A 有 10 个回答 token，平均每 token 损失为 2.0；样本 B 有 90 个回答 token，平均每 token 损失为 1.0。

- 按 token 取平均（对展平后的 batch 调用 `cross_entropy(..., ignore_index)` 就是这样做的）：$(10 \cdot 2.0 + 90 \cdot 1.0)/100 = 1.10$。样本 B 占 90% 的权重。
- 按序列取平均：$(2.0 + 1.0)/2 = 1.50$。每个样本占 50%。

现在把这个 batch 拆成两个微批次做**梯度累积**（每个微批次一个样本），并天真地对两个微批次的损失取平均：你会得到 $(2.0 + 1.0)/2 = 1.50$，也就是按序列加权，尽管单个大 batch 本来会给出 1.10。这种悄无声息的不一致曾是流行训练库中的一个真实 bug；修复办法是把各微批次的 token 损失求和，再除以整个 batch 中未被掩码的 token *总数*。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **SFT 数据集的规模。**开放配方使用的数据量从几千条（LIMA、s1）到几百万条（Tülu 3 的 SFT 混合数据约为一百万条量级），大量偏向来自强模型的合成数据，并经过质量过滤。
- **推理 SFT。**对于推理模型，在长思维链轨迹（每条数千个 token）上做 SFT 是 RL 之前的"冷启动"；16K–32K+ token 的序列长度使打包和节省显存的注意力实现成为必需。
- **库。**TRL 的 `SFTTrainer`、Axolotl、LLaMA-Factory 和 torchtune 都实现了掩码（"只在补全部分训练"）、带正确位置 id 的打包，以及 LoRA。请阅读它们的文档，看看它们默认采用哪种掩码和平均方式。
- **把对话模板放在 tokenizer 里。**把模板和 tokenizer 存在一起（`tokenizer.chat_template`），这样训练和部署就能保持一致。

## 常见误区 {#common-pitfalls-misconceptions}

- **忘记掩掉提示**（或者掩掉了轮次结束 token，导致模型永远学不会停下）。
- **标签错位时差一。**如果从第 0 步起损失就低得可疑，检查一下是不是在预测同一位置的 token。
- **打包时没有重置位置 id，或没有使用块掩码**，导致跨样本的注意力。
- **epoch 太多。**训练损失接近零是警告信号，而不是成功。
- **"SFT 可以教会新知识。"**在模型不知道的事实上微调，往往会增加幻觉：它学会的是用那种风格给出自信的回答，却没有背后的知识。
- **SFT 与推理时的模板不一致**，或训练数据中缺少轮次结束 token。

## 本章要点 {#key-takeaways}

- SFT 是限定在回答 token 上的下一个 token 交叉熵：$-\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$。
- 用 `-100` 掩掉提示和用户 token；对每个助手轮次（包括其轮次结束 token）都做训练。
- 打包能节省算力，但需要按文档重置的位置 id 和块对角因果掩码。
- 使用较小的学习率和 1–3 个 epoch；SFT 很快过拟合，因为模型大而数据小。
- 数据质量和多样性比规模更重要；合成数据和蒸馏数据如今占主导。

## 自测 {#check-your-understanding}

??? question "既然在损失中把提示 token 掩掉了，为什么还要把它们保留在输入中？"
    模型必须以它们为条件：回答的概率 $\pi_\theta(y_t \mid x, y_{<t})$ 依赖于提示 $x$。掩码只是把它们从*目标*中移除，而没有从上下文中移除。

??? question "在包含两个文档的打包行中，如果使用普通的因果掩码，会出什么问题？"
    第二个文档的 token 可以注意到第一个文档，于是模型学会了利用不相关的上下文；而且位置是接着第一个文档继续而不是重新开始，与推理时的条件不一致。

??? question "你的 SFT 训练损失在 6 个 epoch 后是 0.05，模型逐字重复训练集中的回答。发生了什么？你要改什么？"
    它过拟合/背下了数据。使用更少的 epoch（1–3），可能还要降低学习率、使用更多样的数据，并根据留出集上的生成质量来选择检查点。

??? question "Self-Instruct 的核心思想是什么？"
    从一小批人工编写的种子任务出发，提示语言模型生成新的指令和回答，过滤掉低质量和近似重复的，把剩下的加入池中，然后重复这一过程，用很少的人力自举出一个大型指令数据集。

??? question "一个 batch 中有一条 5 个 token 和一条 95 个 token 的回答，平均 token 损失分别为 3.0 和 1.0。按 token 平均的损失是多少？"
    $(5 \cdot 3.0 + 95 \cdot 1.0)/100 = 110/100 = 1.10$。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — Wei et al., 2021 | 在众多任务上做指令微调，能提升在未见任务上的零样本表现。 |
| :material-book-open-variant: [Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416) — Chung et al., 2022 | 扩大任务数量，并在指令微调中加入思维链数据。 |
| :material-book-open-variant: [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al., 2022 | 用模型自身自举指令数据；合成数据流水线的鼻祖。 |
| :material-star: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023 | 1,000 条精选样本就足以带来很强的对话行为；质量 ≫ 数量。 |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | 完全开放的 SFT → DPO → RLVR 配方；请读其中的 SFT 数据混合部分。 |
