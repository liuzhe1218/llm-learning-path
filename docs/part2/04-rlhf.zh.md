# 2.4 基于人类反馈的强化学习（RLHF）

!!! abstract "本章你将学到"
    - 人类偏好数据是如何收集的，以及为什么使用比较而不是打分
    - Bradley–Terry 模型的推导，以及如何用它训练奖励模型（含代码）
    - 完整的 RLHF-PPO 循环及其涉及的四个模型，以及它的显存开销
    - reward hacking（奖励投机）和奖励模型的过度优化，以及 KL 惩罚如何起作用
    - RLAIF 与 Constitutional AI：用 AI 反馈取代人类标注

## 为什么重要 {#why-this-matters-the-problem}

经过 SFT（[第 2.2 章](02-supervised-fine-tuning.md)）之后，模型能以正确的格式回答，但它的回答只能和示范一样好，而且它并不知道在自己所有可能的回答中，人们*更喜欢哪一个*。像"有用但简洁"、"对不确定性保持诚实"、"礼貌地拒绝有害请求"这样的品质，很难穷尽地示范，却很容易识别。

RLHF 把这种识别能力变成训练信号。它是让 InstructGPT 和 ChatGPT 与 GPT-3 在体感上有质的不同的关键要素。它把[第 2.3 章](03-rl-foundations-for-llms.md)中的 RL 工具与一个代替人类评判者的**学习得到的奖励模型**结合在一起。

## 直觉 {#intuition}

我们不可能让人类给 RL 期间生成的数百万条回答逐一打分。所以我们：

1. 让人类比较适量的回答对（几万到几十万对）。
2. 训练一个**奖励模型**（reward model，RM）来预测这些人类判断：它读入提示和回答，输出一个数，越高表示越受偏好。
3. 针对奖励模型运行 RL，它可以即时为无限多的回答打分。
4. 用 KL 惩罚给策略套上缰绳，因为奖励模型只是人类判断的一个*代理*，在不熟悉的回答上会出错。

```text
 提示 ─► SFT 模型采样 2 个以上回答 ─► 人类选出更好的一个
                                                   │
                                                   ▼
                                  奖励模型 r_φ(x, y)  (Bradley–Terry)
                                                   │
 提示 ─► 策略 π_θ 生成回答 ─► r_φ 打分 ─► PPO 更新（对 π_ref 加 KL）
```

## 工作原理 {#how-it-works}

### 收集偏好数据 {#collecting-preference-data}

对每个提示 $x$，SFT 模型（或多个模型）生成两个或更多回答。标注员看到它们后指出哪个更好，有时还带有强度（"好很多"、"稍微好一点"）或者平局。每个判断变成一个三元组 $(x, y_w, y_l)$：**被选中**（"赢家"）的回答 $y_w$ 和**被拒绝**（"输家"）的回答 $y_l$。InstructGPT 让标注员对每个提示的 4 到 9 个回答排序；$K$ 个回答的排序可以产生 $\binom{K}{2}$ 个两两比较。

为什么用比较而不用绝对分数（例如"打 1–10 分"）？不同标注员使用刻度的方式不同（一个人的 7 分是另一个人的 5 分），同一个人也会随时间漂移。相对判断要一致得多。标注员会拿到指导准则（例如有用性、真实性、无害性），而标注员之间的一致性远非完美，这为任何奖励模型所能达到的准确率设定了上限。

### Bradley–Terry 模型的推导 {#the-bradleyterry-model-derived}

我们需要一个概率模型，把每个回答的标量"质量"与人类偏好它的机会联系起来。假设提示 $x$ 的每个回答 $y$ 都有一个潜在强度 $\exp(r(x,y))$，其中 $r$ 是一个实数（即**奖励**），并假设 $y_w$ 击败 $y_l$ 的概率与其强度成正比：

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}.
$$

分子分母同时除以 $\exp(r(x,y_w))$：

$$
P(y_w \succ y_l \mid x) = \frac{1}{1 + \exp\big(-(r(x,y_w) - r(x,y_l))\big)} = \sigma\big(r(x,y_w) - r(x,y_l)\big),
$$

其中 $\sigma(z) = 1/(1+e^{-z})$ 是 logistic sigmoid 函数。用话说：偏好概率只取决于奖励的**差值**，再经过一个 sigmoid。奖励差为 0 意味着五五开；差为 2 则意味着 $\sigma(2) \approx 0.88$。

两个推论：

- 奖励只在**每个提示内相差一个加性常数**的意义下有定义：给同一提示的所有回答都加上 $c(x)$，每个概率都不变。因此 RM 的输出通常在 RL 之前要先做归一化。
- 这和 Elo 评分以及 Chatbot Arena（[第 2.1 章](01-why-post-training.md)）背后的是同一个模型，只是刻度不同。

### 训练奖励模型 {#training-the-reward-model}

奖励模型 $r_\phi$ 通常是把 SFT 模型的词表输出层替换成一个线性头，把最后的隐藏状态映射到一个标量；奖励在回答的**最后一个 token** 处读取。形状：隐藏状态 `[B, L, d_model]` → 头 → `[B, L, 1]` → 取最后一个非填充位置的值 → `[B]`。

我们用 Bradley–Terry 下观测到的偏好的最大似然来拟合 $\phi$，也就是在偏好数据集 $\mathcal{D}$ 上最小化负对数似然：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big].
$$

这就是作用在奖励差上的二元 logistic 回归。令 $\Delta = r_\phi(x,y_w) - r_\phi(x,y_l)$，它的梯度为

$$
\nabla_\phi \mathcal{L}_{\text{RM}} = -\big(1 - \sigma(\Delta)\big)\big(\nabla_\phi r_\phi(x,y_w) - \nabla_\phi r_\phi(x,y_l)\big),
$$

所以模型已经以很大的间隔正确排序的样本对（$\sigma(\Delta)\approx1$）几乎没有贡献，而排错的样本对贡献最大。奖励模型通常只训练一个 epoch，因为它们很快就会过拟合。它们的质量用留出比较集上的**成对准确率**来衡量；由于人类之间也会意见不一，准确率远低于 100%。

### RLHF-PPO 循环 {#the-rlhf-ppo-loop}

奖励模型冻结之后，我们优化[第 2.3 章](03-rl-foundations-for-llms.md)中带 KL 正则的目标：

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

这里 $\pi_\theta$ 是正在训练的策略（由 SFT 初始化），$\pi_{\text{ref}}$ 是冻结的 SFT 模型，$r_\phi$ 是冻结的奖励模型，$\beta$ 是 KL 系数。PPO 需要**四个模型**：

| 模型 | 作用 | 是否训练？ | 初始化自 |
|---|---|---|---|
| 策略（actor）$\pi_\theta$ | 生成回答；被优化 | 是 | SFT 模型 |
| 参考模型 $\pi_{\text{ref}}$ | 为 KL 惩罚计算 $\log\pi_{\text{ref}}$ | 否（冻结） | SFT 模型 |
| 奖励模型 $r_\phi$ | 为完整回答打分 | 否（冻结） | SFT 模型 + 标量头，在偏好数据上训练 |
| 价值模型（critic）$V_\psi$ | 预测每个 token 的期望未来奖励，用于 GAE | 是 | 奖励模型或 SFT 模型 |

一次迭代：

1. **Rollout。**采样一批提示；用 $\pi_\theta$ 为每个提示生成一条回答（这是推理：缓慢的自回归过程）。
2. **打分。**为每条完整回答计算 $r_\phi(x,y)$；通过前向传播计算逐 token 的 $\log\pi_\theta$（作为 $\log\pi_{\text{old}}$）、$\log\pi_{\text{ref}}$ 和 $V_\psi(s_t)$。
3. **构造奖励。**逐 token：$r_t = -\beta(\log\pi_{\text{old}}(y_t\mid s_t) - \log\pi_{\text{ref}}(y_t\mid s_t))$，并在最后一个 token 处加上 $r_\phi(x,y)$。
4. **优势。**对每条回答运行 GAE，得到 $\hat A_t$ 和回报 $\hat R_t$；把优势在整个 batch 上归一化（"白化"）为均值 0、标准差 1。
5. **更新。**在小批次上做若干个 epoch：为 $\pi_\theta$ 最小化 PPO 裁剪策略损失，为 $V_\psi$ 最小化价值损失。
6. 用新的 rollout 重复上述过程。

InstructGPT 还额外混入了预训练的语言建模损失（"PPO-ptx"），以减少在标准 NLP 基准上的退化，即所谓的**对齐税（alignment tax）**。

### 显存方面的影响 {#memory-implications}

四个大模型必须同时驻留，其中两个还带有优化器状态。使用混合精度 Adam 时，一个被训练的模型每个参数大约需要 16 字节（bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + Adam 动量 8），而冻结的推理模型用 bf16 时每个参数需要 2 字节。生成时的激活值和 KV cache 还要另算。这就是为什么 RLHF-PPO 要用模型分片（ZeRO/FSDP）来实现，有时让奖励模型和价值模型比策略模型小（InstructGPT 为其 175B 的策略使用了 6B 的奖励模型），也是无 critic 方法（GRPO，[第 2.7 章](07-rl-with-verifiable-rewards.md)）和无奖励模型方法（DPO，[第 2.5 章](05-direct-preference-optimization.md)）变得流行的原因。

### 奖励投机与过度优化 {#reward-hacking-and-over-optimization}

奖励模型只是一个代理。**古德哈特定律（Goodhart's law）**：当一个指标变成目标时，它就不再是一个好指标。RL 是极其强大的优化器，所以它会发现并利用代理的缺陷：

- **长度**：RM 往往略微偏好更长的回答；策略就学会了注水。
- **重风格轻实质**：自信的语气、项目符号列表、奉承用户（"谄媚，sycophancy"）。
- **对抗性乱码**：在极端情况下，会出现 RM 打高分的怪异 token 序列。

Gao et al.（2022）清晰地测量了这一现象。他们用一个大的"金标准"奖励模型替代人类，在它的标签上训练较小的代理 RM，再针对这些代理优化策略。随着策略离初始化越来越远（以 $d = \sqrt{\mathrm{KL}(\pi\|\pi_{\text{init}})}$ 衡量），代理奖励不断上升，而金标准奖励先上升后**下降**。他们拟合了简单的函数形式：对 best-of-$n$ 采样是 $R_{\text{gold}}(d) = d(\alpha - \beta d)$，对 RL 是 $R_{\text{gold}}(d) = d(\alpha - \beta\log d)$，并发现更大的奖励模型过度优化得更慢。

```text
 奖励
   │        代理 RM 得分  ───────────────/
   │                              ____/
   │                      ___/‾‾‾‾  ‾‾‾‾‾‾\___  真实（金标准）质量
   │                ___/                       ‾‾‾\___
   │          ___/
   │    ___/
   └───────────────────────────────────────────────────► 与初始策略的 KL
              ↑ 最佳点：在此停止（或设定 β 使其停留在此）
```

防御手段：KL 惩罚（$\beta$ 越大 = 离得越近）、依据留出的人类或强评判者评估来提前停止、更大更好的 RM、RM 集成、长度惩罚或长度控制的 RM 训练，以及定期针对*当前*策略的输出收集新的偏好（迭代式 RLHF），使 RM 在策略所在的区域保持准确。

### RLAIF 与 Constitutional AI {#rlaif-and-constitutional-ai}

人类标注又慢又贵，而对于有害内容，生产标注本身也令人不适。**RLAIF**（RL from AI feedback，来自 AI 反馈的强化学习）用一个强 LLM 当标注者：给它看提示和两个回答，让它判断哪个更好，其偏好用来训练奖励模型。Lee et al.（2023）发现，RLAIF 在摘要和有用对话等任务上取得了与 RLHF 相当的结果。

**Constitutional AI**（Bai et al., Anthropic, 2022）围绕一份书面的原则清单，即**宪法（constitution）**（例如"选择危害最小的回答"）来组织 AI 反馈。它分为两个阶段：

1. **监督阶段（批评与修订）。**模型针对一个潜在有害的提示生成回答，然后被要求依据某条原则批评它并重写。对修订后的回答做 SFT。
2. **RL 阶段（RLAIF）。**模型依据随机抽取的原则比较回答对；这些 AI 偏好训练出一个无害性奖励模型，用于 RL（并与人类的有用性偏好结合）。

结果是得到一个更无害、却没有变得同样回避的模型，而且它的价值观是由一份明确、可检视的文档规定的，而不是隐含在成千上万条单独的标注之中。

## 代码 {#code}

### 1. 奖励模型及其 Bradley–Terry 损失 {#1-a-reward-model-and-its-bradleyterry-loss}

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Transformer backbone + scalar head; reward read at the last non-pad token."""
    # 中文说明：Transformer 主干 + 标量头；在最后一个非填充 token 处读取奖励。
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, dtype=torch.float32)  # 不含 LM 头
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, d]
        L = attention_mask.shape[1]                              # 最后一个真实 token 的下标；
        last = L - 1 - attention_mask.flip(1).argmax(dim=1)     # 对左填充、右填充或混合填充都适用
        h_last = h[torch.arange(h.size(0)), last]                # [B, d]
        return self.head(h_last.to(self.head.weight.dtype)).squeeze(-1)  # [B]

def bt_loss(r_chosen, r_rejected):
    """-log sigma(r_w - r_l), averaged. logsigmoid is numerically stable."""
    # 中文说明：-log sigma(r_w - r_l) 的均值；logsigmoid 数值上是稳定的。
    return -F.logsigmoid(r_chosen - r_rejected).mean()

name = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "right"
rm = RewardModel(name)
opt = torch.optim.AdamW(rm.parameters(), lr=1e-5)

pairs = [("Q: What is 2+2?\nA:", " 4.", " 5."),
         ("Q: Capital of France?\nA:", " Paris.", " I don't know, maybe Lyon.")]
for epoch in range(3):
    x, yw, yl = zip(*pairs)
    chosen = tok([a + b for a, b in zip(x, yw)], return_tensors="pt", padding=True)
    rejected = tok([a + b for a, b in zip(x, yl)], return_tensors="pt", padding=True)
    r_w = rm(chosen.input_ids, chosen.attention_mask)
    r_l = rm(rejected.input_ids, rejected.attention_mask)
    loss = bt_loss(r_w, r_l)
    acc = (r_w > r_l).float().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"loss {loss.item():.3f}  pairwise acc {acc.item():.2f}")
```

### 2. 一次 RLHF-PPO 迭代（数据流） {#2-one-rlhf-ppo-iteration-the-data-flow}

这个函数准确展示了四个模型各自计算什么。它复用了[第 2.3 章](03-rl-foundations-for-llms.md)中的 `gae`、`ppo_policy_loss` 和 `value_loss`；假设 `value_model(ids, mask)` 返回逐 token 的价值 `[B, L]`（一个主干加上每个位置上的标量头）。

```python
import torch

def token_logprobs(model, ids, mask):
    """log pi(token_t | prefix) for t = 1..L-1. ids, mask: [B, L] -> [B, L-1]."""
    # 中文说明：对 t = 1..L-1 计算 log pi(token_t | 前缀)。
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

def rlhf_ppo_iteration(policy, ref, reward_model, value_model, tok, prompts,
                       opt_pi, opt_v, beta=0.05, ppo_epochs=2, max_new=64):
    tok.padding_side = "left"                                    # 生成时对提示做左填充
    enc = tok(prompts, return_tensors="pt", padding=True)
    P = enc.input_ids.shape[1]
    with torch.no_grad():
        ids = policy.generate(**enc, max_new_tokens=max_new, do_sample=True, top_p=1.0,
                              pad_token_id=tok.pad_token_id)     # [B, P+T]
        mask = (ids != tok.pad_token_id).long(); mask[:, :P] = enc.attention_mask  # 假设 pad id 不等于 eos id
        resp = torch.zeros_like(ids[:, 1:], dtype=torch.float)   # 回答 token（目标）处为 1
        resp[:, P - 1:] = mask[:, P:].float()
        logp_old = token_logprobs(policy, ids, mask)             # [B, L-1]
        logp_ref = token_logprobs(ref, ids, mask)
        values = value_model(ids, mask)[:, :-1]                  # V(s_t)，与目标对齐
        score = reward_model(ids, mask)                          # [B]，在最后一个真实 token 处读取
        advs, rets = torch.zeros_like(resp), torch.zeros_like(resp)
        for i in range(ids.shape[0]):                            # 逐条回答构造奖励 + GAE
            idx = resp[i].nonzero().squeeze(-1)
            r = -beta * (logp_old[i, idx] - logp_ref[i, idx])
            r[-1] += score[i]
            a, ret = gae(r, values[i, idx])
            advs[i, idx], rets[i, idx] = a, ret
        sel = resp.bool()
        advs[sel] = (advs[sel] - advs[sel].mean()) / (advs[sel].std() + 1e-8)  # 白化
    for _ in range(ppo_epochs):                                  # 复用同一批 rollout
        logp_new = token_logprobs(policy, ids, mask)
        loss_pi = ppo_policy_loss(logp_new, logp_old, advs, resp)
        loss_v = value_loss(value_model(ids, mask)[:, :-1], rets, resp)
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
        opt_v.zero_grad(); loss_v.backward(); opt_v.step()
    return score.mean().item()
```

真实的实现（TRL、OpenRLHF、verl）还会加入小批次划分、梯度累积、自适应 $\beta$ 控制器、在轮次结束 token 处截断，以及用快速推理引擎做分布式生成，但数据流是一样的。

## 实例演算 {#worked-example}

**Bradley–Terry 的数字。**RM 给出 $r(x, y_w) = 1.2$ 和 $r(x, y_l) = 0.4$。

- $\Delta = 0.8$，$P(y_w\succ y_l) = \sigma(0.8) = 1/(1+e^{-0.8}) = 1/(1+0.449) \approx 0.690$。
- 损失：$-\log 0.690 \approx 0.371$。
- 梯度权重：$1 - \sigma(\Delta) \approx 0.310$。如果 RM 把这一对排反了（$\Delta = -0.8$），$\sigma = 0.310$，损失将是 $-\log 0.310 \approx 1.171$，梯度权重为 $0.690$：推动力度超过两倍。

**7B 模型做 RLHF-PPO 的显存**（四个模型都是 7B，bf16，混合精度 Adam，忽略激活值和 KV cache）：

| 模型 | 字节 / 参数 | 显存 |
|---|---|---|
| 策略（训练） | 16 | $7\text{B}\times16 = 112$ GB |
| 价值（训练） | 16 | 112 GB |
| 参考（冻结） | 2 | 14 GB |
| 奖励（冻结） | 2 | 14 GB |
| **合计** | | **≈ 252 GB** |

对比同一模型的 SFT：约 112 GB。RLHF-PPO 在计入生成开销之前就需要两倍以上的显存，这就是为什么哪怕是 7B 模型也需要一个多 GPU 节点，也是去掉价值模型（GRPO）或同时去掉奖励和价值模型（DPO）颇具吸引力的原因。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **偏好数据大多是合成的或混合的。**许多开放流水线（例如 UltraFeedback 风格的数据、Tülu 3）用众多模型生成回答，再由强 LLM 来评判；人类则专注于困难或安全攸关的样例。
- **奖励模型仍然是核心**，用于没有验证器的开放式品质（有用性、风格、安全）。它们越来越多地是*生成式*的：一个在给出分数之前先写出批评的 LLM，有时还以评分细则为条件。
- **混合奖励。**现代 RL 阶段把可验证奖励（数学、代码；[第 2.7 章](07-rl-with-verifiable-rewards.md)）与用于通用聊天的 RM 奖励结合起来，常常分成不同阶段进行。
- **PPO 与其他方案。**一些实验室仍在使用带 critic 的 PPO，但许多开放配方已经转向用 DPO 做偏好调优，用无 critic 的 GRPO 类方法做 RL。
- **反复出现在各类报告中的实用技巧：**对奖励做归一化（按提示或按 batch），对优势做白化，每一步都监控 KL 和回答长度，保留一个由人类或强模型评判的留出集，用 RM 初始化 critic，并为策略使用相当小的学习率（比 SFT 小）。

## 常见误区 {#common-pitfalls-misconceptions}

- **把 RM 分数当作绝对值。**它们只有在同一提示内作差时才有意义；跨提示比较要格外小心。
- **从错误的 token 读取奖励。**左填充/右填充混淆时，"最后一个 token"可能是填充，此时 RM 什么有用的东西也学不到。
- **为奖励上升而庆祝。**代理奖励上升的同时长度或 KL 也在上升，往往就是在投机。务必检查样本，并做独立评估。
- **"RLHF 教会事实。"**它主要塑造行为；甚至可能降低校准度（模型变得更自信），并且如果标注员喜欢被附和，还会增加谄媚。
- **忽视标注员之间的分歧。**嘈杂的偏好集会限制 RM 的准确率，而 RM 的错误会变成策略的目标。

## 本章要点 {#key-takeaways}

- 人类比较回答；比较比绝对评分更一致。
- Bradley–Terry：$P(y_w \succ y_l) = \sigma(r_w - r_l)$；RM 用 $-\log\sigma(r_w - r_l)$ 训练。
- RLHF-PPO 使用四个模型（策略、参考、奖励、价值），最大化 RM 奖励减去与 SFT 模型的 $\beta\,\mathrm{KL}$，非常耗显存。
- 把代理优化得太狠会降低真实质量（过度优化）；KL 惩罚、更好的 RM 和提前停止可以缓解。
- RLAIF 和 Constitutional AI 用在书面原则指导下的 AI 判断取代人类标注。

## 自测 {#check-your-understanding}

??? question "为什么奖励模型的输出只在每个提示内相差一个常数的意义下有定义？"
    Bradley–Terry 概率只取决于同一提示的各回答之间的奖励差；给所有回答都加上 $c(x)$，每个 $\sigma(r_w - r_l)$ 都不变，所以数据无法确定这个常数。

??? question "PPO 的四个模型中，哪些通过梯度更新，哪些只做前向运行？"
    策略模型和价值模型被训练；参考模型和奖励模型被冻结，只做前向运行。

??? question "RLHF 期间 RM 分数稳步上升，但人类评分员说输出变得更差、更长了。发生了什么？你会改什么？"
    奖励投机/过度优化：策略在利用 RM 的偏差，比如对长度的偏好。可以增大 $\beta$、更早停止（依据独立评估选择检查点）、加入长度惩罚或使用长度控制的 RM、在当前策略的输出上重新训练 RM，或者使用 RM 集成。

??? question "Constitutional AI 的两个阶段是什么？"
    一个监督阶段：模型依据宪法原则批评并修订自己的回答，并在修订结果上微调；以及一个 RL 阶段：由原则指导的 AI 偏好标签训练出一个奖励模型，用于 RL（RLAIF）。

??? question "如果 $r_w - r_l = 0$，损失和梯度权重分别是多少？"
    $\sigma(0) = 0.5$，损失 $= -\log 0.5 = \ln 2 \approx 0.693$，梯度权重 $1 - 0.5 = 0.5$。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-book-open-variant: [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al., 2017 | RLHF 的起源：从成对比较中学习奖励模型，再用 RL 优化它。 |
| :material-book-open-variant: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020 | 摘要任务上的 RLHF；大规模的奖励模型 + KL 惩罚配方。 |
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | ChatGPT 背后的 SFT → RM → PPO 流水线，含数据收集细节。 |
| :material-book-open-variant: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao et al., 2022 | 量化古德哈特定律：策略优化代理时，金标准奖励先升后降。 |
| :material-star: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (Anthropic), 2022 | 批评与修订式 SFT，加上由书面宪法指导的 RLAIF。 |
| :material-book-open-variant: [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) — Lee et al., 2023 | AI 标注的偏好在多个任务上可以匹敌人类标注。 |
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | RLHF 内部使用的 RL 算法（在[第 2.3 章](03-rl-foundations-for-llms.md)中推导）。 |
