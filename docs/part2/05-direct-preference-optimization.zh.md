# 2.5 直接偏好优化（DPO）

!!! abstract "本章你将学到"
    - 推导 KL 正则化 RLHF 目标的闭式最优策略
    - 这个闭式解如何把奖励模型变成"隐式奖励"，并得到 DPO 损失
    - 如何用几行 PyTorch 实现 DPO 损失，以及 $\beta$ 控制什么
    - 主要变体（IPO、KTO、ORPO、SimPO）及其区别
    - 离线与在线/迭代式 DPO，以及何时选 DPO 而不是 RL

## 为什么重要 {#why-this-matters-the-problem}

RLHF-PPO（[第 2.4 章](04-rlhf.md)）能用，但很重：要单独训练奖励模型和价值模型，训练循环里还要在线生成，并且有许多敏感的超参数。Rafailov 等人（2023）提出一个问题：既然 RLHF 的目标是一个已知的数学表达式，能不能先用解析方法解出其中一部分，从而绕开强化学习？

答案就是**直接偏好优化（Direct Preference Optimization，DPO）**：它直接在偏好对 $(x, y_w, y_l)$ 上训练策略，损失形式很像逻辑回归。它只需要策略模型和一个冻结的参考模型，训练过程中不需要采样，跑起来几乎和 SFT 一样简单。2023–2024 年间它成为开源模型偏好调优的默认方法，至今仍被广泛使用。

## 直觉 {#intuition}

RLHF 的目标可以这样说："最大化奖励，但不要偏离参考模型太远。"对于任意奖励函数，这个目标下的最优策略形式很简单：取参考模型的概率，乘以 $\exp(\text{reward}/\beta)$，再重新归一化。好的回答被放大，差的回答被压低。

反过来想：如果策略是最优的，那么它相对参考模型的**对数概率比** $\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ *就是*奖励（差一个缩放系数和一个只依赖 prompt 的常数）。所以我们不必先训练奖励模型再训练策略，而是直接用*策略自身*来参数化奖励，并让它去拟合偏好数据。这就是 DPO 论文副标题所说的"你的语言模型暗地里就是一个奖励模型"。

## 工作原理 {#how-it-works}

### 第 1 步：KL 正则化目标 {#step-1-the-kl-regularized-objective}

记号：$x$ 是 prompt，$y$ 是回答，$\pi$ 是策略（回答上的分布），$\pi_{\text{ref}}$ 是冻结的参考（SFT）模型，$r(x,y)$ 是奖励，$\beta > 0$ 是 KL 强度。对单个 prompt，RLHF 目标（[第 2.3 章](03-rl-foundations-for-llms.md)）为

$$
\max_{\pi}\; \mathbb{E}_{y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

### 第 2 步：闭式最优策略 {#step-2-the-closed-form-optimal-policy}

把 KL 写成期望，并提出 $-\beta$：

$$
\mathbb{E}_{y\sim\pi}\Big[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]
= -\beta\, \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\exp\!\big(r(x,y)/\beta\big)}\Big].
$$

分母 $\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ 不是一个概率分布，所以需要归一化。定义**配分函数**

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\exp\!\big(r(x,y)/\beta\big)
$$

以及分布

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big).
$$

把 $\pi_{\text{ref}}\exp(r/\beta) = Z(x)\,\pi^*$ 代入目标：

$$
-\beta\, \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi^*(y\mid x)} - \log Z(x)\Big]
= -\beta\, \mathrm{KL}\big(\pi\,\|\,\pi^*\big) + \beta\log Z(x).
$$

$Z(x)$ 与 $\pi$ 无关，而 KL 散度 $\ge 0$，仅当两个分布相等时取等号。所以目标恰好在 $\pi = \pi^*$ 处取得最大值。

用大白话说：**最优策略就是用指数化的奖励对参考模型重新加权**，$\beta$ 起着温度的作用。$\beta$ 小，奖励占主导，策略集中到最好的回答上；$\beta$ 大，策略停留在 $\pi_{\text{ref}}$ 附近。

我们没法直接用这个公式来*采样*（计算 $Z(x)$ 需要对所有可能的回答求和），但可以在代数上利用它。

### 第 3 步：隐式奖励 {#step-3-the-implicit-reward}

对最优策略公式取对数，并解出 $r$：

$$
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
$$

用大白话说：任何奖励函数都可以用它自己的最优策略来表示，再加上一个只依赖 prompt 的项 $\beta\log Z(x)$。因此，如果我们在策略 $\pi_\theta$ 的空间里搜索，每个策略都定义了一个**隐式奖励**

$$
\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)},
$$

而该策略正是这个隐式奖励对应的最优策略（至多差一个与 prompt 相关的常数）。

### 第 4 步：代入 Bradley–Terry，$Z$ 被消掉 {#step-4-plug-into-bradleyterry-z-cancels}

Bradley–Terry 模型（[第 2.4 章](04-rlhf.md)）只用到同一 prompt 下两个回答的奖励*差*。$\beta\log Z(x)$ 这一项对 $y_w$ 和 $y_l$ 完全相同，所以被消掉：

$$
P(y_w\succ y_l\mid x) = \sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

在偏好数据集 $\mathcal{D}$ 上做极大似然，就得到 **DPO 损失**：

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]\;}
$$

用大白话说：相对参考模型，提高被选中（chosen）回答的对数概率，降低被拒绝（rejected）回答的对数概率，中间经过一个 sigmoid，一旦隐式奖励的间隔足够大它就会饱和。这其实就是 RLHF 里奖励模型的损失，只不过把奖励模型换成了策略自己的对数比。训练它，等于同时拟合了一个奖励模型*并*得到了它的最优策略。

这里 $\log\pi_\theta(y\mid x) = \sum_t \log\pi_\theta(y_t\mid x, y_{<t})$ 是回答所有 token 的对数概率之和，只需一次前向传播即可算出，掩码方式与 SFT 相同（[第 2.2 章](02-supervised-fine-tuning.md)）。

### 梯度：一种加权的 SFT/不似然（unlikelihood）更新 {#the-gradient-a-weighted-sftunlikelihood-update}

记 $\hat r_w = \hat r_\theta(x,y_w)$，$\hat r_l = \hat r_\theta(x,y_l)$。求导得到：

$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{权重}}\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].
$$

用大白话说：提高 $y_w$ 的似然，降低 $y_l$ 的似然，权重取决于当前隐式奖励*错得有多厉害*。已经以较大间隔排对的偏好对，权重接近 0。这个加权很重要：DPO 论文报告说，如果没有它，朴素的"提高 chosen、降低 rejected"目标可能会退化。

### $\beta$ 的作用 {#what-beta-does}

$\beta$ 同时扮演两个相互关联的角色：

1. **原始目标中的 KL 强度**：$\beta$ 越大，隐含的最优策略越贴近 $\pi_{\text{ref}}$。
2. **隐式奖励的尺度**：sigmoid 看到的是 $\beta\times$（对数比间隔）。$\beta$ 大时，对数比的一点小变化就足以让 sigmoid 饱和，训练很早就不再推动策略。$\beta$ 小时，对数比要增长很多损失才会饱和，策略因此会离参考模型更远。

常见取值在 0.01–0.5 之间，默认值常用 0.1。值得做一次扫描。

### 变体 {#variants}

DPO 衍生出许多变体，每个变体改动一个假设。

| 方法 | 数据 | 是否需要参考模型 | 关键改动 | 直觉 |
|---|---|---|---|---|
| **DPO** | 偏好对 | 是 | — | 对对数比间隔使用逻辑损失 |
| **IPO**（Azar et al., 2023） | 偏好对 | 是 | 用平方损失把对数比间隔拉向固定目标 $\tfrac{1}{2\tau}$，而不是用 sigmoid | 当偏好接近确定性时，DPO 可能把间隔推向无穷大而无视 KL；IPO 的有限目标让它保持被正则化 |
| **KTO**（Ethayarajh et al., 2024） | 非成对的点赞/点踩标签 | 是 | 受前景理论启发的逐样本损失，把每个回答的隐式奖励与一个参考点（KL 的估计）比较 | 可以使用比偏好对容易收集得多的二元反馈；对收益和损失的处理是不对称的 |
| **ORPO**（Hong et al., 2024） | 偏好对 | **否** | 在 $y_w$ 上的 SFT 损失，加上 $y_w$ 与 $y_l$ 之间基于长度归一化似然的对数几率比惩罚 | 一个阶段代替"先 SFT 再 DPO"；无需在显存中保留参考模型 |
| **SimPO**（Meng et al., 2024） | 偏好对 | **否** | 隐式奖励 = 长度归一化的对数似然 $\frac{\beta}{\lvert y\rvert}\log\pi_\theta(y\mid x)$，再加一个目标间隔 $\gamma$ | 让奖励与驱动生成的平均对数似然一致；消除长度偏置，也去掉了参考模型 |

SimPO 的损失，具体写出来是：

$$
\mathcal{L}_{\text{SimPO}} = -\log\sigma\Big(\frac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w\mid x) - \frac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l\mid x) - \gamma\Big),
$$

其中 $\lvert y\rvert$ 是 $y$ 的 token 数，$\gamma > 0$ 是间隔。

### 离线与在线/迭代式 DPO {#offline-vs-online-iterative-dpo}

标准 DPO 是**离线**的：偏好对在训练前就固定了，而且常常是由*其他*模型生成的。这会造成分布不匹配：策略学的是给它自己永远不会生成的回答排序，而一旦策略发生变化，它又得不到关于自己*实际*生成内容的反馈。

**在线或迭代式 DPO** 把这个环闭合起来：

1. 用*当前*策略为每个 prompt 采样若干回答。
2. 用奖励模型或 LLM 裁判给回答打标签，组成偏好对（例如最好 vs 最差）。
3. 在这些新鲜的偏好对上跑 DPO；把新的参考模型设为当前策略（或保留原来的）。
4. 重复。

这让 DPO 的行为更接近 RL。2024–2025 年的证据表明，当采样成本可以接受时，**在线、同策略**方法（PPO、GRPO、在线/迭代式 DPO）通常优于纯离线 DPO。

### 何时用 DPO，何时用 RL {#when-to-use-dpo-vs-rl}

| 适合用 DPO（或其变体）的情况 | 适合用 RL（PPO/GRPO）的情况 |
|---|---|
| 有偏好数据，但没有程序化验证器 | 有验证器（数学、代码、测试）——见[第 2.7 章](07-rl-with-verifiable-rewards.md) |
| 算力有限，想要类似 SFT 的简单与稳定 | 能承担在线采样，并想要最好的最终质量 |
| 目标是风格、语气、有用性、安全校准 | 目标是需要探索才能获得的能力（长推理、智能体任务） |
| 需要在 SFT 之后快速做一轮偏好调优 | 需要多轮交互或与环境交互 |

## 代码 {#code}

```python
import torch, torch.nn.functional as F

def sequence_logprob(model, input_ids, attention_mask, labels):
    """回答 token 上 log pi(token) 之和。labels 在 prompt/pad 位置为 -100（与 SFT 相同）。
    input_ids, attention_mask, labels: [B, L] -> 返回 ([B] 对数概率之和, [B] token 数)。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = (targets != -100)
    logp = torch.log_softmax(logits, -1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1), mask.sum(-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """所有输入: [B] 的对数概率之和。返回损失以及用于记录日志的隐式奖励。"""
    r_w = beta * (pi_w - ref_w)                        # chosen 的隐式奖励
    r_l = beta * (pi_l - ref_l)                        # rejected 的隐式奖励
    loss = -F.logsigmoid(r_w - r_l).mean()
    return loss, r_w.detach(), r_l.detach()

def ipo_loss(pi_w, pi_l, ref_w, ref_l, tau=0.1):
    h = (pi_w - ref_w) - (pi_l - ref_l)                # 对数比间隔
    return ((h - 1.0 / (2 * tau)) ** 2).mean()

def simpo_loss(pi_w, pi_l, len_w, len_l, beta=2.0, gamma=1.0):
    r_w = beta * pi_w / len_w                          # 长度归一化，无参考模型
    r_l = beta * pi_l / len_l
    return -F.logsigmoid(r_w - r_l - gamma).mean()

def dpo_step(policy, ref, batch, opt, beta=0.1):
    """batch: 包含 chosen_/rejected_ 的 input_ids、attention_mask、labels 的字典（每项 [B, L]）。"""
    pi_w, _ = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
    pi_l, _ = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"])
    with torch.no_grad():                              # 参考模型是冻结的
        ref_w, _ = sequence_logprob(ref, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
        ref_l, _ = sequence_logprob(ref, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"])
    loss, r_w, r_l = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
    opt.zero_grad(); loss.backward(); opt.step()
    return {"loss": loss.item(), "reward_acc": (r_w > r_l).float().mean().item(),
            "margin": (r_w - r_l).mean().item(), "chosen_logp": pi_w.mean().item()}
```

用[第 2.2 章](02-supervised-fine-tuning.md)的 `build_example` 构造批次：chosen 和 rejected 共享同一个 prompt，只在助手回复部分不同。由于参考模型是冻结的，一个常见技巧是对整个数据集一次性预先算好 `ref_w` 和 `ref_l`，这样训练时 GPU 显存里只需要放策略模型。

## 实例演算 {#worked-example}

**最优策略与 $\beta$。** 某个 prompt 有两个可能的回答，$\pi_{\text{ref}} = (0.5, 0.5)$，奖励 $r = (1, 0)$。

- $\beta = 0.5$：$\pi^* \propto (0.5e^{2}, 0.5e^{0}) = (3.69, 0.5)$，所以 $\pi^*(y_1) = 3.69/4.19 \approx 0.88$。
- $\beta = 2$：$\pi^* \propto (0.5e^{0.5}, 0.5) = (0.824, 0.5)$，所以 $\pi^*(y_1) \approx 0.824/1.324 \approx 0.62$。

$\beta$ 越大，最优策略越接近参考模型的 50/50。

**一次 DPO 损失的计算。** 对某个偏好对，对数概率之和为 $\log\pi_\theta(y_w) = -12$，$\log\pi_{\text{ref}}(y_w) = -13$，$\log\pi_\theta(y_l) = -15$，$\log\pi_{\text{ref}}(y_l) = -14$，$\beta = 0.1$。

- 隐式奖励：$\hat r_w = 0.1\times(-12+13) = 0.1$；$\hat r_l = 0.1\times(-15+14) = -0.1$。
- 间隔 $0.2$；$\sigma(0.2) \approx 0.550$；损失 $= -\log 0.550 \approx 0.598$。
- 初始化时（$\pi_\theta = \pi_{\text{ref}}$），所有间隔都是 0，损失为 $\log 2 \approx 0.693$：这是检查你实现是否正确的一个快速办法。
- 梯度权重 $\sigma(\hat r_l - \hat r_w) = \sigma(-0.2) \approx 0.450$。
- 若 $\beta = 1$，同样的对数概率给出间隔 2，损失 $-\log\sigma(2) \approx 0.127$：损失几乎饱和，训练对这个偏好对的推动会小得多。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **DPO 系列方法仍然是偏好调优的主力**，在开源配方中尤其如此（Zephyr 在 UltraFeedback 上用 AI 裁判打出的偏好做 DPO；Tülu 3 在 SFT 与 RLVR 之间加入一个 DPO 阶段）。之后在可验证领域再用 RL。
- **数据比变体重要。** 许多对比发现，在 DPO/IPO/SimPO/KTO 之间如何选择，其影响不如偏好对的质量和同策略程度。从正在训练的模型本身采样、再由强裁判打标的偏好对效果最好。
- **留意 chosen 的对数概率。** DPO 一个众所周知的现象是：训练中 chosen 和 rejected 的对数概率*都会*下降，而间隔却在增大；如果 chosen 的对数概率下降过多，输出质量可能退化。混入 chosen 上的 SFT 损失（ORPO 在设计上就是这样）或使用更大的 $\beta$ 都有帮助。
- **长度利用。** DPO 的对数概率之和可能偏向更长的回答；长度归一化的变体（SimPO）或控制长度的数据可以解决这个问题。

## 常见误区 {#common-pitfalls-misconceptions}

- **"DPO 没有奖励模型。"** 它有一个*隐式*奖励模型 $\beta\log(\pi_\theta/\pi_{\text{ref}})$；你可以像对待奖励模型那样记录它的准确率和间隔。
- **"DPO 在实践中等价于 RLHF。"** 它*只*在数据的支撑集上优化同一个目标；离线时，它从未见过也无法纠正策略自己新产生的输出。
- **用错参考模型。** 参考模型必须是策略初始化所用的那个模型（通常是 SFT 模型）；换成别的模型会改变隐含的优化目标。
- **忘记掩码。** 对 prompt token 的对数概率求和（chosen 和 rejected 完全相同）既增加噪声又浪费算力；请把它们掩掉。
- **训练轮数过多。** DPO 很快就会过拟合偏好数据（训练集上的奖励准确率 → 100%）；通常 1–3 个 epoch 即可。

## 本章要点 {#key-takeaways}

- KL 正则化的 RLHF 目标有闭式最优解 $\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$。
- 把它反过来写，得到隐式奖励 $\beta\log(\pi_\theta/\pi_{\text{ref}})$；在 Bradley–Terry 中，难以计算的 $Z(x)$ 被消掉。
- DPO 损失：$-\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}]\big)$——每个偏好对每个模型做两次前向传播，无需采样。
- $\beta$ 既控制 KL 强度，也控制损失饱和的快慢。
- 变体分别改变数据格式（KTO）、损失形状（IPO），或去掉参考模型（ORPO、SimPO）；在线/迭代式 DPO 能找回 RL 的大部分优势。

## 自测 {#check-your-understanding}

??? question "为什么配分函数 $Z(x)$ 会从 DPO 损失中消失？"
    隐式奖励包含 $\beta\log Z(x)$，它只依赖 prompt。Bradley–Terry 用的是同一 prompt 下两个回答的奖励之差，所以这一项被消掉了。

??? question "训练刚开始时 DPO 损失是多少，为什么？"
    $\log 2 \approx 0.693$。策略初始化为参考模型，因此所有对数比都是 0，间隔为 0，$-\log\sigma(0) = \log 2$。

??? question "如果你只有针对单个回答的点赞/点踩反馈，应该选哪个 DPO 变体？"
    KTO，它就是为非成对的二元标签设计的。

??? question "为什么在线/迭代式 DPO 可能优于离线 DPO？"
    离线偏好对来自一个固定的（通常是别的模型的）分布，所以策略得不到关于自己当前输出的反馈。从当前策略采样并给这些样本打标签，能让训练保持同策略，使方法像 RL 一样纠正自己的错误。

??? question "两次训练分别用 $\beta=0.05$ 和 $\beta=0.5$，哪个策略会偏离参考模型更远？"
    $\beta = 0.05$：KL 惩罚更弱，而且 sigmoid 要到对数比间隔大得多时才会饱和。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 | 本章的推导出处，另有实验和梯度分析。 |
| :material-book-open-variant: [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036) — Azar et al., 2023 | 统一了 RLHF 与 DPO，并提出 IPO 以避免 DPO 在确定性偏好上的过拟合。 |
| :material-book-open-variant: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024 | 用非成对的点赞/点踩标签做偏好调优。 |
| :material-book-open-variant: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024 | 把 SFT 与偏好损失合并为一个阶段，无需参考模型。 |
| :material-book-open-variant: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024 | 长度归一化、无参考模型、带间隔的奖励；效果强且成本低。 |
| :material-book-open-variant: [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) — Tunstall et al., 2023 | 一份实用配方：先 SFT，再在 AI 裁判打分的 UltraFeedback 偏好上做 DPO。 |
