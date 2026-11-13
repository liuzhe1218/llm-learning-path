# 2.3 大模型的强化学习基础

!!! abstract "本章你将学到"
    - 如何把文本生成看作马尔可夫决策过程（MDP），奖励在末尾给出
    - 序列版本的策略梯度定理，逐步推导，以及 REINFORCE 算法
    - 为什么基线能在不引入偏差的情况下降低方差，以及优势函数是什么
    - 重要性采样，以及让我们能够复用样本的概率比
    - 逐项解读 PPO 的裁剪目标、价值函数、GAE，以及对参考模型的 KL 惩罚

## 为什么重要 {#why-this-matters-the-problem}

SFT（[第 2.2 章](02-supervised-fine-tuning.md)）只能模仿。它需要有人写出理想的回答。但在很多目标上，我们给回答**打分**比写出回答容易得多：人可以判断两个回答哪个更好，程序可以检查数学答案是否正确、代码能否通过测试。我们想要一种算法：它接收对整条生成回答的*分数*，并调整模型，使其产生得分更高的回答。

这就是强化学习（RL）。本部分余下的每个算法（[2.4](04-rlhf.md) 中的 RLHF + PPO、[2.5](05-direct-preference-optimization.md) 中的 DPO、[2.7](07-rl-with-verifiable-rewards.md) 中的 GRPO）都是由本章的这些零件搭成的。我们从零开始、专门针对语言模型来构建它们，这比通用 RL 要简单。

## 直觉 {#intuition}

想象用"冷热游戏"来教学。模型写出一个回答；你只说"好"（+1）或"不好"（0）。模型应该怎么改？

- 如果回答好，就让其中的*每一个 token 选择*都稍微更有可能出现。
- 如果回答不好，就让它们稍微更不可能出现。

这基本上就是全部想法（REINFORCE）。本章余下的内容是在修补它的问题：

1. **噪声。**一个运气好的回答会把它所有的 token 都推高，包括不好的那些。解决办法：与一个期望得分（**基线，baseline**）比较，这样只有*好于预期*的回答才会被强化。
2. **样本效率。**生成回答很昂贵；我们希望对每批生成的回答做多步梯度更新。解决办法：使用带**裁剪**的比值的**重要性采样**（PPO），这样我们就不会偏离产生样本的那个策略太远。
3. **漂移。**把分数优化得太狠，会让模型去利用打分器的缺陷，并忘记如何写正常的文本。解决办法：**KL 惩罚**，让模型保持在参考模型附近。

## 工作原理 {#how-it-works}

### 把生成看作 MDP {#generation-as-an-mdp}

马尔可夫决策过程包含状态、动作、转移和奖励。对于自回归 LLM：

| MDP 概念 | 在 LLM 中的含义 |
|---|---|
| 状态 $s_t$ | 提示加上到目前为止已生成的 token：$s_t = (x, y_1, \dots, y_{t-1})$ |
| 动作 $a_t$ | 下一个 token $y_t$，从大小为 $V$ 的词表中选取 |
| 策略 $\pi_\theta(a_t \mid s_t)$ | LLM 的下一个 token 分布（logits 的 softmax） |
| 转移 | 确定性的：$s_{t+1}$ 就是在 $s_t$ 后面追加 $y_t$ |
| 回合（episode） | 一条完整的回答 $y = (y_1,\dots,y_T)$，在轮次结束 token 或长度上限处结束 |
| 奖励 | 通常是在末尾给出的单个数 $R(x, y)$：当 $t < T$ 时 $r_t = 0$，且 $r_T = R(x,y)$ |

整条回答的概率按 token 分解：

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t}), \qquad
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid s_t).
$$

用话说：一条回答的对数概率等于其各 token 对数概率之和，模型一次前向传播就能算出来。

我们的目标是最大化从提示集 $\mathcal{D}$ 中抽取的提示 $x$ 上的期望奖励：

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\; \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big].
$$

困难在于：$y$ 是*采样*出来的（离散选择），所以我们无法穿过采样这一步做反向传播。奖励函数也可能是一个不可微的程序。

### 序列的策略梯度定理 {#the-policy-gradient-theorem-for-sequences}

固定一个提示 $x$，记 $p(y) = \pi_\theta(y \mid x)$。期望是对所有可能回答的一个（巨大的）求和：

$$
J_x(\theta) = \sum_{y} p_\theta(y)\, R(x,y).
$$

求梯度。$R$ 不依赖于 $\theta$，所以只需对 $p_\theta$ 求导：

$$
\nabla_\theta J_x = \sum_y \nabla_\theta p_\theta(y)\, R(x,y).
$$

现在使用**对数导数技巧**：$\nabla_\theta p = p \,\nabla_\theta \log p$（因为 $\nabla \log p = \nabla p / p$）。代入得：

$$
\nabla_\theta J_x = \sum_y p_\theta(y)\, \nabla_\theta \log p_\theta(y)\, R(x,y)
= \mathbb{E}_{y \sim \pi_\theta}\big[ R(x,y)\, \nabla_\theta \log \pi_\theta(y \mid x) \big].
$$

最后把序列对数概率展开成 token 的形式：

$$
\boxed{\;\nabla_\theta J = \mathbb{E}_{x,\, y \sim \pi_\theta}\Big[ R(x,y) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid s_t) \Big]\;}
$$

用话说：期望奖励的梯度，是对采样得到的回答求平均的"奖励乘以回答中每个 token 对数概率的梯度"。它是一个期望，所以我们可以通过从当前模型采样回答来估计它；我们从不对采样或 $R$ 求导。

如果奖励在中间步骤出现，一个改进是：token $t$ 只应该被记入在它*之后*出现的奖励（即"reward-to-go"，$\sum_{t' \ge t} r_{t'}$），因为一个动作不可能影响过去的奖励。在只有单个终端奖励的情况下，每个 token 的 reward-to-go 都是 $R$，这就是为什么上面框起来的公式本身已经是 reward-to-go 的形式。

### REINFORCE {#reinforce}

最简单的算法把这个公式变成一个损失，其梯度就是该估计量。对于一个包含 $N$ 条采样回答的 batch：

$$
\mathcal{L}_{\text{REINFORCE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} R(x_i, y_i) \sum_{t=1}^{T_i} \log \pi_\theta(y_{i,t} \mid s_{i,t}).
$$

与 SFT 比较：它是在模型*自己的*样本上做的**奖励加权**负对数似然。奖励为 0 的回答没有任何贡献；奖励为 1 的回答则像 SFT 样本一样被训练。（当奖励只有 0 或 1 时，REINFORCE 字面上就是"在你自己成功的样本上做 SFT"，这种方法叫拒绝采样微调（rejection-sampling fine-tuning）或专家迭代（expert iteration），[第 2.6 章](06-reasoning-and-test-time-compute.md)中的 STaR 就用到了它。）

### 方差与基线 {#variance-and-baselines}

REINFORCE 估计量是无偏的，但噪声很大。如果所有奖励都是正的（比如奖励在 5 到 6 之间），每条采样的回答都会被推高，只有各次推动之间的*细微*差别携带信息。

解决办法是减去一个不依赖于所采样回答的**基线** $b$：

$$
\nabla_\theta J = \mathbb{E}_{y}\Big[ \big(R(x,y) - b(x)\big) \nabla_\theta \log \pi_\theta(y \mid x) \Big].
$$

为什么这样仍然是正确的（无偏的）？因为多出来的那一项期望为零：

$$
\mathbb{E}_{y}\big[ b(x)\, \nabla_\theta \log \pi_\theta(y\mid x) \big]
= b(x) \sum_y \pi_\theta(y\mid x)\, \frac{\nabla_\theta \pi_\theta(y\mid x)}{\pi_\theta(y\mid x)}
= b(x)\, \nabla_\theta \underbrace{\sum_y \pi_\theta(y\mid x)}_{=1} = 0.
$$

用话说：概率之和永远是 1，所以对数概率梯度的平均值为零，任何常数乘以它的平均值也为零。一个好的基线是该提示的期望奖励，$b(x) \approx \mathbb{E}_{y}[R(x,y)]$；这样好于平均水平的回答被推高，差于平均水平的被压低。

### 优势函数 {#advantage}

把基线推广到每个 token 上，就得到**优势函数（advantage）**。定义：

- **价值函数** $V(s_t)$：如果从状态 $s_t$ 出发，继续用当前策略采样，所得最终奖励的期望。
- **动作价值** $Q(s_t, a_t)$：如果在状态 $s_t$ 选择 token $a_t$，然后继续按策略采样，所得最终奖励的期望。
- **优势** $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$：在该状态下选择 $a_t$ 比策略的平均选择好多少。

策略梯度可以用优势写成：

$$
\nabla_\theta J = \mathbb{E}\Big[ \sum_t A(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big].
$$

不同算法的区别主要在于**如何估计 $A$**：PPO 学习一个价值网络并使用 GAE（见下文）；GRPO（[第 2.7 章](07-rl-with-verifiable-rewards.md)）则用同一提示的一组样本的 $R - \text{均值}$，让一条回答的每个 token 都获得相同的优势。

### 重要性采样与概率比 {#importance-sampling-and-the-probability-ratio}

生成回答（rollout）是最昂贵的部分。我们希望用策略 $\pi_{\theta_{\text{old}}}$ 采样一批数据，然后做多步梯度更新，在此期间 $\theta$ 会逐渐偏离 $\theta_{\text{old}}$。这时样本就是**离策略（off-policy）**的：它们来自一个与我们正在改进的策略略有不同的策略。

重要性采样（importance sampling）可以修正这一点。对任意函数 $f$：

$$
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \sum_a \pi_\theta(a) f(a) = \sum_a \pi_{\text{old}}(a) \frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} f(a) = \mathbb{E}_{a \sim \pi_{\text{old}}}\Big[\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} f(a)\Big].
$$

逐 token 地应用，定义**概率比**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\big(\log \pi_\theta(a_t\mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t\mid s_t)\big),
$$

以及**代理目标（surrogate objective）** $L(\theta) = \mathbb{E}_{\pi_{\text{old}}}[\rho_t(\theta) A_t]$。当 $\theta = \theta_{\text{old}}$ 时，$\rho_t = 1$ 且 $\nabla \rho_t = \nabla \log \pi_\theta$，所以代理目标的梯度就等于策略梯度。（严格地说，逐 token 的比值只修正了动作分布，而没有修正状态的分布；当 $\theta$ 与 $\theta_{\text{old}}$ 接近时，这种近似没有问题，而这恰恰是裁剪所要保证的。）

问题在于：如果 $\rho_t$ 变得很大，估计的方差会爆炸，少数几个 token 会主导更新。我们需要让 $\pi_\theta$ 保持在 $\pi_{\text{old}}$ 附近。

### PPO：裁剪目标 {#ppo-the-clipped-objective}

近端策略优化（Proximal Policy Optimization，Schulman et al., 2017）对每个 token 最大化

$$
L^{\text{CLIP}}_t(\theta) = \min\Big( \rho_t(\theta)\, A_t,\;\; \operatorname{clip}\big(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\, A_t \Big),
$$

其中 $\epsilon$（通常为 0.1–0.3，常取 0.2）设定信任域，$\operatorname{clip}(z, lo, hi)$ 把 $z$ 限制在 $[lo, hi]$ 内。逐项来看：

- $\rho_t A_t$ 是经重要性采样修正的策略梯度代理目标。
- $\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t$ 与之相同，只是比值不能离开 $[1-\epsilon, 1+\epsilon]$，所以一旦 $\rho_t$ 超出这个范围，它对 $\theta$ 的梯度就是零。
- $\min$ 选取两者中更**悲观**的那个。

各种情形下的效果如下：

| 情形 | 效果 |
|---|---|
| $A_t > 0$（好 token），$\rho_t > 1+\epsilon$ | 裁剪项更小 → 被选中 → 梯度为零。*一个好 token 的概率增长超过 $\epsilon$ 之后就不再继续提高。* |
| $A_t > 0$，$\rho_t < 1+\epsilon$ | 未裁剪 → 正常梯度把 $\rho_t$ 推高。 |
| $A_t < 0$（坏 token），$\rho_t < 1-\epsilon$ | 选中裁剪项 → 梯度为零。*缩小超过 $\epsilon$ 之后就不再继续降低。* |
| $A_t < 0$，$\rho_t > 1-\epsilon$ | 未裁剪 → 正常梯度把 $\rho_t$ 压低，即使 $\rho_t$ 高于 $1+\epsilon$ 也如此（撤销一次错误的变动永远不会被阻止）。 |

所以 PPO 让每次更新把一个 token 的概率*朝优势所希望的方向*最多改变约 $1\pm\epsilon$ 倍，然后就停止推动。这使得对同一批 rollout 做多个 epoch 的小批量更新变得安全。我们最小化的损失是 $-L^{\text{CLIP}}$，对 token 取平均。

### 价值函数与 GAE {#the-value-function-and-gae}

PPO 用一个学习得到的**价值网络**（"critic"，评论家）$V_\phi(s_t)$ 来估计优势，它通常是 LLM 的一个副本，带一个标量头，在每个 token 位置输出一个数。引入折扣因子 $\gamma \in [0,1]$（对 LLM 通常取 $\gamma = 1$），定义**时序差分（TD）误差**

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
$$

它表示走一步之后的实际结果比 critic 预测的好多少（回合结束后 $V_\phi(s_{T+1}) = 0$）。**广义优势估计（Generalized Advantage Estimation，GAE，Schulman et al., 2015）**用参数 $\lambda \in [0,1]$ 对这一量的多步版本取加权平均：

$$
\hat A_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \text{computed backwards as } \hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}.
$$

用话说：$\lambda = 0$ 只用一步 TD 误差（方差低，但如果 critic 不准则有偏）；$\lambda = 1$ 时逐项相消为 $\hat A_t = \sum_{t'\ge t}\gamma^{t'-t} r_{t'} - V_\phi(s_t)$，即实际回报减去基线（无偏，但方差更高）。LLM 的 RLHF 设置常用接近 1 的 $\lambda$。critic 通过向**回报** $\hat R_t = \hat A_t + V_\phi(s_t)$ 回归来训练：

$$
\mathcal{L}_V(\phi) = \tfrac{1}{2}\big(V_\phi(s_t) - \hat R_t\big)^2.
$$

### 对参考模型的 KL 惩罚 {#the-kl-penalty-to-a-reference-model}

在 LLM 的强化学习中，我们几乎总是保留一份起始模型的冻结副本，即**参考策略（参考模型）** $\pi_{\text{ref}}$（通常是 SFT 模型），并惩罚与它的偏离。带 KL 正则的目标是

$$
\max_\theta\; \mathbb{E}_{x,\, y\sim\pi_\theta}\big[R(x,y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big),
$$

其中 $\beta > 0$ 设定强度，并且

$$
\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\Big[\sum_t \log \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}\Big].
$$

为什么要这样做？

1. **奖励并不完美。**学习得到的奖励模型只在它训练时见过的那些回答附近才准确。没有这根缰绳，RL 会找到得分很高但实际很糟的怪异输出（奖励投机，[第 2.4 章](04-rlhf.md)）。
2. **保持语言质量与多样性。**参考模型写出的文本流畅且多样；与它保持接近就能保住这些。
3. **稳定性。**它抑制策略坍缩到少数几个高奖励字符串上。

两种常见实现：

- **把 KL 放进奖励**（经典 RLHF）：使用逐 token 的奖励 $r_t = -\beta\big(\log\pi_\theta(y_t\mid s_t) - \log\pi_{\text{ref}}(y_t\mid s_t)\big)$，并在最后一个 token 处加上 $R(x,y)$，然后照常用 GAE 计算优势。
- **把 KL 放进损失**（GRPO 风格）：把 $\beta \cdot \widehat{\mathrm{KL}}_t$ 直接加到逐 token 损失中，通常使用方差低、始终非负的估计量 $\widehat{\mathrm{KL}}_t = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$（在采样到的 token 上求值）。

2025 年许多采用可验证奖励的推理 RL 配方把 $\beta$ 设得很小或为零，因为基于规则的奖励比学习得到的奖励难被投机得多（[第 2.7 章](07-rl-with-verifiable-rewards.md)）。

## 代码 {#code}

### 1. 在玩具"单 token 回答"问题上，带基线的 REINFORCE {#1-reinforce-with-a-baseline-on-a-toy-one-token-response-problem}

```python
import torch

torch.manual_seed(0)
V = 3                                   # 词表 {A, B, C}；只有 token A（id 0）获得奖励 1
logits = torch.zeros(V, requires_grad=True)   # 整个"策略"就是 3 个 logits
opt = torch.optim.SGD([logits], lr=0.5)
reward = lambda a: (a == 0).float()

for step in range(50):
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, num_samples=16, replacement=True)  # 16 条采样的"回答"
    R = reward(actions)                                                     # [16]
    baseline = R.mean()                                                     # batch 均值基线
    logp = torch.log_softmax(logits, dim=-1)[actions]                       # [16]
    loss = -((R - baseline).detach() * logp).mean()                         # REINFORCE 损失
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(step, [round(p, 3) for p in probs.tolist()])
# P(A) 从 0.333 向 1.0 攀升
```

### 2. GAE 与 PPO 的各项损失 {#2-gae-and-the-ppo-losses}

```python
import torch

def gae(rewards, values, gamma=1.0, lam=0.95):
    """rewards, values: [T] for one response. values[t] = V(s_t).
    Returns advantages [T] and returns [T]."""
    # 中文说明：rewards、values 为单条回答的 [T] 张量；返回优势和回报。
    T = rewards.shape[0]
    adv = torch.zeros(T)
    next_value, running = 0.0, 0.0                 # 回合结束后 V(s_{T+1}) = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values

def ppo_policy_loss(logp_new, logp_old, adv, mask, eps=0.2):
    """All tensors [B, T]; mask is 1 on response tokens. Returns a scalar to MINIMIZE."""
    # 中文说明：所有张量形状为 [B, T]；mask 在回答 token 上为 1；返回需要最小化的标量。
    ratio = torch.exp(logp_new - logp_old)                     # rho_t
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    per_token = -torch.min(unclipped, clipped)                 # L^CLIP 取负
    return (per_token * mask).sum() / mask.sum()

def value_loss(v_new, returns, mask):
    return 0.5 * (((v_new - returns) ** 2) * mask).sum() / mask.sum()

def kl_shaped_rewards(final_reward, logp_policy, logp_ref, beta=0.05):
    """Per-token rewards for one response [T]: -beta*KL at each token, task reward at the end."""
    # 中文说明：单条回答的逐 token 奖励：每个 token 处为 -beta*KL，末尾再加上任务奖励。
    r = -beta * (logp_policy - logp_ref)
    r[-1] = r[-1] + final_reward
    return r

# 小测试
r = kl_shaped_rewards(1.0, torch.tensor([-1.0, -0.5, -0.2]), torch.tensor([-1.2, -0.5, -0.4]))
adv, ret = gae(r, values=torch.tensor([0.5, 0.6, 0.8]))
print(r, adv, ret)
```

## 实例演算 {#worked-example}

**手算 REINFORCE。**单 token 回答，词表 $\{A, B, C\}$，logits $z = (0, 0, 0)$，所以 $p = (\tfrac13, \tfrac13, \tfrac13)$。奖励 $R = 1$ 当且仅当为 $A$，否则为 0。

对于 softmax 策略，$\nabla_z \log p_k = e_k - p$，其中 $e_k$ 是 token $k$ 的独热向量。

*没有基线：*

- 采样到 $A$（概率 1/3）：梯度为 $1\cdot(e_A - p) = (\tfrac23, -\tfrac13, -\tfrac13)$。
- 采样到 $B$ 或 $C$：奖励为 0，梯度为 $0$。
- 期望梯度：$\tfrac13 (\tfrac23, -\tfrac13, -\tfrac13) = (\tfrac29, -\tfrac19, -\tfrac19)$。
- 第一个分量的方差：它以概率 $\tfrac13$ 取 $\tfrac23$，否则取 0。$\mathbb{E}[g^2] = \tfrac13\cdot\tfrac49 = \tfrac{4}{27}$，所以 $\mathrm{Var} = \tfrac{4}{27} - (\tfrac29)^2 = \tfrac{12}{81} - \tfrac{4}{81} = \tfrac{8}{81} \approx 0.099$。

*有基线 $b = \mathbb{E}[R] = \tfrac13$：*

- 采样到 $A$：$(1-\tfrac13)(e_A - p) = \tfrac23(\tfrac23, -\tfrac13, -\tfrac13) = (\tfrac49, -\tfrac29, -\tfrac29)$。
- 采样到 $B$：$(0-\tfrac13)(e_B - p) = -\tfrac13(-\tfrac13, \tfrac23, -\tfrac13) = (\tfrac19, -\tfrac29, \tfrac19)$。
- 采样到 $C$：由对称性得 $(\tfrac19, \tfrac19, -\tfrac29)$。
- 期望梯度：$\tfrac13\big[(\tfrac49+\tfrac19+\tfrac19), \dots\big] = (\tfrac29, -\tfrac19, -\tfrac19)$。**与之前相同：没有偏差。**
- 第一个分量取值为 $\tfrac49, \tfrac19, \tfrac19$：$\mathbb{E}[g^2] = \tfrac13(\tfrac{16}{81}+\tfrac1{81}+\tfrac1{81}) = \tfrac{6}{81}$，所以 $\mathrm{Var} = \tfrac{6}{81} - \tfrac{4}{81} = \tfrac{2}{81} \approx 0.025$。

基线把这个分量的方差降低到原来的 1/4。还要注意，有了基线之后，*失败的*样本现在也携带信息（把 $B$ 压低），而没有基线时它们毫无贡献。

**手算 PPO 裁剪**（$\epsilon = 0.2$）：

| $A_t$ | $\rho_t$ | $\rho_t A_t$ | $\operatorname{clip}(\rho_t)A_t$ | min | 梯度是否流动？ |
|---|---|---|---|---|---|
| +2 | 1.1 | 2.2 | 2.2 | 2.2 | 是 |
| +2 | 1.5 | 3.0 | 2.4 | 2.4 | 否（已经上涨超过 20%） |
| −1 | 0.6 | −0.6 | −0.8 | −0.8 | 否（已经下降超过 20%） |
| −1 | 1.5 | −1.5 | −1.2 | −1.5 | 是（撤销错误的上涨） |

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **用于 RLHF 的 PPO**（InstructGPT 以及许多 2022–2024 年的聊天模型）使用由奖励模型或 SFT 模型初始化的价值网络、GAE、KL 惩罚以及 $\epsilon \approx 0.2$。它需要在显存中同时放四个模型；见[第 2.4 章](04-rlhf.md)。
- **无 critic 的方法**主导了推理 RL：GRPO 及其后继者用组均值基线取代了价值网络（[第 2.7 章](07-rl-with-verifiable-rewards.md)）。RLOO（REINFORCE leave-one-out）是与之密切相关的一种基线，它使用*其他*样本的平均奖励。
- **比值在对数空间中计算**（$\exp(\log\pi_{\text{new}} - \log\pi_{\text{old}})$）以保证数值稳定，而且 $\log\pi_{\text{old}}$ 由训练引擎重新计算，因为生成 rollout 的推理引擎给出的概率可能略有不同（这是[第 2.7 章](07-rl-with-verifiable-rewards.md)讨论的离策略不匹配的一个来源）。
- **每个 batch 做一到几个 PPO epoch** 对 LLM 来说是典型做法；许多推理 RL 训练每个 batch 只做一步梯度更新，这时 $\rho_t = 1$，裁剪只对第一个之后的小批次才有意义。

## 常见误区 {#common-pitfalls-misconceptions}

- **"策略梯度要求奖励可微。"**不是：对数导数技巧把梯度转移到了 $\log\pi_\theta$ 上。奖励可以是任意黑盒。
- **忘记对优势/奖励做 detach。**$\log\pi_\theta$ 前面的权重必须当作常数处理（`.detach()`），否则你会通过策略损失把梯度反传进基线或价值网络。
- **依赖于所采样动作的基线**会引入偏差。batch 均值没问题（偏差为 $O(1/N)$）；leave-one-out 则能完全消除偏差。
- **混淆 KL 的方向。**RLHF 惩罚的是 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$，在来自 $\pi_\theta$ 的样本上估计；这是"寻模（mode-seeking）"的：策略可能会丢掉参考模型的某些行为，但不应该凭空发明参考模型认为极不可能出现的行为。
- **裁剪不是硬约束。**比值仍然可能超出 $1\pm\epsilon$（例如由于其他 token 更新的影响）；请监控"裁剪比例（clip fraction）"以及 $\pi_\theta$ 与 $\pi_{\text{old}}$ 之间的近似 KL。

## 本章要点 {#key-takeaways}

- 把一条回答当作一个回合：状态 = 前缀，动作 = token，奖励在末尾给出。
- 策略梯度：$\nabla J = \mathbb{E}[R \sum_t \nabla\log\pi_\theta(y_t\mid s_t)]$，由对数导数技巧推出；不需要对采样或奖励求梯度。
- 减去基线能保持梯度无偏，并可大幅降低方差；优势函数把这一做法推广到逐 token。
- PPO 用重要性比值复用样本，并将其裁剪，使其保持在 $1\pm\epsilon$ 的信任域内。
- 价值网络 + GAE 估计逐 token 的优势；对冻结参考模型的 KL 惩罚用于防止奖励投机和漂移。

## 自测 {#check-your-understanding}

??? question "为什么减去一个常数基线不会改变期望梯度？"
    因为 $\mathbb{E}_{y\sim\pi}[\nabla\log\pi(y)] = \nabla\sum_y\pi(y) = \nabla 1 = 0$，所以对任何不依赖于 $y$ 的 $b$，都有 $\mathbb{E}[b\,\nabla\log\pi] = 0$。

??? question "在只有单个终端奖励且没有价值网络的情况下，带基线的 REINFORCE 给一条回答的每个 token 分配什么优势？"
    每个 token 的值都相同：$R(x,y) - b(x)$。好回答的所有 token 被同等地推高；回答内部的功劳分配留给对大量样本取平均来解决。

??? question "在 PPO 中，某个 token 的 $A_t = +1$，$\rho_t = 1.3$，$\epsilon=0.2$。此时 $L^{\text{CLIP}}_t$ 及其梯度是多少？"
    $\min(1.3, 1.2) = 1.2$，取自裁剪项，它对 $\theta$ 是常数，所以梯度为零。

??? question "当 $\lambda = 1$ 且 $\gamma = 1$ 时，GAE 化简为什么？"
    $\hat A_t = \sum_{t'\ge t} r_{t'} - V(s_t)$：从 $t$ 起的实际回报减去 critic 的估计（带基线的蒙特卡洛回报）。

??? question "给出对参考模型使用 KL 惩罚的两个理由。"
    它防止策略利用不完美奖励模型的误差，跑到奖励不可靠的区域去；同时它保持了参考模型的流畅性和多样性（也提升了稳定性）。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | 带学习得到的价值函数的裁剪比值策略梯度；经典 RLHF 所用的 RL 算法。 |
| :material-book-open-variant: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015 | GAE 以及 $\lambda$ 在偏差–方差权衡中的作用的出处。 |
| :material-book-open-variant: [RLHF Book](https://rlhfbook.com/) — Nathan Lambert | 专为语言模型写的策略梯度章节。 |
