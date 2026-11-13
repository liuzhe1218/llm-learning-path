# 2.7 可验证奖励强化学习（RLVR）

!!! abstract "本章你将学到"
    - 什么是可验证奖励，以及如何编写它们（数学答案检查、单元测试、格式奖励）
    - 从 PPO 推导出 GRPO：用组内基线代替价值网络，配有数值实例演算和代码
    - DeepSeek-R1-Zero 与 R1 的配方：纯 RL、冷启动、多阶段 RL、拒绝采样 SFT、蒸馏
    - GRPO 的主要失效模式及其修复：DAPO、Dr. GRPO、GSPO、CISPO
    - RLVR 能带来什么、不能带来什么（pass@k 之争），以及为什么 rollout 生成主导了 RL 基础设施

## 为什么重要 {#why-this-matters-the-problem}

RLHF（[第 2.4 章](04-rlhf.md)）优化的是一个*学习得到的*奖励模型，这是一个脆弱的代理：推得太狠，策略就会钻它的空子。这限制了 RL 能跑多久，以及能在多大程度上改变模型。

对许多有价值的任务，我们根本不需要学习得到的裁判。数学题有已知的最终答案；编程题有单元测试；一次工具调用要么成功，要么失败。程序可以精确地检查这些。**可验证奖励强化学习（RL with verifiable rewards，RLVR）**就是把这类检查当作奖励。因为这种奖励（几乎）无法靠风格去糊弄，策略可以优化数千步，2025 年这催生了[第 2.6 章](06-reasoning-and-test-time-compute.md)中的长思维链推理模型。DeepSeek-R1 把这套配方公开，如今它是大多数开源推理模型的基础。

## 直觉 {#intuition}

给模型一道难的数学题，让它尝试 16 次，检查每个最终答案，并强化那些*相对于其他尝试*答对了的尝试。模型总是做对或总是做错的题，没什么可学的（没有对比）；它有时做对的题，才是学习发生的地方。在大量题目上，凡是能提高成功率的习惯（写出步骤、检查结果、发现矛盾后回退）都会被强化，尽管从来没有人示范过它们。回答的长度会自行增长，因为想得更久是划算的。

## 工作原理 {#how-it-works}

### 可验证奖励 {#verifiable-rewards}

可验证奖励是 prompt 和完整回答的一个确定性函数 $R(x, y)$。典型的组成部分：

| 奖励 | 检查方式 | 备注 |
|---|---|---|
| **数学准确性** | 提取最终答案（例如在 `\boxed{}` 内），归一化后与参考答案比较；或检查符号等价 | 归一化很重要：`1/2`、`0.5`、`\frac{1}{2}` |
| **代码正确性** | 在沙箱中运行隐藏的单元测试；全部通过则奖励为 1（或取通过的比例） | 需要超时、隔离、资源限制 |
| **格式** | 回答包含要求的结构（例如推理放在 `<think>...</think>` 里，然后是答案） | 权重要小；它塑造的是输出解析，而不是正确性 |
| **指令约束** | "恰好 3 个要点""不超过 50 个词""用法语回答"（IFEval 风格） | 程序化检查 |
| **智能体结果** | 环境的最终状态（仓库中的测试通过、网页任务成功） | 见[第 2.9 章](09-agents-tool-use-and-safety.md) |

奖励通常是二元的（0/1），或是少数几个分量之和。验证器必须足够鲁棒：任何漏洞（接受多个答案、把崩溃的测试算作通过、测试文件可被编辑）都会被 RL 发现并利用。

### 从 PPO 到 GRPO {#from-ppo-to-grpo}

回顾 PPO（[第 2.3 章](03-rl-foundations-for-llms.md)）：逐 token 的截断代理目标，优势由学习得到的价值网络通过 GAE 给出。价值网络和策略一样大，使需要训练的模型占用的显存翻倍，而且当唯一的奖励出现在一个 10,000 token 回答的末尾时，很难训好。

**组相对策略优化（Group Relative Policy Optimization，GRPO）**由 DeepSeekMath（Shao et al., 2024）提出，它用针对*同一个* prompt 的**一组**样本得到的蒙特卡洛基线来取代价值网络：

1. 对 prompt $x$，从 $\pi_{\theta_{\text{old}}}$ 采样一组 $G$ 个回答 $\{y_1,\dots,y_G\}$。
2. 给每个回答打分：奖励 $r_1,\dots,r_G$。
3. 回答 $i$ 的优势（其所有 token 共享）：

$$
\hat A_{i} = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon_{\text{num}}}.
$$

用大白话说：一个回答是否好，看它是否比同组的兄弟回答得分更高，并以组内的离散程度为单位来度量；$\varepsilon_{\text{num}}$ 是一个防止除以零的极小常数。组内均值是对 $V(x)$（该 prompt 的期望奖励）的估计，这恰好就是基线应有的样子（[第 2.3 章](03-rl-foundations-for-llms.md)）。这就是**组内优势**。

4. 最大化 PPO 风格的截断目标，并把 KL 惩罚直接写进损失：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert y_i\rvert}\sum_{t=1}^{\lvert y_i\rvert}\Big(\min\big(\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i}\big) - \beta\,\widehat{\mathrm{KL}}_{i,t}\Big)\Bigg],
$$

其中 $\lvert y_i\rvert$ 是回答 $i$ 的 token 数，$\rho_{i,t} = \pi_\theta(y_{i,t}\mid s_{i,t})/\pi_{\theta_{\text{old}}}(y_{i,t}\mid s_{i,t})$ 是逐 token 的比值，$\widehat{\mathrm{KL}}_{i,t} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$ 是在被采样 token 处计算的非负 KL 估计量。用大白话说：这就是 PPO 的截断更新，只不过一个回答的每个 token 共享该回答的组内优势，损失先对每个回答内部取平均、再对整组取平均，而且与参考模型的 KL 是一个惩罚项，而不是奖励的一部分。

我们得到了什么：没有价值网络（DeepSeekMath 论文强调了内存和计算的减少），奖励可验证时也不需要奖励模型，而且基线对*这个具体的 prompt* 是准确的。我们付出了什么：每个 prompt 要采样 $G$ 个样本（通常 8–64 个），并且没有逐 token 的信用分配。

### DeepSeek-R1-Zero：从基座模型出发的纯 RL {#deepseek-r1-zero-pure-rl-from-a-base-model}

DeepSeek-R1（DeepSeek-AI, 2025）首先训练了 **R1-Zero**：直接把 GRPO 用在 DeepSeek-V3 的*基座*模型上，完全没有 SFT。

- **提示模板**：一条简单的指令，要求在 `<think> ... </think>` 内推理，然后在 `<answer> ... </answer>` 内给出答案。没有如何推理的示例。
- **奖励**：只用基于规则的奖励。一个准确性奖励（数学答案按指定格式给出，与参考答案核对；代码由编译器对照测试用例检查），以及一个使用标签的格式奖励。作者有意避免使用神经奖励模型，理由是大规模下的奖励投机。
- **报告的结果**：训练期间 AIME 2024 的 pass@1 从 15.6% 升到 71.0%（多数投票为 86.7%），与当时的 OpenAI o1-0912 相当。
- **涌现行为**：训练过程中平均回答长度稳步增长，从几百 token 到几千 token，模型开始重新评估自己的步骤，其中有一次"顿悟时刻"，它明确地停下来重新思考。这些都没有被示范过；RL 之所以发现它们，是因为它们提高了奖励。
- **问题**：可读性差，且回答中混用语言。

### DeepSeek-R1：完整的多阶段配方 {#deepseek-r1-the-full-multi-stage-recipe}

为了修复 R1-Zero 的问题并得到一个通用助手，R1 采用四个阶段：

```text
 V3-Base ─► (1) 冷启动 SFT ─► (2) 推理 RL ─► (3) 拒绝采样 + SFT ─► (4) 全场景 RL ─► R1
            数千条长而        GRPO，基于规则     从第 2 阶段模型采样的         推理用规则奖励，
            可读的 CoT 样本   的奖励 + 语言      约 60 万条推理样本（已        通用场景用奖励模型
                              一致性奖励         过滤）+ 约 20 万条非推理      衡量有用性与无害性
                                                 样本 → 再次 SFT 基座模型
                                                                         │
                                                                         └──► (5) 蒸馏：用同样的约 80 万条样本
                                                                               SFT 小型 Qwen2.5 / Llama 模型
```

1. **冷启动。** 在数千条格式可读的长 CoT 样本上做 SFT。这为 RL 提供了一个更好、更可读的起点。
2. **面向推理的 RL。** 在数学、代码、科学和逻辑 prompt 上用基于规则的奖励做 GRPO，另加一个*语言一致性*奖励（目标语言词汇所占比例）来防止语言混用，代价是准确率略有下降。
3. **拒绝采样与 SFT。** 从 RL 检查点采样大量回答，只保留正确且可读的（约 60 万条推理样本），再加入约 20 万条非推理样本（写作、问答等），并用这约 80 万条数据微调基座模型。这既巩固了 RL 的收益，也恢复了通用能力。
4. **全场景 RL。** 第二个 RL 阶段，在通用 prompt 上把推理的规则奖励与衡量有用性和无害性的奖励模型结合起来。
5. **蒸馏。** 同样的约 80 万条样本被用来 SFT 较小的 Qwen2.5 和 Llama 模型（1.5B 到 70B）。R1 报告发现，从 R1 蒸馏给小模型带来的推理能力，比直接在小模型上做大规模 RL 强得多（[第 2.8 章](08-distillation-and-efficient-fine-tuning.md)）。

Kimi k1.5（Kimi Team, 2025）报告了另一条通往强推理的路线：长上下文（最长 128K token）RL、基于在线镜像下降的策略优化方法、长度惩罚，以及把长 CoT 能力迁移到更短回答里的"long2short"方法。

### 失效模式及其修复 {#failure-modes-and-fixes}

在长 CoT 问题上长时间运行 GRPO 暴露出若干问题。每个修复都是一个小而有针对性的改动。

**1. 熵坍缩 → DAPO 的"clip-higher"。** 策略的 token 熵（即**熵坍缩**，entropy collapse）下降很快；同一组内的样本几乎一模一样，探索停止，学习停滞。一个原因是对称的截断：$\epsilon = 0.2$ 时，$\pi_{\text{old}} = 0.01$ 的 token 每次更新最多只能升到 $0.012$，而 $0.9$ 的 token 却可以升到上限 1。低概率的"探索性" token 被压住了。DAPO（Yu et al., 2025）把截断范围**解耦**：$\operatorname{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$，并使用更大的上界（论文中 $\epsilon_{\text{low}} = 0.2$、$\epsilon_{\text{high}} = 0.28$），让不太可能出现的好 token 增长得更快。

**2. 零优势组 → 动态采样。** 如果全部 $G$ 个回答得到相同的奖励（全对或全错），那么所有优势都为 0，这一组不提供任何梯度，只有来自 KL 项的噪声。随着模型进步，越来越多的 prompt 变成全对，有效批大小随之缩小。DAPO **对 prompt 过采样，并过滤掉准确率为 0 或 1 的组**，不断补充批次，直到每一组都有对比。把 prompt 筛选到模型的"难度前沿"，目的也是一样的。

**3. 长度偏置 → token 级损失（DAPO）与 Dr. GRPO。** $\frac{1}{\lvert y_i\rvert}$ 的归一化先对每个回答内部的 token 取平均，再对各回答取平均。后果是：

- 对正确回答（正优势），较短的回答每个 token 得到更大的更新，使得在正确答案之间倾向于简短。
- 对错误回答（负优势），*更长*的回答把惩罚分摊到更多 token 上，于是每个 token 受到的惩罚更小：**长的错误答案受罚更轻**，这会在不提升准确率的情况下推高长度。

DAPO 改用 **token 级**平均：把批内所有回答的逐 token 损失求和，再除以 token 总数，让每个 token 的权重相同。Dr. GRPO（"GRPO done right"，Liu et al., 2025）认为 GRPO 中的两种归一化都有偏：$\frac{1}{\lvert y_i\rvert}$ 项（长度偏置，如上所述），以及除以 $\operatorname{std}(r)$（它会加大奖励方差较低的 prompt 的权重，也就是那些几乎总是或几乎从不被解出的题，这是一种**难度偏置**）。它把这两者都去掉，使用 $\hat A_i = r_i - \operatorname{mean}(r)$ 和一个常数归一化因子，并报告在准确率相近的情况下有更好的 token 效率（错误回答更短）。DAPO 还加入了**超长奖励整形**：在接近最大长度时施加软长度惩罚，并对被截断的样本做掩码，这样模型就不会因为"推理不错但被截断"而受到带噪声的惩罚。

**4. 带噪声的 token 级比值 → GSPO。** 奖励是针对整个序列给出的，但 GRPO 却对每个 *token* 施加重要性比值和截断。逐 token 的比值带有噪声，而且噪声会在长回答中累积；Zheng 等人（2025，Qwen 团队）认为这会让训练不稳定，对混合专家（MoE）模型尤其如此，因为一个 token 激活的专家在旧策略和新策略之间可能不同。**GSPO**（Group Sequence Policy Optimization）使用**序列级**、经长度归一化的比值：

$$
s_i(\theta) = \Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\Big)^{1/\lvert y_i\rvert} = \exp\Big(\frac{1}{\lvert y_i\rvert}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid s_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid s_{i,t})}\Big),
$$

并对 $s_i$（每个回答一个数）做截断，而不是对每个 token 的比值截断。用大白话说：整个回答要么一起保留、要么一起被截断，与奖励所对应的单位一致。Qwen 团队报告它稳定得多，并把它用于 Qwen3 的 RL。

**5. 稀有 token 被截断掉 → CISPO。** 在 PPO/GRPO 中，一旦某个 token 的比值越出截断范围，它的梯度就是*零*。MiniMax（MiniMax-M1, 2025）观察到，与反思相关的稀有 token（"However"、"Wait"、"Recheck"）概率很低，很快就触及上截断，此后的离策略更新中就不再获得梯度。**CISPO** 截断的是重要性采样*权重*，并把它当作常数（stop-gradient），但对每个 token 都保留策略梯度项：

$$
J_{\text{CISPO}} = \mathbb{E}\Big[\frac{1}{\sum_i \lvert y_i\rvert}\sum_{i,t}\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big)\,\hat A_{i}\,\log\pi_\theta(y_{i,t}\mid s_{i,t})\Big],
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。用大白话说：重要性权重有界以控制方差，但没有任何 token 被彻底静音。这就是带截断重要性权重的 REINFORCE。

| 问题 | 症状 | 修复 |
|---|---|---|
| 熵坍缩 | 熵 → 0，样本完全相同，奖励持平 | Clip-higher（$\epsilon_{\text{high}} > \epsilon_{\text{low}}$）（DAPO） |
| 零优势组 | 许多组全对/全错；有效批大小缩小 | 动态采样/难度过滤（DAPO） |
| 长度偏置 | 长度增长，错误回答变得更长 | token 级损失（DAPO）；去掉 $1/\lvert y\rvert$ 和 std 归一化（Dr. GRPO） |
| token 比值噪声，MoE 不稳定 | 损失尖峰，训练崩溃 | 序列级比值与截断（GSPO） |
| 稀有 token 被截断掉 | 反思行为无法增长 | 截断重要性权重，保留每个 token 的梯度（CISPO） |

### RLVR 带来什么、不带来什么 {#what-rlvr-does-and-doesnt-add}

Yue 等人（2025）用 **pass@k**（[第 2.6 章](06-reasoning-and-test-time-compute.md)）在 $k$ 从 1 到非常大的范围内比较了基座模型和经过 RLVR 训练的版本：

- 在**小 $k$**（例如 pass@1）时，RL 模型明显更好。
- 在**大 $k$** 时，基座模型追了上来，常常还*超过* RL 模型：RL 模型能解出的问题，通常基座模型采样足够多次也能解出，而且 RL 模型能覆盖的可解问题范围可能反而缩小。

他们的解读是：当前的 RLVR 主要是把分布向基座模型本来就能生成的推理路径**锐化**，提高的是采样效率，而不是扩大可解问题的集合；相比之下，来自更强模型的蒸馏可以加入新的推理模式。这与 s1（[第 2.6 章](06-reasoning-and-test-time-compute.md)）以及"表面对齐"的观点（[第 2.1 章](01-why-post-training.md)）一致。

争论仍在继续：让 pass@1 逼近 pass@k 本身就极有价值（用户只拿到一个答案），更长的 RL 训练和更多样的环境也许能突破基座模型的覆盖范围，而上面这些保持熵的修复方法，目的正是让探索保持活力。请把"RL 只是锐化"看作一个在已研究设定中有充分证据的重要假说，而不是定律。

### RL 基础设施 {#rl-infrastructure}

一个 GRPO 步骤有两个性质截然不同的阶段：

```text
  ┌───────────────── rollout（生成）─────────────────┐   ┌──── 训练 ────┐
  │ B 个 prompt × G 个样本，自回归，最长 32K+ token   │──►│ 对所有 token │──► 把新权重
  │ 受内存带宽限制的解码，长度呈长尾分布              │   │ 前向/反向    │    同步给生成器
  └──────────────────────────────────────────────────┘   └──────────────┘
          通常占据挂钟时间的大头
```

- **Rollout 主导成本。** 逐 token 生成数千条长回答既慢，又受内存带宽限制，而训练阶段是并行处理所有 token 的。因此 RL 系统会嵌入一个快速的**推理引擎**（vLLM、SGLang；见第 3 部分）负责生成，再配一个训练框架（FSDP、Megatron）负责更新，并且每一步之后都要把更新后的权重从训练器复制到生成器。
- **长尾。** 一个批次要等最长的那个回答生成完才结束；大部分 GPU 闲着，等少数序列跑到长度上限。解决办法包括部分 rollout（把长的生成暂停，在下一轮迭代里继续，如 Kimi k1.5 所述）和动态批处理。
- **异步/离策略 rollout。** 第 $k+1$ 步的生成与第 $k$ 步的训练并行，所以 rollout 来自一个略微过时的策略。重要性比值（以及截断）校正了这种陈旧性；这是离策略校正在 LLM RL 中重要的一个主要现实原因。
- **训练–推理不匹配。** 推理引擎和训练器对同样的 token 算出的概率略有不同（内核、精度、MoE 路由都可能不同）。在训练器中重新计算 $\log\pi_{\text{old}}$，并对生成器与训练器的概率之间做重要性校正，可以减轻由此带来的不稳定。
- **开源框架：** [verl](https://github.com/volcengine/verl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL](https://github.com/huggingface/trl) 和 [slime](https://github.com/THUDM/slime) 实现了这些模式。

## 代码 {#code}

### 1. 可验证奖励函数 {#1-verifiable-reward-functions}

```python
import re, subprocess, tempfile, os

def normalize(ans):
    ans = ans.strip().replace(" ", "").replace("\\!", "").rstrip(".")
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", ans)
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", ans)      # a/b -> 数值（绝不要用 eval！）
    try:
        val = float(m[1]) / float(m[2]) if m else float(ans)
        return str(round(val, 6))
    except (ValueError, ZeroDivisionError):
        return ans                                                   # 退化为字符串匹配

def last_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1] if m else None

def math_reward(response, reference):
    pred = last_boxed(response)
    return 1.0 if pred is not None and normalize(pred) == normalize(reference) else 0.0

def format_reward(response):
    ok = re.fullmatch(r"\s*<think>.+?</think>\s*\S.*", response, flags=re.DOTALL)
    return 1.0 if ok and response.count("<think>") == 1 else 0.0

def code_reward(program, tests, timeout=5):
    """在子进程中运行程序 + 断言。实际使用时请用真正的沙箱（容器、无网络）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program + "\n\n" + tests)
        path = f.name
    try:
        res = subprocess.run(["python3", path], capture_output=True, timeout=timeout)
        return 1.0 if res.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        os.remove(path)

resp = "<think>1/2 + 1/4 = 3/4</think> The answer is \\boxed{\\frac{3}{4}}"
print(math_reward(resp, "0.75"), format_reward(resp))            # 1.0 1.0
print(code_reward("def add(a,b): return a+b", "assert add(2,3)==5"))  # 1.0
```

### 2. GRPO 优势与损失（带 DAPO / Dr. GRPO / GSPO / CISPO 开关） {#2-grpo-advantages-and-loss-with-dapo-dr-grpo-gspo-cispo-switches}

```python
import torch

def group_advantages(rewards, G, scale_by_std=True, eps=1e-6):
    """rewards: [B*G]，按组依次排列 -> 优势 [B*G]。"""
    r = rewards.view(-1, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:                                   # GRPO；Dr. GRPO 将其设为 False
        adv = adv / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    return adv.view(-1)

def policy_loss(logp, logp_old, adv, mask, mode="grpo", eps_low=0.2, eps_high=0.2,
                logp_ref=None, beta=0.0):
    """logp, logp_old, logp_ref, mask: [N, T]（N = B*G 个回答）；adv: [N]。
    返回一个要最小化的标量。"""
    A = adv.unsqueeze(1)                                # 每个 token 共享同一个优势
    log_ratio = logp - logp_old
    if mode == "gspo":                                  # 每个序列一个经长度归一化的比值
        seq = (log_ratio * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        ratio = torch.exp(seq)
        per_tok = -torch.min(ratio * A, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
        per_tok = per_tok.expand_as(logp)
    elif mode == "cispo":                               # 截断的、stop-gradient 的 IS 权重 * log pi
        w = torch.clamp(torch.exp(log_ratio), 1 - eps_low, 1 + eps_high).detach()
        per_tok = -w * A * logp
    else:                                               # grpo / dapo：逐 token 的 PPO 截断
        ratio = torch.exp(log_ratio)
        per_tok = -torch.min(ratio * A, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
    if beta > 0 and logp_ref is not None:               # k3 KL 估计量，始终 >= 0
        d = logp_ref - logp
        per_tok = per_tok + beta * (torch.exp(d) - d - 1)
    if mode == "grpo":                                  # 先对每个回答取平均，再对回答取平均
        return ((per_tok * mask).sum(1) / mask.sum(1)).mean()
    return (per_tok * mask).sum() / mask.sum()          # token 级平均（DAPO、CISPO、GSPO 风格）

def keep_informative_groups(rewards, G):
    """DAPO 动态采样：返回奖励不全相同的组的索引。"""
    r = rewards.view(-1, G)
    return ((r.max(1).values - r.min(1).values) > 0).nonzero().squeeze(-1)

rewards = torch.tensor([1., 0., 0., 1.,   1., 0., 0., 0.,   1., 1., 1., 1.])
print(group_advantages(rewards, G=4).view(3, 4))
print(keep_informative_groups(rewards, G=4))          # tensor([0, 1])：第 2 组全对
```

一个训练步骤就是：用推理引擎为每个 prompt 采样 $G$ 个回答，计算奖励，丢弃没有信息量的组，用训练器计算 `logp_old`（若 $\beta > 0$ 还要算 `logp_ref`），然后在 `policy_loss` 上做一次或多次优化器更新。设置 `mode="dapo", eps_high=0.28, beta=0` 就得到 DAPO 的损失；`scale_by_std=False` 加一个常数归一化因子就得到 Dr. GRPO。

## 实例演算 {#worked-example}

**$G = 4$ 时的 GRPO 优势**（采用总体标准差，与代码一致；有些实现使用样本标准差，那会改变尺度但不改变符号）。

*A 组*，奖励 $(1, 0, 0, 1)$：均值 $= 0.5$，标准差 $= 0.5$。优势 $= (1-0.5)/0.5, \dots = (+1, -1, -1, +1)$。两个正确的回答被推高，两个错误的被压低，幅度相同。

*B 组*，奖励 $(1, 0, 0, 0)$（一道难题）：均值 $= 0.25$，标准差 $= \sqrt{0.25\cdot0.75} \approx 0.433$。

- 正确的：$(1 - 0.25)/0.433 \approx +1.732$。
- 每个错误的：$(0 - 0.25)/0.433 \approx -0.577$。

唯一的成功样本得到很大的正优势：难题上罕见的成功会被强烈强化。注意一组内的优势之和为零。

*C 组*，奖励 $(1,1,1,1)$：均值 $= 1$，标准差 $= 0$，每个优势 $= 0/(0+\varepsilon) = 0$。**没有学习信号**；动态采样会丢弃这一组。

*B 组上的 Dr. GRPO*：不除以标准差，优势 $= (+0.75, -0.25, -0.25, -0.25)$。与 GRPO 相比，这一组相对于奖励为 $(1,0,1,0)$ 的组（Dr. GRPO 优势为 $\pm0.5$，GRPO 为 $\pm1$）的影响力，不再按奖励的离散程度重新缩放。

**数字上看长度偏置。** 一组中的两个错误回答优势都是 $-1$；一个有 100 个 token，另一个有 1,000 个。按 GRPO 的逐回答平均，短回答的每个 *token* 得到的权重是 $-1/100 = -0.01$，长回答的每个 token 是 $-1/1000 = -0.001$：长的错误回答的 token 被压低的幅度小了 10 倍。若按批内 token 级平均，两者每个 token 的权重相同。

**数字上看 clip-higher。** 对于 $\pi_{\text{old}} = 0.01$、优势为正的 token，$\epsilon = 0.2$ 把新概率（就梯度而言）限制在 $0.012$；$\epsilon_{\text{high}} = 0.28$ 则允许到 $0.0128$。对于概率为 $0.5$ 的 token，上限分别是 $0.6$ 和 $0.64$。无论哪种，稀有 token 的绝对增长空间都很小，这就是没有这些修复时熵会坍缩的原因。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **GRPO 系列算法是开源推理 RL 的默认选择**（DeepSeek-R1、用 GSPO 的 Qwen3、许多带 DAPO 与 Dr. GRPO 改动的学术复现）。DAPO 报告称，用它的四项技术在 Qwen2.5-32B 基座模型上于 AIME 2024 取得 50 分。
- **使用可验证奖励时，KL 往往很小甚至为零**（DAPO 把它去掉了），因为 KL 惩罚的主要理由（可被钻空子的学习型奖励）弱化了。有些训练仍为稳定性保留它。
- **数据整理至关重要**：prompt 要有可靠的参考答案，与评测集去重，并筛选到模型的难度前沿（既不是总做对，也不是从不做对）。
- **混合奖励**：数学/代码/工具用可验证奖励，后期阶段再与通用有用性的奖励模型结合，如 R1 的最后一个阶段。
- **智能体式 RLVR**：多轮环境（带测试的代码仓库、浏览器、终端）是 2025–2026 年的前沿（[第 2.9 章](09-agents-tool-use-and-safety.md)）。
- **监控项**：平均奖励、各难度上的准确率、回答长度、熵、截断比例、零优势组所占比例，以及训练与推理之间的对数概率差距。

## 常见误区 {#common-pitfalls-misconceptions}

- **漏洞百出的验证器。** 在同一个回答里同时接受 `\boxed{3}` *和* `\boxed{4}`，或者匹配文本中的任意数字，很快就会被钻空子。只检查最后一个答案，并惩罚给出多个答案的情形。
- **过度加重格式奖励。** 模型会学会在错误答案外面套上完美的标签。让格式奖励保持很小，或把它设为一道门槛。
- **太简单或太难的 prompt。** 全对和全错的组会浪费 rollout。
- **"回答越长，推理越好。"** 长度可能仅仅因为长度偏置就增长；请跟踪每个 token 的准确率。
- **"RL 教会了模型从零开始推理。"** R1-Zero 是从一个很强的基座模型出发的；证据（pass@k）表明 RL 主要是锐化已有的能力。
- **忽视评测污染。** 许多公开数学数据集与训练数据有重叠；请使用全新的留出题目。

## 本章要点 {#key-takeaways}

- RLVR 用程序化检查取代学习得到的奖励模型，使长时间 RL 训练成为可能而不会被奖励模型钻空子（验证器如果有漏洞，仍然可能被钻空子）。
- GRPO = 用组内优势 $(r_i - \bar r)/\sigma_r$ 代替价值网络的 PPO；一个回答的每个 token 共享它的优势。
- R1-Zero 表明，对基座模型做纯 RL 就能激发出长 CoT 和自我验证；R1 在此基础上加入了冷启动 SFT、多阶段 RL、拒绝采样 SFT 和蒸馏。
- 修复手段：clip-higher 与动态采样（DAPO）、无偏归一化（Dr. GRPO、token 级损失）、序列级比值（GSPO）、截断且 stop-gradient 的 IS 权重（CISPO）。
- RLVR 主要通过锐化来提升 pass@1；它是否扩大了 pass@k 的覆盖范围仍有争议。
- Rollout 生成主导成本；RL 系统嵌入推理引擎，并使用异步、离策略校正。

## 自测 {#check-your-understanding}

??? question "为什么 GRPO 不需要价值网络？"
    它的基线是同一 prompt 下其他样本的平均奖励，由这一组样本经验地估计出来，而不是由一个学习得到的评论家（critic）给出。

??? question "计算奖励为 (0, 0, 1, 1, 1, 1) 时的 GRPO 优势（使用总体标准差）。"
    均值 $= 4/6 \approx 0.667$；方差 $= \frac{2(0.667)^2 + 4(0.333)^2}{6} = \frac{0.889 + 0.444}{6} \approx 0.222$，标准差 $\approx 0.471$。正确的：$(1-0.667)/0.471 \approx +0.707$；错误的：$(0-0.667)/0.471 \approx -1.414$。

??? question "为什么 RLVR 能用于推理，而基于人类偏好的 RLHF-PPO 没有产生长 CoT？"
    可验证奖励准确且难以钻空子，所以策略可以优化很多步；而且强基座模型本来就包含推理行为，RL 把它们放大了。

??? question "DAPO 的动态采样解决了什么问题？"
    每个回答奖励都相同的组优势为零，不提供梯度；随着训练推进，这样的组越来越多，有效批大小随之缩小。动态采样通过过采样，只保留奖励有高有低的组。

??? question "什么是奖励投机？举一个 RLVR 中的例子。"
    策略利用奖励的缺陷，而不是去真正解决任务——例如输出多个答案以匹配一个宽松的解析器，或在编程环境中修改单元测试。

??? question "Yue 等人的 pass@k 研究对 RLVR 说明了什么？"
    经 RL 训练的模型在小 $k$ 时胜过基座模型，但在大 $k$ 时基座模型解出的问题一样多甚至更多；RLVR 主要是让本来就够得着的解法出现得更频繁，而不是增加新的解法（在所研究的设定中）。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024 | 组相对策略优化：以组内归一化的奖励作为优势，无需价值网络。 |
| :material-star: [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | R1-Zero（纯 RL）与多阶段 R1 配方及蒸馏；2025 年最关键的后训练论文。 |
| :material-new-box: [Kimi k1.5: Scaling RL with LLMs](https://arxiv.org/abs/2501.12599) — Kimi Team, 2025 | 长上下文 RL、在线镜像下降、长度惩罚、部分 rollout、long2short。 |
| :material-new-box: [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) — Yu et al. (ByteDance Seed), 2025 | Clip-higher、动态采样、token 级损失、超长奖励整形。 |
| :material-new-box: [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) — Liu et al., 2025 | GRPO 中的长度偏置和 std 归一化偏置，以及一个无偏变体。 |
| :material-new-box: [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — Zheng et al. (Qwen), 2025 | 序列级比值与截断；MoE 模型上稳定的 RL，用于 Qwen3。 |
| :material-new-box: [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) — MiniMax, 2025 | 截断重要性采样权重而非 token 更新，让稀有的反思 token 保留梯度。 |
| :material-book-open-variant: [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — Yue et al., 2025 | pass@k 分析：RL 对采样的锐化多于对能力的增加。 |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | 为其最后阶段首创了"RLVR"一词；代码和数据完全开源。 |
| :material-book-open-variant: [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) · [Open-R1](https://github.com/huggingface/open-r1) | 实用的开源实现，可用来复现一个迷你 R1-Zero。 |
