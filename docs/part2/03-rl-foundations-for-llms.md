# 2.3 RL Foundations for LLMs

!!! abstract "What you'll learn"
    - How to view text generation as a Markov decision process (MDP) with a reward at the end
    - The policy gradient theorem for sequences, derived step by step, and the REINFORCE algorithm
    - Why baselines reduce variance without adding bias, and what an advantage is
    - Importance sampling and the probability ratio that lets us reuse samples
    - The PPO clipped objective term by term, value functions, GAE, and the KL penalty to a reference model

## Why this matters / the problem

SFT ([Chapter 2.2](02-supervised-fine-tuning.md)) can only imitate. It needs someone to write the ideal response. But for many goals we can more easily **score** a response than write one: a human can say which of two answers is better, a program can check whether a math answer is correct or whether code passes its tests. We want an algorithm that takes a *score* for a whole generated response and adjusts the model to produce higher-scoring responses.

That is reinforcement learning (RL). Every algorithm in the rest of this part (RLHF with PPO in [2.4](04-rlhf.md), DPO in [2.5](05-direct-preference-optimization.md), GRPO in [2.7](07-rl-with-verifiable-rewards.md)) is built from the pieces in this chapter. We build them from scratch, specialized to language models, which makes things simpler than general RL.

## Intuition

Imagine teaching by "hot or cold". The model writes an answer; you only say "good" (+1) or "bad" (0). How should the model change?

- If the answer was good, make *every token choice* in it a bit more likely.
- If it was bad, make them a bit less likely.

That is essentially the whole idea (REINFORCE). The rest of the chapter fixes its problems:

1. **Noise.** One lucky answer pushes all its tokens up, including the bad ones. Fix: compare against an expected score (a **baseline**) so only *better-than-expected* answers are reinforced.
2. **Sample efficiency.** Generating answers is expensive; we want to take several gradient steps per batch of generated answers. Fix: **importance sampling** with a **clipped** ratio (PPO) so we do not move too far from the policy that produced the samples.
3. **Drift.** Optimizing a score too hard leads the model to exploit flaws in the scorer and to forget how to write normal text. Fix: a **KL penalty** that keeps the model close to a reference model.

## How it works

### Generation as an MDP

A Markov decision process has states, actions, transitions and rewards. For an autoregressive LLM:

| MDP concept | LLM meaning |
|---|---|
| State $s_t$ | The prompt plus the tokens generated so far: $s_t = (x, y_1, \dots, y_{t-1})$ |
| Action $a_t$ | The next token $y_t$, chosen from a vocabulary of size $V$ |
| Policy $\pi_\theta(a_t \mid s_t)$ | The LLM's next-token distribution (softmax of the logits) |
| Transition | Deterministic: $s_{t+1} = s_t$ with $y_t$ appended |
| Episode | One full response $y = (y_1,\dots,y_T)$, ending at the end-of-turn token or a length limit |
| Reward | Usually a single number $R(x, y)$ given at the end: $r_t = 0$ for $t < T$ and $r_T = R(x,y)$ |

The probability of a whole response factorizes over tokens:

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t}), \qquad
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid s_t).
$$

In words: the log-probability of a response is the sum of the log-probabilities of its tokens, which the model computes in one forward pass.

Our goal is to maximize the expected reward over prompts $x$ drawn from a prompt set $\mathcal{D}$:

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\; \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big].
$$

The difficulty: $y$ is *sampled* (a discrete choice), so we cannot backpropagate through the sampling step. The reward function may also be a non-differentiable program.

### The policy gradient theorem for sequences

Fix one prompt $x$ and write $p(y) = \pi_\theta(y \mid x)$. The expectation is a (huge) sum over all possible responses:

$$
J_x(\theta) = \sum_{y} p_\theta(y)\, R(x,y).
$$

Take the gradient. $R$ does not depend on $\theta$, so only $p_\theta$ is differentiated:

$$
\nabla_\theta J_x = \sum_y \nabla_\theta p_\theta(y)\, R(x,y).
$$

Now use the **log-derivative trick**: $\nabla_\theta p = p \,\nabla_\theta \log p$ (because $\nabla \log p = \nabla p / p$). Substituting:

$$
\nabla_\theta J_x = \sum_y p_\theta(y)\, \nabla_\theta \log p_\theta(y)\, R(x,y)
= \mathbb{E}_{y \sim \pi_\theta}\big[ R(x,y)\, \nabla_\theta \log \pi_\theta(y \mid x) \big].
$$

Finally expand the sequence log-probability into tokens:

$$
\boxed{\;\nabla_\theta J = \mathbb{E}_{x,\, y \sim \pi_\theta}\Big[ R(x,y) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid s_t) \Big]\;}
$$

In words: the gradient of expected reward is the average, over sampled responses, of "reward times the gradient of the log-probability of every token in the response". It is an expectation, so we can estimate it by sampling responses from the current model; we never differentiate through sampling or through $R$.

If rewards arrive at intermediate steps, a refinement says token $t$ should only be credited with rewards that come *after* it (the "reward-to-go" $\sum_{t' \ge t} r_{t'}$), because an action cannot affect past rewards. With a single terminal reward, the reward-to-go is $R$ for every token, which is why the boxed formula is already the reward-to-go form.

### REINFORCE

The simplest algorithm turns the formula into a loss whose gradient is the estimator. For a batch of $N$ sampled responses:

$$
\mathcal{L}_{\text{REINFORCE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} R(x_i, y_i) \sum_{t=1}^{T_i} \log \pi_\theta(y_{i,t} \mid s_{i,t}).
$$

Compare with SFT: it is a **reward-weighted** negative log-likelihood on the model's *own* samples. A response with reward 0 contributes nothing; a response with reward 1 is trained on like an SFT example. (When rewards are only 0 or 1, REINFORCE is literally "SFT on your own successful samples", a method called rejection-sampling fine-tuning or expert iteration, used by STaR in [Chapter 2.6](06-reasoning-and-test-time-compute.md).)

### Variance and baselines

The REINFORCE estimator is unbiased but very noisy. If every reward is positive (say rewards are between 5 and 6), every sampled response is pushed up, and only the *slight* differences between pushes carry information.

The fix is to subtract a **baseline** $b$ that does not depend on the sampled response:

$$
\nabla_\theta J = \mathbb{E}_{y}\Big[ \big(R(x,y) - b(x)\big) \nabla_\theta \log \pi_\theta(y \mid x) \Big].
$$

Why is this still correct (unbiased)? The extra term has expectation zero:

$$
\mathbb{E}_{y}\big[ b(x)\, \nabla_\theta \log \pi_\theta(y\mid x) \big]
= b(x) \sum_y \pi_\theta(y\mid x)\, \frac{\nabla_\theta \pi_\theta(y\mid x)}{\pi_\theta(y\mid x)}
= b(x)\, \nabla_\theta \underbrace{\sum_y \pi_\theta(y\mid x)}_{=1} = 0.
$$

In words: probabilities always sum to 1, so the average gradient of the log-probability is zero, and any constant times it averages to zero. A good baseline is the expected reward for that prompt, $b(x) \approx \mathbb{E}_{y}[R(x,y)]$; then responses better than average get pushed up and worse ones pushed down.

### Advantage

Generalizing the baseline to every token gives the **advantage**. Define:

- The **value** $V(s_t)$: the expected final reward if we continue sampling from state $s_t$ with the current policy.
- The **action value** $Q(s_t, a_t)$: the expected final reward if we take token $a_t$ in state $s_t$ and then continue with the policy.
- The **advantage** $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$: how much better choosing $a_t$ was than the policy's average choice in that state.

The policy gradient can be written with advantages:

$$
\nabla_\theta J = \mathbb{E}\Big[ \sum_t A(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big].
$$

Different algorithms differ mainly in **how they estimate $A$**: PPO learns a value network and uses GAE (below); GRPO ([Chapter 2.7](07-rl-with-verifiable-rewards.md)) uses $R - \text{mean}$ of a group of samples for the same prompt, giving every token of a response the same advantage.

### Importance sampling and the probability ratio

Generating responses (rollouts) is the expensive part. We would like to sample a batch once with the policy $\pi_{\theta_{\text{old}}}$ and then take several gradient steps, during which $\theta$ moves away from $\theta_{\text{old}}$. The samples are then **off-policy**: they came from a slightly different policy than the one we are improving.

Importance sampling corrects for this. For any function $f$:

$$
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \sum_a \pi_\theta(a) f(a) = \sum_a \pi_{\text{old}}(a) \frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} f(a) = \mathbb{E}_{a \sim \pi_{\text{old}}}\Big[\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} f(a)\Big].
$$

Applied per token, define the **probability ratio**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\big(\log \pi_\theta(a_t\mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t\mid s_t)\big),
$$

and the **surrogate objective** $L(\theta) = \mathbb{E}_{\pi_{\text{old}}}[\rho_t(\theta) A_t]$. At $\theta = \theta_{\text{old}}$, $\rho_t = 1$ and $\nabla \rho_t = \nabla \log \pi_\theta$, so the surrogate's gradient equals the policy gradient. (Strictly, a per-token ratio only corrects the action distribution, not the distribution of states; this approximation is fine while $\theta$ stays close to $\theta_{\text{old}}$, which is exactly what clipping enforces.)

The problem: if $\rho_t$ becomes large, the estimate's variance explodes and a few tokens dominate the update. We need to keep $\pi_\theta$ close to $\pi_{\text{old}}$.

### PPO: the clipped objective

Proximal Policy Optimization (Schulman et al., 2017) maximizes, per token,

$$
L^{\text{CLIP}}_t(\theta) = \min\Big( \rho_t(\theta)\, A_t,\;\; \operatorname{clip}\big(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\, A_t \Big),
$$

where $\epsilon$ (typically 0.1–0.3, often 0.2) sets the trust region and $\operatorname{clip}(z, lo, hi)$ limits $z$ to $[lo, hi]$. Term by term:

- $\rho_t A_t$ is the importance-sampled policy-gradient surrogate.
- $\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t$ is the same, but the ratio cannot leave $[1-\epsilon, 1+\epsilon]$, so its gradient with respect to $\theta$ is zero once $\rho_t$ is outside the range.
- The $\min$ picks the more **pessimistic** of the two.

What that does in each case:

| Case | Effect |
|---|---|
| $A_t > 0$ (good token), $\rho_t > 1+\epsilon$ | Clipped term is smaller → chosen → zero gradient. *Stop increasing a good token's probability once it has grown by more than $\epsilon$.* |
| $A_t > 0$, $\rho_t < 1+\epsilon$ | Unclipped → normal gradient pushes $\rho_t$ up. |
| $A_t < 0$ (bad token), $\rho_t < 1-\epsilon$ | Clipped term is chosen → zero gradient. *Stop decreasing once it has shrunk by more than $\epsilon$.* |
| $A_t < 0$, $\rho_t > 1-\epsilon$ | Unclipped → normal gradient pushes $\rho_t$ down, even if $\rho_t$ is above $1+\epsilon$ (undoing a bad move is never blocked). |

So PPO lets each update move a token's probability by at most a factor of about $1\pm\epsilon$ *in the direction the advantage wants*, then stops pushing. This makes it safe to run several epochs of minibatch updates on one batch of rollouts. The loss we minimize is $-L^{\text{CLIP}}$, averaged over tokens.

### The value function and GAE

PPO estimates advantages with a learned **value network** (the "critic") $V_\phi(s_t)$, usually a copy of the LLM with a scalar head that outputs one number per token position. With a discount factor $\gamma \in [0,1]$ (usually $\gamma = 1$ for LLMs), define the **temporal-difference (TD) error**

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
$$

which says how much better things turned out after one step than the critic predicted (with $V_\phi(s_{T+1}) = 0$ after the episode ends). **Generalized Advantage Estimation** (GAE, Schulman et al., 2015) averages multi-step versions of this with a parameter $\lambda \in [0,1]$:

$$
\hat A_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \text{computed backwards as } \hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}.
$$

In words: $\lambda = 0$ uses only the one-step TD error (low variance, but biased if the critic is wrong); $\lambda = 1$ telescopes to $\hat A_t = \sum_{t'\ge t}\gamma^{t'-t} r_{t'} - V_\phi(s_t)$, the actual return minus the baseline (unbiased, higher variance). LLM RLHF setups often use $\lambda$ close to 1. The critic is trained by regression toward the **returns** $\hat R_t = \hat A_t + V_\phi(s_t)$:

$$
\mathcal{L}_V(\phi) = \tfrac{1}{2}\big(V_\phi(s_t) - \hat R_t\big)^2.
$$

### The KL penalty to a reference model

In LLM RL we almost always keep a frozen copy of the starting model, the **reference policy** $\pi_{\text{ref}}$ (usually the SFT model), and penalize divergence from it. The KL-regularized objective is

$$
\max_\theta\; \mathbb{E}_{x,\, y\sim\pi_\theta}\big[R(x,y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big),
$$

where $\beta > 0$ sets the strength and

$$
\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\Big[\sum_t \log \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}\Big].
$$

Why bother?

1. **The reward is imperfect.** A learned reward model is only accurate near the responses it was trained on. Without a leash, RL finds weird outputs that score highly but are bad (reward hacking, [Chapter 2.4](04-rlhf.md)).
2. **Keep language quality and diversity.** The reference model writes fluent, varied text; staying close preserves that.
3. **Stability.** It discourages the policy from collapsing onto a few high-reward strings.

Two common implementations:

- **KL in the reward** (classic RLHF): use the per-token reward $r_t = -\beta\big(\log\pi_\theta(y_t\mid s_t) - \log\pi_{\text{ref}}(y_t\mid s_t)\big)$ plus $R(x,y)$ at the last token, then compute advantages with GAE as usual.
- **KL in the loss** (GRPO-style): add $\beta \cdot \widehat{\mathrm{KL}}_t$ directly to the per-token loss, often with the low-variance, always-non-negative estimator $\widehat{\mathrm{KL}}_t = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$ (evaluated on the sampled token).

Many 2025 reasoning-RL recipes with verifiable rewards set $\beta$ small or to zero, because a rule-based reward is much harder to hack than a learned one ([Chapter 2.7](07-rl-with-verifiable-rewards.md)).

## Code

### 1. REINFORCE with a baseline on a toy "one-token response" problem

```python
import torch

torch.manual_seed(0)
V = 3                                   # vocabulary {A, B, C}; reward 1 only for token A (id 0)
logits = torch.zeros(V, requires_grad=True)   # the whole "policy" is 3 logits
opt = torch.optim.SGD([logits], lr=0.5)
reward = lambda a: (a == 0).float()

for step in range(50):
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, num_samples=16, replacement=True)  # 16 sampled "responses"
    R = reward(actions)                                                     # [16]
    baseline = R.mean()                                                     # batch-mean baseline
    logp = torch.log_softmax(logits, dim=-1)[actions]                       # [16]
    loss = -((R - baseline).detach() * logp).mean()                         # REINFORCE loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(step, [round(p, 3) for p in probs.tolist()])
# P(A) climbs from 0.333 toward 1.0
```

### 2. GAE and the PPO losses

```python
import torch

def gae(rewards, values, gamma=1.0, lam=0.95):
    """rewards, values: [T] for one response. values[t] = V(s_t).
    Returns advantages [T] and returns [T]."""
    T = rewards.shape[0]
    adv = torch.zeros(T)
    next_value, running = 0.0, 0.0                 # V(s_{T+1}) = 0 after the episode
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values

def ppo_policy_loss(logp_new, logp_old, adv, mask, eps=0.2):
    """All tensors [B, T]; mask is 1 on response tokens. Returns a scalar to MINIMIZE."""
    ratio = torch.exp(logp_new - logp_old)                     # rho_t
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    per_token = -torch.min(unclipped, clipped)                 # negative of L^CLIP
    return (per_token * mask).sum() / mask.sum()

def value_loss(v_new, returns, mask):
    return 0.5 * (((v_new - returns) ** 2) * mask).sum() / mask.sum()

def kl_shaped_rewards(final_reward, logp_policy, logp_ref, beta=0.05):
    """Per-token rewards for one response [T]: -beta*KL at each token, task reward at the end."""
    r = -beta * (logp_policy - logp_ref)
    r[-1] = r[-1] + final_reward
    return r

# tiny check
r = kl_shaped_rewards(1.0, torch.tensor([-1.0, -0.5, -0.2]), torch.tensor([-1.2, -0.5, -0.4]))
adv, ret = gae(r, values=torch.tensor([0.5, 0.6, 0.8]))
print(r, adv, ret)
```

## Worked example

**REINFORCE by hand.** One-token responses, vocabulary $\{A, B, C\}$, logits $z = (0, 0, 0)$, so $p = (\tfrac13, \tfrac13, \tfrac13)$. Reward $R = 1$ for $A$, else 0.

For a softmax policy, $\nabla_z \log p_k = e_k - p$, where $e_k$ is the one-hot vector for token $k$.

*Without a baseline:*

- Sample $A$ (prob. 1/3): gradient $1\cdot(e_A - p) = (\tfrac23, -\tfrac13, -\tfrac13)$.
- Sample $B$ or $C$: reward 0, gradient $0$.
- Expected gradient: $\tfrac13 (\tfrac23, -\tfrac13, -\tfrac13) = (\tfrac29, -\tfrac19, -\tfrac19)$.
- Variance of the first component: it is $\tfrac23$ with prob. $\tfrac13$, else 0. $\mathbb{E}[g^2] = \tfrac13\cdot\tfrac49 = \tfrac{4}{27}$, so $\mathrm{Var} = \tfrac{4}{27} - (\tfrac29)^2 = \tfrac{12}{81} - \tfrac{4}{81} = \tfrac{8}{81} \approx 0.099$.

*With baseline $b = \mathbb{E}[R] = \tfrac13$:*

- Sample $A$: $(1-\tfrac13)(e_A - p) = \tfrac23(\tfrac23, -\tfrac13, -\tfrac13) = (\tfrac49, -\tfrac29, -\tfrac29)$.
- Sample $B$: $(0-\tfrac13)(e_B - p) = -\tfrac13(-\tfrac13, \tfrac23, -\tfrac13) = (\tfrac19, -\tfrac29, \tfrac19)$.
- Sample $C$: by symmetry $(\tfrac19, \tfrac19, -\tfrac29)$.
- Expected gradient: $\tfrac13\big[(\tfrac49+\tfrac19+\tfrac19), \dots\big] = (\tfrac29, -\tfrac19, -\tfrac19)$. **Same as before: no bias.**
- First component takes values $\tfrac49, \tfrac19, \tfrac19$: $\mathbb{E}[g^2] = \tfrac13(\tfrac{16}{81}+\tfrac1{81}+\tfrac1{81}) = \tfrac{6}{81}$, so $\mathrm{Var} = \tfrac{6}{81} - \tfrac{4}{81} = \tfrac{2}{81} \approx 0.025$.

The baseline cut the variance of this component by a factor of 4. Note also that with the baseline, *failed* samples now carry information (push $B$ down), whereas without it they contributed nothing.

**PPO clipping by hand** ($\epsilon = 0.2$):

| $A_t$ | $\rho_t$ | $\rho_t A_t$ | $\operatorname{clip}(\rho_t)A_t$ | min | Gradient flows? |
|---|---|---|---|---|---|
| +2 | 1.1 | 2.2 | 2.2 | 2.2 | yes |
| +2 | 1.5 | 3.0 | 2.4 | 2.4 | no (already up by >20%) |
| −1 | 0.6 | −0.6 | −0.8 | −0.8 | no (already down by >20%) |
| −1 | 1.5 | −1.5 | −1.2 | −1.5 | yes (undo the bad increase) |

## In practice (2025–2026)

- **PPO for RLHF** (InstructGPT, many 2022–2024 chat models) uses a learned value network initialized from the reward model or the SFT model, GAE, a KL penalty and $\epsilon \approx 0.2$. It needs four models in memory; see [Chapter 2.4](04-rlhf.md).
- **Critic-free methods** dominate reasoning RL: GRPO and its descendants replace the value network with a group-mean baseline ([Chapter 2.7](07-rl-with-verifiable-rewards.md)). RLOO (REINFORCE leave-one-out) is a closely related baseline that uses the mean reward of the *other* samples.
- **The ratio is computed in log space** ($\exp(\log\pi_{\text{new}} - \log\pi_{\text{old}})$) for numerical stability, and $\log\pi_{\text{old}}$ is recomputed by the training engine, because the inference engine that generated the rollouts can produce slightly different probabilities (a source of off-policy mismatch discussed in [Chapter 2.7](07-rl-with-verifiable-rewards.md)).
- **One or a few PPO epochs per batch** is typical for LLMs; many reasoning RL runs take exactly one gradient step per batch, in which case $\rho_t = 1$ and clipping only matters for minibatches after the first.

## Common pitfalls / misconceptions

- **"The policy gradient requires a differentiable reward."** No: the log-derivative trick moves the gradient onto $\log\pi_\theta$. The reward can be any black box.
- **Forgetting to detach the advantage/reward.** The weight in front of $\log\pi_\theta$ must be treated as a constant (`.detach()`), or you will backpropagate into the baseline or value network through the policy loss.
- **Baselines that depend on the sampled action** introduce bias. A batch mean is fine (the bias is $O(1/N)$); leave-one-out removes it entirely.
- **Confusing the KL directions.** RLHF penalizes $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, estimated on samples from $\pi_\theta$; this is "mode-seeking": the policy may drop some of the reference's behaviors but should not invent behaviors the reference finds very unlikely.
- **Clipping is not a hard constraint.** Ratios can still exceed $1\pm\epsilon$ (e.g. through effects of other tokens' updates); monitor the "clip fraction" and the approximate KL between $\pi_\theta$ and $\pi_{\text{old}}$.

## Key takeaways

- Treat a response as an episode: state = prefix, action = token, reward at the end.
- Policy gradient: $\nabla J = \mathbb{E}[R \sum_t \nabla\log\pi_\theta(y_t\mid s_t)]$, derived with the log-derivative trick; no gradient through sampling or the reward.
- Subtracting a baseline keeps the gradient unbiased and can cut variance a lot; the advantage generalizes this per token.
- PPO reuses samples with importance ratios and clips them to stay within a trust region of $1\pm\epsilon$.
- A value network + GAE estimates per-token advantages; a KL penalty to a frozen reference model guards against reward hacking and drift.

## Check your understanding

??? question "Why does subtracting a constant baseline not change the expected gradient?"
    Because $\mathbb{E}_{y\sim\pi}[\nabla\log\pi(y)] = \nabla\sum_y\pi(y) = \nabla 1 = 0$, so $\mathbb{E}[b\,\nabla\log\pi] = 0$ for any $b$ that does not depend on $y$.

??? question "With a single terminal reward and no value network, what advantage does REINFORCE-with-baseline assign to each token of a response?"
    The same value for every token: $R(x,y) - b(x)$. All tokens of a good response are pushed up equally; credit assignment within the response is left to averaging over many samples.

??? question "In PPO, a token has $A_t = +1$ and $\rho_t = 1.3$ with $\epsilon=0.2$. What is $L^{\text{CLIP}}_t$ and its gradient?"
    $\min(1.3, 1.2) = 1.2$, from the clipped term, which is constant in $\theta$, so the gradient is zero.

??? question "What does GAE reduce to when $\lambda = 1$ and $\gamma = 1$?"
    $\hat A_t = \sum_{t'\ge t} r_{t'} - V(s_t)$: the actual return from $t$ minus the critic's estimate (Monte Carlo return with a baseline).

??? question "Give two reasons for the KL penalty to the reference model."
    It prevents the policy from exploiting errors in an imperfect reward model by straying into regions where the reward is unreliable, and it preserves the fluency and diversity of the reference model (also improving stability).

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | The clipped-ratio policy gradient with a learned value function; the RL algorithm of classic RLHF. |
| :material-book-open-variant: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015 | Where GAE and the bias–variance role of $\lambda$ come from. |
| :material-book-open-variant: [RLHF Book](https://rlhfbook.com/) — Nathan Lambert | Policy-gradient chapters written specifically for language models. |
