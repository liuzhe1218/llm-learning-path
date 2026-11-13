# 2.5 Direct Preference Optimization (DPO)

!!! abstract "What you'll learn"
    - The closed-form optimal policy of the KL-regularized RLHF objective, derived
    - How that closed form turns a reward model into an "implicit reward" and yields the DPO loss
    - How to implement the DPO loss in a few lines of PyTorch, and what $\beta$ controls
    - The main variants (IPO, KTO, ORPO, SimPO) and how they differ
    - Offline vs online/iterative DPO, and when to choose DPO over RL

## Why this matters / the problem

RLHF-PPO ([Chapter 2.4](04-rlhf.md)) works, but it is heavy: a separate reward model, a value model, online generation inside the training loop, and many sensitive hyperparameters. Rafailov et al. (2023) asked: since the RLHF objective is a known mathematical expression, can we solve part of it analytically and skip the RL?

The answer, **Direct Preference Optimization (DPO)**, trains the policy directly on preference pairs $(x, y_w, y_l)$ with a loss that looks like logistic regression. It needs only the policy and a frozen reference model, no sampling during training, and it is about as easy to run as SFT. It became the default preference-tuning method for open models in 2023–2024 and remains widely used.

## Intuition

The RLHF objective says: "maximize reward, but do not move too far from the reference model". For any reward function, the best policy under that objective has a simple form: take the reference model's probabilities and multiply them by $\exp(\text{reward}/\beta)$, then renormalize. Good responses get boosted, bad ones suppressed.

Turn that around: if the policy is optimal, its **log-probability ratio** to the reference, $\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, *is* (up to scale and a per-prompt constant) the reward. So instead of training a reward model and then a policy, we parametrize the reward *by the policy itself* and fit it to preferences directly. "Your language model is secretly a reward model", in the DPO paper's subtitle.

## How it works

### Step 1: the KL-regularized objective

Notation: $x$ a prompt, $y$ a response, $\pi$ a policy (distribution over responses), $\pi_{\text{ref}}$ the frozen reference (SFT) model, $r(x,y)$ a reward, $\beta > 0$ the KL strength. The RLHF objective ([Chapter 2.3](03-rl-foundations-for-llms.md)) for one prompt is

$$
\max_{\pi}\; \mathbb{E}_{y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

### Step 2: the closed-form optimal policy

Write the KL as an expectation and factor out $-\beta$:

$$
\mathbb{E}_{y\sim\pi}\Big[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]
= -\beta\, \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\exp\!\big(r(x,y)/\beta\big)}\Big].
$$

The denominator $\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ is not a probability distribution, so normalize it. Define the **partition function**

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\exp\!\big(r(x,y)/\beta\big)
$$

and the distribution

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big).
$$

Substituting $\pi_{\text{ref}}\exp(r/\beta) = Z(x)\,\pi^*$ into the objective:

$$
-\beta\, \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\pi^*(y\mid x)} - \log Z(x)\Big]
= -\beta\, \mathrm{KL}\big(\pi\,\|\,\pi^*\big) + \beta\log Z(x).
$$

$Z(x)$ does not depend on $\pi$, and a KL divergence is $\ge 0$ with equality only when the distributions are equal. So the objective is maximized exactly at $\pi = \pi^*$.

In words: **the optimal policy reweights the reference model by the exponentiated reward**, with $\beta$ acting as a temperature. Small $\beta$: rewards dominate, the policy concentrates on the best responses. Large $\beta$: the policy stays near $\pi_{\text{ref}}$.

We cannot use this formula directly to *sample* (computing $Z(x)$ requires summing over all possible responses), but we can use it algebraically.

### Step 3: the implicit reward

Take logs of the optimal-policy formula and solve for $r$:

$$
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
$$

In words: any reward function can be written in terms of its own optimal policy, plus a term $\beta\log Z(x)$ that depends only on the prompt. So if we search over policies $\pi_\theta$, each one defines an **implicit reward**

$$
\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)},
$$

for which it is the optimal policy (up to the per-prompt constant).

### Step 4: plug into Bradley–Terry; $Z$ cancels

The Bradley–Terry model ([Chapter 2.4](04-rlhf.md)) uses only reward *differences* between two responses to the same prompt. The $\beta\log Z(x)$ terms are identical for $y_w$ and $y_l$, so they cancel:

$$
P(y_w\succ y_l\mid x) = \sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

Maximum likelihood over the preference dataset $\mathcal{D}$ gives the **DPO loss**:

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]\;}
$$

In words: increase the log-probability of the chosen response relative to the reference, decrease it for the rejected one, through a sigmoid that saturates once the implicit reward margin is large. It is exactly the reward-model loss of RLHF, with the reward model replaced by the policy's own log-ratio. Training it fits a reward model *and* yields its optimal policy at the same time.

Here $\log\pi_\theta(y\mid x) = \sum_t \log\pi_\theta(y_t\mid x, y_{<t})$ is the summed log-probability of the response tokens, computed with one forward pass and the same masking as SFT ([Chapter 2.2](02-supervised-fine-tuning.md)).

### The gradient: a weighted SFT/unlikelihood update

Let $\hat r_w = \hat r_\theta(x,y_w)$ and $\hat r_l = \hat r_\theta(x,y_l)$. Differentiating:

$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{weight}}\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].
$$

In words: raise the likelihood of $y_w$, lower that of $y_l$, weighted by how *wrong* the implicit reward currently is. Pairs already ranked correctly with a big margin get weight near 0. The weighting is important: the DPO paper reports that a naive "raise chosen, lower rejected" objective without it can degenerate.

### What $\beta$ does

$\beta$ plays two linked roles:

1. **KL strength** in the original objective: larger $\beta$ means the implied optimal policy stays closer to $\pi_{\text{ref}}$.
2. **Scale of the implicit reward**: the sigmoid sees $\beta\times$(log-ratio margin). With large $\beta$, a small change in log-ratio already saturates the sigmoid, so training stops moving the policy early. With small $\beta$, the log-ratios must grow a lot before the loss saturates, so the policy moves further from the reference.

Common values are in the range 0.01–0.5, with 0.1 a frequent default. It is worth sweeping.

### Variants

DPO spawned many variants. Each changes one assumption.

| Method | Data | Reference model? | Key change | Intuition |
|---|---|---|---|---|
| **DPO** | pairs | yes | — | Logistic loss on the log-ratio margin |
| **IPO** (Azar et al., 2023) | pairs | yes | Squared loss pulling the log-ratio margin to a fixed target $\tfrac{1}{2\tau}$ instead of a sigmoid | When preferences are near-deterministic, DPO can push the margin toward infinity and ignore the KL; IPO's finite target keeps it regularized |
| **KTO** (Ethayarajh et al., 2024) | unpaired thumbs-up/down labels | yes | A per-example loss inspired by prospect theory, comparing each response's implicit reward to a reference point (an estimate of the KL) | Works with binary feedback that is much easier to collect than pairs; treats gains and losses asymmetrically |
| **ORPO** (Hong et al., 2024) | pairs | **no** | SFT loss on $y_w$ plus a log-odds-ratio penalty between $y_w$ and $y_l$, using length-normalized likelihoods | One stage instead of SFT then DPO; no reference model to keep in memory |
| **SimPO** (Meng et al., 2024) | pairs | **no** | Implicit reward = length-normalized log-likelihood $\frac{\beta}{\lvert y\rvert}\log\pi_\theta(y\mid x)$, plus a target margin $\gamma$ | Matches the reward to the average log-likelihood that drives generation; removes length bias and the reference model |

The SimPO loss, for concreteness:

$$
\mathcal{L}_{\text{SimPO}} = -\log\sigma\Big(\frac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w\mid x) - \frac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l\mid x) - \gamma\Big),
$$

where $\lvert y\rvert$ is the number of tokens in $y$ and $\gamma > 0$ is a margin.

### Offline vs online / iterative DPO

Standard DPO is **offline**: the pairs are fixed before training, and often were generated by *other* models. This creates a distribution mismatch: the policy is taught to rank responses it would never produce itself, and it gets no feedback on the responses it *does* produce once it changes.

**Online or iterative DPO** closes the loop:

1. Sample several responses per prompt from the *current* policy.
2. Label pairs with a reward model or an LLM judge (best vs worst, say).
3. Run DPO on these fresh pairs; set the new reference to the current policy (or keep the original).
4. Repeat.

This makes DPO behave much more like RL, and evidence through 2024–2025 suggests **online, on-policy** methods (PPO, GRPO, online/iterative DPO) generally outperform purely offline DPO when sampling is affordable.

### When to use DPO vs RL

| Use DPO (or a variant) when… | Use RL (PPO/GRPO) when… |
|---|---|
| You have preference data but no programmatic verifier | You have a verifier (math, code, tests) — see [Chapter 2.7](07-rl-with-verifiable-rewards.md) |
| Compute is limited; you want SFT-like simplicity and stability | You can afford online sampling and want the best final quality |
| The goal is style, tone, helpfulness, safety calibration | The goal is capability that needs exploration (long reasoning, agentic tasks) |
| You need a quick preference-tuning pass after SFT | You need multi-turn or environment interaction |

## Code

```python
import torch, torch.nn.functional as F

def sequence_logprob(model, input_ids, attention_mask, labels):
    """Sum of log pi(token) over response tokens. labels = -100 on prompt/pad (as in SFT).
    input_ids, attention_mask, labels: [B, L] -> returns ([B] summed logp, [B] #tokens)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = (targets != -100)
    logp = torch.log_softmax(logits, -1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1), mask.sum(-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """All inputs: [B] summed log-probs. Returns loss and implicit rewards for logging."""
    r_w = beta * (pi_w - ref_w)                        # implicit reward of chosen
    r_l = beta * (pi_l - ref_l)                        # implicit reward of rejected
    loss = -F.logsigmoid(r_w - r_l).mean()
    return loss, r_w.detach(), r_l.detach()

def ipo_loss(pi_w, pi_l, ref_w, ref_l, tau=0.1):
    h = (pi_w - ref_w) - (pi_l - ref_l)                # log-ratio margin
    return ((h - 1.0 / (2 * tau)) ** 2).mean()

def simpo_loss(pi_w, pi_l, len_w, len_l, beta=2.0, gamma=1.0):
    r_w = beta * pi_w / len_w                          # length-normalized, no reference
    r_l = beta * pi_l / len_l
    return -F.logsigmoid(r_w - r_l - gamma).mean()

def dpo_step(policy, ref, batch, opt, beta=0.1):
    """batch: dict with chosen_/rejected_ input_ids, attention_mask, labels (each [B, L])."""
    pi_w, _ = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
    pi_l, _ = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"])
    with torch.no_grad():                              # reference is frozen
        ref_w, _ = sequence_logprob(ref, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
        ref_l, _ = sequence_logprob(ref, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"])
    loss, r_w, r_l = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
    opt.zero_grad(); loss.backward(); opt.step()
    return {"loss": loss.item(), "reward_acc": (r_w > r_l).float().mean().item(),
            "margin": (r_w - r_l).mean().item(), "chosen_logp": pi_w.mean().item()}
```

Build the batches with `build_example` from [Chapter 2.2](02-supervised-fine-tuning.md): chosen and rejected share the prompt and differ only in the assistant turn. Since the reference is frozen, a common trick is to precompute `ref_w` and `ref_l` once for the whole dataset, so only the policy needs to be in GPU memory during training.

## Worked example

**The optimal policy and $\beta$.** A prompt has two possible responses with $\pi_{\text{ref}} = (0.5, 0.5)$ and rewards $r = (1, 0)$.

- $\beta = 0.5$: $\pi^* \propto (0.5e^{2}, 0.5e^{0}) = (3.69, 0.5)$, so $\pi^*(y_1) = 3.69/4.19 \approx 0.88$.
- $\beta = 2$: $\pi^* \propto (0.5e^{0.5}, 0.5) = (0.824, 0.5)$, so $\pi^*(y_1) \approx 0.824/1.324 \approx 0.62$.

Larger $\beta$ keeps the optimal policy closer to the reference's 50/50.

**One DPO loss evaluation.** For a pair, summed log-probs are $\log\pi_\theta(y_w) = -12$, $\log\pi_{\text{ref}}(y_w) = -13$, $\log\pi_\theta(y_l) = -15$, $\log\pi_{\text{ref}}(y_l) = -14$, and $\beta = 0.1$.

- Implicit rewards: $\hat r_w = 0.1\times(-12+13) = 0.1$; $\hat r_l = 0.1\times(-15+14) = -0.1$.
- Margin $0.2$; $\sigma(0.2) \approx 0.550$; loss $= -\log 0.550 \approx 0.598$.
- At initialization ($\pi_\theta = \pi_{\text{ref}}$), every margin is 0 and the loss is $\log 2 \approx 0.693$: a quick sanity check for your implementation.
- Gradient weight $\sigma(\hat r_l - \hat r_w) = \sigma(-0.2) \approx 0.450$.
- With $\beta = 1$ the same log-probs give margin 2 and loss $-\log\sigma(2) \approx 0.127$: the loss is nearly saturated, so training would push this pair much less.

## In practice (2025–2026)

- **DPO-family methods remain the workhorse for preference tuning** in open recipes (Zephyr used DPO on UltraFeedback with AI-judged preferences; Tülu 3 uses a DPO stage between SFT and RLVR). RL is then used for verifiable domains.
- **Data matters more than the variant.** Many comparisons find that the choice among DPO/IPO/SimPO/KTO matters less than the quality and on-policy-ness of the pairs. Pairs sampled from the model being trained and judged by a strong judge work best.
- **Watch the chosen log-probability.** A well-known DPO behavior is that the log-probabilities of *both* chosen and rejected responses decrease during training while the margin grows; if the chosen log-prob falls a lot, outputs can degrade. Mixing in an SFT loss on chosen responses (as ORPO does by construction) or using a larger $\beta$ helps.
- **Length exploitation.** DPO's summed log-probability can favor longer responses; length-normalized variants (SimPO) or length-controlled data address this.

## Common pitfalls / misconceptions

- **"DPO has no reward model."** It has an *implicit* one, $\beta\log(\pi_\theta/\pi_{\text{ref}})$; you can log its accuracy and margins just like an RM.
- **"DPO is equivalent to RLHF in practice."** It optimizes the same objective *only* on the support of the data; offline, it never sees or corrects the policy's own new outputs.
- **Using the wrong reference.** The reference must be the model the policy is initialized from (usually the SFT model); using a different one changes the implied objective.
- **Forgetting masks.** Summing log-probs over prompt tokens (identical for chosen and rejected) adds noise and is wasted compute; mask them out.
- **Too many epochs.** DPO overfits preference data quickly (training reward accuracy → 100%); 1–3 epochs is typical.

## Key takeaways

- The KL-regularized RLHF objective has the closed-form optimum $\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$.
- Inverting it gives an implicit reward $\beta\log(\pi_\theta/\pi_{\text{ref}})$; in Bradley–Terry the intractable $Z(x)$ cancels.
- DPO loss: $-\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}]\big)$ — two forward passes per model per pair, no sampling.
- $\beta$ controls both KL strength and how fast the loss saturates.
- Variants change the data format (KTO), loss shape (IPO), or remove the reference (ORPO, SimPO); online/iterative DPO recovers much of RL's advantage.

## Check your understanding

??? question "Why does the partition function $Z(x)$ disappear from the DPO loss?"
    The implicit reward includes $\beta\log Z(x)$, which depends only on the prompt. Bradley–Terry uses the difference of rewards of two responses to the same prompt, so the term cancels.

??? question "What is the DPO loss at the very start of training, and why?"
    $\log 2 \approx 0.693$. The policy is initialized to the reference, so every log-ratio is 0, the margin is 0, and $-\log\sigma(0) = \log 2$.

??? question "Which DPO variant would you choose if you only have thumbs-up/thumbs-down feedback on single responses?"
    KTO, which is designed for unpaired binary labels.

??? question "Why might online/iterative DPO beat offline DPO?"
    Offline pairs come from a fixed (often different) distribution, so the policy gets no feedback on its own current outputs. Sampling from the current policy and labeling those samples keeps training on-policy and lets the method correct its own mistakes, like RL.

??? question "Two runs use $\beta=0.05$ and $\beta=0.5$. Which policy will drift further from the reference?"
    $\beta = 0.05$: the KL penalty is weaker and the sigmoid saturates only after a much larger log-ratio margin.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 | The derivation in this chapter, plus experiments and the gradient analysis. |
| :material-book-open-variant: [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036) — Azar et al., 2023 | Unifies RLHF and DPO and introduces IPO to avoid DPO's overfitting to deterministic preferences. |
| :material-book-open-variant: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024 | Preference tuning from unpaired thumbs-up/down labels. |
| :material-book-open-variant: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024 | Merges SFT and preference loss into one stage, no reference model. |
| :material-book-open-variant: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024 | Length-normalized, reference-free reward with a margin; strong and cheap. |
| :material-book-open-variant: [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) — Tunstall et al., 2023 | A practical recipe: SFT then DPO on AI-judged UltraFeedback preferences. |
