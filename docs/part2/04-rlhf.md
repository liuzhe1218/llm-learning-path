# 2.4 Reinforcement Learning from Human Feedback (RLHF)

!!! abstract "What you'll learn"
    - How human preference data is collected and why comparisons are used instead of scores
    - The Bradley–Terry model, derived, and how to train a reward model with it (with code)
    - The full RLHF-PPO loop with its four models, and what it costs in memory
    - Reward hacking and reward-model over-optimization, and how the KL penalty helps
    - RLAIF and Constitutional AI: replacing human labels with AI feedback

## Why this matters / the problem

After SFT ([Chapter 2.2](02-supervised-fine-tuning.md)) a model answers in the right format, but its answers are only as good as the demonstrations, and it has no sense of *which* of its possible answers people prefer. Qualities like "helpful but concise", "honest about uncertainty", "polite refusal of a harmful request" are hard to demonstrate exhaustively but easy to recognize.

RLHF turns that recognition into a training signal. It was the key ingredient that made InstructGPT and ChatGPT feel qualitatively different from GPT-3. It combines the RL machinery of [Chapter 2.3](03-rl-foundations-for-llms.md) with a **learned reward model** that stands in for a human judge.

## Intuition

We cannot ask a human to score every one of the millions of responses generated during RL. So we:

1. Ask humans to compare a modest number of response pairs (tens to hundreds of thousands).
2. Train a **reward model** (RM) to predict those human judgments: it reads a prompt and a response and outputs a number, higher = more preferred.
3. Run RL against the reward model, which can score unlimited responses instantly.
4. Keep the policy on a leash (KL penalty) because the reward model is only a *proxy* for human judgment and will be wrong on unfamiliar responses.

```text
 prompts ─► SFT model samples 2+ responses ─► humans pick the better one
                                                   │
                                                   ▼
                                  reward model r_φ(x, y)  (Bradley–Terry)
                                                   │
 prompts ─► policy π_θ generates ─► r_φ scores ─► PPO update (with KL to π_ref)
```

## How it works

### Collecting preference data

For each prompt $x$, the SFT model (or several models) generates two or more responses. An annotator sees them and indicates which is better, sometimes with a strength ("much better", "slightly better") or a tie. Each judgment becomes a triple $(x, y_w, y_l)$: the **chosen** ("winner") response $y_w$ and the **rejected** ("loser") $y_l$. InstructGPT asked labelers to rank between 4 and 9 responses per prompt; a ranking of $K$ responses yields $\binom{K}{2}$ pairwise comparisons.

Why comparisons and not absolute scores (e.g. "rate 1–10")? Different annotators use scales differently (one person's 7 is another's 5), and the same person drifts over time. Relative judgments are much more consistent. Annotators are given guidelines (e.g. helpfulness, truthfulness, harmlessness) and agreement between annotators is far from perfect, which sets a ceiling on how accurate any reward model can be.

### The Bradley–Terry model, derived

We need a probabilistic model linking a scalar "quality" for each response to the chance a human prefers it. Assume each response $y$ for prompt $x$ has a latent strength $\exp(r(x,y))$, where $r$ is a real number (the **reward**), and that the probability of $y_w$ beating $y_l$ is proportional to its strength:

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}.
$$

Divide numerator and denominator by $\exp(r(x,y_w))$:

$$
P(y_w \succ y_l \mid x) = \frac{1}{1 + \exp\big(-(r(x,y_w) - r(x,y_l))\big)} = \sigma\big(r(x,y_w) - r(x,y_l)\big),
$$

where $\sigma(z) = 1/(1+e^{-z})$ is the logistic sigmoid. In words: the preference probability depends only on the **difference** of rewards, passed through a sigmoid. A reward gap of 0 means 50/50; a gap of 2 means $\sigma(2) \approx 0.88$.

Two consequences:

- Rewards are only defined **up to an additive constant per prompt**: adding $c(x)$ to all responses of a prompt leaves every probability unchanged. RM outputs are therefore usually normalized before RL.
- This is the same model behind Elo ratings and Chatbot Arena ([Chapter 2.1](01-why-post-training.md)), with a different scale.

### Training the reward model

The reward model $r_\phi$ is usually the SFT model with its vocabulary output layer replaced by a linear head mapping the final hidden state to one scalar; the reward is read at the **last token** of the response. Shapes: hidden states `[B, L, d_model]` → head → `[B, L, 1]` → take the value at the last non-padding position → `[B]`.

We fit $\phi$ by maximum likelihood of the observed preferences under Bradley–Terry, i.e. minimizing the negative log-likelihood over the preference dataset $\mathcal{D}$:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big].
$$

This is binary logistic regression on the reward difference. Its gradient with $\Delta = r_\phi(x,y_w) - r_\phi(x,y_l)$ is

$$
\nabla_\phi \mathcal{L}_{\text{RM}} = -\big(1 - \sigma(\Delta)\big)\big(\nabla_\phi r_\phi(x,y_w) - \nabla_\phi r_\phi(x,y_l)\big),
$$

so pairs the model already ranks correctly with a big margin ($\sigma(\Delta)\approx1$) contribute almost nothing, and misranked pairs contribute the most. Reward models are typically trained for a single epoch, since they overfit quickly. Their quality is measured by **pairwise accuracy** on held-out comparisons; because humans disagree with each other, accuracies are well below 100%.

### The RLHF-PPO loop

With the reward model frozen, we optimize the KL-regularized objective from [Chapter 2.3](03-rl-foundations-for-llms.md):

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

Here $\pi_\theta$ is the policy being trained (initialized from SFT), $\pi_{\text{ref}}$ is the frozen SFT model, $r_\phi$ the frozen reward model, and $\beta$ the KL coefficient. PPO needs **four models**:

| Model | Role | Trained? | Initialized from |
|---|---|---|---|
| Policy (actor) $\pi_\theta$ | Generates responses; is optimized | Yes | SFT model |
| Reference $\pi_{\text{ref}}$ | Computes $\log\pi_{\text{ref}}$ for the KL penalty | No (frozen) | SFT model |
| Reward model $r_\phi$ | Scores complete responses | No (frozen) | SFT model + scalar head, trained on preferences |
| Value model (critic) $V_\psi$ | Predicts expected future reward per token, for GAE | Yes | Reward model or SFT model |

One iteration:

1. **Rollout.** Sample a batch of prompts; generate a response from $\pi_\theta$ for each (this is inference: slow, autoregressive).
2. **Score.** Compute $r_\phi(x,y)$ for each full response; compute per-token $\log\pi_\theta$ (as $\log\pi_{\text{old}}$), $\log\pi_{\text{ref}}$, and $V_\psi(s_t)$ with forward passes.
3. **Shape rewards.** Per token: $r_t = -\beta(\log\pi_{\text{old}}(y_t\mid s_t) - \log\pi_{\text{ref}}(y_t\mid s_t))$, plus $r_\phi(x,y)$ added at the last token.
4. **Advantages.** Run GAE over each response to get $\hat A_t$ and returns $\hat R_t$; normalize ("whiten") advantages across the batch to mean 0, std 1.
5. **Update.** For a few epochs over minibatches: minimize the PPO clipped policy loss for $\pi_\theta$ and the value loss for $V_\psi$.
6. Repeat with fresh rollouts.

InstructGPT additionally mixed in a pre-training language-modeling loss ("PPO-ptx") to reduce regressions on standard NLP benchmarks, the so-called **alignment tax**.

### Memory implications

Four large models must be resident, two of them with optimizer state. With mixed-precision Adam, a trained model needs roughly 16 bytes per parameter (bf16 weights 2 + bf16 grads 2 + fp32 master weights 4 + Adam moments 8), while a frozen inference model needs 2 bytes per parameter in bf16. Activations and the KV cache for generation come on top. This is why RLHF-PPO is engineered with model sharding (ZeRO/FSDP), sometimes with the reward and value models smaller than the policy (InstructGPT used a 6B reward model for its 175B policy), and why critic-free methods (GRPO, [Chapter 2.7](07-rl-with-verifiable-rewards.md)) and reward-model-free methods (DPO, [Chapter 2.5](05-direct-preference-optimization.md)) became popular.

### Reward hacking and over-optimization

The reward model is a proxy. **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure. RL is an extremely good optimizer, so it finds and exploits the proxy's flaws:

- **Length**: RMs often slightly prefer longer answers; the policy learns to pad.
- **Style over substance**: confident tone, bullet lists, flattering the user ("sycophancy").
- **Adversarial gibberish**: in extreme cases, odd token sequences that the RM scores highly.

Gao et al. (2022) measured this cleanly. They used a large "gold" reward model as stand-in for humans, trained smaller proxy RMs on its labels, and optimized policies against the proxies. As the policy moved further from its initialization (measured by $d = \sqrt{\mathrm{KL}(\pi\|\pi_{\text{init}})}$), the proxy reward kept rising, but the gold reward first rose and then **fell**. They fit simple functional forms, $R_{\text{gold}}(d) = d(\alpha - \beta d)$ for best-of-$n$ sampling and $R_{\text{gold}}(d) = d(\alpha - \beta\log d)$ for RL, and found that larger reward models over-optimize more slowly.

```text
 reward
   │        proxy RM score  ───────────────/
   │                              ____/
   │                      ___/‾‾‾‾  ‾‾‾‾‾‾\___  true (gold) quality
   │                ___/                       ‾‾‾\___
   │          ___/
   │    ___/
   └───────────────────────────────────────────────────► KL from initial policy
              ↑ sweet spot: stop here (or set β so you stay here)
```

Defenses: the KL penalty (larger $\beta$ = stay closer), early stopping based on held-out human or strong-judge evaluation, larger and better RMs, ensembles of RMs, length penalties or length-controlled RM training, and periodically collecting new preferences on the *current* policy's outputs (iterated RLHF) so the RM stays accurate where the policy is.

### RLAIF and Constitutional AI

Human labels are slow and expensive, and for harmful content they are unpleasant to produce. **RLAIF** (RL from AI feedback) uses a strong LLM as the labeler: it is shown the prompt and two responses and asked which is better, and its preferences train the reward model. Lee et al. (2023) found RLAIF achieved results comparable to RLHF on tasks such as summarization and helpful dialogue.

**Constitutional AI** (Bai et al., Anthropic, 2022) structures AI feedback around a written list of principles, the **constitution** (e.g. "choose the response that is least harmful"). It has two phases:

1. **Supervised phase (critique and revise).** The model generates a response to a potentially harmful prompt, then is asked to critique it according to a principle and to rewrite it. SFT on the revised responses.
2. **RL phase (RLAIF).** The model compares pairs of responses according to randomly sampled principles; these AI preferences train a harmlessness reward model, which is used for RL (combined with human helpfulness preferences).

The result was a model that was more harmless without becoming as evasive, and whose values are specified by an explicit, inspectable document rather than implicitly by thousands of individual labels.

## Code

### 1. A reward model and its Bradley–Terry loss

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Transformer backbone + scalar head; reward read at the last non-pad token."""
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, dtype=torch.float32)  # no LM head
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, d]
        L = attention_mask.shape[1]                              # index of last real token; works for
        last = L - 1 - attention_mask.flip(1).argmax(dim=1)     # left, right or mixed padding
        h_last = h[torch.arange(h.size(0)), last]                # [B, d]
        return self.head(h_last.to(self.head.weight.dtype)).squeeze(-1)  # [B]

def bt_loss(r_chosen, r_rejected):
    """-log sigma(r_w - r_l), averaged. logsigmoid is numerically stable."""
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

### 2. One RLHF-PPO iteration (the data flow)

This function shows exactly what the four models compute. It reuses `gae`, `ppo_policy_loss` and `value_loss` from [Chapter 2.3](03-rl-foundations-for-llms.md); `value_model(ids, mask)` is assumed to return per-token values `[B, L]` (a backbone plus a scalar head on every position).

```python
import torch

def token_logprobs(model, ids, mask):
    """log pi(token_t | prefix) for t = 1..L-1. ids, mask: [B, L] -> [B, L-1]."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

def rlhf_ppo_iteration(policy, ref, reward_model, value_model, tok, prompts,
                       opt_pi, opt_v, beta=0.05, ppo_epochs=2, max_new=64):
    tok.padding_side = "left"                                    # left-pad prompts for generation
    enc = tok(prompts, return_tensors="pt", padding=True)
    P = enc.input_ids.shape[1]
    with torch.no_grad():
        ids = policy.generate(**enc, max_new_tokens=max_new, do_sample=True, top_p=1.0,
                              pad_token_id=tok.pad_token_id)     # [B, P+T]
        mask = (ids != tok.pad_token_id).long(); mask[:, :P] = enc.attention_mask  # assumes pad != eos id
        resp = torch.zeros_like(ids[:, 1:], dtype=torch.float)   # 1 on response tokens (targets)
        resp[:, P - 1:] = mask[:, P:].float()
        logp_old = token_logprobs(policy, ids, mask)             # [B, L-1]
        logp_ref = token_logprobs(ref, ids, mask)
        values = value_model(ids, mask)[:, :-1]                  # V(s_t) aligned with targets
        score = reward_model(ids, mask)                          # [B], read at last real token
        advs, rets = torch.zeros_like(resp), torch.zeros_like(resp)
        for i in range(ids.shape[0]):                            # per-response reward shaping + GAE
            idx = resp[i].nonzero().squeeze(-1)
            r = -beta * (logp_old[i, idx] - logp_ref[i, idx])
            r[-1] += score[i]
            a, ret = gae(r, values[i, idx])
            advs[i, idx], rets[i, idx] = a, ret
        sel = resp.bool()
        advs[sel] = (advs[sel] - advs[sel].mean()) / (advs[sel].std() + 1e-8)  # whiten
    for _ in range(ppo_epochs):                                  # reuse the same rollouts
        logp_new = token_logprobs(policy, ids, mask)
        loss_pi = ppo_policy_loss(logp_new, logp_old, advs, resp)
        loss_v = value_loss(value_model(ids, mask)[:, :-1], rets, resp)
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
        opt_v.zero_grad(); loss_v.backward(); opt_v.step()
    return score.mean().item()
```

Real implementations (TRL, OpenRLHF, verl) add minibatching, gradient accumulation, an adaptive $\beta$ controller, truncation at the end-of-turn token, and distributed generation with a fast inference engine, but the data flow is the same.

## Worked example

**Bradley–Terry numbers.** The RM gives $r(x, y_w) = 1.2$ and $r(x, y_l) = 0.4$.

- $\Delta = 0.8$, $P(y_w\succ y_l) = \sigma(0.8) = 1/(1+e^{-0.8}) = 1/(1+0.449) \approx 0.690$.
- Loss: $-\log 0.690 \approx 0.371$.
- Gradient weight: $1 - \sigma(\Delta) \approx 0.310$. If the RM had the pair backwards ($\Delta = -0.8$), $\sigma = 0.310$, the loss would be $-\log 0.310 \approx 1.171$ and the gradient weight $0.690$: more than twice the push.

**Memory for RLHF-PPO on a 7B model** (all four models 7B, bf16, mixed-precision Adam, ignoring activations and KV cache):

| Model | Bytes / param | Memory |
|---|---|---|
| Policy (trained) | 16 | $7\text{B}\times16 = 112$ GB |
| Value (trained) | 16 | 112 GB |
| Reference (frozen) | 2 | 14 GB |
| Reward (frozen) | 2 | 14 GB |
| **Total** | | **≈ 252 GB** |

Compare SFT of the same model: ≈112 GB. RLHF-PPO needs well over twice the memory before counting generation, which is why it requires a multi-GPU node even for a 7B model, and why dropping the value model (GRPO) or both reward and value models (DPO) is attractive.

## In practice (2025–2026)

- **Preference data is mostly synthetic or mixed.** Many open pipelines (e.g. UltraFeedback-style data, Tülu 3) generate responses from many models and have a strong LLM judge them; humans focus on hard or safety-critical cases.
- **Reward models are still central** for open-ended qualities (helpfulness, style, safety) where there is no verifier. They are increasingly *generative*: an LLM that writes a critique before giving a score, sometimes conditioned on a rubric.
- **Hybrid rewards.** Modern RL stages combine verifiable rewards (math, code; [Chapter 2.7](07-rl-with-verifiable-rewards.md)) with RM rewards for general chat, often in separate stages.
- **PPO vs alternatives.** PPO with a critic remains in use at some labs, but many open recipes moved to DPO for preference tuning and to critic-free GRPO-style methods for RL.
- **Practical tips** that recur across reports: normalize rewards (per prompt or batch), whiten advantages, monitor KL and response length every step, keep a held-out set judged by humans or a strong model, initialize the critic from the RM, and use a fairly small learning rate (smaller than SFT) for the policy.

## Common pitfalls / misconceptions

- **Treating RM scores as absolute.** They are only meaningful as differences for the same prompt; compare across prompts with care.
- **Reading the reward from the wrong token.** With left/right padding mixups, the "last token" may be padding; the RM then learns nothing useful.
- **Celebrating rising reward.** Rising proxy reward with rising length or KL is often hacking. Always check samples and an independent evaluation.
- **"RLHF teaches facts."** It mostly shapes behavior; it can even reduce calibration (models become more confident) and increase sycophancy if raters like agreement.
- **Ignoring annotator disagreement.** A noisy preference set caps RM accuracy and the RM's errors become the policy's targets.

## Key takeaways

- Humans compare responses; comparisons are more consistent than absolute ratings.
- Bradley–Terry: $P(y_w \succ y_l) = \sigma(r_w - r_l)$; the RM is trained with $-\log\sigma(r_w - r_l)$.
- RLHF-PPO maximizes RM reward minus $\beta\,\mathrm{KL}$ to the SFT model using four models (policy, reference, reward, value) — memory-hungry.
- Optimizing a proxy too hard lowers true quality (over-optimization); KL penalties, better RMs and early stopping mitigate it.
- RLAIF and Constitutional AI replace human labels with AI judgments guided by written principles.

## Check your understanding

??? question "Why is the reward model's output only defined up to a per-prompt constant?"
    Bradley–Terry probabilities depend only on reward differences between responses to the same prompt; adding $c(x)$ to all of them leaves every $\sigma(r_w - r_l)$ unchanged, so the data cannot pin down the constant.

??? question "Which of the four PPO models are updated by gradients, and which are only run forward?"
    The policy and the value model are trained; the reference model and the reward model are frozen and only run forward.

??? question "During RLHF the RM score rises steadily, but human raters say outputs got worse and longer. What is happening and what would you change?"
    Reward hacking / over-optimization: the policy exploits RM biases such as a preference for length. Increase $\beta$, stop earlier (select checkpoints by independent evaluation), add a length penalty or length-controlled RM, retrain the RM on the current policy's outputs, or use an ensemble of RMs.

??? question "What are the two phases of Constitutional AI?"
    A supervised phase where the model critiques and revises its own responses according to constitutional principles and is fine-tuned on the revisions; and an RL phase where AI preference labels, guided by principles, train a reward model used for RL (RLAIF).

??? question "If $r_w - r_l = 0$, what is the loss and the gradient weight?"
    $\sigma(0) = 0.5$, loss $= -\log 0.5 = \ln 2 \approx 0.693$, gradient weight $1 - 0.5 = 0.5$.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al., 2017 | The origin of RLHF: learn a reward model from pairwise comparisons, then optimize it with RL. |
| :material-book-open-variant: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020 | RLHF on summarization; the reward-model + KL-penalty recipe at scale. |
| :material-star: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 | The SFT → RM → PPO pipeline behind ChatGPT, with data-collection details. |
| :material-book-open-variant: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao et al., 2022 | Quantifies Goodhart: gold reward rises then falls as the policy optimizes a proxy. |
| :material-star: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (Anthropic), 2022 | Critique-and-revise SFT plus RLAIF guided by a written constitution. |
| :material-book-open-variant: [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) — Lee et al., 2023 | AI-labeled preferences can match human labels on several tasks. |
| :material-star: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 | The RL algorithm inside RLHF (derived in [Chapter 2.3](03-rl-foundations-for-llms.md)). |
