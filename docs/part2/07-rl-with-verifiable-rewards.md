# 2.7 RL with Verifiable Rewards (RLVR)

!!! abstract "What you'll learn"
    - What verifiable rewards are and how to write them (math answer checks, unit tests, format rewards)
    - GRPO derived from PPO: a group baseline replaces the value network, with a worked numeric example and code
    - The DeepSeek-R1-Zero and R1 recipes: pure RL, cold start, multi-stage RL, rejection-sampling SFT, distillation
    - The main failure modes of GRPO and their fixes: DAPO, Dr. GRPO, GSPO, CISPO
    - What RLVR does and does not add (the pass@k debate), and why rollout generation dominates RL infrastructure

## Why this matters / the problem

RLHF ([Chapter 2.4](04-rlhf.md)) optimizes a *learned* reward model, which is a fragile proxy: push too hard and the policy hacks it. That limited how long RL could run and how much it could change a model.

For many valuable tasks we do not need a learned judge at all. A math problem has a known final answer; a coding problem has unit tests; a tool call either succeeds or fails. A program can check these exactly. **RL with verifiable rewards (RLVR)** uses such checks as the reward. Because the reward is (nearly) impossible to fool by style, the policy can be optimized for thousands of steps, and in 2025 this produced the long-chain-of-thought reasoning models of [Chapter 2.6](06-reasoning-and-test-time-compute.md). DeepSeek-R1 made this recipe public and it now underlies most open reasoning models.

## Intuition

Give the model a hard math problem, let it try 16 times, check each final answer, and reinforce the attempts that got it right *relative to the others*. Problems it always solves or never solves teach nothing (no contrast); problems it solves sometimes are where learning happens. Over many problems, whatever habits make success more likely (writing out steps, checking work, backtracking after a contradiction) get reinforced, even though no one ever demonstrated them. Response length grows on its own because thinking longer pays off.

## How it works

### Verifiable rewards

A verifiable reward is a deterministic function $R(x, y)$ of the prompt and the full response. Typical components:

| Reward | Check | Notes |
|---|---|---|
| **Math accuracy** | Extract the final answer (e.g. inside `\boxed{}`), normalize, compare with the reference; or check symbolic equivalence | Normalization matters: `1/2`, `0.5`, `\frac{1}{2}` |
| **Code correctness** | Run hidden unit tests in a sandbox; reward = 1 if all pass (or the fraction passed) | Needs timeouts, isolation, resource limits |
| **Format** | Response contains the required structure (e.g. reasoning inside `<think>...</think>` then an answer) | Small weight; it shapes output parsing, not correctness |
| **Instruction constraints** | "Exactly 3 bullet points", "under 50 words", "respond in French" (IFEval-style) | Programmatic checks |
| **Agentic outcome** | The environment's final state (tests pass in the repo, the web task succeeded) | See [Chapter 2.9](09-agents-tool-use-and-safety.md) |

Rewards are usually binary (0/1) or a small sum of components. The verifier must be robust: any loophole (accepting several answers, crashing tests counted as passes, editable test files) will be found and exploited by RL.

### From PPO to GRPO

Recall PPO ([Chapter 2.3](03-rl-foundations-for-llms.md)): per-token clipped surrogate with advantages from a learned value network via GAE. The value network is as large as the policy, doubles the trained-model memory, and is hard to train well when the only reward arrives at the end of a 10,000-token response.

**Group Relative Policy Optimization (GRPO)**, introduced in DeepSeekMath (Shao et al., 2024), replaces the value network with a Monte Carlo baseline from a **group** of samples for the *same* prompt:

1. For prompt $x$, sample a group of $G$ responses $\{y_1,\dots,y_G\}$ from $\pi_{\theta_{\text{old}}}$.
2. Score each: rewards $r_1,\dots,r_G$.
3. Advantage of response $i$, shared by all its tokens:

$$
\hat A_{i} = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon_{\text{num}}}.
$$

In words: a response is good if it scored better than its siblings, measured in units of the group's spread; $\varepsilon_{\text{num}}$ is a tiny constant to avoid division by zero. The group mean is an estimate of $V(x)$, the expected reward for this prompt, which is exactly what a baseline should be ([Chapter 2.3](03-rl-foundations-for-llms.md)).

4. Maximize the PPO-style clipped objective with a KL penalty written directly into the loss:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert y_i\rvert}\sum_{t=1}^{\lvert y_i\rvert}\Big(\min\big(\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i}\big) - \beta\,\widehat{\mathrm{KL}}_{i,t}\Big)\Bigg],
$$

where $\lvert y_i\rvert$ is the number of tokens in response $i$, $\rho_{i,t} = \pi_\theta(y_{i,t}\mid s_{i,t})/\pi_{\theta_{\text{old}}}(y_{i,t}\mid s_{i,t})$ is the per-token ratio, and $\widehat{\mathrm{KL}}_{i,t} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$ is the non-negative KL estimator evaluated at the sampled token. In words: PPO's clipped update, but every token of a response shares the response's group-relative advantage, losses are averaged per response then across the group, and the KL to the reference is a penalty term rather than part of the reward.

What we gained: no value network (the DeepSeekMath paper highlights reduced memory and compute), no reward model if rewards are verifiable, and a baseline that is accurate *for this specific prompt*. What we pay: $G$ samples per prompt (typically 8–64) and no per-token credit assignment.

### DeepSeek-R1-Zero: pure RL from a base model

DeepSeek-R1 (DeepSeek-AI, 2025) first trained **R1-Zero**: GRPO applied directly to the DeepSeek-V3 *base* model, with no SFT at all.

- **Prompt template**: a simple instruction to reason inside `<think> ... </think>` and then give the answer inside `<answer> ... </answer>`. No examples of how to reason.
- **Rewards**: rule-based only. An accuracy reward (math answers in a specified format checked against references; code checked by a compiler against test cases) and a format reward for using the tags. The authors deliberately avoided neural reward models, citing reward hacking at scale.
- **Results reported**: AIME 2024 pass@1 rose from 15.6% to 71.0% during training (86.7% with majority voting), comparable to OpenAI's o1-0912 at the time.
- **Emergent behavior**: the average response length grew steadily during training, from hundreds to thousands of tokens, and the model began to re-evaluate its own steps, including an "aha moment" where it explicitly stops and reconsiders. None of this was demonstrated; RL found it because it raised the reward.
- **Problems**: poor readability and language mixing within responses.

### DeepSeek-R1: the full multi-stage recipe

To fix R1-Zero's problems and make a general assistant, R1 uses four stages:

```text
 V3-Base ─► (1) Cold-start SFT ─► (2) Reasoning RL ─► (3) Rejection sampling + SFT ─► (4) RL for all scenarios ─► R1
            thousands of long,     GRPO, rule-based     ~600K reasoning samples from      rule rewards for reasoning,
            readable CoT examples  rewards + language   the stage-2 model (filtered)      reward models for helpful-
                                   consistency reward   + ~200K non-reasoning samples     ness & harmlessness
                                                        → SFT the base model again
                                                                         │
                                                                         └──► (5) Distillation: the same ~800K samples
                                                                               SFT small Qwen2.5 / Llama models
```

1. **Cold start.** SFT on thousands of long CoT examples in a readable format. This gives RL a better, more readable starting point.
2. **Reasoning-oriented RL.** GRPO on math, code, science and logic prompts with rule-based rewards, plus a *language-consistency* reward (fraction of target-language words) to stop language mixing, at a small cost in accuracy.
3. **Rejection sampling and SFT.** Sample many responses from the RL checkpoint, keep only correct and readable ones (about 600K reasoning samples), add about 200K non-reasoning samples (writing, QA, etc.), and fine-tune the base model on this ~800K set. This consolidates RL gains and restores general abilities.
4. **RL for all scenarios.** A second RL stage that combines rule-based rewards for reasoning with reward models for helpfulness and harmlessness on general prompts.
5. **Distillation.** The same ~800K samples were used to SFT smaller Qwen2.5 and Llama models (1.5B to 70B). The R1 report found that distilling from R1 gave small models much stronger reasoning than running large-scale RL on the small model itself ([Chapter 2.8](08-distillation-and-efficient-fine-tuning.md)).

Kimi k1.5 (Kimi Team, 2025) reported a parallel route to strong reasoning: RL with long contexts (up to 128K tokens), a policy-optimization method based on online mirror descent, a length penalty, and "long2short" methods to transfer long-CoT skill into shorter answers.

### Failure modes and fixes

Running GRPO for long on long-CoT problems exposed several problems. Each fix is a small, targeted change.

**1. Entropy collapse → DAPO's "clip-higher".** The policy's token entropy drops quickly; samples within a group become nearly identical, exploration stops, and learning stalls. One cause is the symmetric clip: with $\epsilon = 0.2$, a token with $\pi_{\text{old}} = 0.01$ can only rise to $0.012$ per update, whereas a token at $0.9$ can rise to the cap of 1. Low-probability "exploration" tokens are held back. DAPO (Yu et al., 2025) **decouples** the clip range: $\operatorname{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$ with a larger upper bound ($\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$ in the paper), letting unlikely good tokens grow faster.

**2. Zero-advantage groups → dynamic sampling.** If all $G$ responses get the same reward (all correct or all wrong), every advantage is 0 and the group contributes no gradient, only noise from the KL term. As the model improves, more prompts become all-correct and the effective batch shrinks. DAPO **over-samples prompts and filters out groups with accuracy 0 or 1**, refilling the batch until every group has contrast. Curating prompts to the model's "difficulty frontier" has the same purpose.

**3. Length bias → token-level loss (DAPO) and Dr. GRPO.** The $\frac{1}{\lvert y_i\rvert}$ normalization averages each response's tokens before averaging across responses. Consequences:

- For a correct response (positive advantage), shorter responses get a larger per-token update, favoring brevity among correct answers.
- For a wrong response (negative advantage), a *longer* response has its penalty spread over more tokens, so each token is penalized less: **long wrong answers are punished less**, which pushes length up without improving accuracy.

DAPO switches to a **token-level** average: sum the per-token losses over all responses in the batch and divide by the total number of tokens, so every token counts equally. Dr. GRPO ("GRPO done right", Liu et al., 2025) argues both normalizations in GRPO are biased: the $\frac{1}{\lvert y_i\rvert}$ term (length bias, as above) and the division by $\operatorname{std}(r)$ (which up-weights prompts whose rewards have low variance, i.e. those that are nearly always or nearly never solved, a **difficulty bias**). It removes both, using $\hat A_i = r_i - \operatorname{mean}(r)$ and a constant normalizer, and reports better token efficiency (shorter wrong answers) at similar accuracy. DAPO also adds **overlong reward shaping**: a soft length penalty near the maximum length and masking of truncated samples, so the model is not punished for good-but-truncated reasoning in a noisy way.

**4. Noisy token-level ratios → GSPO.** The reward is given for the whole sequence, but GRPO applies importance ratios and clipping per *token*. Per-token ratios are noisy, and the noise accumulates over long responses; Zheng et al. (2025, Qwen team) argue this destabilizes training, especially for mixture-of-experts models, where the experts activated for a token can change between the old and new policy. **GSPO** (Group Sequence Policy Optimization) uses a **sequence-level** ratio, length-normalized:

$$
s_i(\theta) = \Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\Big)^{1/\lvert y_i\rvert} = \exp\Big(\frac{1}{\lvert y_i\rvert}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid s_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid s_{i,t})}\Big),
$$

and clips $s_i$ (one number per response) instead of each token's ratio. In words: the whole response is kept or clipped together, matching the unit the reward is assigned to. The Qwen team reports it is much more stable and used it for Qwen3 RL.

**5. Clipped-away rare tokens → CISPO.** In PPO/GRPO, once a token's ratio leaves the clip range its gradient is *zero*. MiniMax (MiniMax-M1, 2025) observed that rare tokens associated with reflection ("However", "Wait", "Recheck") have low probability, quickly hit the upper clip, and then stop receiving gradient in later off-policy updates. **CISPO** clips the importance-sampling *weight* and treats it as a constant (stop-gradient), but keeps the policy-gradient term for every token:

$$
J_{\text{CISPO}} = \mathbb{E}\Big[\frac{1}{\sum_i \lvert y_i\rvert}\sum_{i,t}\operatorname{sg}\big(\operatorname{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big)\,\hat A_{i}\,\log\pi_\theta(y_{i,t}\mid s_{i,t})\Big],
$$

where $\operatorname{sg}$ is stop-gradient. In words: the IS weight is bounded for variance control, but no token is ever silenced. This is REINFORCE with a truncated importance weight.

| Problem | Symptom | Fix |
|---|---|---|
| Entropy collapse | Entropy → 0, identical samples, flat reward | Clip-higher ($\epsilon_{\text{high}} > \epsilon_{\text{low}}$) (DAPO) |
| Zero-advantage groups | Many groups all-correct/all-wrong; shrinking effective batch | Dynamic sampling / difficulty filtering (DAPO) |
| Length bias | Length grows, wrong answers get longer | Token-level loss (DAPO); remove $1/\lvert y\rvert$ and std normalization (Dr. GRPO) |
| Token-ratio noise, MoE instability | Spiky loss, collapse | Sequence-level ratio and clipping (GSPO) |
| Rare tokens clipped away | Reflection behaviors fail to grow | Clip the IS weight, keep every token's gradient (CISPO) |

### What RLVR does and doesn't add

Yue et al. (2025) compared base models and their RLVR-trained versions using **pass@k** ([Chapter 2.6](06-reasoning-and-test-time-compute.md)) for $k$ from 1 to very large values:

- At **small $k$** (e.g. pass@1), the RL model is clearly better.
- At **large $k$**, the base model catches up and often *exceeds* the RL model: problems the RL model solves were usually already solvable by the base model with enough samples, and the RL model's coverage of solvable problems can shrink.

Their interpretation: current RLVR mainly **sharpens** the distribution toward reasoning paths the base model could already produce, improving sampling efficiency rather than expanding the set of solvable problems; distillation from a stronger model, by contrast, can add new reasoning patterns. This is consistent with s1 ([Chapter 2.6](06-reasoning-and-test-time-compute.md)) and with the "superficial alignment" view ([Chapter 2.1](01-why-post-training.md)).

The debate continues: making pass@1 approach pass@k is itself enormously valuable (users get one answer), longer RL runs and more diverse environments may push beyond the base model's coverage, and entropy-preserving fixes above aim exactly at keeping exploration alive. Treat "RL only sharpens" as an important hypothesis with good evidence in the studied settings, not a law.

### RL infrastructure

A GRPO step has two phases with very different profiles:

```text
  ┌───────────────── rollout (generation) ─────────────────┐   ┌──── training ────┐
  │ B prompts × G samples, autoregressive, up to 32K+ tokens│──►│ forward/backward │──► sync new
  │ memory-bound decoding, long-tail lengths               │   │ on all tokens    │    weights to
  └────────────────────────────────────────────────────────┘   └──────────────────┘    generator
          usually the MAJORITY of wall-clock time
```

- **Rollouts dominate cost.** Generating thousands of long responses token by token is slow and memory-bandwidth-bound, while the training pass processes all tokens in parallel. So RL systems embed a fast **inference engine** (vLLM, SGLang; see Part 3) for generation and a training framework (FSDP, Megatron) for updates, and must copy updated weights from trainer to generator after each step.
- **Long tails.** A batch finishes only when its longest response finishes; most GPUs idle while a few sequences run to the length limit. Remedies include partial rollouts (pause long generations and resume them in the next iteration, as Kimi k1.5 described) and dynamic batching.
- **Asynchronous / off-policy rollouts.** Generation for step $k+1$ runs while training step $k$, so rollouts come from a slightly stale policy. The importance ratio (and clipping or truncation) corrects for the staleness; this is a main practical reason off-policy corrections matter in LLM RL.
- **Training–inference mismatch.** The inference engine and the trainer compute slightly different probabilities for the same tokens (different kernels, precision, MoE routing). Recomputing $\log\pi_{\text{old}}$ in the trainer and applying an importance correction between generator and trainer probabilities reduces the resulting instability.
- **Open frameworks:** [verl](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [TRL](https://github.com/huggingface/trl) and [slime](https://github.com/THUDM/slime) implement these patterns.

## Code

### 1. Verifiable reward functions

```python
import re, subprocess, tempfile, os

def normalize(ans):
    ans = ans.strip().replace(" ", "").replace("\\!", "").rstrip(".")
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", ans)
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", ans)      # a/b -> number (never eval!)
    try:
        val = float(m[1]) / float(m[2]) if m else float(ans)
        return str(round(val, 6))
    except (ValueError, ZeroDivisionError):
        return ans                                                   # fall back to string match

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
    """Run program + asserts in a subprocess. Use a real sandbox (container, no network) in practice."""
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

### 2. GRPO advantages and loss (with DAPO / Dr. GRPO / GSPO / CISPO switches)

```python
import torch

def group_advantages(rewards, G, scale_by_std=True, eps=1e-6):
    """rewards: [B*G] ordered group by group -> advantages [B*G]."""
    r = rewards.view(-1, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:                                   # GRPO; Dr. GRPO sets this False
        adv = adv / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    return adv.view(-1)

def policy_loss(logp, logp_old, adv, mask, mode="grpo", eps_low=0.2, eps_high=0.2,
                logp_ref=None, beta=0.0):
    """logp, logp_old, logp_ref, mask: [N, T] (N = B*G responses); adv: [N].
    Returns a scalar to MINIMIZE."""
    A = adv.unsqueeze(1)                                # same advantage for every token
    log_ratio = logp - logp_old
    if mode == "gspo":                                  # one length-normalized ratio per sequence
        seq = (log_ratio * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        ratio = torch.exp(seq)
        per_tok = -torch.min(ratio * A, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
        per_tok = per_tok.expand_as(logp)
    elif mode == "cispo":                               # clipped, stop-gradient IS weight * log pi
        w = torch.clamp(torch.exp(log_ratio), 1 - eps_low, 1 + eps_high).detach()
        per_tok = -w * A * logp
    else:                                               # grpo / dapo: per-token PPO clip
        ratio = torch.exp(log_ratio)
        per_tok = -torch.min(ratio * A, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
    if beta > 0 and logp_ref is not None:               # k3 KL estimator, always >= 0
        d = logp_ref - logp
        per_tok = per_tok + beta * (torch.exp(d) - d - 1)
    if mode == "grpo":                                  # mean per response, then over responses
        return ((per_tok * mask).sum(1) / mask.sum(1)).mean()
    return (per_tok * mask).sum() / mask.sum()          # token-level mean (DAPO, CISPO, GSPO-style)

def keep_informative_groups(rewards, G):
    """DAPO dynamic sampling: indices of groups whose rewards are not all equal."""
    r = rewards.view(-1, G)
    return ((r.max(1).values - r.min(1).values) > 0).nonzero().squeeze(-1)

rewards = torch.tensor([1., 0., 0., 1.,   1., 0., 0., 0.,   1., 1., 1., 1.])
print(group_advantages(rewards, G=4).view(3, 4))
print(keep_informative_groups(rewards, G=4))          # tensor([0, 1]): group 2 is all-correct
```

A training step then is: sample $G$ responses per prompt with the inference engine, compute rewards, drop uninformative groups, compute `logp_old` (and `logp_ref` if $\beta > 0$) with the trainer, and take one or more optimizer steps on `policy_loss`. Setting `mode="dapo", eps_high=0.28, beta=0` gives DAPO's loss; `scale_by_std=False` with a constant normalizer gives Dr. GRPO.

## Worked example

**GRPO advantages for $G = 4$** (population standard deviation, as in the code; some implementations use the sample std, which changes the scale but not the signs).

*Group A*, rewards $(1, 0, 0, 1)$: mean $= 0.5$, std $= 0.5$. Advantages $= (1-0.5)/0.5, \dots = (+1, -1, -1, +1)$. The two correct responses are pushed up, the two wrong ones down, equally.

*Group B*, rewards $(1, 0, 0, 0)$ (a hard prompt): mean $= 0.25$, std $= \sqrt{0.25\cdot0.75} \approx 0.433$.

- Correct: $(1 - 0.25)/0.433 \approx +1.732$.
- Each wrong: $(0 - 0.25)/0.433 \approx -0.577$.

The single success gets a large positive advantage: rare successes on hard problems are strongly reinforced. Note the advantages sum to zero within a group.

*Group C*, rewards $(1,1,1,1)$: mean $= 1$, std $= 0$, every advantage $= 0/(0+\varepsilon) = 0$. **No learning signal**; dynamic sampling would drop this group.

*Dr. GRPO on group B*: no std division, advantages $= (+0.75, -0.25, -0.25, -0.25)$. Compared with GRPO, the group's influence relative to a group with rewards $(1,0,1,0)$ (Dr. GRPO advantages $\pm0.5$, GRPO $\pm1$) is no longer rescaled by the reward spread.

**Length bias in numbers.** Two wrong responses in a group each have advantage $-1$; one has 100 tokens, the other 1,000. With GRPO's per-response averaging, each *token* of the short response receives weight $-1/100 = -0.01$ and each token of the long one $-1/1000 = -0.001$: the long wrong answer's tokens are pushed down 10× less. With a token-level average over the batch, both get the same per-token weight.

**Clip-higher in numbers.** For a token with $\pi_{\text{old}} = 0.01$ and positive advantage, $\epsilon = 0.2$ caps the new probability (for the purpose of the gradient) at $0.012$; $\epsilon_{\text{high}} = 0.28$ allows $0.0128$. For a token at $0.5$, the caps are $0.6$ vs $0.64$. The absolute room to grow is tiny for rare tokens either way, which is why entropy collapses without such fixes.

## In practice (2025–2026)

- **GRPO-family algorithms are the default** for open reasoning RL (DeepSeek-R1, Qwen3 with GSPO, many academic reproductions with DAPO and Dr. GRPO tweaks). DAPO reports 50 points on AIME 2024 with a Qwen2.5-32B base model using its four techniques.
- **KL is often small or zero** with verifiable rewards (DAPO removes it), since the main reason for the KL penalty, a hackable learned reward, is weaker. Some runs still keep it for stability.
- **Data curation is critical**: prompts with reliable reference answers, deduplicated against evaluation sets, filtered to the model's difficulty frontier (neither always nor never solved).
- **Mixed rewards**: verifiable rewards for math/code/tools combined with reward models for general helpfulness in later stages, as in R1's final stage.
- **Agentic RLVR**: multi-turn environments (code repositories with tests, browsers, terminals) are the 2025–2026 frontier ([Chapter 2.9](09-agents-tool-use-and-safety.md)).
- **Monitor**: mean reward, per-difficulty accuracy, response length, entropy, clip fraction, fraction of zero-advantage groups, and the train–inference log-prob gap.

## Common pitfalls / misconceptions

- **Leaky verifiers.** Accepting `\boxed{3}` *and* `\boxed{4}` in the same response, or matching any number in the text, gets hacked fast. Check the last answer only and penalize multiple answers.
- **Over-weighting format rewards.** The model learns to produce perfect tags around wrong answers. Keep format rewards small or make them a gate.
- **Too-easy or too-hard prompts.** All-correct and all-wrong groups waste rollouts.
- **"Longer responses mean better reasoning."** Length can grow from length bias alone; track accuracy per token.
- **"RL taught the model to reason from scratch."** R1-Zero started from a very strong base model; the evidence (pass@k) suggests RL mostly sharpens existing abilities.
- **Ignoring evaluation contamination.** Many public math sets overlap with training data; use fresh held-out problems.

## Key takeaways

- RLVR replaces a learned reward model with programmatic checks, making long RL runs possible without reward-model hacking (verifiers can still be gamed if leaky).
- GRPO = PPO with a group-relative advantage $(r_i - \bar r)/\sigma_r$ instead of a value network; every token of a response shares its advantage.
- R1-Zero showed pure RL on a base model elicits long CoT and self-verification; R1 adds cold-start SFT, multi-stage RL, rejection-sampling SFT, and distillation.
- Fixes: clip-higher and dynamic sampling (DAPO), unbiased normalization (Dr. GRPO, token-level loss), sequence-level ratios (GSPO), clipped stop-gradient IS weights (CISPO).
- RLVR mainly improves pass@1 by sharpening; whether it expands pass@k coverage is debated.
- Rollout generation dominates cost; RL systems embed inference engines and use async, off-policy corrections.

## Check your understanding

??? question "Why does GRPO not need a value network?"
    Its baseline is the mean reward of the other samples for the same prompt, estimated empirically from the group, instead of a learned critic.

??? question "Compute GRPO advantages for rewards (0, 0, 1, 1, 1, 1) with population std."
    Mean $= 4/6 \approx 0.667$; variance $= \frac{2(0.667)^2 + 4(0.333)^2}{6} = \frac{0.889 + 0.444}{6} \approx 0.222$, std $\approx 0.471$. Correct: $(1-0.667)/0.471 \approx +0.707$; wrong: $(0-0.667)/0.471 \approx -1.414$.

??? question "Why did RLVR work for reasoning when RLHF-PPO on human preferences did not produce long CoT?"
    Verifiable rewards are accurate and hard to hack, so the policy can be optimized for many steps; and strong base models already contain the reasoning behaviors, which RL amplifies.

??? question "What problem does DAPO's dynamic sampling solve?"
    Groups where every response has the same reward have zero advantage and provide no gradient; as training progresses there are more of them, shrinking the effective batch. Dynamic sampling over-samples and keeps only groups with mixed rewards.

??? question "What is reward hacking? Give one example from RLVR."
    The policy exploits flaws in the reward instead of solving the task — e.g. outputting multiple answers to match a lenient parser, or editing unit tests in a coding environment.

??? question "What does the pass@k study by Yue et al. suggest about RLVR?"
    RL-trained models beat their base models at small $k$, but at large $k$ the base model solves as many or more problems; RLVR mostly makes already-reachable solutions more likely rather than adding new ones (in the settings studied).

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024 | Group Relative Policy Optimization: group-normalized rewards as advantages, no value network. |
| :material-star: [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025 | R1-Zero (pure RL) and the multi-stage R1 recipe plus distillation; the key post-training paper of 2025. |
| :material-new-box: [Kimi k1.5: Scaling RL with LLMs](https://arxiv.org/abs/2501.12599) — Kimi Team, 2025 | Long-context RL, online mirror descent, length penalties, partial rollouts, long2short. |
| :material-new-box: [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) — Yu et al. (ByteDance Seed), 2025 | Clip-higher, dynamic sampling, token-level loss, overlong reward shaping. |
| :material-new-box: [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) — Liu et al., 2025 | Length and std-normalization biases in GRPO and an unbiased variant. |
| :material-new-box: [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — Zheng et al. (Qwen), 2025 | Sequence-level ratios and clipping; stable RL for MoE models, used for Qwen3. |
| :material-new-box: [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) — MiniMax, 2025 | Clip importance-sampling weights instead of token updates so rare reflection tokens keep their gradient. |
| :material-book-open-variant: [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — Yue et al., 2025 | The pass@k analysis: RL sharpens sampling more than it adds capability. |
| :material-star: [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al. (AI2), 2024 | Coined "RLVR" for its final stage; fully open code and data. |
| :material-book-open-variant: [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) · [Open-R1](https://github.com/huggingface/open-r1) | Practical, open implementations to reproduce a mini R1-Zero. |
