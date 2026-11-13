# 4.8 Ads Systems

!!! abstract "What you'll learn"
    - How ad ranking differs from organic recommendation: a three-sided market, money-valued predictions, advertiser objectives
    - The ad auction from scratch: eCPM $=$ bid × pCTR (× pCVR), second-price and GSP pricing with a worked example, and why truthfulness matters (VCG intuition)
    - Why calibration is critical in ads, and how conversion modeling handles sample selection bias (ESMM) and delayed feedback
    - Budgets, pacing and auto-bidding (target CPA / ROAS) as a control and optimization problem, with a simulation
    - The generative frontier: diffusion-based auto-bidding (AIGB), end-to-end generative advertising (EGA-V2), generative creatives, privacy-constrained modeling and large ads foundation models (Meta GEM)

## Why this matters / the problem

Ads fund most large consumer internet platforms. An ads system is a recommender with three parties instead of two:

- **Users** want relevant, non-annoying content.
- **Advertisers** want outcomes (clicks, installs, purchases) at a price they choose, and they spend real budgets.
- **The platform** wants long-term revenue, which depends on keeping both users and advertisers happy.

The ranking machinery is the same as in Chapters [4.1](01-recsys-fundamentals.md)–[4.4](04-scaling-recommendation-models.md): retrieval, ranking with multi-task models, sequence features. What is new is that predictions are **multiplied by money** in an **auction** that decides which ads are shown *and what advertisers pay*. A prediction error is no longer just a worse ordering; it is a mispriced transaction. And advertisers increasingly hand the bidding to the platform's **auto-bidding** algorithms, which turns the system into a large online optimization problem.

## Intuition

Imagine a single billboard slot that changes every second, auctioned to whoever values it most. Advertisers do not value an *impression*; they value an *outcome*. If advertiser A will pay \$2 per click and her ad has a 5% chance of being clicked by this user, the impression is worth $0.05 \times \$2 = \$0.10$ to her. Ranking ads by this expected value per impression — **eCPM**, effective cost per mille (thousand impressions) — picks the ad that creates the most value, and the platform's job is to estimate the click probability *accurately*, because it directly sets the value.

Then comes pricing: if you simply charge winners what they bid, they will shade their bids and play games. Auctions such as **second-price** charge the winner the minimum needed to win, which makes honest bidding a sensible strategy.

## How it works

### From prediction to expected value

For an ad $a$ shown to user $u$ in context $c$:

- $\text{pCTR}_a = P(\text{click}\mid u,a,c)$ from the ranking model;
- $\text{pCVR}_a = P(\text{conversion}\mid \text{click}, u, a, c)$ for conversion-optimized ads;
- $b_a$ = the advertiser's bid, expressed per click (CPC), per conversion (CPA), or per thousand impressions (CPM).

The expected value per impression, in a common currency, is

$$
\text{eCPM}_a = 1000 \times \begin{cases}
b_a^{\text{CPM}}/1000 & \text{CPM bid}\\
b_a^{\text{CPC}}\cdot \text{pCTR}_a & \text{CPC bid}\\
b_a^{\text{CPA}}\cdot \text{pCTR}_a\cdot\text{pCVR}_a & \text{CPA bid}
\end{cases}
$$

In words: convert every bid into "expected money per impression" using the model's probabilities, so ads with different bidding types can compete in one auction. Real systems add a **quality / user-experience term** (e.g. predicted negative feedback, relevance, landing-page quality) so that a high bid cannot buy a terrible user experience:

$$
\text{rank score}_a = \text{eCPM}_a + \lambda\cdot \text{quality}_a ,
$$

with $\lambda$ tuned by the platform. Ads also compete with organic content for feed slots, which requires converting organic engagement value and ad value into comparable units — another reason absolute calibration matters.

### Auctions: first price, second price, GSP, VCG

**Single slot.** Rank by eCPM; the highest wins.

- *First-price*: the winner pays her own bid. Bidders have an incentive to shade bids below their true value, and the optimal shading depends on guessing competitors — unstable and complex.
- *Second-price (Vickrey)*: the winner pays the minimum bid that would still have won, i.e. determined by the runner-up. Bidding your true value is a **dominant strategy**: overbidding only risks paying more than the item is worth; underbidding only risks losing an auction you would have been happy to win at that price. This property is called **truthfulness** or **incentive compatibility**.

**Multiple slots: GSP.** Search and feed ads have several slots with different visibility. Model the click probability as separable: $P(\text{click on } a \text{ in slot } s) = \theta_s \cdot q_a$, where $\theta_s$ is the slot's position effect (decreasing with $s$) and $q_a$ the ad's own pCTR. Rank ads by $b_a q_a$. In the **generalized second-price** (GSP) auction, the ad in slot $s$ pays per click the minimum bid that would keep its position:

$$
p_s = \frac{b_{s+1}\, q_{s+1}}{q_s},
$$

where index $s+1$ refers to the ad ranked just below. In words: you pay just enough that your eCPM matches the next ad's eCPM.

**VCG.** GSP is *not* truthful with multiple slots: an advertiser can sometimes gain by bidding below value to take a cheaper, lower slot. The **Vickrey–Clarke–Groves** mechanism is truthful: each winner pays the **externality** she imposes on others — the total value others would get if she were absent, minus what they get with her present. VCG is theoretically clean but harder to explain to advertisers; GSP was historically the industry standard in search ads, and with auto-bidding (where the platform bids on behalf of advertisers) the design trade-offs have shifted again. The worked example computes both.

### Why calibration is critical

In organic ranking, any monotone distortion of scores leaves the ordering unchanged. In ads:

- **Mixed bid types.** CPC ads use pCTR, CPA ads use pCTR × pCVR, CPM ads use neither. If pCVR is 2× too high, every CPA ad's eCPM doubles relative to CPC and CPM ads, and the auction is distorted.
- **Per-impression billing for outcome goals.** When the platform charges per impression but advertisers bid per conversion (common with auto-bidding), the charge is computed from predicted probabilities. Over-prediction means advertisers overpay for each actual conversion; under-prediction means the platform under-charges and under-delivers.
- **Ads vs organic and reserve prices.** Deciding whether to show an ad at all, against organic content or a reserve price, compares absolute values.

So ads systems track calibration (Chapter [4.2](02-ranking-and-ctr-models.md)) per segment and maintain online calibration layers.

### Conversion modeling

**Sample selection bias and ESMM.** A CVR model $P(\text{conv}\mid\text{click})$ is naturally trained only on clicked impressions, but it is used to score *all* candidate impressions — a distribution mismatch, and clicks are sparse. The **Entire Space Multi-task Model (ESMM)** uses the funnel impression → click → conversion:

$$
\underbrace{P(\text{click}, \text{conv}\mid x)}_{\text{pCTCVR}} = \underbrace{P(\text{click}\mid x)}_{\text{pCTR}}\times\underbrace{P(\text{conv}\mid \text{click}, x)}_{\text{pCVR}} .
$$

Two towers output pCTR and pCVR (sharing embeddings); the model is supervised on **all impressions** with two losses — click labels on pCTR and click-and-convert labels on pCTCVR. pCVR is never trained directly on the biased clicked subset; it is learned implicitly as the factor that makes the product match. Shared embeddings let the sparse CVR task borrow from the dense CTR task.

**Delayed feedback.** Conversions can arrive hours or days after the click. With streaming training (Chapter 4.1), an impression must be labelled before its conversion has happened — a *fake negative*. Options:

1. Wait for an attribution window before training: correct labels, stale model.
2. Train immediately and correct later: insert the sample as negative, then re-insert as positive when the conversion arrives, with importance weights that correct the induced bias.
3. Model the delay explicitly: if $p(x)$ is the eventual conversion probability and $F(e\mid x)$ the probability that a conversion arrives within elapsed time $e$, then $P(\text{observed conversion by } e) = p(x)\,F(e\mid x)$, and both are learned jointly.

### Budgets, pacing and auto-bidding

Advertisers rarely bid per auction by hand. They set a **budget** and a **goal**, and the platform bids for them:

- **Target CPA (tCPA):** "get as many conversions as possible, average cost per conversion ≤ $T$".
- **Target ROAS:** "revenue / spend ≥ $R$".
- **Max delivery under budget:** spend the daily budget $B$ to maximize conversions.

**Bidding as optimization.** Consider impressions $i = 1..n$ with value $v_i$ (e.g. $T\cdot\text{pCVR}_i$ expected conversion value) and price $c_i$ (the market price, which the advertiser pays on winning in a second-price auction). Choose which to win, $x_i\in\{0,1\}$:

$$
\max_{x}\ \sum_i x_i v_i \quad \text{s.t.}\quad \sum_i x_i c_i \le B .
$$

Relax with a Lagrange multiplier $\lambda \ge 0$ on the budget: maximize $\sum_i x_i (v_i - \lambda c_i)$. Win impression $i$ exactly when $v_i > \lambda c_i$, i.e. when $c_i < v_i/\lambda$. In a second-price auction you win exactly when the price is below your bid, so the optimal policy is

$$
b_i = \frac{v_i}{\lambda}.
$$

In words: bid your value scaled by a single number, the "shadow price" of budget. Large $\lambda$ (tight budget) → bid low and buy only cheap, high-value impressions; small $\lambda$ → bid high. The whole problem reduces to finding the right $\lambda$ — which depends on future traffic you cannot see.

**Pacing as feedback control.** Run a controller during the day: compare actual spend to a target spend curve (e.g. proportional to expected traffic) and adjust $\lambda$ — increase it if overspending, decrease it if underspending. A PI(D) controller or a multiplicative update is typical. tCPA and ROAS constraints add more multipliers in the same way.

**Learning-based bidders.** Because traffic is non-stationary and constraints interact, platforms train bidding policies with RL on historical auction logs.

### Generative auto-bidding (AIGB)

RL bidders model bidding as a Markov decision process, which struggles with long horizons and noisy environments. **AIGB** (Alibaba) reframes auto-bidding as **generative modeling of a whole bid trajectory**: its DiffBid model is a conditional **diffusion model** that generates a trajectory conditioned on the desired return and constraints, avoiding error propagation across time steps. It reports +2.81% GMV and +3.36% ROI in online A/B tests on Alibaba's advertising platform. It mirrors the "decision as conditional generation" idea from offline RL.

### End-to-end generative advertising (EGA-V2)

Just as OneRec replaces the organic cascade (Chapter [4.6](06-end-to-end-generative-recommenders.md)), **EGA-V2** (Meituan) proposes a single generative model for advertising that jointly handles user interests, point-of-interest and creative generation, ad allocation and payment. It uses hierarchical tokenization and multi-token prediction, a permutation-aware reward model and token-level bidding, and — the ads-specific part — decouples allocation from **payment** using differentiable ex-post regret minimization, aiming at *approximate incentive compatibility* (the truthfulness property of the auction section) at the POI level. Its evaluations are offline, and it is a good map of what an "auction inside a generative model" has to solve.

### Generative ad creatives

Major ad platforms now offer LLM- and diffusion-generated ad text, backgrounds, image variations and video. This turns creative selection into a recommendation problem with a generated catalogue: generate many variants, predict performance per user segment, and serve the best — while enforcing brand safety and advertiser approval.

### Privacy and modeling with fewer signals

Platform and regulatory changes (for example Apple's App Tracking Transparency on iOS, which requires user permission for cross-app tracking) have reduced the user-level conversion data available to ad systems. Consequences for modeling:

- Conversion labels become **aggregated, delayed or noisy** (e.g. reported by privacy-preserving attribution APIs), so models are calibrated to aggregate counts rather than individual labels.
- More reliance on **first-party, on-platform signals** (behaviour on the platform itself), **contextual signals**, and **server-side conversion reporting** from advertisers.
- **Modeled conversions**: predicting conversions that cannot be observed directly, which raises the bar for calibration and validation.

### Large ads foundation models

Meta's **GEM** (Generative Ads Recommendation Model) is described as an LLM-scale ads foundation model trained across thousands of GPUs on ad and organic engagement data. Its architecture extends Wukong-style stacked factorization machines (Chapter [4.4](04-scaling-recommendation-models.md)) for non-sequence features and models long behaviour sequences with interleaved sequence and cross-feature layers. GEM's key role is **knowledge transfer**: it teaches hundreds of downstream vertical ads models through direct and hierarchical transfer (distillation, representation learning, parameter sharing). Meta's engineering blog reports a 5% increase in ad conversions on Instagram and 3% on Facebook Feed in Q2 2025 after launch. This is the SUM pattern (Chapter [4.3](03-sequential-user-modeling.md)) taken to foundation-model scale.

## Code

GSP and VCG prices for a separable multi-slot auction, and the effect of a miscalibrated pCTR.

```python
def run_auction(ads, slot_ctr, pctr_override=None):
    """ads: dict name -> (bid_per_click, true_ctr). slot_ctr: position effects theta_s.
    Ranks by bid * predicted CTR. Returns allocation with GSP and VCG prices."""
    pred = {a: (pctr_override or {}).get(a, q) for a, (b, q) in ads.items()}
    order = sorted(ads, key=lambda a: ads[a][0] * pred[a], reverse=True)
    S = len(slot_ctr)
    ecpm = lambda a: ads[a][0] * pred[a]                  # predicted value per unit of slot CTR

    def welfare(ranking):                                 # predicted value of a slot assignment
        return sum(slot_ctr[s] * ecpm(a) for s, a in enumerate(ranking[:S]))

    rows = []
    for s, a in enumerate(order[:S]):
        nxt = order[s + 1] if s + 1 < len(order) else None
        gsp = ecpm(nxt) / pred[a] if nxt else 0.0         # per-click price
        others_without = welfare([x for x in order if x != a])
        others_with = welfare(order) - slot_ctr[s] * ecpm(a)
        vcg = (others_without - others_with) / (slot_ctr[s] * pred[a])
        true_clicks = slot_ctr[s] * ads[a][1]             # actual expected clicks per impression
        rows.append((s + 1, a, round(gsp, 2), round(vcg, 2), true_clicks))
    return rows

ads = {"A": (2.00, 0.05), "B": (4.00, 0.02), "C": (1.00, 0.06), "D": (3.00, 0.01)}
theta = [1.0, 0.5]
for label, override in [("calibrated", None), ("B's pCTR 2x too high", {"B": 0.04})]:
    rows = run_auction(ads, theta, override)
    revenue = sum(gsp * clicks for _, _, gsp, _, clicks in rows)
    value = sum(clicks * ads[a][0] for _, a, _, _, clicks in rows)
    print(label)
    for slot, a, gsp, vcg, _ in rows:
        print(f"  slot {slot}: ad {a}  GSP ${gsp:.2f}/click   VCG ${vcg:.2f}/click")
    print(f"  GSP revenue per impression ${revenue:.3f}, advertiser value created ${value:.3f}")
```

Budget pacing with a single multiplier $\lambda$: bid $= v/\lambda$, adjusted every hour toward a target spend curve.

```python
import random
B, T_CPA, HOURS, PER_HOUR = 1000.0, 50.0, 24, 10000         # budget $1000, target CPA $50
traffic = [0.5 + 0.5 * (8 <= h <= 22) for h in range(HOURS)]   # more traffic in daytime
target_curve = [sum(traffic[:h + 1]) / sum(traffic) * B for h in range(HOURS)]

def simulate(pacing):
    random.seed(0)
    lam, spend, conv, out_at = 1.0, 0.0, 0.0, None
    for h in range(HOURS):
        for _ in range(int(PER_HOUR * traffic[h])):
            pcvr = random.betavariate(1, 400)                   # conversion prob. of impression
            price = random.lognormvariate(-4.0, 0.8)            # competing second price
            value = T_CPA * pcvr                                # expected value at target CPA
            if value / lam > price and spend + price <= B:      # second price: win if bid > price
                spend += price; conv += pcvr
        if out_at is None and spend > 0.99 * B:
            out_at = h
        if pacing:                                              # multiplicative controller
            err = (spend - target_curve[h]) / B                 # ahead of schedule -> err > 0
            lam *= 2.0 ** (8 * err)                             # overspending -> raise lambda
    return spend, conv, out_at

for pacing in (False, True):
    s, c, h = simulate(pacing)
    print(f"pacing={pacing}: spend ${s:.0f}, budget exhausted at hour {h}, "
          f"expected conversions {c:.0f}, CPA ${s / c:.2f}")
```

Without pacing ($\lambda = 1$, bid = value) the advertiser wins almost everything early and runs out of budget around hour 9, missing the evening. With pacing, $\lambda$ rises, the bidder becomes selective (only impressions with a high value-to-price ratio), spend follows the target curve, and the same \$1,000 buys roughly 45% more expected conversions in this simulation.

## Worked example

**A GSP auction.** Two slots with position effects $\theta_1 = 1.0$, $\theta_2 = 0.5$. Four CPC advertisers:

| Ad | Bid / click | pCTR $q$ | $b\cdot q$ (per impression) | eCPM |
|---|---|---|---|---|
| A | \$2.00 | 0.05 | \$0.10 | \$100 |
| B | \$4.00 | 0.02 | \$0.08 | \$80 |
| C | \$1.00 | 0.06 | \$0.06 | \$60 |
| D | \$3.00 | 0.01 | \$0.03 | \$30 |

- **Ranking**: A (slot 1), B (slot 2). B has the highest bid but ranks second; A wins because it is more likely to be clicked.
- **GSP prices** per click: A pays $0.08/0.05 = \$1.60$ (less than her \$2.00 bid); B pays $0.06/0.02 = \$3.00$.
- **Expected revenue per impression**: A: $\theta_1 q_A p_A = 1.0\times0.05\times1.60 = \$0.080$; B: $0.5\times0.02\times3.00 = \$0.030$. Total \$0.110, i.e. \$110 per thousand impressions.

**VCG prices** for the same bids (treating bids as values):

- *Without A*: B gets slot 1 (value $1.0\times0.08 = 0.08$), C slot 2 ($0.5\times0.06 = 0.03$): others' welfare 0.11. *With A*: others get only B in slot 2: 0.04. A's payment $= 0.11 - 0.04 = \$0.07$ per impression $= 0.07/0.05 = \$1.40$ per click.
- *Without B*: A slot 1 (0.10), C slot 2 (0.03): 0.13. *With B*: others get A's 0.10. B pays $\$0.03$ per impression $= 0.03/(0.5\times0.02) = \$3.00$ per click.
- VCG revenue $= \$0.100$ per impression vs GSP's \$0.110. With these (truthful) bids GSP charges the top slot more; the difference is the incentive A has to under-bid in GSP.

**Miscalibration.** Suppose the model over-predicts B's pCTR as 0.04 (true 0.02). B's predicted value per impression becomes $4.00\times0.04 = 0.16$, so B takes slot 1 and A drops to slot 2.

- B pays $0.10/0.04 = \$2.50$ per click; A pays $0.06/0.05 = \$1.20$.
- Actual revenue per impression: B $1.0\times0.02\times2.50 = 0.050$; A $0.5\times0.05\times1.20 = 0.030$; total \$0.080 instead of \$0.110 (−27%).
- Actual value created: B $1.0\times0.02\times4 = 0.08$ + A $0.5\times0.05\times2 = 0.05$ = 0.13 instead of 0.14: a worse allocation for everyone.

Note that if *all* pCTRs were 2× too high, the ranking and GSP per-click prices would not change (they depend on ratios). Damage comes from *relative* miscalibration across ads, and from absolute miscalibration whenever predictions are compared to bids in other units (CPM, CPA billing, organic content, reserve prices).

**Auto-bidding.** An advertiser with tCPA \$50 sees an impression with pCTR × pCVR $= 0.002$. Value $v = 50\times0.002 = \$0.10$ per impression. With budget multiplier $\lambda = 1.25$ the bid is $0.10/1.25 = \$0.08$; if the second price is \$0.06 the advertiser wins and pays \$0.06.

## In practice (2025–2026)

- **Auto-bidding is the default**: most advertiser spend on large platforms goes through platform-run bidding toward tCPA/ROAS/budget goals, solved with multiplier-based control plus learned policies; generative bidders (AIGB) are a frontier.
- **Ranking models are the large-recommendation-model stack** of Chapters 4.3–4.4 (sequence models, scaled backbones), with per-segment calibration layers on top.
- **Foundation models for ads** (Meta's GEM, SUM-style user models) transfer knowledge to hundreds of downstream models.
- **Generative retrieval for ads** (COBRA at Baidu, URM at Alibaba) and **end-to-end generative advertising** (EGA-V2) are emerging.
- **Generative creatives** are standard advertiser tools; **privacy changes** have pushed modeling toward aggregate, on-platform and modeled signals.

## Common pitfalls / misconceptions

- **"Higher AUC = more revenue."** Revenue depends on calibration and the auction; AUC ignores both.
- **Training CVR only on clicks** and scoring all impressions (sample selection bias).
- **Treating fresh impressions as negatives** under delayed conversions without correction.
- **Assuming GSP is truthful.** It is for one slot (where it is just second price), not in general.
- **Ignoring pacing**: a perfect value model with no budget control spends the budget in the morning and misses evening traffic.
- **Optimizing short-term revenue only**: ad load and ad quality affect user retention, which drives long-term revenue.

## Key takeaways

- Ads rank by expected value per impression: eCPM $=$ bid × pCTR (× pCVR), plus quality terms.
- Second-price-style auctions charge the minimum winning price; GSP extends this to slots; VCG is the truthful multi-slot mechanism based on externalities.
- Calibration is essential because probabilities are multiplied by money and compared across bid types.
- Conversion modeling must handle sample selection bias (ESMM) and delayed feedback.
- Auto-bidding reduces to bidding $v/\lambda$ and controlling $\lambda$; generative bidders, end-to-end generative advertising and ads foundation models are the frontier.

## Check your understanding

??? question "Why is truthful bidding a dominant strategy in a single-slot second-price auction?"
    The price you pay is set by others, not by your bid; your bid only determines whether you win. Bidding your value means you win exactly when the price is below your value. Bidding higher can only add wins where you pay more than value; bidding lower can only lose wins that were profitable.

??? question "In the worked GSP example, what does A pay per click if C's bid rises to \$1.50?"
    C's per-impression value becomes $1.50\times0.06=0.09$, so the ranking is A (0.10), C (0.09), B (0.08). A pays $0.09/0.05=\$1.80$ per click, and B drops out of the two slots.

??? question "How does ESMM avoid training pCVR on the biased clicked-only subset?"
    It writes pCTCVR = pCTR × pCVR and supervises pCTR with click labels and pCTCVR with click-and-convert labels over all impressions. pCVR is learned implicitly as the factor that makes the product match, using the entire impression space.

??? question "In budget pacing, what happens to bids when the controller raises λ, and when should it?"
    Bids $v/\lambda$ fall, so the advertiser wins fewer, cheaper impressions. The controller raises λ when actual spend runs ahead of the target spend curve.

??? question "Why might a uniform 2× over-prediction of all pCTRs leave GSP rankings unchanged but still cause harm?"
    Rankings and per-click GSP prices depend only on ratios of predicted CTRs among CPC ads. But the absolute values matter whenever predictions are compared with CPM bids, converted into per-impression charges for outcome-based bids, compared against organic content or reserve prices, or fed into auto-bidders' value estimates.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) — McMahan et al. (Google), KDD 2013 | Large-scale ads CTR in production: FTRL-Proximal, calibration, memory savings, monitoring. |
| :material-book-open-variant: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) — He et al., ADKDD 2014 | GBDT + LR, normalized entropy, data freshness, and the negative-downsampling recalibration formula. |
| :material-star: [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (ESMM)](https://arxiv.org/abs/1804.07931) — Ma et al. (Alibaba), 2018 | Fixes sample selection bias in CVR modeling by learning over the entire impression space. |
| :material-book-open-variant: [Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta](https://arxiv.org/abs/2311.09544) — Zhang et al., 2023 | Foundation user models shared across hundreds of ads models. |
| :material-new-box: [AIGB: Generative Auto-bidding via Conditional Diffusion Modeling](https://arxiv.org/abs/2405.16141) — Guo et al. (Alibaba), KDD 2024 | Auto-bidding as conditional diffusion over bid trajectories, replacing RL bidders. |
| :material-new-box: [EGA-V2: An End-to-end Generative Framework for Industrial Advertising](https://arxiv.org/abs/2505.17549) — Zheng et al. (Meituan), 2025 | Generative modeling of the full ad allocation + payment process. |
| :material-new-box: [Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations (COBRA)](https://arxiv.org/abs/2503.02453) — Yang et al. (Baidu), 2025 | Generative retrieval deployed on an advertising platform. |
| :material-new-box: [Large Language Model as Universal Retriever in Industrial-Scale Recommender System (URM)](https://arxiv.org/abs/2502.03041) — Alibaba, 2025 | LLM-based retrieval deployed in advertising. |
| :material-new-box: [Meta's Generative Ads Recommendation Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) — Meta engineering blog, 2025 | An LLM-scale ads foundation model and how its knowledge is transferred to downstream models. |
