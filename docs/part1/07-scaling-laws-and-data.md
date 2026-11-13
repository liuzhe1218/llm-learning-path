# 1.7 Scaling Laws & Data

!!! abstract "What you'll learn"
    - What a **power law** is and why LLM loss follows one in parameters, data and compute.
    - The compute identity $C \approx 6ND$ and how to use it to size a training run.
    - The **Kaplan vs Chinchilla** disagreement, with a worked compute-optimal sizing example, and why modern models are deliberately **over-trained**.
    - The stages of a web-data pipeline: extraction, language ID, heuristic filters, model-based quality classifiers, **MinHash** deduplication (explained and implemented), decontamination.
    - Data mixtures, annealing/mid-training, synthetic data, and the limits of repeating data.

## Why this matters / the problem

Training a frontier model costs tens of millions of dollars of compute, and you get one shot. Before starting, a lab must decide how many parameters $N$ the model should have and how many tokens $D$ it should see. Too big a model on too little data, or the reverse, wastes a large share of the budget. **Scaling laws** answer this by fitting simple curves to many small, cheap experiments and extrapolating to the big run. The other half of the problem is *which* tokens: two models with the same $N$ and $D$ can differ enormously in quality because of their data. In 2025, data curation is widely regarded as the main competitive advantage between labs with similar architectures.

## Intuition

- **Power laws.** Each 10× more compute buys a roughly constant *fractional* reduction in (reducible) loss. On a log–log plot, loss vs compute is a straight line. This regularity is what makes billion-dollar runs plannable from million-dollar experiments.
- **Balancing N and D.** A big model that sees little data is under-trained; a small model that sees endless data saturates. For a fixed compute budget there is a sweet spot. Chinchilla found that $N$ and $D$ should grow together, at about 20 tokens per parameter.
- **But training is not the only cost.** A model is trained once and then served billions of times. Serving cost depends on $N$, not on $D$. So it often pays to train a *smaller* model on *far more* data than is compute-optimal.
- **Data quality is a multiplier.** Filtering out junk and duplicates means each training token teaches more, which effectively shifts the whole scaling curve down.

## How it works

### Power laws

A **power law** is $y = a x^{-b}$ for constants $a, b > 0$. Taking logs, $\log y = \log a - b \log x$: a straight line on log–log axes with slope $-b$. Kaplan et al. (2020) found that test loss $L$ follows power laws in each resource when the others are not a bottleneck:

$$
L(N) \approx \Big(\frac{N_c}{N}\Big)^{\alpha_N}, \qquad L(D) \approx \Big(\frac{D_c}{D}\Big)^{\alpha_D}, \qquad L(C) \approx \Big(\frac{C_c}{C}\Big)^{\alpha_C},
$$

with fitted exponents around $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$ (constants $N_c, D_c, C_c$ depend on the tokenizer and data). In words: doubling model size multiplies loss by $2^{-0.076} \approx 0.95$, a 5% reduction, and that same 5% is available again at every doubling, as long as data and compute keep up. The trends held over more than six orders of magnitude.

### The compute identity $C \approx 6ND$

From [1.3](03-transformer.md), training costs about $6N$ FLOPs per token, so a run with $N$ parameters on $D$ tokens costs

$$
C \approx 6 N D \ \text{FLOPs}.
$$

This single equation links the three quantities: fix any two and the third follows. Sanity check: Llama 3 405B on 15.6T tokens gives $6 \times 405 \times 10^9 \times 15.6 \times 10^{12} \approx 3.8 \times 10^{25}$ FLOPs, the figure in the Llama 3 report.

### Kaplan vs Chinchilla

Given a budget $C$, how should we split it between $N$ and $D$?

- **Kaplan et al. (2020)** concluded that model size should grow much faster than data: $N_{\text{opt}} \propto C^{0.73}$, $D_{\text{opt}} \propto C^{0.27}$. Following this, GPT-3 (175B) was trained on only 300B tokens, under 2 tokens per parameter.
- **Hoffmann et al. (2022), "Chinchilla"**, trained over 400 models and found $N_{\text{opt}} \propto C^{0.5}$ and $D_{\text{opt}} \propto C^{0.5}$: parameters and tokens should be scaled **equally**, at roughly **20 tokens per parameter**. A main reason for the discrepancy was that Kaplan's runs used a learning-rate schedule whose length did not match each run's number of tokens, which underestimates what small models trained longer can do. To prove the point, Chinchilla (70B, 1.4T tokens) outperformed Gopher (280B, 300B tokens) with a similar compute budget.

Chinchilla also fits a parametric loss surface:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
$$

where $E$ is the irreducible loss (the entropy of text itself, which no model can beat), the second term is the penalty for a finite model and the third for finite data. The paper's fit is $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$. In words: loss approaches a floor, and you pay separately for too few parameters and too few tokens. (A 2024 replication by Besiroglu et al. found this particular fit inconsistent with the paper's other two estimation methods; their corrected fit agrees with the ~20 tokens/parameter rule. Use the rule of thumb, not the constants, for planning.)

**Deriving the optimal split from the rule.** If $D = rN$ with $r \approx 20$, substitute into $C = 6ND$: $C = 6rN^2$, so

$$
N_{\text{opt}} = \sqrt{\frac{C}{6r}} = \sqrt{\frac{C}{120}}, \qquad D_{\text{opt}} = 20\, N_{\text{opt}}.
$$

### Over-training for inference economics

Chinchilla-optimal means *lowest loss for a given training budget*. But lifetime cost is training plus inference, and inference costs $2N$ FLOPs per generated token forever. If you will serve trillions of tokens, a smaller model trained on far more data reaches the same loss with a slightly bigger training bill and a much smaller serving bill (Sardana et al., 2023, formalize this). Loss keeps improving well past 20 tokens per parameter; it just improves more slowly per FLOP. So modern models are deliberately **over-trained**: Llama 3 8B saw about 15T tokens (~1,900 tokens per parameter), and Qwen3's dense models were trained on 36T tokens.

### The data pipeline

Most pre-training tokens come from the web, usually from **Common Crawl**, a public archive of billions of web pages collected since 2008. Raw crawl data is mostly unusable: navigation menus, spam, boilerplate, machine-generated text, near-duplicate pages. A typical pipeline (FineWeb, DCLM, RefinedWeb and Dolma all document theirs) looks like this:

```text
WARC (raw HTML) → text extraction → URL filtering → language ID → heuristic filters
  → model-based quality filtering → deduplication (exact + MinHash) → PII removal
  → decontamination → domain mixing → tokenize
```

1. **Extraction.** Convert HTML to clean text, dropping menus, ads and boilerplate. Tools like `trafilatura` on the raw HTML (WARC files) produce noticeably better text than Common Crawl's pre-extracted text (WET files); FineWeb reports a clear quality difference.
2. **URL filtering.** Block lists of adult, spam and malware domains.
3. **Language identification.** A fast classifier (e.g. fastText's language ID model) labels each document's language with a confidence score; keep documents above a threshold for the target languages.
4. **Heuristic filters.** Cheap rules from C4 and Gopher: minimum and maximum word counts, mean word length, fraction of lines ending in punctuation, symbol-to-word ratio, fraction of repeated lines or n-grams, presence of "lorem ipsum" or "javascript must be enabled". Each rule removes a slice of junk.
5. **Model-based quality classifiers.** Train a classifier to recognize "good" text and keep the top-scoring documents. DCLM trained a simple fastText classifier with instruction-like and explanatory text (OpenHermes data and the ELI5 subreddit) as positives and random web text as negatives; this beat more complex filters. FineWeb-Edu had Llama-3-70B-Instruct rate about 450,000 pages for educational value on a 0–5 scale, trained a small classifier on those ratings, and kept pages scoring at least 3, which gave large gains on knowledge and reasoning benchmarks.
6. **Deduplication.** Exact duplicates are found by hashing documents (or lines/paragraphs). **Near-duplicates** (the same article with different ads, or a template with one word changed) need fuzzy matching, usually **MinHash with locality-sensitive hashing**, explained below. Lee et al. (2021) showed that deduplication reduces memorization and improves training efficiency.
7. **PII removal.** Mask emails, IP addresses and similar personal information.
8. **Decontamination.** Remove training documents that overlap with evaluation benchmarks (e.g. sharing a long n-gram with a test question; GPT-3 used 13-gram overlap) so benchmark scores measure generalization rather than memorization.

### MinHash deduplication, explained

Represent each document as its set of **shingles**, e.g. all 5-word sequences. The similarity of two documents $A$ and $B$ is their **Jaccard similarity** $J(A, B) = |A \cap B| / |A \cup B|$. Computing $J$ for all pairs of billions of documents is impossible, so we use two tricks:

**MinHash estimates Jaccard.** Pick a random hash function $h$ and compute $\min_{s \in A} h(s)$, the minimum hash over $A$'s shingles. Key fact:

$$
\Pr\Big[\min_{s \in A} h(s) = \min_{s \in B} h(s)\Big] = J(A, B).
$$

Why: under a random hash, every shingle in $A \cup B$ is equally likely to have the smallest hash. The two minima are equal exactly when that overall-minimum shingle lies in both sets, i.e. in $A \cap B$, which happens with probability $|A \cap B| / |A \cup B|$. So with $K$ independent hash functions, a document gets a **signature** of $K$ numbers, and the fraction of positions where two signatures agree estimates $J$.

**LSH finds candidates without comparing all pairs.** Split each signature into $b$ **bands** of $r$ values ($K = br$). Put each document into one hash bucket per band, keyed by that band's $r$ values. Two documents become a candidate pair if they land in the same bucket for at least one band. For a pair with Jaccard $J$, one band matches with probability $J^r$, so

$$
\Pr[\text{candidate}] = 1 - \big(1 - J^r\big)^{b}.
$$

This is an S-shaped curve in $J$: near 0 for dissimilar pairs, near 1 for similar ones, with the threshold set by $b$ and $r$. FineWeb uses 112 hash functions in $b = 14$ bands of $r = 8$, targeting documents that are at least about 75% similar. Only documents sharing a bucket are compared, so the cost is roughly linear in the number of documents.

### Data mixtures

The final corpus mixes sources: web text, code, math, scientific papers, books, multilingual text. The proportions strongly affect capabilities, and the best mix is found empirically with small proxy models and scaling-law-style experiments. The Llama 3 report gives its final pre-training mix as roughly 50% general knowledge, 25% mathematical and reasoning tokens, 17% code and 8% multilingual tokens. Mixes are often changed during training, upweighting high-quality sources toward the end.

### Annealing and mid-training

Near the end of pre-training, as the learning rate decays to zero, the model is especially sensitive to the data it sees. Labs exploit this with an **annealing** (or **mid-training**) phase: the last stretch of training uses a curated mix rich in high-quality web text, math, code, instruction-like and synthetic data. OLMo 2 documents a dedicated mid-training stage; Llama 3 annealed on high-quality data at the end and also used annealing on small amounts of a candidate dataset as a cheap way to measure that dataset's value. A long-context stage ([1.6](06-long-context-and-efficient-attention.md)) is usually also part of the late phase.

### Synthetic data

Strong existing models can generate or rewrite training data: textbook-style explanations and exercises (phi-1's "textbook quality" data), rephrasings of web pages into cleaner styles (Maini et al., 2024, "Rephrasing the Web"), and verified solutions to math and code problems. Synthetic data is now a standard part of pre-training mixes, especially in annealing. Risks: reduced diversity, inherited errors and biases of the generator, and benchmark contamination if the generator has seen test sets.

### Repeated data

High-quality text is finite. Muennighoff et al. (2023) trained models on repeated data and found that up to about 4 epochs, repeated tokens are almost as valuable as fresh ones; beyond that, returns diminish quickly and eventually extra epochs add almost nothing. This is why labs work hard on extracting more unique tokens (better extraction, more languages, PDFs, synthetic rephrasing) rather than looping over the same data many times.

## Code

**Compute-optimal sizing and fitting a power law.**

```python
import math, random

def chinchilla_optimal(C, tokens_per_param=20):
    """Compute-optimal (N, D) under C = 6*N*D and D = r*N  =>  N = sqrt(C / (6r))."""
    N = math.sqrt(C / (6 * tokens_per_param))
    return N, tokens_per_param * N

def fit_power_law(xs, ys):
    """Least-squares fit of y = a * x^(-b) via a straight line in log-log space."""
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((u - mx) * (v - my) for u, v in zip(lx, ly)) / sum((u - mx) ** 2 for u in lx)
    return math.exp(my - slope * mx), -slope            # a, b

for C in [1e21, 1e23, 1e24, 3.8e25]:
    N, D = chinchilla_optimal(C)
    print(f"C={C:.1e}: N={N/1e9:7.1f}B  D={D/1e12:6.2f}T")

# Synthetic "experiments": loss = 30 * N^-0.076 with 2% noise; recover the exponent.
random.seed(0)
Ns = [1e7, 3e7, 1e8, 3e8, 1e9, 3e9]
losses = [30 * N ** -0.076 * (1 + random.uniform(-0.02, 0.02)) for N in Ns]
a, b = fit_power_law(Ns, losses)
print(f"fitted: loss = {a:.1f} * N^-{b:.3f}; extrapolated loss at 1e11 params: {a * 1e11 ** -b:.3f}")
```

Real scaling-law fits work the same way, with more careful experimental design: each point is the best run at that size (with a tuned learning-rate schedule), and the fit is checked by holding out the largest runs. Note how 2% noise already shifts the fitted exponent from 0.076 to about 0.079, which matters a lot when extrapolating 30× beyond the data.

**MinHash + LSH near-duplicate detection.**

```python
import random, zlib

def shingles(text, n=5):
    """Set of word n-grams ("shingles") of a document."""
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

P = (1 << 61) - 1                                  # a large prime for universal hashing
random.seed(1)
HASHES = [(random.randrange(1, P), random.randrange(P)) for _ in range(112)]

def minhash(sh):
    """Signature: for each hash function, the minimum hash over all shingles."""
    xs = [zlib.crc32(s.encode()) for s in sh]
    return [min((a * x + b) % P for x in xs) for a, b in HASHES]

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)

def lsh_candidates(sigs, bands=14, rows=8):
    """Docs sharing ALL `rows` values in ANY band become candidate duplicates."""
    buckets, pairs = {}, set()
    for doc_id, sig in sigs.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            for other in buckets.setdefault(key, []):
                pairs.add((other, doc_id))
            buckets[key].append(doc_id)
    return pairs

base = ("pre-training data is collected from common crawl snapshots then extracted filtered "
        "deduplicated and mixed with code math books and papers before the final annealing "
        "phase where the learning rate decays and only the highest quality documents are used")
docs = {
    "a": base,
    "b": base.replace("final", "last"),             # near-duplicate (one word changed)
    "c": "language models are trained on trillions of tokens of filtered web text and code",
}
sh = {k: shingles(v) for k, v in docs.items()}
sigs = {k: minhash(s) for k, s in sh.items()}
est = sum(x == y for x, y in zip(sigs["a"], sigs["b"])) / len(HASHES)
print(f"true Jaccard(a,b)={jaccard(sh['a'], sh['b']):.2f}  MinHash estimate={est:.2f}")
print("candidate duplicate pairs:", lsh_candidates(sigs))
```

This prints a true Jaccard of 0.74, an estimate of 0.72, and flags (a, b) as a candidate pair. Changing a single word alters every 5-gram shingle that contains it (up to 5 of them), which is why the similarity of these short documents is 0.74 rather than near 1. Try other seeds: for $J = 0.74$ the formula gives $1 - (1 - 0.74^8)^{14} \approx 0.73$, so roughly one seed in four will *miss* this pair. LSH is probabilistic by design.

## Worked example

**Sizing a run with $C = 10^{24}$ FLOPs.** Using the Chinchilla rule:

- $N_{\text{opt}} = \sqrt{10^{24} / 120} = \sqrt{8.33 \times 10^{21}} \approx 9.1 \times 10^{10}$, i.e. about **91B parameters**.
- $D_{\text{opt}} = 20 \times 91\text{B} \approx$ **1.8T tokens**.
- Check: $6 \times 9.1 \times 10^{10} \times 1.83 \times 10^{12} \approx 1.0 \times 10^{24}$. ✓

**How over-trained is Llama 3 8B?** $C = 6 \times 8 \times 10^9 \times 15 \times 10^{12} = 7.2 \times 10^{23}$ FLOPs. A Chinchilla-optimal model for that budget would have $\sqrt{7.2 \times 10^{23} / 120} \approx 77\text{B}$ parameters and see 1.55T tokens. Meta instead spent the compute of a ~77B compute-optimal model on an 8B model. The trained 8B is somewhat worse than a 77B would be, but every generated token costs about $2 \times 8\text{B}$ instead of $2 \times 77\text{B}$ FLOPs, nearly 10× cheaper, and it fits on a single consumer GPU.

**Llama 3's own scaling law.** The Llama 3 report fit its own scaling laws on its data mix and predicted that for $3.8 \times 10^{25}$ FLOPs the compute-optimal model would have about 402B parameters trained on 16.55T tokens, about 41 tokens per parameter rather than 20; the chosen 405B/15.6T is close to that prediction. Compute-optimal ratios depend on the data and the setup, which is why labs refit them.

**MinHash LSH thresholds** with $b = 14$, $r = 8$: for $J = 0.5$, $\Pr[\text{candidate}] = 1 - (1 - 0.5^8)^{14} \approx 0.05$; for $J = 0.75$, $\approx 0.77$; for $J = 0.8$, $\approx 0.92$; for $J = 0.9$, $\approx 0.9996$. The curve rises steeply around 0.7–0.8, which is the intended threshold.

## In practice (2025–2026)

| model | parameters (total) | pre-training tokens (reported) |
|---|---|---|
| Llama 3 405B | 405B dense | 15.6T |
| DeepSeek-V3 | 671B MoE (37B active) | 14.8T |
| Kimi K2 | 1.04T MoE (32B active) | 15.5T |
| Qwen3 (all sizes) | 0.6B–235B | ~36T |
| Gemma 3 27B | 27B dense | 14T |

- **Token counts of 15–36T are standard**, far beyond Chinchilla-optimal for most of these sizes. For MoE models, "tokens per parameter" should be computed with *active* parameters for compute and with total parameters for capacity.
- **Open, documented pipelines** (FineWeb/FineWeb-Edu, DCLM, Dolma, OLMo 2's mixes) have made the recipe public; the remaining secrets are mostly in the mixture, synthetic data and annealing choices.
- **Model-based filtering is ubiquitous**, often using an existing LLM to label a sample and a small classifier to scale up.
- **Hyperparameter transfer**: labs tune learning rates and initializations on small models and transfer them using parameterizations such as μP (Yang et al., 2022), so that scaling-law experiments and the final run share optimal settings.
- **Data walls**: with high-quality web text increasingly exhausted, rephrasing, synthetic generation, multilingual expansion and multi-epoch training on the best subsets are the main levers.

## Common pitfalls / misconceptions

- **"Chinchilla says always use 20 tokens/param."** It says that is the *training-compute-optimal* ratio for its setup. If inference matters (it almost always does), train smaller models longer.
- **Extrapolating noisy fits far.** A small error in the exponent becomes a large error 100× out. Hold out your largest runs to validate the fit.
- **Comparing losses across tokenizers or data mixes** to evaluate a scaling law. The constants (and sometimes exponents) are specific to the setup.
- **Deduplicating too aggressively.** Removing every near-duplicate can drop valuable repeated content (licenses, common code idioms) and over-filter some domains; FineWeb found per-snapshot deduplication worked better than deduplicating across all snapshots.
- **Skipping decontamination** inflates benchmark numbers and misleads your own development decisions.
- **Assuming quality classifiers are neutral.** They encode the biases of whatever was labeled "good"; heavy filtering can reduce diversity, dialect coverage, or domain coverage.

## Key takeaways

- Loss falls as a power law in parameters, data and compute; small experiments predict big runs.
- $C \approx 6ND$; the Chinchilla compute-optimal split is about 20 tokens per parameter, giving $N_{\text{opt}} \approx \sqrt{C/120}$.
- Inference cost makes over-training rational: modern models see hundreds to thousands of tokens per parameter.
- A web-data pipeline is extraction → language ID → heuristics → model-based quality filtering → (MinHash) dedup → decontamination → mixing.
- MinHash estimates Jaccard similarity because the minimum hash collides with probability $J$; LSH banding turns that into a cheap near-duplicate filter.
- Late-stage annealing on curated and synthetic data, and up to ~4 epochs of repetition, are the main tools once fresh high-quality data runs short.

## Check your understanding

??? question "You have $6 \times 10^{23}$ FLOPs. What are the Chinchilla-optimal N and D?"
    $N = \sqrt{6 \times 10^{23} / 120} = \sqrt{5 \times 10^{21}} \approx 70.7\text{B}$ and $D \approx 1.41\text{T}$. That is essentially Chinchilla itself (70B, 1.4T).

??? question "Why did Kaplan et al. conclude models should grow faster than data?"
    Among other differences, their runs used learning-rate schedules not matched to each run's length, which made longer training of small models look less useful than it is. Chinchilla tuned the schedule per run and found $N$ and $D$ should scale equally.

??? question "Why is the probability that two MinHash values match equal to the Jaccard similarity?"
    Under a random hash, each element of $A \cup B$ is equally likely to have the minimum hash. The minima match exactly when that element is in both sets, which has probability $|A \cap B| / |A \cup B|$.

??? question "With $b = 20$ bands of $r = 5$, what is the candidate probability for $J = 0.6$?"
    $0.6^5 = 0.0778$; $1 - (1 - 0.0778)^{20} = 1 - 0.9222^{20} \approx 1 - 0.198 = 0.80$.

??? question "Why might a lab train a 7B model on 20T tokens even though it costs more compute than a compute-optimal model of the same loss?"
    Because serving costs scale with $N$. A smaller model is cheaper and faster per token, fits on cheaper hardware, and is used billions of times, so the extra one-time training cost is repaid many times over.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020 | Loss follows power laws in parameters, data and compute. Kick-started "scale is all you need". |
| :material-star: [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022 | Parameters and tokens should scale roughly equally (≈20 tokens/param for compute-optimal). |
| :material-book-open-variant: [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) — Besiroglu et al., 2024 | Re-fits Chinchilla's parametric loss and finds the published fit inconsistent with the other two methods; a useful lesson in fitting scaling laws. |
| :material-book-open-variant: [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) — Sardana et al., 2023 | Adds inference cost to scaling laws: why over-training smaller models is economical. |
| :material-book-open-variant: [Tensor Programs V: Zero-Shot Hyperparameter Transfer (μP)](https://arxiv.org/abs/2203.03466) — Yang et al., 2022 | A parameterization under which optimal learning rates transfer from small to large models: tune small, train big. |
| :material-book-open-variant: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023 | Repeating data up to ~4 epochs is almost as good as fresh data; beyond that returns vanish. |
| :material-star: [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — Penedo et al., 2024 | 15T-token open web corpus with documented filtering/dedup ablations, plus FineWeb-Edu. The best guide to web-data pipelines. |
| :material-star: [DataComp-LM (DCLM)](https://arxiv.org/abs/2406.11794) — Li et al., 2024 | A benchmark for data curation; shows a simple fastText quality classifier is a huge win. |
| :material-book-open-variant: [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Penedo et al., 2023 | Shows that well-filtered and deduplicated web data alone can match curated corpora. |
| :material-book-open-variant: [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159) — Soldaini et al., 2024 | A fully documented open pre-training corpus and toolkit. |
| :material-book-open-variant: [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — Lee et al., 2021 | Exact and near-duplicate removal reduces memorization and improves efficiency. |
| :material-book-open-variant: [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) — Gunasekar et al., 2023 | Small models trained on high-quality and synthetic "textbook" data punch far above their size. |
| :material-book-open-variant: [Rephrasing the Web](https://arxiv.org/abs/2401.16380) — Maini et al., 2024 | Rewriting web documents with an LLM into cleaner styles as a source of pre-training data. |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — OLMo Team (AI2), 2025 | Fully open recipe including **mid-training / annealing** on high-quality data at the end of pre-training. |
