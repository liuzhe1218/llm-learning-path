# 1.1 Language Modeling

!!! abstract "What you'll learn"
    - What a language model *is*, formally: a probability distribution over sequences of tokens.
    - How the chain rule of probability turns that into **next-token prediction**.
    - How to compute **cross-entropy** and **perplexity** by hand, and what their values mean.
    - Why n-gram models hit a wall and how neural LMs get past it.
    - How text is generated from a model: greedy decoding, temperature, top-k and top-p sampling.

## Why this matters / the problem

Every LLM you will meet in this course, from a 100-line toy GPT to a trillion-parameter MoE, is trained with the same objective: *given the text so far, predict the next token*. Pre-training (this whole Part 1) is nothing more than minimizing a single number, the average cross-entropy of that prediction over trillions of tokens. Before we discuss Transformers, tokenizers or GPU clusters, we need to be precise about what that number is, why minimizing it is a good idea, and how a model that only predicts one token at a time can produce whole essays.

## Intuition

Think of a language model as an extremely well-read autocomplete. Given "The capital of France is", it does not output one answer; it outputs a **probability for every possible next token**: maybe 0.8 for " Paris", 0.03 for " a", 0.01 for " located", and tiny amounts for the other ~100,000 tokens in its vocabulary.

To generate text you pick one token from that distribution, append it, and ask again. To **train** the model you show it real text and measure how surprised it was by each actual next token: assigning 0.8 to the right answer is good, 0.001 is bad. The training signal is "be less surprised by real text".

Why should that lead to anything resembling intelligence? Because to be unsurprised by the next token of a physics textbook, a legal contract, Python code or a dialogue, the model must implicitly learn grammar, facts, arithmetic, code semantics and something about how people reason. Prediction is a universal task: almost any skill that leaves a trace in text helps reduce the loss somewhere.

## How it works

### A language model is a distribution over sequences

Let a **vocabulary** $\mathcal{V}$ be a finite set of tokens (chapter [1.2](02-tokenization.md) explains where tokens come from), with size $V = |\mathcal{V}|$. A text is a sequence $x = (x_1, x_2, \dots, x_T)$ with each $x_t \in \mathcal{V}$. A **language model** assigns a probability $p(x_1, \dots, x_T)$ to every such sequence, and these probabilities sum to 1 over all sequences (an end-of-sequence token makes lengths finite).

Modeling the joint distribution directly is hopeless: with $V = 50{,}000$ and $T = 100$ there are $50{,}000^{100}$ possible sequences. The trick is to factor it.

### The chain rule turns it into next-token prediction

The **chain rule of probability** says that any joint distribution can be written as a product of conditionals:

$$
p(x_1, \dots, x_T) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2) \cdots = \prod_{t=1}^{T} p(x_t \mid x_{<t}),
$$

where $x_{<t} = (x_1, \dots, x_{t-1})$ is the **context** (or prefix) before position $t$. In words: the probability of a whole text equals the probability of its first token, times the probability of the second given the first, and so on. This is an identity, not an approximation. So a model that can output $p(x_t \mid x_{<t})$ for any prefix *is* a full language model. This left-to-right style is called **autoregressive** (each output feeds back as input).

A neural LM implements this with parameters $\theta$: a network $f_\theta$ reads the prefix and outputs a vector of **logits** $z \in \mathbb{R}^{V}$, one real number per token, which the **softmax** converts into probabilities:

$$
p_\theta(x_t = i \mid x_{<t}) = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

The softmax exponentiates each logit (making it positive) and divides by the sum (making them add to 1). Bigger logit, bigger probability.

### The training objective: maximum likelihood = cross-entropy

Given a training corpus, we choose $\theta$ to make the real text as probable as possible (**maximum likelihood**). Products of many small probabilities underflow, so we take logs, which turns the product into a sum, and flip the sign so we can minimize:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

This is the **average negative log-likelihood (NLL)** per token. It is also called the **cross-entropy loss**, because for each position it equals the cross-entropy $H(q, p) = -\sum_i q_i \log p_i$ between the "true" one-hot distribution $q$ (all mass on the token that actually occurred) and the model's distribution $p$: with a one-hot $q$, the sum collapses to $-\log p_{\text{correct}}$.

Some properties worth internalizing:

- If the model puts probability 1 on the correct token, the loss for that position is $-\log 1 = 0$. It can never be negative.
- If it puts probability $\varepsilon \to 0$ on the correct token, the loss goes to $+\infty$. Confident mistakes are punished hard.
- A model that knows nothing and predicts uniformly over $V$ tokens gets loss $\log V$. For $V = 50{,}000$ that is $\ln 50{,}000 \approx 10.82$ nats. Every LLM training curve starts near this value.

!!! note "Nats vs bits"
    With the natural log the unit is **nats**; with $\log_2$ it is **bits**. Divide nats by $\ln 2 \approx 0.693$ to get bits. PyTorch's `F.cross_entropy` uses natural logs.

In a Transformer, all $T$ positions are predicted in parallel in a single forward pass (the causal mask in [1.3](03-transformer.md) makes sure position $t$ cannot peek at $x_t$ or later). Training on a sequence of length $T$ therefore gives $T$ training examples at once. That parallelism is one reason Transformers replaced recurrent networks.

### Perplexity

**Perplexity (PPL)** is the exponential of the average NLL:

$$
\mathrm{PPL} = \exp\!\Big(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\Big) = \Big(\prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})\Big)^{-1/T}.
$$

In words: perplexity is the inverse of the geometric-mean probability the model gave to each correct token. It has a friendly interpretation as an **effective branching factor**: a perplexity of 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 tokens. The uniform model has perplexity exactly $V$.

!!! warning "Perplexities are only comparable with the same tokenizer"
    Loss and perplexity are *per token*. A tokenizer with longer tokens has fewer, harder predictions per sentence. To compare models with different tokenizers, normalize by something tokenizer-independent such as bits per byte (total bits divided by the number of UTF-8 bytes of the text).

### From n-grams to neural language models

Before neural networks, the standard LM was the **n-gram model**. It makes a **Markov assumption**: the next token depends only on the previous $n-1$ tokens,

$$
p(x_t \mid x_{<t}) \approx p(x_t \mid x_{t-n+1}, \dots, x_{t-1}),
$$

and estimates each conditional by counting: $p(w \mid u) = \mathrm{count}(u, w) / \mathrm{count}(u)$, where $u$ is the $(n-1)$-token context. A **bigram** model ($n = 2$) conditions on just one previous token.

N-grams hit two walls:

1. **Sparsity.** The table has $V^n$ entries. With $V = 50{,}000$, a trigram table has $1.25 \times 10^{14}$ cells; most contexts are never seen, so their counts are zero, and an unseen event gets probability 0 (infinite loss). Smoothing tricks (add-one, Kneser–Ney, back-off) patch this but do not fix it.
2. **No generalization across similar contexts.** Having seen "the cat sat on the mat" teaches an n-gram model nothing about "the dog sat on the rug", because "cat" and "dog" are just unrelated indices.

The **neural language model** (Bengio et al., 2003) fixed both. Each token gets a learned **embedding vector**, a dense vector in $\mathbb{R}^d$; similar tokens end up with similar vectors, so evidence about "cat" transfers to "dog". A neural network maps the context's embeddings to logits, so the parameter count grows with $d$ and the network size, not with $V^n$. Recurrent networks (RNNs, LSTMs) then removed the fixed window, and Transformers ([1.3](03-transformer.md)) made the computation parallel over positions and able to attend directly to any earlier token.

### Generating text: decoding strategies

At inference time we repeatedly compute $p_\theta(\cdot \mid x_{<t})$ and choose a token. The main options:

- **Greedy decoding:** always take $\arg\max_i p_i$. Deterministic and good for short factual answers, but prone to bland, repetitive loops in long text.
- **Sampling:** draw $x_t \sim p_\theta(\cdot \mid x_{<t})$. Diverse, but the long tail of thousands of individually unlikely tokens collectively gets a lot of mass, so sampling occasionally picks nonsense that derails the rest of the text.
- **Temperature $\tau$:** divide logits by $\tau > 0$ before the softmax, $p_i \propto e^{z_i/\tau}$. $\tau < 1$ sharpens the distribution (more confident, closer to greedy as $\tau \to 0$); $\tau > 1$ flattens it (more random, uniform as $\tau \to \infty$).
- **Top-k:** keep only the $k$ highest-probability tokens, renormalize, sample.
- **Top-p (nucleus) sampling** (Holtzman et al., 2019): keep the smallest set of tokens whose probabilities sum to at least $p$, renormalize, sample. Unlike top-k, the candidate set adapts: it is small when the model is confident and large when many continuations are plausible.

These are applied in sequence in practice (temperature, then top-k and/or top-p). Inference engines and more advanced decoding (speculative decoding, constrained decoding) are covered in [Part 3](../part3/index.md).

### Why "just predict the next token" yields general capabilities

The cross-entropy objective is *proper*: its expected value is minimized exactly when the model's distribution equals the true data distribution. So a model with ever-lower loss is, by definition, an ever-better model of the text-generating process, which for internet-scale text includes people solving equations, writing code, translating, and answering questions. Several observations make this concrete:

- **Prediction requires understanding.** To predict the next token of "17 × 23 = ", the model has to compute 391 (or memorize it). To predict the next line of a proof, it must follow the argument.
- **Tasks are just text.** GPT-2 showed that a model trained only on web text can translate or summarize when prompted in the right format ("TL;DR:"), because such patterns occur naturally on the web. GPT-3 (175B parameters) showed **in-context learning**: given a few examples in the prompt, a large enough model continues the pattern on a new input, with no weight updates.
- **Loss improves predictably with scale.** Chapter [1.7](07-scaling-laws-and-data.md) shows that loss falls as a smooth power law in parameters, data and compute, which is what justified building ever-larger models.

A base model is still not an assistant: it continues text, it does not follow instructions or refuse harmful requests. Turning it into one is the job of post-training ([Part 2](../part2/index.md)).

## Code

A complete bigram language model, trained by gradient descent with cross-entropy, plus a sampler supporting temperature, top-k and top-p. This is the same training loop as a GPT, just with a much simpler $f_\theta$ (a lookup table of logits).

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
text = "the cat sat on the mat. the dog sat on the rug. the cat ran. "
chars = sorted(set(text))                       # character-level vocabulary
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in text])    # [T]

# Bigram model: logits for next token depend only on the current token.
# W[i] holds the V logits for "next token given current token i".
W = torch.zeros(V, V, requires_grad=True)
opt = torch.optim.Adam([W], lr=0.1)

x, y = data[:-1], data[1:]                      # inputs and next-token targets
for step in range(300):
    logits = W[x]                               # [T-1, V]
    loss = F.cross_entropy(logits, y)           # mean NLL in nats
    opt.zero_grad(); loss.backward(); opt.step()
print(f"uniform loss = {torch.log(torch.tensor(V)).item():.3f}, "
      f"trained loss = {loss.item():.3f}, perplexity = {loss.exp().item():.2f}")

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None:                       # mask everything below the k-th largest
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:                       # nucleus: smallest set with mass >= top_p
        sorted_p, idx = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        keep = cum - sorted_p < top_p           # keep a token if mass *before* it < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(start="t", n=60, **kw):
    out = [stoi[start]]
    for _ in range(n):
        out.append(sample_next(W[out[-1]], **kw))
    return "".join(itos[i] for i in out)

print(generate(temperature=0.7, top_p=0.9))
```

Running it prints a trained loss of about 0.75 nats against a uniform $\ln V \approx 2.71$, and a string such as "the t the sathe cathe og. rug." Fragments of the training text appear, but there is no coherence beyond a couple of characters, which is all a model that only sees the previous character can achieve. Chapter [1.3](03-transformer.md) replaces the lookup table with a Transformer that sees the whole prefix.

## Worked example

**Cross-entropy and perplexity by hand.** A model reads the three-token text "the cat sat" and assigns these probabilities to the correct tokens:

| position | context | correct token | $p_\theta$ | $-\ln p_\theta$ |
|---|---|---|---|---|
| 1 | (start) | the | 0.05 | 2.996 |
| 2 | the | cat | 0.01 | 4.605 |
| 3 | the cat | sat | 0.20 | 1.609 |

- Total NLL $= 2.996 + 4.605 + 1.609 = 9.210$ nats.
- Average loss $= 9.210 / 3 = 3.070$ nats per token $= 3.070 / 0.693 \approx 4.43$ bits per token.
- Perplexity $= e^{3.070} \approx 21.5$. Check with the other formula: $(0.05 \times 0.01 \times 0.20)^{-1/3} = (10^{-4})^{-1/3} \approx 21.5$. The model is "as confused as a uniform choice among ~21.5 tokens".
- A uniform model over $V = 50{,}000$ tokens would score $\ln 50{,}000 \approx 10.82$ nats and perplexity 50,000.

**Temperature and top-p.** Suppose the logits over a 3-token vocabulary are $z = (2.0, 1.0, 0.1)$.

| temperature | scaled logits | probabilities |
|---|---|---|
| $\tau = 1$ | (2.0, 1.0, 0.1) | (0.659, 0.242, 0.099) |
| $\tau = 0.5$ | (4.0, 2.0, 0.2) | (0.864, 0.117, 0.019) |
| $\tau = 2$ | (1.0, 0.5, 0.05) | (0.502, 0.304, 0.194) |

For $\tau=1$: $e^{2} = 7.389$, $e^{1} = 2.718$, $e^{0.1} = 1.105$, sum $11.212$, so $p = (0.659, 0.242, 0.099)$. With top-p $= 0.9$, sort descending: cumulative mass is $0.659$, then $0.901 \ge 0.9$, so we keep the first two tokens and renormalize to $(0.731, 0.269, 0)$. With top-k $= 1$ we recover greedy decoding.

## In practice (2025–2026)

- **The objective has not changed.** Frontier base models (Llama 3, DeepSeek-V3, Qwen3, Kimi K2) are all trained with next-token cross-entropy on causal Transformers. Additions are auxiliary: DeepSeek-V3 adds a **multi-token prediction** loss that also predicts tokens further ahead (chapter [1.9](09-frontier-model-case-studies.md)), and MoE models add load-balancing terms ([1.5](05-mixture-of-experts.md)).
- **Loss values.** Training curves start near $\ln V$ (about 11.8–12.5 nats for 128K–262K vocabularies), drop steeply in the first few thousand steps, then decline slowly for the rest of training. The final value depends heavily on the tokenizer and data mix, so never compare raw losses across labs.
- **Vocabularies are large** (128K for Llama 3 and DeepSeek-V3, ~150K for Qwen3, ~200K for gpt-oss, ~262K for Gemma 3), so the softmax over $V$ is itself a noticeable cost; see [1.2](02-tokenization.md).
- **Decoding defaults** for chat models typically combine a temperature around 0.6–1.0 with top-p around 0.9–0.95; model cards often publish recommended values. Reasoning models are usually sampled rather than decoded greedily, because greedy decoding can fall into repetition loops in very long outputs.

## Common pitfalls / misconceptions

- **"Perplexity 10 is better than perplexity 12" across different tokenizers.** Not necessarily; normalize per byte or per word.
- **Confusing the loss on training data with generalization.** A model can memorize a small corpus to near-zero loss (our bigram toy can't, but a big network can). Always track held-out loss.
- **Thinking temperature 0 is a special mode.** It is the limit of greedy decoding; implementations special-case it because dividing by zero is undefined.
- **Assuming a model "knows" the answer it samples.** Sampling draws from a distribution; the same prompt can produce different answers, and a low-probability wrong token can be chosen.
- **Off-by-one targets.** Inputs are $x_{1:T-1}$, targets are $x_{2:T}$. Forgetting the shift trains the model to copy its input.

## Key takeaways

- A language model is a probability distribution over token sequences; the chain rule makes it a product of next-token conditionals.
- Training minimizes average negative log-likelihood (cross-entropy); a uniform model scores $\log V$, a perfect one 0.
- Perplexity $= e^{\text{loss}}$ is an effective branching factor and is only comparable with the same tokenizer.
- Neural LMs beat n-grams by sharing statistical strength through embeddings instead of counting exact contexts.
- Generation is repeated sampling; temperature, top-k and top-p trade diversity against coherence.
- Because almost every skill leaves a trace in text, better next-token prediction implies better general capability, and that improves predictably with scale.

## Check your understanding

??? question "A model's average loss on a test set is 2.3 nats per token. What is its perplexity, and its loss in bits?"
    Perplexity $= e^{2.3} \approx 9.97$, i.e. about 10. In bits: $2.3 / \ln 2 \approx 3.32$ bits per token.

??? question "Why is the chain-rule factorization not an approximation, while the n-gram model is?"
    The chain rule $p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$ is an exact identity for any distribution. The n-gram model additionally *assumes* that $p(x_t \mid x_{<t})$ only depends on the last $n-1$ tokens, which throws information away.

??? question "What happens to sampling as temperature goes to 0 and to infinity?"
    As $\tau \to 0$, the largest logit dominates and sampling becomes greedy (argmax). As $\tau \to \infty$, all scaled logits approach 0 and the distribution becomes uniform over the vocabulary.

??? question "Why does top-p adapt better than top-k?"
    Top-k always keeps exactly $k$ tokens. When the model is confident (one token has 0.97), $k = 50$ still admits 49 unlikely tokens; when many tokens are plausible, $k$ may be too few. Top-p keeps however many tokens are needed to cover probability mass $p$, so the candidate set shrinks and grows with the model's uncertainty.

??? question "A freshly initialized model with a 128,000-token vocabulary reports an initial loss of 11.76. Is that a bug?"
    No. $\ln 128{,}000 \approx 11.76$, which is exactly the uniform-prediction loss. A random initialization with small output logits predicts nearly uniformly.

## Further reading

| Paper | Why read it |
|---|---|
| :material-book-open-variant: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Bengio et al., 2003 | The first neural LM with learned word embeddings; the argument against n-gram sparsity is still the clearest motivation for neural LMs. |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | Decoder-only LM + pre-norm + large web corpus → zero-shot task transfer. Establishes the "just scale next-token prediction" recipe. |
| :material-star: [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020 | 175B parameters; in-context (few-shot) learning emerges with scale. Read for the framing and evaluation methodology. |
| :material-book-open-variant: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al., 2019 | Why greedy and pure sampling both produce bad text, and the introduction of nucleus (top-p) sampling. |
