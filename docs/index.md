# LLM Learning Path

A curated, opinionated path through how modern large language models are **built**, **aligned**, and **served** — with links to the original papers and a short summary of what each one contributes and why it still matters.

!!! tip "Language / 语言"
    Use the language switcher (🌐 icon in the header) to toggle between **English** and **中文**. Every page exists in both languages.

## How the path is organized

| Part | Focus | You will be able to… |
|---|---|---|
| [**Part 1 · Pre-training & Architecture**](part1/index.md) | Transformer internals, tokenization, scaling laws, data, MoE, long context, efficient attention, distributed training | Explain every block of a 2025–2026 frontier model (e.g. DeepSeek-V3, Qwen3, Llama 4, gpt-oss) and why it is designed that way |
| [**Part 2 · Post-training & Reasoning**](part2/index.md) | SFT, RLHF, DPO & friends, RL with verifiable rewards (GRPO/DAPO/GSPO), reasoning models, test-time scaling, agentic/tool-use training, PEFT | Design a post-training recipe that turns a base model into a chat / reasoning / agent model |
| [**Part 3 · Inference Systems & Infra**](part3/index.md) | KV cache, PagedAttention, continuous batching, prefix caching, speculative decoding, quantization, prefill/decode disaggregation, MoE serving | Reason about latency/throughput/cost of serving and pick the right engine and optimizations |
| [**Part 4 · LLMs in Ads & Recommendation**](part4/index.md) | Industrial recsys/ads cascades, CTR models, scaling laws for recommenders (HSTU, Wukong, RankMixer), semantic IDs & generative retrieval, end-to-end generative recommenders (OneRec), LLMs as feature providers / user models, generative bidding | Explain how LLM-era ideas are rebuilding ranking and recommendation, and design an LLM-augmented or generative recommender |

## Suggested order

```text
Prerequisites ──► Part 1 (2–4 weeks) ──► Part 2 (2–3 weeks) ──► Part 3 (2–3 weeks) ──► Part 4 (2–3 weeks)
                       │                         │                       │                        │
                  build nanoGPT            fine-tune + GRPO          serve with vLLM/SGLang    semantic-ID generative recommender
```

- **Prerequisites:** linear algebra, probability, PyTorch basics, backprop. Good refreshers: Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and the free [Dive into Deep Learning](https://d2l.ai/) book.
- Each part is a **from-scratch course** split into chapters. Every chapter goes **intuition → how it works (math, derived step by step) → runnable code → worked example → what frontier systems do in 2025–2026 → pitfalls → takeaways → self-check questions**, and ends with **Further reading**: the original papers, for when you want to go deeper.
- Papers are tagged :material-star: **must-read**, :material-book-open-variant: **deep-dive**, or :material-new-box: **frontier**.

## How to read a paper efficiently

1. Read the abstract, look at Figure 1 and the main results table.
2. Ask: *what bottleneck does this remove?* (compute, memory, data, stability, latency…)
3. Skim the method; re-derive the one key equation by hand.
4. Check whether the idea survived — is it used in today's open models? The summaries here try to answer that for you.

## Contributing

Found a broken link, outdated claim, or a paper that should be here? Open an issue or PR. Each page is a plain Markdown file under `docs/` (`*.md` = English, `*.zh.md` = 中文).

*Last updated: November 2026.*
