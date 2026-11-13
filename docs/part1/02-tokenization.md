# 1.2 Tokenization

!!! abstract "What you'll learn"
    - Why LLMs operate on **subword tokens** rather than characters or words.
    - How the **Byte-Pair Encoding (BPE)** training algorithm works, traced merge by merge on a toy corpus.
    - What **byte-level BPE** adds, and how encoding and decoding work.
    - How vocabulary size trades sequence length against embedding size and multilingual coverage.
    - Tokenizer quirks (numbers, whitespace, CJK text) and how they leak into model behavior.

## Why this matters / the problem

A neural network consumes numbers, not text. Chapter [1.1](01-language-modeling.md) assumed a vocabulary $\mathcal{V}$ of tokens; this chapter decides what those tokens are. The choice matters more than it looks:

- It fixes **how many prediction steps** a text costs. Attention cost grows with sequence length (quadratically, see [1.6](06-long-context-and-efficient-attention.md)), and generation time grows linearly with the number of output tokens.
- It fixes the size of the **embedding and output layers** ($V \times d$ parameters each).
- It decides what the model can "see". A model that receives "2025" as one opaque token must learn its digits indirectly; a model that sees "2", "0", "2", "5" can do arithmetic digit by digit.
- It determines fairness across languages: the same sentence can cost several times more tokens in one language than another.

The tokenizer is trained once, before the model, and is frozen for the model's lifetime. Getting it wrong is expensive.

## Intuition

There are three obvious granularities, and each fails in a different way:

| unit | vocabulary | sequence length | problem |
|---|---|---|---|
| characters / bytes | tiny (256 bytes) | very long | the model must spend capacity and compute assembling words from letters |
| words | huge, open-ended | short | any unseen word ("ChatGPT", a typo, a new name) is out-of-vocabulary; morphology ("run", "running") is not shared |
| **subwords** | chosen (32K–262K) | medium | the compromise everyone uses |

Subword tokenizers keep frequent strings ("the", " because", "ing") as single tokens and break rare words into reusable pieces ("tokenization" → "token" + "ization"). Nothing is ever out-of-vocabulary because, in the worst case, a word is spelled out with small pieces. BPE finds these pieces with a simple compression idea: **repeatedly glue together the most frequent adjacent pair of symbols**.

## How it works

### BPE training

Byte-Pair Encoding was a data-compression algorithm before Sennrich et al. (2015) adapted it to NLP. Training takes a corpus and a target vocabulary size $V$:

1. **Pre-tokenize**: split text into chunks (roughly words, with punctuation and spaces handled by rules). Merges never cross chunk boundaries. Count how often each chunk occurs.
2. **Initialize** the vocabulary with all base symbols (characters, or the 256 byte values in byte-level BPE) and write each chunk as a sequence of base symbols.
3. **Count** every adjacent pair of symbols across all chunks, weighted by chunk frequency.
4. **Merge** the most frequent pair $(a, b)$ into a new symbol $ab$: add it to the vocabulary, record the rule "$a\ b \to ab$", and rewrite every occurrence.
5. **Repeat** steps 3–4 until the vocabulary has $V$ symbols.

The output is the vocabulary plus an **ordered list of merge rules**. The order matters: encoding replays the merges in the order they were learned.

### Encoding and decoding

To **encode** new text: pre-tokenize, turn each chunk into base symbols, then repeatedly apply the *earliest-learned* merge rule that matches any adjacent pair in the chunk, until no rule applies. Map each final symbol to its integer id.

To **decode**: map ids back to their strings (or byte strings) and concatenate. Decoding is a lookup; encoding is the only non-trivial direction.

### Byte-level BPE

If the base symbols are Unicode characters, the base vocabulary is large (Unicode assigns about 150,000 characters) and an unseen character (an emoji, a rare CJK character) still breaks the "never out-of-vocabulary" promise. GPT-2 introduced **byte-level BPE**: first encode text as UTF-8 bytes, so the base vocabulary is exactly the 256 byte values, and run BPE over bytes. Any string in any language, and even invalid text, can be represented. Common multi-byte sequences (a Chinese character is 3 UTF-8 bytes, "é" is 2) get merged if they are frequent enough.

The price: a single byte is not always valid UTF-8 on its own, so a model can emit a token sequence whose bytes do not decode cleanly. Decoders handle this by buffering bytes or replacing invalid sequences.

### Pre-tokenization

Running BPE over raw text would happily learn merges such as "dog." or "the cat" that straddle punctuation and word boundaries, wasting vocabulary on combinations. GPT-style tokenizers therefore first split text with a regular expression into chunks, and BPE merges only within a chunk. The GPT-2 pattern roughly splits into:

- an optional leading space followed by letters (" hello", "world"),
- an optional leading space followed by digits (later tokenizers limit digit runs to 1–3 characters),
- an optional leading space followed by punctuation (" ...", "!"),
- runs of whitespace,
- common English contractions ("'s", "'t", "'re").

So `"Hello world, it's 2025!"` becomes `["Hello", " world", ",", " it", "'s", " 2025", "!"]` before any merges (with GPT-2's rules; newer patterns would split " 2025" further). Note that the space belongs to the *following* word; this is why " world" and "world" are different tokens.

### Special tokens

Besides learned tokens, every vocabulary reserves **special tokens** that never come out of BPE: an end-of-text marker (e.g. `<|endoftext|>`) placed between documents during pre-training, and later markers for chat roles, tool calls or images. They are matched before pre-tokenization and mapped directly to reserved ids. A user typing the literal string "<|endoftext|>" must not be able to inject the real control token, so production tokenizers treat special-token parsing as an explicit opt-in.

### Other subword algorithms

- **WordPiece** (BERT) is BPE-like but picks the merge that most increases the likelihood of the training data rather than the raw most frequent pair.
- **Unigram LM** (Kudo, 2018) works top-down: start with a large candidate vocabulary and prune pieces whose removal hurts a unigram language model's likelihood least. It can produce several segmentations of the same word with probabilities, which enables "subword regularization".
- **SentencePiece** is a library, not an algorithm: it implements BPE and Unigram and treats the input as a raw stream (spaces become a visible symbol "▁"), so it needs no language-specific pre-tokenizer. Llama 1 and 2 used SentencePiece BPE with a 32K vocabulary.

### Vocabulary-size trade-offs

Let $V$ be the vocabulary size and $d$ the model's hidden size.

- **Sequence length.** Larger $V$ means more and longer merges, so the same text becomes fewer tokens. Fewer tokens means less attention compute, more text per context window, and faster generation.
- **Parameters.** The input embedding matrix is $V \times d$ and the output (unembedding) matrix is $d \times V$ unless they are tied ([1.4](04-modern-block.md)). For Llama 3 8B ($V = 128{,}256$, $d = 4096$) each matrix has $128{,}256 \times 4096 \approx 525\text{M}$ parameters, so the two together are about 1.05B of the model's 8B.
- **Compute in the output layer.** Every position computes $V$ logits: $2dV$ FLOPs per token for the output projection. For small models this is a large fraction of total compute.
- **Rare tokens are undertrained.** Each token's embedding only gets gradient when that token occurs. With a huge $V$, many tokens are rare and their embeddings are poorly learned.
- **Multilinguality.** A vocabulary trained mostly on English allocates few merges to other scripts, which then fragment into many tokens. Larger vocabularies, trained on multilingual data, are the main fix; this is one reason vocabularies grew from 32K (Llama 2) to 128K (Llama 3), ~150K (Qwen), ~200K (gpt-oss) and ~262K (Gemma 3).

## Code

A complete, runnable byte-level BPE tokenizer: training, encoding and decoding, in about 60 lines. It follows the structure of GPT-2-style tokenizers (regex pre-tokenization, then BPE on UTF-8 bytes) with a simplified regex.

```python
import re
from collections import Counter

# Simplified GPT-2-style pre-tokenizer: optional leading space + letters,
# digits in runs of 1-3, punctuation runs, whitespace.
PAT = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+")

def pair_counts(chunks):
    counts = Counter()
    for ids, freq in chunks.items():
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += freq
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return tuple(out)

class ByteBPE:
    def train(self, text, vocab_size):
        assert vocab_size >= 256
        # 1) pre-tokenize, 2) each chunk -> tuple of UTF-8 byte values, with its count
        chunks = Counter(tuple(c.encode("utf-8")) for c in PAT.findall(text))
        self.merges = {}                                   # (a, b) -> new id, in learned order
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes
        for new_id in range(256, vocab_size):
            counts = pair_counts(chunks)
            if not counts:
                break
            best = max(counts, key=counts.get)             # most frequent adjacent pair
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunks = Counter({merge(ids, best, new_id): f for ids, f in chunks.items()})
        return self

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            # apply the earliest-learned merge present in the chunk
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text):
        out = []
        for chunk in PAT.findall(text):
            out += self._encode_chunk(tuple(chunk.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

corpus = "low lower lowest newest widest " * 20 + "the newest model is the lowest risk. 你好世界 2025"
tok = ByteBPE().train(corpus, vocab_size=280)
ids = tok.encode("the lowest newest 你好")
print(len("the lowest newest 你好".encode()), "bytes ->", len(ids), "tokens:", ids)
print([tok.vocab[i] for i in ids])
assert tok.decode(ids) == "the lowest newest 你好"
```

Output: 24 bytes become 12 tokens. " lowest" and " newest" are single tokens (they were frequent in the training text), "the" stays as three bytes (it was too rare to earn merges with only 24 merges available), and each Chinese character stays as its 3 raw UTF-8 bytes because the corpus had too little Chinese. That last point is the multilingual problem in miniature.

!!! tip "Making it fast"
    This implementation recounts all pairs after every merge, which is $O(\text{merges} \times \text{corpus})$. Production trainers (Hugging Face `tokenizers`, SentencePiece) update pair counts incrementally and are written in Rust or C++. For encoding at inference time, OpenAI's `tiktoken` is a fast BPE encoder.

## Worked example

We train BPE on the classic toy corpus from Sennrich et al.: the words "low" ×5, "lower" ×2, "newest" ×6, "widest" ×3. We work at the character level and append an end-of-word marker "\_" so that "est\_" (a word ending) can differ from "est" inside a word. Ties are broken by whichever pair was seen first.

**Initial state** (each word as characters, with its count):

```text
l o w _          ×5
l o w e r _      ×2
n e w e s t _    ×6
w i d e s t _    ×3
```

**Pair counts.** "e s" occurs in newest (6) and widest (3) = 9; likewise "s t" = 9 and "t \_" = 9. "w e" = 2 (lower) + 6 (newest) = 8. "l o" = "o w" = 5 + 2 = 7. The winner is "e s" with 9.

| merge # | pair merged | count | corpus after the merge |
|---|---|---|---|
| 1 | e + s → es | 9 | `n e w es t _`, `w i d es t _` |
| 2 | es + t → est | 9 | `n e w est _`, `w i d est _` |
| 3 | est + \_ → est\_ | 9 | `n e w est_`, `w i d est_` |
| 4 | l + o → lo | 7 | `lo w _`, `lo w e r _` |
| 5 | lo + w → low | 7 | `low _`, `low e r _` |
| 6 | n + e → ne | 6 | `ne w est_` |
| 7 | ne + w → new | 6 | `new est_` |
| 8 | new + est\_ → newest\_ | 6 | `newest_` |

After 8 merges the vocabulary is the base characters plus {es, est, est\_, lo, low, ne, new, newest\_}. The algorithm discovered the suffix "est" and the stem "low" with no linguistic knowledge, purely from frequency.

**Encoding an unseen word, "lowest".** Start from `l o w e s t _` and replay merges in order:

1. merge 1 (e s): `l o w es t _`
2. merge 2 (es t): `l o w est _`
3. merge 3 (est \_): `l o w est_`
4. merge 4 (l o): `lo w est_`
5. merge 5 (lo w): `low est_`
6. merges 6–8 do not apply.

Result: **[low, est\_]**, two tokens for a word never seen in training, composed of meaningful pieces.

**What vocabulary size costs.** Take a small model with $d = 1024$, about 0.6B parameters in total, and a Qwen-sized vocabulary of $V = 151{,}936$.

- Embedding matrix: $151{,}936 \times 1024 \approx 156\text{M}$ parameters, about a quarter of the model. Tying input and output embeddings ([1.4](04-modern-block.md)) avoids paying this twice.
- Output projection compute: $2dV \approx 311\text{M}$ FLOPs per token, against a whole forward pass of roughly $2N \approx 1.2$ GFLOPs: about a quarter of all compute goes into producing logits.
- For a 70B model with $d = 8192$ the same vocabulary costs $2 \times 8192 \times 151{,}936 \approx 2.5$ GFLOPs per token, under 2% of its $\approx 140$ GFLOPs forward pass.

Large vocabularies are therefore cheap for large models and expensive for small ones, which is why the smallest models in a family often share the big vocabulary but tie their embeddings.

## In practice (2025–2026)

- **Byte-level BPE dominates.** GPT-2 onward, Llama 3 (128K vocabulary, built on OpenAI's `tiktoken` vocabulary plus extra tokens for other languages), Qwen (~150K), DeepSeek-V3 (128K) and gpt-oss (`o200k_harmony`, ~200K) all use byte-level BPE. Gemma uses a SentencePiece vocabulary of about 262K tokens.
- **Bigger vocabularies pay off.** The Llama 3 report says its new tokenizer improved compression on English from 3.17 to 3.94 characters per token compared with Llama 2's, meaning the model reads more text for the same compute.
- **Digit handling is a deliberate choice.** Llama 1/2 split numbers into individual digits. GPT-4-era tokenizers (and Llama 3's) split digit runs into chunks of up to 3. Either way, the goal is to stop arbitrary multi-digit numbers from becoming single opaque tokens.
- **Special tokens** such as `<|begin_of_text|>`, `<|end_of_text|>` and chat-role markers are added to the vocabulary outside BPE. Chat templates built from them are a post-training concern ([Part 2](../part2/index.md)).
- **Tokenizer-free models** that operate directly on bytes with learned dynamic grouping are an active research area, but no frontier model has replaced subword tokenization yet.

## Common pitfalls / misconceptions

- **Leading spaces are part of tokens.** In GPT-style tokenizers " hello" and "hello" are different tokens. A prompt that ends with a trailing space forces the model to continue with an unusual token that has no leading space, and output quality drops.
- **Numbers are tokenized inconsistently.** In GPT-2's tokenizer, "1234" and "1235" might split differently, which makes arithmetic harder to learn. Counting characters ("how many r's in strawberry?") is hard for the same reason: the model sees tokens, not letters.
- **Whitespace in code.** Early tokenizers spent one token per space of indentation; modern ones have tokens for runs of spaces and tabs.
- **CJK and other scripts cost more.** With byte-level BPE, a Chinese character that was not merged costs 3 tokens. Petrov et al. (2023) measured large differences in token counts for the same content across languages, which affects cost, latency and effective context length.
- **Glitch tokens.** A token that was frequent in the tokenizer's training data but rare in the model's training data (for example a username from a scraped forum) has an almost untrained embedding, and prompting with it can produce bizarre output. Train the tokenizer on data representative of the model's corpus.
- **Changing the tokenizer means retraining.** Token ids are baked into the embedding matrix; you cannot swap tokenizers on a trained model without at least re-learning the embeddings.

## Key takeaways

- Subword tokenization balances vocabulary size against sequence length and never produces out-of-vocabulary tokens.
- BPE training = repeatedly merge the most frequent adjacent pair; encoding replays the learned merges in order.
- Byte-level BPE starts from the 256 byte values, so it can encode any string in any language.
- Larger vocabularies shorten sequences and help multilingual text, at the cost of bigger embedding/output matrices and rarely-trained tokens.
- Tokenizer quirks (spaces, digits, scripts) show up directly as model quirks; many "LLMs can't count letters" failures are tokenization artifacts.

## Check your understanding

??? question "Why does BPE need an ordered list of merges rather than just the final vocabulary?"
    The same string can be segmented in many ways using the vocabulary's tokens. Replaying merges in the learned order gives a deterministic segmentation that matches what the model saw during training. (Greedy longest-match over the vocabulary would give different, inconsistent segmentations.)

??? question "With byte-level BPE, can any input string produce an unknown token?"
    No. Every string is a sequence of bytes, and all 256 byte values are in the base vocabulary, so in the worst case a string is encoded byte by byte.

??? question "Doubling the vocabulary from 64K to 128K with $d = 4096$ adds how many parameters, if input and output embeddings are untied?"
    Each matrix grows by $64{,}000 \times 4096 \approx 262\text{M}$ (using 64K = 64,000 for round numbers), so both together add about 524M parameters. Whether it is worth it depends on how much shorter the sequences become.

??? question "In the worked example, why did we add the end-of-word marker '\_'?"
    It lets BPE distinguish a suffix at the end of a word ("est\_" in "newest") from the same letters inside a word. Byte-level BPE achieves a similar effect differently: pre-tokenization attaches the leading space to a word, so " est" at a word start differs from "est" mid-word.

??? question "A model is bad at reversing words letter by letter. Is that a reasoning failure?"
    Often it is largely a tokenization artifact: the model receives multi-character tokens and has to have memorized how each token is spelled. Splitting the word into single characters in the prompt (e.g. separated by spaces) usually helps a lot.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2015 | Introduced BPE for NLP: start from characters, repeatedly merge the most frequent pair. The basis of nearly every LLM tokenizer today; the toy corpus in this chapter comes from here. |
| :material-book-open-variant: [SentencePiece](https://arxiv.org/abs/1808.06226) — Kudo & Richardson, 2018 | Language-agnostic tokenizer that treats text as raw Unicode (whitespace included), popular for multilingual and CJK-heavy models. |
| :material-book-open-variant: [Subword Regularization](https://arxiv.org/abs/1804.10959) — Kudo, 2018 | The Unigram LM tokenizer and sampling multiple segmentations during training. |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | Section 2.2 introduces byte-level BPE and explains the pre-tokenization rules. |
| :material-book-open-variant: [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) — Petrov et al., 2023 | Measures how token counts for the same content differ across languages, and the cost/latency consequences. |
