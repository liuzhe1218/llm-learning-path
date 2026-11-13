# Authoring guide

This site is a **from-scratch course**, not a paper list. Every chapter must teach the topic so that a reader with only the prerequisites (linear algebra, probability, Python/PyTorch basics) can understand it **without opening any paper**. Papers are supplementary "go deeper" material at the end.

## File layout

```
docs/
  index.md / index.zh.md                 # site home
  partN/index.md / index.zh.md           # part overview
  partN/NN-slug.md / NN-slug.zh.md       # chapters (EN / 中文)
```

`*.md` = English, `*.zh.md` = Chinese (mkdocs-static-i18n suffix mode). Both must exist and mirror each other 1:1.

## Part overview page (`partN/index.md`)

1. H1 title (e.g. `# Part 1 · Pre-training & Model Architecture`)
2. What this part covers and why it matters (2–3 paragraphs)
3. Prerequisites and estimated time
4. Chapter map: a table linking every chapter (relative links like `01-language-modeling.md`) with a one-line description
5. A "mental model" diagram (ASCII in a ```text block, or a Mermaid-free table) showing how the chapters connect
6. How to study this part (suggested order, which hands-on projects to do)

## Chapter template (`partN/NN-slug.md`)

```markdown
# N.M Chapter Title

!!! abstract "What you'll learn"
    - 3–5 concrete learning outcomes

## Why this matters / the problem
Motivate with the bottleneck or question this chapter answers.

## Intuition
Plain-language explanation, analogies, a small diagram.

## How it works
The mechanics, step by step. Define every symbol before using it.
Math in $inline$ and $$display$$ form. Derive key results, don't just state them.
Track tensor shapes explicitly, e.g. `x: [batch, seq, d_model]`.

## Code
Minimal, runnable, commented PyTorch/Python (no pseudo-APIs). 20–80 lines per block.

## Worked example
Concrete numbers: e.g. compute a KV-cache size, FLOPs, a BPE merge, a GRPO advantage, an auction price.

## In practice (2025–2026)
What frontier systems actually do today and why; common design choices; numbers from real models.

## Common pitfalls / misconceptions

## Key takeaways
3–6 bullets.

## Check your understanding
3–5 `??? question "..."` collapsible Q&As (answers inside, indented 4 spaces).

## Further reading
Table: | Paper | Why read it | with :material-star: must-read · :material-book-open-variant: deep-dive · :material-new-box: frontier tags.
```

Use admonitions where useful: `!!! note`, `!!! tip`, `!!! warning`, `!!! example`, collapsible `??? info "Derivation"` for long proofs.

## Style rules

- Teach, don't list. Prose explanations first; each concept builds on the previous chapter (link back with relative links like `../part1/03-transformer.md`).
- Every formula is followed by a sentence in words saying what it means.
- Prefer one fully worked numeric example over three vague ones.
- Code must be correct and self-contained; keep it short enough to type in.
- Be accurate. Never invent numbers, paper titles, or links. Only cite numbers you are sure of, and attribute them ("DeepSeek-V3 reports…").
- **Links:** reuse arXiv links already on the site (they were verified). Any new arXiv link must be verified against the arXiv API (`https://export.arxiv.org/api/query?id_list=<id>`) for the exact title before use. Non-arXiv links must return HTTP 200.
- Length: a chapter is typically 250–600 lines of Markdown.
- Chinese pages: fluent technical Chinese; keep paper titles, URLs, code, math and model names unchanged; keep common English terms (token, KV cache, RoPE, MoE…) with Chinese on first use.

## Build

```bash
.venv/bin/mkdocs build --strict
```
