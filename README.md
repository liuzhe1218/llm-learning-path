# LLM Learning Path · LLM 学习路线

A curated, bilingual (English / 中文) learning path for large language models — with paper links, summaries, hands-on projects and self-check questions.

一份中英双语的大语言模型学习路线，包含论文链接、要点总结、动手项目与自测题。

**🌐 Website / 网站:** https://liuzhe1218.github.io/llm-learning-path/ (toggle EN / 中文 in the header)

| Part | Topic | 主题 |
|---|---|---|
| 1 | [Pre-training & Model Architecture](docs/part1/index.md) | [预训练与模型架构](docs/part1/index.zh.md) |
| 2 | [Post-training, Alignment & Reasoning](docs/part2/index.md) | [后训练、对齐与推理能力](docs/part2/index.zh.md) |
| 3 | [Inference Systems & Infrastructure](docs/part3/index.md) | [推理系统与基础设施](docs/part3/index.zh.md) |
| 4 | [LLMs & Generative Modeling in Ads and Recommendation](docs/part4/index.md) | [LLM 与生成式建模在广告推荐中的应用](docs/part4/index.zh.md) |

## Run locally / 本地预览

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/mkdocs serve
```

Open http://127.0.0.1:8000. Pushing to `main` deploys to GitHub Pages via `.github/workflows/deploy.yml`.

## Contributing / 贡献

English pages are `docs/*.md`; Chinese pages are `docs/*.zh.md`. Please update both when adding a paper. Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) + [mkdocs-static-i18n](https://github.com/ultrabug/mkdocs-static-i18n).

## License

Content: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper summaries are original writing; papers belong to their authors.
