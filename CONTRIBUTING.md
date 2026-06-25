# Contributing to GenAI Engineering Hub

Thanks for your interest in contributing. This document explains how to add to or improve the hub.

---

## 🎯 Scope

This hub covers **modern Generative AI engineering** end-to-end:

| In scope | Out of scope |
|---|---|
| LLM/VLM architecture (2023–2026+) | Pre-Transformer NLP (LSTM/GRU era, except for reference) |
| Open-weight AND closed-source models | Vendor product comparisons / "best LLM" rankings |
| Attention / serving kernels (production-grade) | Toy demos / single-call examples |
| Self-hosted and open-weight serving patterns | Pure marketing material |
| Papers with clear engineering impact | Speculative / unverified claims |

> Note: RAG, agent, AWS production, security and end-to-end blueprint sections (7-12) were moved to an external production repo. This hub now focuses on the **core research / engineering layers** (architecture → serving → finetuning → alignment).

---

## 📝 How to Add Content

### Add a new model to the model map

1. Find the relevant section in `docs/01-architecture/README.md` (or the matching `examples/` README)
2. Add a row to the relevant table with:
   - Model name + family
   - One-line "what changed / why interesting"
   - Primary source (arXiv, official report, blog)
   - Maturity tag: `main` / `watchlist`
3. If it's a `main` entry, also add it to `end_to_end_*_master_notebook.ipynb` in the model map section.

### Add a new fine-tuning / alignment pattern

1. Open an issue first with the pattern name + 2-3 sentence motivation
2. Once approved:
   - Add the conceptual explanation to the relevant `docs/0X-*/README.md`
   - Add a runnable script in `examples/0X_*/`
   - If it's a major new pattern, add a section to the master notebook

---

## ✅ Style Rules

### Tables

- Always include a header row
- Use pipe alignment (`---:`) for numeric columns
- Keep cells concise — link out for details
- Use `main` / `watchlist` columns where maturity matters

### Links

- Prefer **official sources** (arXiv, official model report, official blog)
- If using a blog post, mark it `[blog]` in the link text
- For closed-source systems, always include the source caveat (e.g., "limited architecture disclosure")

### Code

- All Python examples must be **runnable on a clean environment** with `pip install -r requirements.txt`
- No hardcoded API keys — use `os.environ` or `.env`
- Include docstrings on every public function
- Add a `# WHY:` comment on non-obvious lines

### Watchlist vs Main

Anything in the **main roadmap** must satisfy at least ONE of:

```text
1. Used in a released model
2. Covered in a major

*(This section now uses `emerging` instead of `watchlist` to avoid the “still waiting” connotation.)* technical report
3. Widely used in serving frameworks
4. Architecturally central to 2025–2026 models
```

Otherwise it goes to **watchlist** with a reason for the uncertainty.

---

## 🔍 Review Process

1. Open a PR with a clear title (e.g., "Add Qwen3-VL to multimodal section")
2. Reference any related issue
3. Keep PRs **focused** — one topic per PR
4. Maintainers will review within 7 days

---

## 🐛 Reporting an Error

If you spot:

- A factual mistake (wrong paper link, wrong year, wrong attribution)
- A broken example
- An outdated recommendation

Please open an issue with the section + the error. Even a one-line PR fixing a typo is welcome.

---

## 📜 License

By contributing, you agree your contributions will be licensed under Apache 2.0.