# Examples — Runnable Code for Each Section

This directory contains **runnable Python / YAML / config files** that demonstrate the concepts in the corresponding `docs/` section.

| Folder | Topic | Maps to |
|---|---|---|
| [`03_finetune/`](03_finetune/) | LoRA / QLoRA / DPO training scripts | [docs/05-finetuning-peft](../docs/05-finetuning-peft/README.md), [docs/06-alignment-rlhf](../docs/06-alignment-rlhf/README.md) |
| [`04_serving/`](04_serving/) | vLLM config + serving patterns | [docs/02-attention-serving](../docs/02-attention-serving/README.md) |

> Note: RAG / Agent / AWS / Eval examples (sections 7-12) were moved to an external production repo. This hub now focuses on the **core research / engineering layers** (architecture → serving → finetuning → alignment).

---

## 🚀 Quick Start

```bash
# 1. Install dependencies (use a virtual environment!)
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r ../requirements.txt

# 2. Pick an example and run it
cd 03_finetune
python lora_sft.py
```

---

## 📜 License

CC BY 4.0 — same as the parent repo. See [LICENSE](../LICENSE).