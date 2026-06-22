# Examples — Runnable Code for Each Section

This directory contains **runnable Python / YAML / config files** that demonstrate the concepts in the corresponding `docs/` section.

| Folder | Topic | Maps to |
|---|---|---|
| [`01_rag/`](01_rag/) | End-to-end RAG with hybrid search + reranker | [docs/07-ragops](../docs/07-ragops/README.md) |
| [`02_agent/`](02_agent/) | Multi-agent supervisor with LangGraph | [docs/08-agentops](../docs/08-agentops/README.md) |
| [`03_finetune/`](03_finetune/) | LoRA / QLoRA / DPO training scripts | [docs/05-finetuning-peft](../docs/05-finetuning-peft/README.md), [docs/06-alignment-rlhf](../docs/06-alignment-rlhf/README.md) |
| [`04_serving/`](04_serving/) | vLLM config + KServe manifest | [docs/02-attention-serving](../docs/02-attention-serving/README.md), [docs/10-aws-production](../docs/10-aws-production/README.md) |
| [`05_bedrock/`](05_bedrock/) | AWS Bedrock SDK + Knowledge Base RAG | [docs/10-aws-production](../docs/10-aws-production/README.md) |
| [`06_eval/`](06_eval/) | LLM-as-judge eval harness | [docs/09-llmops-evalops](../docs/09-llmops-evalops/README.md) |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies (use a virtual environment!)
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r ../requirements.txt

# 2. Pick an example and run it
cd 01_rag
python rag_basic.py
```

---

## 📜 License

Apache 2.0 — same as the parent repo.