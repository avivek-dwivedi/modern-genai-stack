# GenAI Engineering Hub

> **The complete, end-to-end technical reference for modern Generative AI Engineering.**
> From LLM architecture internals to production RAG systems, agents, fine-tuning, RLHF, AWS deployment, and security.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Maintained](https://img.shields.io/badge/Maintained-2026-green)]()

---

## What This Hub Is

This is a **single, comprehensive technical knowledge base** for engineers building production-grade Generative AI systems. It is structured as a **learning path** from first principles all the way to deployed, monitored, secure AI products on AWS.

Unlike scattered blog posts or vendor docs, this hub is:

- **Vendor-neutral** — covers open-weight AND closed-source systems
- **Architecture-deep** — goes from attention internals → serving kernels → production ops
- **Production-focused** — every section ends with how to ship it
- **Code-backed** — every major section has runnable examples in `examples/`

---

## 📚 Learning Path (Read In Order)

| # | Topic | What You Will Learn | Doc |
|---:|---|---|---|
| 1 | **LLM Architecture** | Transformer, RoPE, Attention variants, MoE, Beyond-Transformer | [01-architecture](docs/01-architecture/README.md) |
| 2 | **Attention & Serving Kernels** | FlashAttention, PagedAttention, KV-cache, inference optimization | [02-attention-serving](docs/02-attention-serving/README.md) |
| 3 | **VLM & Multimodal** | CLIP, BLIP-2, LLaVA, Qwen-VL, Q-Former, multimodal embeddings | [03-vlm-multimodal](docs/03-vlm-multimodal/README.md) |
| 4 | **Coding Models** | Code Llama, DeepSeek Coder, Qwen Coder, agentic coding | [04-coding-models](docs/04-coding-models/README.md) |
| 5 | **Fine-Tuning & PEFT** | LoRA, QLoRA, DoRA, full FT, capacity planning | [05-finetuning-peft](docs/05-finetuning-peft/README.md) |
| 6 | **Alignment & RLHF** | PPO → DPO → ORPO → GRPO, reasoning models | [06-alignment-rlhf](docs/06-alignment-rlhf/README.md) |
| 7 | **RAGOps** | Basic RAG → hybrid → reranker → multimodal RAG → GraphRAG | [07-ragops](docs/07-ragops/README.md) |
| 8 | **AgentOps** | ReAct, planner-executor, LangGraph, memory, multi-agent | [08-agentops](docs/08-agentops/README.md) |
| 9 | **LLMOps & EvalOps** | Prompt registry, evaluation, observability, version control | [09-llmops-evalops](docs/09-llmops-evalops/README.md) |
| 10 | **AWS Production** | Bedrock, EKS+KServe, vLLM, SageMaker, Ray | [10-aws-production](docs/10-aws-production/README.md) |
| 11 | **Security & Guardrails** | Prompt injection, RAG poisoning, memory safety, PII | [11-security-guardrails](docs/11-security-guardrails/README.md) |
| 12 | **End-to-End Blueprint** | The complete production system map | [12-e2e-blueprint](docs/12-e2e-blueprint/README.md) |

---

## 🗂️ Repository Layout

```text
genai-engineering-hub/
├── README.md                          # ← you are here
├── LICENSE                            # Apache 2.0
├── CONTRIBUTING.md                    # how to contribute
├── CHANGELOG.md                       # release notes
│
├── docs/                              # structured knowledge base (12 sections)
│   ├── 01-architecture/
│   ├── 02-attention-serving/
│   ├── 03-vlm-multimodal/
│   ├── 04-coding-models/
│   ├── 05-finetuning-peft/
│   ├── 06-alignment-rlhf/
│   ├── 07-ragops/
│   ├── 08-agentops/
│   ├── 09-llmops-evalops/
│   ├── 10-aws-production/
│   ├── 11-security-guardrails/
│   └── 12-e2e-blueprint/
│
├── examples/                          # runnable Python code for each section
│   ├── 01_rag/         # end-to-end RAG with hybrid + reranker
│   ├── 02_agent/       # LangGraph multi-agent supervisor
│   ├── 03_finetune/    # LoRA + DPO training scripts
│   ├── 04_serving/     # vLLM + KServe deployment
│   ├── 05_bedrock/     # AWS Bedrock SDK + AgentCore
│   └── 06_eval/        # EvalOps harness with LLM-as-judge
│
├── assets/
│   ├── diagrams/                      # ASCII + Mermaid source for all diagrams
│   └── images/                        # rendered architecture images
│
├── references/                        # curated paper/resource index
│   └── papers.md
│
└── end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb
                                      # ← the master learning notebook
```

---

## 🚀 Quick Start

### 1. Clone and explore the docs

```bash
git clone https://github.com/your-org/genai-engineering-hub.git
cd genai-engineering-hub
```

### 2. Open the master notebook

The single Jupyter notebook in the root is the **canonical, end-to-end learning artifact**.
It is structured so you can read it top-to-bottom, or use it as a reference notebook.

```bash
jupyter lab end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb
```

### 3. Run an example

```bash
cd examples/01_rag
pip install -r requirements.txt
python rag_basic.py
```

---

## 🎯 Who This Hub Is For

| Audience | What to Read |
|---|---|
| **AI Engineer starting out** | Sections 1 → 6 (architecture → alignment), then 7 → 12 |
| **ML Engineer moving to LLMs** | Sections 1, 5, 6, 7, 9 |
| **Backend Engineer building AI products** | Sections 7, 8, 9, 10, 11, 12 |
| **Solutions Architect** | Sections 9, 10, 11, 12 + AWS section in `examples/05_bedrock/` |
| **Tech Lead / Engineering Manager** | Section 12 (E2E blueprint) + decision trees throughout |

---

## 🧭 Core Principle

```text
Architecture tells which model to choose.
Fine-tuning tells how to adapt it.
RAG connects private/external knowledge.
Agents connect tools, workflows and memory.
Serving runs the system at scale.
LLMOps evaluates, monitors, and rolls back.
Security prevents leakage, poisoning, and abuse.
```

---

## 📖 Notebook vs Docs vs Examples

| Artifact | When to use |
|---|---|
| **`end_to_end_*_master_notebook.ipynb`** | Linear reading, study sessions, teaching material |
| **`docs/*.md`** | Quick lookup, reference, linkable deep-dives |
| **`examples/*/*.py`** | When you need runnable code you can copy/adapt |

The three formats are **synchronized**. Each `docs/` section has a sibling in the notebook and a corresponding example in `examples/`.

---

## 🤝 Contributing

We welcome additions in:

- **New architecture methods** (with paper link + one-line teaching summary)
- **Updated model reports** (Qwen, DeepSeek, Kimi, MiniMax, Llama, Gemma, Jamba, RWKV, etc.)
- **Real-world RAG / agent / fine-tuning patterns**
- **AWS architecture diagrams**

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rules.

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 📅 Maintenance

- **Current version:** V1.4 Master
- **Last updated:** 2026-06
- **Refresh cadence:** quarterly review of new models / papers / frameworks