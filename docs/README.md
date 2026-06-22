# 📚 GenAI Engineering Hub — Knowledge Base

> The structured, deep-dive companion to the master notebook.

This directory contains **12 topic-focused sections** that mirror the master notebook but go deeper into each topic, with decision trees, comparison tables, and source links.

---

## 🗺️ Section Map

| # | Section | One-line Summary | Doc |
|---:|---|---|---|
| 1 | **LLM Architecture** | Transformer → RoPE → Attention variants → MoE → Beyond-Transformer | [01-architecture](01-architecture/README.md) |
| 2 | **Attention & Serving Kernels** | FlashAttention, PagedAttention, KV-cache quantization, inference optimization | [02-attention-serving](02-attention-serving/README.md) |
| 3 | **VLM & Multimodal** | CLIP → BLIP-2 → LLaVA → Qwen-VL; multimodal embeddings | [03-vlm-multimodal](03-vlm-multimodal/README.md) |
| 4 | **Coding Models** | Why coding models are different; the coding-model landscape | [04-coding-models](04-coding-models/README.md) |
| 5 | **Fine-Tuning & PEFT** | LoRA, QLoRA, DoRA, full fine-tuning, capacity planning | [05-finetuning-peft](05-finetuning-peft/README.md) |
| 6 | **Alignment & RLHF** | PPO → DPO → ORPO → SimPO → GRPO; reasoning model training | [06-alignment-rlhf](06-alignment-rlhf/README.md) |
| 7 | **RAGOps** | Basic RAG → Hybrid → Reranker → Multimodal RAG → GraphRAG | [07-ragops](07-ragops/README.md) |
| 8 | **AgentOps** | ReAct, Planner-Executor, LangGraph, memory, multi-agent | [08-agentops](08-agentops/README.md) |
| 9 | **LLMOps & EvalOps** | Prompt registry, evaluation, observability, version control | [09-llmops-evalops](09-llmops-evalops/README.md) |
| 10 | **AWS Production** | Bedrock, EKS+KServe, vLLM, SageMaker, Ray | [10-aws-production](10-aws-production/README.md) |
| 11 | **Security & Guardrails** | Prompt injection, RAG poisoning, memory safety, PII | [11-security-guardrails](11-security-guardrails/README.md) |
| 12 | **End-to-End Blueprint** | The complete production system map & decision tree | [12-e2e-blueprint](12-e2e-blueprint/README.md) |

---

## 🔀 Section Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Section 1 — LLM Architecture                                       │
│  "What is this model? What changed inside the transformer?"         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Section 2 — Attention & Serving Kernels                            │
│  "How do we run this model fast and cheap in production?"           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Sections 3–6 — Model Adaptation                                    │
│  VLM → Coding → Fine-Tuning → RLHF                                  │
│  "How do we adapt the model to our task?"                           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Sections 7–8 — Knowledge & Action                                  │
│  RAG → Agents                                                      │
│  "How do we connect the model to data and tools?"                   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Sections 9–11 — Operations & Safety                                │
│  LLMOps → AWS → Security                                            │
│  "How do we ship it safely at scale?"                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Section 12 — End-to-End Blueprint                                  │
│  "How does it all fit together?"                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔖 Reading Mode

Each `docs/0X-*/README.md` follows the same internal structure:

1. **Why this exists** — the problem this section solves
2. **Taxonomy / Evolution timeline** — how the field got here
3. **Cheat sheet table** — method-by-method comparison
4. **Decision tree** — when to use what
5. **Production rule** — the one-line takeaway
6. **Source links** — primary references

This makes every section skimmable AND deep-dive-able.

---

## 🔗 Cross-Links

Every section links to:
- Its corresponding notebook section in `end_to_end_*_master_notebook.ipynb`
- Its corresponding example in `examples/`
- Its diagrams in `assets/diagrams/`

This is intentional — the three formats (notebook / docs / examples) form a **learning triangle**:

```text
        notebook (linear reading)
              /\
             /  \
            /    \
   docs /    \ examples
(reference) (runnable code)
```