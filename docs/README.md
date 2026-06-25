# 📚 GenAI Engineering Hub — Knowledge Base

> The structured, deep-dive companion to the master notebook.

This directory contains **6 topic-focused sections** that mirror the master notebook but go deeper into each topic, with decision trees, comparison tables, and source links. (Sections 7–12 — RAG, agents, AWS production, security, E2E blueprint — were moved to an external production repo.)

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
| 7 | *RAGOps* | Moved to external repo | — |
| 8 | *AgentOps* | Moved to external repo | — |
| 9 | *LLMOps & EvalOps* | Moved to external repo | — |
| 10 | *AWS Production* | Moved to external repo | — |
| 11 | *Security & Guardrails* | Moved to external repo | — |
| 12 | *End-to-End Blueprint* | Moved to external repo | — |

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
│  Sections 7–12 — Moved to external production repo                   │
│  RAG → Agents → LLMOps → AWS → Security → E2E                       │
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