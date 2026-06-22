# Section 12 — End-to-End Production Blueprint

> **How does it all fit together?**

---

## Why This Section Exists

Most AI projects fail not because of any one technical choice but because the team didn't see the **whole system**.

This section is the integrated view — for tech leads, engineering managers, and senior engineers who need to ship.

---

## 12.1 Complete System Map

```text
User / App
→ API Layer
→ Auth / Tenant / Rate Limit
→ Orchestrator
   ├── Prompt Registry
   ├── RAG Retriever
   ├── Reranker
   ├── Memory
   ├── Tool Calling
   ├── Guardrails
   └── Model Router
→ Model Serving
   ├── Bedrock
   ├── vLLM / TGI
   ├── KServe / SageMaker
   └── fallback model
→ Observability
→ Feedback Collection
→ Offline Evaluation
→ Fine-Tuning / Preference Training
→ Model Registry
→ Deployment / Rollback
```

---

## 12.2 End-to-End AI Engineer Roadmap

```text
Step 1:  Understand model architecture
Step 2:  Choose model for task
Step 3:  Build RAG if knowledge needed
Step 4:  Add reranker / hybrid / metadata filtering
Step 5:  Add agent workflow only if tools/actions needed
Step 6:  Add memory only if persistence needed
Step 7:  Add evals before production
Step 8:  Add observability
Step 9:  Add guardrails/security
Step 10: Add fine-tuning only when prompting/RAG is not enough
Step 11: Add preference optimization when behavior quality needs alignment
Step 12: Serve with managed or self-hosted runtime
Step 13: Monitor, rollback, improve
```

---

## 12.3 Decision Tree

```text
Need private documents?
    → RAG

Need domain-specific response format?
    → SFT / LoRA

Need preferred behavior/style?
    → DPO / ORPO / SimPO

Need reasoning with verifiable reward?
    → GRPO / verifier-based RL

Need visual understanding?
    → VLM / multimodal RAG / VLM fine-tuning

Need actions/tools?
    → Agent / workflow

Need enterprise production?
    → LLMOps + guardrails + observability + AWS architecture

Need scale/cost control?
    → quantization + serving optimization + caching + batching
```

---

## 12.4 When to Add Each Layer

| Symptom | Add |
|---|---|
| Model hallucinates facts | RAG |
| Wrong style / format | SFT |
| Right info, wrong tone | DPO / ORPO |
| Math / code reasoning failures | GRPO / verifier |
| Can't read documents / charts | VLM / ColPali |
| User wants to do actions | Agent + tools |
| User wants continuity | Memory |
| Hallucinations in production | Eval + observability |
| Security incidents | Guardrails + injection scanning |
| Cost blowing up | Quantization + caching + serving optimization |
| Latency too high | Speculative decoding + prefix caching |

---

## 12.5 The Build Order (Pragmatic)

For most enterprise projects, this is the **order of attack**:

```text
PHASE 1 — Foundations (Weeks 1–2)
  - Pick a model (Bedrock for fastest, vLLM+Qwen for open-weight)
  - Pick an evaluation harness
  - Pick a tracing/logging system

PHASE 2 — RAG (Weeks 3–4)
  - Pick chunking strategy
  - Add hybrid search (BM25 + dense)
  - Add a reranker
  - Add metadata filtering
  - Eval retrieval quality

PHASE 3 — Application (Weeks 5–6)
  - Build prompt templates + prompt registry
  - Add input/output guardrails
  - Add observability + tracing
  - Add cost/latency tracking

PHASE 4 — Agents (if needed) (Weeks 7–8)
  - Build tool allow-list + permissions
  - Build state machine (LangGraph-style)
  - Add memory (if needed)
  - Add audit log

PHASE 5 — Adaptation (if needed) (Weeks 9–12)
  - Collect preference data
  - Run DPO/ORPO alignment
  - (Optional) SFT on domain data
  - Re-eval + canary + rollback plan

PHASE 6 — Scale (Weeks 13+)
  - Quantization
  - Speculative decoding
  - Disaggregated serving
  - Multi-region
  - Cost optimization
```

---

## 12.6 Final Mental Model

```text
LLM app = model + data + retrieval + tools + memory + evals + serving + security + operations.
```

**Not:** "LLM app = a model + a prompt"

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part I`
- **All other sections** in this hub — this section is the integration view
- **Diagrams:** [assets/diagrams/full-stack.md](../../assets/diagrams/full-stack.md)