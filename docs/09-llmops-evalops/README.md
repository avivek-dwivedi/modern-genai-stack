# Section 9 — LLMOps & EvalOps

> **How do we evaluate, monitor, and version LLM systems like software?**

---

## Why This Section Exists

LLM apps deploy a **system**, not only a model:

```text
model
prompt
retriever
tools
memory
guardrails
evaluation
serving layer
user feedback
```

Without LLMOps discipline, you cannot:
- Roll back a bad prompt change
- Prove answer quality is improving
- Detect regressions
- Reproduce past behavior
- Audit decisions

---

## 9.1 PromptOps

PromptOps manages prompts like software artifacts:

```text
prompt template
prompt version
test set
approval
deployment
rollback
A/B test
monitoring
```

### Prompt Registry Example

```json
{
  "prompt_id": "rag_answer_v3",
  "version": "3.2.1",
  "owner": "ai-platform",
  "model": "qwen-72b",
  "temperature": 0.2,
  "status": "production"
}
```

---

## 9.2 EvalOps

| Eval type | Purpose |
|---|---|
| Golden dataset eval | Compare against expected answers |
| LLM-as-judge | Judge quality using another model |
| Human eval | Expert scoring |
| RAG faithfulness eval | Answer grounded in retrieved context |
| Tool-call eval | Correct tool and arguments |
| Safety eval | Jailbreak, PII, toxicity |
| Regression eval | New version not worse than old |
| Cost-latency eval | Production feasibility |

---

## 9.3 Observability

Track:

```text
prompt
model
retrieved chunks
tool calls
latency
token usage
cost
error
fallback
user feedback
trace ID
```

---

## 9.4 Version Everything

```text
model version
adapter version
prompt version
retriever/index version
embedding model version
reranker version
tool schema version
guardrail version
```

### Why?

```text
Without version pinning, you cannot:
  - reproduce a past good answer
  - roll back a regression
  - blame a specific change for a quality drop
  - prove to auditors that nothing unsafe changed
```

---

## 9.5 Decision Tree

```text
Need to A/B test prompts?
    → Prompt registry + traffic split

Need to catch regressions automatically?
    → Golden dataset eval + CI gate

Need qualitative feedback?
    → LLM-as-judge on sampled traffic

Need real-world quality signal?
    → User feedback + thumbs up/down + outcome tracking

Need end-to-end traceability?
    → Distributed tracing (OpenTelemetry) + trace IDs
```

---

## 9.6 Production Rule

```text
Treat prompts and adapters as versioned artifacts.
Run eval suite on every change.
Block deploys that fail the suite.
Track business metrics (acceptance, completion) alongside technical metrics.
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part F`
- **Code example:** [examples/06_eval/](../examples/06_eval/) (EvalOps harness with LLM-as-judge)
- **Diagrams:** [assets/diagrams/llmops-pipeline.md](../../assets/diagrams/llmops-pipeline.md)