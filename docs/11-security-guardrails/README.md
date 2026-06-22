# Section 11 — Security & Guardrails

> **How do we prevent leakage, poisoning, and abuse?**

---

## Why This Section Exists

LLM apps have **attack surfaces that traditional apps do not**:

- **Prompt injection** through user input or retrieved documents
- **RAG poisoning** through malicious documents entering the vector index
- **Memory poisoning** through malicious facts stored as long-term memory
- **Feedback poisoning** through bad feedback entering training data
- **Model poisoning** through curated data updates that contain backdoors
- **Data leakage** through logs, memory, retrieved chunks, and tool outputs
- **Tool misuse** through agents calling dangerous APIs

Security must be **layered** — no single guardrail catches everything.

---

## 11.1 Main Risks

| Risk | Where it happens |
|---|---|
| Prompt injection | User prompt / retrieved document / tool output |
| Data leakage | Prompt, memory, logs, retrieved chunks |
| Tool misuse | Agent calls dangerous API |
| RAG poisoning | Malicious document enters vector index |
| Feedback poisoning | Bad feedback enters training data |
| Model poisoning | Bad curated data updates model weights |
| Memory poisoning | Malicious facts stored as memory |
| Eval poisoning | Contaminated eval set hides failure |

---

## 11.2 Safe Feedback-to-Training Flow

### Unsafe

```text
user feedback → model immediately updates
```

### Safe

```text
user feedback
→ raw logs
→ filtering
→ review
→ curated dataset
→ offline training
→ evaluation
→ model registry
→ canary deploy
→ rollback if needed
```

---

## 11.3 Data Storage Layers

```text
1. Raw interaction logs
2. Feedback / preference data
3. Sanitized dataset
4. Curated training dataset
5. Model checkpoint / adapter version
```

Each layer should have **different access controls** and **different retention policies**.

---

## 11.4 Preference Data Example

```json
{
  "prompt": "Explain LoRA",
  "chosen": "Correct detailed answer",
  "rejected": "Incorrect or unsafe answer",
  "source": "expert_review",
  "trust_score": 0.95
}
```

The `trust_score` is critical — it lets the training pipeline weight examples by their source quality.

---

## 11.5 Poisoning Mitigation

```text
do not train from raw logs directly
PII redaction
toxicity and jailbreak filtering
prompt injection scanning
trust scoring
human expert review
deduplication
outlier detection
dataset versioning
eval gates
canary deploy
rollback
```

---

## 11.6 Guardrail Layers

| Layer | Guardrail |
|---|---|
| Input | Prompt injection / PII / abuse detection |
| Retrieval | Source trust + metadata permission |
| Generation | Policy prompt + output validation |
| Tool | Schema validation + permissions + approval |
| Memory | Write filters + user scope |
| Training | Curated dataset + eval gate |
| Deployment | Canary + rollback |

### Core Rule

```text
Raw feedback is signal.
Curated feedback is training data.
Only reviewed, versioned, evaluated data should touch model weights.
```

---

## 11.7 Decision Tree

```text
User input could be hostile?
    → Input filter (prompt injection + jailbreak detection)

Retrieval returns documents from external sources?
    → Source trust scoring + injection scanning on ingest

Output could leak PII or violate policy?
    → Output validator + PII redaction

Agent calls external APIs?
    → Tool allow-list + permission check + audit log

User data stored long-term?
    → Memory write filter + user scope + expiry

Training on production feedback?
    → Trust score + human review + canary + rollback

Auditors need to know what changed?
    → Dataset versioning + model registry + audit log
```

---

## 11.8 Production Rule

```text
Defense in depth:
  - filter at INPUT (cheap, catches most attacks)
  - filter at RETRIEVAL (medium cost, catches RAG poisoning)
  - filter at OUTPUT (expensive, catches leakage)
  - filter at TRAINING (most expensive, prevents backdoors)
  - filter at DEPLOYMENT (canary + rollback catches regressions)

No single layer is enough. Stack them.
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part H`
- **Code example:** see [examples/02_agent/](../examples/02_agent/) for tool allow-list pattern
- **Diagrams:** [assets/diagrams/guardrail-layers.md](../../assets/diagrams/guardrail-layers.md)