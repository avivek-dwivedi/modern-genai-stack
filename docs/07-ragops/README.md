# Section 7 — RAGOps

> **How do we connect the LLM to private/external knowledge?**

---

## Why This Section Exists

Base LLMs have three problems:

```text
1. Knowledge cutoff
2. Hallucination
3. No access to private enterprise data
```

RAG (Retrieval-Augmented Generation) connects the LLM to external knowledge:

```text
User query → retrieve context → inject context → generate grounded answer
```

RAGOps is the discipline of running RAG **as a production system** — with monitoring, versioning, security, and feedback loops.

---

## 7.1 Basic RAG Pipeline

```text
Documents
→ parsing / OCR
→ chunking
→ embedding
→ vector database
→ retrieval
→ prompt context
→ LLM answer
```

---

## 7.2 Where Basic RAG Fails

| Failure | Meaning |
|---|---|
| Bad chunking | Important meaning split incorrectly |
| Weak embedding | Query and document do not match semantically |
| Low recall | Relevant chunk not retrieved |
| Low precision | Irrelevant chunks retrieved |
| No reranker | Best chunk not ranked at top |
| No metadata filter | Wrong user/tenant/source retrieved |
| Prompt injection | Document attacks the model |
| No eval | No proof answer is grounded |
| No index versioning | Cannot reproduce old behavior |

---

## 7.3 Advanced RAG Components

| Component | Purpose |
|---|---|
| Hybrid search | BM25 keyword + dense vector retrieval |
| Reranker | Reorders retrieved chunks with stronger model |
| Query rewriting | Converts vague query into retriever-friendly query |
| Multi-query retrieval | Creates multiple query variants |
| Metadata filtering | Filters by user, tenant, source, date, permission |
| Contextual compression | Keeps only relevant context |
| Parent-child retrieval | Retrieves small chunks, returns parent document |
| GraphRAG | Uses entity/relation graph for reasoning |
| Multi-vector retrieval | Multiple vectors per document/page |
| Multimodal RAG | Retrieves images, tables, charts, PDFs, video/audio |

### Hybrid RAG

```text
BM25 / keyword search
+
dense vector search
+
score fusion
+
reranking
```

Dense retrieval catches meaning. BM25 catches exact words, IDs, names, legal clauses, product codes and error strings.

### Reranker

```text
Retriever = fast recall
Reranker = slower precision
```

Retriever gives candidates. Reranker deeply compares query + chunk and reorders top results.

### GraphRAG

GraphRAG helps when the answer needs relationships:

```text
Company → Project → Person → Contract → Risk → Decision
```

### Multimodal RAG

Use when documents are not pure text:

```text
PDF pages
tables
charts
screenshots
diagrams
images
videos
audio transcripts
```

---

## 7.4 Production RAGOps Pipeline

```text
Document upload
→ validation
→ malware / prompt injection scan
→ OCR / parsing
→ chunking
→ embedding
→ metadata tagging
→ vector index versioning
→ retrieval evaluation
→ deployment
→ monitoring
→ re-indexing
```

---

## 7.5 RAGOps Metrics

| Layer | Metrics |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR, nDCG |
| Reranking | Hit rate after rerank, MRR improvement |
| Generation | Faithfulness, groundedness, answer relevance |
| Safety | Injection resistance, PII leakage |
| System | Latency, cost, cache hit rate |
| Business | Task completion, user acceptance |

---

## 7.6 Decision Tree

```text
Need exact IDs / legal clauses?
    → Hybrid search (BM25 + dense)

Need semantic answers?
    → Dense retrieval

Need better top-5 chunks?
    → Add a reranker

Need document layout / charts?
    → Multimodal RAG / ColPali-style retrieval

Need relationship reasoning?
    → GraphRAG

Need strict access control?
    → Metadata filters + tenant isolation

Need production reliability?
    → Eval + monitoring + index versioning
```

---

## 7.7 Production Rule

```text
Basic RAG works for ~20% of real use cases.
For the remaining 80%, you need:
  - hybrid search (always, by default)
  - a reranker (Cross-encoder or ColBERT-style)
  - metadata filtering (for permission / multi-tenant)
  - eval harness (faithfulness + groundedness)
  - index versioning (so you can roll back)
  - injection scanning (on document upload)
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part D`
- **Code example:** [examples/01_rag/](../examples/01_rag/) (end-to-end RAG with hybrid + reranker)
- **Diagrams:** [assets/diagrams/rag-pipeline.md](../../assets/diagrams/rag-pipeline.md)