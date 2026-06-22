# Diagrams

This folder contains **Mermaid source files** (`.mmd`) for all architecture diagrams used in the hub.

## How to Use

### Option 1 — Render inline in GitHub
Mermaid is natively supported by GitHub. Just include in any markdown:

```markdown
```mermaid
graph TD
    A[User] --> B[LLM]
```
```

### Option 2 — Render locally
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Render
mmdc -i transformer-block.mmd -o transformer-block.png
```

### Option 3 — View in VS Code
Install the **Markdown Preview Mermaid Support** extension.

---

## Diagram Index

| Diagram | Source | Used in |
|---|---|---|
| Transformer block | [transformer-block.mmd](transformer-block.mmd) | docs/01-architecture |
| RAG pipeline | [rag-pipeline.mmd](rag-pipeline.mmd) | docs/07-ragops |
| Serving stack | [serving-stack.mmd](serving-stack.mmd) | docs/02-attention-serving |
| Agent architecture | [agent-architecture.mmd](agent-architecture.mmd) | docs/08-agentops |
| AWS RAG | [aws-rag.mmd](aws-rag.mmd) | docs/10-aws-production |
| Guardrail layers | [guardrail-layers.mmd](guardrail-layers.mmd) | docs/11-security-guardrails |
| RLHF pipeline | [rlhf-pipeline.mmd](rlhf-pipeline.mmd) | docs/06-alignment-rlhf |
| Training pipeline | [training-pipeline.mmd](training-pipeline.mmd) | docs/10-aws-production |
| Model landscape | [model-landscape.mmd](model-landscape.mmd) | docs/01-architecture |
| Full stack | [full-stack.mmd](full-stack.mmd) | docs/12-e2e-blueprint |

---

## How to Add a New Diagram

1. Create `your-diagram.mmd` here
2. Add an entry to the table above
3. Reference it in the relevant `docs/` section with a markdown image link

```markdown
![Diagram description](assets/diagrams/your-diagram.png)
```