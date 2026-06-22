# Section 10 — AWS Production Architecture

> **How do we deploy and operate GenAI systems on AWS?**

---

## Why This Section Exists

AWS gives you many options — Bedrock, SageMaker, EKS+KServe, ECS, EC2, Lambda. Picking the wrong one wastes months.

This section gives you a **decision framework** + reference architectures for the most common patterns.

---

## 10.1 Managed vs Self-Hosted

| Choice | Best when |
|---|---|
| AWS Bedrock | Managed models, governance, fast delivery |
| Bedrock Agents / AgentCore | Managed agent workflows/tools/memory/guardrails where supported |
| SageMaker | Managed training/deployment/model registry |
| EKS + KServe | Kubernetes-native model platform |
| ECS | Simpler container serving |
| Lambda | Lightweight async workflows |
| Ray on AWS | Distributed training/orchestration |
| vLLM/TGI on EC2/EKS | Self-hosted open-weight serving |
| OpenSearch | Hybrid/vector search on AWS |
| S3 | Dataset/model/log storage |
| CloudWatch | Infra/app logs and metrics |
| IAM/KMS/Secrets Manager | Security and access control |

### Bedrock vs Self-Hosted

```text
Bedrock = managed foundation model consumption.
Self-hosted = platform control over open-weight models.
```

**Choose Bedrock when:**

```text
governance matters
fast delivery matters
managed scaling matters
custom model internals are not required
```

**Choose self-hosted when:**

```text
need custom LoRA/QLoRA adapters
need open-weight model
need custom decoding/serving
need unit economics at scale
need full runtime control
```

---

## 10.2 AWS GPU Choices

| Instance family | GPUs | Best for |
|---|---|---|
| g5 | NVIDIA A10G | 7B/13B LoRA, inference, small labs |
| g6 | NVIDIA L4 | efficient inference, smaller training |
| p4d / p4de | A100 | large training, distributed finetuning |
| p5 | H100 | high-end training, RLHF, large VLM |
| trn / inf | Trainium / Inferentia | AWS-native managed training/inference, framework-dependent |

---

## 10.3 Reference AWS GenAI Architecture

```text
Frontend
→ API Gateway / ALB
→ FastAPI / service layer
→ Auth / IAM / Cognito
→ Orchestrator
   ├── Bedrock / self-hosted vLLM
   ├── RAG retriever
   ├── tools/APIs
   ├── memory store
   └── guardrail layer
→ Observability
→ Logs / feedback
→ Offline training/evaluation pipeline
```

## 10.4 AWS RAG Architecture

```text
S3 document bucket
→ Textract / parser / custom OCR
→ chunking job
→ embedding job
→ OpenSearch / vector DB
→ retriever API
→ reranker
→ LLM generation
→ answer + citations
```

## 10.5 AWS Fine-Tuning Architecture

```text
S3 dataset
→ SageMaker / EC2 / EKS training job
→ MLflow / W&B / CloudWatch tracking
→ model artifact in S3
→ model registry
→ SageMaker / EKS / vLLM / TGI deployment
→ monitor
```

## 10.6 Decision Tree

```text
Quickest path to a working GenAI app?
    → Bedrock + managed RAG + Bedrock Agents

Need fine-tuning on your own data?
    → SageMaker training jobs (managed) or EC2/EKS (custom)

Need Kubernetes-native serving?
    → EKS + KServe + vLLM/TGI

Need distributed training at scale?
    → SageMaker distributed training or Ray on EKS

Need to optimize cost at high QPS?
    → Self-hosted vLLM on EC2 with autoscaling

Need managed vector search?
    → OpenSearch Service / Aurora pgvector / Bedrock Knowledge Base
```

---

## 10.7 Production Rule

```text
Bedrock is managed AI consumption.
EKS/KServe/vLLM is AI platform engineering.
Ray/SageMaker/DeepSpeed is training orchestration.
```

For most teams, the realistic path is:

1. **Prototype on Bedrock** (fastest delivery)
2. **Move to self-hosted vLLM** when cost or customization matters
3. **Adopt Bedrock Agents / AgentCore** if you want managed agent infra
4. **Use SageMaker** for serious training jobs
5. **Add OpenSearch / pgvector** for RAG

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part G / Section 18`
- **Code example:** [examples/05_bedrock/](../examples/05_bedrock/) (Bedrock SDK + AgentCore)
- **Diagrams:** [assets/diagrams/aws-reference.md](../../assets/diagrams/aws-reference.md)