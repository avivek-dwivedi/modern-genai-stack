<div align="center">

# Modern GenAI Stack

**The end-to-end engineering reference for production Generative AI systems.**

From LLM architecture internals to RAG, agents, fine-tuning, RLHF, AWS, and security — in one place.

[📓 Master Notebook](end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb) · [📚 Docs](docs/) · [💻 Examples](examples/) · [📑 Papers](references/papers.md)

</div>

---

## Why This Repo Exists

Building a production GenAI system means stitching together 12+ technical layers — model architecture, serving, retrieval, agents, fine-tuning, alignment, security, and AWS. **Most docs cover one layer. This repo covers all of them, in order, with runnable code.**

| | |
|---|---|
| 🏗️ **Architecture-deep** | Attention internals, MoE routing, beyond-transformer (Mamba, RWKV, KDA) |
| 🚀 **Serving-aware** | FlashAttention, PagedAttention, KV-cache quantization, speculative decoding |
| 🤖 **Application-ready** | RAG, agents, memory, tools, evaluation — with runnable examples |
| ☁️ **AWS-native** | Bedrock, AgentCore, EKS+KServe, SageMaker, Ray |
| 🔒 **Security-first** | Prompt injection, RAG poisoning, memory safety, defense in depth |

---

## 🚀 Quick Start

```bash
git clone https://github.com/avivek-dwivedi/modern-genai-stack.git
cd modern-genai-stack

# Open the master notebook
jupyter lab end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb

# Run an example
cd examples/01_rag
pip install -r ../../requirements.txt
python rag_basic.py
```

---

## 🗺️ The 12-Section Learning Path

| # | Section | What's inside |
|---:|---|---|
| **01** | [LLM Architecture](docs/01-architecture/README.md) | RoPE, GQA, MLA, MoE, Mamba, RWKV |
| **02** | [Attention & Serving Kernels](docs/02-attention-serving/README.md) | FlashAttention, PagedAttention, KV-cache |
| **03** | [VLM & Multimodal](docs/03-vlm-multimodal/README.md) | CLIP, BLIP-2, LLaVA, Qwen-VL, ColPali |
| **04** | [Coding Models](docs/04-coding-models/README.md) | Code Llama, DeepSeek Coder, Qwen Coder, Kimi K2 |
| **05** | [Fine-Tuning & PEFT](docs/05-finetuning-peft/README.md) | LoRA, QLoRA, DoRA, capacity planning |
| **06** | [Alignment & RLHF](docs/06-alignment-rlhf/README.md) | PPO → DPO → ORPO → GRPO |
| **07** | [RAGOps](docs/07-ragops/README.md) | Hybrid search, rerankers, multimodal RAG |
| **08** | [AgentOps](docs/08-agentops/README.md) | ReAct, LangGraph, memory, tools |
| **09** | [LLMOps & EvalOps](docs/09-llmops-evalops/README.md) | Prompt registry, evaluation, observability |
| **10** | [AWS Production](docs/10-aws-production/README.md) | Bedrock, EKS+KServe, vLLM, SageMaker |
| **11** | [Security & Guardrails](docs/11-security-guardrails/README.md) | Injection, poisoning, defense in depth |
| **12** | [E2E Blueprint](docs/12-e2e-blueprint/README.md) | The complete production system map |

---

## 💻 Examples

Runnable code for every major section.

| | | |
|---|---|---|
| 🟦 [examples/01_rag/](examples/01_rag/) | Hybrid RAG + reranker + faithfulness check | [`rag_basic.py`](examples/01_rag/rag_basic.py) |
| 🤖 [examples/02_agent/](examples/02_agent/) | Multi-agent supervisor with LangGraph | [`langgraph_supervisor.py`](examples/02_agent/langgraph_supervisor.py) |
| 🎯 [examples/03_finetune/](examples/03_finetune/) | LoRA + QLoRA + DPO training | [`lora_sft.py`](examples/03_finetune/lora_sft.py) · [`dpo_example.py`](examples/03_finetune/dpo_example.py) |
| ⚡ [examples/04_serving/](examples/04_serving/) | vLLM config + KServe manifest | [`vllm_config.py`](examples/04_serving/vllm_config.py) · [`kserve_manifest.yaml`](examples/04_serving/kserve_manifest.yaml) |
| ☁️ [examples/05_bedrock/](examples/05_bedrock/) | AWS Bedrock + Knowledge Base RAG | [`bedrock_basic.py`](examples/05_bedrock/bedrock_basic.py) |
| 📊 [examples/06_eval/](examples/06_eval/) | LLM-as-judge eval harness | [`llm_as_judge.py`](examples/06_eval/llm_as_judge.py) |

---

## 🧭 The Core Principle

```text
Architecture tells which model to choose.
Fine-tuning tells how to adapt it.
RAG connects private/external knowledge.
Agents connect tools, workflows, and memory.
Serving runs the system at scale.
LLMOps evaluates, monitors, and rolls back.
Security prevents leakage, poisoning, and abuse.
```

---

## 📊 What's Tracked

| Category | Coverage |
|---|---|
| LLM models | Qwen, DeepSeek, Kimi, MiniMax, Llama, Gemma, Jamba, RWKV, Nemotron, Titans |
| Architecture methods | RoPE, GQA, MLA, MoE, KDA, Mamba, Mamba-2 — all main + watchlist |
| Serving methods | FlashAttention 1–3, SageAttention 1–3, PagedAttention, vAttention, RadixAttention |
| Fine-tuning | Full FT, LoRA, QLoRA, DoRA, AdaLoRA, IA3 |
| Alignment | PPO, DPO, ORPO, SimPO, KTO, GRPO, RLAIF |
| AWS services | Bedrock, Bedrock Agents, AgentCore, SageMaker, EKS+KServe, ECS, OpenSearch, Ray |
| Diagrams | 10 Mermaid architecture diagrams |
| Paper references | 90+ with canonical URLs |

---

## 🛠️ Who This Is For

| Audience | Where to start |
|---|---|
| 🧑‍💻 **AI Engineer starting out** | Read the notebook top-to-bottom |
| 👩‍🔬 **ML Engineer moving to LLMs** | Sections 1, 5, 6, 7, 9 |
| 🧱 **Backend Engineer building AI products** | Sections 7, 8, 9, 10, 11, 12 |
| 🏗️ **Solutions Architect** | Sections 9, 10, 11, 12 + `examples/05_bedrock/` |
| 📊 **Tech Lead / EM** | Section 12 (E2E blueprint) + decision trees throughout |

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New model reports, architecture methods, and production patterns are welcome — please follow the watchlist vs. main classification.

---

<div align="center">

**⭐ Star this repo if it helps you build production GenAI systems.**

</div>