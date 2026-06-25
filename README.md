<div align="center">

# Modern GenAI Stack

**The research and engineering reference for modern Generative AI systems.**

From LLM architecture internals to attention/serving kernels, multimodal models, fine-tuning, and alignment — in one place.

[📓 Master Notebook](end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb) · [📚 Docs](docs/) · [💻 Examples](examples/) · [📑 Papers](references/papers.md)

</div>

---

## Why This Repo Exists

Building modern LLMs, VLMs and reasoning systems requires understanding model internals, attention/serving kernels, multimodal extensions, fine-tuning, and alignment. **Most docs cover one layer. This hub covers the research and engineering layers, in order, with runnable code.**

| | | |
|---|---|---|
| 🏗️ **Architecture-deep** | Attention internals, MoE routing, beyond-transformer (Mamba, RWKV, KDA) |
| 🚀 **Serving-aware** | FlashAttention, SageAttention, PagedAttention, KV-cache quantization, speculative decoding |
| 🖼️ **Multimodal-ready** | CLIP, BLIP-2, LLaVA, Qwen-VL, ColPali, MRoPE / iRoPE / DeepStack |
| 🎯 **Fine-tuning & alignment** | LoRA / QLoRA / DoRA, PPO → DPO → ORPO → GRPO |

> Sections 7-12 (RAG, agents, AWS production, security, E2E blueprint) have been moved to an **external production repo**. This hub focuses on the research and engineering layers.

---

## 🚀 Quick Start

```bash
git clone https://github.com/avivek-dwivedi/modern-genai-stack.git
cd modern-genai-stack

# Open the master notebook
jupyter lab end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb

# Run a fine-tuning example
cd examples/03_finetune
pip install -r ../../requirements.txt
python lora_sft.py
```

---

## 🗺️ The 6-Section Learning Path

| # | Section | What's inside |
|---:|---|---|
| **01** | [LLM Architecture](docs/01-architecture/README.md) | RoPE, GQA, MLA, MoE, Mamba, RWKV |
| **02** | [Attention & Serving Kernels](docs/02-attention-serving/README.md) | FlashAttention, PagedAttention, KV-cache |
| **03** | [VLM & Multimodal](docs/03-vlm-multimodal/README.md) | CLIP, BLIP-2, LLaVA, Qwen-VL, ColPali |
| **04** | [Coding Models](docs/04-coding-models/README.md) | Code Llama, DeepSeek Coder, Qwen Coder, Kimi K2 |
| **05** | [Fine-Tuning & PEFT](docs/05-finetuning-peft/README.md) | LoRA, QLoRA, DoRA, capacity planning |
| **06** | [Alignment & RLHF](docs/06-alignment-rlhf/README.md) | PPO → DPO → ORPO → GRPO |

> Sections 7-12 (RAG / Agents / LLMOps / AWS / Security / E2E Blueprint) are now in an external production repo.

---

## 💻 Examples

Runnable code for the active sections.

| | | | |
|---|---|---|---|
| 🎯 [examples/03_finetune/](examples/03_finetune/) | LoRA + QLoRA + DPO training | [`lora_sft.py`](examples/03_finetune/lora_sft.py) · [`dpo_example.py`](examples/03_finetune/dpo_example.py) |
| ⚡ [examples/04_serving/](examples/04_serving/) | vLLM config + serving patterns | [`vllm_config.py`](examples/04_serving/vllm_config.py) |

> RAG / Agent / AWS / Eval examples were moved to the external production repo.

---

## 🧭 The Core Principle

```text
Architecture tells which model to choose.
Attention/serving tells how cheaply it runs.
Multimodal extends it to vision and audio.
Fine-tuning adapts it to your task.
Alignment shapes its behavior with PPO / DPO / GRPO.
```

---

## 📊 What's Tracked

| Category | Coverage |
|---|---|
| LLM models | Qwen, DeepSeek, Kimi, MiniMax, Llama, Gemma, Jamba, RWKV, Nemotron, Titans |
| Architecture methods | RoPE (incl. iRoPE / P-RoPE / Interleaved-MRoPE), GQA, MLA, MoE, KDA, Gated DeltaNet, Mamba, Mamba-2 |
| Attention mechanisms | Mamba, RWKV, RetNet, KDA, Gated DeltaNet, NSA, MLA, DSA, Lightning Attention |
| Serving kernels | FlashAttention 1-3, SageAttention 1-3 (+2++), PagedAttention, vAttention, RadixAttention, FlashMLA, FlashInfer, TurboQuant |
| VLM serving | DSA (DeepSeek-V3.2), DeepStack, Local-Global Vision Attention, LazyAttention |
| Fine-tuning | Full FT, LoRA, QLoRA, DoRA, AdaLoRA, IA3 |
| Alignment | PPO, DPO, ORPO, SimPO, KTO, GRPO, RLAIF |
| Diagrams | Mermaid architecture diagrams |
| Paper references | 90+ with canonical URLs |

---

## 🛠️ Who This Is For

| Audience | Where to start |
|---|---|
| 🧑‍💻 **AI Engineer starting out** | Read the notebook top-to-bottom (sections 1-6) |
| 👩‍🔬 **ML Engineer moving to LLMs** | Sections 1, 5, 6 |
| 🧱 **Backend Engineer serving models** | Section 2 (attention/serving) + examples/04_serving |
| 🏗️ **Solutions Architect picking models** | Section 1 (architecture) + Section 3 (VLM/multimodal) |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New model reports, architecture methods, and attention/serving patterns are welcome — please follow the `emerging` vs. `main` classification.

---

## 📜 License

CC BY 4.0 — see [LICENSE](LICENSE). This hub is a curated aggregation of publicly available research papers and blog references; original sources remain the property of their authors.

---

<div align="center">

**⭐ Star this repo if it helps you build modern Generative AI systems.**

</div>
