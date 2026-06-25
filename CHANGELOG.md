# Changelog

All notable changes to this hub are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [V1.5] — 2026-06-25

### 🧭 Scope Refocus

The hub now focuses **exclusively** on the **research and engineering layers** of modern Generative AI. Sections 7–12 (RAG, agents, AWS production, security, E2E blueprint) have been **moved to an external production repo**.

Sections remaining in this hub (sections 1–6):

1. LLM Architecture (PE, attention, MoE, beyond-Transformer, multimodal backbone)
2. Attention & Serving Kernels (FlashAttention, SageAttention, PagedAttention, KV-cache quantization, VLM/inference kernels)
3. VLM & Multimodal (CLIP, BLIP-2, LLaVA, Qwen-VL, ColPali, MRoPE / Interleaved-MRoPE / iRoPE / DeepStack)
4. Coding Models
5. Fine-Tuning & PEFT
6. Alignment & RLHF

### ✨ Added (2026 coverage update)

- **New PE methods** with verified canonical sources: Periodic RoPE / P-RoPE ([arXiv:2605.27980](https://arxiv.org/abs/2605.27980)), LazyAttention ([arXiv:2606.04302](https://arxiv.org/abs/2606.04302)), iRoPE (Llama 4 blog), Interleaved-MRoPE + DeepStack (Qwen3-VL).
- **New attention mechanisms**: DSA (DeepSeek-V3.2 sparse attention), Gated DeltaNet, FlashMLA (DeepSeek MLA decoding kernels).
- **New FFN/MoE methods**: REAM ([arXiv:2604.04356](https://arxiv.org/abs/2604.04356)), SoftMoE ([arXiv:2308.00951](https://arxiv.org/abs/2308.00951)), Expert-Choice Routing ([arXiv:2202.09368](https://arxiv.org/abs/2202.09368)).
- **New serving kernels**: TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), SageAttention2++ (in-tree), FlashMLA + DSA.
- **New VLM/multimodal attention**: DSA, DeepStack, Local-Global Vision Attention (Gemma 3), iRoPE (Llama 4), ModRWKV.
- **New section**: `docs/02-attention-serving/README.md` now has a dedicated `2.4 VLM / Multimodal Inference` sub-section.

### 🗑️ Removed (moved to external production repo)

- `docs/07-ragops/` → external repo
- `docs/08-agentops/` → external repo
- `docs/09-llmops-evalops/` → external repo
- `docs/10-aws-production/` → external repo
- `docs/11-security-guardrails/` → external repo
- `docs/12-e2e-blueprint/` → external repo
- `examples/01_rag/`, `examples/02_agent/`, `examples/05_bedrock/`, `examples/06_eval/` → external repo
- 42 cells from `end_to_end_llm_vlm_rag_agentops_master_notebook.ipynb`

### 🔧 Fixed

- **Broken placeholder links**: Replaced every `Watchlist` / `TBD` entry in `references/papers.md` and section READMEs with verified arXiv / GitHub sources.
- **Stale "Notebook section" lines** in `docs/04-coding-models`, `docs/05-finetuning-peft`, `docs/06-alignment-rlhf`: now point to the correct single section (4, 5, 6 respectively).
- **Duplicated table** in `docs/02-attention-serving` 2.5 Architecture Cheat Sheet (was rendered twice).
- **`.github/README.md`**: clarified that GitHub renders this folder-level README, with pointer back to the hub's main `/README.md`.

### 🧹 Cleaned

- All `Watchlist` maturity labels → `emerging` (more accurate framing).
- All `TBD` / "search official sources" entries → real verified links.
- Removed `AWS / Bedrock` and `LangChain / LangGraph / LlamaIndex / Ray` from `references/papers.md` (production-stack references moved to external repo).

---

## [V1.4] — 2026-06-22

### ✨ Added

- **Section reorganization**: Notebook split into 12 cleanly-ordered sections (architecture → E2E blueprint)
- **Master notebook rewrite**: Removed 3 duplicate intro cells, deduplicated model maps, fixed Section 7A placement
- **Per-section explanatory headers**: Each notebook section now has a 1-line "what this section is for" callout
- **`docs/` tree**: 12 section READMEs with decision trees, taxonomies, and reference links
- **`examples/` tree**: Runnable code for RAG, agents, fine-tuning, serving, Bedrock, and eval
- **`assets/diagrams/`**: Mermaid source for architecture diagrams
- **`references/papers.md`**: Curated paper index with links
- **Issue templates**: GitHub issue templates for content proposals and bug reports
- **`.gitignore`**: Standard ML/AI gitignore (Python, models, checkpoints, secrets)

### 🔧 Fixed

- Removed duplicate "Master Taxonomy" tables (consolidated into `docs/01-architecture/`)
- Removed duplicate "Closed-Source Tracking Notes" (single canonical location in `docs/01-architecture/closed-source.md`)
- Removed duplicate "GitHub Page Structure" (now lives only in `docs/README.md`)
- Removed duplicate "Model-Wise Architecture Map" (single canonical source: `docs/01-architecture/model-map.md`)
- Removed duplicate "Master Reading List" (single source: `references/papers.md`)
- **Section 7A placement**: Attention/Serving optimization is now its own top-level section (Section 2), not wedged inside architecture
- **Section 8 (Closed-source)**: Moved from inside model-architecture to an appendix in `docs/01-architecture/closed-source.md`

### 🧹 Cleaned

- All content sections now follow the same internal structure:
  1. Why this topic exists
  2. Evolution timeline
  3. Method-by-method cheat sheet (table)
  4. Decision tree
  5. Code example link
  6. Source links

---

## [V1.3] — 2025-08

### ✨ Added

- VLM / multimodal architecture evolution
- Multimodal embeddings (CLIP, SigLIP, BGE, ColPali)
- Coding models deep-dive (Code Llama, StarCoder, DeepSeek Coder, Qwen Coder, Codestral, Kimi K2)
- Fine-tuning method comparison
- PEFT methods (LoRA, QLoRA, DoRA, AdaLoRA, IA3)
- RLHF / preference optimization evolution (PPO → DPO → ORPO → GRPO → SimPO)
- Training frameworks (Transformers, PEFT, TRL, Unsloth, Axolotl, LLaMA-Factory, DeepSpeed, FSDP, NeMo)
- VLM fine-tuning recipes
- Hardware and AWS capacity planning

---

## [V1.2] — 2025-06

### ✨ Added

- Gap audit after deep search
- New positional encoding methods (LongRoPE2, Periodic RoPE, MHRoPE)
- New attention architecture methods (Native Sparse Attention, Lightning Attention)
- Attention inference / kernel optimization (FlashAttention-3, SageAttention2/3, FlashMLA, FlexAttention)
- KV-cache quantization methods (KVTuner, KVLinC, TurboQuant)
- Disaggregated prefill/decode, speculative decoding
- MoE routing (SeqTopK, SoftMoE, expert-choice)
- Beyond-Transformer additions (Gated DeltaNet, ATLAS, ModRWKV, VMamba, Vision Mamba, xLSTM)
- Watchlist vs Main roadmap separation

---

## [V1.1] — 2025-04

### ✨ Added

- Initial LLM architecture roadmap
- Positional encoding (RoPE family, multimodal RoPE)
- Attention mechanisms (MQA, GQA, MLA, KDA)
- FFN / MoE (SwiGLU, DeepSeekMoE, High-Sparsity MoE)
- Beyond Transformer (Mamba, Mamba-2, Jamba, RWKV, RetNet, Titans)
- Multimodal architecture (Qwen-VL family)
- Model-wise architecture map (Qwen, DeepSeek, Kimi, Gemma, Llama, Jamba, Nemotron)
- Four-week study plan