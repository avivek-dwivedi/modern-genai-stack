# Changelog

All notable changes to this hub are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

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