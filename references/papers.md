# Curated Paper Index

> Single source of truth for all papers, reports, and blog posts referenced across the hub.
> Organized by section. Links are official sources where possible.

---

## Architecture (Section 1)

### Foundation
- **Transformer** — https://arxiv.org/abs/1706.03762
- **RoPE / RoFormer** — https://arxiv.org/abs/2104.09864
- **GQA** — https://arxiv.org/abs/2305.13245
- **MQA** — https://arxiv.org/abs/1911.02150

### Positional Encoding
- **YaRN** — https://arxiv.org/abs/2309.00071
- **LongRoPE** — https://arxiv.org/abs/2402.13753
- **LongRoPE2** — https://arxiv.org/abs/2502.20082
- **NoPE** — https://arxiv.org/abs/2505.11199
- **Periodic RoPE / P-RoPE** — https://arxiv.org/abs/2605.27980
- **MHRoPE / MRoPE-I** — https://arxiv.org/abs/2510.23095
- **HoPE (Hyperbolic)** — https://arxiv.org/abs/2509.05218
- **HoPE (Hybrid VLM)** — https://arxiv.org/abs/2505.20444
- **iRoPE (Implicit RoPE)** — https://ai.meta.com/blog/llama-4-multimodal-intelligence/ (introduced in Llama 4)
- **TMRoPE (Time-aligned RoPE)** — https://arxiv.org/abs/2503.20215 (Qwen2.5-Omni)
- **Interleaved-MRoPE** — https://arxiv.org/abs/2511.21631 (Qwen3-VL DeepStack + MRoPE)
- **LazyAttention** — https://arxiv.org/abs/2606.04302 (deferred PE for KV reuse in RAG)

### Attention Mechanisms
- **MLA (DeepSeek-V3)** — https://arxiv.org/abs/2412.19437
- **NSA (Native Sparse Attention)** — https://arxiv.org/abs/2502.11089
- **Lightning Attention** — https://arxiv.org/abs/2501.08313 (MiniMax-01)
- **Lightning Attention (M1)** — https://arxiv.org/abs/2506.13585
- **KDA (Kimi Delta Attention)** — https://arxiv.org/abs/2510.26692
- **Gated DeltaNet** — https://arxiv.org/abs/2412.06410 (linear attention, Qwen3-Next / Kimi Linear)
- **DeepSeek Sparse Attention (DSA)** — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp (DeepSeek-V3.2)

### FFN / MoE
- **GLU Family / SwiGLU** — https://arxiv.org/abs/2002.05202
- **DeepSeekMoE** — https://arxiv.org/abs/2401.06066
- **Adaptive Routing (Ada-K)** — https://arxiv.org/abs/2410.10456
- **Mixture-of-Depths** — https://arxiv.org/abs/2404.02258
- **PEER** — https://arxiv.org/abs/2407.04153
- **MoE++** — https://arxiv.org/abs/2410.07348
- **SeqTopK** — https://arxiv.org/abs/2511.06494
- **Sub-MoE / Expert Merging** — https://arxiv.org/abs/2506.23266
- **REAM** — https://arxiv.org/abs/2604.04356
- **SoftMoE (Soft Mixture-of-Experts)** — https://arxiv.org/abs/2308.00951
- **Expert-Choice Routing** — https://arxiv.org/abs/2202.09368

### Beyond Transformer
- **Mamba** — https://arxiv.org/abs/2312.00752
- **Mamba-2** — https://arxiv.org/abs/2405.21060
- **Jamba** — https://arxiv.org/abs/2403.19887
- **Nemotron-H** — https://arxiv.org/abs/2504.03624
- **RWKV-7** — https://arxiv.org/abs/2503.14456
- **RetNet** — https://arxiv.org/abs/2307.08621
- **Titans** — https://arxiv.org/abs/2501.00663
- **xLSTM** — https://arxiv.org/abs/2405.04517
- **ATLAS / DeepTransformers** — https://arxiv.org/abs/2505.23735
- **ModRWKV** — https://arxiv.org/abs/2505.14505
- **VMamba** — https://arxiv.org/abs/2401.10166
- **Vision Mamba / Vim** — https://arxiv.org/abs/2401.09417
- **MambaVision** — https://arxiv.org/abs/2407.08083

### Model Reports
- **Qwen3** — https://arxiv.org/abs/2505.09388
- **Qwen2.5-VL** — https://arxiv.org/abs/2502.13923
- **Qwen2.5-Omni** — https://arxiv.org/abs/2503.20215
- **Qwen3-VL** — https://arxiv.org/abs/2511.21631
- **Qwen3-Next (Gated DeltaNet)** — https://arxiv.org/abs/2505.09388 (Qwen3 family)
- **DeepSeek-V3** — https://arxiv.org/abs/2412.19437
- **DeepSeek-R1** — https://arxiv.org/abs/2501.12948
- **DeepSeek-V3.2-Exp (DSA)** — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- **Kimi Linear** — https://arxiv.org/abs/2510.26692
- **MiniMax-01** — https://arxiv.org/abs/2501.08313
- **MiniMax-M1** — https://arxiv.org/abs/2506.13585
- **Gemma 3** — https://arxiv.org/abs/2503.19786
- **Llama 4 (iRoPE)** — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Jamba** — https://arxiv.org/abs/2403.19887
- **Nemotron-H** — https://arxiv.org/abs/2504.03624

---

## Attention / Serving Kernels (Section 2)

- **FlashAttention** — https://arxiv.org/abs/2205.14135
- **FlashAttention-2** — https://arxiv.org/abs/2307.08691
- **FlashAttention-3** — https://arxiv.org/abs/2407.08608
- **SageAttention** — https://arxiv.org/abs/2410.02367
- **SageAttention2** — https://arxiv.org/abs/2411.10958
- **SageAttention2++** — https://github.com/thu-ml/SageAttention (in-tree extension of SageAttention2)
- **SageAttention3** — https://arxiv.org/abs/2505.11594
- **FlashInfer** — https://arxiv.org/abs/2501.01005
- **FlexAttention** — https://arxiv.org/abs/2412.05496
- **xFormers** — https://github.com/facebookresearch/xformers
- **PagedAttention** — https://arxiv.org/abs/2309.06180
- **vAttention** — https://arxiv.org/abs/2405.04437
- **RadixAttention** — https://arxiv.org/abs/2312.07104
- **FlashMLA** — https://github.com/deepseek-ai/FlashMLA (DeepSeek MLA decoding kernels)
- **KV Cache Quantization** — https://arxiv.org/abs/2401.18079
- **KVTuner** — https://arxiv.org/abs/2502.04420
- **KVLinC** — https://arxiv.org/abs/2510.05373
- **TurboQuant** — https://arxiv.org/abs/2504.19874 (Google KV cache quantization)
- **RingAttention** — https://arxiv.org/abs/2310.01889
- **DeepSpeed-Ulysses** — https://arxiv.org/abs/2309.14509
- **USP** — https://arxiv.org/abs/2405.07719
- **LazyAttention (RAG kernel)** — https://arxiv.org/abs/2606.04302

---

## VLM / Multimodal (Section 3)

### Foundations
- **CLIP** — https://arxiv.org/abs/2103.00020
- **BLIP-2** — https://arxiv.org/abs/2301.12597
- **LLaVA project** — https://llava-vl.github.io/
- **InternVL** — https://arxiv.org/abs/2312.14238
- **SigLIP** — https://arxiv.org/abs/2303.15343
- **ColPali** — https://arxiv.org/abs/2407.01449
- **HF VLM Design Blog** — https://huggingface.co/blog/gigant/vlm-design

### Frontier VLMs (2025–2026)
- **Qwen2.5-VL** — https://arxiv.org/abs/2502.13923 (MRoPE)
- **Qwen2.5-Omni** — https://arxiv.org/abs/2503.20215 (TMRoPE for time alignment)
- **Qwen3-VL** — https://arxiv.org/abs/2511.21631 (Interleaved-MRoPE + DeepStack)
- **Gemma 3** — https://arxiv.org/abs/2503.19786 (local-global attention for vision)
- **Llama 4 (multimodal + iRoPE)** — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **MiniMax-01 / MiniMax-M1 (VLM lightning attention)** — https://arxiv.org/abs/2501.08313 / https://arxiv.org/abs/2506.13585

### VLM Inference & Attention
- **LazyAttention (RAG with deferred PE)** — https://arxiv.org/abs/2606.04302 (serves multimodal long-context RAG)
- **DeepSeek Sparse Attention (DSA)** — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp (image-text token-level sparse)
- **FlashMLA (multimodal serving)** — https://github.com/deepseek-ai/FlashMLA (powers DeepSeek-V3.2 multimodal serving)

### Position Encoding for Multimodal (cross-ref Section 1)
- **Interleaved-MRoPE** — https://arxiv.org/abs/2511.21631
- **TMRoPE (text-time RoPE)** — https://arxiv.org/abs/2503.20215
- **HoPE (Hybrid PE for VLM / long-video)** — https://arxiv.org/abs/2505.20444
- **MHRoPE / MRoPE-I (multimodal RoPE variants)** — https://arxiv.org/abs/2510.23095

---

## Coding Models (Section 4)

- **Code Llama** — https://huggingface.co/blog/codellama
- **StarCoder** — https://huggingface.co/blog/starcoder
- **DeepSeek Coder** — https://github.com/deepseek-ai/DeepSeek-Coder
- **Qwen2.5-Coder** — https://www.alibabacloud.com/blog/qwen2-5-coder-series-powerful-diverse-practical_601765
- **Codestral** — https://mistral.ai/news/codestral
- **Devstral** — https://mistral.ai/news/devstral

---

## Fine-Tuning / PEFT (Section 5)

- **LoRA** — https://arxiv.org/abs/2106.09685
- **QLoRA** — https://arxiv.org/abs/2305.14314
- **DoRA** — https://arxiv.org/abs/2402.09353
- **PEFT library** — https://github.com/huggingface/peft
- **Unsloth** — https://github.com/unslothai/unsloth
- **TRL** — https://github.com/huggingface/trl
- **Axolotl** — https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** — https://github.com/hiyouga/LLaMA-Factory
- **DeepSpeed** — https://www.deepspeed.ai/

---

## Alignment / RLHF (Section 6)

- **DPO** — https://arxiv.org/abs/2305.18290
- **ORPO** — https://arxiv.org/abs/2403.07691
- **KTO** — https://arxiv.org/abs/2402.01306
- **GRPO / DeepSeekMath** — https://arxiv.org/abs/2402.03300
- **SimPO** — https://arxiv.org/abs/2405.14734

---

## Frameworks (for research/training — production frameworks moved to external repo)

- **vLLM** — https://docs.vllm.ai/
- **TGI** — https://huggingface.co/docs/text-generation-inference
- **OpenRLHF** — https://github.com/OpenRLHF/OpenRLHF
- **verl** — https://github.com/volcengine/verl
- **NeMo** — https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/

> Note: LangChain, LangGraph, LlamaIndex, Ray, Bedrock, AgentCore, KServe, OpenSearch and similar application/production-stack references were moved to an external production repo (sections 7-12 of the docs).

---

## How to Add a New Reference

Edit this file following the existing structure:

```markdown
- **Paper Name** — https://link-to-official-source
```

Keep entries in **alphabetical order within each section**. Use the canonical/most-authoritative URL (arXiv, official blog, GitHub).