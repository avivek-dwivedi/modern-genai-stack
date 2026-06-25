# Section 3 — VLM & Multimodal

> **How do we extend text LLMs to see, hear, and speak?**

This section covers Vision-Language Models (VLMs), multimodal embeddings, and the bridge patterns that connect vision/audio encoders to LLM decoders.

---

## Why This Section Exists

A VLM is **not** an LLM plus an image input. It needs:

1. A **vision encoder** (usually ViT-style)
2. A **bridge / projector** to map vision features into LLM token space
3. **Multimodal position encoding** (MRoPE / TMRoPE)
4. **Multimodal instruction data**
5. **Special tokenization** for image / video / audio

Choosing the right pattern determines token cost, latency, and capability.

---

## 3.1 Architecture Evolution

```text
CNN / ViT
↓
CLIP / ALIGN
↓
BLIP
↓
BLIP-2 + Q-Former
↓
LLaVA / MiniGPT-4
↓
InternVL / Qwen-VL
↓
Qwen-Omni / Omni models
↓
Unified multimodal / native omni architectures
```

---

## 3.2 Architecture Patterns

### Pattern 1 — Dual Encoder

```text
Image → Vision Encoder → Image Embedding
Text  → Text Encoder   → Text Embedding
Image/Text similarity through contrastive loss
```

**Best for:** image retrieval, text-image matching, zero-shot classification
**Not best for:** visual reasoning, long visual QA, multi-turn image chat

**Examples:** CLIP, OpenCLIP, ALIGN, SigLIP

### Pattern 2 — Vision Encoder + Linear Projector + LLM

```text
Image
→ CLIP / ViT vision encoder
→ linear projector / MLP projector
→ image tokens in LLM embedding space
→ LLM decoder generates answer
```

**Best for:** visual chat, document QA, image reasoning
**Limitation:** projector may lose fine-grained visual information

**Examples:** LLaVA, MiniGPT-4, early open VLMs

### Pattern 3 — Q-Former / Learned Query Bridge

```text
Image patches
→ Vision encoder
→ Q-Former learned queries attend to image features
→ compressed visual tokens
→ LLM
```

**Best for:** token-budget-constrained VLMs
**Examples:** BLIP-2, InstructBLIP

### Pattern 4 — Perceiver Resampler

```text
Image features
→ learned latent queries
→ fixed number of visual tokens
→ language model with cross-attention
```

**Examples:** Flamingo-style architectures

### Pattern 5 — Unified / Interleaved Multimodal Token Architecture

```text
Text tokens + image tokens + video tokens + audio tokens
→ unified model context
→ multimodal attention
→ text/audio/image/video response
```

**Best for:** deep cross-modal reasoning (video + timestamp + text + speech)
**Examples:** Qwen-VL, Qwen3-VL, Qwen-Omni, Gemini-style, GPT-4o-style

---

## 3.3 Multimodal Position Encoding (2025–2026 update)

Modern VLMs extend RoPE into multimodal space. Each variant handles a different combination of text, image, video, and time.

| Method | Handles | Used In | Source |
|---|---|---|---|
| MRoPE | Spatial-temporal (image + video) | Qwen2.5-VL | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| TMRoPE | Text-time RoPE (text + audio + video) | Qwen2.5-Omni | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215) |
| Interleaved-MRoPE | Interleaved text/image/video tokens | Qwen3-VL | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |
| DeepStack | Multi-level ViT feature fusion (VLM) | Qwen3-VL | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |
| iRoPE | Implicit RoPE for vision-language | Llama 4 | [Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| HoPE (Hybrid VLM) | VLM / long-video | Research | [arXiv:2505.20444](https://arxiv.org/abs/2505.20444) |
| MHRoPE / MRoPE-I | Multi-head frequency multimodal | Research | [arXiv:2510.23095](https://arxiv.org/abs/2510.23095) |
| Local-Global Vision Attention | Image patches split into local + global | Gemma 3 | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| LazyAttention (deferred PE) | RAG over multimodal long docs | Research (ICML 2026) | [arXiv:2606.04302](https://arxiv.org/abs/2606.04302) |

### Core Takeaway

```text
MRoPE / TMRoPE put position into image/video/audio time.
Interleaved-MRoPE + DeepStack (Qwen3-VL) is the current open-weight SOTA pattern.
iRoPE (Llama 4) is the leading closed-ish open-weight alternative.
LazyAttention solves RAG-side position-agnostic KV reuse, not a PE per se.
```

---

## 3.4 Multimodal Serving Attention (cross-ref Section 2)

These are the attention / serving pieces that VLMs actually run on in production.

| Method | Role | Source |
|---|---|---|
| DSA (DeepSeek Sparse Attention) | Token-level sparse multimodal attention | [DeepSeek-V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| FlashMLA + DSA | Multimodal MLA decoding + sparse FP8 KV | [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA) |
| MambaVision / Vision Mamba | Mamba-based vision backbones | [arXiv:2407.08083](https://arxiv.org/abs/2407.08083) / [arXiv:2401.09417](https://arxiv.org/abs/2401.09417) |
| ModRWKV | Multimodal RWKV | [arXiv:2505.14505](https://arxiv.org/abs/2505.14505) |
| Lightning Attention (M1) | Long-context hybrid attention for VLMs | [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) |

---

## 3.5 Multimodal Embeddings

| Type | Meaning | Examples |
|---|---|---|
| Dense text embedding | One vector per text / chunk | OpenAI embeddings, BGE, Jina |
| Dense image embedding | One vector per image | CLIP, SigLIP, OpenCLIP |
| Cross-modal embedding | Image and text in same space | CLIP, SigLIP |
| Sparse embedding | Lexical / token-weighted representation | BM25, SPLADE-style |
| Multi-vector embedding | Multiple vectors per document/page/image | ColBERT, ColPali |
| Multimodal embedding | Text + image + layout-aware embedding | ColPali, multimodal retrievers |

### ColPali / Multi-vector Document Embedding

Instead of OCR-ing a page first:

```text
PDF page / image
→ vision-language encoder
→ multiple page-level visual/text vectors
→ late interaction retrieval
```

**Why useful:** tables, layout, charts, and visual structure can be retrieved without OCR-only loss.

### Core Takeaway

```text
CLIP/SigLIP are strong for image-text similarity.
BGE/Jina/NV-Embed are strong for text retrieval.
ColPali-style models are strong for document/page retrieval because they preserve visual layout.
```

---

## 3.6 Decision Tree — Which VLM Pattern?

```text
Image-text similarity / retrieval only?
    → Dual Encoder (CLIP / SigLIP)

Visual chat on natural images?
    → Vision Encoder + Projector + LLM (LLaVA pattern)

Limited token budget / many image patches?
    → Q-Former / Perceiver

Document understanding (PDFs, charts)?
    → ColPali-style multi-vector retrieval

Real-time multimodal interaction (text + audio + video)?
    → Unified / Interleaved (Qwen-VL / Qwen-Omni / Llama 4)

Long-context multimodal RAG over images/docs?
    → Unified tokens + LazyAttention (deferred PE for KV reuse)
```

---

## 3.7 Production Rule

```text
Most enterprise VLMs in 2025–2026 use Pattern 2 (projector + LLM) for chat
or Pattern 5 (unified tokens) for advanced multimodal reasoning.

For document QA, prefer ColPali-style multi-vector retrieval over text-only OCR pipelines.

For multimodal serving at scale, use a backend that supports your VLM's RoPE variant
(MRoPE / Interleaved-MRoPE / TMRoPE / iRoPE) and consider sparse attention (DSA) for
token-heavy image+text prefills.
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part C2 / Section 3 / Section 4`
- **Code example:** [examples/01_rag/](../examples/01_rag/) (ColPali RAG)
- **Diagrams:** [assets/diagrams/vlm-patterns.md](../../assets/diagrams/vlm-patterns.md)