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

## 3.3 Multimodal Embeddings

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

## 3.4 Decision Tree — Which VLM Pattern?

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
    → Unified / Interleaved (Qwen-VL / Qwen-Omni)
```

---

## 3.5 Production Rule

```text
Most enterprise VLMs in 2025–2026 use Pattern 2 (projector + LLM) for chat
or Pattern 5 (unified tokens) for advanced multimodal reasoning.

For document QA, prefer ColPali-style multi-vector retrieval over text-only OCR pipelines.
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part C2 / Section 3 / Section 4`
- **Code example:** [examples/01_rag/](../examples/01_rag/) (ColPali RAG)
- **Diagrams:** [assets/diagrams/vlm-patterns.md](../../assets/diagrams/vlm-patterns.md)