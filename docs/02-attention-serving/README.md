# Section 2 — Attention & Serving Kernels

> **How do we run this model fast and cheap in production?**

This section is the **performance bridge** between architecture (Section 1) and downstream model adaptation (Sections 3–6).

---

## Why This Section Exists

Most attention/serving optimizations are **NOT new model architectures** — they are kernel, memory, and scheduling tricks that make the same model run faster and cheaper. Confusing these buckets leads to bad decisions.

### The Three Buckets (don't mix them)

```text
Model architecture attention:
  MHA, MQA, GQA, MLA, sparse attention, local-global attention, KDA

Attention kernel optimization:
  FlashAttention, SageAttention, FlashInfer, FlexAttention

KV-cache / serving optimization:
  PagedAttention, vAttention, RadixAttention, prefix caching, chunked prefill

Distributed long-context systems:
  RingAttention, DeepSpeed-Ulysses, USP
```

---

## 2.1 Attention Kernels

| Method | Bucket | What It Solves | Source |
|---|---|---|---|
| FlashAttention | Kernel | Computes exact attention faster by reducing HBM memory traffic | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2 | Kernel | Better parallelism and work partitioning | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |
| FlashAttention-3 | Kernel | Hopper/H100 async + FP8 attention | [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) |
| xFormers Memory-Efficient Attention | Library | Practical attention backend in PyTorch ecosystem | [xFormers GitHub](https://github.com/facebookresearch/xformers) |
| SageAttention | Quantized kernel | 8-bit attention acceleration | [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) |
| SageAttention2 | Quantized kernel | INT4/FP8 with smoothing/outlier handling | [arXiv:2411.10958](https://arxiv.org/abs/2411.10958) |
| SageAttention3 | Quantized kernel | FP4 attention direction for Blackwell-era GPUs | [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) |
| SageAttention2++ | Quantized kernel | Extension of SageAttention2 (in-tree) | [SageAttention GitHub](https://github.com/thu-ml/SageAttention) |
| FlashInfer | Inference engine | Optimized attention kernels and runtime pieces | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) |
| FlexAttention | Programmable | Express attention variants, get optimized kernels | [arXiv:2412.05496](https://arxiv.org/abs/2412.05496) |
| FlashMLA | MLA-specific runtime | Optimized serving for MLA-style attention (DeepSeek) | [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA) |

### Decision Tree

```text
Default attention backend?
    → FlashAttention-2 / FlashAttention-3

H100 / Hopper GPU?
    → FlashAttention-3 (FP8)

Blackwell GPU?
    → SageAttention3 (FP4)

Need custom attention pattern?
    → FlexAttention

Need MLA-specific kernel?
    → FlashMLA (DeepSeek, FP8 sparse + dense)

Need sparse token-level attention (VLM / long-context)?
    → DSA (DeepSeek-V3.2 sparse kernels in FlashMLA repo)
```

---

## 2.2 KV-Cache & Serving Optimization

| Method | Bucket | What It Solves | Source |
|---|---|---|---|
| PagedAttention | KV-cache memory | Manages KV cache in blocks to reduce fragmentation | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| vAttention | KV-cache virtual memory | Virtual memory to manage KV cache with logical contiguity | [arXiv:2405.04437](https://arxiv.org/abs/2405.04437) |
| RadixAttention | Prefix/KV reuse | Reuses KV cache for shared prompt prefixes | [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) |
| Prefix Caching | Serving | Reuses precomputed KV for repeated prompts | [vLLM docs](https://docs.vllm.ai/en/latest/automatic_prefix_caching/details.html) |
| Chunked Prefill | Scheduling | Splits long prompt prefill into chunks | [vLLM optimization docs](https://docs.vllm.ai/en/latest/performance/optimization.html) |
| Continuous Batching | Scheduling | Dynamically batches requests during decoding | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| KV Cache Quantization | Memory | Reduces KV-cache footprint with lower precision | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVTuner | Quantization | Layer-wise mixed precision KV quantization | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVLinC | Quantization | Hadamard rotation + linear correction | [arXiv:2510.05373](https://arxiv.org/abs/2510.05373) |
| TurboQuant | Quantization | Extreme KV/vector compression (Google, 2.5–3.5 bits) | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| Disaggregated Prefill/Decode | Architecture | Separates prefill-heavy and decode-heavy workloads | vLLM / SGLang / TensorRT-LLM docs |
| Speculative Decoding | Decode accel | Draft model predicts, target verifies | Various serving docs |
| KV Offloading | Memory | Moves KV/cache/state to CPU/SSD where needed | FlexGen / InstInfer papers |

### Decision Tree

```text
OOM on long context?
    → PagedAttention (default in vLLM) → vAttention → KV Quantization → KV Offloading

Many requests with same system prompt?
    → Prefix Caching + RadixAttention

Mixed prefill + decode traffic?
    → Disaggregated Prefill/Decode + Continuous Batching

Need 2–4x decode speedup?
    → Speculative Decoding
```

---

## 2.3 Distributed Long-Context Attention

| Method | What It Solves | Source |
|---|---|---|
| RingAttention | Splits sequence attention across devices in a ring | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) |
| DeepSpeed-Ulysses | Long-context training through sequence parallelism | [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) |
| USP | Combines sequence parallel strategies for long context | [arXiv:2405.07719](https://arxiv.org/abs/2405.07719) |

---

## 2.4 VLM / Multimodal Inference

Modern VLMs add attention challenges on top of the text-only stack.

| Method | Bucket | What It Solves | Source |
|---|---|---|---|
| DSA (DeepSeek Sparse Attention) | Sparse serving kernel | Token-level sparse attention for image+text prefill/decode | [DeepSeek-V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| FlashMLA + DSA | Multimodal serving | MLA decoding + FP8 sparse KV for multimodal DeepSeek | [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA) |
| Interleaved-MRoPE | Multimodal PE | Position encoding for interleaved text/image/video tokens | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) (Qwen3-VL) |
| DeepStack (multi-level ViT fusion) | Multimodal fusion | Tighter vision-language alignment via multi-level ViT features | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) (Qwen3-VL) |
| Local-Global Vision Attention | Multimodal PE | Image patches split into local + global streams | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) (Gemma 3) |
| TMRoPE (text-time RoPE) | Multimodal PE | Time-aligned multimodal RoPE for video/audio | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215) (Qwen2.5-Omni) |
| iRoPE | Multimodal PE | Implicit RoPE for vision-language | [Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| HoPE (Hybrid VLM) | Multimodal PE | VLM / long-video position encoding | [arXiv:2505.20444](https://arxiv.org/abs/2505.20444) |
| ModRWKV | Multimodal architecture | Multimodal RWKV for vision-language | [arXiv:2505.14505](https://arxiv.org/abs/2505.14505) |
| Vision Mamba / MambaVision | Multimodal architecture | Mamba-based vision backbone | [arXiv:2401.09417](https://arxiv.org/abs/2401.09417) / [arXiv:2407.08083](https://arxiv.org/abs/2407.08083) |
| Lightning Attention (M1) | Multimodal kernel | Long-context hybrid attention for VLMs | [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) (MiniMax-M1) |
| LazyAttention (RAG) | RAG inference kernel | Deferred PE → position-agnostic KV reuse for long-doc RAG | [arXiv:2606.04302](https://arxiv.org/abs/2606.04302) |

### Decision Tree — Multimodal Serving

```text
Multimodal long-context retrieval (RAG over images/docs)?
    → LazyAttention + RadixAttention

Image+text sparse attention (DeepSeek-V3.2 style)?
    → DSA kernels (FlashMLA repo)

Vision-language position encoding?
    → MRoPE / Interleaved-MRoPE / iRoPE / TMRoPE depending on modality

Multimodal backbone?
    → Local-Global Vision Attention (Gemma 3) or MambaVision (Mamba-based)
```

---

## 2.5 Architecture vs Kernel vs Serving — A Decision Cheat Sheet

| Method | Changes Model Architecture? | Main Layer |
|---|---:|---|
| GQA | Yes | Model attention |
| MLA | Yes | Model attention |
| KDA / Linear Attention | Yes | Model attention / recurrent state |
| Gated DeltaNet | Yes | Model attention / recurrent state |
| Interleaved-MRoPE / iRoPE / TMRoPE | Yes (multimodal) | Multimodal position encoding |
| DeepStack | Yes (multimodal) | Vision-language fusion |
| FlashAttention | No | Kernel |
| SageAttention | No | Quantized kernel |
| FlashMLA + DSA | No | MLA + sparse serving kernel |
| PagedAttention | No | KV-cache memory manager |
| vAttention | No | KV-cache virtual memory manager |
| RadixAttention | No | Prefix/KV-cache reuse |
| LazyAttention | No | RAG serving kernel (position-agnostic KV reuse) |
| RingAttention | System-level | Distributed long-context attention |

---

## 2.6 Production Rule

```text
Architecture methods decide what the model is.
Serving methods decide how cheaply and reliably the model runs in production.
```

For 90% of production deployments, you don't need to invent a new kernel — you need to:

1. **Use vLLM or TGI** (they bundle PagedAttention + Continuous Batching + FlashAttention)
2. **Enable prefix caching** if you have system prompts or RAG
3. **Quantize** the model (INT8 or INT4) if VRAM is tight
4. **Use speculative decoding** if decode latency matters
5. **Move to disaggregated serving** only if prefill is starving decode
6. **For VLMs** — pick a backend (vLLM / SGLang / TensorRT-LLM) that supports your VLM's multimodal RoPE variant (MRoPE / TMRoPE / iRoPE); use FlashMLA + DSA for DeepSeek-V3.2 style sparse multimodal serving
7. **For long-doc RAG** — add LazyAttention (deferred PE) so the KV cache can be reused across retrieved documents with different positions, instead of re-materializing it per query

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Section 2`
- **Code example:** [examples/04_serving/](../examples/04_serving/) (vLLM config)
- **Diagrams:** [assets/diagrams/serving-stack.md](../../assets/diagrams/serving-stack.md)