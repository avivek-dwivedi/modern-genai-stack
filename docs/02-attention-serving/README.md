# Section 2 — Attention & Serving Kernels

> **How do we run this model fast and cheap in production?**

This section is the **performance bridge** between architecture (Section 1) and deployment (Section 10).

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
| SageAttention2++ | Quantized kernel | Extension of SageAttention2 | Watchlist |
| FlashInfer | Inference engine | Optimized attention kernels and runtime pieces | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) |
| FlexAttention | Programmable | Express attention variants, get optimized kernels | [arXiv:2412.05496](https://arxiv.org/abs/2412.05496) |
| FlashMLA | MLA-specific runtime | Optimized serving for MLA-style attention | Watchlist — search official sources |

### Decision Tree

```text
Default attention backend?
    → FlashAttention-2 / FlashAttention-3

H100 / Hopper GPU?
    → FlashAttention-3 (FP8)

Blackwell GPU?
    → SageAttention3 (FP4, watchlist)

Need custom attention pattern?
    → FlexAttention

Need MLA-specific kernel?
    → FlashMLA (watchlist)
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
| TurboQuant | Quantization | Extreme KV/vector compression | Watchlist |
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

## 2.4 Architecture vs Kernel vs Serving — A Decision Cheat Sheet

| Method | Changes Model Architecture? | Main Layer |
|---|---:|---|
| GQA | Yes | Model attention |
| MLA | Yes | Model attention |
| KDA / Linear Attention | Yes | Model attention / recurrent state |
| FlashAttention | No | Kernel |
| SageAttention | No | Quantized kernel |
| PagedAttention | No | KV-cache memory manager |
| vAttention | No | KV-cache virtual memory manager |
| RadixAttention | No | Prefix/KV-cache reuse |
| RingAttention | System-level | Distributed long-context attention |

---

## 2.5 Production Rule

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

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Section 7A`
- **Code example:** [examples/04_serving/](../examples/04_serving/) (vLLM config, KServe manifest)
- **Diagrams:** [assets/diagrams/serving-stack.md](../../assets/diagrams/serving-stack.md)