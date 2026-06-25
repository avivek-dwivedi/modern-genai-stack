# Section 1 — LLM Architecture

> **What is this model? What changed inside the transformer?**

This section is the **conceptual foundation**. Before you can fine-tune, serve, or secure a model, you need to understand what's happening at the block level.

---

## Why This Section Exists

Modern LLMs are not "just Transformers" anymore. Since 2023, the architecture has fragmented into many variants:

- **Positional encoding** has gone from RoPE → multimodal RoPE → periodic RoPE
- **Attention** has gone from MHA → GQA → MLA → sparse → linear/KDA
- **FFN** has gone from dense → MoE → high-sparsity MoE → PEER million-expert
- **Beyond-Transformer** has become a real alternative (Mamba, RWKV, RetNet, Titans)

If you don't know these, you can't read a 2025–2026 model report.

---

## 1.1 Positional Encoding

**The problem:** Transformers are permutation-invariant — without position info, "dog bites man" and "man bites dog" look the same.

### Evolution

```text
Sinusoidal PE
→ Learned Absolute PE
→ Relative Position Bias
→ ALiBi
→ RoPE
→ RoPE base/theta scaling
→ NTK-aware RoPE
→ Dynamic RoPE
→ Position Interpolation
→ YaRN
→ LongRoPE
→ LongRoPE2
→ MRoPE
→ TMRoPE
→ Interleaved-MRoPE
→ MHRoPE / MRoPE-I
→ iRoPE (Llama 4 implicit RoPE)
→ NoPE
→ P-RoPE / Periodic RoPE
→ HoPE variants (Hyperbolic / Hybrid VLM)
→ LazyAttention (PE + KV reuse bridge for RAG)
```

### Cheat Sheet

| Method | Study Why | Maturity | Used In | Source |
|---|---|---|---|---|
| Sinusoidal PE | Original Transformer baseline | main | Early Transformer | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| RoPE | Modern decoder LLM default | main | LLaMA, Qwen, DeepSeek | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| YaRN | Efficient RoPE context extension | main | Long-context LLaMA / Qwen configs | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| LongRoPE | Non-uniform RoPE interpolation | main | Long-context research | [arXiv:2402.13753](https://arxiv.org/abs/2402.13753) |
| LongRoPE2 | Continuation of LongRoPE context scaling | emerging | Research | [arXiv:2502.20082](https://arxiv.org/abs/2502.20082) |
| Periodic RoPE / P-RoPE | Infinite-context PE direction | emerging | Research | [arXiv:2605.27980](https://arxiv.org/abs/2605.27980) |
| MRoPE | Multimodal spatial/temporal PE | main | Qwen2.5-VL family | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| TMRoPE | Time-aligned multimodal RoPE | main | Qwen2.5-Omni | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215) |
| Interleaved-MRoPE | Interleaved text-image-video PE | main | Qwen3-VL | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |
| MHRoPE | Multi-head frequency allocation for multimodal | emerging | Research | [arXiv:2510.23095](https://arxiv.org/abs/2510.23095) |
| MRoPE-I / MRoPE-Interleave | Interleaved multimodal RoPE variant | emerging | Research | [arXiv:2510.23095](https://arxiv.org/abs/2510.23095) |
| NoPE | Transformers without explicit PE | emerging | Research direction | [arXiv:2505.11199](https://arxiv.org/abs/2505.11199) |
| iRoPE | Implicit RoPE for vision-language | emerging | Llama 4 | [Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| HoPE (Hyperbolic RoPE) | Hyperbolic geometry PE direction | emerging | Research | [arXiv:2509.05218](https://arxiv.org/abs/2509.05218) |
| HoPE (Hybrid of Position Embedding) | VLM / long-video PE | emerging | Research | [arXiv:2505.20444](https://arxiv.org/abs/2505.20444) |
| LazyAttention | Deferred PE for position-agnostic KV reuse (RAG) | emerging | Research (ICML 2026) | [arXiv:2606.04302](https://arxiv.org/abs/2606.04302) |

### Core Takeaway

```text
RoPE gives position through rotation.
YaRN and LongRoPE stretch RoPE for long context.
MRoPE/TMRoPE extend position from text into image, video, and audio time.
```

---

## 1.2 Attention Mechanism

**The problem:** Vanilla self-attention has O(n²) cost and huge KV-cache. Real systems need to reduce both.

### Evolution

```text
Self-Attention
→ Multi-Head Attention (MHA)
→ Multi-Query Attention (MQA)
→ Grouped-Query Attention (GQA)
→ Local / Sliding Window Attention
→ Local-Global Attention
→ Multi-Head Latent Attention (MLA)
→ Sparse Attention
→ Native Sparse Attention (NSA)
→ Lightning Attention
→ Linear Attention / KDA
→ Multimodal Self-Attention / Cross-Attention
```

### Cheat Sheet

| Method | Problem Solved | Maturity | Used In | Source |
|---|---|---|---|---|
| MHA | Full attention baseline | main | Original Transformer / GPT-style | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| MQA | Reduces KV cache by sharing K/V | main | PaLM-era / efficient decoders | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| GQA | Balance MHA quality + MQA speed | main | LLaMA 3, Qwen, Gemma line | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| Local Attention | Reduces cost over long context | main | Gemma 3 local/global design | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| MLA | Compresses KV cache into latent representation | main | DeepSeek-V3 / DeepSeek line | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Sparse Attention | Skips irrelevant tokens/blocks | main | DeepSeek sparse direction | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| NSA | Sparse long-context attention w/ token compression + selection | main | Research | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| DSA (DeepSeek Sparse Attention) | Token-level sparse attention for prefill + decode | main | DeepSeek-V3.2-Exp | [DeepSeek-V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| Lightning Attention | Efficient long-context hybrid attention | main | MiniMax-01 / MiniMax-M1 | [arXiv:2501.08313](https://arxiv.org/abs/2501.08313), [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) |
| KDA / Kimi Delta Attention | Linear-attention fixed-state memory | main | Kimi Linear | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |
| Gated DeltaNet | Linear attention with data-controlled gating | main | Qwen3-Next / Kimi-style | [arXiv:2412.06410](https://arxiv.org/abs/2412.06410) |
| Multimodal Self-Attention | Text/image/video tokens reason together | main | Qwen3-VL | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |
| Vision Local-Global Attention | Image patches split into local + global streams | main | Gemma 3 multimodal | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| DeepStack (multi-level ViT fusion) | Tighter vision-language alignment | main | Qwen3-VL | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |

### Core Takeaway

```text
GQA reduces KV heads.
MLA compresses KV cache.
Sparse attention skips tokens.
Linear/KDA replaces growing KV cache with fixed recurrent state.
```

---

## 1.3 FFN / MoE

**The problem:** Scaling dense FFNs gets expensive. MoE decouples **total params** from **active params per token**.

### Evolution

```text
Dense FFN
→ GELU FFN
→ Gated FFN
→ SwiGLU
→ Sparse MoE
→ Top-K Routing
→ Shared Experts
→ Fine-Grained Experts
→ Aux-loss-free Load Balancing
→ High-Sparsity MoE
→ Adaptive Routing
→ SeqTopK
→ Expert-Choice Routing (reversed routing)
→ SoftMoE (differentiable soft assignment)
→ Sub-MoE / Expert Merging
→ REAM (Router-weighted Expert Activation Merging)
→ PEER (Million tiny experts)
→ Memory Layers / MoD
```

### Cheat Sheet

| Method | Core Idea | Maturity | Used In | Source |
|---|---|---|---|---|
| Dense FFN | Same MLP for every token | main | GPT/LLaMA/Gemma dense | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| SwiGLU | Gated FFN activation | main | LLaMA-style, Kimi K2 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) |
| Sparse MoE | Many FFNs, activate few experts | main | Mixtral, DeepSeek, Qwen MoE, Llama 4 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| DeepSeekMoE | Shared experts + fine-grained experts | main | DeepSeek-V2/V3/R1 | [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) |
| Aux-loss-free Load Balancing | Balances experts without strong aux loss | main | DeepSeek-V3 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| High-Sparsity MoE | Huge total params, low active params | main | Qwen3 MoE / Qwen3-Next | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| Adaptive Expert Routing | Different tokens get different expert budgets | emerging | Research | [arXiv:2410.10456](https://arxiv.org/abs/2410.10456) |
| SeqTopK | Sequence-level expert budget allocation | emerging | Research | [arXiv:2511.06494](https://arxiv.org/abs/2511.06494) |
| SoftMoE | Differentiable soft expert assignment (weighted token combos) | emerging | Research (ICLR 2024) | [arXiv:2308.00951](https://arxiv.org/abs/2308.00951) |
| Expert-Choice Routing | Experts choose top-k tokens (reversed routing) | emerging | Research | [arXiv:2202.09368](https://arxiv.org/abs/2202.09368) |
| Sub-MoE / Expert Merging | Merge experts to reduce MoE deployment cost | emerging | Research | [arXiv:2506.23266](https://arxiv.org/abs/2506.23266) |
| REAM | Router-weighted Expert Activation Merging (preserves quality) | emerging | Research | [arXiv:2604.04356](https://arxiv.org/abs/2604.04356) |
| MoE++ | Cheap zero/copy/constant experts | emerging | Research | [arXiv:2410.07348](https://arxiv.org/abs/2410.07348) |
| PEER | Many tiny retrieved experts | emerging | Research | [arXiv:2407.04153](https://arxiv.org/abs/2407.04153) |
| Mixture-of-Depths | Route tokens through different compute depth | emerging | Research | [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) |

### Core Takeaway

```text
Dense FFN gives every token the same network.
MoE gives each token a selected set of expert networks.
Modern MoE focuses on routing, balancing, and reducing active compute.
```

---

## 1.4 Beyond Transformer

**The question:** Can we replace the transformer entirely with linear-time or recurrent architectures?

### Cheat Sheet

| Architecture | Core Idea | Maturity | Used In | Source |
|---|---|---|---|---|
| Mamba | Selective state-space sequence model | main | SSM research direction | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Mamba-2 | Connects SSM and attention via State Space Duality | main | SSD research | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| Jamba | Hybrid Transformer + Mamba + MoE | main | AI21 hybrid | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| Nemotron-H | Hybrid Mamba-Transformer LLM | main | NVIDIA efficient inference | [arXiv:2504.03624](https://arxiv.org/abs/2504.03624) |
| RWKV-7 | RNN-like LLM with constant memory/time per token | main | RWKV community | [arXiv:2503.14456](https://arxiv.org/abs/2503.14456) |
| RetNet | Retention: parallel training + recurrent inference | main | Microsoft research | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |
| Kimi Linear | KDA + MLA hybrid efficient attention | main | Moonshot | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |
| Titans | Neural long-term memory | main | Google research | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| Gated DeltaNet | Linear/recurrent attention with data-controlled gating | main | Qwen3-Next / Kimi Linear | [arXiv:2412.06410](https://arxiv.org/abs/2412.06410) |
| DeltaNet variants | Linear attention family (Mamba-2 SSD / Gated DeltaNet) | main | Research / Qwen3-Next | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| ATLAS / DeepTransformers | Test-time memory / deep memory | emerging | Research | [arXiv:2505.23735](https://arxiv.org/abs/2505.23735) |
| ModRWKV | Multimodal RWKV for vision-language | emerging | Research | [arXiv:2505.14505](https://arxiv.org/abs/2505.14505) |
| VMamba | Vision SSM (2D selective scan) | main | Vision research | [arXiv:2401.10166](https://arxiv.org/abs/2401.10166) |
| Vision Mamba / Vim | Mamba adapted for visual patches | main | Vision research | [arXiv:2401.09417](https://arxiv.org/abs/2401.09417) |
| MambaVision | Hybrid Mamba-Transformer vision backbone | main | Vision research | [arXiv:2407.08083](https://arxiv.org/abs/2407.08083) |
| xLSTM | Modern recurrent LLM (LSTM scaled back) | emerging | Research | [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) |

### Core Takeaway

```text
Transformer remembers using KV cache.
Mamba / RWKV / RetNet / KDA remember using recurrent or fixed-size state.
Titans adds trainable long-term memory.
```

---

## 1.5 Multimodal Architecture

**The question:** How do we go from text-only LLMs to models that see, hear, and speak?

### Architecture Patterns

```text
1. Dual Encoder (CLIP, SigLIP)
2. Vision Encoder + Linear Projector + LLM (LLaVA, MiniGPT-4)
3. Q-Former / Learned Query Bridge (BLIP-2, InstructBLIP)
4. Perceiver Resampler (Flamingo)
5. Unified / Interleaved Multimodal Token Architecture (Qwen-VL, Qwen3-VL, Qwen-Omni)
```

| Model / Family | Pattern | What to Study | Source |
|---|---|---|---|
| CLIP | Dual encoder | Contrastive image-text embedding | OpenAI paper |
| OpenCLIP | Dual encoder | Open reproduction / scaling | OpenCLIP repo |
| SigLIP | Dual encoder | Sigmoid loss for image-text pretraining | SigLIP paper |
| BLIP | VL pretraining | Captioning + filtering + bootstrapping | Salesforce BLIP |
| BLIP-2 | Vision + Q-Former + LLM | Frozen vision/LLM, trainable bridge | [Salesforce BLIP-2](https://arxiv.org/abs/2301.12597) |
| LLaVA | CLIP + projector + LLaMA | Two-stage visual instruction tuning | [LLaVA project](https://llava-vl.github.io/) |
| InternVL | Large vision encoder + LLM bridge | Scaled vision foundation | [arXiv:2312.14238](https://arxiv.org/abs/2312.14238) |
| Qwen2.5-VL | Modern VLM | Dynamic resolution, window attention, absolute time | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| Qwen2.5-Omni | Omni model | TMRoPE, Thinker-Talker | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215) |
| Qwen3-VL | Modern VLM | Interleaved-MRoPE, DeepStack, dense + MoE VLM | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) |
| Gemma 3 multimodal | Open multimodal | Local/global attention | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| Llama 4 (multimodal) | Open multimodal | iRoPE, vision + text + speech | [Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| DeepSeek-V3.2 multimodal | Sparse VLM | Token-level sparse attention (DSA) | [DeepSeek-V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| MiniCPM-V | Efficient VLM | Small-device / efficient VLM | OpenBMB |
| GPT-4o / Gemini | Closed omni | Behavior and product direction only | Limited disclosure |

### Core Takeaway

```text
Multimodal models are not just text models with images.
They must solve token explosion, spatial position, temporal alignment, and cross-modal fusion.
```

---

## 1.6 Closed-Source Tracking Notes

Closed models are useful for **capability and product direction** but usually weak for exact architecture.

| Model Family | What to Track | Architecture Disclosure |
|---|---|---|
| OpenAI GPT-5 / GPT-5.x | System card, model behavior, routing/reasoning | Low |
| Claude 4 / Opus / Sonnet | System card, tool use, extended thinking, agent behavior | Low |
| Gemini 2.5 / Gemini 3 | Model cards, multimodal reasoning, long context | Low |
| Grok 4 / 4.x | Product behavior, tool use, reasoning benchmarks | Low |

**Rule:**

```text
Use closed-source docs for system behavior.
Use open/open-weight reports for real architecture study.
```

---

## 1.7 Decision Tree — Which Architecture Should I Care About?

```text
Just want to use a model?
    → Read "Cheat Sheet" rows marked main, focus on GQA + SwiGLU + RoPE

Training your own LLM?
    → Add MLA + DeepSeekMoE + auxiliary-loss-free balancing

Long context (1M+ tokens)?
    → Add YaRN / LongRoPE / Lightning Attention / Linear Attention (KDA)

Multimodal?
    → Qwen2.5-VL / Qwen3-VL architecture patterns + MRoPE / TMRoPE

Pushing inference cost down?
    → Read Section 2 (Attention & Serving Kernels)

Curious about what's next?
    → Emerging items (NoPE, P-RoPE, HoPE, xLSTM, Gated DeltaNet, iRoPE, SoftMoE, DSA)
```

---

## 📎 Related Resources

- **Notebook section:** see the master notebook's `Part A → Part C2`
- **Code examples:** none for this section — architecture is conceptual
- **Diagrams:** see [assets/diagrams/architecture-*.md](../../assets/diagrams/)
- **Paper index:** see [references/papers.md](../../references/papers.md)