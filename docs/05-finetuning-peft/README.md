# Section 5 — Fine-Tuning & PEFT

> **How do we adapt a base model to our specific task?**

---

## Why This Section Exists

Base models know general knowledge. Fine-tuning adapts them to:

```text
Domain style
Instruction following
Specific tasks
Reasoning traces
Tool-calling format
Safety behavior
Enterprise data style
Coding conventions
VLM instruction behavior
```

Choosing the right fine-tuning method determines cost, quality, and how easily you can iterate.

---

## 5.1 Method Evolution

```text
Full Fine-Tuning
↓
Adapters
↓
Prompt Tuning
↓
Prefix Tuning
↓
LoRA
↓
QLoRA
↓
DoRA
↓
AdaLoRA / modern PEFT variants
↓
RLHF / Preference Optimization
```

---

## 5.2 Fine-Tuning Categories

| Category | What changes | Cost |
|---|---|---|
| Full fine-tuning | All model weights | Highest |
| SFT | Usually all weights or adapters on instruction data | Medium to high |
| PEFT | Small trainable modules | Low |
| Preference optimization | Aligns preferred outputs | Medium |
| Continued pretraining | More tokens on domain corpus | Very high |
| VLM fine-tuning | Vision bridge / projector / LLM / vision encoder | Varies |

---

## 5.3 PEFT Methods Comparison

| Method | How it works | Best for | Limitations |
|---|---|---|---|
| Adapter tuning | Inserts small trainable layers | Modular task adaptation | Adds inference path |
| Prompt tuning | Learns soft prompt embeddings | Very low parameter tuning | Lower capacity |
| Prefix tuning | Learns prefix key/value vectors | Generation control | Can be weaker than LoRA |
| **LoRA** | Adds low-rank matrices to attention/MLP weights | Most common PEFT | Rank choice matters |
| **QLoRA** | Quantizes base model to 4-bit and trains LoRA | Large model on limited GPU | Slower due to quant/dequant |
| **DoRA** | Decomposes weight into magnitude + direction, applies LoRA to direction | Better LoRA quality | More complex/newer |
| AdaLoRA | Dynamically allocates rank budget | Efficient rank allocation | More moving parts |
| IA3 | Learns scaling vectors | Very parameter-efficient | Less flexible than LoRA |

### LoRA — the canonical PEFT method

```text
Base weight W is frozen.
Trainable low-rank update ΔW = A × B is added.
Only A and B are trained.
```

### QLoRA — LoRA on a quantized base

```text
Base model is loaded in 4-bit (NF4).
LoRA adapters are trained on top.
Allows 65B models on a single 48GB GPU.
```

### DoRA — better LoRA via decomposition

```text
Weight = magnitude × direction
DoRA keeps magnitude trainable
and applies LoRA-style low-rank update to direction only.
```

---

## 5.4 Decision Tree — Which Method Should I Use?

```text
Need domain knowledge from huge corpus?
    → Continued pretraining

Need instruction-following for task format?
    → SFT

Limited GPU?
    → LoRA or QLoRA

Very limited GPU?
    → QLoRA + Unsloth

Need best accuracy and have GPUs?
    → Full fine-tuning / FSDP / DeepSpeed

Need preference alignment?
    → DPO / ORPO / SimPO / GRPO  (see Section 6)

Need reasoning improvement?
    → SFT with reasoning traces + GRPO / rejection sampling / distillation

Need VLM adaptation?
    → Projector tuning → multimodal SFT → optional multimodal preference tuning
```

### Practical Rule

```text
Start with SFT + LoRA/QLoRA.
Use DPO/ORPO/SimPO only after you have preference pairs.
Use GRPO if you have verifiable rewards or reasoning tasks.
Use full fine-tuning only when PEFT is not enough and budget exists.
```

---

## 5.5 Capacity Planning

### Memory Intuition

```text
FP32 = 4 bytes per parameter
FP16/BF16 = 2 bytes per parameter
INT8 = 1 byte per parameter
INT4/NF4 = 0.5 byte per parameter
```

But training needs more than weights:

```text
weights
+ gradients
+ optimizer states
+ activations
+ LoRA/adapters
```

### Capacity Table

| GPU VRAM | Practical LoRA | Practical QLoRA | Notes |
|---:|---|---|---|
| 12GB | 1B–3B comfortable; 7B possible with aggressive settings | 3B–7B possible | Good for demos, small labs |
| 16GB | 7B LoRA possible | 7B–13B QLoRA possible | Good low-cost training |
| 24GB | 7B LoRA comfortable | 13B QLoRA possible | RTX 4090 class |
| 48GB | 13B–34B LoRA depending seq length | 34B QLoRA possible | A6000 / L40S class |
| 80GB | 34B LoRA; 70B QLoRA possible | 70B QLoRA | A100/H100 class |
| Multi-GPU 80GB | 70B+ with FSDP/ZeRO | 100B+ possible with sharding | Production/advanced training |

### Things That Increase Memory

```text
Long sequence length
Large batch size
Full fine-tuning
Vision tokens in VLM
RLHF with multiple models
PPO with policy + reference + reward + value model
High-resolution images
Video frames
```

### Things That Reduce Memory

```text
QLoRA
Gradient checkpointing
FlashAttention
FSDP / ZeRO
Gradient accumulation
Lower sequence length
Smaller batch size
Freeze vision encoder
Projector-only tuning
```

---

## 5.6 Frameworks

| Framework | Best for | Not best for |
|---|---|---|
| Hugging Face Transformers | General training and model loading | Large distributed RLHF alone |
| PEFT | LoRA/QLoRA/adapters | Full RLHF pipeline |
| TRL | PPO, DPO, GRPO, reward modeling | Very large infra orchestration alone |
| Unsloth | Fast single-GPU LoRA/QLoRA/GRPO | Open multi-GPU at massive scale |
| Axolotl | YAML SFT/DPO/PPO/GRPO pipelines | Beginners who dislike config complexity |
| LLaMA-Factory | UI-friendly fine-tuning | Advanced distributed RLHF |
| DeepSpeed | ZeRO, large distributed training | Simple local tuning |
| FSDP | PyTorch-native model sharding | Beginners |
| Megatron-LM | Very large-scale pretraining/finetuning | Small experiments |
| NeMo | NVIDIA enterprise training stack | Lightweight hobby training |
| Ray Train | Distributed orchestration | Model-specific tuning alone |
| OpenRLHF | RLHF training stack | Simple LoRA only |
| verl | RLHF/GRPO style training | Simple beginner LoRA |
| SkyPilot | Cloud orchestration | Training algorithm itself |

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Section 8 / Section 9 / Section 17`
- **Code example:** [examples/03_finetune/](../examples/03_finetune/) (LoRA + QLoRA scripts)
- **Diagrams:** [assets/diagrams/training-pipeline.md](../../assets/diagrams/training-pipeline.md)