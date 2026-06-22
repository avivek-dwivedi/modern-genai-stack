# Section 6 — Alignment & RLHF

> **How do we make the model behave the way we want?**

---

## Why This Section Exists

A base model is a next-token predictor. It doesn't naturally:

- Refuse harmful requests
- Follow a specific output format
- Prefer concise over verbose answers
- Reason step-by-step
- Align with enterprise tone/style

Alignment techniques add these behaviors — and modern post-training has evolved rapidly from PPO-style RLHF to simpler, reference-free methods.

---

## 6.1 Method Evolution

```text
SFT
↓
Reward Model
↓
PPO-based RLHF
↓
RLAIF
↓
DPO
↓
IPO
↓
KTO
↓
ORPO
↓
SimPO
↓
GRPO
```

---

## 6.2 Method Comparison

| Method | Needs reward model? | Needs reference model? | Online RL? | Best for | Weakness |
|---|---:|---:|---:|---|---|
| PPO RLHF | Yes | Usually yes | Yes | Strong alignment with reward model | Complex, unstable, expensive |
| DPO | No explicit RM | Yes | No | Offline preference pairs | Sensitive to preference data quality |
| IPO | No explicit RM | Yes | No | Alternative preference optimization | Less common |
| KTO | No explicit RM | Usually no/varies | No | Binary desirable/undesirable feedback | Newer, less standard |
| ORPO | No | No | No | Simple SFT + preference in one stage | Objective tuning needed |
| SimPO | No | No | No | Reference-free preference training | Newer; needs validation |
| GRPO | Reward/verifier useful | No critic | Yes-ish / online sampling | Reasoning/verifiable tasks | More complex than DPO/ORPO |
| RLAIF | AI feedback instead of human | Varies | Varies | Scalable preference labels | Can amplify model bias |

---

## 6.3 The Core Idea Behind Each Method

### Traditional RLHF (PPO)

```text
1. SFT model on demonstrations
2. Collect human preference pairs
3. Train reward model
4. Use PPO to optimize policy model against reward model
```

**Pros:** Powerful alignment method, can optimize behavior beyond supervised data
**Cons:** Complex, expensive, needs reward model, needs sampling loop, tuning can be unstable

### DPO

```text
Use preference pairs directly.
Avoid explicit reward model and PPO loop.
```

### ORPO

```text
Combine SFT and preference alignment into one objective.
No separate reference model.
```

### GRPO

```text
Generate multiple responses for the same prompt.
Compare group rewards.
Avoid separate value critic used in PPO.
```

### SimPO

```text
Use length-normalized log probability as an implicit reward.
Reference-free preference optimization.
```

---

## 6.4 Reasoning Model Training

### Common Training Stages

```text
Base LLM
↓
SFT on instruction data
↓
SFT on chain-of-thought / reasoning traces
↓
Preference optimization
↓
RL with verifiable rewards
↓
Distillation into smaller models
```

### Reward Types

| Reward type | Meaning | Use case |
|---|---|---|
| Outcome reward | Final answer is correct/incorrect | Math, code tests |
| Process reward | Intermediate reasoning steps are scored | Step-by-step reasoning |
| Rule-based reward | Programmatic verifier | Coding, math, structured tasks |
| Human preference reward | Human ranks outputs | Chat/helpfulness |
| AI feedback reward | Strong model ranks outputs | RLAIF |

---

## 6.5 Decision Tree

```text
Need basic tone/style alignment?
    → SFT + small DPO dataset

Need verifiable-task alignment (math/code)?
    → GRPO + rule-based verifier

Need a single unified stage?
    → ORPO

Need reference-free preference training?
    → SimPO

Already have a strong reward model?
    → PPO

Scaling preference labels without humans?
    → RLAIF (use a stronger model as judge)
```

### Practical Rule

```text
For enterprise chatbot: DPO/ORPO may be enough.
For math/code reasoning: GRPO/verifier-based RL becomes more relevant.
```

---

## 6.6 Source Links

```text
DPO: https://arxiv.org/abs/2305.18290
ORPO: https://arxiv.org/abs/2403.07691
KTO: https://arxiv.org/abs/2402.01306
GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
SimPO: https://arxiv.org/abs/2405.14734
TRL: https://github.com/huggingface/trl
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Section 11 / Section 12 / Section 13`
- **Code example:** [examples/03_finetune/dpo_example.py](../examples/03_finetune/) (DPO training script)
- **Diagrams:** [assets/diagrams/rlhf-pipeline.md](../../assets/diagrams/rlhf-pipeline.md)