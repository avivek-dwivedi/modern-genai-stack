# Section 4 — Coding Models

> **Why are coding models better at coding?**

---

## Why This Section Exists

A coding model is usually improved through:

```text
1. More code-heavy pretraining
2. Repository-level data
3. Multi-language code data
4. Code-specific instruction tuning
5. Infilling / fill-in-the-middle training
6. Longer context
7. Tool-use / agentic coding data
8. Compiler/test feedback
9. Bug fixing and code repair datasets
```

**Most coding improvements come from data + tokenizer + context length + instruction tuning + tool-use training + evaluation loop — NOT from a radically different transformer architecture.**

---

## 4.1 Why Coding Models Differ

| Factor | Why it matters |
|---|---|
| Code-heavy tokens | Model sees syntax, libraries, APIs, patterns |
| Multi-file repo data | Learns imports, project structure, dependencies |
| Fill-in-the-middle | Better IDE-style completion |
| Long context | Can read more of a codebase |
| Test/repair data | Better debugging and patching |
| Agentic data | Better tool use, edit-run-debug loops |
| Code tokenizer | Better compression of symbols/indentation/API names |

---

## 4.2 Model Timeline

```text
CodeGen
↓
StarCoder
↓
Code Llama
↓
DeepSeek Coder
↓
Qwen Coder
↓
Codestral / Devstral
↓
Kimi K2 / agentic coding models
↓
Tool-use coding agents
```

## 4.3 Key Models

| Model | Why important | Source |
|---|---|---|
| Code Llama | Llama 2 continued on 500B code tokens; code completion/instruction variants | [Hugging Face blog](https://huggingface.co/blog/codellama) |
| StarCoder | 15B model trained on permissively licensed GitHub code, 80+ languages | [Hugging Face blog](https://huggingface.co/blog/starcoder) |
| DeepSeek Coder | Strong open coding family, code/math reasoning direction | [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) |
| Qwen2.5-Coder | Strong open-source coding model, Qwen family | [Alibaba Cloud blog](https://www.alibabacloud.com/blog/qwen2-5-coder-series-powerful-diverse-practical_601765) |
| Codestral | Mistral coding model family | [Mistral Codestral](https://mistral.ai/news/codestral) |
| Devstral | Agentic coding / software engineering direction | [Mistral Devstral](https://mistral.ai/news/devstral) |
| Kimi K2 | Agentic / tool-use coding direction | Moonshot / Kimi official |

---

## 4.4 Evaluation Benchmarks

| Benchmark | Measures |
|---|---|
| HumanEval | Function-level code generation |
| MBPP | Basic Python programming |
| EvalPlus | More rigorous HumanEval/MBPP tests |
| LiveCodeBench | Recent coding tasks |
| SWE-bench | Real GitHub issue resolution |
| Aider benchmark | Code repair / agentic coding workflow |
| BigCodeBench | Practical code generation |
| RepoBench | Repository-level understanding |

### Core Takeaway

```text
HumanEval checks function generation.
SWE-bench checks software engineering.
A model can be good at HumanEval but weak at repo-level debugging.
```

---

## 4.5 Decision Tree — Which Coding Model?

```text
Need a coding assistant inside an IDE?
    → Qwen2.5-Coder / Codestral / DeepSeek Coder

Need agentic coding (read repo, edit, run tests)?
    → Devstral / Kimi K2 / DeepSeek Coder + tools

Need local / offline coding model?
    → Qwen2.5-Coder (1.5B / 7B)

Need best function-level completion?
    → Codestral / Code Llama

Best open reasoning + coding?
    → DeepSeek Coder / Qwen2.5-Coder-32B
```

---

## 4.6 Production Rule

```text
For agentic coding, the model is only 30% of the system.
The other 70% is:
  - tool framework (file read, grep, shell, tests)
  - sandbox (no accidental file deletion)
  - retry + repair loop
  - evaluation harness (does the patch pass tests?)
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Section 6 / Section 7`
- **Code example:** [examples/02_agent/](../examples/02_agent/) (coding agent with tool use)
- **Diagrams:** none — this section is more about data than architecture