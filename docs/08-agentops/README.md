# Section 8 — AgentOps

> **How do we build agents that can plan, call tools, use memory, and recover from errors?**

---

## Why This Section Exists

A simple chatbot:

```text
User → LLM → answer
```

An agent:

```text
User → planner → tool call → observation → reasoning → next tool → final answer
```

Agents can **reason, plan, call tools, use memory, observe results, continue steps, and return final answers**. But they're also fragile, expensive, and dangerous if not operated carefully.

AgentOps is the discipline of running agents safely in production.

---

## 8.1 Agent vs Workflow

| System | Meaning | Use when |
|---|---|---|
| Workflow | Fixed sequence | Process is predictable |
| Agent | LLM decides next step | Task is open-ended |
| Multi-agent | Multiple specialized agents | Task needs role separation |
| Graph agent | State machine controls flow | Need reliability and control |

---

## 8.2 Agentic Design Patterns

| Pattern | Meaning |
|---|---|
| ReAct | Reason + Act + Observe |
| Planner-executor | One component plans, another executes |
| Router | Routes task to specialist chain/agent |
| Reflection | Model critiques and revises |
| Tool-use agent | Calls APIs/functions |
| Human-in-loop | Human approval before risky action |
| Supervisor-worker | Supervisor coordinates agents |
| State machine agent | Graph controls allowed transitions |

### LangGraph-style Control

```text
State
→ Node
→ Conditional edge
→ Tool / Agent
→ Update state
→ Continue or stop
```

Graphs make agent behavior **inspectable, debuggable and recoverable**.

---

## 8.3 Tool Calling Requirements

```text
tool schema validation
input sanitization
timeout
retry
permission check
audit log
human approval for high-risk tools
```

---

## 8.4 Agent Memory

| Memory type | Meaning |
|---|---|
| Short-term memory | Current session state |
| Long-term memory | Persisted facts/preferences |
| Episodic memory | Past interactions |
| Semantic memory | User/business knowledge |
| Tool memory | Previous tool outputs |
| Vector memory | Retrieved embedded memory |
| Structured memory | SQL/JSON profile/entity store |

### Memory Risks

```text
wrong memory
outdated memory
cross-user leakage
private data stored without consent
prompt injection stored as memory
malicious user teaches bad facts
```

### Safe Memory Design

```text
memory write should be explicit
sensitive memory should be blocked
memory should have source + timestamp
memory should be user-scoped
memory should be editable/deletable
retrieval should be filtered
```

---

## 8.5 Multi-Agent Architecture

```text
User request
→ Supervisor agent
→ Research agent
→ Code agent
→ Critic agent
→ Safety agent
→ Final response
```

**Use multi-agent only when the task needs role separation. Do not use it for simple Q&A.**

---

## 8.6 AgentOps Monitoring

Track:

```text
agent steps
tool calls
tool errors
latency per tool
cost per step
loop count
state transitions
memory reads/writes
failure reason
user feedback
```

---

## 8.7 Decision Tree

```text
Simple Q&A or fixed sequence?
    → Plain chain / workflow

Need a model to choose next step?
    → ReAct agent

Need role separation?
    → Multi-agent (supervisor + workers)

Need reliability / audit trail?
    → State machine / LangGraph-style graph

Need user-specific memory?
    → Add memory layer (with safety controls)

Need tool access?
    → Tool-use agent (with sandbox + permissions)
```

---

## 8.8 Production Rule

```text
Agent systems fail in three ways:
  1. Loop forever (no max steps)
  2. Call unsafe tools (no permission check)
  3. Hallucinate tool outputs (no schema validation)

Production-grade agents must enforce:
  - max steps + timeout
  - tool allow-list + permission check
  - schema validation + sandbox
  - audit log + user feedback
  - rollback to last good state
```

---

## 📎 Related Resources

- **Notebook section:** Master notebook `Part E`
- **Code example:** [examples/02_agent/](../examples/02_agent/) (LangGraph multi-agent supervisor)
- **Diagrams:** [assets/diagrams/agent-architecture.md](../../assets/diagrams/agent-architecture.md)