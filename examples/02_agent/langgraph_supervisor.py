"""
examples/02_agent/langgraph_supervisor.py

Multi-agent supervisor pattern using LangGraph.
This is the canonical "graph agent" architecture referenced in
docs/08-agentops/README.md "LangGraph-style Control".

What this example demonstrates:
- State machine control of agent flow
- Multi-agent supervisor + worker pattern
- Tool calling with permission check + audit log
- Max steps enforcement (prevents infinite loops)
- Memory writes that are user-scoped and timestamped

NOTE: This is a reference. Install langgraph + langchain to run:
    pip install langgraph langchain langchain-openai
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator
import time


# ---------- State ----------

class AgentState(TypedDict):
    """State passed between agent nodes."""
    user_id: str
    query: str
    plan: list[str]
    step: int
    max_steps: int
    results: Annotated[list[str], operator.add]
    audit_log: Annotated[list[dict], operator.add]
    final_answer: str


# ---------- Tools ----------

def web_search(query: str) -> str:
    """Mock web search tool."""
    # In production: call SerpAPI, Tavily, or your search backend
    return f"[Mock search result for: {query}]"


def read_document(doc_id: str) -> str:
    """Mock document reader."""
    return f"[Mock document content for: {doc_id}]"


def calculator(expression: str) -> str:
    """Mock calculator tool."""
    try:
        return str(eval(expression))  # NEVER do this in production
    except Exception:
        return "error"


TOOL_REGISTRY = {
    "web_search": web_search,
    "read_document": read_document,
    "calculator": calculator,
}

# Tool allow-list — only these tools can be called
ALLOWED_TOOLS = {"web_search", "read_document", "calculator"}


# ---------- Supervisor node ----------

def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor: decides what the next step should be.
    In production, this is an LLM call that returns structured output.
    """
    step = state["step"]
    max_steps = state["max_steps"]

    # Enforce max steps (prevents infinite loops)
    if step >= max_steps:
        return {
            **state,
            "plan": ["finish"],
            "audit_log": [{"event": "max_steps_reached", "step": step}],
        }

    # Placeholder: simple planning logic
    if step == 0:
        plan = ["research", "answer"]
    else:
        plan = ["finish"]

    return {**state, "plan": plan, "step": step + 1}


# ---------- Worker node ----------

def worker_node(state: AgentState) -> AgentState:
    """
    Worker: executes the next action from the plan.
    """
    plan = state.get("plan", [])
    if not plan:
        return state

    next_action = plan[0]
    remaining_plan = plan[1:]

    audit_entry = {
        "event": "tool_call",
        "tool": next_action,
        "step": state["step"],
        "timestamp": time.time(),
        "user_id": state["user_id"],
    }

    # Tool call with permission check
    if next_action not in ALLOWED_TOOLS:
        result = f"TOOL NOT ALLOWED: {next_action}"
        audit_entry["result"] = "denied"
    else:
        # Mock tool call (in production, parse LLM output for tool name + args)
        tool_fn = TOOL_REGISTRY.get(next_action)
        try:
            result = tool_fn(state["query"])
            audit_entry["result"] = "success"
        except Exception as e:
            result = f"TOOL ERROR: {e}"
            audit_entry["result"] = "error"

    return {
        **state,
        "plan": remaining_plan,
        "results": [result],
        "audit_log": [audit_entry],
    }


# ---------- Answer node ----------

def answer_node(state: AgentState) -> AgentState:
    """Generate final answer from collected results."""
    results = state["results"]
    final = f"Based on my research:\n\n" + "\n".join(results)
    return {**state, "final_answer": final}


# ---------- Graph construction ----------

def build_agent_graph():
    """Build the multi-agent supervisor graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("supervisor")

    # Supervisor decides: continue or finish
    def route_supervisor(state: AgentState) -> Literal["worker", "answer"]:
        if state.get("plan") == ["finish"]:
            return "answer"
        return "worker"

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"worker": "worker", "answer": "answer"},
    )

    # Worker loops back to supervisor
    workflow.add_edge("worker", "supervisor")

    # Answer ends the graph
    workflow.add_edge("answer", END)

    return workflow.compile()


# ---------- Run ----------

if __name__ == "__main__":
    app = build_agent_graph()

    initial_state = {
        "user_id": "user-123",
        "query": "What is the capital of France and what is 2+2?",
        "plan": [],
        "step": 0,
        "max_steps": 5,
        "results": [],
        "audit_log": [],
        "final_answer": "",
    }

    result = app.invoke(initial_state)

    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['final_answer']}")
    print(f"\nAudit Log ({len(result['audit_log'])} entries):")
    for entry in result["audit_log"]:
        print(f"  - {entry}")