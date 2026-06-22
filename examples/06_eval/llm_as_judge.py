"""
examples/06_eval/llm_as_judge.py

LLM-as-judge evaluation harness for RAG faithfulness.
Maps to docs/09-llmops-evalops/README.md "EvalOps".

What this demonstrates:
- Golden dataset eval
- LLM-as-judge scoring
- Multiple eval metrics (faithfulness, relevance, helpfulness)
- Aggregation + reporting

Run:
    python llm_as_judge.py
"""

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalExample:
    """One example in the eval dataset."""
    query: str
    answer: str
    contexts: List[str]
    expected_answer: str = ""  # for golden dataset eval
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating one example."""
    example: EvalExample
    scores: dict  # e.g., {"faithfulness": 4, "relevance": 5, ...}
    notes: str = ""


# ---------- LLM judge ----------

JUDGE_PROMPT = """You are evaluating a RAG system's output.

Query: {query}

Context provided to the system:
{contexts}

Answer produced:
{answer}

Rate the answer on these dimensions, each 1–5:

1. **Faithfulness**: Is the answer grounded in the context? (1 = hallucinated, 5 = fully grounded)
2. **Relevance**: Does the answer address the query? (1 = off-topic, 5 = directly addresses)
3. **Helpfulness**: Is the answer useful and complete? (1 = useless, 5 = excellent)
4. **Citation Quality**: Are sources cited correctly? (1 = none/wrong, 5 = all correct)

Respond with JSON:
{{
  "faithfulness": <int 1-5>,
  "relevance": <int 1-5>,
  "helpfulness": <int 1-5>,
  "citation_quality": <int 1-5>,
  "notes": "<one-sentence justification>"
}}
"""


def call_judge_llm(prompt: str) -> dict:
    """
    Call the judge LLM.

    In production: use Bedrock, OpenAI, vLLM, etc.
    Here: mock for demonstration.
    """
    # Replace with real LLM call
    return {
        "faithfulness": 4,
        "relevance": 5,
        "helpfulness": 4,
        "citation_quality": 5,
        "notes": "Answer is grounded in context; sources cited correctly.",
    }


def judge_example(example: EvalExample) -> EvalResult:
    """Run the LLM judge on one example."""
    contexts_text = "\n\n---\n\n".join(
        f"[{i+1}] {c}" for i, c in enumerate(example.contexts)
    )
    prompt = JUDGE_PROMPT.format(
        query=example.query,
        contexts=contexts_text,
        answer=example.answer,
    )
    scores = call_judge_llm(prompt)
    return EvalResult(example=example, scores=scores, notes=scores.get("notes", ""))


def aggregate_results(results: List[EvalResult]) -> dict:
    """Compute aggregate metrics across all examples."""
    if not results:
        return {}

    n = len(results)
    metric_keys = ["faithfulness", "relevance", "helpfulness", "citation_quality"]

    aggregates = {}
    for key in metric_keys:
        scores = [r.scores.get(key, 0) for r in results]
        aggregates[f"avg_{key}"] = sum(scores) / n
        aggregates[f"min_{key}"] = min(scores)
        aggregates[f"max_{key}"] = max(scores)
        aggregates[f"pct_{key}_gte_4"] = sum(1 for s in scores if s >= 4) / n

    aggregates["total_examples"] = n
    return aggregates


def run_eval_suite(examples: List[EvalExample]) -> dict:
    """Run the full eval suite."""
    results = [judge_example(ex) for ex in examples]
    aggregates = aggregate_results(results)
    return {"results": results, "aggregates": aggregates}


# ---------- Demo ----------

SAMPLE_EVAL_SET = [
    EvalExample(
        query="What is QLoRA?",
        answer="QLoRA loads the base model in 4-bit and trains LoRA on top, allowing 65B models on a single 48GB GPU.",
        contexts=[
            "QLoRA (Quantized LoRA) loads the base model in 4-bit (NF4) and trains LoRA adapters on top.",
        ],
    ),
    EvalExample(
        query="What is DoRA?",
        answer="DoRA is some kind of optimization.",
        contexts=[
            "DoRA decomposes weight into magnitude and direction, applying LoRA to direction only.",
        ],
    ),
]


if __name__ == "__main__":
    output = run_eval_suite(SAMPLE_EVAL_SET)

    print("=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)

    for key, val in output["aggregates"].items():
        if isinstance(val, float):
            print(f"{key:30s}: {val:.3f}")
        else:
            print(f"{key:30s}: {val}")

    print("\n--- Per-example notes ---")
    for r in output["results"]:
        print(f"Q: {r.example.query}")
        print(f"   Notes: {r.notes}")
        print()