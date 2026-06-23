"""Binary reward from reference answers via the paper LLM-as-judge."""

from __future__ import annotations

from typing import Any

from . import _bootstrap  # noqa: F401
from metrics.llm_judge import evaluate_one


def judge_reward(
    *,
    question: str,
    model_answer: str,
    reference_answer: str,
    task_type: str,
) -> tuple[float, str | None]:
    """
    Returns reward in ``{{0.0, 1.0}}`` and optional short reasoning (judge notes).
    """
    ref = reference_answer.strip() if reference_answer else ""
    if not ref:
        return 0.0, "empty_reference"

    labels, meta = evaluate_one(
        question=question,
        model_answer=model_answer,
        reference_answer=reference_answer,
        task_type=task_type or "v_plus_k",
        temperature=0.0,
    )
    notes = getattr(labels, "notes", None)
    return float(labels.correctness), notes


def reference_from_row(row: dict[str, Any]) -> str:
    for k in ("answer", "reference_answer", "ground_truth"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
