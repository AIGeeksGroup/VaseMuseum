"""Automated evaluation aligned with VaseMuseum paper metrics (LLM-as-judge)."""

from .llm_judge import (
    EvalSample,
    PaperMetricLabels,
    aggregate_paper_metrics,
    evaluate_batch,
    evaluate_one,
)

__all__ = [
    "EvalSample",
    "PaperMetricLabels",
    "aggregate_paper_metrics",
    "evaluate_batch",
    "evaluate_one",
]
