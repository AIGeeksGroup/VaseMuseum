from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutRecord:
    """One sampled rollout for training-free GRPO experience extraction."""

    raw_question: str
    image_url: str
    sample_id: str
    task_type: str
    correct_answer: str
    trajectory_text: str
    model_answer: str
    reward: float
    reasoning: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def rollout_record_to_dict(r: RolloutRecord) -> dict[str, Any]:
    return {
        "raw_question": r.raw_question,
        "image_url": r.image_url,
        "sample_id": r.sample_id,
        "task_type": r.task_type,
        "correct_answer": r.correct_answer,
        "trajectory_text": r.trajectory_text,
        "model_answer": r.model_answer,
        "reward": r.reward,
        "reasoning": r.reasoning,
        "metadata": dict(r.metadata),
    }


def rollout_record_from_dict(d: dict[str, Any]) -> RolloutRecord:
    return RolloutRecord(
        raw_question=str(d.get("raw_question") or ""),
        image_url=str(d.get("image_url") or ""),
        sample_id=str(d.get("sample_id") or ""),
        task_type=str(d.get("task_type") or ""),
        correct_answer=str(d.get("correct_answer") or ""),
        trajectory_text=str(d.get("trajectory_text") or ""),
        model_answer=str(d.get("model_answer") or ""),
        reward=float(d.get("reward") or 0.0),
        reasoning=(None if d.get("reasoning") is None else str(d.get("reasoning"))),
        metadata=dict(d.get("metadata") or {}),
    )


@dataclass
class TaskRecorder:
    """Tracks global experience strings keyed by string IDs (``\"0\"``, ``\"1\"``, ...)."""

    experiment_name: str
    experiences: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def experiences_update(self, new_map: dict[str, str]) -> None:
        self.experiences = dict(new_map)

    def stat_update(self, patch: dict[str, Any]) -> None:
        self.stats.update(patch)
