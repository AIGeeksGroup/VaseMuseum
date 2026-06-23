"""
LLM-based experience distillation (single rollout summary → group advantage → pool merge),
ported from youtu-agent ``ExperienceUpdater`` and adapted to :class:`RolloutRecord`.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

from . import _bootstrap  # noqa: F401

if TYPE_CHECKING:
    from openai import OpenAI
from .prompts_embedded import (
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP,
    SINGLE_QUERY_GROUP_ADVANTAGE_SP,
    SINGLE_QUERY_GROUP_ADVANTAGE_UP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP,
)
from pathlib import Path

from .types import RolloutRecord, TaskRecorder
from llm_env import get_llm_config
from metrics.llm_judge import extract_first_json_object


def _assistant_text_from_response(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return ""
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            str(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    if isinstance(usage, dict):
        return dict(usage)
    return {"repr": str(usage)}


class ExperienceUpdater:
    def __init__(
        self,
        *,
        agent_objective: str,
        learning_objective: str,
        llm_temperature: float = 0.2,
        trace_jsonl: Path | None = None,
    ) -> None:
        self.agent_objective = agent_objective
        self.learning_objective = learning_objective
        self.llm_temperature = llm_temperature
        self._trace_path = trace_jsonl
        self._trace_lock = threading.Lock()
        self._trace_step_id = ""
        try:
            from openai import OpenAI as _OpenAI  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Experience extraction requires the `openai` package (same dependency as vase-agent)."
            ) from e
        cfg = get_llm_config()
        self._model = cfg["model"]
        self._client = _OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], max_retries=2)

    def _append_trace(self, meta: dict[str, Any], system: str, user: str, assistant: str) -> None:
        if self._trace_path is None:
            return
        rec: dict[str, Any] = {
            "trace_step_id": self._trace_step_id,
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            **meta,
        }
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_lock:
            with self._trace_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()

    def _query_llm(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        trace_meta: dict[str, Any] | None = None,
    ) -> str:
        temp = self.llm_temperature if temperature is None else temperature
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            timeout=180,
        )
        text = _assistant_text_from_response(resp).strip()
        tm = dict(trace_meta or {})
        tm.setdefault("usage", _usage_to_dict(getattr(resp, "usage", None)))
        self._append_trace(tm, system, user, text)
        return text

    def run(
        self,
        rollouts: Sequence[RolloutRecord],
        recorder: TaskRecorder,
        *,
        concurrency: int = 8,
        given_ground_truth: bool = True,
        num_experiences: int = 2,
        partial_groups_only: bool = True,
        diag_log: Callable[[str], None] | None = None,
        trace_step_id: str | None = None,
    ) -> dict[str, str]:
        """Return updated global experience map ``{{\"0\": text, ...}}``."""
        self._trace_step_id = trace_step_id or ""

        def _log(msg: str) -> None:
            if diag_log:
                diag_log(msg)

        problem_to_rollouts: dict[str, list[RolloutRecord]] = defaultdict(list)
        for r in rollouts:
            problem_to_rollouts[r.raw_question].append(r)

        all_rollouts_to_process: list[RolloutRecord] = []
        for group in problem_to_rollouts.values():
            if not group:
                continue
            if given_ground_truth:
                scores = [each.reward for each in group]
                avg_score = sum(scores) / len(scores)
                # Match youtu: only "disagreement" groups yield contrastive signal — unless relaxed.
                take = (avg_score > 0.0 and avg_score < 1.0) if partial_groups_only else True
                if take:
                    all_rollouts_to_process.extend(group)
            else:
                all_rollouts_to_process.extend(group)

        summarized = self._single_rollout_summary(all_rollouts_to_process, concurrency=concurrency, given_ground_truth=given_ground_truth)
        grouped = self._group_advantage(
            summarized,
            concurrency=concurrency,
            given_ground_truth=given_ground_truth,
            num_experiences=num_experiences,
            partial_groups_only=partial_groups_only,
        )
        n_roll_summ = sum(len(v) for v in summarized.values())
        _log(
            f"[experience] distill: rollouts_in={len(all_rollouts_to_process)} summarized_ok={n_roll_summ} "
            f"questions={len(summarized)} group_advantage_blocks={len(grouped)}"
        )

        critiques = self._group_update(recorder, grouped, concurrency=concurrency)
        _log(f"[experience] group_update_ok={len(critiques)}/{len(grouped)}")
        merged = self._batch_update(recorder, critiques)
        if not merged:
            fb = self._fallback_pool_from_grouped(grouped)
            if fb:
                merged = fb
                _log(
                    f"[experience] batch merge yielded empty pool; using fallback from <Experiences> text "
                    f"({len(fb)} line(s))"
                )
            else:
                _log(
                    "[experience] batch merge empty and no <Experiences> text to fall back to "
                    "(check meta-LLM JSON / tags; VL judge may mark all rollouts correct so contrast is weak)."
                )

        renamed = {f"G{i}": exp for i, exp in enumerate(merged.values())}
        recorder.experiences_update(renamed)
        return renamed

    def _single_rollout_summary(
        self,
        rollouts: list[RolloutRecord],
        *,
        concurrency: int,
        given_ground_truth: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        def work(item: RolloutRecord) -> dict[str, Any] | None:
            try:
                sp = SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP.format(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                )
                up = SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP.format(
                    question=item.raw_question,
                    answer=item.correct_answer if given_ground_truth else "[REDACTED]",
                    critique=item.reasoning or "[No critique provided]",
                    trajectory=item.trajectory_text,
                )
                summary = self._query_llm(
                    system=sp,
                    user=up,
                    trace_meta={
                        "stage": "single_rollout_summary",
                        "sample_id": item.sample_id,
                        "question": (item.raw_question or "")[:800],
                    },
                )
                return {
                    "trajectory_summary": summary,
                    "raw_question": item.raw_question,
                    "reward": item.reward,
                    "trajectory_text": item.trajectory_text,
                    "correct_answer": item.correct_answer,
                    "reasoning": item.reasoning,
                    "model_answer": item.model_answer,
                }
            except Exception as e:
                return {"_error": str(e), "raw_question": item.raw_question}

        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if not rollouts:
            return {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [ex.submit(work, r) for r in rollouts]
            for fut in as_completed(futs):
                result = fut.result()
                if result is None or "_error" in result:
                    continue
                q = str(result.get("raw_question") or "")
                results[q].append(result)
        return dict(results)

    def _group_advantage(
        self,
        problem_to_summarized: dict[str, list[dict[str, Any]]],
        *,
        concurrency: int,
        given_ground_truth: bool,
        num_experiences: int,
        partial_groups_only: bool = True,
    ) -> list[dict[str, Any]]:
        batches: list[list[dict[str, Any]]] = []
        for rollouts in problem_to_summarized.values():
            if not rollouts:
                continue
            if given_ground_truth:
                scores = [float(each["reward"]) for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if partial_groups_only and not (avg_score > 0.0 and avg_score < 1.0):
                    continue
            batches.append(rollouts)

        sem_worker = max(1, concurrency)

        def work(rollouts_per_problem: list[dict[str, Any]]) -> dict[str, Any] | None:
            try:
                formatted_trajectories = "\n\n".join(
                    [
                        f"Attempt {i + 1} (Reward {each['reward'] if given_ground_truth else '[REDACTED]'}):\n"
                        f"{each['trajectory_summary']}"
                        for i, each in enumerate(rollouts_per_problem)
                    ]
                )
                sp = SINGLE_QUERY_GROUP_ADVANTAGE_SP.format(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                    num_experiences=num_experiences,
                )
                up = SINGLE_QUERY_GROUP_ADVANTAGE_UP.format(
                    question=rollouts_per_problem[0]["raw_question"],
                    answer=rollouts_per_problem[0]["correct_answer"] if given_ground_truth else "[REDACTED]",
                    trajectories=formatted_trajectories,
                )
                response = self._query_llm(
                    system=sp,
                    user=up,
                    trace_meta={
                        "stage": "group_advantage",
                        "question": str(rollouts_per_problem[0].get("raw_question") or "")[:800],
                    },
                )
                pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>", re.DOTALL | re.IGNORECASE)
                match = pattern.search(response)
                experiences = match.group(1).strip() if match else ""
                return {"rollouts": rollouts_per_problem, "critique": response, "experiences": experiences}
            except Exception:
                return None

        out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=sem_worker) as ex:
            futs = [ex.submit(work, b) for b in batches]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    out.append(r)
        return out

    def _group_update(
        self,
        recorder: TaskRecorder,
        new_experiences: list[dict[str, Any]],
        *,
        concurrency: int,
    ) -> list[dict[str, Any]]:
        curr = recorder.experiences or {}
        formatted_experiences = "\n".join([f"[{i}]. {e}" for i, e in curr.items()]) if curr else "None"

        def work(ne: dict[str, Any]) -> dict[str, Any] | None:
            try:
                sp = GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP.format(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                )
                up = GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    existing_experiences=formatted_experiences,
                    new_experiences=ne.get("experiences") or "",
                )
                response = self._query_llm(
                    system=sp,
                    user=up,
                    trace_meta={"stage": "group_update"},
                )
                chunk = response.split("```json")[-1].split("```")[0]
                try:
                    operations = json.loads(chunk)
                except json.JSONDecodeError:
                    parsed = extract_first_json_object(response)
                    operations = parsed if isinstance(parsed, list) else None
                if not isinstance(operations, list):
                    return None
                return {"operations": operations, **ne}
            except Exception:
                return None

        merged: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [ex.submit(work, ne) for ne in new_experiences]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    merged.append(r)
        return merged

    def _parse_revision_plan_json(self, text: str | None) -> list[dict[str, Any]] | None:
        """Parse batch-merge JSON: top-level array, or dict wrapping ``revision_plan`` / ``operations`` / ``changes``."""
        if not text or not str(text).strip():
            return None
        s = str(text).strip()
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            obj = extract_first_json_object(s)
            if isinstance(obj, dict):
                for key in ("revision_plan", "operations", "changes"):
                    v = obj.get(key)
                    if isinstance(v, list):
                        return v
            return None
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("revision_plan", "operations", "changes"):
                v = raw.get(key)
                if isinstance(v, list):
                    return v
        return None

    def _batch_update(self, recorder: TaskRecorder, critiques: list[dict[str, Any]], max_retries: int = 3) -> dict[str, str]:
        all_operations: list[dict[str, Any]] = []
        for each in critiques:
            ops = each.get("operations")
            if isinstance(ops, list):
                all_operations.extend(ops)

        experiences = recorder.experiences or {}
        revision_plan: list[dict[str, Any]] = []
        for attempt in range(max_retries):
            try:
                sp = BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP.format(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                )
                up = BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    experiences_and_operations=self._format_exp_and_ops(experiences, all_operations),
                )
                response = self._query_llm(
                    system=sp,
                    user=up,
                    trace_meta={"stage": "batch_merge", "attempt": attempt},
                )
                tail = response.split("```json")[-1].split("```")[0]
                # Meta-LLM returns a JSON *array* of ops; ``extract_first_json_object`` only accepts dicts.
                parsed_list = self._parse_revision_plan_json(tail) or self._parse_revision_plan_json(response)
                if isinstance(parsed_list, list) and parsed_list:
                    revision_plan = parsed_list
                    break
            except Exception:
                continue

        new_experiences = copy.deepcopy(experiences)

        def _next_numeric_id(pool: dict[str, str]) -> str:
            best = -1
            for k in pool.keys():
                if k.isdigit():
                    best = max(best, int(k))
                elif k.startswith("G") and k[1:].isdigit():
                    best = max(best, int(k[1:]))
            return str(best + 1)

        for plan in revision_plan:
            if not isinstance(plan, dict):
                continue
            operation = str(plan.get("operation") or "ADD").upper()
            content = str(plan.get("content") or "").strip()
            target_id = plan.get("id")
            target_key = str(target_id) if target_id is not None and str(target_id).strip() else None

            if not content and operation != "DELETE":
                continue

            if operation == "ADD":
                nk = _next_numeric_id(new_experiences)
                new_experiences[nk] = content
            elif operation == "UPDATE":
                if target_key and target_key in new_experiences:
                    new_experiences[target_key] = content or new_experiences[target_key]
                else:
                    nk = _next_numeric_id(new_experiences)
                    new_experiences[nk] = content
            elif operation == "DELETE":
                if target_key and target_key in new_experiences:
                    del new_experiences[target_key]
            elif operation == "NONE":
                continue

        return new_experiences

    def _format_exp_and_ops(self, experiences: dict[str, str], operations: list[dict[str, Any]]) -> str:
        if not operations:
            return "No batch operations."
        parts: list[str] = []
        for eid, exp in experiences.items():
            curr = f"Experience {eid}:\nContent: {exp}\n"
            related = [op for op in operations if str(op.get("id")) == str(eid)]
            if related:
                curr += "Related Operations:\n" + "\n".join(json.dumps(op, ensure_ascii=False) for op in related)
            else:
                curr += "No related operations."
            parts.append(curr)
        no_id = [op for op in operations if not op.get("id", None)]
        if no_id:
            blob = "\n".join(json.dumps(op, ensure_ascii=False) for op in no_id)
            parts.append("Operations without specific Experience ID:\n" + blob)
        return "\n\n".join(parts)

    def _fallback_pool_from_grouped(self, grouped: list[dict[str, Any]]) -> dict[str, str]:
        """
        If JSON merge steps produced no entries, still persist non-empty ``<Experiences>`` lines.

        Models often omit fences or return invalid JSON; the group-advantage step may still emit tagged text.
        """
        out: dict[str, str] = {}
        n = 0
        for g in grouped:
            raw = (g.get("experiences") or "").strip()
            if not raw and isinstance(g.get("critique"), str):
                m = re.search(r"<Experiences>\s*(.*?)\s*</Experiences>", g["critique"], re.DOTALL | re.IGNORECASE)
                if m:
                    raw = m.group(1).strip()
            for line in raw.splitlines():
                line = line.strip()
                if len(line) < 4:
                    continue
                line = re.sub(r"^[\d]+[\).\s]+", "", line).strip()
                if line:
                    out[str(n)] = line
                    n += 1
        return out
