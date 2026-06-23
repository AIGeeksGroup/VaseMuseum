"""
Practice dataset → repeated rollouts → reward labeling → experience pool updates.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import _bootstrap  # noqa: F401
from .control_profiles import CONTROL_PROFILES, ControlProfile, apply_control_profile_to_env
from .experience_updater import ExperienceUpdater
from .reward import judge_reward, reference_from_row
from .trajectory_format import messages_to_trajectory_text
from .resume_state import (
    ROLLOUT_RECORDS,
    RUN_STATE,
    append_rollout_record,
    config_fingerprint,
    experiences_step_path,
    load_experiences_json,
    load_rollout_cache,
    load_run_state,
    load_steps_complete,
    rollout_key,
    save_run_state,
    save_steps_complete,
    step_key,
    wipe_checkpoint_files,
)
from .types import RolloutRecord, TaskRecorder
from agent_run import VaseAgent, _assistant_text
from llm_env import get_tool_mode


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_id(row: dict[str, Any], line_idx: int) -> str:
    for k in ("id", "sample_id", "uuid"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    vr = row.get("vase_row_id")
    fd = row.get("field")
    if vr is not None and fd is not None:
        return f"{vr}_{fd}"
    return str(line_idx)


def _task_type(row: dict[str, Any]) -> str:
    t = row.get("task_type") or row.get("task") or ""
    if isinstance(t, str) and t.strip():
        return t.strip().lower()
    return "v_plus_k"


@dataclass
class AccumulationConfig:
    practice_jsonl: Path
    out_dir: Path
    control_key: str = "vase_full"
    grpo_n: int = 4
    experience_batch_size: int = 8
    epochs: int = 1
    shuffle: bool = False
    max_rows: int = 0
    start: int = 0
    max_tool_rounds: int = 10
    rollout_temperature: float = 0.7
    rollout_concurrency: int = 2
    experience_concurrency: int = 8
    num_experiences_per_query: int = 2
    given_ground_truth: bool = True
    tool_mode: str | None = None
    agent_objective: str = (
        "You are a Greek vase / museum QA agent that may call search tools and must answer faithfully."
    )
    learning_objective: str = (
        "Improve grounded, tool-assisted answers for museum metadata and imagery: better retrieval, "
        "claim–evidence alignment, and calibrated uncertainty."
    )
    log_rollouts_jsonl: bool = True
    seed: int | None = None
    #: If True (youtu default), only question groups whose mean reward is strictly between 0 and 1 feed extraction.
    partial_groups_only: bool = True
    #: Load ``rollout_records.jsonl`` / ``steps_complete.json`` and skip finished work.
    resume: bool = True
    #: Delete checkpoint files in ``out_dir`` and start over (implies no resume).
    fresh: bool = False
    #: Append meta-LLM (experience distillation) turns to ``experience_llm_traces.jsonl``.
    log_experience_llm_traces: bool = True


def _fingerprint_payload(cfg: AccumulationConfig, n_rows: int) -> dict[str, Any]:
    return {
        "practice_jsonl": str(cfg.practice_jsonl.resolve()),
        "start": cfg.start,
        "max_rows": cfg.max_rows,
        "n_rows_slice": n_rows,
        "control_key": cfg.control_key,
        "grpo_n": cfg.grpo_n,
        "experience_batch_size": cfg.experience_batch_size,
        "epochs": cfg.epochs,
        "shuffle": cfg.shuffle,
        "seed": cfg.seed,
        "partial_groups_only": cfg.partial_groups_only,
    }


def _restore_pool_from_last_step(out_dir: Path, completed: set[str]) -> dict[str, str]:
    if not completed:
        return {}
    best: tuple[int, int] | None = None
    for sk in completed:
        try:
            a, b = sk.split(":")
            t = (int(a), int(b))
        except ValueError:
            continue
        if best is None or t > best:
            best = t
    if best is None:
        return {}
    path = experiences_step_path(out_dir, best[0], best[1])
    return load_experiences_json(path)


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _batch_partial_stats(batch_records: list[RolloutRecord]) -> tuple[int, int]:
    """Returns (num_distinct_questions, num_partial_mean_reward_groups)."""
    by_q: dict[str, list[float]] = defaultdict(list)
    for r in batch_records:
        by_q[r.raw_question].append(float(r.reward))
    partial_n = 0
    for rs in by_q.values():
        if not rs:
            continue
        m = sum(rs) / len(rs)
        if 0.0 < m < 1.0:
            partial_n += 1
    return len(by_q), partial_n


def _one_rollout(
    agent: VaseAgent,
    profile: ControlProfile,
    row: dict[str, Any],
    *,
    line_idx: int,
    repeat_idx: int,
    cfg: AccumulationConfig,
) -> RolloutRecord:
    q = str(row.get("question") or "").strip()
    img = str(row.get("image") or row.get("image_url") or "").strip()
    ref = reference_from_row(row)
    sid = _sample_id(row, line_idx)
    tt = _task_type(row)

    apply_control_profile_to_env(profile)
    tm = cfg.tool_mode if cfg.tool_mode in ("openai", "xml") else None

    meta = agent.run(
        q,
        img,
        max_tool_rounds=cfg.max_tool_rounds,
        verbose=False,
        response_control_gate=profile.response_control_gate,
        apply_uncertain_preamble=profile.apply_uncertain_preamble,
        log_jsonl=False,
        return_metadata=True,
        tool_mode=tm,
        temperature=cfg.rollout_temperature,
    )
    assistant = meta["assistant"]
    messages = meta["messages"]
    text = messages_to_trajectory_text(messages)
    ans = _assistant_text(assistant)
    reward, notes = judge_reward(question=q, model_answer=ans, reference_answer=ref, task_type=tt)

    return RolloutRecord(
        raw_question=q,
        image_url=img,
        sample_id=f"{sid}__{repeat_idx}",
        task_type=tt,
        correct_answer=ref,
        trajectory_text=text,
        model_answer=ans,
        reward=float(reward),
        reasoning=notes,
        metadata={
            "control_key": profile.key,
            "line_idx": line_idx,
            "repeat_idx": repeat_idx,
        },
    )


def run_accumulation(cfg: AccumulationConfig, progress: Callable[[str], None] | None = None) -> dict[str, str]:
    """Run rollout collection + experience updates; returns final experience map."""
    if cfg.control_key not in CONTROL_PROFILES:
        raise ValueError(f"Unknown control key {cfg.control_key!r}; choose from {sorted(CONTROL_PROFILES.keys())}")

    profile = CONTROL_PROFILES[cfg.control_key]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(cfg.practice_jsonl)
    if cfg.start:
        rows = rows[cfg.start :]
    if cfg.max_rows and cfg.max_rows > 0:
        rows = rows[: cfg.max_rows]

    n_rows = len(rows)
    fp_payload = _fingerprint_payload(cfg, n_rows)
    fp = config_fingerprint(fp_payload)

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    use_resume = bool(cfg.resume) and not cfg.fresh
    if cfg.fresh:
        removed = wipe_checkpoint_files(cfg.out_dir)
        if removed:
            log(f"[resume] --fresh: removed {len(removed)} checkpoint file(s).")
        use_resume = False

    rollout_cache: dict[str, RolloutRecord] = {}
    steps_complete: set[str] = set()
    stored_rs = load_run_state(cfg.out_dir)
    if use_resume:
        rr_path = cfg.out_dir / ROLLOUT_RECORDS
        rs_path = cfg.out_dir / RUN_STATE
        if rr_path.is_file() and not rs_path.is_file():
            raise RuntimeError(
                f"[resume] Found {ROLLOUT_RECORDS} but missing {RUN_STATE} — cannot rebuild row order. "
                "Use --fresh to discard caches or restore run_state.json from backup."
            )
        if stored_rs and stored_rs.get("fingerprint") != fp:
            raise RuntimeError(
                f"[resume] Config fingerprint mismatch (saved {stored_rs.get('fingerprint')!r} vs now {fp!r}). "
                "Change dataset slice / grpo_n / batch size / epochs / shuffle / seed? Use --fresh to restart."
            )
        rollout_cache = load_rollout_cache(cfg.out_dir)
        steps_complete = load_steps_complete(cfg.out_dir)
        log(
            f"[resume] loaded rollout_cache={len(rollout_cache)} keys, completed_steps={len(steps_complete)} "
            f"({sorted(steps_complete, key=lambda s: tuple(int(x) for x in s.split(':')) )[:8]}"
            f"{'…' if len(steps_complete) > 8 else ''})"
        )

    if cfg.seed is not None:
        random.seed(cfg.seed)

    # Row order: must be stable across resumes when shuffle=True (stored in run_state.json).
    order: list[int]
    if (
        stored_rs
        and stored_rs.get("fingerprint") == fp
        and isinstance(stored_rs.get("order"), list)
        and len(stored_rs["order"]) == n_rows
    ):
        order = [int(x) for x in stored_rs["order"]]
        log("[resume] using saved row order from run_state.json")
    else:
        order = list(range(n_rows))
        if cfg.shuffle:
            random.shuffle(order)
        save_run_state(
            cfg.out_dir,
            {
                "fingerprint": fp,
                "fingerprint_payload": fp_payload,
                "order": order,
                "n_rows": n_rows,
            },
        )

    manifest: dict[str, Any] = {
        "practice_jsonl": str(cfg.practice_jsonl.resolve()),
        "n_rows": n_rows,
        "control_key": cfg.control_key,
        "grpo_n": cfg.grpo_n,
        "experience_batch_size": cfg.experience_batch_size,
        "epochs": cfg.epochs,
        "shuffle": cfg.shuffle,
        "max_tool_rounds": cfg.max_tool_rounds,
        "rollout_temperature": cfg.rollout_temperature,
        "tool_mode_env": get_tool_mode(),
        "tool_mode_override": cfg.tool_mode,
        "agent_objective": cfg.agent_objective,
        "learning_objective": cfg.learning_objective,
        "partial_groups_only": cfg.partial_groups_only,
        "resume": use_resume,
        "config_fingerprint": fp,
        "checkpoint_note": "Resume: rollout_records.jsonl (per-rollout) + steps_complete.json (per batch step) + run_state.json (shuffle order + fingerprint)",
        "experience_llm_traces_jsonl": str((cfg.out_dir / "experience_llm_traces.jsonl").resolve())
        if cfg.log_experience_llm_traces
        else None,
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    recorder = TaskRecorder(experiment_name=cfg.out_dir.name)
    recorder.experiences = _restore_pool_from_last_step(cfg.out_dir, steps_complete)

    _exp_trace = (cfg.out_dir / "experience_llm_traces.jsonl") if cfg.log_experience_llm_traces else None
    updater = ExperienceUpdater(
        agent_objective=cfg.agent_objective,
        learning_objective=cfg.learning_objective,
        trace_jsonl=_exp_trace,
    )

    agent = VaseAgent.from_env()
    rollouts_path = cfg.out_dir / "practice_rollouts.jsonl"
    if cfg.log_rollouts_jsonl and not use_resume and rollouts_path.exists():
        rollouts_path.unlink()

    global_idx = _count_lines(rollouts_path) if cfg.log_rollouts_jsonl and use_resume else 0

    t0 = time.perf_counter()
    num_batches = (n_rows + cfg.experience_batch_size - 1) // cfg.experience_batch_size

    for epoch in range(cfg.epochs):
        for batch_no in range(num_batches):
            bi = batch_no * cfg.experience_batch_size
            chunk_idx = order[bi : bi + cfg.experience_batch_size]
            sk = step_key(epoch, batch_no)

            if sk in steps_complete:
                exp_p = experiences_step_path(cfg.out_dir, epoch, batch_no)
                pool = load_experiences_json(exp_p)
                if pool:
                    recorder.experiences = pool
                log(f"[resume] skip step {sk} (epoch {epoch + 1} batch {batch_no}/{num_batches - 1}) — already complete")
                continue

            tasks: list[tuple[int, int]] = []
            for i_row in chunk_idx:
                for rep in range(cfg.grpo_n):
                    tasks.append((i_row, rep))

            def run_cell(i_row: int, rep: int) -> tuple[str, RolloutRecord, bool]:
                rk = rollout_key(epoch, i_row, rep)
                if rk in rollout_cache:
                    return rk, rollout_cache[rk], True
                rec = _one_rollout(agent, profile, rows[i_row], line_idx=i_row, repeat_idx=rep, cfg=cfg)
                return rk, rec, False

            batch_records: list[RolloutRecord] = []
            workers = max(1, min(cfg.rollout_concurrency, len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_cell, i_row, rep): (i_row, rep) for i_row, rep in tasks}
                for fut in as_completed(futs):
                    rk, rec, cached = fut.result()
                    batch_records.append(rec)
                    if not cached:
                        rollout_cache[rk] = rec
                        append_rollout_record(cfg.out_dir, rk, rec)
                        if cfg.log_rollouts_jsonl:
                            global_idx += 1
                            payload = {
                                "epoch": epoch,
                                "global_idx": global_idx,
                                "sample_id": rec.sample_id,
                                "reward": rec.reward,
                                "control_key": profile.key,
                                "question": rec.raw_question,
                                "reference_answer": rec.correct_answer,
                                "model_answer": rec.model_answer,
                            }
                            with rollouts_path.open("a", encoding="utf-8") as rf:
                                rf.write(json.dumps(payload, ensure_ascii=False) + "\n")

            keyed: dict[str, RolloutRecord] = {}
            for r in batch_records:
                li = int(r.metadata.get("line_idx", -1))
                ri = int(r.metadata.get("repeat_idx", -1))
                if li >= 0 and ri >= 0:
                    keyed[rollout_key(epoch, li, ri)] = r
            ordered: list[RolloutRecord] = []
            missing_keys: list[str] = []
            for i_row in chunk_idx:
                for rep in range(cfg.grpo_n):
                    k = rollout_key(epoch, i_row, rep)
                    if k not in keyed:
                        missing_keys.append(k)
                    else:
                        ordered.append(keyed[k])
            if missing_keys or len(ordered) != len(batch_records):
                raise RuntimeError(
                    f"Internal batch assembly failed (missing {missing_keys[:6]}, "
                    f"got {len(ordered)} vs {len(batch_records)} records)."
                )

            n_q, n_partial = _batch_partial_stats(ordered)
            log(
                f"[epoch {epoch + 1}/{cfg.epochs}] step {sk} batch rows={len(chunk_idx)} rollouts={len(ordered)} "
                f"| distinct_questions={n_q} partial_reward_groups={n_partial} "
                f"(partial-only={'on' if cfg.partial_groups_only else 'off'}) → updating pool…"
            )
            if cfg.partial_groups_only and n_partial == 0:
                log(
                    "[experience] No question group has mean reward in (0,1); extractor input is empty unless "
                    "you pass --no-partial-only on run_tf_grpo_accumulate.py."
                )

            updater.run(
                ordered,
                recorder,
                concurrency=cfg.experience_concurrency,
                given_ground_truth=cfg.given_ground_truth,
                num_experiences=cfg.num_experiences_per_query,
                partial_groups_only=cfg.partial_groups_only,
                diag_log=log,
                trace_step_id=sk,
            )

            step_path = experiences_step_path(cfg.out_dir, epoch, batch_no)
            step_path.write_text(json.dumps(recorder.experiences, indent=2, ensure_ascii=False), encoding="utf-8")
            steps_complete.add(sk)
            save_steps_complete(cfg.out_dir, steps_complete)
            log(f"[resume] marked step {sk} complete → {step_path.name}")

    final_path = cfg.out_dir / "experiences_final.json"
    final_path.write_text(json.dumps(recorder.experiences, indent=2, ensure_ascii=False), encoding="utf-8")

    text_blob = "\n".join([f"[{i}]. {e}" for i, e in recorder.experiences.items()])
    (cfg.out_dir / "experiences_final.txt").write_text(text_blob, encoding="utf-8")

    log(f"done in {time.perf_counter() - t0:.1f}s; wrote {final_path}")
    if not recorder.experiences:
        tail = (
            "partial_groups_only is on and every question had all-correct or all-wrong K rollouts; "
            if cfg.partial_groups_only
            else "all rollouts may share the same judge score (weak contrast); "
        )
        log(
            "[experience] Final pool is empty. Common causes: (1) "
            + tail
            + "(2) meta-LLM JSON merge / group_update parse failed; (3) group-advantage omitted "
            "<Experiences>...</>. See stderr lines prefixed with [experience] distill / batch merge."
        )
        if cfg.partial_groups_only:
            log("[experience] Tip: try --no-partial-only to distill from every question group.")
    return dict(recorder.experiences)


def format_prompt_block(experiences: dict[str, str]) -> str:
    """Render experiences as an instruction appendix (for downstream eval prompts)."""
    if not experiences:
        return ""
    lines = "\n".join([f"[{i}]. {e}" for i, e in experiences.items()])
    return (
        "\n\nWhen solving problems, read these distilled experiences first:\n"
        f"{lines}\n"
    )
