"""
File-based checkpoint / resume (similar in spirit to youtu DB ``ExperienceCache`` + ``restart_step``).

Artifacts under ``out_dir``:
  - ``run_state.json`` — config fingerprint + row ``order`` (for reproducible resume with shuffle)
  - ``rollout_records.jsonl`` — one completed rollout per line (full :class:`RolloutRecord`, for crash-safe incremental save)
  - ``steps_complete.json`` — which experience-update steps finished successfully
  - ``experience_llm_traces.jsonl`` — meta-LLM chat turns for experience distillation (optional; appended across resume)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import RolloutRecord, rollout_record_from_dict, rollout_record_to_dict

RUN_STATE = "run_state.json"
ROLLOUT_RECORDS = "rollout_records.jsonl"
STEPS_COMPLETE = "steps_complete.json"
EXPERIENCE_LLM_TRACES = "experience_llm_traces.jsonl"


def config_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def rollout_key(epoch: int, i_row: int, rep: int) -> str:
    return f"{epoch}:{i_row}:{rep}"


def step_key(epoch: int, batch_no: int) -> str:
    return f"{epoch}:{batch_no}"


def wipe_checkpoint_files(out_dir: Path) -> list[str]:
    """Remove resume artifacts; returns list of removed paths (as strings)."""
    removed: list[str] = []
    for name in (RUN_STATE, ROLLOUT_RECORDS, STEPS_COMPLETE, EXPERIENCE_LLM_TRACES):
        p = out_dir / name
        if p.is_file():
            p.unlink()
            removed.append(str(p))
    for p in sorted(out_dir.glob("experiences_epoch*_batch*.json")):
        p.unlink()
        removed.append(str(p))
    return removed


def load_run_state(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / RUN_STATE
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_run_state(out_dir: Path, state: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RUN_STATE).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rollout_cache(out_dir: Path) -> dict[str, RolloutRecord]:
    p = out_dir / ROLLOUT_RECORDS
    if not p.is_file():
        return {}
    cache: dict[str, RolloutRecord] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = str(obj.get("key") or "")
            rec = obj.get("record")
            if k and isinstance(rec, dict):
                cache[k] = rollout_record_from_dict(rec)
    return cache


def append_rollout_record(out_dir: Path, key: str, record: RolloutRecord) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ROLLOUT_RECORDS
    payload = {"key": key, "record": rollout_record_to_dict(record)}
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def load_steps_complete(out_dir: Path) -> set[str]:
    p = out_dir / STEPS_COMPLETE
    if not p.is_file():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    lst = data.get("completed")
    if isinstance(lst, list):
        return {str(x) for x in lst}
    return set()


def save_steps_complete(out_dir: Path, completed: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(completed, key=lambda s: tuple(int(x) for x in s.split(":")))
    (out_dir / STEPS_COMPLETE).write_text(
        json.dumps({"completed": ordered}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def experiences_step_path(out_dir: Path, epoch: int, batch_no: int) -> Path:
    return out_dir / f"experiences_epoch{epoch}_batch{batch_no:04d}.json"


def load_experiences_json(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}
