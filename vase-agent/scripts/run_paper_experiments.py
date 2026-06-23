#!/usr/bin/env python3
"""
Run paper-aligned evaluation protocols on a JSONL eval set.

Aligned with §Baselines / ablations over **source control** (evidence pool) vs **response control** (gate + preamble).

Intermediate rows are ``neither_control``, ``source_control_only``, and ``response_control_only``.

Inference-time K-sample GRPO is not implemented in agent_run; ``vase_full`` means tools + source + response control.

Prediction JSONL files are appended incrementally (flush after each sample). Pass ``--resume`` to skip
samples already present in each ``predictions_<method>.jsonl`` and continue appending (same ``--eval-jsonl``
slice as the saved manifest).

Pass ``--judge-now`` to run LLM-as-judge only on existing ``predictions_all.jsonl`` (no inference).

Judge step skips rows whose ``sample_id`` already appears in ``judged_per_sample.jsonl`` (resume-friendly).
Each new judgment is appended to that file immediately (flush + fsync) so interrupts preserve progress;
the file is rewritten in prediction order when the judge batch finishes.

Eval JSONL (one JSON object per line), expected keys:
  - question (str)
  - image (str): http(s) URL or local file path
  - answer (str, optional): ground truth for metrics
  - task_type (str, optional): e.g. v_only | v_plus_k | ambiguous; default v_plus_k
  - id / sample_id / vase_row_id: optional identifier

Example: dataset/data/grpo_tf_600_flat.jsonl (field `answer` is used as reference if present).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

# vase-agent root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_run import VaseAgent, _assistant_text  # noqa: E402
from llm_env import get_tool_mode  # noqa: E402


# Method keys (choose subset with --methods).
METHODS: dict[str, dict[str, Any]] = {
    "direct": {
        "label": "Direct — VLM only (no tools)",
        "runner": "direct",
    },
    # Raw retrieval: no source-control pool, no response gate.
    "neither_control": {
        "label": "+Tools / search — neither source nor response control",
        "runner": "agent",
        "response_control_gate": False,
        "apply_uncertain_preamble": False,
        "disable_source_control": True,
    },
    # Evidence pool on; response gate off.
    "source_control_only": {
        "label": "Source control only — structured evidence pool; no response gate",
        "runner": "agent",
        "response_control_gate": False,
        "apply_uncertain_preamble": False,
        "disable_source_control": False,
    },
    # Response gate on; source-control pool off.
    "response_control_only": {
        "label": "Response control only — gate + preamble; source-control pool disabled",
        "runner": "agent",
        "response_control_gate": True,
        "apply_uncertain_preamble": True,
        "disable_source_control": True,
    },
    "vase_full": {
        "label": "Full — source + response control (no K-sample GRPO in repo)",
        "runner": "agent",
        "response_control_gate": True,
        "apply_uncertain_preamble": True,
        "disable_source_control": False,
    },
}


def _set_source_control_env(disable: bool) -> None:
    if disable:
        os.environ["VASE_DISABLE_SOURCE_CONTROL"] = "1"
    else:
        os.environ.pop("VASE_DISABLE_SOURCE_CONTROL", None)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _reference_answer(row: dict[str, Any]) -> str:
    for k in ("answer", "reference_answer", "ground_truth"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _task_type(row: dict[str, Any]) -> str:
    t = row.get("task_type") or row.get("task") or ""
    if isinstance(t, str) and t.strip():
        return t.strip().lower()
    return "v_plus_k"


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


def _progress_enumerate(
    rows: list[dict[str, Any]],
    *,
    desc: str,
    enabled: bool,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Enumerate rows; optionally wrap with tqdm on stderr."""
    n = len(rows)
    it: Iterator[tuple[int, dict[str, Any]]] = iter(enumerate(rows))
    if enabled and n > 0:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            it = tqdm(
                enumerate(rows),
                total=n,
                desc=desc,
                unit="sample",
                file=sys.stderr,
                leave=True,
            )
        except ImportError:
            print(f"[progress] {desc} ({n} samples)", file=sys.stderr, flush=True)
    yield from it


def _iter_indexed_samples(
    indexed: list[tuple[int, dict[str, Any]]],
    *,
    desc: str,
    enabled: bool,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Like :func:`_progress_enumerate` but for pre-built ``(line_idx, row)`` pairs (for resume skips)."""
    n = len(indexed)
    it: Iterator[tuple[int, dict[str, Any]]] = iter(indexed)
    if enabled and n > 0:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            it = tqdm(
                indexed,
                total=n,
                desc=desc,
                unit="sample",
                file=sys.stderr,
                leave=True,
            )
        except ImportError:
            print(f"[progress] {desc} ({n} samples)", file=sys.stderr, flush=True)
    yield from it


def _load_done_line_indices(pred_path: Path) -> set[int]:
    """``line_idx`` values already written to a per-method predictions JSONL."""
    done: set[int] = set()
    if not pred_path.is_file():
        return done
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                li = rec.get("line_idx")
                if isinstance(li, int):
                    done.add(li)
            except json.JSONDecodeError:
                continue
    return done


def _infer_methods_order(combined_path: Path) -> list[str]:
    """First-seen order of ``method`` field in ``predictions_all.jsonl``."""
    order: list[str] = []
    seen: set[str] = set()
    with combined_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                m = rec.get("method")
                if isinstance(m, str) and m not in seen:
                    seen.add(m)
                    order.append(m)
            except json.JSONDecodeError:
                continue
    return order


def _resolve_methods_for_judge(args: Any, combined_path: Path) -> list[str]:
    """Prefer ``manifest.json`` methods; else ``--methods``; else first-seen order in combined JSONL."""
    man_path = args.out_dir / "manifest.json"
    if man_path.is_file():
        prev = json.loads(man_path.read_text(encoding="utf-8"))
        pm = prev.get("methods")
        if isinstance(pm, list) and pm:
            out = [str(x) for x in pm]
            for m in out:
                if m not in METHODS:
                    raise SystemExit(f"judge: unknown method {m!r} in manifest.json")
            return out

    raw_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods: list[str] = []
    seen: set[str] = set()
    for m in raw_methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown method: {m!r}. Known keys: {sorted(METHODS.keys())}")
        if m not in seen:
            methods.append(m)
            seen.add(m)
    if methods:
        return methods

    inferred = _infer_methods_order(combined_path)
    if not inferred:
        raise SystemExit(
            "judge: could not infer methods from predictions_all.jsonl; pass --methods or keep manifest.json."
        )
    return inferred


def _load_judged_by_id(judged_path: Path) -> dict[str, dict[str, Any]]:
    """Map ``sample_id`` (e.g. ``direct:uuid``) -> row from ``judged_per_sample.jsonl``."""
    out: dict[str, dict[str, Any]] = {}
    if not judged_path.is_file():
        return out
    with judged_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("sample_id")
                if isinstance(sid, str) and sid:
                    out[sid] = row
            except json.JSONDecodeError:
                continue
    return out


def _run_llm_judge(
    out_dir: Path,
    combined_path: Path,
    methods: list[str],
    judge_workers: int,
    *,
    show_progress: bool = True,
) -> None:
    from metrics.llm_judge import aggregate_paper_metrics, evaluate_batch  # noqa: WPS433

    judge_samples: list[dict[str, Any]] = []

    with combined_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ms = rec.get("model_answer") or ""
            if not str(ms).strip():
                continue
            sid = f'{rec["method"]}:{rec["sample_id"]}'
            judge_samples.append(
                {
                    "id": sid,
                    "question": rec["question"],
                    "model_answer": ms,
                    "reference_answer": rec.get("reference_answer") or "",
                    "task_type": rec.get("task_type") or "v_plus_k",
                }
            )

    judged_path = out_dir / "judged_per_sample.jsonl"
    if judge_samples:
        existing = _load_judged_by_id(judged_path)
        pending = [s for s in judge_samples if str(s.get("id")) not in existing]
        n_skip = len(judge_samples) - len(pending)
        _info("")
        _info(f"== LLM judge ({len(judge_samples)} ok answers, workers={max(1, judge_workers)}) ==")
        if n_skip:
            _info(f"    resume: skip {n_skip} already in {judged_path.name}; run {len(pending)} pending")

        t_j = time.perf_counter()
        new_by_id: dict[str, dict[str, Any]] = {}
        append_lock = threading.Lock()

        def _persist_row(row: dict[str, Any]) -> None:
            sid = row.get("sample_id")
            if not isinstance(sid, str):
                return
            line = json.dumps(row, ensure_ascii=False) + "\n"
            with append_lock:
                new_by_id[sid] = row
                with judged_path.open("a", encoding="utf-8", newline="\n") as jf:
                    jf.write(line)
                    jf.flush()
                    try:
                        os.fsync(jf.fileno())
                    except OSError:
                        pass

        if pending:
            _info(f"    incremental save: each judged row append+fsync → {judged_path.name}")
            evaluate_batch(
                pending,
                max_workers=max(1, judge_workers),
                progress=show_progress,
                progress_desc="LLM judge",
                on_completed=_persist_row,
            )

        judged_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for s in judge_samples:
            jid = str(s["id"])
            if jid in new_by_id:
                judged_rows.append(new_by_id[jid])
            elif jid in existing:
                judged_rows.append(existing[jid])
            else:
                missing.append(jid)
        if missing:
            raise RuntimeError(f"judge merge failed (missing rows for {len(missing)} ids); delete {judged_path} and retry.")

        _info(f"judge done in {time.perf_counter() - t_j:.1f}s (rewriting merged file in prediction order)")
        tmp_j = judged_path.with_suffix(".jsonl.tmp")
        with tmp_j.open("w", encoding="utf-8", newline="\n") as jf:
            for jr in judged_rows:
                jf.write(json.dumps(jr, ensure_ascii=False) + "\n")
        tmp_j.replace(judged_path)

        summary_all = aggregate_paper_metrics(judged_rows)
        (out_dir / "metrics_summary_all.json").write_text(
            json.dumps(summary_all, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        by_method_metrics: dict[str, Any] = {}
        for mk in methods:
            prefixed = [r for r in judged_rows if str(r.get("sample_id", "")).startswith(mk + ":")]
            by_method_metrics[mk] = aggregate_paper_metrics(prefixed)

        (out_dir / "metrics_by_method.json").write_text(
            json.dumps(by_method_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Wrote judged_per_sample.jsonl, metrics_summary_all.json, metrics_by_method.json")
    else:
        print("Judge skipped: no successful predictions with non-empty answers.")


def _count_ok_err_in_predictions(pred_path: Path) -> tuple[int, int]:
    ok = err = 0
    if not pred_path.is_file():
        return 0, 0
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    ok += 1
                else:
                    err += 1
            except json.JSONDecodeError:
                err += 1
    return ok, err


def _info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_one_method(
    agent: VaseAgent,
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    method_key: str,
    max_rounds: int,
    verbose: bool,
    tool_mode_override: str | None,
    log_jsonl: bool,
    show_progress: bool,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    skip_line_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    runner = cfg["runner"]
    out: list[dict[str, Any]] = []
    skip = skip_line_indices or set()
    indexed: list[tuple[int, dict[str, Any]]] = [(i, rows[i]) for i in range(len(rows)) if i not in skip]

    if runner == "direct":
        _set_source_control_env(False)
        for i, row in _iter_indexed_samples(indexed, desc=f"{method_key} [direct]", enabled=show_progress):
            q = str(row.get("question") or "").strip()
            img = str(row.get("image") or row.get("image_url") or "").strip()
            ref = _reference_answer(row)
            sid = _sample_id(row, i)
            tt = _task_type(row)
            rec: dict[str, Any] = {
                "method": method_key,
                "sample_id": sid,
                "line_idx": i,
                "question": q,
                "reference_answer": ref,
                "task_type": tt,
                "image": img,
                "run_label": cfg.get("label"),
            }
            try:
                assistant = agent.run_direct(q, img, verbose=verbose, log_jsonl=log_jsonl)
                rec["model_answer"] = _assistant_text(assistant)
                rec["ok"] = True
            except Exception as e:
                rec["ok"] = False
                rec["error"] = str(e)
                rec["model_answer"] = ""
            out.append(rec)
            if on_sample is not None:
                on_sample(rec)
        return out

    # agent path
    rc = bool(cfg["response_control_gate"])
    ap = bool(cfg["apply_uncertain_preamble"])
    dsc = bool(cfg["disable_source_control"])
    _set_source_control_env(dsc)

    tm = tool_mode_override if tool_mode_override in ("openai", "xml") else None

    for i, row in _iter_indexed_samples(indexed, desc=f"{method_key} [agent]", enabled=show_progress):
        q = str(row.get("question") or "").strip()
        img = str(row.get("image") or row.get("image_url") or "").strip()
        ref = _reference_answer(row)
        sid = _sample_id(row, i)
        tt = _task_type(row)
        rec: dict[str, Any] = {
            "method": method_key,
            "sample_id": sid,
            "line_idx": i,
            "question": q,
            "reference_answer": ref,
            "task_type": tt,
            "image": img,
            "run_label": cfg.get("label"),
            "env_VASE_DISABLE_SOURCE_CONTROL": os.getenv("VASE_DISABLE_SOURCE_CONTROL", ""),
        }
        try:
            assistant = agent.run(
                q,
                img,
                max_tool_rounds=max_rounds,
                verbose=verbose,
                response_control_gate=rc,
                apply_uncertain_preamble=ap,
                log_jsonl=log_jsonl,
                tool_mode=tm,
            )
            rec["model_answer"] = _assistant_text(assistant)
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)
            rec["model_answer"] = ""
        out.append(rec)
        if on_sample is not None:
            on_sample(rec)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-style baseline / VaseAgent experiments")
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Evaluation JSONL (question + image + optional answer); omit when using --judge-now only.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions + summaries")
    p.add_argument(
        "--methods",
        type=str,
        default="direct,neither_control,source_control_only,response_control_only,vase_full",
        help="Comma-separated method keys (see METHODS dict in this script)",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Use only first N rows (0 = all)")
    p.add_argument("--start", type=int, default=0, help="Skip first START rows")
    p.add_argument("--max-tool-rounds", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--tool-mode",
        choices=["openai", "xml", ""],
        default="",
        help="Override VASE_TOOL_MODE for agent methods only (empty = use .env)",
    )
    p.add_argument("--judge", action="store_true", help="Run LLM-as-judge metrics after inference")
    p.add_argument(
        "--judge-now",
        action="store_true",
        help="Skip inference; run judge on existing predictions_all.jsonl under --out-dir only.",
    )
    p.add_argument("--judge-workers", type=int, default=1)
    p.add_argument(
        "--log-jsonl",
        action="store_true",
        help="Append per-run traces to agent_runs.jsonl (same as interactive agent_run logging)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm on stderr (inference phases and LLM judge batch).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue each predictions_<method>.jsonl: skip line_idx already present; append to predictions_all.jsonl.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Remove predictions_*.jsonl and predictions_all.jsonl in --out-dir before running (ignore resume).",
    )

    args = p.parse_args()

    if args.judge_now and args.fresh:
        raise SystemExit("--judge-now cannot be combined with --fresh")
    if not args.judge_now and args.eval_jsonl is None:
        raise SystemExit("--eval-jsonl is required unless --judge-now")

    if args.judge_now:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        combined_only = args.out_dir / "predictions_all.jsonl"
        if not combined_only.is_file():
            raise SystemExit("--judge-now requires an existing predictions_all.jsonl under --out-dir")
        methods_only = _resolve_methods_for_judge(args, combined_only)
        _info("")
        _info("== paper experiments (judge-only) ==")
        _info(f"out_dir: {args.out_dir.resolve()}")
        _info(f"methods:   {', '.join(methods_only)}")
        _info("")
        _run_llm_judge(
            args.out_dir,
            combined_only,
            methods_only,
            args.judge_workers,
            show_progress=not args.no_progress,
        )
        return

    assert args.eval_jsonl is not None
    rows = load_jsonl(args.eval_jsonl)
    if args.start:
        rows = rows[args.start :]
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = args.out_dir / "predictions_all.jsonl"
    if args.fresh:
        for p_jsonl in sorted(args.out_dir.glob("predictions_*.jsonl")):
            p_jsonl.unlink()
        if combined_path.is_file():
            combined_path.unlink()
        for judge_name in ("judged_per_sample.jsonl", "metrics_summary_all.json", "metrics_by_method.json"):
            jp = args.out_dir / judge_name
            if jp.is_file():
                jp.unlink()

    use_resume = bool(args.resume) and not args.fresh
    tool_mode_override = args.tool_mode if args.tool_mode else None

    raw_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods: list[str] = []
    seen: set[str] = set()
    for m in raw_methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown method: {m!r}. Known keys: {sorted(METHODS.keys())}")
        if m not in seen:
            methods.append(m)
            seen.add(m)

    if use_resume:
        man_path = args.out_dir / "manifest.json"
        if man_path.is_file():
            prev = json.loads(man_path.read_text(encoding="utf-8"))
            if prev.get("eval_jsonl") != str(args.eval_jsonl.resolve()):
                raise SystemExit(
                    "resume: manifest eval_jsonl mismatch — use same --eval-jsonl as the prior run or --fresh."
                )
            if int(prev.get("n_rows", -1)) != len(rows):
                raise SystemExit(
                    "resume: manifest n_rows mismatch — use the same --start/--max-samples as before or --fresh."
                )
            pm = prev.get("methods")
            if isinstance(pm, list) and set(pm) != set(methods):
                raise SystemExit(
                    "resume: manifest methods mismatch — use the same --methods as before or --fresh."
                )

    manifest = {
        "eval_jsonl": str(args.eval_jsonl.resolve()),
        "n_rows": len(rows),
        "methods": methods,
        "tool_mode_env": get_tool_mode(),
        "tool_mode_override": tool_mode_override or None,
        "agent_log_jsonl": bool(args.log_jsonl),
        "note": "Full baseline omits K-sample GRPO selection (not implemented in agent_run).",
        "resume": use_resume,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    show_progress = not args.no_progress
    n_methods = len(methods)
    _info("")
    _info("== paper experiments ==")
    _info(f"eval_jsonl: {args.eval_jsonl.resolve()}")
    _info(f"out_dir:    {args.out_dir.resolve()}")
    _info(f"samples:    {len(rows)}  (start={args.start}, max_samples={args.max_samples or 'all'})")
    _info(f"methods:    ({n_methods}) {', '.join(methods)}")
    _info(f"max_rounds: {args.max_tool_rounds}  tool_mode: {tool_mode_override or get_tool_mode()} (env)")
    _info(f"judge:      {'yes' if args.judge else 'no'}  log_jsonl: {'yes' if args.log_jsonl else 'no'}")
    _info(f"resume:     {'yes' if use_resume else 'no'}")
    _info("")

    t_run0 = time.perf_counter()
    agent = VaseAgent.from_env()

    def _flush_record(rec: dict[str, Any], pf: TextIO, combined_f: TextIO) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        pf.write(line)
        combined_f.write(line)
        pf.flush()
        combined_f.flush()

    combined_mode = "a" if (use_resume and combined_path.is_file()) else "w"
    with combined_path.open(combined_mode, encoding="utf-8", newline="\n") as combined_f:
        for mi, method_key in enumerate(methods, start=1):
            cfg = METHODS[method_key]
            _info(f"[{mi}/{n_methods}] {method_key} — {cfg.get('label', '')}")
            t0 = time.perf_counter()
            per_path = args.out_dir / f"predictions_{method_key}.jsonl"
            skip_done = _load_done_line_indices(per_path) if use_resume else set()
            if skip_done:
                _info(f"    resume: skip {len(skip_done)} samples already in {per_path.name}")
            file_mode = "a" if (use_resume and per_path.is_file()) else "w"
            with per_path.open(file_mode, encoding="utf-8", newline="\n") as pf:

                def _on_sample(rec: dict[str, Any]) -> None:
                    _flush_record(rec, pf, combined_f)

                run_one_method(
                    agent,
                    cfg,
                    rows,
                    method_key=method_key,
                    max_rounds=args.max_tool_rounds,
                    verbose=args.verbose,
                    tool_mode_override=tool_mode_override,
                    log_jsonl=args.log_jsonl,
                    show_progress=show_progress,
                    on_sample=_on_sample,
                    skip_line_indices=skip_done,
                )
            dt = time.perf_counter() - t0
            ok_n, bad_n = _count_ok_err_in_predictions(per_path)
            _info(f"    done in {dt:.1f}s  cumulative in file: ok={ok_n}  err={bad_n}")

    _info(f"inference total: {time.perf_counter() - t_run0:.1f}s")
    print(f"Wrote {combined_path} and per-method predictions_*.jsonl under {args.out_dir}")

    if args.judge:
        _run_llm_judge(
            args.out_dir,
            combined_path,
            methods,
            args.judge_workers,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    main()
