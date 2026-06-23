"""
LLM-as-judge metrics aligned with the paper (§Evaluation Tasks and Metrics):

- Answer Accuracy (correctness vs reference / acceptable answers)
- Hallucination rate (unsupported or fabricated claims)
- Link validity / citation plausibility (external URLs or named scholarly sources)
- Neutrality score (0–5) under ambiguity
- Knowledge grounding (optional; especially V+K)

Uses :func:`llm_env.get_eval_judge_config` (``EVAL_JUDGE_*`` overrides).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable, Sequence

from openai import OpenAI

# package import when run as ``python -m metrics.llm_judge`` from vase-agent/
_VASE_ROOT = Path(__file__).resolve().parents[1]
if str(_VASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VASE_ROOT))

from llm_env import get_eval_judge_config


JUDGE_SYSTEM = """You are an expert evaluator for a museum vision–language QA system (ancient Greek pottery / cultural heritage).

Your judgments must be conservative: prefer marking hallucination when claims cannot be verified from the given references or are likely fabricated.

Output ONLY one JSON object, no markdown fences, no text before or after."""


def _task_type_description(task_type: str) -> str:
    t = (task_type or "unknown").strip().lower()
    if t in ("v_only", "v-only", "visual", "visual_only"):
        return "Visual-only (V-Only): the reference should be checkable from visual appearance; correctness focuses on visual facts."
    if t in ("v_plus_k", "v+k", "vk", "visual_knowledge", "visual+knowledge"):
        return "Visual + Knowledge (V+K): correctness may require combining visuals with metadata/knowledge; grounding matters."
    if t in ("amb", "ambiguous", "ambiguity"):
        return "Ambiguous (Amb.): multiple interpretations may be valid; reward neutrality and hedging; correctness allows any acceptable reference answer."
    return "Task type unknown; use general museum QA criteria."


def build_judge_user_prompt(
    *,
    question: str,
    model_answer: str,
    reference_answer: str | Sequence[str] | None,
    task_type: str = "",
    extra_context: str | None = None,
) -> str:
    ref = reference_answer
    if ref is None:
        ref_text = "(none — judge factual consistency with common scholarly consensus only; be conservative on hallucination)."
    elif isinstance(ref, (list, tuple)):
        ref_text = "Accept ANY of the following as correct:\n" + "\n".join(f"- {r}" for r in ref)
    else:
        ref_text = str(ref)

    ctx = (extra_context or "").strip()
    ctx_block = f"\nAdditional context (metadata / evidence snippets, optional):\n{ctx}\n" if ctx else ""

    return f"""{_task_type_description(task_type)}

Question:
{question.strip()}

Reference / acceptable answer(s):
{ref_text}
{ctx_block}
Model answer (to evaluate):
{model_answer.strip()}

Return a JSON object with EXACTLY these keys:
- "correctness": integer 0 or 1. 1 if the model answer is compatible with the reference OR any acceptable interpretation for ambiguous tasks; 0 if clearly wrong.
- "hallucination": integer 0 or 1. 1 if the answer contains fabricated facts, invented citations/URLs, or claims not reasonably supported by reference/context; 0 otherwise.
- "neutrality": integer from 1 to 5. Higher if the answer appropriately acknowledges uncertainty or multiple interpretations when needed (especially Ambiguous tasks). Lower if overconfident when evidence is weak.
- "grounding": integer 0 or 1. 1 if it appropriately uses plausible domain knowledge / provided metadata when required (especially V+K); 0 if it ignores known constraints or misuses facts.
- "has_external_citation": boolean. True if the answer cites specific external URLs, DOIs, or clearly named institutional databases (LIMC, Beazley, museum accession pages, etc.).
- "citation_plausible": boolean or null. If has_external_citation is false, use null. If true, true only if citations look like real scholarly/museum sources (not obvious placeholders); false if fabricated or nonsensical links.

Optional key "notes": one short English sentence explaining your decision (for debugging)."""


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_first_json_object(text: str | None) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # brace slice fallback
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _to_bit(x: Any, *, default: int = 0) -> int:
    try:
        return 1 if int(float(x)) == 1 else 0
    except (TypeError, ValueError):
        return default


def _normalize_labels(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["correctness"] = _to_bit(raw.get("correctness"))

    out["hallucination"] = _to_bit(raw.get("hallucination"))

    n = raw.get("neutrality")
    try:
        ni = int(float(n))
    except (TypeError, ValueError):
        ni = 3
    out["neutrality"] = max(1, min(5, ni))

    out["grounding"] = _to_bit(raw.get("grounding"))

    he = raw.get("has_external_citation")
    out["has_external_citation"] = bool(he)

    cp = raw.get("citation_plausible")
    if not out["has_external_citation"]:
        out["citation_plausible"] = None
    else:
        out["citation_plausible"] = None if cp is None else bool(cp)

    if "notes" in raw and raw["notes"] is not None:
        out["notes"] = str(raw["notes"])[:500]
    return out


@dataclass
class PaperMetricLabels:
    """Normalized labels from one judge call."""

    correctness: int
    hallucination: int
    neutrality: int
    grounding: int
    has_external_citation: bool
    citation_plausible: bool | None
    notes: str | None = None


def evaluate_one(
    *,
    question: str,
    model_answer: str,
    reference_answer: str | Sequence[str] | None = None,
    task_type: str = "",
    extra_context: str | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> tuple[PaperMetricLabels, dict[str, Any]]:
    """
    Run one LLM judge call. Returns (:class:`PaperMetricLabels`, raw metadata).

    ``client`` / ``model`` default to :func:`get_eval_judge_config` when omitted.
    """
    cfg = get_eval_judge_config()
    mdl = model or cfg["model"]
    cli = client or OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], max_retries=2)

    user_msg = build_judge_user_prompt(
        question=question,
        model_answer=model_answer,
        reference_answer=reference_answer,
        task_type=task_type,
        extra_context=extra_context,
    )
    resp = cli.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        timeout=120,
    )
    ch0 = (resp.choices or [None])[0]
    msg = getattr(ch0, "message", None) if ch0 else None
    raw_text = ""
    if msg is not None:
        c = getattr(msg, "content", None)
        if isinstance(c, str):
            raw_text = c
        elif isinstance(c, list):
            raw_text = "\n".join(
                str(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text"
            )

    parsed = extract_first_json_object(raw_text)
    if not parsed:
        raise RuntimeError(f"judge returned non-JSON or empty: {raw_text[:800]!r}")

    norm = _normalize_labels(parsed)
    labels = PaperMetricLabels(
        correctness=norm["correctness"],
        hallucination=norm["hallucination"],
        neutrality=norm["neutrality"],
        grounding=norm["grounding"],
        has_external_citation=norm["has_external_citation"],
        citation_plausible=norm["citation_plausible"],
        notes=norm.get("notes"),
    )
    meta = {
        "judge_model": mdl,
        "raw_assistant_text": raw_text[:8000],
        "parsed": norm,
    }
    return labels, meta


@dataclass
class EvalSample:
    """One row for batch evaluation."""

    question: str
    model_answer: str
    reference_answer: str | Sequence[str] | None = None
    task_type: str = ""
    extra_context: str | None = None
    sample_id: str | None = None


def evaluate_batch(
    samples: Sequence[EvalSample | dict[str, Any]],
    *,
    max_workers: int = 1,
    temperature: float = 0.0,
    progress: bool = False,
    progress_desc: str = "Judge",
    on_completed: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate many samples. Each dict may use keys:
    ``question``, ``model_answer``, ``reference_answer`` | ``ground_truth``,
    ``task_type``, ``extra_context``, ``id`` | ``sample_id``.

    If ``on_completed`` is set, it is invoked with each result row as soon as it is ready
    (thread-safe callbacks should use a lock). Useful for incremental persistence.
    """
    normalized: list[EvalSample] = []
    for s in samples:
        if isinstance(s, EvalSample):
            normalized.append(s)
            continue
        d = dict(s)
        ref = d.get("reference_answer")
        if ref is None:
            ref = d.get("ground_truth")
        normalized.append(
            EvalSample(
                question=str(d.get("question", "") or ""),
                model_answer=str(d.get("model_answer", d.get("answer", "")) or ""),
                reference_answer=ref,
                task_type=str(d.get("task_type", "") or ""),
                extra_context=d.get("extra_context"),
                sample_id=str(d["id"]) if d.get("id") is not None else d.get("sample_id"),
            )
        )

    def _one(es: EvalSample) -> dict[str, Any]:
        labels, meta = evaluate_one(
            question=es.question,
            model_answer=es.model_answer,
            reference_answer=es.reference_answer,
            task_type=es.task_type,
            extra_context=es.extra_context,
            temperature=temperature,
        )
        row: dict[str, Any] = {
            "sample_id": es.sample_id,
            "task_type": es.task_type or None,
            "labels": {
                "correctness": labels.correctness,
                "hallucination": labels.hallucination,
                "neutrality": labels.neutrality,
                "grounding": labels.grounding,
                "has_external_citation": labels.has_external_citation,
                "citation_plausible": labels.citation_plausible,
                "notes": labels.notes,
            },
            "judge_model": meta["judge_model"],
        }
        return row

    if max_workers <= 1:
        it_es = normalized
        if progress:
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]

                it_es = tqdm(
                    normalized,
                    total=len(normalized),
                    desc=progress_desc,
                    unit="sample",
                    file=sys.stderr,
                    leave=True,
                )
            except ImportError:
                pass
        out_seq: list[dict[str, Any]] = []
        for es in it_es:
            row = _one(es)
            if on_completed is not None:
                on_completed(row)
            out_seq.append(row)
        return out_seq

    results: list[dict[str, Any] | None] = [None] * len(normalized)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx: dict[Any, int] = {}
        for i, es in enumerate(normalized):
            fut = ex.submit(_one, es)
            fut_to_idx[fut] = i

        if progress:
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]

                pbar = tqdm(
                    total=len(normalized),
                    desc=progress_desc,
                    unit="sample",
                    file=sys.stderr,
                    leave=True,
                )
            except ImportError:
                pbar = None
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                row = fut.result()
                results[i] = row
                if on_completed is not None:
                    on_completed(row)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        else:
            for fut, i in fut_to_idx.items():
                row = fut.result()
                results[i] = row
                if on_completed is not None:
                    on_completed(row)

    return results


def aggregate_paper_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate judge rows (each with ``labels`` dict) into paper-style percentages.

    Returns keys including ``answer_accuracy_pct``, ``hallucination_rate_pct``,
    ``neutrality_mean``, ``grounding_rate_pct``, ``link_validity_pct`` (among
    answers with ``has_external_citation``), and counts.
    """
    rows_list = list(rows)
    n = len(rows_list)
    if n == 0:
        return {"n": 0}

    acc = []
    hall = []
    neu = []
    grd = []
    cite_subset_plausible: list[int] = []

    by_task: dict[str, list[dict[str, Any]]] = {}

    for row in rows_list:
        lab = row.get("labels") or {}
        tt = (row.get("task_type") or "unknown").strip() or "unknown"
        by_task.setdefault(tt, []).append(lab)

        acc.append(int(lab.get("correctness", 0)))
        hall.append(int(lab.get("hallucination", 0)))
        neu.append(float(lab.get("neutrality", 3)))
        grd.append(int(lab.get("grounding", 0)))
        if lab.get("has_external_citation"):
            cp = lab.get("citation_plausible")
            if cp is True:
                cite_subset_plausible.append(1)
            elif cp is False:
                cite_subset_plausible.append(0)

    def _pct(xs: list[int]) -> float:
        return round(100.0 * sum(xs) / len(xs), 2) if xs else 0.0

    link_valid = (
        round(100.0 * sum(cite_subset_plausible) / len(cite_subset_plausible), 2)
        if cite_subset_plausible
        else None
    )

    summary: dict[str, Any] = {
        "n": n,
        "answer_accuracy_pct": _pct(acc),
        "hallucination_rate_pct": _pct(hall),
        "neutrality_mean": round(sum(neu) / len(neu), 3) if neu else 0.0,
        "grounding_rate_pct": _pct(grd),
        "n_with_external_citation": len(cite_subset_plausible),
        "link_validity_pct": link_valid,
    }

    per_task: dict[str, Any] = {}
    for tt, labs in by_task.items():
        a = [int(x.get("correctness", 0)) for x in labs]
        h = [int(x.get("hallucination", 0)) for x in labs]
        nm = [float(x.get("neutrality", 3)) for x in labs]
        per_task[tt] = {
            "n": len(labs),
            "answer_accuracy_pct": _pct(a),
            "hallucination_rate_pct": _pct(h),
            "neutrality_mean": round(sum(nm) / len(nm), 3) if nm else 0.0,
        }
    summary["by_task_type"] = per_task
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LLM-as-judge metrics (paper-aligned)")
    p.add_argument("--input", type=Path, help="JSONL with question, model_answer, reference_answer (optional)")
    p.add_argument("--output", type=Path, help="Write per-row judge JSONL")
    p.add_argument("--aggregate-out", type=Path, help="Write aggregate summary JSON")
    p.add_argument("--workers", type=int, default=1, help="Parallel judge calls")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)

    if args.input:
        samples = _load_jsonl(args.input)
        rows = evaluate_batch(samples, max_workers=max(1, args.workers), temperature=args.temperature)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        agg = aggregate_paper_metrics(rows)
        print(json.dumps(agg, ensure_ascii=False, indent=2))
        if args.aggregate_out:
            args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)
            args.aggregate_out.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        p.print_help()


if __name__ == "__main__":
    _main()
