"""
Response-control: structured claim–evidence alignment, conflict/gap heuristics, and answer gating.

Uses the same ``evidence_pool`` schema as ``source_control``. No extra model calls by default;
optional ``llm_decompose`` hook can supply finer claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .source_control import _default_tokenize, _jaccard


def simple_decompose_query(query: str) -> list[dict[str, Any]]:
    """
    Split a question into coarse atomic claims for coverage checks.
    Claims are split on sentence-ending punctuation (., !, ?, ;, newlines).
    Each claim must be at least 2 characters.
    """
    q = (query or "").strip()
    if not q:
        return [{"id": "c0", "text": "answer the user question", "keywords": set()}]
    chunks = re.split(r"[。！？；;.\!\?\n]+", q)
    claims: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        t = ch.strip()
        if len(t) < 2:
            continue
        claims.append({"id": f"c{len(claims)}", "text": t, "keywords": _default_tokenize(t)})
    if not claims:
        claims.append({"id": "c0", "text": q, "keywords": _default_tokenize(q)})
    return claims


def _evidence_text(e: dict[str, Any]) -> str:
    return str(e.get("text") or e.get("title") or "")


def align_claims_lexical(
    claims: list[dict[str, Any]], evidence_pool: list[dict[str, Any]], support_threshold: float = 0.08
) -> list[dict[str, Any]]:
    """
    Align claims with evidence pool via keyword Jaccard overlap.
    Sources above ``support_threshold`` (default 0.08) count as supporting a claim.
    """
    rows: list[dict[str, Any]] = []
    for c in claims:
        kw = c.get("keywords") or _default_tokenize(c["text"])
        support: list[str] = []
        scores: dict[str, float] = {}
        for e in evidence_pool:
            etoks = _default_tokenize(_evidence_text(e))
            s = _jaccard(kw, etoks) if kw and etoks else 0.0
            scores[e["source_id"]] = round(s, 4)
            if s >= support_threshold:
                support.append(e["source_id"])
        rows.append(
            {
                "claim_id": c["id"],
                "claim_text": c["text"],
                "support_sources": support,
                "contradict_sources": [],
                "per_source_score": scores,
            }
        )
    return rows


def _pairwise_contradiction_strength(a: str, b: str) -> float:
    """Very light heuristic: strong negation + high token overlap suggests conflict."""
    if not a or not b:
        return 0.0
    ta, tb = _default_tokenize(a), _default_tokenize(b)
    j = _jaccard(ta, tb)
    if j < 0.25:
        return 0.0
    neg = ("不", "否", "非", "无", "没有", "不是", "未", "not ", "no ", "without ")
    na = any(x in a.lower() for x in neg)
    nb = any(x in b.lower() for x in neg)
    if na ^ nb:
        return min(1.0, j + 0.2)
    return 0.0


def detect_cross_source_conflicts(evidence_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Detect cross-source conflicts.
    Pairwise evidence texts with Jaccard >= 0.25 and asymmetric negation cues
    (English/Chinese heuristics) yield conflict strength; strength > 0.45 is recorded.
    """
    conflicts: list[dict[str, Any]] = []
    n = len(evidence_pool)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = evidence_pool[i], evidence_pool[j]
            sa, sb = _evidence_text(a), _evidence_text(b)
            strength = _pairwise_contradiction_strength(sa, sb)
            if strength > 0.45:
                conflicts.append(
                    {
                        "source_a": a["source_id"],
                        "source_b": b["source_id"],
                        "strength": round(strength, 3),
                    }
                )
    return conflicts


def compute_answer_confidence(
    matrix: list[dict[str, Any]],
    *,
    conflicts: list[dict[str, Any]],
    evidence_pool: list[dict[str, Any]],
) -> float:
    """Map claim coverage and consistency to a single 0–1 score (answer gate).

    Combines supported-claim ratio, consistency penalty, multi-source agreement,
    evidence-pool presence, and capped conflict penalties.
    """
    if not matrix:
        return 0.0
    covered = sum(1 for r in matrix if r["support_sources"])
    c_coverage = covered / len(matrix)
    # consistency: penalize claims with no support
    k_consistency = 1.0 - (len(matrix) - covered) / max(len(matrix) * 2, 1)
    # cross-source: at least two distinct sources for majority of supported claims
    multi = 0
    supp = 0
    for r in matrix:
        if r["support_sources"]:
            supp += 1
            if len(set(r["support_sources"])) >= 2:
                multi += 1
    o_agree = (multi / supp) if supp else 0.0
    m_modal = 1.0 if evidence_pool else 0.0
    conf_penalty = min(0.35, len(conflicts) * 0.12)
    raw = 0.4 * c_coverage + 0.3 * k_consistency + 0.2 * o_agree + 0.1 * m_modal
    return max(0.0, min(1.0, raw - conf_penalty))


@dataclass
class ResponseControlConfig:
    confidence_threshold: float = 0.52
    support_threshold: float = 0.08


class ResponseControl:
    def __init__(self, config: ResponseControlConfig | None = None):
        self.config = config or ResponseControlConfig()

    def run(
        self,
        *,
        query: str,
        evidence_pool: list[dict[str, Any]],
        assistant_draft: str | None = None,
        claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        claims = claims or simple_decompose_query(query)
        matrix = align_claims_lexical(claims, evidence_pool, support_threshold=self.config.support_threshold)
        conflicts = detect_cross_source_conflicts(evidence_pool)
        conf = compute_answer_confidence(matrix, conflicts=conflicts, evidence_pool=evidence_pool)
        mode = "answer" if conf >= self.config.confidence_threshold else "uncertain"
        missing = [r["claim_text"] for r in matrix if not r["support_sources"]]
        supported = [r["claim_text"] for r in matrix if r["support_sources"]]

        audit = {
            "mode": mode,
            "confidence": round(conf, 4),
            "claim_evidence_matrix": matrix,
            "cross_source_conflicts": conflicts,
            "supported_claims": supported,
            "unsupported_claims": missing,
        }
        if assistant_draft:
            audit["assistant_draft_chars"] = len(assistant_draft)
        return audit

    def format_uncertain_preamble(self, audit: dict[str, Any]) -> str:
        """User-facing conservative prefix when mode is ``uncertain``."""
        supported = audit.get("supported_claims") or []
        missing = audit.get("unsupported_claims") or []
        conflicts = audit.get("cross_source_conflicts") or []
        lines = [
            "[Evidence audit] Retrieved evidence is insufficient to support all sub-questions with equal confidence.",
            f"Overall confidence ~ {audit.get('confidence', 0):.2f} (below the answer gate threshold).",
        ]
        if supported:
            lines.append("Partially supported aspects: " + "; ".join(supported[:6]))
        if missing:
            lines.append("Under-supported or misaligned sub-questions: " + "; ".join(missing[:6]))
        if conflicts:
            lines.append(
                "Potential conflicts were detected across sources; more authoritative cross-checking is needed."
            )
        lines.append("The answer below is conservative and grounded only in aligned evidence.")
        return "\n".join(lines)


# Optional async hook type for LLM-based claim decomposition
LLMDecomposeFn = Callable[[str], Awaitable[list[dict[str, Any]]]]
