"""
Source-control: turn SERP text/image hits into a filtered, deduplicated, structured evidence pool.

Pipeline: hard filter (accessibility proxy + sufficiency) → score (relevance, reliability) →
MMR-style diversity selection → structured ``evidence_pool`` for response-control / GRPO metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from .context import get_user_query

TokenizeFn = Callable[[str], set[str]]


def _default_tokenize(text: str) -> set[str]:
    """Tokenize text; keep letters, digits, CJK characters, and underscores."""
    t = (text or "").lower()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", t)
    return {p for p in parts if len(p) > 1}


def _host(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        return h.lower().lstrip("www.")
    except Exception:
        return ""


def _accessibility_proxy(url: str) -> float:
    """Heuristic 0–1 score without network I/O (optional HEAD checks can be added later)."""
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return 0.0
    if len(u) > 2048:
        return 0.35
    low = u.lower()
    for bad in ("javascript:", "data:", "file:", "blob:"):
        if bad in low:
            return 0.0
    return 0.95


def _sufficiency(title: str, snippet: str, *, min_snippet: int, min_title: int) -> float:
    """Score title/snippet length relative to thresholds (0–1)."""
    sn = (snippet or "").strip()
    ti = (title or "").strip()
    if len(ti) < min_title:
        return 0.2
    if len(sn) < min_snippet:
        return 0.35 + 0.25 * min(1.0, len(sn) / max(min_snippet, 1))
    return min(1.0, 0.5 + 0.5 * min(1.0, len(sn) / 400))


def _domain_reliability(host: str) -> float:
    """Heuristic domain reliability from TLD / known scholarly hosts."""
    if not host:
        return 0.35
    if any(host.endswith(s) for s in (".gov", ".edu", ".ac.uk", ".ac.jp")):
        return 0.92
    if "arxiv.org" in host or "nature.com" in host or "science.org" in host:
        return 0.9
    if "wikipedia.org" in host or "wikimedia.org" in host:
        return 0.78
    if host.endswith((".org", ".museum")):
        return 0.65
    if host.endswith(".com"):
        return 0.55
    return 0.5


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class SourceControlConfig:
    """Tuning for filtering and MMR selection."""

    max_pool_size: int = 5
    mmr_lambda: float = 0.72  # MMR trade-off; higher favors relevance
    min_snippet_chars: int = 18
    min_title_chars: int = 2
    min_composite_to_keep: float = 0.22
    # composite pre-diversity: 0.35*R + 0.25*Q + 0.20*A + 0.20*D
    w_relevance: float = 0.35
    w_quality: float = 0.25
    w_access_suff: float = 0.20
    w_diversity: float = 0.20
    tokenize: TokenizeFn = _default_tokenize


def flatten_text_search_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ``TextSearchTool.run`` JSON into row dicts."""
    rows: list[dict[str, Any]] = []
    for block in payload.get("results") or []:
        q = str(block.get("query") or "")
        for org in block.get("organic") or []:
            if not isinstance(org, dict):
                continue
            rows.append(
                {
                    "modality": "text",
                    "query_hint": q,
                    "title": str(org.get("title") or ""),
                    "snippet": str(org.get("snippet") or ""),
                    "url": str(org.get("url") or ""),
                    "displayed_link": str(org.get("displayed_link") or ""),
                }
            )
    return rows


def flatten_image_search_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ``ImageSearchTool.run`` JSON into row dicts."""
    rows: list[dict[str, Any]] = []
    for block in payload.get("results") or []:
        q_img = str(block.get("query_image_url") or "")
        for vm in block.get("visual_matches") or []:
            if not isinstance(vm, dict):
                continue
            url = str(vm.get("source_page_url") or "")
            rows.append(
                {
                    "modality": "image_hit",
                    "query_image_url": q_img,
                    "title": str(vm.get("title") or ""),
                    "snippet": str(vm.get("description") or ""),
                    "url": url,
                    "source": str(vm.get("source") or ""),
                    "image_thumb_url": str(vm.get("image_thumb_url") or ""),
                }
            )
    return rows


class SourceControl:
    """Filter, score, and select a compact evidence pool from search hits."""

    def __init__(self, config: SourceControlConfig | None = None):
        self.config = config or SourceControlConfig()

    def _hard_filter(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.config
        out: list[dict[str, Any]] = []
        for h in hits:
            url = h.get("url") or ""
            a = _accessibility_proxy(str(url))
            title = str(h.get("title") or "")
            snippet = str(h.get("snippet") or "")
            c = _sufficiency(title, snippet, min_snippet=cfg.min_snippet_chars, min_title=cfg.min_title_chars)
            s_valid = 0.55 * a + 0.45 * c
            if s_valid < 0.4:
                continue
            if a < 0.5:
                continue
            h = dict(h)
            h["_access"] = a
            h["_suff"] = c
            h["_s_valid"] = s_valid
            out.append(h)
        return out

    def _score_hit(self, hit: dict[str, Any], query_tokens: set[str], tokenize: TokenizeFn) -> dict[str, Any]:
        title = str(hit.get("title") or "")
        snippet = str(hit.get("snippet") or "")
        text = f"{title}\n{snippet}"
        doc_tokens = tokenize(text)
        rel = _jaccard(query_tokens, doc_tokens) if query_tokens else 0.25

        url = str(hit.get("url") or "")
        host = _host(url)
        reliab = _domain_reliability(host)
        quality = 0.55 * reliab + 0.45 * float(hit.get("_suff", 0.5))
        access = float(hit.get("_access", 0.5))
        suff = float(hit.get("_suff", 0.5))
        a_combo = 0.5 * access + 0.5 * suff

        comp = (
            self.config.w_relevance * rel
            + self.config.w_quality * quality
            + self.config.w_access_suff * a_combo
        )
        hit = dict(hit)
        hit["_tokens"] = doc_tokens
        hit["_scores"] = {
            "accessibility": round(access, 4),
            "sufficiency": round(suff, 4),
            "relevance": round(rel, 4),
            "reliability": round(reliab, 4),
            "quality": round(quality, 4),
            "composite_pre_diversity": round(comp, 4),
        }
        return hit

    def _mmr_select(self, scored: list[dict[str, Any]], query_tokens: set[str], tokenize: TokenizeFn) -> list[dict[str, Any]]:
        """Maximum Marginal Relevance selection balancing relevance and diversity."""
        cfg = self.config
        lam = cfg.mmr_lambda
        k = min(cfg.max_pool_size, len(scored))
        if k <= 0:
            return []

        # seed by best composite + reliability
        def prelim_key(h: dict[str, Any]) -> float:
            sc = h["_scores"]
            return sc["composite_pre_diversity"] + 0.05 * sc["reliability"]

        pool = sorted(scored, key=prelim_key, reverse=True)
        selected: list[dict[str, Any]] = []
        domains_seen: set[str] = set()

        def diversity_bonus(h: dict[str, Any]) -> float:
            dom = _host(str(h.get("url") or ""))
            bonus = 0.15 if dom and dom not in domains_seen else 0.0
            return bonus

        while len(selected) < k and pool:
            best_i = -1
            best_mmr = -1e9
            for i, h in enumerate(pool):
                if h in selected:
                    continue
                rel = h["_scores"]["relevance"]
                max_sim = 0.0
                if selected:
                    t0 = h["_tokens"]
                    for s in selected:
                        max_sim = max(max_sim, _jaccard(t0, s["_tokens"]))
                div_marginal = 1.0 - max_sim
                d_bonus = diversity_bonus(h)
                mmr = lam * rel + (1.0 - lam) * div_marginal + d_bonus
                # fold in reliability lightly for tie-break inside MMR
                mmr += 0.04 * h["_scores"]["reliability"]
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_i = i
            if best_i < 0:
                break
            pick = pool.pop(best_i)
            dom = _host(str(pick.get("url") or ""))
            if dom:
                domains_seen.add(dom)
            selected.append(pick)

        # final composite including diversity contribution (approximate)
        for rank, h in enumerate(selected):
            div_contrib = 0.2 + 0.8 * (1.0 - rank / max(len(selected), 1))
            sc = h["_scores"]
            final_c = (
                cfg.w_relevance * sc["relevance"]
                + cfg.w_quality * sc["quality"]
                + cfg.w_access_suff * (0.5 * sc["accessibility"] + 0.5 * sc["sufficiency"])
                + cfg.w_diversity * div_contrib
            )
            sc["source_composite"] = round(float(final_c), 4)
        return selected

    def build_evidence_pool(self, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        for i, h in enumerate(selected):
            url = str(h.get("url") or "")
            modality = str(h.get("modality") or "text")
            text_for_model = f"{h.get('title', '')}\n{h.get('snippet', '')}".strip()
            pool.append(
                {
                    "source_id": f"S{i}",
                    "url": url,
                    "domain": _host(url),
                    "modality": modality,
                    "title": str(h.get("title") or ""),
                    "text": text_for_model,
                    "scores": h.get("_scores", {}),
                    "meta": {
                        k: v
                        for k, v in h.items()
                        if k
                        in (
                            "query_hint",
                            "query_image_url",
                            "image_thumb_url",
                            "source",
                            "displayed_link",
                        )
                        and v
                    },
                }
            )
        return pool

    def run(
        self,
        *,
        query: str | None = None,
        hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Args:
            query: Override relevance query; if None, uses ``get_user_query()`` then falls back to
                concatenated ``query_hint`` from hits.
            hits: Flattened rows from ``flatten_*`` or compatible dicts (title, snippet, url, modality).

        Returns:
            (evidence_pool, debug_stats)
        """
        q = (query or get_user_query() or "").strip()
        if not q:
            q = " ".join(str(h.get("query_hint") or "") for h in hits[:3])
        tokenize = self.config.tokenize
        query_tokens = tokenize(q)

        filtered = self._hard_filter(hits)
        scored = [self._score_hit(h, query_tokens, tokenize) for h in filtered]
        scored = [h for h in scored if h["_scores"]["composite_pre_diversity"] >= self.config.min_composite_to_keep]
        selected = self._mmr_select(scored, query_tokens, tokenize)
        pool = self.build_evidence_pool(selected)
        stats = {
            "n_input": len(hits),
            "n_after_filter": len(filtered),
            "n_after_score_cut": len(scored),
            "n_selected": len(selected),
            "query_used": q[:500],
        }
        return pool, stats


def attach_evidence_pool_to_search_payload(
    payload: dict[str, Any],
    *,
    tool: str,
    query: str | None = None,
    config: SourceControlConfig | None = None,
) -> dict[str, Any]:
    """
    Mutate a copy of a search tool JSON payload to add ``evidence_pool`` and ``source_control_stats``.

    ``tool`` is ``"text_search"`` or ``"image_search"``.
    """
    out = dict(payload)
    sc = SourceControl(config)
    if tool == "text_search":
        hits = flatten_text_search_results(out)
    elif tool == "image_search":
        hits = flatten_image_search_results(out)
    else:
        out["evidence_pool"] = []
        out["source_control_stats"] = {"error": "unknown_tool", "tool": tool}
        return out

    pool, stats = sc.run(query=query, hits=hits)
    out["evidence_pool"] = pool
    out["source_control_stats"] = stats
    return out
