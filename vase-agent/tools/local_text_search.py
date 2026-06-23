# Local museum KB text search (LIMC + Beazley descriptions)
#
# Output format intentionally mirrors `serp_search.TextSearchTool.run`:
# { ok, tool, queries_echo, results: [{query, organic:[{rank,title,snippet,url,displayed_link,date}]}], total_snippets }

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retriever.local_llm import embed_texts
from retriever.pipeline import load_index

import dotenv
dotenv.load_dotenv()

LOCAL_KB_INDEX_DIR = os.getenv("LOCAL_KB_INDEX_DIR")

def _first(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _truncate(s: str, n: int = 320) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


class LocalTextSearchTool:
    """
    Agent tool:
    Query a local retriever index built from LIMC.csv + descriptions.csv.

    Env:
      - LOCAL_KB_INDEX_DIR: path to index build dir (default: retriever_runs/structured_full)
      - LOCAL_KB_TOPK: default k per query (default: 10)
      - EMBEDDING_MODEL: embedding model name/id for OpenAI-compatible /v1 (default: text-embedding-3-large)
    """

    def __init__(self):
        repo_root = Path(__file__).resolve().parents[2]
        default_idx = repo_root / "retriever_runs" / "structured_full"
        self.index_dir = Path(_first("LOCAL_KB_INDEX_DIR", default=str(default_idx))).expanduser()
        self.default_k = int(_first("LOCAL_KB_TOPK", default="10") or "10")
        self.embedding_model = _first("EMBEDDING_MODEL", default="text-embedding-3-large")
        self._index = None

    def _get_index(self):
        if self._index is None:
            self._index = load_index(str(self.index_dir))
        return self._index

    def run(self, queries: list[str], *, num: int | None = None):
        qs = [str(q).strip() for q in queries if str(q).strip()]
        k = int(num if (num is not None and int(num) > 0) else self.default_k)

        if not qs:
            return {
                "ok": False,
                "tool": "kb_text_search",
                "error": "queries is empty",
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }

        if not self.index_dir.exists():
            return {
                "ok": False,
                "tool": "kb_text_search",
                "error": f"LOCAL_KB_INDEX_DIR does not exist: {self.index_dir}",
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }

        try:
            idx = self._get_index()
            per_query: list[dict[str, Any]] = []
            total = 0

            is_embedding = getattr(idx, "kind", "") == "embedding"
            for q in qs[:8]:
                if is_embedding:
                    q_emb = embed_texts(texts=[q], model=self.embedding_model)[0]
                    hits = idx.search(q_emb, k=k)
                else:
                    hits = idx.search(q, k=k)

                organic: list[dict[str, Any]] = []
                for i, h in enumerate(hits, start=1):
                    title = h.title or h.doc_id
                    snippet = _truncate(h.text or "")
                    url = h.uri or ""
                    displayed = str(h.source)
                    organic.append(
                        {
                            "rank": i,
                            "title": title,
                            "snippet": snippet,
                            "url": url,
                            "displayed_link": displayed,
                            "date": "",
                            "score": float(h.score),
                            "doc_id": h.doc_id,
                            "meta": h.meta,
                        }
                    )

                total += len(organic)
                per_query.append(
                    {
                        "query": q,
                        "index_dir": str(self.index_dir),
                        "index_kind": getattr(idx, "kind", "unknown"),
                        "organic": organic,
                    }
                )

            ok = any(block.get("organic") for block in per_query)
            return {
                "ok": ok,
                "tool": "kb_text_search",
                "queries_echo": qs,
                "results": per_query,
                "total_snippets": total,
            }
        except Exception as e:
            return {
                "ok": False,
                "tool": "kb_text_search",
                "error": str(e),
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }


if __name__ == "__main__":
    import json

    tool = LocalTextSearchTool()
    print(json.dumps(tool.run(["Triptolemos winged chariot Demeter Persephone"], num=5), ensure_ascii=False, indent=2))

