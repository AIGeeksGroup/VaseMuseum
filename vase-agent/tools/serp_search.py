# Serp text search (Google Light)
# Docs: https://serpapi.com/integrations/python
# https://serpapi.com/google-light-api

import os
from collections.abc import Mapping
from typing import Any

import dotenv
import serpapi

import serp_cache

dotenv.load_dotenv()

SERP_TEXT_SEARCH_KEY = os.getenv("SERP_TEXT_SEARCH_KEY")


def _json_endpoint(serp: Any) -> str:
    # SerpResults is a UserDict, not a plain dict; use Mapping
    if serp is None or isinstance(serp, (str, bytes)) or not isinstance(serp, Mapping):
        return ""
    meta = serp.get("search_metadata") or {}
    if not isinstance(meta, Mapping):
        return ""
    u = meta.get("json_endpoint")
    return str(u).strip() if u else ""


def _compact_answer_box(raw: Any) -> dict[str, Any] | None:
    """Normalize Google answer_box payloads into a stable small dict for the model."""
    if not raw or not isinstance(raw, dict):
        return None

    out: dict[str, Any] = {}
    for key in (
        "type",
        "title",
        "link",
        "displayed_link",
        "snippet",
        "result",
        "answer",
        "snippet_highlighted_words",
        "thumbnail",
    ):
        if key in raw and raw[key] not in (None, "", []):
            out[key] = raw[key]

    # Some answer_box types include lists or tables
    if raw.get("list"):
        out["list"] = raw["list"]
    if raw.get("table"):
        out["table"] = raw["table"]

    return out or None


def _organic_entries(items: list[Any], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for it in (items or [])[:limit]:
        if not isinstance(it, dict):
            continue
        pos = int(it.get("position") or len(rows) + 1)
        rows.append(
            {
                "rank": pos,
                "title": str(it.get("title") or ""),
                "snippet": str(it.get("snippet") or ""),
                "url": str(it.get("link") or ""),
                "displayed_link": str(it.get("displayed_link") or ""),
                "date": str(it.get("date") or ""),
            }
        )
    return rows


class TextSearchTool:
    def __init__(self):
        self.api_key = SERP_TEXT_SEARCH_KEY

    def run(self, queries: list[str], *, num: int = 10):
        """
        Agent tool:
        Text search via SerpAPI google_light. Each query returns up to ``num`` organic_results;
        if answer_box is present it is included for reference (cross-check with organic hits).
        Successful results are cached on disk (default vase-agent/tools/.serp_cache; override with SERP_CACHE_DIR).
        """
        qs = [str(q).strip() for q in queries if str(q).strip()]

        if not self.api_key:
            return {
                "ok": False,
                "tool": "text_search",
                "error": "missing environment variable SERP_TEXT_SEARCH_KEY",
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }

        if not qs:
            return {
                "ok": False,
                "tool": "text_search",
                "error": "queries is empty",
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }

        try:
            client = serpapi.Client(api_key=self.api_key)
            per_query: list[dict[str, Any]] = []
            total_snippets = 0

            for q in qs:
                cpath = serp_cache.entry_path("google_light", q, str(num))
                cached = serp_cache.load_entry(cpath)
                if cached is not None:
                    per_query.append({**cached, "from_cache": True})
                    total_snippets += len(cached.get("organic") or [])
                    continue

                serp = client.search(
                    {
                        "engine": "google_light",
                        "q": q,
                        "num": num,
                    }
                )
                if isinstance(serp, str):
                    entry = {
                        "query": q,
                        "json_endpoint": "",
                        "error": serp[:2000],
                        "answer_box": None,
                        "organic": [],
                    }
                    per_query.append(entry)
                    continue

                err = serp.get("error")
                if err:
                    entry = {
                        "query": q,
                        "json_endpoint": _json_endpoint(serp),
                        "error": str(err),
                        "answer_box": None,
                        "organic": [],
                    }
                    serp_cache.save_entry(cpath, entry)
                    per_query.append(entry)
                    continue

                organic_raw = serp.get("organic_results") or []
                organic = _organic_entries(organic_raw, num)
                total_snippets += len(organic)

                entry = {
                    "query": q,
                    "json_endpoint": _json_endpoint(serp),
                    "answer_box": _compact_answer_box(serp.get("answer_box")),
                    "organic": organic,
                }
                # if not entry.get("error"):
                #     serp_cache.save_entry(cpath, entry)
                serp_cache.save_entry(cpath, entry)
                per_query.append(entry)

            ok = any(
                (entry.get("organic") or entry.get("answer_box"))
                for entry in per_query
                if not entry.get("error")
            )

            return {
                "ok": ok,
                "tool": "text_search",
                "queries_echo": qs,
                "results": per_query,
                "total_snippets": total_snippets,
            }

        except Exception as e:
            return {
                "ok": False,
                "tool": "text_search",
                "error": str(e),
                "queries_echo": qs,
                "results": [],
                "total_snippets": 0,
            }


if __name__ == "__main__":
    import json
    tool = TextSearchTool()
    result = tool.run([
        "what's the period of red-figure?",
        "what's the period of black-figure?"
        ])
    print(json.dumps(result, indent=4))
    with open("txt-eg-return.json", "w") as f:
        json.dump(result, f, indent=4)
    
