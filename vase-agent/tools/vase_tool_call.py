"""
OpenAI-compatible tool definitions and execution for the vase agent.

text_search: tools/serp_search.py (requires SERP_TEXT_SEARCH_KEY)
image_search: tools/serp_image.py (requires SERP_IMG_SEARCH_KEY)
visit: tools/visit_extract.py (Playwright + Readability)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = Path(__file__).resolve().parent
for p in (_AGENT_ROOT, _TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.source_control import attach_evidence_pool_to_search_payload
from local_text_search import LocalTextSearchTool
from serp_image import ImageSearchTool
from serp_search import TextSearchTool
from visit_extract import VisitTool

def _disable_source_control() -> bool:
    """Read env at call time so experiments can toggle without reloading the module."""
    return os.getenv("VASE_DISABLE_SOURCE_CONTROL", "").strip().lower() in ("1", "true", "yes")


_image_search_tool = ImageSearchTool()
_text_search_tool = TextSearchTool()
_kb_text_search_tool = LocalTextSearchTool()
_visit_tool = VisitTool()

# OpenAI Chat Completions `tools` list (function specs).
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "kb_text_search",
            "description": "Search the local Museum KB (LIMC + Beazley descriptions) and return top-k entry summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search queries (1–3 recommended).",
                    },
                    "num": {
                        "type": "integer",
                        "description": "Number of entries per query (top-k). Default 10.",
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Google SERP reverse image search; search only the input image(s). Use at most once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Image URLs to use as queries.",
                    }
                },
                "required": ["image_urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit a web page and return a summary of its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the web page to visit."},
                    "goal": {"type": "string", "description": "Goal of visiting the page."},
                },
                "required": ["url", "goal"],
            },
        },
    },
]


def _text_search_result(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("queries", [])
    queries = [str(q).strip() for q in raw] if isinstance(raw, list) else [str(raw)]
    queries = [q for q in queries if q][:8]
    payload = _text_search_tool.run(queries, num=10)
    if not _disable_source_control() and payload.get("ok"):
        q_for_rel = " ".join(queries[:3])
        payload = attach_evidence_pool_to_search_payload(payload, tool="text_search", query=q_for_rel)
    return payload


def _kb_text_search_result(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("queries", [])
    queries = [str(q).strip() for q in raw] if isinstance(raw, list) else [str(raw)]
    queries = [q for q in queries if q][:8]
    num = arguments.get("num")
    try:
        k = int(num) if num is not None else 10
    except Exception:
        k = 10
    payload = _kb_text_search_tool.run(queries, num=k)
    if not _disable_source_control() and payload.get("ok"):
        q_for_rel = " ".join(queries[:3])
        # reuse source-control pipeline (expects title/snippet/url fields under results->organic)
        payload = attach_evidence_pool_to_search_payload(payload, tool="text_search", query=q_for_rel)
    return payload


def _image_search_result(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("image_urls", [])
    urls = list(raw) if isinstance(raw, list) else [raw]
    urls = [str(u).strip() for u in urls if str(u).strip()][:4]
    payload = _image_search_tool.run(urls)
    if not _disable_source_control() and payload.get("ok"):
        payload = attach_evidence_pool_to_search_payload(payload, tool="image_search", query=None)
    return payload


def _visit_result(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "") or "").strip()
    goal = str(arguments.get("goal", "") or "").strip()

    if not url:
        return {
            "ok": False,
            "tool": "visit",
            "error": "url is empty",
            "request": {"url": url, "goal": goal},
            "title": "",
            "content": "",
        }

    raw = _visit_tool.run(url, goal)
    err = raw.get("error")
    payload: dict[str, Any] = {
        "ok": not bool(err),
        "tool": "visit",
        "request": {"url": raw.get("url", url), "goal": raw.get("goal", goal)},
        "title": raw.get("title") or "",
        "content": raw.get("content") or "",
    }
    if err:
        payload["error"] = err
    return payload


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Run a single tool by name. Returns JSON string for the model's tool / tool_response message.
    """
    if name == "text_search":
        payload = _text_search_result(arguments)
    elif name == "kb_text_search":
        payload = _kb_text_search_result(arguments)
    elif name == "image_search":
        payload = _image_search_result(arguments)
    elif name == "visit":
        payload = _visit_result(arguments)
    else:
        payload = {"ok": False, "error": "unknown_tool", "name": name, "arguments": arguments}

    return json.dumps(payload, ensure_ascii=False)


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_parse_error": "invalid_json", "raw": raw}
