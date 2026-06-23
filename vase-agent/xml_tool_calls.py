"""
Parse / format XML-style tool I/O (Qwen / Nous / WebWatcher-style <tool_call>, <tool_response>).

Used when the upstream model does not support OpenAI ``tools`` / ``tool_calls``.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)


def format_openai_tools_as_prompt_lines(tool_definitions: list[dict[str, Any]]) -> str:
    """Turn Chat Completions ``tools`` entries into comma-separated JSON objects for <tools> block."""
    parts: list[str] = []
    for item in tool_definitions:
        fn = item.get("function")
        if isinstance(fn, dict):
            parts.append(json.dumps(fn, ensure_ascii=False))
    return ",\n".join(parts)


def extract_tool_calls_xml(assistant_text: str) -> list[dict[str, Any]]:
    """
    Extract ``{"name": str, "arguments": dict}`` from each ``<tool_call>...</tool_call>`` block.

    Inner payload must be JSON: ``{"name": "...", "arguments": {...}}`` (Nous / Qwen convention).
    Invalid blocks are skipped.
    """
    out: list[dict[str, Any]] = []
    if not assistant_text or not assistant_text.strip():
        return out
    for raw_inner in _TOOL_CALL_BLOCK.findall(assistant_text):
        inner = raw_inner.strip()
        if not inner:
            continue
        inner = _strip_json_fence(inner)
        try:
            obj = json.loads(inner)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if not isinstance(name, str) or not name.strip():
            continue
        if args is None:
            args = {}
        if not isinstance(args, dict):
            continue
        out.append({"name": name.strip(), "arguments": args})
    return out


def _strip_json_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def format_tool_responses_xml(tool_results: list[str]) -> str:
    """Paste executor outputs (JSON strings) into one user message."""
    blocks: list[str] = []
    for raw in tool_results:
        t = (raw or "").strip()
        blocks.append(f"<tool_response>\n{t}\n</tool_response>")
    return "\n\n".join(blocks)
