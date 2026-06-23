"""
Serialize ``agent_run.VaseAgent.run(..., return_metadata=True)`` messages into a
compact textual trajectory for LLM summarization (youtu-style).
"""

from __future__ import annotations

import json
from typing import Any


def _shorten_str(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 20] + f"\n… [truncated, {len(s)} chars total]"


def messages_to_trajectory_text(
    messages: list[dict[str, Any]],
    *,
    max_chars_per_content: int = 14_000,
) -> str:
    """Turn chat messages into readable lines for experience prompts."""
    lines: list[str] = []
    for i, m in enumerate(messages):
        role = str(m.get("role") or "")
        content = m.get("content")
        if isinstance(content, str):
            text = _shorten_str(content, max_chars_per_content)
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "image_url":
                    parts.append("[image attachment]")
            text = _shorten_str("\n".join(parts), max_chars_per_content)
        else:
            text = _shorten_str(json.dumps(content, ensure_ascii=False, default=str), max_chars_per_content)

        extra = ""
        if role == "assistant" and m.get("tool_calls"):
            try:
                extra = "\n tool_calls: " + json.dumps(m["tool_calls"], ensure_ascii=False, default=str)[:8000]
            except Exception:
                extra = "\n tool_calls: [unserializable]"

        lines.append(f"--- Step {i} | {role} ---\n{text}{extra}")

    return "\n\n".join(lines)
