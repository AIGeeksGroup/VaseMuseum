#!/usr/bin/env python3
"""
HTTP API for the Digital Exhibition assistant.

Expects the vase-agent environment (``LLM_BASE_URL``, ``LLM_MODEL``, ``.env``, etc.).

Run from repository root::

    python deploy/DigitalExhibition/exhibition_chat_server.py --host 0.0.0.0 --port 8765

Endpoint: POST /v1/exhibition/chat
JSON body::

    {
      "question": "...",
      "image": "data:image/jpeg;base64,...",
      "deep_research": false
    }

When ``deep_research`` is true, runs :meth:`VaseAgent.run` (tools / search).
Otherwise runs :meth:`VaseAgent.run_direct` (single-turn vision only).

Tool rounds are capped (see ``EXHIBITION_MAX_TOOL_ROUNDS`` / ``--max-tool-rounds``) because each
search/visit result is appended to context and can exceed the model window quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vase-agent"))

from agent_run import VaseAgent, _assistant_text, _redact_for_storage  # noqa: E402


def _exhibition_max_tool_rounds() -> int:
    raw = os.environ.get("EXHIBITION_MAX_TOOL_ROUNDS", "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(1, min(20, n))


def _exhibition_ui_step_chars() -> int:
    raw = os.environ.get("EXHIBITION_UI_STEP_CHARS", "12000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 12000
    return max(500, min(100_000, n))


def _exhibition_tool_preview_chars() -> int:
    raw = os.environ.get("EXHIBITION_TOOL_PREVIEW_CHARS", "8000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8000
    return max(500, min(200_000, n))


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n… (truncated, {len(s)} chars total)"


def _content_user_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        lines: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                lines.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image_url":
                lines.append("[image passed to model]")
            elif isinstance(block, dict):
                lines.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(lines)
    return json.dumps(content, ensure_ascii=False)


def _assistant_visible_text(msg: dict[str, object]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return json.dumps(c, ensure_ascii=False)
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


def _normalize_tool_calls(msg: dict[str, object], arg_limit: int) -> list[dict[str, str]]:
    raw = msg.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        out.append(
            {
                "name": str(fn.get("name") or ""),
                "arguments": _clip(str(fn.get("arguments") or ""), arg_limit),
            }
        )
    return out


def _build_ui_steps(messages: list[dict[str, object]], *, max_chars: int) -> list[dict[str, object]]:
    """Turn agent messages into compact UI steps (already redacted / size-capped)."""
    steps: list[dict[str, object]] = []
    user_i = 0
    asst_i = 0
    tool_i = 0
    arg_limit = min(8000, max_chars)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            txt = msg.get("content")
            if isinstance(txt, str) and txt.strip():
                steps.append(
                    {
                        "kind": "system",
                        "title": "System prompt (excerpt)",
                        "text": _clip(txt, max_chars),
                    }
                )
            continue

        if role == "user":
            user_i += 1
            text = _content_user_to_text(msg.get("content"))
            steps.append(
                {
                    "kind": "user",
                    "title": f"User input #{user_i}",
                    "text": _clip(text, max_chars),
                }
            )
            continue

        if role == "assistant":
            asst_i += 1
            text = _assistant_visible_text(msg)
            tcalls = _normalize_tool_calls(msg, arg_limit)
            steps.append(
                {
                    "kind": "assistant",
                    "title": f"Assistant · turn {asst_i}",
                    "text": _clip(text, max_chars) if text.strip() else "",
                    "tool_calls": tcalls,
                }
            )
            continue

        if role == "tool":
            tool_i += 1
            content = msg.get("content")
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            elif not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tcid = str(msg.get("tool_call_id") or "")
            steps.append(
                {
                    "kind": "tool",
                    "title": f"Tool output #{tool_i}",
                    "tool_call_id": tcid,
                    "text": _clip(str(content), max_chars),
                }
            )
            continue

    return steps


def _messages_without_final_assistant(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    """Drop last assistant message so UI can show it only as `answer` (avoid duplicate)."""
    if not messages:
        return [], False
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        return messages[:-1], True
    return messages, False


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/health"):
            body = json.dumps({"ok": True, "service": "exhibition-chat"}).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/v1/exhibition/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            req_json = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": "invalid_json", "detail": str(e)})
            return

        question = str(req_json.get("question") or "").strip()
        image = req_json.get("image") or req_json.get("image_base64")
        deep = bool(req_json.get("deep_research") or req_json.get("deepResearch"))

        if not question:
            self._json_response(400, {"error": "missing_question"})
            return
        if not image:
            self._json_response(400, {"error": "missing_image"})
            return

        if isinstance(image, str) and not image.startswith("data:"):
            image = f"data:image/jpeg;base64,{image}"

        try:
            agent = VaseAgent.from_env()
            if deep:
                meta = agent.run(
                    question=question,
                    image_url=str(image),
                    log_jsonl=False,
                    verbose=False,
                    return_metadata=True,
                    max_tool_rounds=_exhibition_max_tool_rounds(),
                    log_full_trajectory=False,
                )
                mode = "search"
                final_msg = meta.get("assistant")
                if not isinstance(final_msg, dict):
                    raise RuntimeError("deep run: missing assistant")
                text = _assistant_text(final_msg)
                raw_msgs = meta.get("messages") or []
                if not isinstance(raw_msgs, list):
                    raw_msgs = []
                safe_list: list[dict[str, object]] = []
                for m in raw_msgs:
                    if isinstance(m, dict):
                        safe_list.append(dict(m))
                safe_msgs = _redact_for_storage(
                    safe_list,
                    redact_data_urls=True,
                    max_tool_content_chars=_exhibition_tool_preview_chars(),
                )
                if not isinstance(safe_msgs, list):
                    safe_msgs = []
                safe_typed = [x for x in safe_msgs if isinstance(x, dict)]
                inter_msgs, had_final = _messages_without_final_assistant(safe_typed)
                steps = _build_ui_steps(inter_msgs, max_chars=_exhibition_ui_step_chars())
                payload: dict[str, object] = {
                    "answer": text,
                    "mode": mode,
                    "question": question,
                    "steps": steps,
                    "rounds": meta.get("rounds") or [],
                }
                if not had_final:
                    payload["steps"] = _build_ui_steps(safe_typed, max_chars=_exhibition_ui_step_chars())
            else:
                out = agent.run_direct(
                    question=question,
                    image_url=str(image),
                    log_jsonl=False,
                    verbose=False,
                )
                mode = "direct"
                text = _assistant_text(out)
                payload = {
                    "answer": text,
                    "mode": mode,
                    "question": question,
                    "steps": [],
                    "rounds": [],
                }
            self._json_response(200, payload)
        except Exception as e:
            traceback.print_exc()
            self._json_response(
                500,
                {"error": "inference_failed", "detail": str(e)},
            )

    def _json_response(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description="Digital Exhibition chat API (VaseAgent)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        metavar="N",
        help="Cap agent tool rounds (sets EXHIBITION_MAX_TOOL_ROUNDS). Default: env or 4.",
    )
    args = p.parse_args()
    if args.max_tool_rounds is not None:
        os.environ["EXHIBITION_MAX_TOOL_ROUNDS"] = str(args.max_tool_rounds)
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Exhibition chat API http://{args.host}:{args.port}/v1/exhibition/chat", flush=True)
    print(f"max_tool_rounds (deep research) = {_exhibition_max_tool_rounds()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
