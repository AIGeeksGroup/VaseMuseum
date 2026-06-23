import importlib.util
import json
import base64
import mimetypes
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_prompt import (
    SYSTEM_APPEND_XML,
    build_user_prompt_xml,
    system_prompt,
    user_prompt,
)
from llm_env import get_llm_config
from xml_tool_calls import (
    extract_tool_calls_xml,
    format_openai_tools_as_prompt_lines,
    format_tool_responses_xml,
)

_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from core.context import agent_evidence_context
from core.response_control import ResponseControl


def _tools():
    p = Path(__file__).resolve().parent / "tools" / "vase_tool_call.py"
    spec = importlib.util.spec_from_file_location("_vtc", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TOOL_DEFINITIONS, m.execute_tool, m.parse_tool_arguments


_TOOL_DEFS, _tool_exec, _tool_parse = _tools()

_SYSTEM_EXTRA = (
    "Use text_search, image_search, and visit via function calling; "
    "when you have enough information, give the final answer without calling more tools."
)

_DEFAULT_JSONL = Path(__file__).resolve().parent / "agent_runs.jsonl"

# Log storage: skip embedding multi‑MB base64 in JSON by default
_DEFAULT_MAX_TOOL_CONTENT_CHARS = 400_000


def _messages_for_verbose_print(messages: list[dict[str, Any]]) -> list[Any]:
    """Truncate data: URLs for readable terminal output."""

    def shorten(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: shorten(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [shorten(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("data:") and len(obj) > 120:
            return f"data:<… truncated, {len(obj)} chars>"
        return obj

    return [shorten(m) for m in messages]


def _redact_for_storage(
    obj: Any,
    *,
    redact_data_urls: bool,
    max_tool_content_chars: int | None,
) -> Any:
    """Make messages JSON-safe for logs: optional data-URL redaction and tool payload size cap."""

    def redact_string(s: str) -> str | dict[str, Any]:
        if redact_data_urls and s.startswith("data:") and ";base64," in s:
            try:
                head, _b64 = s.split(";base64,", 1)
            except ValueError:
                return s
            return {
                "_redacted": "data_url",
                "prefix": head[:120],
                "approx_chars": len(s),
            }
        if max_tool_content_chars is not None and len(s) > max_tool_content_chars:
            return {
                "_truncated": True,
                "head": s[:20_000],
                "total_chars": len(s),
            }
        return s

    if isinstance(obj, dict):
        return {k: _redact_for_storage(v, redact_data_urls=redact_data_urls, max_tool_content_chars=max_tool_content_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_for_storage(x, redact_data_urls=redact_data_urls, max_tool_content_chars=max_tool_content_chars) for x in obj]
    if isinstance(obj, str):
        return redact_string(obj)
    return obj


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    if isinstance(usage, dict):
        return dict(usage)
    return {"repr": str(usage)}


def _assistant_text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>", re.IGNORECASE)


def _merge_evidence_pools_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_url: set[str] = set()

    def consume_payload(raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        pool = payload.get("evidence_pool")
        if not isinstance(pool, list):
            return
        for e in pool:
            if not isinstance(e, dict):
                continue
            u = str(e.get("url") or "")
            key = u or str(id(e))
            if key in seen_url:
                continue
            seen_url.add(key)
            merged.append(dict(e))

    for m in messages:
        role = m.get("role")
        if role == "tool":
            raw = m.get("content")
            if isinstance(raw, str):
                consume_payload(raw)
            continue
        if role == "user":
            c = m.get("content")
            if isinstance(c, str):
                for block in _TOOL_RESPONSE_BLOCK.findall(c):
                    consume_payload(block.strip())

    for i, e in enumerate(merged):
        e["source_id"] = f"S{i}"
    return merged


def _prepend_text_to_assistant_message(msg: dict[str, Any], prefix: str) -> None:
    c = msg.get("content")
    if isinstance(c, str):
        msg["content"] = f"{prefix}\n\n{c}" if (c or "").strip() else prefix
        return
    if isinstance(c, list):
        msg["content"] = [{"type": "text", "text": prefix}, *c]
        return
    msg["content"] = prefix


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_trajectory_sidecar(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class VaseAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        tool_mode: str = "openai",
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
        self.tool_mode = tool_mode if tool_mode in ("openai", "xml") else "openai"

    @classmethod
    def from_env(cls) -> "VaseAgent":
        c = get_llm_config()
        return cls(
            c["model"],
            c["base_url"],
            c["api_key"],
            tool_mode=str(c.get("tool_mode", "openai")),
        )

    @staticmethod
    def _image_url_to_data_url(image_url: str) -> str:
        if image_url.startswith(("http://", "https://")):
            pass
        else:
            p = Path(image_url).expanduser()
            if p.is_file():
                data = p.read_bytes()
                mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{b64}"
            return image_url
        with urllib.request.urlopen(image_url, timeout=15) as resp:
            ct = resp.headers.get("Content-Type")
            data = resp.read()
        mime = ct or mimetypes.guess_type(image_url)[0] or "application/octet-stream"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def run_direct(
        self,
        question: str,
        image_url: str,
        *,
        temperature: float = 0.7,
        timeout: float = 120.0,
        system_prompt_text: str | None = None,
        log_jsonl: bool = False,
        jsonl_path: Path | str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Paper **Direct** baseline: single-turn VLM, no tools / no search.

        ``image_url`` may be http(s) or a local file path (read and sent as data URL).
        """
        try:
            img = self._image_url_to_data_url(image_url)
        except Exception:
            img = image_url

        sys_txt = (system_prompt_text or "").strip() or (
            "You are an expert on ancient Greek vases and museum exhibits. "
            "Answer the user's question using the image. Be concise and factual; "
            "if uncertain, say so briefly."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_txt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question.strip()},
                    {"type": "image_url", "image_url": {"url": img}},
                ],
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError("run_direct: empty choices from chat.completions")
        msg = getattr(choices[0], "message", None)
        if msg is None:
            raise RuntimeError("run_direct: missing message on choice")
        assistant = msg.model_dump()
        if verbose:
            print("[run_direct] assistant:", _assistant_text(assistant)[:2000], flush=True)

        if log_jsonl:
            out_path = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
            rec: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "mode": "direct",
                "input": {"question": question, "image_url": image_url},
                "final_assistant": assistant,
                "output": assistant,
            }
            _append_jsonl(out_path, rec)

        return assistant

    def _run_openai_loop(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tool_rounds: int,
        verbose: bool,
        temperature: float = 0.7,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        last: dict[str, Any] | None = None
        round_meta: list[dict[str, Any]] = []
        for round_i in range(1, max_tool_rounds + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_TOOL_DEFS,
                tool_choice="auto",
                temperature=temperature,
                timeout=120,
            )
            choices = getattr(resp, "choices", None) or []
            if not choices:
                meta: dict[str, Any] = {}
                for key in ("id", "model", "object", "service_tier"):
                    v = getattr(resp, key, None)
                    if v is not None:
                        meta[key] = v
                raise RuntimeError(
                    "chat.completions returned no choices (choices is null or empty). "
                    "The upstream API may have rejected the request, hit a limit, or returned a non-standard payload. "
                    f"response_fields={meta}"
                )
            ch0 = choices[0]
            msg = getattr(ch0, "message", None)
            if msg is None:
                meta = {
                    "finish_reason": getattr(ch0, "finish_reason", None),
                    "response_id": getattr(resp, "id", None),
                }
                raise RuntimeError(
                    "chat.completions choice has no message (message is null). "
                    f"Often caused by API errors, content filters, or an unexpected provider payload. meta={meta}"
                )
            assistant = msg.model_dump()
            round_meta.append(
                {
                    "round": round_i,
                    "response_id": getattr(resp, "id", None),
                    "model": getattr(resp, "model", None),
                    "finish_reason": getattr(ch0, "finish_reason", None),
                    "usage": _usage_to_dict(getattr(resp, "usage", None)),
                }
            )
            messages.append(assistant)
            last = assistant

            tcalls = msg.tool_calls or []
            if verbose:
                print(f"[agent round {round_i}] tool_calls (openai):", flush=True)
                if not tcalls:
                    print("  (none — model stopped calling tools this turn)", flush=True)
                else:
                    for tc in tcalls:
                        fn = tc.function
                        args_preview = (fn.arguments or "")[:800]
                        if len(fn.arguments or "") > 800:
                            args_preview += "…"
                        print(
                            f"  - id={tc.id} name={fn.name} arguments={args_preview}",
                            flush=True,
                        )

            if not tcalls:
                break

            for tc in tcalls:
                fn = tc.function
                try:
                    out = _tool_exec(fn.name, _tool_parse(fn.arguments or ""))
                except Exception as e:
                    out = json.dumps({"error": str(e), "tool": fn.name}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})

        return last, messages, round_meta

    def _run_xml_loop(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tool_rounds: int,
        verbose: bool,
        temperature: float = 0.7,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse ``<tool_call>`` from assistant text and append ``<tool_response>`` user turns (no API ``tools``)."""
        last: dict[str, Any] | None = None
        round_meta: list[dict[str, Any]] = []
        for round_i in range(1, max_tool_rounds + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=120,
            )
            choices = getattr(resp, "choices", None) or []
            if not choices:
                meta: dict[str, Any] = {}
                for key in ("id", "model", "object", "service_tier"):
                    v = getattr(resp, key, None)
                    if v is not None:
                        meta[key] = v
                raise RuntimeError(
                    "chat.completions returned no choices (choices is null or empty). "
                    "The upstream API may have rejected the request, hit a limit, or returned a non-standard payload. "
                    f"response_fields={meta}"
                )
            ch0 = choices[0]
            msg = getattr(ch0, "message", None)
            if msg is None:
                meta = {
                    "finish_reason": getattr(ch0, "finish_reason", None),
                    "response_id": getattr(resp, "id", None),
                }
                raise RuntimeError(
                    "chat.completions choice has no message (message is null). "
                    f"Often caused by API errors, content filters, or an unexpected provider payload. meta={meta}"
                )
            assistant = msg.model_dump()
            round_meta.append(
                {
                    "round": round_i,
                    "response_id": getattr(resp, "id", None),
                    "model": getattr(resp, "model", None),
                    "finish_reason": getattr(ch0, "finish_reason", None),
                    "usage": _usage_to_dict(getattr(resp, "usage", None)),
                }
            )
            messages.append(assistant)
            last = assistant

            text = _assistant_text(assistant)
            calls = extract_tool_calls_xml(text)
            if verbose:
                print(f"[agent round {round_i}] xml <tool_call> blocks: {len(calls)}", flush=True)
                for c in calls:
                    ap = json.dumps(c.get("arguments") or {}, ensure_ascii=False)[:800]
                    print(f"  - name={c.get('name')} arguments={ap}", flush=True)

            if not calls:
                break

            outs: list[str] = []
            for call in calls:
                name = call.get("name") or ""
                args = call.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                try:
                    out = _tool_exec(str(name), args)
                except Exception as e:
                    out = json.dumps({"error": str(e), "tool": name}, ensure_ascii=False)
                outs.append(out)
            messages.append({"role": "user", "content": format_tool_responses_xml(outs)})

        return last, messages, round_meta

    def run(
        self,
        question: str,
        image_url: str,
        *,
        max_tool_rounds: int = 10,
        log_jsonl: bool = True,
        jsonl_path: Path | str | None = None,
        verbose: bool = False,
        response_control_gate: bool = False,
        apply_uncertain_preamble: bool = False,
        return_metadata: bool = False,
        log_full_trajectory: bool = True,
        log_redact_image_data_url: bool = True,
        log_max_tool_content_chars: int | None = _DEFAULT_MAX_TOOL_CONTENT_CHARS,
        trajectory_json_path: Path | str | None = None,
        tool_mode: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        mode = tool_mode if tool_mode in ("openai", "xml") else self.tool_mode

        try:
            img = self._image_url_to_data_url(image_url)
        except Exception:
            img = image_url

        tool_lines = format_openai_tools_as_prompt_lines(_TOOL_DEFS)
        if mode == "xml":
            system_text = f"{system_prompt.strip()}\n\n{SYSTEM_APPEND_XML.strip()}"
            user_text = build_user_prompt_xml(question, image_url, tool_lines)
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                },
            ]
        else:
            system_text = f"{system_prompt.strip()}\n\n{_SYSTEM_EXTRA}"
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt.replace("{Question}", question).replace("{Image_url}", image_url),
                        },
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                },
            ]

        last: dict[str, Any] | None = None
        evidence_audit: dict[str, Any] | None = None
        round_meta: list[dict[str, Any]] = []
        with agent_evidence_context(user_query=question):
            if mode == "xml":
                last, messages, round_meta = self._run_xml_loop(
                    messages,
                    max_tool_rounds=max_tool_rounds,
                    verbose=verbose,
                    temperature=temperature,
                )
            else:
                last, messages, round_meta = self._run_openai_loop(
                    messages,
                    max_tool_rounds=max_tool_rounds,
                    verbose=verbose,
                    temperature=temperature,
                )

        if last is None:
            raise RuntimeError("no assistant message")

        if response_control_gate:
            pool = _merge_evidence_pools_from_messages(messages)
            rc = ResponseControl()
            evidence_audit = rc.run(
                query=question,
                evidence_pool=pool,
                assistant_draft=_assistant_text(last),
            )
            if apply_uncertain_preamble and evidence_audit.get("mode") == "uncertain":
                preamble = rc.format_uncertain_preamble(evidence_audit)
                _prepend_text_to_assistant_message(last, preamble)

        if verbose:
            print("[agent] full messages (image data URLs shortened):", flush=True)
            print(
                json.dumps(
                    _messages_for_verbose_print(messages),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                flush=True,
            )

        if log_jsonl:
            out_path = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
            rec: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "input": {"question": question, "image_url": image_url},
                "run": {
                    "tool_mode": mode,
                    "temperature": temperature,
                    "max_tool_rounds": max_tool_rounds,
                    "response_control_gate": response_control_gate,
                    "apply_uncertain_preamble": apply_uncertain_preamble,
                    "log_full_trajectory": log_full_trajectory,
                    "log_redact_image_data_url": log_redact_image_data_url,
                    "log_max_tool_content_chars": log_max_tool_content_chars,
                },
                "tool_definitions": _TOOL_DEFS,
                "rounds": round_meta,
                "final_assistant": last,
                "output": last,
            }
            if evidence_audit is not None:
                rec["evidence_audit"] = evidence_audit
            if log_full_trajectory:
                rec["trajectory"] = _redact_for_storage(
                    messages,
                    redact_data_urls=log_redact_image_data_url,
                    max_tool_content_chars=log_max_tool_content_chars,
                )
            _append_jsonl(out_path, rec)
            sidecar = Path(trajectory_json_path) if trajectory_json_path else None
            if sidecar is not None:
                _write_trajectory_sidecar(sidecar, rec)

        if return_metadata:
            return {
                "assistant": last,
                "evidence_audit": evidence_audit,
                "messages": messages,
                "rounds": round_meta,
            }
        return last


if __name__ == "__main__":
    a = VaseAgent.from_env()
    print(
        a.run(
            question="What is the identity of the woman between the warriors on the amphora?",
            image_url="https://YOUR_IMAGE_HOST/images/4B339C11-B634-4F10-957C-300C4D462AE1_5_ac001001.jpg",
            verbose=True,
        )
    )
