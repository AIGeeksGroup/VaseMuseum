"""Load LLM_BACKEND (vllm | modelscope) and base_url / api_key / model from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV = Path(__file__).resolve().parent / ".env"


def _first(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def get_tool_mode() -> str:
    """How the agent drives tools: ``openai`` (native tool_calls) or ``xml`` (<tool_call> in text)."""
    load_dotenv(_ENV, override=False)
    raw = _first("VASE_TOOL_MODE", "TOOL_CALL_MODE", default="openai").strip().lower()
    if raw in ("xml", "ominisearch", "text", "legacy"):
        return "xml"
    if raw in ("openai", "oai", "tools", "native", "function_calling"):
        return "openai"
    return "openai"


def get_llm_config() -> dict[str, str]:
    load_dotenv(_ENV, override=False)
    raw = _first("LLM_BACKEND", default="vllm").lower()
    if raw in ("modelscope", "ms", "model_scope"):
        base = _first(
            "MODELSCOPE_BASE_URL",
            default="https://api-inference.modelscope.cn/v1",
        ).rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        key = _first("MODELSCOPE_ACCESS_TOKEN", "MODELSCOPE_API_KEY")
        if not key:
            raise ValueError("LLM_BACKEND=modelscope requires MODELSCOPE_ACCESS_TOKEN")
        model = _first("MODELSCOPE_MODEL", "LLM_MODEL", default="Qwen/Qwen3.5-35B-A3B")
        return {
            "backend": "modelscope",
            "model": model,
            "base_url": base,
            "api_key": key,
            "tool_mode": get_tool_mode(),
        }

    base = _first("LLM_BASE_URL", "VLLM_BASE_URL")
    if not base:
        host = _first("VLLM_HOST", "LLM_HOST", default="127.0.0.1")
        port = _first("VLLM_PORT", "LLM_PORT", default="8001")
        base = f"http://{host}:{port}/v1"
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return {
        "backend": "vllm",
        "model": _first("LLM_MODEL", "VLLM_MODEL", default="Qwen3-VL-8B-Instruct"),
        "base_url": base,
        "api_key": _first("LLM_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY", default="EMPTY"),
        "tool_mode": get_tool_mode(),
    }


def get_eval_judge_config() -> dict[str, str]:
    """
    LLM-as-judge endpoint for evaluation metrics (accuracy, hallucination, etc.).

    Override with ``EVAL_JUDGE_MODEL`` / ``EVAL_JUDGE_BASE_URL`` / ``EVAL_JUDGE_API_KEY``
    (aliases ``EVAL_LLM_*``); otherwise uses the same settings as :func:`get_llm_config`.
    """
    load_dotenv(_ENV, override=False)
    base_cfg = get_llm_config()
    model = _first("EVAL_JUDGE_MODEL", "EVAL_LLM_MODEL") or base_cfg["model"]
    base_url = _first("EVAL_JUDGE_BASE_URL", "EVAL_LLM_BASE_URL") or base_cfg["base_url"]
    api_key = _first("EVAL_JUDGE_API_KEY", "EVAL_LLM_API_KEY") or base_cfg["api_key"]
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return {"model": model, "base_url": base_url, "api_key": api_key}
