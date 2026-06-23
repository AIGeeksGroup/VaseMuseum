from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


@dataclass(frozen=True)
class LocalLLMConfig:
    base_url: str
    api_key: str
    model: str


def _first(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _load_llm_config() -> LocalLLMConfig:
    """
    Mirror `vase-agent/llm_env.py` env conventions, but keep this module standalone.
    Defaults to local OpenAI-compatible server at http://127.0.0.1:8001/v1.
    """
    env_path = Path(__file__).resolve().parents[1] / "vase-agent" / ".env"
    if env_path.exists() and load_dotenv is not None:
        load_dotenv(env_path, override=False)

    base = _first("LLM_BASE_URL", "VLLM_BASE_URL")
    if not base:
        host = _first("VLLM_HOST", "LLM_HOST", default="127.0.0.1")
        port = _first("VLLM_PORT", "LLM_PORT", default="8001")
        base = f"http://{host}:{port}/v1"
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    return LocalLLMConfig(
        base_url=base,
        api_key=_first("LLM_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY", default="EMPTY"),
        model=_first("LLM_MODEL", "VLLM_MODEL", default="Qwen3-VL-8B-Instruct"),
    )


def get_local_client(config: LocalLLMConfig | None = None):
    cfg = config or _load_llm_config()
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'openai'. Install it in the environment used to run retriever "
            "(e.g. `pip install openai`) to enable nl-caption / embeddings."
        ) from e

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return client, cfg


def nl_caption(
    *,
    structured_caption: str,
    extra_instructions: str = "",
    config: LocalLLMConfig | None = None,
    temperature: float = 0.2,
) -> str:
    client, cfg = get_local_client(config)
    sys = (
        "You are a domain assistant for classical archaeology and museum cataloging. "
        "Rewrite structured metadata into a concise natural-language caption that improves embedding retrieval. "
        "Be factual; do not invent missing details."
    )
    if extra_instructions.strip():
        sys += "\n\nExtra instructions:\n" + extra_instructions.strip()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": "Turn the following structured metadata into a single caption (1-3 sentences):\n\n"
            + structured_caption.strip(),
        },
    ]
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=temperature,
    )
    out = (resp.choices[0].message.content or "").strip()
    return out


def embed_texts(
    *,
    texts: list[str],
    model: str,
    config: LocalLLMConfig | None = None,
) -> list[list[float]]:
    client, cfg = get_local_client(config)
    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    # OpenAI SDK returns objects in original order.
    return [d.embedding for d in resp.data]

