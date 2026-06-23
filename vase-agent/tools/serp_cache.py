# On-disk cache for Serp tool results (keyed by hashed query parameters)

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_dir() -> Path:
    env = os.getenv("SERP_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / ".serp_cache"


def _key_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def entry_path(kind: str, *parts: str) -> Path:
    return cache_dir() / f"{kind}_{_key_hash(*parts)[:40]}.json"


def load_entry(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_entry(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(path)
