from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SourceName = Literal["LIMC", "BEAZLEY_DESC"]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    source: SourceName
    uri: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    source: SourceName
    title: str
    uri: str
    text: str
    meta: dict[str, Any]

