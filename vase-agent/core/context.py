"""Thread-local / async-safe context so search tools can score hits against the user query."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

_user_query: contextvars.ContextVar[str] = contextvars.ContextVar("vase_user_query", default="")


def get_user_query() -> str:
    """Current user question for relevance scoring in source-control (may be empty)."""
    return _user_query.get()


@contextmanager
def agent_evidence_context(*, user_query: str = "") -> Iterator[None]:
    """Set user query for the duration of an agent step (e.g. one `VaseAgent.run` or one tool round)."""
    token = _user_query.set(user_query or "")
    try:
        yield
    finally:
        _user_query.reset(token)
