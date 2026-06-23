"""Evidence pipeline: source-control (post-retrieval) and response-control (pre-/post-generation)."""

from .context import agent_evidence_context, get_user_query
from .response_control import ResponseControl, ResponseControlConfig, simple_decompose_query
from .source_control import (
    SourceControl,
    SourceControlConfig,
    attach_evidence_pool_to_search_payload,
    flatten_image_search_results,
    flatten_text_search_results,
)

__all__ = [
    "agent_evidence_context",
    "get_user_query",
    "SourceControl",
    "SourceControlConfig",
    "attach_evidence_pool_to_search_payload",
    "flatten_image_search_results",
    "flatten_text_search_results",
    "ResponseControl",
    "ResponseControlConfig",
    "simple_decompose_query",
]
