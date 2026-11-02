"""Read token usage and gateway cost metadata from a model response.

The gateway reports, next to the provider's own usage, a block shaped like::

    "usage": {
        "input_tokens": 812,
        "output_tokens": 44,
        "completion_tokens_details": {"reasoning_tokens": 12},
        "opencode": {
            "list_cost": 0.00017,
            "allowance": "go",
            "billing_source": "subscription",
            "tokens": {"input_tokens": 812, "output_tokens": 44},
        },
    }

``measured_cost_usd`` is that ``list_cost`` when the gateway sent one. It is the
measured cost; the price table in :mod:`which_ai_model.core.cost` is only the fallback.
Nothing here reads a key, a header or a base URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

__all__ = [
    "EMPTY",
    "TRUNCATED",
    "Usage",
    "empty_answer_reason",
    "extract_usage",
    "measured_cost_usd",
]

#: Reason for an answer with no content and no tool calls (``PLAN.md`` §7).
EMPTY = "EMPTY"
#: Reason for an answer cut off by the output limit (``PLAN.md`` §7).
TRUNCATED = "TRUNCATED"

_FINISH_REASONS = ("finish_reason", "stop_reason", "stopReason")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def _get(mapping: Any, *names: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class Usage:
    """Normalised usage for one call. Every field may be unknown (``None``)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    measured_cost_usd: float | None = None
    gateway: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe usage block for traces and recordings."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "measured_cost_usd": self.measured_cost_usd,
            "finish_reason": self.finish_reason,
            "gateway": dict(self.gateway),
        }


def _message_metadata(message: BaseMessage) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = getattr(message, "usage_metadata", None)
    response = getattr(message, "response_metadata", None)
    return (
        usage if isinstance(usage, dict) else {},
        response if isinstance(response, dict) else {},
    )


def _raw_usage_from(response_metadata: dict[str, Any]) -> dict[str, Any]:
    """The provider's raw usage object, as kept by the LangChain integration."""
    candidates = (
        response_metadata.get("token_usage"),
        response_metadata.get("usage"),
        response_metadata.get("usage_metadata"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _finish_reason(response_metadata: dict[str, Any]) -> str | None:
    for name in _FINISH_REASONS:
        value = response_metadata.get(name)
        if isinstance(value, str):
            return value
    return None


def extract_usage(source: Any) -> Usage:
    """Read usage from an ``AIMessage``, a trace/recording dict, or a raw usage dict.

    Accepts every shape the layer meets in practice:

    - a LangChain message: ``usage_metadata`` plus ``response_metadata``;
    - a recording row response: ``{"content": ..., "tool_calls": ..., "usage": {...}}``;
    - a bare provider usage object (``{"prompt_tokens": ...}``).
    """
    usage_metadata: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {}

    if isinstance(source, BaseMessage):
        usage_metadata, response_metadata = _message_metadata(source)
        raw = _raw_usage_from(response_metadata) or usage_metadata
    elif isinstance(source, dict):
        if isinstance(source.get("usage_metadata"), dict):
            usage_metadata = source["usage_metadata"]
        if isinstance(source.get("response_metadata"), dict):
            response_metadata = source["response_metadata"]
        raw = _first(
            source.get("usage"),
            source.get("token_usage"),
            source.get("usage_metadata"),
            source,
        )
        raw = raw if isinstance(raw, dict) else {}
        if not usage_metadata:
            usage_metadata = raw
    else:
        raw = {}

    gateway = raw.get("opencode")
    gateway = gateway if isinstance(gateway, dict) else {}

    input_tokens = _first(
        _as_int(_get(usage_metadata, "input_tokens")),
        _as_int(_get(raw, "input_tokens", "prompt_tokens")),
    )
    output_tokens = _first(
        _as_int(_get(usage_metadata, "output_tokens")),
        _as_int(_get(raw, "output_tokens", "completion_tokens")),
    )

    detail = _first(
        _get(usage_metadata, "output_token_details"),
        _get(raw, "completion_tokens_details"),
        _get(raw, "output_tokens_details"),
    )
    reasoning_tokens = _as_int(_get(detail, "reasoning", "reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _as_int(_get(gateway, "reasoning_tokens"))

    total_tokens = _first(
        _as_int(_get(usage_metadata, "total_tokens")),
        _as_int(_get(raw, "total_tokens")),
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        measured_cost_usd=_as_float(gateway.get("list_cost")),
        gateway=dict(gateway),
        finish_reason=_finish_reason(response_metadata),
    )


def measured_cost_usd(source: Any) -> float | None:
    """The gateway's own ``list_cost`` for this call, or ``None``.

    This is a measurement, not an estimate: it is only ever the value the gateway
    sent. A missing block is ``None`` and callers fall back to the price table.
    """
    return extract_usage(source).measured_cost_usd


def _has_answer_content(message: BaseMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        if content.strip():
            return True
    elif content:
        return True
    tool_calls = getattr(message, "tool_calls", None)
    return bool(tool_calls)


def empty_answer_reason(message: BaseMessage) -> str | None:
    """Why an answer is unusable, or ``None`` when it has something in it.

    ``TRUNCATED`` means the provider stopped on the length limit (often with
    ``content: null``); ``EMPTY`` means no text and no tool call. Either way the
    answer is schema-invalid, never an exception.
    """
    if _has_answer_content(message):
        return None
    _, response_metadata = _message_metadata(message)
    reason = _finish_reason(response_metadata)
    if reason is not None and reason.lower() in {"length", "max_tokens", "truncated"}:
        return TRUNCATED
    return EMPTY
