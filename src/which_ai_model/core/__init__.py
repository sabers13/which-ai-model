"""T02 — the shared model layer used by every tool, benchmark and RAG path.

Public surface (``docs/INTERFACES.md`` §3): build a chat model, get one with retries and
fallback, ask for structured output, record/replay responses, trace every call, read
usage and cost, and pace calls per connection.
"""

from __future__ import annotations

from .cost import Price, cost_usd, load_prices, usage_cost_usd
from .models import (
    FallbackChatModel,
    ModelBuilder,
    get_model,
    make_agent,
    make_chat_model,
)
from .network_guard import NetworkAttempted, NetworkGuard
from .ratelimit import DEFAULT_LIMITS, Limits, RateLimiter, load_limits
from .record import (
    Recorder,
    RecordingMode,
    ReplayChatModel,
    ReplayMiss,
    recording_key,
    replay_model,
    request_digest,
    response_from_message,
    scrub_response,
)
from .structured import StructuredAnswer, make_structured_model, structured
from .trace import (
    CallTraceHandler,
    TraceWriter,
    maybe_enable_langsmith,
    trace_path,
)
from .usage import EMPTY, TRUNCATED, Usage, empty_answer_reason, extract_usage, measured_cost_usd

__all__ = [
    "DEFAULT_LIMITS",
    "EMPTY",
    "TRUNCATED",
    "CallTraceHandler",
    "FallbackChatModel",
    "Limits",
    "ModelBuilder",
    "NetworkAttempted",
    "NetworkGuard",
    "Price",
    "RateLimiter",
    "Recorder",
    "RecordingMode",
    "ReplayChatModel",
    "ReplayMiss",
    "StructuredAnswer",
    "TraceWriter",
    "Usage",
    "cost_usd",
    "empty_answer_reason",
    "extract_usage",
    "get_model",
    "load_limits",
    "load_prices",
    "make_agent",
    "make_chat_model",
    "make_structured_model",
    "maybe_enable_langsmith",
    "measured_cost_usd",
    "recording_key",
    "replay_model",
    "request_digest",
    "response_from_message",
    "scrub_response",
    "structured",
    "trace_path",
    "usage_cost_usd",
]
