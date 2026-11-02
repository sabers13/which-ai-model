"""Classify transport failures and read ``Retry-After`` without provider imports.

The model layer only retries and falls back for the two outcomes ``PLAN.md`` §5 names:
``RATE_LIMITED`` and ``ENDPOINT_UNREACHABLE``. A schema-invalid answer, a 400 or a
credential error is raised or recorded, never retried to fish for a better answer.

Classification is duck-typed on purpose: it works for the OpenAI, Anthropic and Google
SDK exception classes and for ``httpx`` without importing any of them, so this module
stays on the LangChain side of the dependency graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from which_ai_model.contracts import ProbeReason

__all__ = [
    "RETRYABLE_REASONS",
    "classify_exception",
    "retry_after_seconds",
]

#: The only reasons that may trigger a retry or a fallback model.
RETRYABLE_REASONS: frozenset[ProbeReason] = frozenset(
    {ProbeReason.RATE_LIMITED, ProbeReason.ENDPOINT_UNREACHABLE}
)

_RATE_LIMIT_NAMES = ("ratelimit", "toomanyrequests", "resourceexhausted", "quotaexceeded")
_UNREACHABLE_NAMES = (
    "connecterror",
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "timeout",
    "apiconnectionerror",
    "apiconnectiontimeout",
    "endpointconnectionerror",
    "serviceunavailable",
    "networkerror",
)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 8:
        seen.append(current)
        nested = current.__cause__ or current.__context__
        current = nested if isinstance(nested, BaseException) else None
    return seen


def _status_code(exc: BaseException) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_exception(exc: BaseException) -> ProbeReason | None:
    """Map an exception to a probe reason, or ``None`` when it is not transport."""

    for item in _exception_chain(exc):
        status = _status_code(item)
        if status == 429:
            return ProbeReason.RATE_LIMITED
        if status is not None and 500 <= status <= 599:
            return ProbeReason.ENDPOINT_UNREACHABLE

        name = type(item).__name__.lower()
        if any(fragment in name for fragment in _RATE_LIMIT_NAMES):
            return ProbeReason.RATE_LIMITED
        if any(fragment in name for fragment in _UNREACHABLE_NAMES):
            return ProbeReason.ENDPOINT_UNREACHABLE

        code = getattr(item, "code", None)
        if isinstance(code, int):
            if code == 429:
                return ProbeReason.RATE_LIMITED
            if 500 <= code <= 599:
                return ProbeReason.ENDPOINT_UNREACHABLE

    return None


def _headers(exc: BaseException) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in _exception_chain(exc):
        for holder in (getattr(item, "response", None), item):
            mapping = getattr(holder, "headers", None)
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    if isinstance(key, str) and isinstance(value, str):
                        headers.setdefault(key.lower(), value)
    return headers


def _parse_retry_after(value: str, *, now: datetime) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - now).total_seconds()
    if seconds < 0:
        return 0.0
    return seconds


def retry_after_seconds(exc: BaseException, *, now: datetime | None = None) -> float | None:
    """Seconds the server asked us to wait, from a ``Retry-After`` header or attribute."""
    moment = now or datetime.now(UTC)
    headers = _headers(exc)
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
        value = headers.get(key)
        if value is not None:
            parsed = _parse_retry_after(value, now=moment)
            if parsed is not None:
                return parsed
    for item in _exception_chain(exc):
        attribute = getattr(item, "retry_after", None)
        if isinstance(attribute, (int, float)) and not isinstance(attribute, bool):
            return max(0.0, float(attribute))
        if isinstance(attribute, str):
            parsed = _parse_retry_after(attribute, now=moment)
            if parsed is not None:
                return parsed
    return None
