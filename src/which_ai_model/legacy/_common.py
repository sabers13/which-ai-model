"""Shared plumbing for the five legacy ("before") tools.

The tools themselves are deliberately inconsistent and each one wires its own
provider. This module only holds what all five need: the result type, the "old way"
environment lookup, timing, error scrubbing and the dict shape used by ``run()``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

#: Repository root from ``src/which_ai_model/legacy/_common.py``.
REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "legacy"

#: Errors are short and never carry a credential value.
MAX_ERROR_CHARS = 500


@dataclass(frozen=True)
class LegacyResult:
    """What every "before" tool returns; fields match ``docs/INTERFACES.md`` §4."""

    content: str
    tool_calls: list[dict[str, Any]]
    latency_ms: int
    error: str | None


@overload
def env(name: str, default: str) -> str: ...


@overload
def env(name: str, default: None = None) -> str | None: ...


def env(name: str, default: str | None = None) -> str | None:
    """Read the tool's own environment variable, the "old way" (empty means unset)."""
    value = os.environ.get(name)
    return value if value else default


def elapsed_ms(started: float) -> int:
    """Whole milliseconds since ``started`` (``time.perf_counter()``)."""
    return max(0, round((time.perf_counter() - started) * 1000))


def redact(text: str, *secrets: str | None) -> str:
    """Remove known credential values from an error before it is returned."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text[:MAX_ERROR_CHARS]


def error_text(exc: BaseException, *secrets: str | None) -> str:
    """A short, scrubbed description of a failure."""
    text = str(exc).strip() or type(exc).__name__
    return redact(text, *secrets)


def result_dict(result: LegacyResult, **extra: Any) -> dict[str, Any]:
    """``run(input) -> dict``: the legacy fields plus a tool's parsed extras."""
    payload: dict[str, Any] = {
        "content": result.content,
        "tool_calls": result.tool_calls,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }
    payload.update(extra)
    return payload
