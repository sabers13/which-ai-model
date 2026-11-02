"""L1 — ticket router ("before").

Raw ``httpx`` POST to an OpenAI-compatible ``/chat/completions`` endpoint. The label is
parsed from the first line of free text and matched against the 77 Banking77 intents;
there is no schema and no retry. Configuration comes from this tool's own environment
variables (``ROUTER_BASE_URL``, ``ROUTER_MODEL``, ``ROUTER_KEY``).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from which_ai_model.legacy._common import LegacyResult, elapsed_ms, env, error_text, result_dict
from which_ai_model.legacy._intents import BY_LOWER, INTENTS

__all__ = ["INTENTS", "parse_label", "route_message", "run"]


_PROMPT = (
    "You route banking support messages. Reply with exactly one intent label on the "
    "first line, chosen from the supported label list. Do not explain.\n\nMessage: {message}"
)


def parse_label(content: str) -> str | None:
    """Match the first line of the reply against the 77 intents (case-insensitive)."""
    first_line = content.splitlines()[0].strip() if content.strip() else ""
    return BY_LOWER.get(first_line.lower())


def route_message(
    message: str,
    *,
    transport: httpx.BaseTransport | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> LegacyResult:
    """Send one raw chat-completions request and return the unparsed reply."""
    started = time.perf_counter()
    key = api_key if api_key is not None else env("ROUTER_KEY")
    if not key:
        return LegacyResult("", [], elapsed_ms(started), "ROUTER_KEY is not set")
    url = (base_url or env("ROUTER_BASE_URL", "http://localhost:8080/v1") or "").rstrip("/")
    body = {
        "model": model or env("ROUTER_MODEL", "router-small"),
        "messages": [{"role": "user", "content": _PROMPT.format(message=message)}],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(transport=transport, timeout=timeout_s, follow_redirects=False) as client:
            response = client.post(f"{url}/chat/completions", headers=headers, json=body)
            if response.status_code >= 400:
                return LegacyResult(
                    "", [], elapsed_ms(started), f"API request failed (HTTP {response.status_code})"
                )
            data: Any = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc, key))
    try:
        content = data["choices"][0]["message"]["content"] or ""
        if not isinstance(content, str):
            raise TypeError("assistant content is not a string")
    except (KeyError, IndexError, TypeError) as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc))
    return LegacyResult(content, [], elapsed_ms(started), None)


def run(message: str, **kwargs: Any) -> dict[str, Any]:
    """``run(input) -> dict`` with the legacy fields plus the parsed ``label``."""
    result = route_message(message, **kwargs)
    return result_dict(result, label=parse_label(result.content))
