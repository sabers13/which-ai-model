"""L2 — reply drafter ("before").

Uses the Anthropic SDK's ``messages.create`` and then pulls JSON out of free text with a
single regex. No structured output, no schema validation, no retry. Configuration comes
from this tool's own environment variables (``DRAFTER_MODEL``, ``DRAFTER_BASE_URL`` and
the SDK's ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import anthropic

from which_ai_model.legacy._common import LegacyResult, elapsed_ms, env, error_text, result_dict

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_PROMPT = (
    "Draft a customer-service reply. Return one JSON object with keys "
    '"intent" (short snake_case label), "urgency" (low, medium or high) and '
    '"reply" (one short paragraph). You may add a sentence before the JSON.\n\n'
    "Message: {message}"
)


def extract_json(text: str) -> dict[str, Any] | None:
    """First ``{...}`` block that parses as a JSON object, or ``None``."""
    match = _JSON_RE.search(text)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _text_of(message: Any) -> str:
    parts = [
        block.text for block in message.content if isinstance(block, anthropic.types.TextBlock)
    ]
    return "\n".join(parts)


def draft_reply(
    message: str,
    *,
    httpx_client: Any = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> LegacyResult:
    """Ask Anthropic for a reply and return the raw text (JSON not enforced)."""
    started = time.perf_counter()
    key = api_key if api_key is not None else env("ANTHROPIC_API_KEY")
    if not key:
        return LegacyResult("", [], elapsed_ms(started), "ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(
        api_key=key,
        base_url=base_url or env("DRAFTER_BASE_URL"),
        http_client=httpx_client,
        max_retries=0,
        timeout=timeout_s,
    )
    try:
        message_obj = client.messages.create(
            model=model or env("DRAFTER_MODEL", "claude-haiku-5-5"),
            max_tokens=1024,
            system="You are a concise banking support assistant.",
            messages=[{"role": "user", "content": _PROMPT.format(message=message)}],
        )
        content = _text_of(message_obj)
    except (anthropic.AnthropicError, ValueError) as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc, key))
    return LegacyResult(content, [], elapsed_ms(started), None)


def run(message: str, **kwargs: Any) -> dict[str, Any]:
    """``run(input) -> dict`` with the legacy fields plus ``parsed`` JSON."""
    result = draft_reply(message, **kwargs)
    return result_dict(result, parsed=extract_json(result.content))
