"""L3 — scope checker ("before").

Uses the Google Gen AI SDK (``google-genai``) and decides scope with a substring
search for "yes"/"no" in the model's free text. Configuration comes from this tool's own
environment variables (``SCOPE_MODEL``, ``SCOPE_BASE_URL`` and ``GEMINI_API_KEY``).
"""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from which_ai_model.legacy._common import LegacyResult, elapsed_ms, env, error_text, result_dict

_PROMPT = (
    "You check whether a customer message is within the supported scope of a retail-bank "
    "support assistant (cards, transfers, top-ups, fees and account access). Answer with "
    "the single word yes or no.\n\nMessage: {message}"
)

IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"


def parse_scope(content: str) -> str | None:
    """Deliberately brittle: first "yes" wins, then any "no" anywhere."""
    lowered = content.lower()
    if "yes" in lowered:
        return IN_SCOPE
    if "no" in lowered:
        return OUT_OF_SCOPE
    return None


def check_scope(
    message: str,
    *,
    httpx_client: Any = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LegacyResult:
    """Ask Gemini for a yes/no answer and return its raw text."""
    started = time.perf_counter()
    key = api_key if api_key is not None else env("GEMINI_API_KEY")
    if not key:
        return LegacyResult("", [], elapsed_ms(started), "GEMINI_API_KEY is not set")
    options = types.HttpOptions(
        base_url=base_url or env("SCOPE_BASE_URL"), httpx_client=httpx_client
    )
    client = genai.Client(api_key=key, http_options=options)
    try:
        response = client.models.generate_content(
            model=model or env("SCOPE_MODEL", "gemini-4-argon"),
            contents=_PROMPT.format(message=message),
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=1024),
        )
        content = response.text or ""
    except (genai.errors.APIError, ValueError) as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc, key))
    return LegacyResult(content, [], elapsed_ms(started), None)


def run(message: str, **kwargs: Any) -> dict[str, Any]:
    """``run(input) -> dict`` with the legacy fields plus the parsed ``label``."""
    result = check_scope(message, **kwargs)
    return result_dict(result, label=parse_scope(result.content))
