"""L4 — docs helper ("before").

Pastes the whole ``docs/guide/`` text into the prompt and makes one OpenAI-compatible
chat call. No retrieval, no passage ids, no citations; when the guide has not been
written yet it falls back to the committed test fixture. Configuration comes from this
tool's own environment variables (``DOCS_BASE_URL``, ``DOCS_MODEL``, ``DOCS_KEY`` and
optionally ``DOCS_GUIDE_PATH``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import openai

from which_ai_model.legacy._common import (
    LEGACY_FIXTURES,
    REPO_ROOT,
    LegacyResult,
    elapsed_ms,
    env,
    error_text,
    result_dict,
)

_PROMPT = (
    "You are the documentation helper for this repository. Answer the question using "
    "only the guide below. If the guide does not cover it, say so.\n\n"
    "--- GUIDE START ---\n{guide}\n--- GUIDE END ---\n\nQuestion: {question}"
)


def guide_text(path: Path | None = None) -> str:
    """Every ``*.md`` under ``docs/guide/`` in path order, else the fixture excerpt."""
    root = path or Path(env("DOCS_GUIDE_PATH") or REPO_ROOT / "docs" / "guide")
    if root.is_dir():
        chunks = [file.read_text(encoding="utf-8") for file in sorted(root.rglob("*.md"))]
        if chunks:
            return "\n\n".join(chunks)
    return (LEGACY_FIXTURES / "guide_fallback.md").read_text(encoding="utf-8")


def docs_answer(
    question: str,
    *,
    httpx_client: Any = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    guide_path: Path | None = None,
    timeout_s: float = 30.0,
) -> LegacyResult:
    """One call with the full guide pasted in; the raw answer is the result."""
    started = time.perf_counter()
    key = api_key if api_key is not None else env("DOCS_KEY")
    if not key:
        return LegacyResult("", [], elapsed_ms(started), "DOCS_KEY is not set")
    try:
        guide = guide_text(guide_path)
    except OSError as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc))
    client = openai.OpenAI(
        api_key=key,
        base_url=base_url or env("DOCS_BASE_URL", "http://localhost:8080/v1"),
        http_client=httpx_client,
        max_retries=0,
        timeout=timeout_s,
    )
    try:
        completion = client.chat.completions.create(
            model=model or env("DOCS_MODEL", "docs-small"),
            temperature=0,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(guide=guide, question=question),
                }
            ],
        )
        content = completion.choices[0].message.content or ""
    except (openai.OpenAIError, IndexError, ValueError) as exc:
        return LegacyResult("", [], elapsed_ms(started), error_text(exc, key))
    return LegacyResult(content, [], elapsed_ms(started), None)


def run(question: str, **kwargs: Any) -> dict[str, Any]:
    """``run(input) -> dict`` with the legacy fields; the answer is ``content``."""
    return result_dict(docs_answer(question, **kwargs))
