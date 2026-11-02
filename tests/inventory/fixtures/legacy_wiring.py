"""Legacy-style wiring: env-helper model, payload variable, regex parsed elsewhere."""

from __future__ import annotations

import os
import re

import httpx

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def parse(content: str) -> str | None:
    match = _JSON_RE.search(content)
    return match.group(0) if match else None


def ask(
    message: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> str:
    key = api_key or env("ASK_KEY")
    body = {
        "model": model or env("ASK_MODEL", "small"),
        "messages": [{"role": "user", "content": f"Message: {message}"}],
    }
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(
            f"{base_url or env('ASK_BASE_URL', 'http://localhost:8080/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
        )
    content: str = response.json()["choices"][0]["message"]["content"]
    return content
