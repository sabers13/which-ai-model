"""L1-style raw HTTP call to an OpenAI-compatible endpoint (hand-written example)."""

from __future__ import annotations

import os

import httpx

BASE_URL = "https://api.openai.com/v1"


def route_message(message: str) -> str:
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": message}],
            "response_format": {"type": "json_object"},
        },
        timeout=30.0,
    )
    content: str = response.json()["choices"][0]["message"]["content"]
    return content
