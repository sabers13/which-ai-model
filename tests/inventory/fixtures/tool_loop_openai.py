"""L5-style hand-written OpenAI-compatible tool loop."""

from __future__ import annotations

import os

import httpx

URL = "https://api.openai.com/v1/chat/completions"


def run_agent(question: str) -> str:
    messages: list[dict[str, object]] = [{"role": "user", "content": question}]
    while True:
        response = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": "gpt-4o-mini", "messages": messages, "tools": []},
            timeout=30.0,
        )
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return str(message.get("content", ""))
        messages.append(message)
