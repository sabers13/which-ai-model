"""Mock-transport helpers shared by the legacy tool tests."""

from __future__ import annotations

import json
from typing import Any

import httpx


def scripted(responses: list[dict[str, Any]]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A mock handler that replays ``responses`` and records requests."""
    remaining = list(responses)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if not remaining:
            raise AssertionError("unexpected extra request")
        return httpx.Response(200, json=remaining.pop(0))

    return httpx.MockTransport(handler), requests


def chat_completion(
    content: str = "", *, calls: list[dict[str, Any]] | None = None, finish: str = "stop"
) -> dict[str, Any]:
    """One OpenAI-compatible chat-completion payload."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One OpenAI-compatible tool call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
