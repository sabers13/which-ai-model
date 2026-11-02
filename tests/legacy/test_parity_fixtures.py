"""The 50 committed parity items replay through every legacy tool."""

from __future__ import annotations

from typing import Any

import httpx
import httpx2
import pytest
from _helpers import chat_completion, scripted, tool_call

from which_ai_model.legacy.l1_router import route_message
from which_ai_model.legacy.l2_drafter import draft_reply, extract_json
from which_ai_model.legacy.l3_scope import check_scope, parse_scope
from which_ai_model.legacy.l4_docs import docs_answer
from which_ai_model.legacy.l5_agent import run_agent


def test_fixture_shape(parity_items: list[dict[str, Any]]) -> None:
    assert len(parity_items) == 50
    assert len({item["item_id"] for item in parity_items}) == 50
    assert sum(1 for item in parity_items if item.get("docs_question")) == 10
    assert sum(1 for item in parity_items if item["responses"]["l5"]["tool_calls"]) == 10
    for item in parity_items:
        assert set(item["responses"]) == {"l1", "l2", "l3", "l4", "l5"}
        assert set(item["responses"]["l5"]) == {"content", "tool_calls"}


def test_l1_replays_every_item(
    monkeypatch: pytest.MonkeyPatch, parity_items: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("ROUTER_KEY", "test-router-key")
    for item in parity_items:
        content = item["responses"]["l1"]["content"]
        transport = httpx.MockTransport(
            lambda request, body=content: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": body}}]},
            )
        )
        result = route_message(item["message"], transport=transport)
        assert result.error is None
        assert result.content == content


def test_l2_replays_every_item(
    monkeypatch: pytest.MonkeyPatch, parity_items: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    for item in parity_items:
        content = item["responses"]["l2"]["content"]

        def handler(request: httpx2.Request, body: str = content) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": body}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        result = draft_reply(
            item["message"], httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )
        assert result.error is None
        parsed = extract_json(result.content)
        assert parsed is not None
        assert parsed["intent"] == item["intent"]


def test_l3_replays_every_item(
    monkeypatch: pytest.MonkeyPatch, parity_items: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    for item in parity_items:
        content = item["responses"]["l3"]["content"]

        def handler(request: httpx.Request, body: str = content) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": body}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            )

        result = check_scope(
            item["message"], httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        assert result.error is None
        expected = "in_scope" if "yes" in content.lower() else "out_of_scope"
        assert parse_scope(result.content) == expected


def test_l4_replays_every_item(
    monkeypatch: pytest.MonkeyPatch, parity_items: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("DOCS_KEY", "test-docs-key")
    for item in parity_items:
        content = item["responses"]["l4"]["content"]
        question = item.get("docs_question", item["message"])

        def handler(request: httpx2.Request, body: str = content) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "docs-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": body},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        result = docs_answer(
            question, httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )
        assert result.error is None
        assert result.content == content


def test_l5_replays_every_item(
    monkeypatch: pytest.MonkeyPatch, parity_items: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("AGENT_KEY", "test-agent-key")
    for item in parity_items:
        response = item["responses"]["l5"]
        calls = response["tool_calls"]
        script: list[dict[str, Any]] = [
            chat_completion(calls=[tool_call(f"call_{index}", call["name"], call["arguments"])])
            for index, call in enumerate(calls)
        ]
        script.append(chat_completion(response["content"]))
        transport, _ = scripted(script)

        result = run_agent(item["message"], transport=transport)

        assert result.error is None
        assert result.content == response["content"]
        assert [call["name"] for call in result.tool_calls] == [call["name"] for call in calls]
        assert [call["arguments"] for call in result.tool_calls] == [
            call["arguments"] for call in calls
        ]
