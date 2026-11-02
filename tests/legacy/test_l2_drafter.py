"""L2 drafter: Anthropic SDK with a mocked transport and a regex JSON extraction."""

from __future__ import annotations

import json

import httpx2
import pytest

from which_ai_model.legacy.l2_drafter import draft_reply, extract_json, run


def anthropic_client(handler: object) -> httpx2.Client:
    transport = httpx2.MockTransport(handler)  # type: ignore[arg-type]
    return httpx2.Client(transport=transport)


def message_payload(text: str) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 8},
    }


def test_uses_sdk_with_own_env_and_extracts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DRAFTER_MODEL", "claude-test")
    monkeypatch.setenv("DRAFTER_BASE_URL", "https://drafter.test")
    raw = 'Sure:\n{"intent": "card_arrival", "urgency": "low", "reply": "It is on its way."}'
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=message_payload(raw))

    result = draft_reply("Where is my card?", httpx_client=anthropic_client(handler))

    assert result.error is None
    assert result.content == raw
    assert extract_json(result.content) == {
        "intent": "card_arrival",
        "urgency": "low",
        "reply": "It is on its way.",
    }
    assert str(seen[0].url) == "https://drafter.test/v1/messages"
    assert json.loads(seen[0].content)["model"] == "claude-test"


def test_run_returns_parsed_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    raw = '{"intent": "change_pin", "urgency": "medium", "reply": "Open the app."}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message_payload(raw))

    result = run("Change my PIN", httpx_client=anthropic_client(handler))

    assert result["parsed"]["intent"] == "change_pin"
    assert set(result) == {"content", "tool_calls", "latency_ms", "error", "parsed"}


def test_unparseable_text_is_content_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message_payload("I cannot do that."))

    result = run("hello", httpx_client=anthropic_client(handler))

    assert result["error"] is None
    assert result["parsed"] is None
    assert result["content"] == "I cannot do that."


def test_brittle_regex_rejects_nested_or_broken_json() -> None:
    assert extract_json('{"a": {"b": 1}}') is None
    assert extract_json("{not json}") is None
    assert extract_json("no braces") is None
    assert extract_json("") is None


def test_api_error_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"type": "invalid_request", "message": "no"}})

    result = draft_reply("hello", httpx_client=anthropic_client(handler))

    assert result.error is not None
    assert "test-anthropic-key" not in result.error


def test_missing_key_never_builds_a_client() -> None:
    result = draft_reply("hello")
    assert result.error == "ANTHROPIC_API_KEY is not set"
