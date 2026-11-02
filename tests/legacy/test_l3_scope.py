"""L3 scope: google-genai client with a mocked httpx transport, substring parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from which_ai_model.legacy.l3_scope import IN_SCOPE, OUT_OF_SCOPE, check_scope, parse_scope, run


def scope_client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def gemini_payload(text: str) -> dict[str, object]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
    }


def test_calls_genai_with_own_env_and_reads_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("SCOPE_MODEL", "gemini-test")
    monkeypatch.setenv("SCOPE_BASE_URL", "https://scope.test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=gemini_payload("Yes, that is supported."))

    result = check_scope("My card has not arrived", httpx_client=scope_client(handler))

    assert result.error is None
    assert result.content == "Yes, that is supported."
    assert str(seen[0].url) == "https://scope.test/v1beta/models/gemini-test:generateContent"
    assert (
        "My card has not arrived" in json.loads(seen[0].content)["contents"][0]["parts"][0]["text"]
    )


def test_no_means_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=gemini_payload("No, that is outside the scope."))

    assert run("Book a flight", httpx_client=scope_client(handler))["label"] == OUT_OF_SCOPE


def test_brittle_substring_rules() -> None:
    assert parse_scope("YES") == IN_SCOPE
    assert parse_scope("yes, but no fee") == IN_SCOPE
    assert parse_scope("No idea, sorry") == OUT_OF_SCOPE
    assert parse_scope("maybe") is None
    assert parse_scope("") is None


def test_api_error_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500, "message": "boom"}})

    result = check_scope("hello", httpx_client=scope_client(handler))
    assert result.error is not None
    assert "test-gemini-key" not in result.error


def test_missing_key_is_an_error() -> None:
    assert check_scope("hello").error == "GEMINI_API_KEY is not set"
