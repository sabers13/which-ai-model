"""L1 router: raw httpx transport, first-line label parsing, old-style config."""

from __future__ import annotations

import json

import httpx
import pytest
from _helpers import chat_completion, scripted

from which_ai_model.legacy.l1_router import INTENTS, parse_label, route_message, run


def test_posts_to_chat_completions_with_own_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTER_KEY", "test-router-key")
    monkeypatch.setenv("ROUTER_MODEL", "router-test")
    monkeypatch.setenv("ROUTER_BASE_URL", "https://router.test/v1")
    transport, requests = scripted([chat_completion("card_arrival\nconfidence: high")])

    result = route_message("Where is my card?", transport=transport)

    assert result.error is None
    assert result.content == "card_arrival\nconfidence: high"
    assert result.latency_ms >= 0
    request = requests[0]
    assert str(request.url) == "https://router.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-router-key"
    payload = json.loads(request.content)
    assert payload["model"] == "router-test"
    assert "Where is my card?" in payload["messages"][0]["content"]


def test_run_returns_legacy_dict_with_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTER_KEY", "test-router-key")
    transport, _ = scripted([chat_completion("card_arrival\nbecause")])

    result = run("Where is my card?", transport=transport)

    assert set(result) == {"content", "tool_calls", "latency_ms", "error", "label"}
    assert result["label"] == "card_arrival"
    assert result["tool_calls"] == []
    assert result["error"] is None


def test_first_line_only_and_case_insensitive() -> None:
    assert parse_label("CARD_ARRIVAL\nsecond line") == "card_arrival"
    assert parse_label("refund_not_showing_up") == "Refund_not_showing_up"
    assert parse_label("reverted_card_payment?") == "reverted_card_payment?"
    assert parse_label("") is None


def test_exact_match_only_so_punctuation_misses() -> None:
    assert parse_label("card_arrival.") is None
    assert parse_label("I think this is card_arrival") is None


def test_the_label_list_has_77_unique_intents() -> None:
    assert len(INTENTS) == 77
    assert len({intent.lower() for intent in INTENTS}) == 77


def test_missing_key_is_an_error_before_any_request() -> None:
    result = route_message("hello", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert result.content == ""
    assert result.error == "ROUTER_KEY is not set"


def test_http_error_is_reported_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTER_KEY", "test-router-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "slow"}))
    result = route_message("hello", transport=transport)
    assert result.error == "API request failed (HTTP 429)"
    assert "test-router-key" not in (result.error or "")
