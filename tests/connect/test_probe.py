"""Probe mapping: one model-list request per protocol, fixed reason vocabulary."""

from __future__ import annotations

import httpx
import pytest

from which_ai_model.connect.presets import GATEWAY_MODELS_URL, USER_AGENT, preset_connections
from which_ai_model.connect.probe import models_url, probe_connection
from which_ai_model.contracts import Connection, ProbeReason, Protocol

SECRET = "sk-" + "test-not-a-real-key-1234567890"


def _transport(
    status: int = 200,
    payload: object | None = None,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=payload if payload is not None else {"data": []})

    return httpx.MockTransport(handler)


def _raising(error: Exception, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        raise error

    return httpx.MockTransport(handler)


def _openai() -> Connection:
    return preset_connections("openai")[0]


@pytest.mark.parametrize(
    ("status", "payload", "reason"),
    [
        (200, {"data": [{"id": "gpt-6-luna"}]}, ProbeReason.OK),
        (400, {"error": "bad"}, ProbeReason.CONFIG_INVALID),
        (401, {"error": "no"}, ProbeReason.CREDENTIAL_REJECTED),
        (403, {"error": "no"}, ProbeReason.CREDENTIAL_REJECTED),
        (404, {"error": "no"}, ProbeReason.CONFIG_INVALID),
        (405, {"error": "no"}, ProbeReason.PROTOCOL_UNSUPPORTED),
        (418, {"error": "teapot"}, ProbeReason.UNKNOWN),
        (429, {"error": "slow"}, ProbeReason.RATE_LIMITED),
        (500, {"error": "boom"}, ProbeReason.ENDPOINT_UNREACHABLE),
    ],
)
def test_status_to_reason(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: dict[str, str],
    reason: ProbeReason,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    result = probe_connection(_openai(), transport=_transport(status, payload))
    assert result.reason is reason
    assert result.connection_id == "openai"
    assert result.latency_ms >= 0


def test_transport_errors_are_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    connect = probe_connection(_openai(), transport=_raising(httpx.ConnectError("nope")))
    assert connect.reason is ProbeReason.ENDPOINT_UNREACHABLE
    assert connect.detail == "ConnectError"

    timeout = probe_connection(_openai(), transport=_raising(httpx.ReadTimeout("slow")))
    assert timeout.reason is ProbeReason.ENDPOINT_UNREACHABLE
    assert timeout.detail == "timeout"


def test_missing_key_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected request to {request.url}")

    result = probe_connection(_openai(), transport=httpx.MockTransport(handler))
    assert result.reason is ProbeReason.CREDENTIAL_MISSING
    assert result.latency_ms == 0


def test_missing_base_url_is_config_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUSTOM_API_KEY", SECRET)
    custom = preset_connections("custom")[0]
    result = probe_connection(custom, transport=_transport())
    assert result.reason is ProbeReason.CONFIG_INVALID


def test_gateway_lists_on_the_shared_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    seen: list[httpx.Request] = []
    transport = _transport(200, {"data": [{"id": "gpt-6-luna"}]}, seen)
    for connection in preset_connections("opencode-go"):
        result = probe_connection(connection, session_id="run-1", transport=transport)
        assert result.reason is ProbeReason.OK
    assert {str(request.url) for request in seen} == {GATEWAY_MODELS_URL}
    assert all(
        not str(request.url).startswith("https://opencode.ai/inference/go/openai/v1/models")
        for request in seen
    )


def test_gateway_headers_are_identical_within_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    seen: list[httpx.Request] = []
    transport = _transport(200, {"data": []}, seen)
    connection = preset_connections("opencode-go")[0]
    probe_connection(connection, session_id="run-2026-10-28", transport=transport)
    probe_connection(connection, session_id="run-2026-10-28", transport=transport)
    probe_connection(connection, session_id="run-other", transport=transport)
    assert [request.headers["x-opencode-session"] for request in seen] == [
        "run-2026-10-28",
        "run-2026-10-28",
        "run-other",
    ]
    assert {request.headers["user-agent"] for request in seen} == {USER_AGENT}


def test_gateway_probes_with_the_key_and_reports_a_wrong_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    seen: list[httpx.Request] = []
    result = probe_connection(
        preset_connections("opencode-go")[2],
        session_id="run",
        transport=_transport(401, seen=seen),
    )
    assert result.reason is ProbeReason.CREDENTIAL_REJECTED
    assert result.detail == "HTTP 401"
    assert SECRET not in repr(result)
    assert seen[0].headers["x-api-key"] == SECRET
    assert SECRET not in str(result.detail)


@pytest.mark.parametrize(
    ("preset", "env_name", "url", "header"),
    [
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/models", "authorization"),
        ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/models", "x-api-key"),
        (
            "gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/models",
            "x-goog-api-key",
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1/models",
            "authorization",
        ),
    ],
)
def test_first_party_auth_headers_and_urls(
    monkeypatch: pytest.MonkeyPatch, preset: str, env_name: str, url: str, header: str
) -> None:
    monkeypatch.setenv(env_name, SECRET)
    seen: list[httpx.Request] = []
    connection = preset_connections(preset)[0]
    assert models_url(connection) == url
    result = probe_connection(connection, transport=_transport(200, {"data": []}, seen))
    assert result.reason is ProbeReason.OK
    assert str(seen[0].url) == url
    assert SECRET in seen[0].headers[header]
    assert "x-opencode-session" not in seen[0].headers


def test_models_url_handles_versioned_anthropic_bases() -> None:
    connection = Connection(
        id="custom-anthropic",
        provider="custom",
        protocol=Protocol.ANTHROPIC,
        base_url="https://example.test/v1",
        credential_ref="env:CUSTOM_API_KEY",
        preset="custom",
    )
    assert models_url(connection) == "https://example.test/v1/models"
