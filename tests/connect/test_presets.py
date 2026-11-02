"""Presets: four gateway connections on one key, plus the first-party presets."""

from __future__ import annotations

from which_ai_model.connect.presets import (
    BENCHMARK_MODELS,
    GATEWAY_MODELS_URL,
    USER_AGENT,
    all_presets,
    connection_headers,
    default_connections,
    preset_connections,
)
from which_ai_model.contracts import Protocol


def test_gateway_preset_has_four_protocols_on_one_key() -> None:
    connections = preset_connections("opencode-go")
    assert [c.id for c in connections] == [
        "opencode-go-openai",
        "opencode-go-responses",
        "opencode-go-anthropic",
        "opencode-go-google",
    ]
    assert {c.protocol for c in connections} == set(Protocol)
    assert {c.credential_ref for c in connections} == {"env:OPENCODE_GO_API_KEY"}
    assert {c.preset for c in connections} == {"opencode-go"}


def test_gateway_base_urls_match_the_notes() -> None:
    by_id = {c.id: c for c in preset_connections("opencode-go")}
    assert by_id["opencode-go-openai"].base_url == "https://opencode.ai/inference/go/openai/v1"
    assert by_id["opencode-go-responses"].base_url == "https://opencode.ai/inference/go/openai/v1"
    assert by_id["opencode-go-anthropic"].base_url == "https://opencode.ai/inference/go/anthropic"
    assert by_id["opencode-go-google"].base_url == "https://opencode.ai/inference/go/google/"


def test_every_preset_is_declared() -> None:
    assert set(all_presets()) == {
        "opencode-go",
        "openai",
        "anthropic",
        "gemini",
        "openrouter",
        "local-openai-compatible",
        "custom",
    }
    assert preset_connections("local-openai-compatible")[0].credential_ref is None
    assert preset_connections("custom")[0].base_url is None


def test_gateway_headers_carry_user_agent_and_session() -> None:
    connection = preset_connections("opencode-go")[0]
    headers = connection_headers(connection, "run-2026-10-28")
    assert headers == {
        "User-Agent": USER_AGENT,
        "x-opencode-session": "run-2026-10-28",
    }
    assert "x-opencode-session" not in connection_headers(preset_connections("openai")[0], "run")


def test_default_connections_leave_the_keyless_local_preset_opt_in() -> None:
    ids = {c.id for c in default_connections()}
    assert "local-openai-compatible" not in ids
    assert "local-openai-compatible" in {c.id for c in default_connections(include_local=True)}


def test_benchmark_models_and_gateway_endpoint_are_fixed() -> None:
    assert len(BENCHMARK_MODELS) == 7
    assert BENCHMARK_MODELS[0] == "claude-haiku-5-5"
    assert GATEWAY_MODELS_URL == "https://opencode.ai/inference/go/v1/models"
