"""Preset -> LangChain class mapping, constructed offline."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from which_ai_model.connect.presets import USER_AGENT, preset_connections
from which_ai_model.connect.to_langchain import chat_model
from which_ai_model.contracts import Connection, Protocol

SECRET = "sk-" + "test-not-a-real-key-1234567890"

# ChatOpenAI probes OS socket support by creating (not connecting) a socket.
pytestmark = pytest.mark.filterwarnings("ignore:A test tried to use socket.socket:UserWarning")


def _gateway() -> dict[Protocol, Connection]:
    return {connection.protocol: connection for connection in preset_connections("opencode-go")}


def test_openai_compatible_maps_to_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(
        _gateway()[Protocol.OPENAI_COMPATIBLE], "deepseek-v4.1-flash", session_id="run-1"
    )
    assert isinstance(model, ChatOpenAI)
    assert model.use_responses_api in (False, None)
    assert model.openai_api_base == "https://opencode.ai/inference/go/openai/v1"
    assert model.default_headers is not None
    assert model.default_headers["x-opencode-session"] == "run-1"
    assert model.default_headers["User-Agent"] == USER_AGENT
    assert SECRET not in repr(model)


def test_responses_maps_to_chat_openai_in_responses_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(_gateway()[Protocol.OPENAI_RESPONSES], "gpt-6-luna", session_id="run-1")
    assert isinstance(model, ChatOpenAI)
    assert model.use_responses_api is True


def test_anthropic_maps_to_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(_gateway()[Protocol.ANTHROPIC], "claude-haiku-5-5", session_id="run-1")
    assert isinstance(model, ChatAnthropic)
    assert str(model.anthropic_api_url).startswith("https://opencode.ai/inference/go/anthropic")


def test_gemini_maps_to_chat_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(_gateway()[Protocol.GEMINI], "gemini-4-argon", session_id="run-1")
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert SECRET not in repr(model)


def test_keyless_local_connection_still_builds() -> None:
    model = chat_model(preset_connections("local-openai-compatible")[0], "llama3", reasoning=None)
    assert isinstance(model, ChatOpenAI)


def test_options_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(
        _gateway()[Protocol.OPENAI_COMPATIBLE],
        "glm-5.3-flash",
        max_tokens=2048,
        timeout_s=30.0,
        session_id="run",
    )
    assert isinstance(model, ChatOpenAI)
    assert model.max_tokens == 2048


PROTOCOLS = [
    Protocol.OPENAI_COMPATIBLE,
    Protocol.OPENAI_RESPONSES,
    Protocol.ANTHROPIC,
    Protocol.GEMINI,
]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_temperature_is_zero_on_every_protocol(
    protocol: Protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PLAN.md`` §7: temperature 0 wherever the API allows, on every wire format."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(_gateway()[protocol], "some-model", session_id="run")
    assert model.temperature == 0.0
    assert "temperature" in model.model_fields_set


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_temperature_can_be_left_out(protocol: Protocol, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that rejects the parameter is told so explicitly with ``None``."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", SECRET)
    model = chat_model(_gateway()[protocol], "some-model", temperature=None, session_id="run")
    # Not passed at all: the provider's own default applies (Gemini's class default is
    # 0.7), which is what "where the API allows" means for a provider that rejects it.
    assert "temperature" not in model.model_fields_set
