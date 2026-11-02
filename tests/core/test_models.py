"""Model layer: builder policy, retries with Retry-After, fallback on transport errors."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from core_fakes import FakeBuilder, ScriptedChatModel, StructuredFakeModel, answer_with_tool
from langchain_core.messages import AIMessage

from which_ai_model.contracts import Connection, ModelRef, RouteOut
from which_ai_model.core.models import get_model, make_agent, make_chat_model
from which_ai_model.core.structured import structured


class RateLimitError(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        self.status_code = 429
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(headers=headers)


class ConnectError(Exception):
    pass


class FakeClock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _refs() -> tuple[ModelRef, ModelRef]:
    return (
        ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash"),
        ModelRef(connection_id="opencode-go-anthropic", model_id="claude-haiku-5-5"),
    )


def test_make_chat_model_passes_policy_options(
    connection: Connection,
    fallback_connection: Connection,
) -> None:
    builder = FakeBuilder({"opencode-go-openai": ScriptedChatModel(script=[])})
    model = make_chat_model(
        connection,
        "deepseek-v4.1-flash",
        reasoning="minimal",
        max_tokens=2048,
        session_id="run-1",
        builder=builder,
    )
    assert model is builder.models["opencode-go-openai"]
    built_connection, model_id, options = builder.calls[0]
    assert built_connection is connection
    assert model_id == "deepseek-v4.1-flash"
    assert options == {
        "reasoning": "minimal",
        "max_tokens": 2048,
        "max_retries": 0,
        "temperature": 0.0,
        "session_id": "run-1",
    }


def test_make_chat_model_can_omit_temperature(connection: Connection) -> None:
    """A provider that rejects ``temperature`` gets no parameter at all (PLAN.md §7)."""
    builder = FakeBuilder({"opencode-go-openai": ScriptedChatModel(script=[])})
    make_chat_model(
        connection,
        "deepseek-v4.1-flash",
        temperature=None,
        builder=builder,
    )
    assert builder.calls[0][2]["temperature"] is None


def test_make_chat_model_enforces_the_1024_token_floor(connection: Connection) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        make_chat_model(connection, "m", max_tokens=512, builder=FakeBuilder())


def test_make_chat_model_defaults_to_connect_builder(
    monkeypatch: pytest.MonkeyPatch, connection: Connection
) -> None:
    module = ModuleType("which_ai_model.connect.to_langchain")
    builder = FakeBuilder({"opencode-go-openai": ScriptedChatModel(script=[])})
    module.chat_model = builder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "which_ai_model.connect.to_langchain", module)
    model = make_chat_model(connection, "deepseek-v4.1-flash")
    assert isinstance(model, ScriptedChatModel)
    assert builder.calls[0][2]["max_retries"] == 0


def test_retry_honours_retry_after(connection: Connection, fallback_connection: Connection) -> None:
    primary = ScriptedChatModel(script=[RateLimitError(7), AIMessage(content="finally")])
    builder = FakeBuilder({"opencode-go-openai": primary})
    clock = FakeClock()
    ref, _ = _refs()
    model = get_model(
        ref,
        connections=[connection, fallback_connection],
        builder=builder,
        sleep_fn=clock.sleep,
    )
    message = model.invoke("hello")
    assert message.content == "finally"
    assert clock.sleeps == [7.0]
    assert primary.calls == ["call", "call"]


def test_primary_retries_then_falls_back(
    connection: Connection, fallback_connection: Connection
) -> None:
    primary = ScriptedChatModel(script=[ConnectError("down") for _ in range(5)])
    backup = ScriptedChatModel(script=[AIMessage(content="from fallback")])
    builder = FakeBuilder({"opencode-go-openai": primary, "opencode-go-anthropic": backup})
    clock = FakeClock()
    ref, fallback = _refs()
    model = get_model(
        ref,
        fallback=fallback,
        connections=[connection, fallback_connection],
        builder=builder,
        sleep_fn=clock.sleep,
        base_delay=0.01,
    )
    assert model.invoke("hello").content == "from fallback"
    assert len(primary.calls) == 5
    assert backup.calls == ["call"]
    assert len(clock.sleeps) == 4
    assert all(delay >= 0 for delay in clock.sleeps)


def test_non_transport_error_is_raised_and_not_retried(
    connection: Connection, fallback_connection: Connection
) -> None:
    primary = ScriptedChatModel(script=[ValueError("bad request")])
    backup = ScriptedChatModel(script=[AIMessage(content="should not happen")])
    builder = FakeBuilder({"opencode-go-openai": primary, "opencode-go-anthropic": backup})
    ref, fallback = _refs()
    model = get_model(
        ref,
        fallback=fallback,
        connections=[connection, fallback_connection],
        builder=builder,
        sleep_fn=FakeClock().sleep,
    )
    with pytest.raises(ValueError, match="bad request"):
        model.invoke("hello")
    assert primary.calls == ["call"]
    assert backup.calls == []


def test_async_fallback(connection: Connection, fallback_connection: Connection) -> None:
    primary = ScriptedChatModel(script=[RateLimitError(0) for _ in range(5)])
    backup = ScriptedChatModel(script=[AIMessage(content="async fallback")])
    builder = FakeBuilder({"opencode-go-openai": primary, "opencode-go-anthropic": backup})
    ref, fallback = _refs()
    model = get_model(
        ref,
        fallback=fallback,
        connections=[connection, fallback_connection],
        builder=builder,
        base_delay=0.0,
    )
    message = asyncio.run(model.ainvoke("hello"))
    assert message.content == "async fallback"


def test_structured_output_keeps_the_retry_policy(
    connection: Connection, fallback_connection: Connection
) -> None:
    structured_primary = StructuredFakeModel(reply=RateLimitError(0))
    backup = StructuredFakeModel(reply=answer_with_tool("RouteOut", {"intent": "card_arrival"}))
    builder = FakeBuilder(
        {"opencode-go-openai": structured_primary, "opencode-go-anthropic": backup}
    )
    ref, fallback = _refs()
    model = get_model(
        ref,
        fallback=fallback,
        connections=[connection, fallback_connection],
        builder=builder,
        base_delay=0.0,
    )
    parsed, valid, _raw = structured(model, RouteOut, "my card never arrived")
    assert valid is True
    assert isinstance(parsed, RouteOut)
    assert parsed.intent == "card_arrival"


def test_unknown_connection_is_a_clear_error(connection: Connection) -> None:
    ref = ModelRef(connection_id="does-not-exist", model_id="m")
    with pytest.raises(KeyError, match="does-not-exist"):
        get_model(ref, connections=[connection], builder=FakeBuilder())


def test_make_agent_builds_offline(connection: Connection, fallback_connection: Connection) -> None:
    builder = FakeBuilder(
        {
            "opencode-go-openai": StructuredFakeModel(),
            "opencode-go-anthropic": StructuredFakeModel(),
        }
    )
    agent = make_agent(
        connection,
        fallback_connection,
        "deepseek-v4.1-flash",
        [],
        system_prompt="Route the message.",
        response_format=RouteOut,
        builder=builder,
    )
    assert agent is not None
    assert len(builder.calls) == 2


def test_trace_rows_paths_are_lane_specific(tmp_path: Path) -> None:
    from which_ai_model.core.trace import trace_path

    assert trace_path("run-1", mode="replay", root=tmp_path) == tmp_path / "replay" / "run-1.jsonl"
    assert trace_path("run-1", mode="live", root=tmp_path) == tmp_path / "live" / "run-1.jsonl"
