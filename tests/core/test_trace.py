"""Tracing: one CallRecord per call, scrubbed rows, LangSmith off without its key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from core_fakes import ScriptedChatModel
from langchain_core.messages import AIMessage

from which_ai_model.contracts import ModelRef
from which_ai_model.core.trace import (
    CallTraceHandler,
    TraceWriter,
    maybe_enable_langsmith,
    trace_path,
)

MODEL = ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash")
KEY = "a" * 64


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def now(self) -> float:
        self.value += 0.25
        return self.value


def _message() -> AIMessage:
    return AIMessage(
        content="the answer",
        usage_metadata={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        response_metadata={
            "finish_reason": "stop",
            "token_usage": {"opencode": {"list_cost": 0.00042}},
        },
    )


def _handler(path: Path, **overrides: Any) -> CallTraceHandler:
    options: dict[str, Any] = {
        "run_id": "run-1",
        "task": "T-ROUTE",
        "tool": "router",
        "item_id": "item-1",
        "model": MODEL,
        "prompt_version": "route-v1",
        "response_key": KEY,
        "clock": FakeClock().now,
    }
    options.update(overrides)
    return CallTraceHandler(TraceWriter(path), **options)


def test_one_call_record_per_call_with_tokens_and_measured_cost(tmp_path: Path) -> None:
    path = trace_path("run-1", mode="live", root=tmp_path)
    handler = _handler(path)
    model = ScriptedChatModel(script=[_message()])
    model.invoke("a prompt that must not be written", config={"callbacks": [handler]})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-1"
    assert row["task"] == "T-ROUTE"
    assert row["tool"] == "router"
    assert row["item_id"] == "item-1"
    assert row["model"] == {
        "connection_id": "opencode-go-openai",
        "model_id": "deepseek-v4.1-flash",
    }
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 7
    assert row["cost_usd"] == pytest.approx(0.00042)
    assert row["schema_valid"] is None
    assert row["error"] is None
    assert row["response_key"] == KEY
    assert isinstance(row["latency_ms"], int)
    assert row["latency_ms"] >= 0
    assert "a prompt that must not be written" not in path.read_text()


def test_failed_call_writes_an_error_row_not_a_secret(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    handler = _handler(path)
    model = ScriptedChatModel(script=[ValueError("Bearer sk-not-a-real-key")])
    with pytest.raises(ValueError, match="Bearer"):
        model.invoke("hi", config={"callbacks": [handler]})

    row = json.loads(path.read_text().splitlines()[0])
    assert row["error"] == "ValueError"
    assert row["schema_valid"] is None
    assert row["input_tokens"] is None
    assert "sk-not-a-real-key" not in path.read_text()


def test_langsmith_is_off_without_a_key_and_on_with_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert maybe_enable_langsmith() is False
    assert "LANGSMITH_TRACING" not in __import__("os").environ

    env: dict[str, str] = {}
    assert maybe_enable_langsmith(env) is False
    assert env == {}

    env = {"LANGSMITH_API_KEY": "lsv2_test_value"}
    assert maybe_enable_langsmith(env) is True
    assert env["LANGSMITH_TRACING"] == "true"
    assert env["LANGCHAIN_TRACING_V2"] == "true"


def test_trace_paths_keep_replay_and_live_apart(tmp_path: Path) -> None:
    assert trace_path("r", mode="replay", root=tmp_path) == tmp_path / "replay" / "r.jsonl"
    assert trace_path("r", mode="record", root=tmp_path) == tmp_path / "live" / "r.jsonl"
