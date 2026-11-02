"""Record/replay: identical objects, replay miss never reaches a model, scrubbed rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from core_fakes import answer_with_tool
from langchain_core.messages import AIMessage

from which_ai_model.contracts import RouteOut
from which_ai_model.core.network_guard import NetworkGuard
from which_ai_model.core.record import (
    Recorder,
    RecordingMode,
    ReplayMiss,
    recording_key,
    replay_model,
    request_digest,
    response_from_message,
    scrub_response,
)
from which_ai_model.core.structured import structured

DIGEST = request_digest([{"role": "user", "content": "my card never arrived"}])
KEY = recording_key("deepseek-v4.1-flash", "route-v1", DIGEST)


def _record(tmp_path: Path, key: str = KEY, content: str = "first") -> Recorder:
    recorder = Recorder(tmp_path, mode=RecordingMode.RECORD, task="T-ROUTE")
    recorder.record(
        key,
        "deepseek-v4.1-flash",
        "route-v1",
        DIGEST,
        {"content": content, "tool_calls": [], "usage": {"input_tokens": 4}},
    )
    return recorder


def test_record_then_replay_returns_identical_object(tmp_path: Path) -> None:
    _record(tmp_path)
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    expected = {"content": "first", "tool_calls": [], "usage": {"input_tokens": 4}}
    assert replay.lookup(KEY) == expected
    assert replay.lookup(KEY) == expected


def test_last_matching_row_wins(tmp_path: Path) -> None:
    _record(tmp_path)
    _record(tmp_path, content="second")
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    assert replay.lookup(KEY)["content"] == "second"


def test_a_fresh_root_under_the_default_root_does_not_shadow(tmp_path: Path) -> None:
    """A fresh re-check row answers the same key; it must never change the v1 replay."""
    _record(tmp_path, content="v1 answer")
    fresh = tmp_path / "fresh" / "recheck-demo" / "T-ROUTE"
    fresh.mkdir(parents=True)
    (fresh / "deepseek-v4-1-flash.jsonl").write_text(
        json.dumps(
            {
                "key": KEY,
                "model_id": "deepseek-v4.1-flash",
                "prompt_version": "route-v1",
                "request_digest": DIGEST,
                "response": {"content": "fresh answer", "tool_calls": [], "usage": {}},
                "recorded_at": "2026-11-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    assert replay.lookup(KEY, task="T-ROUTE")["content"] == "v1 answer"


def test_replay_miss_raises_and_never_calls_a_model(tmp_path: Path) -> None:
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    with pytest.raises(ReplayMiss):
        replay.lookup("0" * 64)
    with pytest.raises(ReplayMiss):
        replay_model(replay, "0" * 64, "deepseek-v4.1-flash")


def test_replay_model_opens_no_socket(tmp_path: Path) -> None:
    _record(tmp_path)
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    model = replay_model(replay, KEY, "deepseek-v4.1-flash")
    with NetworkGuard() as guard:
        message = model.invoke("my card never arrived")
    assert message.content == "first"
    assert guard.attempts == []


def test_replay_serves_a_structured_answer(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, mode=RecordingMode.RECORD, task="T-ROUTE")
    message = answer_with_tool("RouteOut", {"intent": "card_arrival"})
    recorder.record_message(KEY, "deepseek-v4.1-flash", "route-v1", DIGEST, message)
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY)
    model = replay_model(replay, KEY, "deepseek-v4.1-flash")
    parsed, valid, _raw = structured(model, RouteOut, "my card never arrived")
    assert isinstance(parsed, RouteOut)
    assert valid is True
    assert parsed.intent == "card_arrival"


def test_rows_are_scrubbed_to_content_tool_calls_and_usage(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, mode=RecordingMode.RECORD, task="T-ROUTE")
    recorder.record(
        KEY,
        "deepseek-v4.1-flash",
        "route-v1",
        DIGEST,
        {
            "content": "ok",
            "tool_calls": [{"name": "RouteOut", "args": {"intent": "x"}}],
            "usage": {
                "input_tokens": 4,
                "authorization": "Bearer sk-not-a-real-key",
                "headers": {"x-api-key": "nope"},
                "opencode": {"list_cost": 0.1},
            },
            "base_url": "https://user:pass@example.invalid/v1",
        },
    )
    row = json.loads((tmp_path / "T-ROUTE" / "deepseek-v4-1-flash.jsonl").read_text())
    assert set(row) == {
        "key",
        "model_id",
        "prompt_version",
        "request_digest",
        "response",
        "recorded_at",
    }
    assert set(row["response"]) == {"content", "tool_calls", "usage"}
    text = json.dumps(row)
    assert "authorization" not in text
    assert "sk-not-a-real-key" not in text
    assert "user:pass" not in text
    assert row["response"]["usage"]["opencode"]["list_cost"] == 0.1


def test_file_path_uses_task_and_model_slug(tmp_path: Path) -> None:
    recorder = _record(tmp_path)
    assert recorder.root == tmp_path
    assert (tmp_path / "T-ROUTE" / "deepseek-v4-1-flash.jsonl").is_file()
    assert len((tmp_path / "T-ROUTE" / "deepseek-v4-1-flash.jsonl").read_text().splitlines()) == 1


def test_live_mode_writes_nothing_and_does_not_raise_on_miss(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, mode=RecordingMode.LIVE)
    assert recorder.record(KEY, "m", "v1", DIGEST, {"content": "x"}) is None
    assert not list(tmp_path.rglob("*.jsonl"))
    assert recorder.lookup(KEY) is None


def test_recording_in_replay_mode_is_refused(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, mode=RecordingMode.REPLAY, task="T-ROUTE")
    with pytest.raises(RuntimeError):
        recorder.record(KEY, "m", "v1", DIGEST, {"content": "x"})


def test_key_material_is_documented_hash() -> None:
    assert hashlib.sha256(f"deepseek-v4.1-flash\nroute-v1\n{DIGEST}".encode()).hexdigest() == KEY
    assert len(KEY) == 64
    assert request_digest({"content": "x", "role": "user"}) == request_digest(
        {"role": "user", "content": "x"}
    )


def test_response_from_message_keeps_usage_and_drops_headers() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        response_metadata={
            "token_usage": {
                "opencode": {"list_cost": 0.02},
                "headers": {"authorization": "Bearer nope"},
            }
        },
    )
    response = response_from_message(message)
    assert response["content"] == "ok"
    assert response["usage"]["opencode"]["list_cost"] == 0.02
    assert "headers" not in response["usage"]
    assert scrub_response({"content": None, "tool_calls": [], "usage": {}}) == {
        "content": None,
        "tool_calls": [],
        "usage": {},
    }
