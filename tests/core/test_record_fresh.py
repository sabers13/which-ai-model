"""The fresh recording root: never serve a saved answer, replay it when asked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from which_ai_model.core.record import (
    DEFAULT_FRESH_RECORDINGS_ROOT,
    Recorder,
    RecordingMode,
    fresh_recording_root,
    recording_key,
    request_digest,
)

DIGEST = request_digest([{"role": "user", "content": "is this out of scope?"}])
KEY = recording_key("fake-a", "v1", DIGEST)


def _record(root: Path, *, content: str, reuse: bool = True) -> Recorder:
    recorder = Recorder(root, mode=RecordingMode.RECORD, task="T-SCOPE", reuse=reuse)
    recorder.record(
        KEY,
        "fake-a",
        "v1",
        DIGEST,
        {"content": content, "tool_calls": [], "usage": {"input_tokens": 4}},
    )
    return recorder


def test_fresh_root_is_one_safe_path_component(tmp_path: Path) -> None:
    assert fresh_recording_root("recheck-demo", root=tmp_path) == tmp_path / "recheck-demo"
    assert fresh_recording_root("run.1_a-b", root=tmp_path) == tmp_path / "run.1_a-b"
    assert Path("recordings/fresh") == DEFAULT_FRESH_RECORDINGS_ROOT
    for bad in ("", ".", "..", "a/b", "../x", "a b", "-leading"):
        with pytest.raises(ValueError, match="invalid fresh run id"):
            fresh_recording_root(bad, root=tmp_path)


def test_fresh_recorder_never_serves_a_saved_row(tmp_path: Path) -> None:
    _record(tmp_path, content="saved answer")
    fresh = Recorder(tmp_path, mode=RecordingMode.RECORD, task="T-SCOPE", reuse=False)
    assert fresh.lookup(KEY) is None
    _record(tmp_path, content="fresh answer", reuse=False)
    assert fresh.lookup(KEY) is None
    # The default recorder is unchanged: it still serves the last saved row.
    assert Recorder(tmp_path, mode=RecordingMode.RECORD).lookup(KEY)["content"] == "fresh answer"
    assert Recorder(tmp_path, mode=RecordingMode.RECORD).reuse is True


def test_fresh_recordings_are_replayable(tmp_path: Path) -> None:
    _record(tmp_path, content="fresh answer", reuse=False)
    replay = Recorder(tmp_path, mode=RecordingMode.REPLAY, reuse=False)
    assert replay.lookup(KEY)["content"] == "fresh answer"


def test_default_row_shape_is_unchanged(tmp_path: Path) -> None:
    _record(tmp_path, content="saved answer")
    row = json.loads((tmp_path / "T-SCOPE" / "fake-a.jsonl").read_text(encoding="utf-8"))
    assert set(row) == {
        "key",
        "model_id",
        "prompt_version",
        "request_digest",
        "response",
        "recorded_at",
    }
