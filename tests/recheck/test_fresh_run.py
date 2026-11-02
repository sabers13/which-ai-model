"""A fresh run writes only under the fresh root; a v1 replay keeps its old manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import (
    MODEL_ID,
    run_kwargs,
    scope_invoker,
    write_config,
    write_samples,
    write_scope_items,
)

from which_ai_model.bench.prompts import PROMPT_VERSION, prompt_for
from which_ai_model.bench.run import run
from which_ai_model.contracts import TaskName
from which_ai_model.core.record import (
    Recorder,
    recording_key,
    request_digest,
)

V1_MANIFEST_KEYS = {
    "run_id",
    "sample_hash",
    "prompt_version",
    "scorer_version",
    "models",
    "started_at",
    "finished_at",
}


def _saved_row(tmp_path: Path) -> Path:
    """A saved, valid-looking answer for the exact key a fresh run would use."""
    item = {"item_id": "s0", "text": "message 0", "out_of_scope": True}
    digest = request_digest(list(prompt_for(TaskName.SCOPE).messages(item)))
    key = recording_key(MODEL_ID, PROMPT_VERSION, digest)
    path = tmp_path / "recordings" / "T-SCOPE" / f"{MODEL_ID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "key": key,
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "request_digest": digest,
        "response": {
            "content": json.dumps({"out_of_scope": True}),
            "tool_calls": [],
            "usage": {"input_tokens": 99, "output_tokens": 99},
        },
        "recorded_at": "2026-10-29T00:00:00Z",
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_fresh_run_never_reads_the_saved_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_config(tmp_path)
    write_samples(tmp_path)
    items_dir = write_scope_items(tmp_path)
    saved_row = _saved_row(tmp_path)
    saved_root = tmp_path / "recordings"
    fresh_root = saved_root / "fresh"
    fresh_run = fresh_root / "fresh-run"

    seen: list[tuple[Path, str]] = []
    original = Recorder.lookup

    def spy(self: Recorder, key: str, *, task: str | None = None) -> dict[str, object] | None:
        seen.append((self.root, key))
        return original(self, key, task=task)

    monkeypatch.setattr(Recorder, "lookup", spy)
    invoker = scope_invoker(items_dir)
    run(
        "record",
        run_id="fresh-run",
        tasks=[TaskName.SCOPE],
        invoker=invoker,
        workers=1,
        fresh=True,
        fresh_root=fresh_root,
        **run_kwargs(tmp_path, config),
    )

    assert len(invoker.calls) == 4  # type: ignore[attr-defined]
    assert seen
    assert {root for root, _key in seen} == {fresh_run}
    assert saved_root not in {root for root, _key in seen}
    assert len(saved_row.read_text(encoding="utf-8").splitlines()) == 1  # untouched
    rows = (fresh_run / "T-SCOPE" / f"{MODEL_ID}.jsonl").read_text(encoding="utf-8")
    assert len(rows.splitlines()) == 4


def test_replay_manifest_keeps_the_v1_shape(tmp_path: Path, repo_root: Path) -> None:
    fixture = repo_root / "tests" / "fixtures" / "bench"
    manifest = run(
        "replay",
        run_id="replay-v1",
        config=fixture / "config",
        models_path=fixture / "models.lock.toml",
        tasks=[TaskName.ROUTE, TaskName.SCOPE, TaskName.REPLY, TaskName.DOCS],
        recordings_dir=fixture / "recordings",
        samples_dir=fixture / "samples",
        items_dir=fixture / "items",
        runs_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports",
        baseline_metrics=fixture / "baseline-metrics.json",
        recommendations_path=tmp_path / "recommendations.json",
    )
    document = json.loads((tmp_path / "runs" / "replay-v1" / "manifest.json").read_text())
    assert set(document) == V1_MANIFEST_KEYS
    assert "settings" not in document
    assert "lock_sha256" not in document
    assert manifest.run_id == "replay-v1"
