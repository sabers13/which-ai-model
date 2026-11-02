"""Manifest settings: the v1 body stays intact, old and new manifests both load."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from support import (
    MODEL_ID,
    run_kwargs,
    scope_invoker,
    write_config,
    write_samples,
    write_scope_items,
)

from which_ai_model.bench.run import manifest_settings, run, write_manifest_settings
from which_ai_model.contracts import ModelRef, RunManifest, RunSettings, TaskName

V1_MANIFEST_KEYS = {
    "run_id",
    "sample_hash",
    "prompt_version",
    "scorer_version",
    "models",
    "started_at",
    "finished_at",
}


def _v1_manifest() -> RunManifest:
    return RunManifest(
        run_id="live",
        sample_hash="s" * 64,
        prompt_version="v1",
        scorer_version="v1",
        models=[ModelRef(connection_id="fake", model_id=MODEL_ID)],
        started_at=datetime(2026, 10, 29, tzinfo=UTC),
        finished_at=datetime(2026, 10, 29, 0, 10, tzinfo=UTC),
    )


def test_write_manifest_settings_preserves_every_other_key(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(_v1_manifest().model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = json.loads(path.read_text(encoding="utf-8"))
    settings = {MODEL_ID: RunSettings(temperature=0.0, reasoning="none", max_output_tokens=2048)}

    write_manifest_settings(path, settings, lock_sha256="a" * 64)

    after = json.loads(path.read_text(encoding="utf-8"))
    for key, value in before.items():
        assert after[key] == value
    assert after["settings"] == manifest_settings(settings)
    assert after["lock_sha256"] == "a" * 64
    assert set(before) == V1_MANIFEST_KEYS

    parsed = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert parsed.settings == manifest_settings(settings)
    assert parsed.lock_sha256 == "a" * 64
    # The v1 serialisation shape is untouched: a dump excludes the v1.1 additions.
    assert set(parsed.model_dump(mode="json")) == V1_MANIFEST_KEYS


def test_old_manifest_without_settings_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(_v1_manifest().model_dump_json(), encoding="utf-8")
    parsed = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert parsed.settings == {}
    assert parsed.lock_sha256 is None


def test_record_run_writes_settings_and_the_lock_hash(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    write_samples(tmp_path)
    items_dir = write_scope_items(tmp_path)
    invoker = scope_invoker(items_dir)
    run(
        "record",
        run_id="recorded",
        tasks=[TaskName.SCOPE],
        invoker=invoker,
        workers=1,
        **run_kwargs(tmp_path, config),
    )
    manifest_path = tmp_path / "runs" / "recorded" / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["settings"] == {
        MODEL_ID: {
            "temperature": 0.0,
            "reasoning": "minimal",
            "max_output_tokens": 1024,
        }
    }
    lock = config / "models.lock.toml"
    assert document["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
