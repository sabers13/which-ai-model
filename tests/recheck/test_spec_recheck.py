"""A spec-task re-check dispatches to the §A5 gate, not the v1 cell gate (M11, S7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from which_ai_model.bench.score import SpecCellResult, write_cells
from which_ai_model.contracts import ModelRef, RunManifest
from which_ai_model.gate import GateOutcome
from which_ai_model.recheck import GateRequest, SettingsCheck, evaluate_default_gate

MODEL = "model-a"
STARTED = datetime(2026, 11, 1, tzinfo=UTC)


def _write_run(path: Path, primary: float) -> Path:
    path.mkdir(parents=True)
    manifest = RunManifest(
        run_id=path.name,
        sample_hash="a" * 64,
        prompt_version="d1",
        scorer_version="s1",
        models=[ModelRef(connection_id="fixture-openai", model_id=MODEL)],
        started_at=STARTED,
        finished_at=STARTED,
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cell = SpecCellResult(
        task="defect",
        model=ModelRef(connection_id="fixture-openai", model_id=MODEL),
        model_id=MODEL,
        n=20,
        primary_metric=primary,
        metrics={
            "safety_relevant.recall": 0.95,
            "safety_relevant.precision": 0.90,
            "schema_valid_rate": 0.99,
        },
        ci={},
        latency_p50=10.0,
        latency_p95=20.0,
        cost_per_1k=0.1,
    )
    write_cells(path / "cells.json", [], spec_cells=[cell])
    return path


def test_a_spec_task_recheck_uses_the_spec_gate(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", 0.80)
    candidate = _write_run(tmp_path / "candidate", 0.79)
    request = GateRequest(
        baseline=baseline,
        candidate=candidate,
        tasks=("defect",),
        baseline_model=MODEL,
        candidate_model=MODEL,
        settings=SettingsCheck(comparable=True),
    )

    verdict = evaluate_default_gate(request)

    assert verdict.outcome is GateOutcome.PASS
    assert verdict.rules
    assert any(rule["name"] == "primary_drop" for rule in verdict.rules)


def test_a_dropped_spec_cell_fails_the_recheck_gate(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", 0.90)
    candidate = _write_run(tmp_path / "candidate", 0.80)
    request = GateRequest(
        baseline=baseline,
        candidate=candidate,
        tasks=("defect",),
        baseline_model=MODEL,
        candidate_model=MODEL,
        settings=SettingsCheck(comparable=True),
    )

    verdict = evaluate_default_gate(request)

    assert verdict.outcome is GateOutcome.FAIL
    assert any(rule["name"] == "primary_drop" and not rule["passed"] for rule in verdict.rules)
