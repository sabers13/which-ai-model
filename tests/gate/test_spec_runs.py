"""`wam gate` on a spec-only run (M11, S7/S8): spec cells, manifest comparability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from which_ai_model.bench.score import SpecCellResult, write_cells
from which_ai_model.cli import app
from which_ai_model.contracts import ModelRef, RunManifest
from which_ai_model.gate import GateOutcome, evaluate_spec_gate, evaluate_spec_runs

MODEL = "model-a"
STARTED = datetime(2026, 11, 1, tzinfo=UTC)


def _cell(primary: float = 0.79, *, model_id: str = MODEL) -> SpecCellResult:
    return SpecCellResult(
        task="defect",
        model=ModelRef(connection_id="fixture-openai", model_id=model_id),
        model_id=model_id,
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


def _manifest(**overrides: str) -> RunManifest:
    values = {
        "run_id": "run",
        "sample_hash": "a" * 64,
        "prompt_version": "d1",
        "scorer_version": "s1",
        "models": [ModelRef(connection_id="fixture-openai", model_id=MODEL)],
        "started_at": STARTED,
        "finished_at": STARTED,
    }
    values.update(overrides)
    return RunManifest(**values)


def _write_run(path: Path, *, primary: float = 0.79, **manifest: str) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(_manifest(**manifest).model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cells(path / "cells.json", [], spec_cells=[_cell(primary)])
    return path


def test_a_spec_run_can_be_gated_through_the_cli(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", primary=0.80)
    candidate = _write_run(tmp_path / "candidate", primary=0.79)

    result = evaluate_spec_runs(baseline, candidate)

    assert result.outcome is GateOutcome.PASS
    assert result.rules
    cli = CliRunner().invoke(
        app, ["gate", "--baseline", str(baseline), "--candidate", str(candidate)]
    )
    assert cli.exit_code == 0, cli.output
    assert "PASS" in cli.output
    assert "spec rule" in cli.output


def test_a_spec_gate_refuses_a_different_prompt_version(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", primary=0.80)
    candidate = _write_run(tmp_path / "candidate", primary=0.79, prompt_version="d2")

    result = evaluate_spec_runs(baseline, candidate)
    assert result.outcome is GateOutcome.REFUSED
    assert any("prompt_version differs" in reason for reason in result.reasons)

    cli = CliRunner().invoke(
        app, ["gate", "--baseline", str(baseline), "--candidate", str(candidate)]
    )
    assert cli.exit_code == 2
    assert "prompt_version differs" in cli.output


def test_evaluate_spec_gate_checks_the_manifest_fields_directly() -> None:
    spec = _defect_spec()
    cell = _cell()
    result = evaluate_spec_gate(
        spec,
        {MODEL: cell},
        {MODEL: cell},
        baseline_manifest={"sample_hash": "a" * 64, "prompt_version": "d1", "scorer_version": "s1"},
        candidate_manifest={
            "sample_hash": "b" * 64,
            "prompt_version": "d1",
            "scorer_version": "s1",
        },
    )
    assert result.outcome is GateOutcome.REFUSED
    assert result.reasons == (
        "runs are not comparable: sample_hash differs ('" + "a" * 64 + "' vs '" + "b" * 64 + "')",
    )


def _defect_spec():
    from which_ai_model.bench.run import load_specs

    return load_specs(["defect"])[0]
