"""Regression gate: PASS/FAIL/REFUSED, comparability first, CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from which_ai_model.bench.score import write_cells
from which_ai_model.cli import app
from which_ai_model.contracts import CellResult, ModelRef, RunManifest, TaskName
from which_ai_model.gate import GateOutcome, evaluate_gate


def _cell(
    task: TaskName,
    primary: float,
    *,
    model_id: str = "m",
    schema: float | None = None,
) -> CellResult:
    metrics = {}
    if schema is not None:
        metrics["schema_valid_rate"] = schema
    return CellResult(
        task=task,
        model=ModelRef(connection_id="c", model_id=model_id),
        n=10,
        primary_metric=primary,
        metrics=metrics,
        ci={},
        latency_p50=10.0,
        latency_p95=20.0,
        cost_per_1k=1.0,
    )


def _write_run(
    path: Path,
    cells: list[CellResult],
    *,
    sample_hash: str = "sample",
    prompt_version: str = "v1",
    scorer_version: str = "v1",
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=path.name,
        sample_hash=sample_hash,
        prompt_version=prompt_version,
        scorer_version=scorer_version,
        models=sorted({cell.model for cell in cells}, key=lambda model: model.model_id),
        started_at="2026-10-28T00:00:00Z",  # type: ignore[arg-type]
        finished_at="2026-10-28T00:10:00Z",  # type: ignore[arg-type]
    )
    (path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    write_cells(path / "cells.json", cells)
    return path


def _pair(
    tmp_path: Path, baseline_primary: float, candidate_primary: float, **kwargs
) -> tuple[Path, Path]:
    baseline = _write_run(
        tmp_path / "baseline",
        [
            _cell(TaskName.ROUTE, baseline_primary),
            _cell(TaskName.REPLY, 1.0, schema=1.0),
        ],
        **kwargs,
    )
    candidate = _write_run(
        tmp_path / "candidate",
        [
            _cell(TaskName.ROUTE, candidate_primary),
            _cell(TaskName.REPLY, 1.0, schema=1.0),
        ],
        **kwargs,
    )
    return baseline, candidate


def test_gate_passes_a_small_drop(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path, 0.90, 0.885)
    result = evaluate_gate(baseline, candidate)
    assert result.outcome is GateOutcome.PASS
    assert result.exit_code == 0
    assert result.reasons == []


def test_gate_fails_on_primary_drop_over_two_points(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path, 0.90, 0.87)
    result = evaluate_gate(baseline, candidate)
    assert result.outcome is GateOutcome.FAIL
    assert result.exit_code == 1
    assert any("dropped" in reason for reason in result.reasons)


def test_gate_fails_when_reply_schema_valid_drops_below_98(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path / "baseline",
        [_cell(TaskName.REPLY, 1.0, schema=1.0)],
    )
    candidate = _write_run(
        tmp_path / "candidate",
        [_cell(TaskName.REPLY, 1.0, schema=0.97)],
    )
    result = evaluate_gate(baseline, candidate)
    assert result.outcome is GateOutcome.FAIL
    assert any("schema-valid" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_hash", "other"),
        ("prompt_version", "v2"),
        ("scorer_version", "v2"),
    ],
)
def test_gate_refuses_incomparable_runs(tmp_path: Path, field: str, value: str) -> None:
    baseline = _write_run(tmp_path / "baseline", [_cell(TaskName.ROUTE, 0.9)])
    candidate = _write_run(tmp_path / "candidate", [_cell(TaskName.ROUTE, 0.9)], **{field: value})
    result = evaluate_gate(baseline, candidate)
    assert result.outcome is GateOutcome.REFUSED
    assert result.exit_code == 2
    assert any(field in reason for reason in result.reasons)


def test_gate_refuses_a_missing_run(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", [_cell(TaskName.ROUTE, 0.9)])
    result = evaluate_gate(baseline, tmp_path / "missing")
    assert result.outcome is GateOutcome.REFUSED


def test_gate_selects_models(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path / "baseline",
        [
            _cell(TaskName.ROUTE, 0.9, model_id="flagship"),
            _cell(TaskName.ROUTE, 0.8, model_id="cheap"),
        ],
    )
    candidate = _write_run(
        tmp_path / "candidate",
        [
            _cell(TaskName.ROUTE, 0.89, model_id="flagship"),
            _cell(TaskName.ROUTE, 0.7, model_id="cheap"),
        ],
    )
    passed = evaluate_gate(baseline, candidate, model_a="flagship", model_b="flagship")
    failed = evaluate_gate(baseline, candidate, model_a="flagship", model_b="cheap")
    assert passed.outcome is GateOutcome.PASS
    assert failed.outcome is GateOutcome.FAIL
    assert evaluate_gate(baseline, candidate, model_a="flagship").outcome is GateOutcome.REFUSED


def test_gate_cli_exit_codes_and_json_report(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path, 0.90, 0.80)
    report = tmp_path / "gate.json"
    runner = CliRunner()
    failed = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--json",
            str(report),
        ],
    )
    assert failed.exit_code == 1
    assert "FAIL" in failed.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["outcome"] == "FAIL"
    assert payload["exit_code"] == 1

    baseline_ok, candidate_ok = _pair(tmp_path / "ok", 0.90, 0.89)
    passed = runner.invoke(
        app,
        ["gate", "--baseline", str(baseline_ok), "--candidate", str(candidate_ok)],
    )
    assert passed.exit_code == 0

    candidate_v2 = _write_run(
        tmp_path / "candidate-v2", [_cell(TaskName.ROUTE, 0.9)], prompt_version="v2"
    )
    refused = runner.invoke(
        app,
        ["gate", "--baseline", str(baseline_ok), "--candidate", str(candidate_v2)],
    )
    assert refused.exit_code == 2
    assert "REFUSED" in refused.output
