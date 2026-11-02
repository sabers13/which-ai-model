"""Regression gate (``PLAN.md`` §7).

``wam gate --baseline <run> --candidate <run>`` runs the comparability check **first**:

* different ``sample_hash``, ``prompt_version`` or ``scorer_version`` → **REFUSED**
  (exit 2). A refusal is never a pass.
* otherwise compare the primary metric per task; a drop of more than **2.0 points**
  fails.
* T-REPLY schema-valid rate below **98%** fails.

Outcome is ``PASS`` (exit 0), ``FAIL`` (exit 1) or ``REFUSED`` (exit 2). The optional
``--json`` path writes the full report; the console always gets a one-line verdict plus
the reasons.

Spec tasks carry their own thresholds: ``spec.py`` adds the per-task gate, the sampling
settings check and the "good enough" picker for them (``PLAN-v1.1`` §A4, §A5). The v1
rules, thresholds and exit codes above are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from which_ai_model.bench.score import load_cells, load_spec_cells
from which_ai_model.contracts import CellResult, RunManifest, TaskName

__all__ = [
    "GATE_PRIMARY_DROP_POINTS",
    "GATE_SCHEMA_VALID_FLOOR",
    "PICKER_WITHIN_POINTS",
    "Comparison",
    "GateOutcome",
    "GateResult",
    "GateRuleResult",
    "Pick",
    "SettingsCheck",
    "SpecCellLike",
    "SpecGateResult",
    "check_gate",
    "check_settings",
    "evaluate_gate",
    "evaluate_spec_gate",
    "evaluate_spec_runs",
    "pick",
    "write_report",
    "write_spec_report",
]

GATE_PRIMARY_DROP_POINTS = 2.0
GATE_SCHEMA_VALID_FLOOR = 0.98
_EPSILON = 1e-12


class GateOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    REFUSED = "REFUSED"

    @property
    def exit_code(self) -> int:
        return {GateOutcome.PASS: 0, GateOutcome.FAIL: 1, GateOutcome.REFUSED: 2}[self]


@dataclass(frozen=True)
class Comparison:
    """One model x task comparison in the report."""

    task: TaskName
    baseline_model: str
    candidate_model: str
    baseline_primary: float
    candidate_primary: float
    primary_delta: float
    schema_valid_baseline: float | None = None
    schema_valid_candidate: float | None = None
    failures: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "baseline_primary": self.baseline_primary,
            "candidate_primary": self.candidate_primary,
            "primary_delta": self.primary_delta,
            "schema_valid_baseline": self.schema_valid_baseline,
            "schema_valid_candidate": self.schema_valid_candidate,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class GateResult:
    """Verdict, reasons and the per-task comparisons."""

    outcome: GateOutcome
    reasons: list[str] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    baseline: str = ""
    candidate: str = ""
    rules: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "outcome": self.outcome.value,
            "exit_code": self.exit_code,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "reasons": list(self.reasons),
            "comparisons": [row.as_json() for row in self.comparisons],
        }
        if self.rules:
            payload["rules"] = list(self.rules)
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def _load_manifest(run: Path) -> RunManifest:
    return RunManifest.model_validate_json((run / "manifest.json").read_text(encoding="utf-8"))


def _match(model: Any, wanted: str) -> bool:
    model_id = str(getattr(model, "model_id", ""))
    connection_id = str(getattr(model, "connection_id", ""))
    return wanted in {model_id, f"{connection_id}/{model_id}"}


def _select(
    cells: list[CellResult],
    *,
    task: TaskName,
    model_id: str | None,
) -> list[CellResult]:
    rows = [cell for cell in cells if cell.task is task]
    if model_id is not None:
        rows = [cell for cell in rows if _match(cell.model, model_id)]
    return rows


def _compare(
    baseline_cell: CellResult,
    candidate_cell: CellResult,
) -> Comparison:
    delta = baseline_cell.primary_metric - candidate_cell.primary_metric
    failures: list[str] = []
    if delta > (GATE_PRIMARY_DROP_POINTS / 100.0) + _EPSILON:
        failures.append(
            f"{baseline_cell.task.value} primary metric dropped {delta * 100:.2f} points "
            f"({baseline_cell.primary_metric:.4f} → {candidate_cell.primary_metric:.4f}); "
            f"limit {GATE_PRIMARY_DROP_POINTS:.1f}"
        )
    schema_baseline = baseline_cell.metrics.get("schema_valid_rate")
    schema_candidate = candidate_cell.metrics.get("schema_valid_rate")
    if (
        baseline_cell.task is TaskName.REPLY
        and schema_candidate is not None
        and schema_candidate < GATE_SCHEMA_VALID_FLOOR - _EPSILON
    ):
        failures.append(
            f"T-REPLY schema-valid rate {schema_candidate:.4f} is below "
            f"{GATE_SCHEMA_VALID_FLOOR:.2f}"
        )
    return Comparison(
        task=baseline_cell.task,
        baseline_model=baseline_cell.model.model_id,
        candidate_model=candidate_cell.model.model_id,
        baseline_primary=baseline_cell.primary_metric,
        candidate_primary=candidate_cell.primary_metric,
        primary_delta=delta,
        schema_valid_baseline=schema_baseline,
        schema_valid_candidate=schema_candidate,
        failures=tuple(failures),
    )


def _spec_run_gate(
    baseline: Path,
    candidate: Path,
    *,
    model_a: str | None,
    model_b: str | None,
) -> GateResult:
    """Adapt the spec-run verdict to the ``wam gate`` result shape."""
    from which_ai_model.gate.spec import evaluate_spec_runs

    result = evaluate_spec_runs(baseline, candidate, model_a=model_a, model_b=model_b)
    return GateResult(
        outcome=result.outcome,
        reasons=list(result.reasons),
        baseline=str(baseline),
        candidate=str(candidate),
        rules=[rule.as_json() for rule in result.rules],
        warnings=list(result.warnings),
    )


def evaluate_gate(
    baseline: Path,
    candidate: Path,
    *,
    model_a: str | None = None,
    model_b: str | None = None,
) -> GateResult:
    """Evaluate the gate without raising for FAIL/REFUSED."""
    baseline = Path(baseline)
    candidate = Path(candidate)
    if (model_a is None) != (model_b is None):
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=["--model-a and --model-b must be given together"],
            baseline=str(baseline),
            candidate=str(candidate),
        )
    try:
        baseline_manifest = _load_manifest(baseline)
        candidate_manifest = _load_manifest(candidate)
        baseline_cells = load_cells(baseline / "cells.json")
        candidate_cells = load_cells(candidate / "cells.json")
        baseline_spec = load_spec_cells(baseline / "cells.json")
        candidate_spec = load_spec_cells(candidate / "cells.json")
    except (OSError, ValueError) as exc:
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=[f"cannot read a run: {exc}"],
            baseline=str(baseline),
            candidate=str(candidate),
        )

    refusals: list[str] = []
    for field_name, left, right in (
        ("sample_hash", baseline_manifest.sample_hash, candidate_manifest.sample_hash),
        ("prompt_version", baseline_manifest.prompt_version, candidate_manifest.prompt_version),
        ("scorer_version", baseline_manifest.scorer_version, candidate_manifest.scorer_version),
    ):
        if left != right:
            refusals.append(
                f"runs are not comparable: {field_name} differs ({left!r} vs {right!r})"
            )
    if refusals:
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=refusals,
            baseline=str(baseline),
            candidate=str(candidate),
        )

    tasks = sorted({cell.task for cell in baseline_cells}, key=lambda task: task.value)
    if not tasks:
        if baseline_spec or candidate_spec:
            return _spec_run_gate(baseline, candidate, model_a=model_a, model_b=model_b)
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=["baseline run has no cells"],
            baseline=str(baseline),
            candidate=str(candidate),
        )

    if model_a is None:
        baseline_models = {cell.model.model_id for cell in baseline_cells}
        candidate_models = {cell.model.model_id for cell in candidate_cells}
        common = sorted(baseline_models & candidate_models)
        if not common:
            return GateResult(
                outcome=GateOutcome.REFUSED,
                reasons=["no model id appears in both runs; pass --model-a/--model-b"],
                baseline=str(baseline),
                candidate=str(candidate),
            )
        pairs: list[tuple[str, str]] = [(model_id, model_id) for model_id in common]
    else:
        assert model_b is not None
        pairs = [(model_a, model_b)]

    comparisons: list[Comparison] = []
    fails: list[str] = []
    for baseline_model, candidate_model in pairs:
        for task in tasks:
            baseline_rows = _select(baseline_cells, task=task, model_id=baseline_model)
            if not baseline_rows:
                refusals.append(f"baseline has no {task.value} cell for {baseline_model!r}")
                continue
            candidate_rows = _select(candidate_cells, task=task, model_id=candidate_model)
            if not candidate_rows:
                refusals.append(f"candidate has no {task.value} cell for {candidate_model!r}")
                continue
            comparison = _compare(baseline_rows[0], candidate_rows[0])
            comparisons.append(comparison)
            fails.extend(comparison.failures)
            schema_candidate = comparison.schema_valid_candidate
            if task is TaskName.REPLY and schema_candidate is None:
                refusals.append(
                    f"candidate has no schema_valid_rate for {task.value}/{candidate_model}"
                )

    if refusals:
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=refusals,
            comparisons=comparisons,
            baseline=str(baseline),
            candidate=str(candidate),
        )
    if not comparisons:
        return GateResult(
            outcome=GateOutcome.REFUSED,
            reasons=["no comparable model x task cell"],
            baseline=str(baseline),
            candidate=str(candidate),
        )
    if fails:
        return GateResult(
            outcome=GateOutcome.FAIL,
            reasons=fails,
            comparisons=comparisons,
            baseline=str(baseline),
            candidate=str(candidate),
        )
    return GateResult(
        outcome=GateOutcome.PASS,
        reasons=[],
        comparisons=comparisons,
        baseline=str(baseline),
        candidate=str(candidate),
    )


def check_gate(baseline: Path, candidate: Path) -> tuple[bool, list[str]]:
    """``(passed, reasons)``; REFUSED is not a pass, exactly as the plan requires."""
    result = evaluate_gate(baseline, candidate)
    return result.outcome is GateOutcome.PASS, result.reasons


def write_report(path: Path, result: GateResult) -> None:
    """Write the JSON report deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Spec tasks (PLAN-v1.1 §A4, §A5). Imported last because ``spec`` imports ``GateOutcome``
# from this module; every v1 name above exists before the import runs.
from which_ai_model.gate.spec import (  # noqa: E402
    PICKER_WITHIN_POINTS,
    GateRuleResult,
    Pick,
    SettingsCheck,
    SpecCellLike,
    SpecGateResult,
    check_settings,
    evaluate_spec_gate,
    evaluate_spec_runs,
    pick,
    write_spec_report,
)
