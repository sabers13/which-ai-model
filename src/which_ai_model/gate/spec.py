"""Spec-task gate, run-settings comparability and the picker (``PLAN-v1.1`` §A4, §A5).

A spec task carries its own gate thresholds, so the v1 gate in ``which_ai_model.gate``
stays exactly as it is and this module adds the spec rules:

* the primary metric may not drop by more than ``gate.primary_drop_points``;
* every ``gate.floors`` entry must stay at or above ``min`` and/or within
  ``max_drop_points`` of the baseline;
* the schema-valid rate must stay at or above ``gate.schema_valid_floor``.

Each rule is reported by name with its measured value, its threshold and its verdict, so
the JSON report says which rule failed and by how much. ``value`` and ``threshold`` share
a unit: a drop rule measures points on the 0 to 100 scale the spec uses (``2.0`` means
two points), every other rule measures a rate between 0 and 1. A required metric a cell
does not carry (a floor metric, or ``schema_valid_rate``) makes the outcome REFUSED with
a reason, never a silent pass.

Run settings (temperature, reasoning and maximum output tokens) must match the baseline
run. A difference is REFUSED unless the caller accepted it; when accepted, the normal
verdict stands and every difference becomes a ``WARN`` line. A v1 run recorded no
temperature, so it reads as ``provider-default`` (``RunSettings.provider_default()``).

``pick`` is the §A4 "good enough" rule for spec tasks, next to the v1 rule in
``bench.score.build_recommendations``: within 3.0 points of the best primary metric, never
below a floor, and an explicit "no model meets the floor" outcome. It is a pure function
so the card builder (M6) can call it.

This module imports :class:`GateOutcome` from its own package, so ``gate/__init__``
imports it last, after the enum exists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from which_ai_model.bench.score import GOOD_ENOUGH_POINTS
from which_ai_model.contracts import GateFloor, GateSpec, RunSettings, TaskSpec
from which_ai_model.gate import GateOutcome

__all__ = [
    "PICKER_WITHIN_POINTS",
    "GateRuleResult",
    "Pick",
    "SettingsCheck",
    "SpecCellLike",
    "SpecGateResult",
    "check_settings",
    "compare_manifests",
    "evaluate_spec_gate",
    "evaluate_spec_runs",
    "pick",
    "write_spec_report",
]

#: The §A4 "good enough" band: 3.0 points of the primary metric, as in v1 §7.
PICKER_WITHIN_POINTS = GOOD_ENOUGH_POINTS
_EPSILON = 1e-12
_SETTING_FIELDS = ("temperature", "reasoning", "max_output_tokens")


class SpecCellLike(Protocol):
    """What the spec gate and the picker need from one model x task cell."""

    @property
    def model_id(self) -> str: ...

    @property
    def primary_metric(self) -> float: ...

    @property
    def metrics(self) -> Mapping[str, float]: ...

    @property
    def cost_per_1k(self) -> float | None: ...

    @property
    def latency_p50(self) -> float | None: ...


@dataclass(frozen=True)
class GateRuleResult:
    """One gate rule: its name, measured value, threshold and verdict.

    ``model_id`` names the model the rule was evaluated for. ``value`` and ``threshold``
    share a unit: points for the drop rules, a rate between 0 and 1 otherwise.
    """

    name: str
    value: float
    threshold: float
    passed: bool
    model_id: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SpecGateResult:
    """The spec gate verdict, every rule it evaluated, its WARN lines and refusals."""

    outcome: GateOutcome
    rules: tuple[GateRuleResult, ...]
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code

    def as_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "exit_code": self.exit_code,
            "rules": [rule.as_json() for rule in self.rules],
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SettingsCheck:
    """Whether two runs used comparable sampling settings, and how they differ."""

    comparable: bool
    differences: tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        return not self.comparable


def _show(value: object) -> str:
    """Quote strings so ``'provider-default'`` cannot be read as a number."""
    return repr(value)


def check_settings(
    baseline: Mapping[str, RunSettings],
    candidate: Mapping[str, RunSettings],
    *,
    accept_mismatch: bool = False,
) -> SettingsCheck:
    """Compare per-model sampling settings; a difference is REFUSED unless accepted.

    Only models present in both runs are compared: a model with no baseline cell cannot
    be gated anyway, and the gate refuses it with its own reason. A v1 run has no
    recorded settings, so the caller reads it with ``RunSettings.unrecorded()``:
    temperature is ``provider-default`` and reasoning is ``not-recorded``, which is
    skipped rather than reported as an invented difference.
    """
    differences: list[str] = []
    for model_id in sorted(set(baseline) & set(candidate)):
        left = baseline[model_id]
        right = candidate[model_id]
        for field_name in _SETTING_FIELDS:
            before = getattr(left, field_name)
            after = getattr(right, field_name)
            if before == after or "not-recorded" in (before, after):
                continue
            differences.append(
                f"{field_name} differs for {model_id}: "
                f"baseline {_show(before)}, candidate {_show(after)}"
            )
    return SettingsCheck(
        comparable=accept_mismatch or not differences,
        differences=tuple(differences),
    )


def _metric(cell: SpecCellLike, metric: str) -> float | None:
    return cell.metrics.get(metric)


def _model_rules(
    gate: GateSpec,
    baseline: SpecCellLike,
    candidate: SpecCellLike,
) -> tuple[list[GateRuleResult], list[str]]:
    """Every rule for one model pair, plus the reasons a rule cannot be evaluated."""
    rules: list[GateRuleResult] = []
    reasons: list[str] = []
    model_id = candidate.model_id

    drop_points = (baseline.primary_metric - candidate.primary_metric) * 100.0
    rules.append(
        GateRuleResult(
            name="primary_drop",
            value=drop_points,
            threshold=gate.primary_drop_points,
            passed=drop_points <= gate.primary_drop_points + _EPSILON,
            model_id=model_id,
        )
    )

    for floor in gate.floors:
        value = _metric(candidate, floor.metric)
        if floor.min is not None:
            if value is None:
                reasons.append(
                    f"{model_id!r} has no {floor.metric!r} metric to check against its floor"
                )
            else:
                rules.append(
                    GateRuleResult(
                        name=f"floor_min:{floor.metric}",
                        value=value,
                        threshold=floor.min,
                        passed=value >= floor.min - _EPSILON,
                        model_id=model_id,
                    )
                )
        if floor.max_drop_points is not None:
            before = _metric(baseline, floor.metric)
            if before is None or value is None:
                reasons.append(f"{model_id!r} has no {floor.metric!r} metric to check for its drop")
            else:
                floor_drop = (before - value) * 100.0
                rules.append(
                    GateRuleResult(
                        name=f"floor_drop:{floor.metric}",
                        value=floor_drop,
                        threshold=floor.max_drop_points,
                        passed=floor_drop <= floor.max_drop_points + _EPSILON,
                        model_id=model_id,
                    )
                )

    schema = _metric(candidate, "schema_valid_rate")
    if schema is None:
        reasons.append(f"{model_id!r} has no 'schema_valid_rate' metric for the schema-valid floor")
    else:
        rules.append(
            GateRuleResult(
                name="schema_valid_floor",
                value=schema,
                threshold=gate.schema_valid_floor,
                passed=schema >= gate.schema_valid_floor - _EPSILON,
                model_id=model_id,
            )
        )
    return rules, reasons


def _manifest_value(document: Mapping[str, Any] | object | None, name: str) -> Any:
    if document is None:
        return None
    if isinstance(document, Mapping):
        return document.get(name)
    return getattr(document, name, None)


def compare_manifests(
    baseline_manifest: Mapping[str, Any] | object | None,
    candidate_manifest: Mapping[str, Any] | object | None,
) -> tuple[str, ...]:
    """§A5 comparability for a spec run: sample hash, prompt version, scorer version."""
    reasons: list[str] = []
    for name in ("sample_hash", "prompt_version", "scorer_version"):
        left = _manifest_value(baseline_manifest, name)
        right = _manifest_value(candidate_manifest, name)
        if left is not None and right is not None and left != right:
            reasons.append(f"runs are not comparable: {name} differs ({left!r} vs {right!r})")
    return tuple(reasons)


def evaluate_spec_gate(
    spec: TaskSpec,
    baseline: Mapping[str, SpecCellLike],
    candidate: Mapping[str, SpecCellLike],
    *,
    settings: SettingsCheck | None = None,
    baseline_manifest: Mapping[str, Any] | object | None = None,
    candidate_manifest: Mapping[str, Any] | object | None = None,
) -> SpecGateResult:
    """Gate a candidate run against a baseline run for one spec task.

    Comparability comes first: different sample, prompt or scorer versions are REFUSED,
    exactly as in v1, and incomparable settings are REFUSED, never a pass. Accepted
    settings differences keep the normal verdict and add one ``WARN`` line each.
    """
    comparability = compare_manifests(baseline_manifest, candidate_manifest)
    if comparability:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=comparability,
        )
    if settings is not None and settings.refused:
        refusal_reasons = tuple(
            f"runs are not comparable: {difference}" for difference in settings.differences
        )
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=refusal_reasons or ("runs are not comparable: sampling settings differ",),
        )
    if not candidate:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=("candidate run has no cells for this task",),
        )

    warnings = (
        tuple(f"WARN: {difference}" for difference in settings.differences)
        if settings is not None
        else ()
    )
    rules: list[GateRuleResult] = []
    reasons: list[str] = []
    for model_id in sorted(candidate):
        baseline_cell = baseline.get(model_id)
        if baseline_cell is None:
            reasons.append(f"baseline run has no comparable cell for {model_id!r}")
            continue
        model_rules, model_reasons = _model_rules(spec.gate, baseline_cell, candidate[model_id])
        rules.extend(model_rules)
        reasons.extend(model_reasons)

    if reasons:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=tuple(rules),
            warnings=warnings,
            reasons=tuple(reasons),
        )
    if not rules:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            warnings=warnings,
            reasons=("no comparable model cell to gate",),
        )
    outcome = GateOutcome.PASS if all(rule.passed for rule in rules) else GateOutcome.FAIL
    return SpecGateResult(outcome=outcome, rules=tuple(rules), warnings=warnings)


def write_spec_report(path: Path, result: SpecGateResult) -> None:
    """Write the JSON report deterministically: outcome, every rule, WARN lines, reasons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _meets_floor(cell: SpecCellLike, floor: GateFloor) -> bool:
    """Whether a cell clears a floor's ``min``; the drop bound needs a baseline."""
    if floor.min is None:
        return True
    value = _metric(cell, floor.metric)
    return value is not None and value >= floor.min - _EPSILON


def _cost_key(cell: SpecCellLike) -> tuple[bool, float, str]:
    """Unknown cost sorts last, then the cheaper model wins, then the model id."""
    return (
        cell.cost_per_1k is None,
        cell.cost_per_1k if cell.cost_per_1k is not None else 0.0,
        cell.model_id,
    )


@dataclass(frozen=True)
class Pick:
    """The §A4 picks for one spec task; ``None`` when there is nothing to pick."""

    cheapest: str | None
    most_accurate: str | None
    fastest: str | None
    good_enough: tuple[str, ...]
    no_model_meets_floor: bool


def pick(cells: Sequence[SpecCellLike], *, floors: Sequence[GateFloor] = ()) -> Pick:
    """Pick cheapest / most accurate / fastest under the §A4 good-enough rule.

    A model below a floor ``min`` is never good enough. When no model meets the floors,
    cheapest and fastest are ``None`` and ``no_model_meets_floor`` is ``True`` so the
    card can say so plainly instead of showing a cheapest pick; ``most_accurate`` still
    names the best model so the card can point at the one to watch. Only ``min`` is
    applied here; the baseline-relative ``max_drop_points`` bound belongs to the gate.
    """
    rows = list(cells)
    if not rows:
        return Pick(
            cheapest=None,
            most_accurate=None,
            fastest=None,
            good_enough=(),
            no_model_meets_floor=False,
        )
    eligible = [cell for cell in rows if all(_meets_floor(cell, floor) for floor in floors)]
    if floors and not eligible:
        # §A4: name the most accurate model even when it is below a floor, so the card
        # can say which model to watch; no cheapest or fastest pick is shown.
        accurate_cell = min(rows, key=lambda cell: (-cell.primary_metric, *_cost_key(cell)))
        return Pick(
            cheapest=None,
            most_accurate=accurate_cell.model_id,
            fastest=None,
            good_enough=(),
            no_model_meets_floor=True,
        )

    best = max(cell.primary_metric for cell in eligible)
    threshold = best - (PICKER_WITHIN_POINTS / 100.0) - _EPSILON
    good = [cell for cell in eligible if cell.primary_metric >= threshold]
    ordered_good = sorted(good, key=lambda cell: (-cell.primary_metric, *_cost_key(cell)))
    cheapest = min(good, key=_cost_key).model_id
    most_accurate = min(
        eligible, key=lambda cell: (-cell.primary_metric, *_cost_key(cell))
    ).model_id
    fastest = min(
        (cell for cell in good if cell.latency_p50 is not None),
        key=lambda cell: (
            cell.latency_p50 if cell.latency_p50 is not None else 0.0,
            *_cost_key(cell),
        ),
        default=None,
    )
    return Pick(
        cheapest=cheapest,
        most_accurate=most_accurate,
        fastest=None if fastest is None else fastest.model_id,
        good_enough=tuple(cell.model_id for cell in ordered_good),
        no_model_meets_floor=False,
    )


def _rows_by_model(cells: Sequence[SpecCellLike]) -> dict[str, SpecCellLike]:
    return {cell.model_id: cell for cell in cells}


def evaluate_spec_runs(
    baseline: Path,
    candidate: Path,
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    tasks_dir: Path = Path("tasks"),
) -> SpecGateResult:
    """Gate two run directories that carry spec cells (``wam gate`` for a spec task).

    The run manifests supply the §A5 comparability fields and the committed spec
    supplies the thresholds. ``model_a``/``model_b`` select one pair; otherwise every
    model id present in both runs is gated. A run with no spec cell for a task, or no
    common model, is REFUSED with a reason instead of quietly passing.
    """
    from which_ai_model.bench.run import load_specs
    from which_ai_model.bench.score import load_spec_cells

    baseline = Path(baseline)
    candidate = Path(candidate)
    if (model_a is None) != (model_b is None):
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=("--model-a and --model-b must be given together",),
        )
    try:
        baseline_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
        candidate_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        baseline_cells = load_spec_cells(baseline / "cells.json")
        candidate_cells = load_spec_cells(candidate / "cells.json")
    except (OSError, ValueError) as exc:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=(f"cannot read a spec run: {exc}",),
        )
    if not baseline_cells or not candidate_cells:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=(
                "baseline run has no spec cells"
                if not baseline_cells
                else "candidate run has no spec cells",
            ),
        )

    baseline_by_task: dict[str, dict[str, SpecCellLike]] = {}
    for cell in baseline_cells:
        baseline_by_task.setdefault(cell.task, {})[cell.model_id] = cell
    candidate_by_task: dict[str, dict[str, SpecCellLike]] = {}
    for cell in candidate_cells:
        candidate_by_task.setdefault(cell.task, {})[cell.model_id] = cell

    task_ids = sorted(set(baseline_by_task) | set(candidate_by_task))
    try:
        specs = load_specs(task_ids, tasks_dir=tasks_dir)
    except FileNotFoundError as exc:
        return SpecGateResult(outcome=GateOutcome.REFUSED, rules=(), reasons=(str(exc),))

    rules: list[GateRuleResult] = []
    warnings: list[str] = []
    reasons: list[str] = []
    outcomes: list[GateOutcome] = []
    for spec in specs:
        baseline_rows = baseline_by_task.get(spec.id)
        candidate_rows = candidate_by_task.get(spec.id)
        if not baseline_rows:
            reasons.append(f"baseline run has no {spec.id!r} cell")
            continue
        if not candidate_rows:
            reasons.append(f"candidate run has no {spec.id!r} cell")
            continue
        if model_a is not None:
            pairs = [(model_a, model_b or model_a)]
        else:
            common = sorted(set(baseline_rows) & set(candidate_rows))
            pairs = [(model_id, model_id) for model_id in common]
        if not pairs:
            reasons.append(f"no model id appears in both runs for {spec.id!r}")
            continue
        for baseline_model, candidate_model in pairs:
            baseline_cell = baseline_rows.get(baseline_model)
            candidate_cell = candidate_rows.get(candidate_model)
            if baseline_cell is None or candidate_cell is None:
                reasons.append(
                    f"{spec.id}: baseline {baseline_model!r} or candidate "
                    f"{candidate_model!r} has no cell"
                )
                continue
            result = evaluate_spec_gate(
                spec,
                {candidate_model: baseline_cell},
                {candidate_model: candidate_cell},
                baseline_manifest=baseline_manifest,
                candidate_manifest=candidate_manifest,
            )
            rules.extend(result.rules)
            warnings.extend(result.warnings)
            reasons.extend(result.reasons)
            outcomes.append(result.outcome)
    if GateOutcome.REFUSED in outcomes or reasons:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=tuple(rules),
            warnings=tuple(warnings),
            reasons=tuple(reasons) or ("spec gate refused",),
        )
    if not outcomes:
        return SpecGateResult(
            outcome=GateOutcome.REFUSED,
            rules=(),
            reasons=("no comparable spec cell to gate",),
        )
    outcome = GateOutcome.FAIL if GateOutcome.FAIL in outcomes else GateOutcome.PASS
    return SpecGateResult(
        outcome=outcome,
        rules=tuple(rules),
        warnings=tuple(warnings),
    )
