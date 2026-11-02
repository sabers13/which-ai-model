"""Spec gate (``PLAN-v1.1`` §A5): PASS / FAIL / REFUSED, floors, schema floor, report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from which_ai_model.cli import app
from which_ai_model.contracts import (
    DataSpec,
    GateFloor,
    GateSpec,
    MetricSpec,
    OutputKind,
    OutputSpec,
    PromptSpec,
    RunSettings,
    TaskSpec,
)
from which_ai_model.gate import (
    GateOutcome,
    SettingsCheck,
    check_settings,
    evaluate_spec_gate,
    write_spec_report,
)


def _spec(gate: GateSpec | None = None) -> TaskSpec:
    return TaskSpec(
        id="defect",
        title="Sort electrical complaints and flag safety",
        description="Sort owner complaints into component groups and flag the safety-relevant ones.",
        prompt_version="d1",
        scorer_version="s1",
        data=DataSpec(file="data/defect/test.jsonl"),
        prompt=PromptSpec(template_file="tasks/defect.prompt.txt"),
        outputs=[
            OutputSpec(
                name="component_group",
                kind=OutputKind.LABEL,
                truth_field="component_group",
                labels_file="data/defect/groups.toml",
                allow_unclear=True,
            ),
            OutputSpec(name="safety_relevant", kind=OutputKind.BOOL, truth_field="safety_relevant"),
        ],
        metrics=MetricSpec(
            primary="component_group.macro_f1",
            report=[
                "component_group.accuracy",
                "component_group.coverage",
                "component_group.accuracy_answered",
                "safety_relevant.recall",
                "safety_relevant.precision",
                "schema_valid_rate",
            ],
        ),
        gate=gate or _gate(),
    )


def _gate(
    *,
    primary_drop_points: float = 2.0,
    schema_valid_floor: float = 0.98,
    floors: list[GateFloor] | None = None,
) -> GateSpec:
    return GateSpec(
        primary_drop_points=primary_drop_points,
        schema_valid_floor=schema_valid_floor,
        floors=floors
        if floors is not None
        else [GateFloor(metric="safety_relevant.recall", min=0.90, max_drop_points=2.0)],
    )


@dataclass(frozen=True)
class _Cell:
    model_id: str
    primary_metric: float
    metrics: dict[str, float]
    cost_per_1k: float | None = 1.0
    latency_p50: float | None = 10.0


def _cell(
    primary: float = 0.80,
    *,
    model_id: str = "model-a",
    recall: float = 0.95,
    schema: float = 0.99,
) -> _Cell:
    return _Cell(
        model_id=model_id,
        primary_metric=primary,
        metrics={
            "safety_relevant.recall": recall,
            "safety_relevant.precision": 0.80,
            "schema_valid_rate": schema,
        },
    )


def _rules(result) -> dict[str, object]:
    return {rule.name: rule for rule in result.rules}


def test_spec_gate_passes_within_every_limit() -> None:
    result = evaluate_spec_gate(_spec(), {"model-a": _cell(0.80)}, {"model-a": _cell(0.79)})
    assert result.outcome is GateOutcome.PASS
    assert result.exit_code == 0
    assert result.reasons == ()
    assert result.warnings == ()
    assert set(_rules(result)) == {
        "primary_drop",
        "floor_min:safety_relevant.recall",
        "floor_drop:safety_relevant.recall",
        "schema_valid_floor",
    }
    drop = _rules(result)["primary_drop"]
    assert drop.passed is True  # type: ignore[attr-defined]
    assert drop.value == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert drop.threshold == 2.0  # type: ignore[attr-defined]


def test_spec_gate_fails_on_a_primary_drop_over_the_limit() -> None:
    result = evaluate_spec_gate(_spec(), {"model-a": _cell(0.80)}, {"model-a": _cell(0.77)})
    assert result.outcome is GateOutcome.FAIL
    assert result.exit_code == 1
    drop = _rules(result)["primary_drop"]
    assert drop.passed is False  # type: ignore[attr-defined]
    assert drop.value == pytest.approx(3.0)  # type: ignore[attr-defined]


def test_spec_gate_fails_on_a_floor_even_when_the_primary_passes() -> None:
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80, recall=0.95)},
        {"model-a": _cell(0.80, recall=0.85)},
    )
    assert result.outcome is GateOutcome.FAIL
    rules = _rules(result)
    assert rules["primary_drop"].passed is True  # type: ignore[attr-defined]
    floor = rules["floor_min:safety_relevant.recall"]
    assert floor.passed is False  # type: ignore[attr-defined]
    assert floor.value == pytest.approx(0.85)  # type: ignore[attr-defined]
    assert floor.threshold == 0.90  # type: ignore[attr-defined]


def test_spec_gate_fails_on_a_floor_drop_over_its_limit() -> None:
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80, recall=0.95)},
        {"model-a": _cell(0.80, recall=0.92)},
    )
    assert result.outcome is GateOutcome.FAIL
    floor = _rules(result)["floor_drop:safety_relevant.recall"]
    assert floor.passed is False  # type: ignore[attr-defined]
    assert floor.value == pytest.approx(3.0)  # type: ignore[attr-defined]
    assert floor.threshold == 2.0  # type: ignore[attr-defined]


def test_spec_gate_fails_below_the_schema_valid_floor() -> None:
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80)},
        {"model-a": _cell(0.80, schema=0.97)},
    )
    assert result.outcome is GateOutcome.FAIL
    rule = _rules(result)["schema_valid_floor"]
    assert rule.passed is False  # type: ignore[attr-defined]
    assert rule.value == pytest.approx(0.97)  # type: ignore[attr-defined]
    assert rule.threshold == 0.98  # type: ignore[attr-defined]


def test_spec_gate_fails_when_any_model_fails() -> None:
    result = evaluate_spec_gate(
        _spec(),
        {"a": _cell(0.80, model_id="a"), "b": _cell(0.80, model_id="b")},
        {
            "a": _cell(0.80, model_id="a"),
            "b": _cell(0.70, model_id="b", recall=0.50),
        },
    )
    assert result.outcome is GateOutcome.FAIL
    assert {rule.model_id for rule in result.rules} == {"a", "b"}


def test_spec_gate_refuses_on_a_settings_mismatch() -> None:
    settings = check_settings(
        {"model-a": RunSettings.provider_default()},
        {"model-a": RunSettings(temperature=0.0)},
    )
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80)},
        {"model-a": _cell(0.80)},
        settings=settings,
    )
    assert result.outcome is GateOutcome.REFUSED
    assert result.exit_code == 2
    assert result.rules == ()
    assert result.warnings == ()
    assert any("temperature" in reason for reason in result.reasons)
    assert any("not comparable" in reason for reason in result.reasons)


def test_spec_gate_warns_when_a_settings_mismatch_is_accepted() -> None:
    settings = SettingsCheck(comparable=True, differences=("temperature differs: x",))
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80)},
        {"model-a": _cell(0.80)},
        settings=settings,
    )
    assert result.outcome is GateOutcome.PASS
    assert result.exit_code == 0
    assert result.warnings == ("WARN: temperature differs: x",)


def test_spec_gate_refuses_a_candidate_model_without_a_baseline_cell() -> None:
    result = evaluate_spec_gate(
        _spec(),
        {"model-a": _cell(0.80)},
        {"model-b": _cell(0.80, model_id="model-b")},
    )
    assert result.outcome is GateOutcome.REFUSED
    assert result.exit_code == 2
    assert any("model-b" in reason for reason in result.reasons)


def test_spec_gate_refuses_an_empty_candidate() -> None:
    result = evaluate_spec_gate(_spec(), {"model-a": _cell(0.80)}, {})
    assert result.outcome is GateOutcome.REFUSED
    assert result.exit_code == 2


def test_spec_gate_refuses_a_missing_floor_metric() -> None:
    cell = _Cell(model_id="model-a", primary_metric=0.80, metrics={"schema_valid_rate": 0.99})
    result = evaluate_spec_gate(_spec(), {"model-a": cell}, {"model-a": cell})
    assert result.outcome is GateOutcome.REFUSED
    assert any("safety_relevant.recall" in reason for reason in result.reasons)


def test_spec_gate_report_names_each_rule_value_and_verdict(tmp_path: Path) -> None:
    result = evaluate_spec_gate(_spec(), {"model-a": _cell(0.80)}, {"model-a": _cell(0.77)})
    path = tmp_path / "gate" / "spec.json"
    write_spec_report(path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "FAIL"
    assert payload["exit_code"] == 1
    named = {rule["name"]: rule for rule in payload["rules"]}
    assert named["primary_drop"]["value"] == pytest.approx(3.0)
    assert named["primary_drop"]["threshold"] == 2.0
    assert named["primary_drop"]["passed"] is False
    assert named["schema_valid_floor"]["passed"] is True
    assert all(
        "model_id" in rule and "value" in rule and "passed" in rule for rule in payload["rules"]
    )


def test_spec_gate_keeps_the_v1_cli_exit_codes(tmp_path: Path) -> None:
    """The spec gate is a library API; the v1 CLI still exits 0/1/2 exactly as before."""
    runner = CliRunner()
    missing = runner.invoke(
        app,
        ["gate", "--baseline", str(tmp_path / "missing"), "--candidate", str(tmp_path)],
    )
    assert missing.exit_code == 2
