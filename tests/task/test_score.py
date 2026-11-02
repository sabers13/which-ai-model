"""The spec scorer on hand-computed numbers, including unclear and invalid answers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from which_ai_model.bench.metrics import macro_f1
from which_ai_model.contracts import ModelRef, TaskSpec
from which_ai_model.task import SpecPrediction, load_spec, score_spec

FIXTURE = Path("tests/fixtures/task")
MODEL = ModelRef(connection_id="fake", model_id="fake-model")

# Six items. Three are correct, one answers "unclear" where "technical" was right, one
# answer is schema-invalid (empty), and one bool answer is a false positive.
TRUTH: dict[str, dict[str, Any]] = {
    "i1": {"queue": "billing", "urgent": True},
    "i2": {"queue": "billing", "urgent": True},
    "i3": {"queue": "technical", "urgent": False},
    "i4": {"queue": "technical", "urgent": False},
    "i5": {"queue": "account", "urgent": True},
    "i6": {"queue": "account", "urgent": False},
}

OUTPUTS: dict[str, dict[str, Any]] = {
    "i1": {"queue": "billing", "urgent": True},
    "i2": {"queue": "technical", "urgent": False},
    "i3": {"queue": "technical", "urgent": True},
    "i4": {"queue": "unclear", "urgent": False},
    "i5": {"queue": "account", "urgent": True},
    "i6": {},
}

VALID = {"i1": True, "i2": True, "i3": True, "i4": True, "i5": True, "i6": False}


def _spec(repo_root: Path) -> TaskSpec:
    return load_spec(repo_root / FIXTURE / "example.toml")


def _predictions() -> list[SpecPrediction]:
    return [
        SpecPrediction(
            item_id=item_id,
            task="example",
            model=MODEL,
            output=output,
            raw_key=f"key-{item_id}",
        )
        for item_id, output in OUTPUTS.items()
    ]


def test_label_metrics_hand_computed(repo_root: Path) -> None:
    score = score_spec(_spec(repo_root), _predictions(), TRUTH, schema_valid=VALID)

    # 3 of 6 exact; 4 answered (i4 is unclear, i6 invalid); 3 of those 4 correct.
    assert score.metrics["queue.accuracy"] == pytest.approx(3 / 6)
    assert score.metrics["queue.coverage"] == pytest.approx(4 / 6)
    assert score.metrics["queue.accuracy_answered"] == pytest.approx(3 / 4)
    assert score.metrics["schema_valid_rate"] == pytest.approx(5 / 6)

    y_true = [TRUTH[item_id]["queue"] for item_id in sorted(TRUTH)]
    y_pred = ["billing", "technical", "technical", "unclear", "account", "unclear"]
    expected = macro_f1(y_true, y_pred)
    assert score.metrics["queue.macro_f1"] == pytest.approx(expected)
    assert score.primary == pytest.approx(expected)


def test_unclear_counts_as_wrong_and_is_not_covered(repo_root: Path) -> None:
    predictions = [
        replace(row, output={"queue": "technical", "urgent": False}) if row.item_id == "i4" else row
        for row in _predictions()
    ]
    answered = score_spec(_spec(repo_root), predictions, TRUTH, schema_valid=VALID)
    abstained = score_spec(_spec(repo_root), _predictions(), TRUTH, schema_valid=VALID)

    assert abstained.metrics["queue.coverage"] < answered.metrics["queue.coverage"]
    assert abstained.metrics["queue.accuracy_answered"] == pytest.approx(3 / 4)
    assert answered.metrics["queue.accuracy_answered"] == pytest.approx(4 / 5)
    assert abstained.metrics["queue.macro_f1"] < answered.metrics["queue.macro_f1"]


def test_bool_metrics_hand_computed(repo_root: Path) -> None:
    score = score_spec(_spec(repo_root), _predictions(), TRUTH, schema_valid=VALID)
    # truth urgent: yes, yes, no, no, yes, no; predicted: yes, no, yes, no, yes, yes
    # (the invalid last answer is always wrong, so it reads as a false positive here):
    # tp=2, fp=2, fn=1 -> precision 1/2, recall 2/3, f1 4/7.
    assert score.metrics["urgent.precision"] == pytest.approx(1 / 2)
    assert score.metrics["urgent.recall"] == pytest.approx(2 / 3)
    assert score.metrics["urgent.f1"] == pytest.approx(4 / 7)


def test_schema_invalid_answer_is_wrong(repo_root: Path) -> None:
    without_flags = score_spec(_spec(repo_root), _predictions(), TRUTH)
    assert without_flags.metrics["schema_valid_rate"] == pytest.approx(5 / 6)
    assert without_flags.metrics["queue.accuracy"] == pytest.approx(3 / 6)


def test_confidence_intervals_bracket_their_metric(repo_root: Path) -> None:
    score = score_spec(_spec(repo_root), _predictions(), TRUTH, schema_valid=VALID)
    for key in (
        "queue.accuracy",
        "queue.macro_f1",
        "queue.coverage",
        "queue.accuracy_answered",
        "urgent.precision",
        "urgent.recall",
        "urgent.f1",
        "schema_valid_rate",
    ):
        assert f"{key}_low" in score.ci
        assert f"{key}_high" in score.ci
        assert score.ci[f"{key}_low"] <= score.metrics[key] <= score.ci[f"{key}_high"]


def test_primary_can_be_a_bool_metric(repo_root: Path) -> None:
    spec = _spec(repo_root)
    spec = spec.model_copy(
        update={"metrics": spec.metrics.model_copy(update={"primary": "urgent.f1"})}
    )
    score = score_spec(spec, _predictions(), TRUTH, schema_valid=VALID)
    assert score.primary == pytest.approx(4 / 7)


def test_no_predictions_is_an_error(repo_root: Path) -> None:
    with pytest.raises(ValueError, match="no predictions"):
        score_spec(_spec(repo_root), [], TRUTH)


def test_missing_truth_row_is_named(repo_root: Path) -> None:
    truth = {key: value for key, value in TRUTH.items() if key != "i4"}
    with pytest.raises(ValueError, match="i4"):
        score_spec(_spec(repo_root), _predictions(), truth, schema_valid=VALID)


def test_duplicate_prediction_ids_are_rejected(repo_root: Path) -> None:
    predictions = [*_predictions(), _predictions()[0]]
    with pytest.raises(ValueError, match="repeat item ids"):
        score_spec(_spec(repo_root), predictions, TRUTH, schema_valid=VALID)
