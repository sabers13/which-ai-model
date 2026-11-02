"""Score one model x spec cell (``PLAN-v1.1`` §A3).

Deterministic and offline: every metric comes from the predictions and the committed
truth rows. The confidence intervals reuse :mod:`which_ai_model.bench.metrics` exactly
as v1 does — a paired bootstrap for macro-F1 and every F1, Wilson for rates.

Label metrics:

* ``accuracy`` — exact matches over every item;
* ``macro_f1`` — v1's unweighted mean F1; ``unclear`` and a schema-invalid answer are
  wrong, so abstaining is never free;
* ``coverage`` — the share of items answered (schema-valid and not ``unclear``);
* ``accuracy_answered`` — accuracy among the answered items (0 when none were answered).

Bool metrics are precision, recall and F1 for the ``true`` class. ``schema_valid_rate``
is reported once per cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from which_ai_model.bench.metrics import (
    DEFAULT_SEED,
    accuracy,
    bootstrap_ci,
    bootstrap_metric_ci,
    macro_f1,
    precision_recall_f1,
    wilson_ci,
)
from which_ai_model.bench.score import TaskScore
from which_ai_model.contracts import OutputKind, OutputSpec, TaskSpec
from which_ai_model.task.predict import SpecPrediction
from which_ai_model.task.spec import UNCLEAR

__all__ = ["score_spec"]


def _validity(
    predictions: Sequence[SpecPrediction],
    schema_valid: Mapping[str, bool] | None,
) -> dict[str, bool]:
    if schema_valid is None:
        return {row.item_id: bool(row.output) for row in predictions}
    return {
        row.item_id: bool(schema_valid.get(row.item_id, bool(row.output))) for row in predictions
    }


def _label_value(row: SpecPrediction, output: OutputSpec, valid: bool) -> str:
    value = row.output.get(output.name)
    if not valid or not isinstance(value, str) or not value:
        return UNCLEAR
    return value


def _bool_value(
    row: SpecPrediction,
    output: OutputSpec,
    valid: bool,
    truth_value: bool,
) -> bool:
    """The yes/no answer, with an invalid answer made the opposite of the truth.

    A schema-invalid answer is wrong by rule, so it can never be scored as a correct
    "no" just because a missing field reads as false. ``schema_valid_rate`` counts the
    invalid answer separately as well.
    """
    value = row.output.get(output.name)
    if valid and isinstance(value, bool):
        return value
    return not truth_value


def _wilson(successes: int, n: int) -> tuple[float, float]:
    return wilson_ci(successes, n) if n else (0.0, 1.0)


def _bool_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return precision_recall_f1(y_true, y_pred, "yes")[2]


def _score_label(
    output: OutputSpec,
    truth: Mapping[str, Mapping[str, Any]],
    predictions: Sequence[SpecPrediction],
    valid: Mapping[str, bool],
    metrics: dict[str, float],
    ci: dict[str, float],
) -> None:
    y_true = [str(truth[row.item_id][output.truth_field]) for row in predictions]
    y_pred = [_label_value(row, output, valid[row.item_id]) for row in predictions]
    correct = [
        1.0 if truth_value == predicted else 0.0
        for truth_value, predicted in zip(y_true, y_pred, strict=True)
    ]
    answered = [
        1.0 if valid[row.item_id] and _label_value(row, output, True) != UNCLEAR else 0.0
        for row in predictions
    ]
    answered_count = int(sum(answered))
    answered_correct = sum(
        1.0
        for row, truth_value, predicted in zip(predictions, y_true, y_pred, strict=True)
        if valid[row.item_id] and predicted != UNCLEAR and truth_value == predicted
    )

    prefix = output.name
    metrics[f"{prefix}.accuracy"] = accuracy(y_true, y_pred)
    metrics[f"{prefix}.macro_f1"] = macro_f1(y_true, y_pred)
    metrics[f"{prefix}.coverage"] = answered_count / len(predictions)
    metrics[f"{prefix}.accuracy_answered"] = (
        answered_correct / answered_count if answered_count else 0.0
    )

    accuracy_low, accuracy_high = bootstrap_ci(correct, seed=DEFAULT_SEED)
    macro_low, macro_high = bootstrap_metric_ci(y_true, y_pred, seed=DEFAULT_SEED)
    coverage_low, coverage_high = _wilson(answered_count, len(predictions))
    answered_low, answered_high = _wilson(int(answered_correct), answered_count)
    ci[f"{prefix}.accuracy_low"] = accuracy_low
    ci[f"{prefix}.accuracy_high"] = accuracy_high
    ci[f"{prefix}.macro_f1_low"] = macro_low
    ci[f"{prefix}.macro_f1_high"] = macro_high
    ci[f"{prefix}.coverage_low"] = coverage_low
    ci[f"{prefix}.coverage_high"] = coverage_high
    ci[f"{prefix}.accuracy_answered_low"] = answered_low
    ci[f"{prefix}.accuracy_answered_high"] = answered_high


def _score_bool(
    output: OutputSpec,
    truth: Mapping[str, Mapping[str, Any]],
    predictions: Sequence[SpecPrediction],
    valid: Mapping[str, bool],
    metrics: dict[str, float],
    ci: dict[str, float],
) -> None:
    truth_values = [bool(truth[row.item_id][output.truth_field]) for row in predictions]
    y_true = ["yes" if value else "no" for value in truth_values]
    y_pred = [
        "yes" if _bool_value(row, output, valid[row.item_id], truth_value) else "no"
        for row, truth_value in zip(predictions, truth_values, strict=True)
    ]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, "yes")
    true_positive = sum(
        1
        for truth_value, predicted in zip(y_true, y_pred, strict=True)
        if truth_value == predicted == "yes"
    )
    predicted_positive = sum(1 for predicted in y_pred if predicted == "yes")
    support = sum(1 for truth_value in y_true if truth_value == "yes")

    prefix = output.name
    metrics[f"{prefix}.precision"] = precision
    metrics[f"{prefix}.recall"] = recall
    metrics[f"{prefix}.f1"] = f1

    precision_low, precision_high = _wilson(true_positive, predicted_positive)
    recall_low, recall_high = _wilson(true_positive, support)
    f1_low, f1_high = bootstrap_metric_ci(y_true, y_pred, metric=_bool_f1, seed=DEFAULT_SEED)
    ci[f"{prefix}.precision_low"] = precision_low
    ci[f"{prefix}.precision_high"] = precision_high
    ci[f"{prefix}.recall_low"] = recall_low
    ci[f"{prefix}.recall_high"] = recall_high
    ci[f"{prefix}.f1_low"] = f1_low
    ci[f"{prefix}.f1_high"] = f1_high


def score_spec(
    spec: TaskSpec,
    predictions: Sequence[SpecPrediction],
    truth: Mapping[str, Mapping[str, Any]],
    *,
    schema_valid: Mapping[str, bool] | None = None,
) -> TaskScore:
    """Score one model x spec cell; reuse ``bench.metrics`` for the CIs.

    ``schema_valid`` overrides the default reading, which is "a non-empty parsed output
    is valid". The primary metric is ``spec.metrics.primary``; every metric the spec
    names is present, plus the rest of its output kind's metrics.
    """
    if not predictions:
        raise ValueError(f"no predictions to score for {spec.id}")
    ordered = sorted(predictions, key=lambda row: row.item_id)
    ids = [row.item_id for row in ordered]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"predictions for {spec.id} repeat item ids: {duplicates[:3]}")
    missing = [row.item_id for row in ordered if row.item_id not in truth]
    if missing:
        raise ValueError(f"predictions for {spec.id} have no truth row: {missing[:3]}")

    valid = _validity(ordered, schema_valid)
    metrics: dict[str, float] = {}
    ci: dict[str, float] = {}
    for output in spec.outputs:
        if output.kind is OutputKind.LABEL:
            _score_label(output, truth, ordered, valid, metrics, ci)
        else:
            _score_bool(output, truth, ordered, valid, metrics, ci)

    valid_count = sum(1 for flag in valid.values() if flag)
    metrics["schema_valid_rate"] = valid_count / len(ordered)
    schema_low, schema_high = _wilson(valid_count, len(ordered))
    ci["schema_valid_rate_low"] = schema_low
    ci["schema_valid_rate_high"] = schema_high

    primary = spec.metrics.primary
    if primary not in metrics:  # the contract checks this; keep the failure plain
        raise ValueError(f"the spec's primary metric {primary!r} was not scored")
    return TaskScore(primary=metrics[primary], metrics=metrics, ci=ci)
