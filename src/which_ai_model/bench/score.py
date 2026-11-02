"""Deterministic scoring and recommendations (``PLAN.md`` §7).

The scorer version is ``v1``. It reads predictions and per-call metadata only; it never
looks at the clock, the network or a random seed that is not fixed. Replaying the same
recordings therefore produces byte-identical ``cells.json`` files.

Metrics per task:

* **T-ROUTE** — accuracy and macro-F1, 95% bootstrap CI (2,000 resamples, fixed seed);
  primary metric ``macro_f1``.
* **T-SCOPE** — precision, recall and F1 for ``out_of_scope``, Wilson 95% CI; primary
  ``f1``.
* **T-REPLY** — schema-valid rate and intent-correct rate among valid outputs, Wilson
  95% CI; primary ``intent_correct``.
* **T-DOCS** — hit@4, citation precision and abstain-correct on unanswerable questions;
  primary ``hit_at_4``.

Recommendations use the fixed "good enough" rule: a candidate is good enough when its
primary metric is within **3.0 points** of the best candidate on that task. Per task the
report names the cheapest, the most accurate and the fastest good-enough candidate. Ties
go to the lower cost. The TF-IDF baseline row is included when its committed metrics
file exists; it has no latency, so it can be the cheapest or most accurate pick but never
the fastest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from which_ai_model.bench.metrics import (
    DEFAULT_SEED,
    accuracy,
    bootstrap_ci,
    bootstrap_metric_ci,
    citation_precision,
    latency_percentiles,
    macro_f1,
    precision_recall_f1,
    wilson_ci,
)
from which_ai_model.contracts import CellResult, ModelRef, Prediction, TaskName

__all__ = [
    "GOOD_ENOUGH_POINTS",
    "SCORER_VERSION",
    "Candidate",
    "SpecCellResult",
    "TaskScore",
    "baseline_candidates",
    "build_recommendations",
    "build_recommendations_with_specs",
    "candidates_from_cells",
    "evaluate_cell",
    "evaluate_spec_cell",
    "load_cells",
    "load_prediction_file",
    "load_spec_cells",
    "score_task",
    "spec_recommendation",
    "write_cells",
    "write_recommendations",
]

SCORER_VERSION = "v1"
GOOD_ENOUGH_POINTS = 3.0
_EPSILON = 1e-12


@dataclass(frozen=True)
class TaskScore:
    """Primary metric, the flat metric table and the confidence intervals for one task."""

    primary: float
    metrics: dict[str, float]
    ci: dict[str, float]


def _wilson(successes: int, n: int) -> tuple[float, float]:
    return wilson_ci(successes, n) if n else (0.0, 1.0)


def _route_score(y_true: Sequence[str], y_pred: Sequence[str]) -> TaskScore:
    correct = [
        1.0 if truth == predicted else 0.0 for truth, predicted in zip(y_true, y_pred, strict=True)
    ]
    accuracy_low, accuracy_high = bootstrap_ci(correct, seed=DEFAULT_SEED)
    f1_low, f1_high = bootstrap_metric_ci(y_true, y_pred, metric=macro_f1, seed=DEFAULT_SEED)
    metrics = {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}
    ci = {
        "accuracy_low": accuracy_low,
        "accuracy_high": accuracy_high,
        "macro_f1_low": f1_low,
        "macro_f1_high": f1_high,
    }
    return TaskScore(primary=metrics["macro_f1"], metrics=metrics, ci=ci)


def _scope_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return precision_recall_f1(y_true, y_pred, "out_of_scope")[2]


def _scope_score(y_true: Sequence[str], y_pred: Sequence[str]) -> TaskScore:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, "out_of_scope")
    true_positive = sum(
        1
        for truth, predicted in zip(y_true, y_pred, strict=True)
        if truth == predicted == "out_of_scope"
    )
    predicted_positive = sum(1 for predicted in y_pred if predicted == "out_of_scope")
    support = sum(1 for truth in y_true if truth == "out_of_scope")
    precision_low, precision_high = _wilson(true_positive, predicted_positive)
    recall_low, recall_high = _wilson(true_positive, support)
    f1_low, f1_high = bootstrap_metric_ci(y_true, y_pred, metric=_scope_f1, seed=DEFAULT_SEED)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy(y_true, y_pred),
    }
    ci = {
        "precision_low": precision_low,
        "precision_high": precision_high,
        "recall_low": recall_low,
        "recall_high": recall_high,
        "f1_low": f1_low,
        "f1_high": f1_high,
    }
    return TaskScore(primary=f1, metrics=metrics, ci=ci)


def _reply_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    valid: Sequence[bool],
) -> TaskScore:
    valid_count = sum(1 for flag in valid if flag)
    correct = sum(
        1
        for truth, predicted, flag in zip(y_true, y_pred, valid, strict=True)
        if flag and truth == predicted
    )
    schema_valid_rate = valid_count / len(valid)
    intent_correct = correct / valid_count if valid_count else 0.0
    schema_low, schema_high = _wilson(valid_count, len(valid))
    intent_low, intent_high = _wilson(correct, valid_count)
    metrics = {
        "schema_valid_rate": schema_valid_rate,
        "intent_correct": intent_correct,
        "intent_correct_all": correct / len(valid),
        "n_valid": float(valid_count),
    }
    ci = {
        "schema_valid_rate_low": schema_low,
        "schema_valid_rate_high": schema_high,
        "intent_correct_low": intent_low,
        "intent_correct_high": intent_high,
    }
    return TaskScore(primary=intent_correct, metrics=metrics, ci=ci)


def _docs_score(
    items: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> TaskScore:
    hits: list[float] = []
    citations: list[float] = []
    abstain_hits: list[float] = []
    for item in items:
        item_id = str(item["item_id"])
        output = predictions.get(item_id, {})
        gold = [str(value) for value in item.get("gold_ids", [])]
        retrieved = output.get("retrieved_ids", output.get("cited_ids", []))
        retrieved_ids = [str(value) for value in retrieved] if isinstance(retrieved, list) else []
        cited = output.get("cited_ids", [])
        cited_ids = [str(value) for value in cited] if isinstance(cited, list) else []
        if gold:
            hits.append(1.0 if set(gold) & set(retrieved_ids[:4]) else 0.0)
        citations.append(citation_precision(cited_ids, gold))
        if not bool(item.get("answerable", bool(gold))):
            abstain_hits.append(1.0 if bool(output.get("abstained", False)) else 0.0)

    hit_at_4 = sum(hits) / len(hits) if hits else 0.0
    citation_mean = sum(citations) / len(citations) if citations else 0.0
    abstain_correct = sum(abstain_hits) / len(abstain_hits) if abstain_hits else 0.0
    if hits:
        hit_low, hit_high = bootstrap_ci(hits, seed=DEFAULT_SEED)
    else:
        hit_low, hit_high = 0.0, 1.0
    if citations:
        citation_low, citation_high = bootstrap_ci(citations, seed=DEFAULT_SEED)
    else:
        citation_low, citation_high = 0.0, 1.0
    metrics = {
        "hit_at_4": hit_at_4,
        "citation_precision": citation_mean,
        "abstain_correct": abstain_correct,
        "n_answerable": float(len(hits)),
        "n_unanswerable": float(len(abstain_hits)),
    }
    ci = {
        "hit_at_4_low": hit_low,
        "hit_at_4_high": hit_high,
        "citation_precision_low": citation_low,
        "citation_precision_high": citation_high,
    }
    return TaskScore(primary=hit_at_4, metrics=metrics, ci=ci)


def score_task(
    task: TaskName,
    predictions: Sequence[Prediction],
    truth: Mapping[str, Mapping[str, Any]],
    *,
    schema_valid: Mapping[str, bool] | None = None,
) -> TaskScore:
    """Score one model x task cell against the sample's truth values.

    ``truth`` is keyed by item id and holds the gold fields (``intent``,
    ``out_of_scope``, ``gold_ids``/``answerable``). Predictions are matched by item id so
    a reduced T-ROUTE cell is scored against exactly the items it answered.
    """
    ordered = sorted(predictions, key=lambda prediction: prediction.item_id)
    if not ordered:
        msg = f"no predictions to score for {task}"
        raise ValueError(msg)
    missing = [prediction.item_id for prediction in ordered if prediction.item_id not in truth]
    if missing:
        msg = f"predictions for {task} have no truth row: {missing[:3]}"
        raise ValueError(msg)

    if task is TaskName.ROUTE:
        y_true = [str(truth[row.item_id]["intent"]) for row in ordered]
        y_pred = [str(row.output.get("intent", "")) for row in ordered]
        return _route_score(y_true, y_pred)
    if task is TaskName.SCOPE:
        y_true = [
            "out_of_scope" if bool(truth[row.item_id].get("out_of_scope", False)) else "in_scope"
            for row in ordered
        ]
        y_pred = [
            "out_of_scope" if bool(row.output.get("out_of_scope", False)) else "in_scope"
            for row in ordered
        ]
        return _scope_score(y_true, y_pred)
    if task is TaskName.REPLY:
        y_true = [str(truth[row.item_id]["intent"]) for row in ordered]
        y_pred = [str(row.output.get("intent", "")) for row in ordered]
        if schema_valid is None:
            valid = [bool(row.output) for row in ordered]
        else:
            valid = [bool(schema_valid.get(row.item_id, False)) for row in ordered]
        return _reply_score(y_true, y_pred, valid)

    items = [dict(truth[row.item_id]) for row in ordered]
    outputs = {row.item_id: row.output for row in ordered}
    return _docs_score(items, outputs)


def evaluate_cell(
    task: TaskName,
    model: ModelRef,
    predictions: Sequence[Prediction],
    truth: Mapping[str, Mapping[str, Any]],
    calls: Sequence[Any],
    *,
    schema_valid: Mapping[str, bool] | None = None,
) -> CellResult:
    """The runner's path: score with truth, then attach latency and cost."""
    score = score_task(task, predictions, truth, schema_valid=schema_valid)
    return _cell_from_score(task, model, score, calls, n=len(predictions))


def _cell_from_score(
    task: TaskName,
    model: ModelRef,
    score: TaskScore,
    calls: Sequence[Any],
    *,
    n: int | None = None,
) -> CellResult:
    latencies = [float(getattr(call, "latency_ms", 0)) for call in calls]
    if latencies:
        latency_p50, latency_p95 = latency_percentiles(latencies)
    else:
        latency_p50, latency_p95 = 0.0, 0.0
    costs = [getattr(call, "cost_usd", None) for call in calls]
    item_count = n if n is not None else len(calls)
    cost_per_1k: float | None
    if calls and item_count and all(cost is not None for cost in costs):
        cost_per_1k = float(sum(cost for cost in costs if cost is not None)) / item_count * 1_000
    else:
        cost_per_1k = None
    return CellResult(
        task=task,
        model=model,
        n=n if n is not None else len(calls) or 0,
        primary_metric=score.primary,
        metrics=score.metrics,
        ci=score.ci,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        cost_per_1k=cost_per_1k,
    )


@dataclass(frozen=True)
class SpecCellResult:
    """Aggregate for one model x spec task cell, written beside the v1 ``CellResult``s.

    ``CellResult.task`` is a ``TaskName`` and stays built-in-only (``PLAN-v1.1`` §A2.1),
    so a spec cell carries its spec id in ``task`` as plain text. Every other field,
    and every number, is the same as a v1 cell's.
    """

    task: str
    model: ModelRef
    model_id: str
    n: int
    primary_metric: float
    metrics: dict[str, float]
    ci: dict[str, float]
    latency_p50: float | None
    latency_p95: float | None
    cost_per_1k: float | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model.model_dump(mode="json"),
            "n": self.n,
            "primary_metric": self.primary_metric,
            "metrics": dict(self.metrics),
            "ci": dict(self.ci),
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "cost_per_1k": self.cost_per_1k,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> SpecCellResult:
        return cls(
            task=str(row["task"]),
            model=ModelRef.model_validate(row["model"]),
            model_id=str(ModelRef.model_validate(row["model"]).model_id),
            n=int(row["n"]),
            primary_metric=float(row["primary_metric"]),
            metrics={str(name): float(value) for name, value in dict(row["metrics"]).items()},
            ci={str(name): float(value) for name, value in dict(row.get("ci", {})).items()},
            latency_p50=(None if row.get("latency_p50") is None else float(row["latency_p50"])),
            latency_p95=(None if row.get("latency_p95") is None else float(row["latency_p95"])),
            cost_per_1k=None if row.get("cost_per_1k") is None else float(row["cost_per_1k"]),
        )


def evaluate_spec_cell(
    spec: Any,
    model: ModelRef,
    predictions: Sequence[Any],
    truth: Mapping[str, Mapping[str, Any]],
    calls: Sequence[Any],
    *,
    schema_valid: Mapping[str, bool] | None = None,
) -> SpecCellResult:
    """The runner's path for one spec cell: M1's scorer plus latency and cost.

    ``spec`` and ``predictions`` are imported lazily so this module never imports the
    task package at module load (the task package imports this one).
    """
    from which_ai_model.task.score import score_spec

    score = score_spec(spec, predictions, truth, schema_valid=schema_valid)
    # A replayed call has no measured latency (the runner serves it from a recording),
    # so a zero is "unknown", never "instant". A cell with no measured call at all
    # reports null and the picker's fastest rule skips it.
    latencies = [
        float(getattr(call, "latency_ms", 0))
        for call in calls
        if float(getattr(call, "latency_ms", 0)) > 0
    ]
    if latencies:
        latency_p50, latency_p95 = latency_percentiles(latencies)
    else:
        latency_p50, latency_p95 = None, None
    costs = [getattr(call, "cost_usd", None) for call in calls]
    item_count = len(predictions)
    cost_per_1k: float | None
    if calls and item_count and all(cost is not None for cost in costs):
        cost_per_1k = float(sum(cost for cost in costs if cost is not None)) / item_count * 1_000
    else:
        cost_per_1k = None
    return SpecCellResult(
        task=str(spec.id),
        model=model,
        model_id=model.model_id,
        n=item_count,
        primary_metric=score.primary,
        metrics=score.metrics,
        ci=score.ci,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        cost_per_1k=cost_per_1k,
    )


def write_cells(
    path: Path,
    cells: Sequence[CellResult],
    *,
    spec_cells: Sequence[SpecCellResult] = (),
) -> None:
    """Write ``cells.json`` deterministically (sorted by task, connection, model).

    ``spec_cells`` are written under their own key beside the v1 rows, so a v1
    ``cells.json`` keeps exactly its old bytes when no spec cell ran (``PLAN-v1.1`` §A2).
    """
    ordered = sorted(
        cells,
        key=lambda cell: (cell.task.value, cell.model.connection_id, cell.model.model_id),
    )
    payload: dict[str, Any] = {
        "scorer_version": SCORER_VERSION,
        "cells": [cell.model_dump(mode="json") for cell in ordered],
    }
    if spec_cells:
        ordered_spec = sorted(
            spec_cells,
            key=lambda cell: (cell.task, cell.model.connection_id, cell.model.model_id),
        )
        payload["spec_cells"] = [cell.as_json() for cell in ordered_spec]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cells(path: Path) -> list[CellResult]:
    """Read ``cells.json`` written by :func:`write_cells` (v1 rows only)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("cells", data) if isinstance(data, dict) else data
    return [CellResult.model_validate(row) for row in rows]


def load_spec_cells(path: Path) -> list[SpecCellResult]:
    """Read the spec rows written beside the v1 cells; empty when there are none."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("spec_cells", []) if isinstance(data, dict) else []
    return [SpecCellResult.from_json(row) for row in rows if isinstance(row, dict)]


def load_prediction_file(path: Path) -> list[Prediction]:
    """Read one ``predictions/*.jsonl`` file."""
    rows: list[Prediction] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(Prediction.model_validate_json(line))
    return rows


@dataclass(frozen=True)
class Candidate:
    """One comparable row for the recommendation rule (a model cell or the baseline)."""

    label: str
    task: TaskName
    model: ModelRef
    primary: float
    cost_per_1k: float | None
    latency_p50: float | None
    baseline: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model.model_id,
            "connection_id": self.model.connection_id,
            "primary_metric": self.primary,
            "cost_per_1k": self.cost_per_1k,
            "latency_p50": self.latency_p50,
            "baseline": self.baseline,
        }


def candidates_from_cells(cells: Sequence[CellResult]) -> list[Candidate]:
    """Turn model cells into recommendation candidates."""
    return [
        Candidate(
            label=f"{cell.model.connection_id}/{cell.model.model_id}",
            task=cell.task,
            model=cell.model,
            primary=cell.primary_metric,
            cost_per_1k=cell.cost_per_1k,
            latency_p50=cell.latency_p50,
        )
        for cell in cells
    ]


def baseline_candidates(path: Path) -> list[Candidate]:
    """Read T04's ``results/baseline/metrics.json`` as baseline candidate rows.

    The baseline is deterministic and costs 0. It has no latency measurement (it runs
    locally, not through a gateway), so ``latency_p50`` is ``None`` and it never wins
    "fastest". A missing or unreadable file simply leaves the baseline out.
    """
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    model = ModelRef(connection_id="baseline", model_id="tfidf-logreg")
    candidates: list[Candidate] = []
    route = data.get("route")
    if isinstance(route, dict) and isinstance(route.get("macro_f1"), (int, float)):
        candidates.append(
            Candidate(
                label="baseline/tfidf-logreg",
                task=TaskName.ROUTE,
                model=model,
                primary=float(route["macro_f1"]),
                cost_per_1k=0.0,
                latency_p50=None,
                baseline=True,
            )
        )
    scope = data.get("scope")
    if isinstance(scope, dict) and isinstance(scope.get("f1"), (int, float)):
        candidates.append(
            Candidate(
                label="baseline/tfidf-logreg",
                task=TaskName.SCOPE,
                model=model,
                primary=float(scope["f1"]),
                cost_per_1k=0.0,
                latency_p50=None,
                baseline=True,
            )
        )
    return candidates


def _cost_key(candidate: Candidate) -> tuple[bool, float, str]:
    return (
        candidate.cost_per_1k is None,
        candidate.cost_per_1k if candidate.cost_per_1k is not None else 0.0,
        candidate.label,
    )


def _pick_cheapest(candidates: Sequence[Candidate]) -> Candidate | None:
    return min(candidates, key=_cost_key, default=None)


def _pick_most_accurate(candidates: Sequence[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: (-c.primary, *_cost_key(c)))


def _pick_fastest(candidates: Sequence[Candidate]) -> Candidate | None:
    timed = [candidate for candidate in candidates if candidate.latency_p50 is not None]
    if not timed:
        return None
    return min(
        timed,
        key=lambda c: (c.latency_p50 if c.latency_p50 is not None else 0.0, *_cost_key(c)),
    )


def build_recommendations(
    cells: Sequence[CellResult],
    *,
    baseline: Sequence[Candidate] | None = None,
    good_enough_points: float = GOOD_ENOUGH_POINTS,
) -> dict[str, Any]:
    """Apply the fixed good-enough rule per task and return the report payload."""
    candidates = [*candidates_from_cells(cells), *(baseline or [])]
    tasks: dict[str, Any] = {}
    for task in TaskName:
        rows = [candidate for candidate in candidates if candidate.task is task]
        if not rows:
            continue
        best = max(row.primary for row in rows)
        threshold = best - (good_enough_points / 100.0) - _EPSILON
        good = [row for row in rows if row.primary >= threshold]
        ordered_good = sorted(good, key=lambda c: (-c.primary, *_cost_key(c)))
        tasks[task.value] = {
            "best_primary": best,
            "good_enough": [row.as_json() for row in ordered_good],
            "cheapest": _json_or_none(_pick_cheapest(good)),
            "most_accurate": _json_or_none(_pick_most_accurate(rows)),
            "fastest": _json_or_none(_pick_fastest(good)),
        }
    return {
        "scorer_version": SCORER_VERSION,
        "good_enough_points": good_enough_points,
        "tasks": tasks,
    }


def _json_or_none(candidate: Candidate | None) -> dict[str, Any] | None:
    return None if candidate is None else candidate.as_json()


def _spec_row_json(cell: SpecCellResult | None) -> dict[str, Any] | None:
    if cell is None:
        return None
    return {
        "model": cell.model.model_id,
        "connection_id": cell.model.connection_id,
        "primary_metric": cell.primary_metric,
        "cost_per_1k": cell.cost_per_1k,
        "latency_p50": cell.latency_p50,
        "baseline": False,
    }


@dataclass
class _PickerCell:
    """A mutable view of a spec cell for M3's gate picker protocol."""

    model_id: str
    primary_metric: float
    metrics: Mapping[str, float]
    cost_per_1k: float | None
    latency_p50: float | None


def spec_recommendation(
    spec: Any,
    cells: Sequence[SpecCellResult],
) -> dict[str, Any]:
    """One spec task's entry in ``results/recommendations.json`` via M3's picker.

    A model below a gate floor is never eligible, and the no-model-meets-the-floor
    outcome is explicit: cheapest and fastest are ``None``,
    ``most_accurate`` names the best model, and ``no_model_meets_floor`` is ``True``
    (``PLAN-v1.1`` §A4). The shape matches the v1 entries, with the extra flag and rule
    so a card can state why nothing was picked.
    """
    from which_ai_model.gate.spec import PICKER_WITHIN_POINTS, pick

    rows = sorted(cells, key=lambda cell: (cell.model.connection_id, cell.model.model_id))
    picker_rows = [
        _PickerCell(
            model_id=cell.model_id,
            primary_metric=cell.primary_metric,
            metrics=cell.metrics,
            cost_per_1k=cell.cost_per_1k,
            latency_p50=cell.latency_p50,
        )
        for cell in rows
    ]
    picked = pick(picker_rows, floors=spec.gate.floors)
    by_id = {cell.model_id: cell for cell in rows}
    best = max((cell.primary_metric for cell in rows), default=None)
    return {
        "best_primary": best,
        "good_enough": [_spec_row_json(by_id[model_id]) for model_id in picked.good_enough],
        "cheapest": _spec_row_json(None if picked.cheapest is None else by_id.get(picked.cheapest)),
        "most_accurate": _spec_row_json(
            None if picked.most_accurate is None else by_id.get(picked.most_accurate)
        ),
        "fastest": _spec_row_json(None if picked.fastest is None else by_id.get(picked.fastest)),
        "no_model_meets_floor": picked.no_model_meets_floor,
        "gate_floors": [floor.model_dump(mode="json") for floor in spec.gate.floors],
        "picker_rule": (
            f"within {PICKER_WITHIN_POINTS:.1f} points of the best model on the primary "
            "metric; a model below any gate floor is not good enough; ties go to lower cost"
        ),
    }


def build_recommendations_with_specs(
    cells: Sequence[CellResult],
    spec_cells: Sequence[SpecCellResult],
    specs: Sequence[Any],
    *,
    baseline: Sequence[Candidate] | None = None,
    good_enough_points: float = GOOD_ENOUGH_POINTS,
) -> dict[str, Any]:
    """The v1 payload with each spec task added beside its four built-in entries.

    With no spec cells this is byte-for-byte the v1 payload: the v1 tasks keep their
    exact entries and key order.
    """
    payload = build_recommendations(
        cells,
        baseline=baseline,
        good_enough_points=good_enough_points,
    )
    rows_by_task: dict[str, list[SpecCellResult]] = {}
    for cell in spec_cells:
        rows_by_task.setdefault(cell.task, []).append(cell)
    for spec in specs:
        rows = rows_by_task.get(str(spec.id))
        if rows:
            payload["tasks"][str(spec.id)] = spec_recommendation(spec, rows)
    return payload


def write_recommendations(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``results/recommendations.json`` deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
