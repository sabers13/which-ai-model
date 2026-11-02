"""Use-case cards: one per task, built only from committed ``results/`` (``PLAN-v1.1`` §A7).

A card states the job, the data and its licence, a per-model results table with the
primary metric's interval, the cost-0 baseline where one exists, the recommendation and
the rule that produced it, the gate rule, when to re-check and the exact command, the
cost at monthly volume, and the known limits.

The four built-in cards are read from the committed v1 live run; no model is called and
nothing is recomputed from a live answer. A spec card is read from a replay run's
``spec_cells`` (``results/runs/live-v1.1/cells.json``) and the spec's own gate floors.
The recommendation is the same rule the committed ``results/recommendations.json``
records: v1's 3.0-point band for built-ins, and M3's floor-aware picker for a spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from which_ai_model.bench.score import (
    SpecCellResult,
    baseline_candidates,
    build_recommendations,
    load_cells,
    load_spec_cells,
)
from which_ai_model.contracts import (
    CardData,
    CardModelRow,
    CellResult,
    GateSpec,
    TaskName,
    TaskRef,
    TaskSpec,
    builtin_task,
    task_key,
)
from which_ai_model.report.prompt_library import export_prompt_library
from which_ai_model.report.volume import volume_rows
from which_ai_model.task import load_spec

__all__ = [
    "BUILTIN_RUN_ID",
    "SPEC_RUN_ID",
    "build_card",
    "build_cards",
    "build_spec_card",
    "write_card",
    "write_prompt_library",
]

#: The committed v1 live run every built-in card reads (byte-identical v1 evidence).
BUILTIN_RUN_ID = "live"
#: The committed v1.1 spec run id (``PLAN-v1.1`` §A6/§A10).
SPEC_RUN_ID = "live-v1.1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASKS_DIR = Path("tasks")
_PICKER_RULE = (
    "within 3.0 points of the best model on the primary metric; "
    "a model below any gate floor is not good enough; ties go to lower cost"
)


@dataclass(frozen=True)
class _BuiltinMeta:
    """The curated prose for one built-in card; every number still comes from results."""

    title: str
    job: str
    data_source: str
    licence: str
    metric_order: tuple[str, ...]
    primary_ci: tuple[str, str]
    baseline_key: str | None = None
    limits: tuple[str, ...] = ()


_BUILTIN: dict[TaskName, _BuiltinMeta] = {
    TaskName.ROUTE: _BuiltinMeta(
        title="Route a support message",
        job="Send each customer message to the one of 77 banking intents that fits it.",
        data_source="Banking77 (PolyAI), test split, 8 messages per intent",
        licence="CC-BY-4.0",
        metric_order=("accuracy", "macro_f1"),
        primary_ci=("macro_f1_low", "macro_f1_high"),
        baseline_key="route",
        limits=(
            "77 fixed English-language banking intents; a different intent set needs its own task.",
            "Public Banking77 messages may be known to the models.",
        ),
    ),
    TaskName.SCOPE: _BuiltinMeta(
        title="Check whether a message is in scope",
        job="Decide whether a support message is inside the supported banking scope or outside it.",
        data_source="CLINC150 plus (CLINC), test split, 150 out-of-scope + 150 in-scope",
        licence="CC-BY-3.0",
        metric_order=("precision", "recall", "f1", "accuracy"),
        primary_ci=("f1_low", "f1_high"),
        baseline_key="scope",
        limits=(
            "Binary decision on one fixed English-language sample; the in-scope mix decides the score.",
            "Public CLINC150 messages may be known to the models.",
        ),
    ),
    TaskName.REPLY: _BuiltinMeta(
        title="Draft a structured reply",
        job="Turn a customer message into one typed object: intent, urgency and reply text.",
        data_source="Banking77 (PolyAI), a seeded 150-item subset of the routing sample",
        licence="CC-BY-4.0",
        metric_order=("schema_valid_rate", "intent_correct", "intent_correct_all"),
        primary_ci=("intent_correct_low", "intent_correct_high"),
        limits=(
            "Intent-correct is measured among schema-valid answers; reply text is not judged by a model or a person.",
            "No non-AI baseline is defined for free-text replies.",
        ),
    ),
    TaskName.DOCS: _BuiltinMeta(
        title="Answer from the guide",
        job="Answer a question about this stack from retrieved guide passages and cite the passages used.",
        data_source="docs/guide in this repository (this project's own text)",
        licence="MIT (this repository)",
        metric_order=(
            "hit_at_4",
            "citation_precision",
            "abstain_correct",
            "n_answerable",
            "n_unanswerable",
        ),
        primary_ci=("hit_at_4_low", "hit_at_4_high"),
        limits=(
            "The corpus is this project's guide, not a general knowledge base.",
            "Five questions are deliberately unanswerable; abstaining is the correct answer there.",
        ),
    ),
}

_SPEC_META: dict[str, dict[str, Any]] = {
    "defect": {
        "data_source": (
            "NHTSA ODI vehicle owner complaints (2020-2024 and 2025-2026 received-date "
            "flat files), sampled into "
            "data/defect/test.jsonl"
        ),
        "licence": "U.S. government work, public domain",
        "limits": (
            "The sample oversamples safety-relevant complaints; the natural rate and the PII drops are in data/defect/STATS.md.",
            "A model below the safety-recall floor is never recommended, whatever its group accuracy.",
            "One fixed English-language sample; another fleet, period or language needs its own run.",
        ),
    }
}

_GENERIC_LIMITS = (
    "One fixed sample; a different domain, language or message mix needs its own run.",
    "Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.",
    "Point estimates carry 95% intervals; compare intervals before treating a small gap as real.",
)


# --------------------------------------------------------------------- card rows


def _ci(cell: Mapping[str, Any], keys: tuple[str, str]) -> tuple[float, float] | None:
    low, high = keys
    values = cell.get("ci")
    if not isinstance(values, Mapping) or low not in values or high not in values:
        return None
    return (float(values[low]), float(values[high]))


def _row_from_cell(cell: CellResult, meta: _BuiltinMeta) -> CardModelRow:
    return CardModelRow(
        model_id=cell.model.model_id,
        primary_metric=cell.primary_metric,
        primary_ci=_ci(cell.model_dump(mode="json"), meta.primary_ci),
        metrics={name: float(value) for name, value in cell.metrics.items()},
        cost_per_1k=cell.cost_per_1k,
        latency_p50=cell.latency_p50,
    )


def _spec_primary_ci(spec: TaskSpec, cell: SpecCellResult) -> tuple[float, float] | None:
    primary = spec.metrics.primary
    low = cell.ci.get(f"{primary}_low")
    high = cell.ci.get(f"{primary}_high")
    if low is None or high is None:
        return None
    return (float(low), float(high))


def _row_from_spec_cell(
    spec: TaskSpec, cell: SpecCellResult, primary_ci: tuple[float, float] | None
) -> CardModelRow:
    return CardModelRow(
        model_id=cell.model.model_id,
        primary_metric=cell.primary_metric,
        primary_ci=primary_ci,
        metrics={name: float(value) for name, value in cell.metrics.items()},
        cost_per_1k=cell.cost_per_1k,
        latency_p50=cell.latency_p50,
    )


# --------------------------------------------------------------------- baseline


def _route_baseline(results_dir: Path) -> CardModelRow | None:
    path = results_dir / "baseline" / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    route = data.get("route") if isinstance(data, Mapping) else None
    if not isinstance(route, Mapping) or not isinstance(route.get("macro_f1"), (int, float)):
        return None
    return CardModelRow(
        model_id="tfidf-logreg",
        primary_metric=float(route["macro_f1"]),
        primary_ci=_ci(route, ("macro_f1_low", "macro_f1_high")),
        metrics={
            "accuracy": float(route.get("accuracy", 0.0)),
            "macro_f1": float(route["macro_f1"]),
        },
        cost_per_1k=0.0,
        latency_p50=None,
        is_baseline=True,
    )


def _scope_baseline(results_dir: Path) -> CardModelRow | None:
    path = results_dir / "baseline" / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    scope = data.get("scope") if isinstance(data, Mapping) else None
    if not isinstance(scope, Mapping) or not isinstance(scope.get("f1"), (int, float)):
        return None
    return CardModelRow(
        model_id="tfidf-logreg",
        primary_metric=float(scope["f1"]),
        primary_ci=_ci(scope, ("f1_low", "f1_high")),
        metrics={
            "precision": float(scope.get("precision", 0.0)),
            "recall": float(scope.get("recall", 0.0)),
            "f1": float(scope["f1"]),
            "accuracy": float(scope.get("accuracy", 0.0)),
        },
        cost_per_1k=0.0,
        latency_p50=None,
        is_baseline=True,
    )


def _spec_baseline(results_dir: Path, task_id: str) -> CardModelRow | None:
    path = results_dir / "baseline" / f"{task_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    group = data.get("component_group") if isinstance(data, Mapping) else None
    if not isinstance(group, Mapping) or not isinstance(group.get("macro_f1"), (int, float)):
        return None
    safety = data.get("safety_relevant", {}) if isinstance(data, Mapping) else {}
    safety = safety if isinstance(safety, Mapping) else {}
    metrics = {
        "component_group.accuracy": float(group.get("accuracy", 0.0)),
        "component_group.macro_f1": float(group["macro_f1"]),
        "component_group.coverage": float(group.get("coverage", 0.0)),
        "component_group.accuracy_answered": float(group.get("accuracy_answered", 0.0)),
        "safety_relevant.recall": float(safety.get("recall", 0.0)),
        "safety_relevant.precision": float(safety.get("precision", 0.0)),
        "safety_relevant.f1": float(safety.get("f1", 0.0)),
        "schema_valid_rate": float(data.get("schema_valid_rate", 1.0)),
    }
    return CardModelRow(
        model_id="tfidf-logreg + keyword rule",
        primary_metric=float(group["macro_f1"]),
        primary_ci=_ci(group, ("macro_f1_low", "macro_f1_high")),
        metrics=metrics,
        cost_per_1k=0.0,
        latency_p50=None,
        is_baseline=True,
    )


def _baseline_for(task: TaskName, results_dir: Path) -> CardModelRow | None:
    if task is TaskName.ROUTE:
        return _route_baseline(results_dir)
    if task is TaskName.SCOPE:
        return _scope_baseline(results_dir)
    return None


# ------------------------------------------------------------------ built-in card


def _builtin_picks(
    cells: Sequence[CellResult], task: TaskName, results_dir: Path
) -> Mapping[str, Mapping[str, Any] | None]:
    baseline = list(baseline_candidates(results_dir / "baseline" / "metrics.json"))
    payload = build_recommendations(
        [cell for cell in cells if cell.task is task],
        baseline=[row for row in baseline if row.task is task],
    )
    entry = payload["tasks"].get(task.value, {})
    return entry if isinstance(entry, Mapping) else {}


def _pick_id(entry: Mapping[str, Any], name: str) -> str | None:
    value = entry.get(name)
    if isinstance(value, Mapping) and isinstance(value.get("model"), str):
        return str(value["model"])
    return None


def _builtin_prompt_version(runs_dir: Path) -> str:
    path = runs_dir / BUILTIN_RUN_ID / "manifest.json"
    if not path.is_file():
        return "v1"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping) and isinstance(data.get("prompt_version"), str):
        return str(data["prompt_version"])
    return "v1"


def _builtin_card(
    task: TaskName,
    cells: Sequence[CellResult],
    results_dir: Path,
    *,
    runs_dir: Path = Path("results/runs"),
) -> CardData:
    meta = _BUILTIN[task]
    rows = sorted(
        (cell for cell in cells if cell.task is task),
        key=lambda cell: (-cell.primary_metric, cell.model.model_id),
    )
    if not rows:
        msg = f"no committed cells for {task.value} in results/runs/{BUILTIN_RUN_ID}"
        raise FileNotFoundError(msg)
    entry = _builtin_picks(rows, task, results_dir)
    baseline = _baseline_for(task, results_dir)
    card = CardData(
        task=task.value,
        title=meta.title,
        job=meta.job,
        data_source=meta.data_source,
        licence=meta.licence,
        sample_size=rows[0].n,
        prompt_version=_builtin_prompt_version(runs_dir),
        models=[_row_from_cell(cell, meta) for cell in rows],
        baseline=baseline,
        cheapest=_pick_id(entry, "cheapest"),
        most_accurate=_pick_id(entry, "most_accurate"),
        fastest=_pick_id(entry, "fastest"),
        picker_rule=_PICKER_RULE,
        gate=GateSpec(),
        recheck=(
            f"`wam recheck run --tasks {task.value} --against results/runs/{BUILTIN_RUN_ID}` "
            "when a new model is released, when a model changes under the same id, or "
            "after a prompt change (that needs a new baseline run first)."
        ),
        limits=[*meta.limits, *_GENERIC_LIMITS],
        notes=[
            (
                f"Every number is read from `results/runs/{BUILTIN_RUN_ID}/cells.json`, "
                "`results/baseline/metrics.json` and the committed recommendations. "
                "No model call was made to build this card."
            )
        ],
    )
    return card.model_copy(update={"volumes": volume_rows(card)})


# ---------------------------------------------------------------------- spec card


@dataclass
class _PickerCell:
    """A mutable view of a spec cell for M3's gate picker protocol."""

    model_id: str
    primary_metric: float
    metrics: Mapping[str, float]
    cost_per_1k: float | None
    latency_p50: float | None


def _spec_picks(
    spec: TaskSpec, cells: Sequence[SpecCellResult]
) -> tuple[str | None, str | None, str | None, bool]:
    from which_ai_model.gate.spec import pick

    picker_rows = [
        _PickerCell(
            model_id=cell.model_id,
            primary_metric=cell.primary_metric,
            metrics=cell.metrics,
            cost_per_1k=cell.cost_per_1k,
            latency_p50=cell.latency_p50,
        )
        for cell in cells
    ]
    picked = pick(picker_rows, floors=spec.gate.floors)
    return picked.cheapest, picked.most_accurate, picked.fastest, picked.no_model_meets_floor


def build_spec_card(
    spec: TaskSpec,
    cells: Sequence[SpecCellResult],
    *,
    baseline: CardModelRow | None = None,
) -> CardData:
    """One spec task's card from its cells; no model call and no live answer."""
    ordered = sorted(cells, key=lambda cell: (-cell.primary_metric, cell.model_id))
    if not ordered:
        msg = f"no spec cells to build the {spec.id!r} card"
        raise FileNotFoundError(msg)
    spec_meta = _SPEC_META.get(spec.id, {})
    cheapest, most_accurate, fastest, no_floor = _spec_picks(spec, ordered)
    # PLAN-v1.1 §A10 may cut one cell to fewer items to stay under the cap. The card
    # states the full sample and names any cell that was scored on fewer items, so a
    # reader never compares a 400-item interval with a 600-item one unaware.
    counts = {cell.model_id: cell.n for cell in ordered}
    full_sample = max(counts.values())
    partial = sorted((model, n) for model, n in counts.items() if n != full_sample)
    reduction_limits = (
        [
            "Scored on a reduced sample after the §A10 budget cut: "
            + ", ".join(f"{model} on {n} of {full_sample} items" for model, n in partial)
            + "; its interval is wider, so compare intervals before treating a gap as real."
        ]
        if partial
        else []
    )
    card = CardData(
        task=spec.id,
        title=spec.title,
        job=spec.description,
        data_source=str(spec_meta.get("data_source", spec.data.file)),
        licence=str(spec_meta.get("licence", "See THIRD_PARTY.md for the task's data licence.")),
        sample_size=full_sample,
        prompt_version=spec.prompt_version,
        models=[_row_from_spec_cell(spec, cell, _spec_primary_ci(spec, cell)) for cell in ordered],
        baseline=baseline,
        cheapest=cheapest,
        most_accurate=most_accurate,
        fastest=fastest,
        no_model_meets_floor=no_floor,
        picker_rule=_PICKER_RULE,
        gate=spec.gate,
        recheck=(
            f"`wam recheck run --tasks {spec.id} --against results/runs/{SPEC_RUN_ID}` "
            "when a new model is released, when a model changes under the same id, or "
            "after a prompt change (that needs a new baseline run first)."
        ),
        limits=[*spec_meta.get("limits", ()), *reduction_limits, *_GENERIC_LIMITS],
        notes=[
            (
                f"Every number is read from `results/runs/{SPEC_RUN_ID}/cells.json` and "
                f"`results/baseline/{spec.id}.json`. No model call was made to build this card."
            )
        ],
    )
    return card.model_copy(update={"volumes": volume_rows(card)})


def _find_spec_run(task_id: str, runs_dir: Path) -> Path | None:
    """The committed run that holds this spec task's cells, most specific first."""
    candidates = [runs_dir / SPEC_RUN_ID]
    if runs_dir.is_dir():
        candidates.extend(
            path for path in sorted(runs_dir.iterdir()) if path.is_dir() and path not in candidates
        )
    for run_dir in candidates:
        cells_path = run_dir / "cells.json"
        if not cells_path.is_file():
            continue
        try:
            cells = load_spec_cells(cells_path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if any(cell.task == task_id for cell in cells):
            return run_dir
    return None


def _spec_card(task_id: str, runs_dir: Path, results_dir: Path) -> CardData:
    spec_path = _REPO_ROOT / _TASKS_DIR / f"{task_id}.toml"
    if not spec_path.is_file():
        msg = f"no task spec for {task_id!r} at {_TASKS_DIR / (task_id + '.toml')}"
        raise FileNotFoundError(msg)
    spec = load_spec(spec_path)
    run_dir = _find_spec_run(task_id, runs_dir)
    if run_dir is None:
        msg = f"no committed spec cells for {task_id!r} under {runs_dir}"
        raise FileNotFoundError(msg)
    cells = [cell for cell in load_spec_cells(run_dir / "cells.json") if cell.task == task_id]
    return build_spec_card(spec, cells, baseline=_spec_baseline(results_dir, task_id))


# ------------------------------------------------------------------- public API


def build_card(
    task: TaskRef,
    *,
    runs_dir: Path = Path("results/runs"),
    results_dir: Path = Path("results"),
) -> CardData:
    """Every number from committed files under ``results/``; no model call."""
    builtin = builtin_task(task)
    if builtin is not None:
        cells = load_cells(runs_dir / BUILTIN_RUN_ID / "cells.json")
        return _builtin_card(builtin, cells, results_dir, runs_dir=runs_dir)
    return _spec_card(str(task), runs_dir, results_dir)


def _render_cell(value: float, metric: str) -> str:
    if metric.startswith("n_") or metric == "n_valid":
        return f"{value:.0f}"
    return f"{value:.4f}"


def _render_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "—"
    return f"{ci[0]:.4f} to {ci[1]:.4f}"


def _render_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _render_latency(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _metric_headers(card: CardData, metric_order: Sequence[str]) -> list[str]:
    available = {name for row in card.models for name in row.metrics}
    primary = _primary_metric(card)
    return [metric for metric in metric_order if metric in available and metric != primary]


def _baseline_row(card: CardData, metrics: Sequence[str]) -> list[str]:
    row = card.baseline
    if row is None:
        return []
    cells = [
        f"| {row.model_id} (baseline) | {row.primary_metric:.4f} "
        f"(95% CI {_render_ci(row.primary_ci)}) |"
    ]
    cells.extend(f" {_render_cell(row.metrics.get(metric, 0.0), metric)} |" for metric in metrics)
    cells.append(f" {_render_cost(row.cost_per_1k)} | — |")
    return cells


def _markdown(card: CardData, *, metric_order: Sequence[str]) -> str:
    metrics = _metric_headers(card, metric_order)
    lines = [
        f"# {card.title}",
        "",
        f"**Task:** `{card.task}`",
        "",
        "## The job",
        "",
        card.job,
        "",
        "## The data",
        "",
        f"- Source: {card.data_source}.",
        f"- Licence: {card.licence}.",
        f"- Sample: {card.sample_size} items.",
        f"- Prompt version: `{card.prompt_version}`.",
        "",
        "## Results",
        "",
    ]
    header = (
        "| Model | Primary (95% CI) | "
        + " | ".join(f"`{name}`" for name in metrics)
        + " | Cost / 1k items (USD) | p50 latency |"
    )
    divider = "| --- | ---: | " + " | ".join("---:" for _ in metrics) + " | ---: | ---: |"
    lines.extend([header, divider])
    for row in card.models:
        cells = [
            f"| {row.model_id} | {row.primary_metric:.4f} (95% CI {_render_ci(row.primary_ci)}) |"
        ]
        cells.extend(f" {_render_cell(row.metrics.get(name, 0.0), name)} |" for name in metrics)
        cells.append(f" {_render_cost(row.cost_per_1k)} | {_render_latency(row.latency_p50)} |")
        lines.append("".join(cells))
    baseline = _baseline_row(card, metrics)
    if baseline:
        lines.append("".join(baseline))
    if card.baseline is None:
        lines.extend(["", "No non-AI baseline is defined for this task."])
    lines.extend(
        [
            "",
            f"Primary metric: `{_primary_metric(card)}`. "
            + (
                "Its interval is a 2,000-resample bootstrap."
                if _primary_metric(card) in {"macro_f1", "component_group.macro_f1"}
                else "Rate intervals are Wilson 95% intervals, as in `results/`."
            ),
            "",
            "## Recommendation",
            "",
        ]
    )
    if card.no_model_meets_floor:
        lines.extend(
            [
                "No model meets this task's gate floors yet. "
                + (
                    f"The most accurate model is `{card.most_accurate}`. "
                    if card.most_accurate is not None
                    else ""
                )
                + "Keep the current baseline and a human check; no cheapest pick is shown.",
                "",
            ]
        )
    else:
        for label, picked_model in (
            ("Cheapest good enough", card.cheapest),
            ("Most accurate", card.most_accurate),
            ("Fastest good enough", card.fastest),
        ):
            if picked_model is not None:
                lines.append(f"- {label}: `{picked_model}`.")
        lines.append("")
    lines.extend(
        [
            f"The rule that produced them: {card.picker_rule}.",
            "",
            "## Gate rule",
            "",
            f"- The primary metric may not drop by more than "
            f"{card.gate.primary_drop_points:g} points against the baseline run.",
            f"- The schema-valid rate must stay at or above {card.gate.schema_valid_floor:.0%}.",
        ]
    )
    for floor in card.gate.floors:
        bound = "at or above" if floor.min is not None else "within its drop bound"
        text = f"- `{floor.metric}` must stay {bound}"
        if floor.min is not None:
            text += f" {floor.min:.2f}"
        if floor.max_drop_points is not None:
            text += f" and may not drop by more than {floor.max_drop_points:g} points"
        lines.append(text + ".")
    lines.extend(
        [
            "",
            "## When to re-check",
            "",
            card.recheck,
            "",
            "## Cost at volume",
            "",
            "| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    by_model: dict[str, dict[int, float | None]] = {}
    for volume in card.volumes:
        by_model.setdefault(volume.model_id, {})[volume.items_per_month] = volume.cost_usd
    for card_row in card.models:
        values = by_model.get(card_row.model_id, {})
        lines.append(
            f"| {card_row.model_id} | "
            + " | ".join(_render_cost(values.get(items)) for items in (1_000, 10_000, 100_000))
            + " |"
        )
    lines.extend(
        [
            "",
            "Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. "
            "An unknown measured cost stays unknown.",
            "",
            "## Known limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in card.limits)
    if card.notes:
        lines.extend(["", "## Sources", ""])
        lines.extend(f"- {note}" for note in card.notes)
    lines.append("")
    return "\n".join(lines)


_PRIMARY_LABELS: dict[str, str] = {
    "T-ROUTE": "macro_f1",
    "T-SCOPE": "f1",
    "T-REPLY": "intent_correct",
    "T-DOCS": "hit_at_4",
}


def _primary_metric(card: CardData) -> str:
    builtin = builtin_task(card.task)
    if builtin is not None:
        return _PRIMARY_LABELS[builtin.value]
    spec_path = _REPO_ROOT / _TASKS_DIR / f"{card.task}.toml"
    if spec_path.is_file():
        return load_spec(spec_path).metrics.primary
    return "primary_metric"


def write_card(card: CardData, *, out_dir: Path = Path("results/cards")) -> tuple[Path, Path]:
    """``<task>.md`` and ``<task>.json``; both deterministic."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = task_key(card.task)
    markdown_path = out_dir / f"{key}.md"
    json_path = out_dir / f"{key}.json"
    metric_order = _metric_order(card)
    markdown_path.write_text(_markdown(card, metric_order=metric_order), encoding="utf-8")
    json_path.write_text(
        json.dumps(card.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return markdown_path, json_path


def _metric_order(card: CardData) -> tuple[str, ...]:
    builtin = builtin_task(card.task)
    if builtin is not None:
        return _BUILTIN[builtin].metric_order
    spec_path = _REPO_ROOT / _TASKS_DIR / f"{card.task}.toml"
    if spec_path.is_file():
        spec = load_spec(spec_path)
        return tuple(spec.metrics.report)
    return tuple(sorted({name for row in card.models for name in row.metrics}))


def write_prompt_library(
    card: CardData,
    spec_or_builtin: TaskSpec | TaskName,
    *,
    out_dir: Path = Path("results/prompt-library"),
) -> tuple[Path, Path]:
    """Write one card as its §A9 prompt-library entry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown, json_text = export_prompt_library(card, spec_or_builtin)
    key = task_key(card.task)
    markdown_path = out_dir / f"{key}.md"
    json_path = out_dir / f"{key}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    return markdown_path, json_path


def _available_spec_ids(runs_dir: Path) -> list[str]:
    task_ids: list[str] = []
    if not _TASKS_DIR.is_dir():
        return task_ids
    for path in sorted((_REPO_ROOT / _TASKS_DIR).glob("*.toml")):
        try:
            spec = load_spec(path)
        except ValueError:
            continue
        if _find_spec_run(spec.id, runs_dir) is not None:
            task_ids.append(spec.id)
    return task_ids


def build_cards(
    tasks: Sequence[TaskRef],
    *,
    runs_dir: Path = Path("results/runs"),
    results_dir: Path = Path("results"),
    out_dir: Path = Path("results/cards"),
) -> list[Path]:
    """Build and write the named cards; returns the markdown paths in order."""
    paths: list[Path] = []
    for task in tasks:
        card = build_card(task, runs_dir=runs_dir, results_dir=results_dir)
        markdown_path, _json_path = write_card(card, out_dir=out_dir)
        paths.append(markdown_path)
    return paths


def _write_prompt_library_for(
    task: TaskRef,
    *,
    runs_dir: Path,
    results_dir: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    card = build_card(task, runs_dir=runs_dir, results_dir=results_dir)
    builtin = builtin_task(task)
    spec_or_builtin: TaskSpec | TaskName
    if builtin is not None:
        spec_or_builtin = builtin
    else:
        spec_or_builtin = load_spec(_REPO_ROOT / _TASKS_DIR / f"{task_key(task)}.toml")
    return write_prompt_library(card, spec_or_builtin, out_dir=out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build use-case cards and the prompt library.")
    parser.add_argument("--cards", action="store_true", help="Write results/cards/.")
    parser.add_argument(
        "--prompt-library", action="store_true", help="Write results/prompt-library/."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    do_cards = args.cards or not args.prompt_library
    do_library = args.prompt_library or not args.cards

    tasks: list[str] = [task.value for task in TaskName]
    tasks.extend(_available_spec_ids(args.runs_dir))
    failures = 0
    for task in tasks:
        try:
            if do_cards:
                card = build_card(task, runs_dir=args.runs_dir, results_dir=args.results_dir)
                paths = write_card(card)
                print("card: " + ", ".join(str(path) for path in paths))  # noqa: T201
            if do_library:
                paths = _write_prompt_library_for(
                    task,
                    runs_dir=args.runs_dir,
                    results_dir=args.results_dir,
                    out_dir=Path("results/prompt-library"),
                )
                print("prompt-library: " + ", ".join(str(path) for path in paths))  # noqa: T201
        except FileNotFoundError as exc:
            failures += 1
            print(f"{task}: {exc}", file=sys.stderr)  # noqa: T201
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
