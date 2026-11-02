"""Use-case cards (``PLAN-v1.1`` §A7) from committed v1 results and spec cells."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from which_ai_model.bench.score import load_cells
from which_ai_model.contracts import ModelRef, TaskName
from which_ai_model.report.cards import (
    build_card,
    build_cards,
    build_spec_card,
    write_card,
)
from which_ai_model.report.volume import volume_rows
from which_ai_model.task import load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "results" / "runs"
RESULTS = REPO_ROOT / "results"
SPEC = REPO_ROOT / "tasks" / "defect.toml"


def _committed_cells():
    return load_cells(RUNS / "live" / "cells.json")


def _spec_cell(model_id: str, *, recall: float, primary: float = 0.70):
    from which_ai_model.bench.score import SpecCellResult

    return SpecCellResult(
        task="defect",
        model=ModelRef(connection_id="fixture-openai", model_id=model_id),
        model_id=model_id,
        n=20,
        primary_metric=primary,
        metrics={
            "component_group.macro_f1": primary,
            "component_group.accuracy": primary,
            "component_group.coverage": 1.0,
            "component_group.accuracy_answered": primary,
            "safety_relevant.recall": recall,
            "safety_relevant.precision": 0.80,
            "schema_valid_rate": 1.0,
        },
        ci={
            "component_group.macro_f1_low": primary - 0.05,
            "component_group.macro_f1_high": primary + 0.05,
        },
        latency_p50=10.0,
        latency_p95=20.0,
        cost_per_1k=0.10,
    )


@pytest.mark.parametrize("task", list(TaskName))
def test_a_v1_card_reads_its_number_from_the_committed_cells(task: TaskName) -> None:
    card = build_card(task.value, runs_dir=RUNS, results_dir=RESULTS)
    cells = [cell for cell in _committed_cells() if cell.task is task]
    assert card.task == task.value
    assert card.prompt_version == "v1"
    assert card.sample_size == cells[0].n
    assert len(card.models) == len(cells) == 7
    assert card.job
    assert card.title
    assert card.data_source
    assert card.licence
    assert "recheck run --tasks" in card.recheck
    assert card.gate.primary_drop_points == 2.0
    assert card.gate.schema_valid_floor == 0.98
    assert len(card.volumes) == 21
    assert card.notes
    # Every row's primary is a committed cell, not a fresh measurement.
    expected = {(cell.model.model_id, cell.primary_metric) for cell in cells}
    assert {(row.model_id, row.primary_metric) for row in card.models} == expected


def test_baseline_rows_come_from_the_committed_baseline_files() -> None:
    route = build_card("T-ROUTE", runs_dir=RUNS, results_dir=RESULTS)
    scope = build_card("T-SCOPE", runs_dir=RUNS, results_dir=RESULTS)
    assert route.baseline is not None
    assert scope.baseline is not None
    assert route.baseline.is_baseline is True
    assert route.baseline.cost_per_1k == 0.0
    assert route.baseline.latency_p50 is None
    assert build_card("T-REPLY", runs_dir=RUNS, results_dir=RESULTS).baseline is None
    assert build_card("T-DOCS", runs_dir=RUNS, results_dir=RESULTS).baseline is None


def test_the_card_recommendation_is_the_committed_v1_recommendation() -> None:
    recommendations = json.loads((RESULTS / "recommendations.json").read_text(encoding="utf-8"))[
        "tasks"
    ]
    for task in TaskName:
        card = build_card(task.value, runs_dir=RUNS, results_dir=RESULTS)
        entry = recommendations[task.value]
        for label in ("cheapest", "most_accurate", "fastest"):
            committed = entry[label]
            expected = committed["model"] if committed else None
            assert getattr(card, label) == expected


def test_markdown_has_the_a7_sections_and_the_volume_arithmetic(tmp_path: Path) -> None:
    card = build_card("T-SCOPE", runs_dir=RUNS, results_dir=RESULTS)
    markdown_path, json_path = write_card(card, out_dir=tmp_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    for heading in (
        "## The job",
        "## The data",
        "## Results",
        "## Recommendation",
        "## Gate rule",
        "## When to re-check",
        "## Cost at volume",
        "## Known limits",
        "## Sources",
    ):
        assert heading in markdown
    assert "tfidf-logreg (baseline)" in markdown
    assert "CC-BY-3.0" in markdown
    assert "300 items" in markdown
    # A row per model plus the baseline, each with the same number of columns as the header.
    results_block = markdown.split("## Recommendation", 1)[0]
    table = [line for line in results_block.splitlines() if line.startswith("| ")]
    header_columns = table[0].count("|")
    assert all(line.count("|") == header_columns for line in table)
    first = card.models[0]
    assert first.cost_per_1k is not None
    assert f"${first.cost_per_1k * 10_000 / 1_000:.2f}" in markdown

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["task"] == "T-SCOPE"
    assert len(saved["volumes"]) == len(volume_rows(card))
    assert saved["baseline"]["is_baseline"] is True


def test_writing_a_card_twice_is_byte_identical(tmp_path: Path) -> None:
    card = build_card("T-ROUTE", runs_dir=RUNS, results_dir=RESULTS)
    first_md, first_json = write_card(card, out_dir=tmp_path / "a")
    second_md, second_json = write_card(card, out_dir=tmp_path / "b")
    assert first_md.read_bytes() == second_md.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()


def test_build_cards_writes_the_four_v1_cards(tmp_path: Path) -> None:
    tasks = [task.value for task in TaskName]
    paths = build_cards(tasks, runs_dir=RUNS, results_dir=RESULTS, out_dir=tmp_path)
    assert [path.name for path in paths] == [f"{task}.md" for task in tasks]
    for task in tasks:
        assert (tmp_path / f"{task}.md").is_file()
        payload = json.loads((tmp_path / f"{task}.json").read_text(encoding="utf-8"))
        assert payload["task"] == task
        assert payload["models"]


def test_a_spec_card_states_no_model_meets_the_floor(tmp_path: Path) -> None:
    spec = load_spec(SPEC)
    cells = [_spec_cell("weak-one", recall=0.40), _spec_cell("weak-two", recall=0.45, primary=0.72)]
    card = build_spec_card(spec, cells)

    assert card.no_model_meets_floor is True
    assert card.cheapest is None
    assert card.most_accurate == "weak-two"
    assert card.fastest is None
    assert card.sample_size == 20
    assert card.gate.floors[0].metric == "safety_relevant.recall"
    markdown_path, _ = write_card(card, out_dir=tmp_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "No model meets this task's gate floors yet." in markdown
    assert "The most accurate model is `weak-two`." in markdown
    assert "`safety_relevant.recall` must stay at or above 0.90" in markdown
    assert "No non-AI baseline is defined for this task." in markdown


def test_a_spec_card_picks_only_models_above_the_floor() -> None:
    spec = load_spec(SPEC)
    cells = [
        _spec_cell("below-floor", recall=0.50, primary=0.90),
        _spec_cell("above-floor", recall=0.95, primary=0.70),
    ]
    card = build_spec_card(spec, cells)
    assert card.no_model_meets_floor is False
    assert card.most_accurate == "above-floor"
    assert card.cheapest == "above-floor"


def test_a_spec_card_names_a_cell_scored_on_fewer_items() -> None:
    """§A10 may cut one cell to 400 items; the card says so instead of implying 600."""
    spec = load_spec(SPEC)
    cells = [
        _spec_cell("full-cell", recall=0.95),
        _spec_cell("cut-cell", recall=0.95, primary=0.72),
    ]
    cells[1] = dataclasses.replace(cells[1], n=15)
    card = build_spec_card(spec, cells)

    assert card.sample_size == 20
    assert any("cut-cell on 15 of 20 items" in limit for limit in card.limits)
