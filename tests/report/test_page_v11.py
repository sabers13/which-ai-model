"""The v1.1 section reads the committed T-DEFECT card, re-check demo and volumes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from which_ai_model.report.page import build_page
from which_ai_model.report.page_data import REPO_ROOT, TASK_ORDER, render_json


def _read(relative: str) -> Any:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_defect_table_is_the_committed_card(page: dict[str, Any]) -> None:
    card = _read("results/cards/defect.json")
    defect = page["v11"]["defect"]
    assert defect["job"] == card["job"]
    assert defect["sample"]["value"] == card["sample_size"]
    rows = defect["rows"]
    assert len(rows) == len(card["models"]) + 1
    for shown, committed in zip(rows, card["models"], strict=False):
        assert shown["model"] == committed["model_id"]
        assert shown["primary"]["value"] == round(committed["primary_metric"], 4)
        assert shown["coverage"]["value"] == round(
            committed["metrics"]["component_group.coverage"], 4
        )
        assert shown["answered"]["value"] == round(
            committed["metrics"]["component_group.accuracy_answered"], 4
        )
        assert shown["safety"]["value"] == round(committed["metrics"]["safety_relevant.recall"], 4)
        assert shown["cost"]["value"] == round(committed["cost_per_1k"], 4)
    baseline = rows[-1]
    assert baseline["baseline"] is True
    assert baseline["model"] == card["baseline"]["model_id"]
    assert baseline["cost"]["display"] == "$0"


def test_defect_rows_are_the_committed_cells_and_baseline(page: dict[str, Any]) -> None:
    cells = {
        cell["model"]["model_id"]: cell
        for cell in _read("results/runs/live-v1.1/cells.json")["spec_cells"]
    }
    rows = page["v11"]["defect"]["rows"]
    for row in rows[:-1]:
        cell = cells[row["model"]]
        assert row["primary"]["value"] == round(cell["primary_metric"], 4)
        assert row["coverage"]["value"] == round(cell["metrics"]["component_group.coverage"], 4)
        assert row["answered"]["value"] == round(
            cell["metrics"]["component_group.accuracy_answered"], 4
        )
        assert row["safety"]["value"] == round(cell["metrics"]["safety_relevant.recall"], 4)
        assert row["cost"]["value"] == round(cell["cost_per_1k"], 4)
        assert row["items"] == cell["n"]
    baseline = _read("results/baseline/defect.json")
    row = rows[-1]
    assert row["primary"]["value"] == round(baseline["component_group"]["macro_f1"], 4)
    assert row["coverage"]["value"] == round(baseline["component_group"]["coverage"], 4)
    assert row["safety"]["value"] == round(baseline["safety_relevant"]["recall"], 4)
    assert row["cost"]["value"] == 0.0


def test_safety_floor_and_coverage_are_stated_plainly(page: dict[str, Any]) -> None:
    card = _read("results/cards/defect.json")
    floor = page["v11"]["defect"]["floor"]
    assert floor["value"]["value"] == card["gate"]["floors"][0]["min"]
    assert floor["met"] is True
    assert floor["baseline_met"] is False
    assert "cleared" in floor["sentence"]
    assert "baseline" in floor["sentence"]
    coverage = page["v11"]["defect"]["coverage"]["sentence"]
    assert "Coverage ran from" in coverage
    assert "unclear" in coverage


def test_recheck_demo_is_the_committed_outcome(page: dict[str, Any]) -> None:
    refused = _read("results/runs/recheck-demo/recheck.json")
    accepted = _read("results/runs/recheck-demo-accepted/recheck.json")
    demo = page["v11"]["recheck"]["demo"]
    assert demo["refused"]["outcome"] == refused["gate"]["outcome"] == "REFUSED"
    assert demo["accepted"]["outcome"] == accepted["gate"]["outcome"] == "PASS"
    assert demo["accepted"]["warning_count"]["value"] == len(accepted["gate"]["warnings"])
    comparison = accepted["gate"]["pairs"][0]["comparisons"][0]
    assert demo["accepted"]["baseline"]["value"] == round(comparison["baseline_primary"], 4)
    assert demo["accepted"]["candidate"]["value"] == round(comparison["candidate_primary"], 4)
    assert demo["refused"]["differences"] == refused["settings"]["pairs"][0]["differences"]


def test_volume_rows_are_the_card_arithmetic(page: dict[str, Any]) -> None:
    card = _read("results/cards/defect.json")
    volume = page["v11"]["volume"]
    assert [header["items"] for header in volume["headers"]] == [1_000, 10_000, 100_000]
    assert [header["display"] for header in volume["headers"]] == ["1,000", "10,000", "100,000"]
    committed = {
        (row["model_id"], row["items_per_month"]): row["cost_usd"] for row in card["volumes"]
    }
    assert len(volume["rows"]) == len(card["models"])
    for row in volume["rows"]:
        assert len(row["volumes"]) == 3
        for entry in row["volumes"]:
            expected = committed[(row["model"], entry["items"])]
            assert entry["amount"]["value"] == round(expected, 4)
            assert entry["amount"]["display"].startswith("$")


def test_five_cards_in_task_order(page: dict[str, Any]) -> None:
    entries = page["v11"]["cards"]["entries"]
    assert [entry["task"] for entry in entries] == list(TASK_ORDER)
    for entry in entries:
        assert entry["job"]
        assert entry["picks"]
        assert (REPO_ROOT / entry["source"]).is_file()


def test_the_section_and_calculator_are_in_the_built_page(page: dict[str, Any]) -> None:
    built = build_page(REPO_ROOT, data=page)
    assert 'id="your-task"' in built
    assert 'href="#your-task"' in built
    assert 'id="volume-items"' in built
    assert 'id="volume-table"' in built
    assert built.count("data-cost=") == len(page["v11"]["volume"]["rows"])
    assert page["v11"]["volume"]["default_items"] == 10_000
    for row in page["v11"]["volume"]["rows"]:
        for entry in row["volumes"]:
            assert entry["amount"]["display"] in built


def test_v11_sources_exist_and_are_committed(page: dict[str, Any]) -> None:
    for relative in (
        "results/cards/defect.json",
        "results/runs/live-v1.1/cells.json",
        "results/runs/live-v1.1/manifest.json",
        "results/gate_demo_defect.json",
        "results/runs/recheck-demo/recheck.json",
        "results/runs/recheck-demo-accepted/recheck.json",
    ):
        assert Path(relative) in [Path(source) for source in page["sources"]]
    text = render_json(page)
    assert "TODO" not in text
    assert "TBD" not in text
