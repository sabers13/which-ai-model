"""Page data must copy its measurements from committed evidence, with sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from which_ai_model.report.page_data import (
    BUILTIN_TASK_ORDER,
    REPO_ROOT,
    TASK_ORDER,
    build_page_data,
    render_json,
)


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_headline_numbers_match_the_committed_results(page: dict[str, Any]) -> None:
    cards = {card["id"]: card for card in page["numbers"]["cards"]}
    assert cards["tools-moved"]["value"]["value"] == 5
    assert cards["tools-moved"]["value"]["source"] == "results/inventory/before.json"
    assert cards["messages-per-model"]["value"]["value"] == 1106
    assert cards["messages-per-model"]["value"]["display"] == "1,106"
    assert cards["models"]["value"]["value"] == 7
    assert cards["spend"]["value"]["display"] == "$1.94"
    assert cards["gate-blocked"]["value"]["display"] == "2"

    baseline = page["numbers"]["baseline"]
    assert baseline["route"]["display"] == "85.9%"
    assert baseline["scope"]["display"] == "68.4%"
    assert page["numbers"]["best_route"]["display"] == "88.9%"
    assert page["numbers"]["best_scope"]["display"] == "73.1%"


def test_every_measurement_names_an_existing_file(page: dict[str, Any]) -> None:
    measured = 0
    for node in _walk(page):
        if "value" not in node or "source" not in node:
            continue
        measured += 1
        source = Path(str(node["source"]))
        assert (REPO_ROOT / source).is_file(), f"missing source {source}"
        assert "display" in node
        assert str(node["source"]).startswith(("results/",)), node
    assert measured > 60


def test_task_picks_and_full_tables(page: dict[str, Any]) -> None:
    assert TASK_ORDER[-1] == "T-DEFECT"
    assert [task["id"] for task in page["tasks"]] == list(BUILTIN_TASK_ORDER)
    for task in page["tasks"]:
        assert set(task["picks"]) == {"cheapest", "most_accurate", "fastest"}
        assert len(task["rows"]) == 7
        for priority in task["picks"].values():
            assert priority["metric"]["source"] == "results/recommendations.json"
            assert priority["cost"]["source"] == "results/recommendations.json"
            assert priority["latency"]["source"] == "results/recommendations.json"


def test_example_is_from_the_committed_sample(page: dict[str, Any]) -> None:
    example = page["example"]
    sample = (REPO_ROOT / example["source"]).read_text(encoding="utf-8").splitlines()
    assert example["item_id"] in sample
    assert example["total"] == 7
    assert example["correct"] == 4
    assert example["true_label"] == "card_payment_fee_charged"
    for row in example["rows"]:
        assert row["latency"]["source"] == "results/runs/live/calls.jsonl"
        assert row["cost"]["source"] == "results/runs/live/calls.jsonl"
        assert row["correct"] == (row["answer"] == example["true_label"])


def test_gate_reasons_come_from_the_gate_file(page: dict[str, Any]) -> None:
    gate_file = json.loads((REPO_ROOT / "results/gate_demo.json").read_text(encoding="utf-8"))
    saved = [reason for pair in gate_file["pairs"] for reason in pair["reasons"]]
    shown = [reason for pair in page["gate"]["pairs"] for reason in pair["reasons"]]
    assert shown == saved
    assert page["gate"]["refusal"]["outcome"] == "REFUSED"


def test_docs_examples_have_answers_and_cited_passages(page: dict[str, Any]) -> None:
    assert len(page["docs"]["examples"]) == 3
    for item in page["docs"]["examples"]:
        assert item["question"].endswith("?")
        assert item["answer"]
        assert item["passages"]
        for passage in item["passages"]:
            assert (REPO_ROOT / passage["source"]).is_file()


def test_line_counts_match_the_source_inventory(page: dict[str, Any]) -> None:
    for row in page["before_after"]["line_counts"]:
        report = (REPO_ROOT / row["source"]).read_text(encoding="utf-8")
        assert str(row["before"]["value"]) in report
        assert str(row["after"]["value"]) in report


def test_page_data_is_deterministic() -> None:
    first = build_page_data()
    second = build_page_data()
    assert render_json(first) == render_json(second)
    assert json.loads(render_json(first)) == first


def test_page_data_has_no_clock_or_timestamp(page: dict[str, Any]) -> None:
    text = render_json(page)
    assert "generated_at" not in text
    assert "built_at" not in text


def test_all_sources_exist(page: dict[str, Any]) -> None:
    for source in page["sources"]:
        assert (REPO_ROOT / source).is_file(), source


def test_content_has_no_placeholders(page: dict[str, Any]) -> None:
    text = render_json(page)
    assert "TODO" not in text
    assert "TBD" not in text
    assert "lorem ipsum" not in text.lower()
