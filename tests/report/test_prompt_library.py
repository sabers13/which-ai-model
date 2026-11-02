"""Prompt-library export (``PLAN-v1.1`` §A9): one card becomes one ready entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from which_ai_model.contracts import (
    CardData,
    CardModelRow,
    GateFloor,
    GateSpec,
    PromptLibraryEntry,
    TaskName,
)
from which_ai_model.report.prompt_library import export_prompt_library
from which_ai_model.task import load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "tests" / "fixtures" / "task" / "example.toml"


def _card(
    models: list[CardModelRow],
    *,
    task: str = "T-ROUTE",
    floors: list[GateFloor] | None = None,
    no_model_meets_floor: bool = False,
    prompt_version: str = "v1",
) -> CardData:
    return CardData(
        task=task,
        title="Route a message",
        job="Send each support message to the right queue.",
        data_source="Banking77",
        licence="CC-BY-4.0",
        sample_size=616,
        prompt_version=prompt_version,
        models=models,
        cheapest=next((model.model_id for model in models), None),
        no_model_meets_floor=no_model_meets_floor,
        gate=GateSpec(floors=floors or []),
        recheck="wam recheck run --tasks T-ROUTE --against results/runs/live",
    )


def _entry(markdown: str, json_text: str) -> PromptLibraryEntry:
    return PromptLibraryEntry.model_validate(json.loads(json_text))


def test_builtin_export_has_the_a9_fields() -> None:
    card = _card(
        [
            CardModelRow(model_id="gpt-6-luna", primary_metric=0.80, cost_per_1k=0.15),
            CardModelRow(model_id="claude-haiku-5-5", primary_metric=0.81, cost_per_1k=0.25),
            CardModelRow(model_id="deepseek-v4.1-flash", primary_metric=0.60, cost_per_1k=0.05),
        ]
    )
    markdown, json_text = export_prompt_library(card, TaskName.ROUTE)
    entry = _entry(markdown, json_text)

    assert entry.task == "T-ROUTE"
    assert entry.title == card.title
    assert entry.when_to_use == card.job
    assert entry.prompt_version == "v1"
    assert entry.recommended_model == "gpt-6-luna"
    assert entry.fallback_model == "claude-haiku-5-5"
    assert entry.recheck_trigger == card.recheck
    assert entry.guardrails
    assert "intent" in entry.output_schema["properties"]
    assert "Banking77" in entry.prompt
    assert "{text}" in entry.prompt


def test_markdown_states_the_recommendation_prompt_and_guardrails() -> None:
    card = _card(
        [
            CardModelRow(model_id="gpt-6-luna", primary_metric=0.80, cost_per_1k=0.15),
            CardModelRow(model_id="claude-haiku-5-5", primary_metric=0.81, cost_per_1k=0.25),
        ]
    )
    markdown, _ = export_prompt_library(card, "T-ROUTE")

    assert markdown.startswith("# Route a message")
    assert "## Task and when to use" in markdown
    assert "**Recommended:** `gpt-6-luna` — $0.15 per 1,000 items." in markdown
    assert "**Fallback:** `claude-haiku-5-5` — $0.25 per 1,000 items." in markdown
    assert "## Prompt (version `v1`)" in markdown
    assert "## Output schema" in markdown
    assert "## Guardrails" in markdown
    assert "unclear" in markdown
    assert "Never auto-close" in markdown
    assert f"## Re-check\n\n{card.recheck}" in markdown


def test_the_fallback_is_the_next_good_enough_model_by_cost() -> None:
    card = _card(
        [
            CardModelRow(model_id="a-cheap", primary_metric=0.80, cost_per_1k=0.10),
            CardModelRow(model_id="b-cheap", primary_metric=0.80, cost_per_1k=0.10),
            CardModelRow(model_id="c-dear", primary_metric=0.80, cost_per_1k=0.20),
        ]
    )
    entry = _entry(*export_prompt_library(card, TaskName.ROUTE))
    assert entry.recommended_model == "a-cheap"
    assert entry.fallback_model == "b-cheap"


def test_a_model_below_a_gate_floor_is_never_picked() -> None:
    floor = GateFloor(metric="urgent.recall", min=0.90)
    card = _card(
        [
            CardModelRow(
                model_id="cheap-but-weak",
                primary_metric=0.78,
                metrics={"urgent.recall": 0.50},
                cost_per_1k=0.01,
            ),
            CardModelRow(
                model_id="good-model",
                primary_metric=0.80,
                metrics={"urgent.recall": 0.95},
                cost_per_1k=0.30,
            ),
        ],
        floors=[floor],
    )
    markdown, json_text = export_prompt_library(card, TaskName.ROUTE)
    entry = _entry(markdown, json_text)
    assert entry.recommended_model == "good-model"
    assert entry.fallback_model is None
    assert "cheap-but-weak" not in markdown
    assert "`urgent.recall` must stay at or above 0.90" in entry.guardrails[1]


def test_no_model_meets_the_floor_is_stated_plainly() -> None:
    floor = GateFloor(metric="urgent.recall", min=0.90)
    card = _card(
        [
            CardModelRow(
                model_id="weak-one",
                primary_metric=0.70,
                metrics={"urgent.recall": 0.40},
                cost_per_1k=0.01,
            ),
            CardModelRow(
                model_id="weak-two",
                primary_metric=0.72,
                metrics={"urgent.recall": 0.45},
                cost_per_1k=0.02,
            ),
        ],
        floors=[floor],
        no_model_meets_floor=True,
    )
    markdown, json_text = export_prompt_library(card, TaskName.ROUTE)
    entry = _entry(markdown, json_text)
    assert entry.recommended_model is None
    assert entry.fallback_model is None
    assert "No model meets this task's gate floors yet." in markdown


def test_a_floor_that_no_model_meets_is_stated_even_without_the_flag() -> None:
    floor = GateFloor(metric="urgent.recall", min=0.90)
    card = _card(
        [
            CardModelRow(
                model_id="weak",
                primary_metric=0.70,
                metrics={"urgent.recall": 0.40},
                cost_per_1k=0.01,
            )
        ],
        floors=[floor],
    )
    markdown, json_text = export_prompt_library(card, TaskName.ROUTE)
    entry = _entry(markdown, json_text)
    assert entry.recommended_model is None
    assert entry.fallback_model is None
    assert "No model meets this task's gate floors yet." in markdown


def test_an_unknown_cost_is_shown_as_unknown_never_as_zero() -> None:
    card = _card(
        [
            CardModelRow(model_id="unknown-cost", primary_metric=0.80, cost_per_1k=None),
            CardModelRow(model_id="known-cost", primary_metric=0.80, cost_per_1k=0.20),
        ]
    )
    markdown, _ = export_prompt_library(card, TaskName.ROUTE)
    assert "**Recommended:** `known-cost` — $0.20 per 1,000 items." in markdown
    assert "**Fallback:** `unknown-cost` — cost per 1,000 items: unknown." in markdown
    assert "$0.00" not in markdown


def test_tied_costs_prefer_the_better_primary_metric_then_the_model_id() -> None:
    card = _card(
        [
            CardModelRow(model_id="a-lower", primary_metric=0.80, cost_per_1k=0.10),
            CardModelRow(model_id="b-higher", primary_metric=0.81, cost_per_1k=0.10),
        ]
    )
    entry = _entry(*export_prompt_library(card, TaskName.ROUTE))
    assert entry.recommended_model == "b-higher"
    assert entry.fallback_model == "a-lower"


def test_only_one_good_enough_model_has_no_fallback() -> None:
    card = _card([CardModelRow(model_id="solo", primary_metric=0.80, cost_per_1k=0.10)])
    markdown, json_text = export_prompt_library(card, TaskName.ROUTE)
    entry = _entry(markdown, json_text)
    assert entry.recommended_model == "solo"
    assert entry.fallback_model is None
    assert "only one model is good enough" in markdown


def test_export_is_byte_identical_twice() -> None:
    card = _card(
        [
            CardModelRow(model_id="gpt-6-luna", primary_metric=0.80, cost_per_1k=0.15),
            CardModelRow(model_id="claude-haiku-5-5", primary_metric=0.81, cost_per_1k=0.25),
        ]
    )
    assert export_prompt_library(card, TaskName.ROUTE) == export_prompt_library(
        card, TaskName.ROUTE
    )


def test_spec_export_uses_the_committed_template_and_output_model() -> None:
    spec = load_spec(SPEC_PATH)
    card = _card(
        [CardModelRow(model_id="gpt-6-luna", primary_metric=0.80, cost_per_1k=0.15)],
        task=spec.id,
        prompt_version=spec.prompt_version,
    )
    markdown, json_text = export_prompt_library(card, spec)
    entry = _entry(markdown, json_text)

    template = (REPO_ROOT / spec.prompt.template_file).read_text(encoding="utf-8")
    assert entry.prompt == template
    assert "{labels}" in entry.prompt
    assert "{text}" in entry.prompt
    assert entry.prompt_version == "e1"
    queue = entry.output_schema["properties"]["queue"]
    assert queue["enum"] == ["billing", "technical", "account", "unclear"]
    assert entry.output_schema["properties"]["urgent"]["type"] == "boolean"
    assert "## Task and when to use" in markdown


def test_a_card_for_another_task_is_refused() -> None:
    card = _card([CardModelRow(model_id="m", primary_metric=0.80, cost_per_1k=0.1)])
    with pytest.raises(ValueError, match="does not match"):
        export_prompt_library(card, load_spec(SPEC_PATH))
