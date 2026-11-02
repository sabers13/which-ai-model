"""Cost at volume (``PLAN-v1.1`` §A8): linear, and unknown is never zero."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from which_ai_model.contracts import CardData, CardModelRow
from which_ai_model.report.volume import DEFAULT_VOLUMES, volume_rows


def _card(models: list[CardModelRow], *, baseline: CardModelRow | None = None) -> CardData:
    return CardData(
        task="T-ROUTE",
        title="Route a message",
        job="Send each support message to the right queue.",
        data_source="Banking77",
        licence="CC-BY-4.0",
        sample_size=616,
        prompt_version="v1",
        models=models,
        baseline=baseline,
        recheck="wam recheck run --tasks T-ROUTE --against results/runs/live",
    )


def test_rows_are_model_major_with_the_default_volumes() -> None:
    card = _card(
        [
            CardModelRow(model_id="gpt-6-luna", primary_metric=0.8, cost_per_1k=0.5),
            CardModelRow(model_id="claude-haiku-5-5", primary_metric=0.81, cost_per_1k=0.25),
        ]
    )
    rows = volume_rows(card)
    assert DEFAULT_VOLUMES == (1_000, 10_000, 100_000)
    assert [(row.model_id, row.items_per_month) for row in rows] == [
        ("gpt-6-luna", 1_000),
        ("gpt-6-luna", 10_000),
        ("gpt-6-luna", 100_000),
        ("claude-haiku-5-5", 1_000),
        ("claude-haiku-5-5", 10_000),
        ("claude-haiku-5-5", 100_000),
    ]


def test_cost_is_the_linear_product() -> None:
    card = _card([CardModelRow(model_id="m", primary_metric=0.5, cost_per_1k=0.5)])
    assert [row.cost_usd for row in volume_rows(card)] == pytest.approx([0.5, 5.0, 50.0])


def test_an_unknown_cost_stays_unknown_at_every_volume() -> None:
    card = _card(
        [
            CardModelRow(model_id="known", primary_metric=0.5, cost_per_1k=0.25),
            CardModelRow(model_id="unknown", primary_metric=0.6, cost_per_1k=None),
        ]
    )
    rows = volume_rows(card)
    unknown = [row for row in rows if row.model_id == "unknown"]
    assert len(unknown) == 3
    assert all(row.cost_usd is None for row in unknown)
    assert all(row.cost_usd != 0 for row in unknown)
    known = [row for row in rows if row.model_id == "known"]
    assert [row.cost_usd for row in known] == pytest.approx([0.25, 2.5, 25.0])


def test_a_measured_zero_cost_stays_zero() -> None:
    card = _card([CardModelRow(model_id="tfidf_logreg", primary_metric=0.5, cost_per_1k=0.0)])
    rows = volume_rows(card)
    assert [row.cost_usd for row in rows] == [0.0, 0.0, 0.0]


def test_custom_volumes_are_used_in_the_order_given() -> None:
    card = _card([CardModelRow(model_id="m", primary_metric=0.5, cost_per_1k=0.4)])
    rows = volume_rows(card, volumes=(2_000, 250))
    assert [row.items_per_month for row in rows] == [2_000, 250]
    assert [row.cost_usd for row in rows] == pytest.approx([0.8, 0.1])


def test_no_models_gives_no_rows() -> None:
    assert volume_rows(_card([])) == []


def test_the_baseline_is_not_duplicated() -> None:
    baseline = CardModelRow(
        model_id="tfidf_logreg", primary_metric=0.55, cost_per_1k=0.0, is_baseline=True
    )
    card = _card(
        [CardModelRow(model_id="gpt-6-luna", primary_metric=0.8, cost_per_1k=0.5)],
        baseline=baseline,
    )
    rows = volume_rows(card)
    assert {row.model_id for row in rows} == {"gpt-6-luna"}


def test_tied_costs_keep_both_models_in_card_order() -> None:
    card = _card(
        [
            CardModelRow(model_id="b-model", primary_metric=0.8, cost_per_1k=0.2),
            CardModelRow(model_id="a-model", primary_metric=0.8, cost_per_1k=0.2),
        ]
    )
    rows = volume_rows(card, volumes=(1_000,))
    assert [(row.model_id, row.cost_usd) for row in rows] == [
        ("b-model", pytest.approx(0.2)),
        ("a-model", pytest.approx(0.2)),
    ]


def test_a_negative_volume_is_rejected_by_the_contract() -> None:
    card = _card([CardModelRow(model_id="m", primary_metric=0.5, cost_per_1k=0.2)])
    with pytest.raises(ValidationError):
        volume_rows(card, volumes=(-1,))
