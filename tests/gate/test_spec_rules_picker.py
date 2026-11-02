"""The §A4 picker for spec tasks: floors first, then the 3.0-point band."""

from __future__ import annotations

from dataclasses import dataclass

from which_ai_model.contracts import GateFloor
from which_ai_model.gate import Pick, pick

RECALL_FLOOR = GateFloor(metric="safety_relevant.recall", min=0.90)


@dataclass(frozen=True)
class _Cell:
    model_id: str
    primary_metric: float
    metrics: dict[str, float]
    cost_per_1k: float | None = 1.0
    latency_p50: float | None = 10.0


def _cell(
    model_id: str,
    primary: float,
    *,
    recall: float = 0.95,
    cost: float | None = 1.0,
    latency: float | None = 10.0,
) -> _Cell:
    return _Cell(
        model_id=model_id,
        primary_metric=primary,
        metrics={"safety_relevant.recall": recall},
        cost_per_1k=cost,
        latency_p50=latency,
    )


def test_pick_keeps_models_within_three_points_as_good_enough() -> None:
    result = pick(
        [
            _cell("best", 0.80, cost=2.0),
            _cell("close", 0.78, cost=0.5),
            _cell("far", 0.76, cost=0.1),
        ]
    )
    assert isinstance(result, Pick)
    assert result.good_enough == ("best", "close")
    assert result.cheapest == "close"
    assert result.most_accurate == "best"
    assert result.no_model_meets_floor is False


def test_pick_rejects_a_model_below_a_floor() -> None:
    result = pick(
        [
            _cell("accurate-but-unsafe", 0.90, recall=0.50, cost=0.1),
            _cell("safe", 0.80, recall=0.95, cost=1.0),
        ],
        floors=[RECALL_FLOOR],
    )
    assert result.good_enough == ("safe",)
    assert result.cheapest == "safe"
    assert result.most_accurate == "safe"
    assert result.no_model_meets_floor is False


def test_pick_says_plainly_when_no_model_meets_a_floor() -> None:
    result = pick(
        [
            _cell("a", 0.90, recall=0.80),
            _cell("b", 0.85, recall=0.89),
        ],
        floors=[RECALL_FLOOR],
    )
    assert result == Pick(
        cheapest=None,
        most_accurate="a",
        fastest=None,
        good_enough=(),
        no_model_meets_floor=True,
    )


def test_pick_treats_a_missing_floor_metric_as_not_good_enough() -> None:
    no_metric = _Cell(model_id="no-metric", primary_metric=0.90, metrics={})
    result = pick([no_metric, _cell("safe", 0.80)], floors=[RECALL_FLOOR])
    assert result.good_enough == ("safe",)
    assert result.no_model_meets_floor is False


def test_pick_ignores_a_drop_only_floor() -> None:
    """``max_drop_points`` is baseline-relative; only the gate can enforce it."""
    drop_only = GateFloor(metric="safety_relevant.recall", max_drop_points=2.0)
    result = pick([_cell("a", 0.90, recall=0.10)], floors=[drop_only])
    assert result.good_enough == ("a",)
    assert result.most_accurate == "a"


def test_pick_prefers_the_cheaper_model_on_a_tie() -> None:
    result = pick([_cell("pricey", 0.80, cost=2.0), _cell("cheap", 0.80, cost=0.5)])
    assert result.good_enough == ("cheap", "pricey")
    assert result.cheapest == "cheap"
    assert result.most_accurate == "cheap"


def test_pick_puts_an_unknown_cost_last() -> None:
    result = pick([_cell("unknown", 0.80, cost=None), _cell("known", 0.80, cost=1.0)])
    assert result.cheapest == "known"
    assert result.good_enough == ("known", "unknown")


def test_pick_fastest_ignores_models_without_a_latency() -> None:
    result = pick(
        [
            _cell("slow", 0.80, latency=50.0),
            _cell("fast", 0.79, latency=10.0),
            _cell("unmeasured", 0.78, latency=None),
        ]
    )
    assert result.fastest == "fast"


def test_pick_has_nothing_to_pick_without_cells() -> None:
    assert pick([]) == Pick(
        cheapest=None,
        most_accurate=None,
        fastest=None,
        good_enough=(),
        no_model_meets_floor=False,
    )
