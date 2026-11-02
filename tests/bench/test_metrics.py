"""Metric correctness: hand-computed examples plus scikit-learn agreement."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from which_ai_model.bench.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_metric_ci,
    citation_precision,
    hit_at_k,
    latency_percentiles,
    macro_f1,
    precision_recall_f1,
    wilson_ci,
)


def test_accuracy_hand_computed() -> None:
    assert accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)
    assert accuracy(["a"], ["a"]) == 1.0
    with pytest.raises(ValueError, match="at least one"):
        accuracy([], [])
    with pytest.raises(ValueError, match="y_pred"):
        accuracy(["a"], [])


def test_macro_f1_matches_sklearn() -> None:
    y_true = ["a", "a", "b", "b", "c", "a"]
    y_pred = ["a", "b", "b", "b", "a", "a"]
    labels = sorted(set(y_true) | set(y_pred))
    expected = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_macro_f1_hand_computed_and_degenerate_label() -> None:
    assert macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx((2 / 3 + 0.8) / 2)
    # A label only in the predictions has precision 0 and contributes F1 0.
    assert macro_f1(["a", "a"], ["a", "z"]) == pytest.approx((2 / 3 + 0.0) / 2)


def test_precision_recall_f1_hand_computed() -> None:
    truth = ["pos", "pos", "neg", "neg"]
    predicted = ["pos", "neg", "pos", "neg"]
    precision, recall, f1 = precision_recall_f1(truth, predicted, "pos")
    assert (precision, recall, f1) == (pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.5))

    precision, recall, f1 = precision_recall_f1(["neg"], ["neg"], "pos")
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_wilson_ci_known_values_and_bounds() -> None:
    low, high = wilson_ci(8, 10)
    assert (low, high) == (pytest.approx(0.49016247), pytest.approx(0.94331785))
    assert wilson_ci(0, 10)[0] == 0.0
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n > 0"):
        wilson_ci(0, 0)
    with pytest.raises(ValueError, match="between"):
        wilson_ci(11, 10)


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    values = [0.5 + 0.1 * index for index in range(20)]
    first = bootstrap_ci(values)
    assert first == bootstrap_ci(values)
    mean = float(np.mean(values))
    assert first[0] <= mean <= first[1]
    assert first[1] - first[0] < 1.0
    assert bootstrap_ci(values, seed=7) != first

    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="positive"):
        bootstrap_ci([1.0], n_resamples=0)


def test_bootstrap_metric_ci_is_seeded() -> None:
    truth = ["a", "b", "a", "b"] * 5
    predicted = ["a", "b", "b", "b"] * 5
    first = bootstrap_metric_ci(truth, predicted)
    assert first == bootstrap_metric_ci(truth, predicted)
    low, high = first
    assert 0.0 <= low <= macro_f1(truth, predicted) <= high <= 1.0


def test_latency_percentiles_linear_interpolation() -> None:
    assert latency_percentiles(list(range(1, 101))) == (
        pytest.approx(50.5),
        pytest.approx(95.05),
    )
    assert latency_percentiles([7]) == (7.0, 7.0)
    with pytest.raises(ValueError, match="at least one"):
        latency_percentiles([])


def test_hit_at_k() -> None:
    retrieved = [["a", "b", "c"], ["c", "d", "e"], ["x", "y", "z"]]
    gold = [["b"], ["z"], ["x"]]
    assert hit_at_k(retrieved, gold, k=1) == pytest.approx(1 / 3)
    assert hit_at_k(retrieved, gold, k=2) == pytest.approx(2 / 3)
    assert hit_at_k(retrieved, gold, k=4) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="retrieval rows"):
        hit_at_k(retrieved, gold[:2])
    with pytest.raises(ValueError, match="k must be positive"):
        hit_at_k(retrieved, gold, k=0)


def test_citation_precision_is_subset_indicator() -> None:
    assert citation_precision(["p1", "p2"], ["p1", "p2", "p3"]) == 1.0
    assert citation_precision(["p1", "p9"], ["p1", "p2"]) == 0.0
    assert citation_precision([], ["p1"]) == 0.0
    assert citation_precision(["p1"], []) == 0.0
