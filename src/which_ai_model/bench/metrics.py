"""Deterministic metrics for the benchmark (``PLAN.md`` §7).

Every function is pure: same inputs, same output, no clock, no network. Bootstrap
resampling uses ``numpy.random.default_rng`` with an explicit seed, so a replayed result
is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_SEED",
    "Z_95",
    "accuracy",
    "bootstrap_ci",
    "bootstrap_metric_ci",
    "citation_precision",
    "hit_at_k",
    "latency_percentiles",
    "macro_f1",
    "precision_recall_f1",
    "wilson_ci",
]

DEFAULT_SEED = 20261028
Z_95 = 1.959963984540054


def _check_pairs(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        msg = f"y_true has {len(y_true)} items but y_pred has {len(y_pred)}"
        raise ValueError(msg)
    if not y_true:
        msg = "metrics need at least one item"
        raise ValueError(msg)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Share of exactly matching labels."""
    _check_pairs(y_true, y_pred)
    correct = sum(1 for truth, predicted in zip(y_true, y_pred, strict=True) if truth == predicted)
    return correct / len(y_true)


def _per_class_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label: str,
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for truth, predicted in zip(y_true, y_pred, strict=True):
        if truth == label and predicted == label:
            true_positive += 1
        elif truth != label and predicted == label:
            false_positive += 1
        elif truth == label and predicted != label:
            false_negative += 1
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted mean F1 over every label that occurs in truth or prediction.

    Matches ``sklearn.metrics.f1_score(..., average="macro")`` for those labels, with a
    zero F1 when a class has no true or predicted positive.
    """
    _check_pairs(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    scores = [_f1(*_per_class_counts(y_true, y_pred, label)) for label in labels]
    return sum(scores) / len(scores)


def precision_recall_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive: str,
) -> tuple[float, float, float]:
    """Precision, recall and F1 for ``positive`` as the positive class."""
    _check_pairs(y_true, y_pred)
    true_positive, false_positive, false_negative = _per_class_counts(y_true, y_pred, positive)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def wilson_ci(
    successes: int,
    n: int,
    *,
    z: float = Z_95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        msg = "Wilson interval needs n > 0"
        raise ValueError(msg)
    if successes < 0 or successes > n:
        msg = f"successes must be between 0 and n={n}, got {successes}"
        raise ValueError(msg)
    proportion = successes / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean of ``values``.

    Pass per-item scores (0/1 correctness, latency, …); the resample size equals the
    number of values, as in the plan's 2,000-resample bootstrap.
    """
    if not values:
        msg = "bootstrap needs at least one value"
        raise ValueError(msg)
    if n_resamples <= 0:
        msg = "n_resamples must be positive"
        raise ValueError(msg)
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    means = sample[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_metric_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    metric: Callable[[Sequence[str], Sequence[str]], float] = macro_f1,
    n_resamples: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI of a paired metric such as ``macro_f1``.

    Each resample draws item indices with replacement and recomputes the metric on the
    paired truth/prediction lists, which is what ``PLAN.md`` §7 asks for.
    """
    _check_pairs(y_true, y_pred)
    if n_resamples <= 0:
        msg = "n_resamples must be positive"
        raise ValueError(msg)

    if metric is macro_f1:
        return _bootstrap_macro_f1(y_true, y_pred, n_resamples=n_resamples, seed=seed)

    rng = np.random.default_rng(seed)
    size = len(y_true)
    scores = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        positions = rng.integers(0, size, size=size)
        sampled_true = [y_true[position] for position in positions]
        sampled_pred = [y_pred[position] for position in positions]
        scores[index] = metric(sampled_true, sampled_pred)
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Vectorised macro-F1 bootstrap over the full label set (fast path).

    Labels stay fixed at the union of the observed truth and predictions, so a resample
    that happens to miss a rare class still scores that class as 0 instead of silently
    shrinking the macro average.
    """
    labels = sorted(set(y_true) | set(y_pred))
    lookup = {label: index for index, label in enumerate(labels)}
    truth = np.asarray([lookup[label] for label in y_true], dtype=np.int64)
    predicted = np.asarray([lookup[label] for label in y_pred], dtype=np.int64)
    size = truth.size
    label_count = len(labels)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=float)

    # Count each (truth, prediction) pair per resample in one bincount per chunk; the
    # chunk keeps the R x L x L array small (L is 77 or 151).
    chunk = max(1, min(100, 4_000_000 // (label_count * label_count)))
    for start in range(0, n_resamples, chunk):
        count = min(chunk, n_resamples - start)
        positions = rng.integers(0, size, size=(count, size))
        resampled_truth = truth[positions]
        resampled_pred = predicted[positions]
        pair = resampled_truth * label_count + resampled_pred
        offsets = (np.arange(count) * label_count * label_count)[:, np.newaxis]
        flat = (pair + offsets).ravel()
        matrix = np.bincount(flat, minlength=count * label_count * label_count).reshape(
            count, label_count, label_count
        )
        diagonal = np.diagonal(matrix, axis1=1, axis2=2)
        false_positive = matrix.sum(axis=1) - diagonal
        false_negative = matrix.sum(axis=2) - diagonal
        denominator = 2 * diagonal + false_positive + false_negative
        f1 = np.divide(
            2 * diagonal,
            denominator,
            out=np.zeros_like(diagonal, dtype=float),
            where=denominator > 0,
        )
        scores[start : start + count] = f1.mean(axis=1)

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def latency_percentiles(values: Sequence[float]) -> tuple[float, float]:
    """p50 and p95 latency, linear interpolation."""
    if not values:
        msg = "latency percentiles need at least one value"
        raise ValueError(msg)
    p50, p95 = np.percentile(np.asarray(values, dtype=float), [50, 95])
    return float(p50), float(p95)


def hit_at_k(
    retrieved: Sequence[Sequence[str]],
    gold: Sequence[Iterable[str]],
    *,
    k: int = 4,
) -> float:
    """Share of questions with any gold passage id in the top ``k`` retrieved ids."""
    if len(retrieved) != len(gold):
        msg = f"{len(retrieved)} retrieval rows but {len(gold)} gold rows"
        raise ValueError(msg)
    if not retrieved:
        msg = "hit@k needs at least one question"
        raise ValueError(msg)
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    hits = 0
    for ranked, gold_ids in zip(retrieved, gold, strict=True):
        wanted = set(gold_ids)
        if any(item_id in wanted for item_id in list(ranked)[:k]):
            hits += 1
    return hits / len(retrieved)


def citation_precision(cited: Iterable[str], gold: Iterable[str]) -> float:
    """Per-item citation score, using ``PLAN.md`` §7's rule: cited ids ⊆ gold.

    ``1.0`` means at least one citation and every cited id is in the gold set. ``0.0``
    covers an answer with no citation as well as one citing anything outside gold. The
    task-level citation precision is the mean over items.
    """
    cited_ids = list(cited)
    if not cited_ids:
        return 0.0
    gold_ids = set(gold)
    return 1.0 if all(item_id in gold_ids for item_id in cited_ids) else 0.0
