"""TF-IDF + logistic-regression baselines (``PLAN.md`` §7).

Two fixed pipelines, no tuning:

* trained on the Banking77 **train** split, scored on the committed T-ROUTE ids;
* trained on the CLINC150 ``plus`` **train** split with ``oos`` as a class, scored on
  the committed T-SCOPE ids.

``make baseline`` (via ``uv run python -m which_ai_model.bench.baseline``) writes
``results/baseline/route_predictions.jsonl``, ``scope_predictions.jsonl`` and
``metrics.json``. The baseline costs 0 and is reported beside the models. Tests exercise
the same path on a synthetic three-class toy set; downloading or training on the real
files is never part of ``make test``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from which_ai_model.bench.data import (
    CLINC_OOS_LABEL,
    RAW_DIR,
    load_banking77,
    load_clinc150,
)
from which_ai_model.bench.metrics import (
    DEFAULT_SEED,
    accuracy,
    bootstrap_ci,
    bootstrap_metric_ci,
    macro_f1,
    precision_recall_f1,
    wilson_ci,
)
from which_ai_model.bench.sample import SAMPLE_DIR, SAMPLE_SEED, load_sample_ids, sample_hash
from which_ai_model.contracts import SCHEMA_VERSION, ModelRef, Prediction, TaskName

__all__ = [
    "BASELINE_MODEL",
    "Baseline",
    "run_baseline",
]

RESULTS_DIR = Path("results/baseline")
BASELINE_MODEL = ModelRef(connection_id="baseline", model_id="tfidf-logreg")
MODEL_DESCRIPTION = "TF-IDF word 1-2 grams + LogisticRegression(max_iter=2000)"


class Baseline:
    """A deterministic TF-IDF (1-2-grams) + logistic-regression classifier."""

    def __init__(self, *, random_state: int = DEFAULT_SEED) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._model = LogisticRegression(max_iter=2_000, random_state=random_state)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Baseline:
        if len(texts) != len(labels):
            msg = f"{len(texts)} texts but {len(labels)} labels"
            raise ValueError(msg)
        if not texts:
            msg = "cannot fit on an empty training set"
            raise ValueError(msg)
        features = self._vectorizer.fit_transform(list(texts))
        self._model.fit(features, list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        features = self._vectorizer.transform(list(texts))
        return [str(label) for label in self._model.predict(features)]


def _frame_by_id(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    return {
        str(row.item_id): (str(row.text), str(row.intent)) for row in frame.itertuples(index=False)
    }


def _texts_for_ids(
    index: dict[str, tuple[str, str]],
    ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    missing = [item_id for item_id in ids if item_id not in index]
    if missing:
        msg = f"sample id(s) missing from the dataset: {missing[:5]}"
        raise ValueError(msg)
    texts = [index[item_id][0] for item_id in ids]
    labels = [index[item_id][1] for item_id in ids]
    return texts, labels


def _write_predictions(
    path: Path,
    task: TaskName,
    rows: Sequence[tuple[str, dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item_id, output in rows:
            prediction = Prediction(
                item_id=item_id,
                task=task,
                model=BASELINE_MODEL,
                output=output,
                raw_key=f"baseline:{item_id}",
            )
            handle.write(prediction.model_dump_json() + "\n")


def _scope_label(intent: str) -> str:
    return "out_of_scope" if intent == CLINC_OOS_LABEL else "in_scope"


def _scope_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return precision_recall_f1(y_true, y_pred, "out_of_scope")[2]


def _wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson interval, or an uninformative (0, 1) when there is no observation."""
    return wilson_ci(successes, n) if n else (0.0, 1.0)


def run_baseline(
    *,
    data_root: Path = RAW_DIR,
    samples_dir: Path = SAMPLE_DIR,
    results_dir: Path = RESULTS_DIR,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Train both baselines, write predictions and metrics, and return the metrics."""
    sample_ids_hash = sample_hash(samples_dir)

    route_ids = load_sample_ids("route", samples_dir=samples_dir)
    banking_train = load_banking77("train", root=data_root)
    banking_test = load_banking77("test", root=data_root)
    route_index = _frame_by_id(banking_test)
    route_texts, route_truth = _texts_for_ids(route_index, route_ids)
    route_model = Baseline(random_state=seed).fit(
        banking_train["text"].astype(str).tolist(),
        banking_train["intent"].astype(str).tolist(),
    )
    route_pred = route_model.predict(route_texts)

    scope_ids = load_sample_ids("scope", samples_dir=samples_dir)
    clinc_train = load_clinc150("train", root=data_root)
    clinc_test = load_clinc150("test", root=data_root)
    scope_index = _frame_by_id(clinc_test)
    scope_texts, scope_intent = _texts_for_ids(scope_index, scope_ids)
    scope_truth = [_scope_label(intent) for intent in scope_intent]
    scope_model = Baseline(random_state=seed).fit(
        clinc_train["text"].astype(str).tolist(),
        clinc_train["intent"].astype(str).tolist(),
    )
    scope_pred_intent = scope_model.predict(scope_texts)
    scope_pred = [_scope_label(intent) for intent in scope_pred_intent]

    route_correct = [
        1.0 if truth == predicted else 0.0
        for truth, predicted in zip(route_truth, route_pred, strict=True)
    ]
    route_accuracy_low, route_accuracy_high = bootstrap_ci(route_correct, seed=seed)
    route_f1_low, route_f1_high = bootstrap_metric_ci(
        route_truth, route_pred, metric=macro_f1, seed=seed
    )
    scope_precision, scope_recall, scope_f1 = precision_recall_f1(
        scope_truth, scope_pred, "out_of_scope"
    )
    scope_f1_low, scope_f1_high = bootstrap_metric_ci(
        scope_truth, scope_pred, metric=_scope_f1, seed=seed
    )
    scope_tp = sum(
        1
        for truth, predicted in zip(scope_truth, scope_pred, strict=True)
        if truth == predicted == "out_of_scope"
    )
    scope_predicted_positive = sum(1 for predicted in scope_pred if predicted == "out_of_scope")
    scope_support = sum(1 for truth in scope_truth if truth == "out_of_scope")
    precision_low, precision_high = _wilson(scope_tp, scope_predicted_positive)
    recall_low, recall_high = _wilson(scope_tp, scope_support)

    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": BASELINE_MODEL.model_dump(mode="json"),
        "description": MODEL_DESCRIPTION,
        "sample_hash": sample_ids_hash,
        "seed": seed,
        "cost_per_1k_usd": 0.0,
        "route": {
            "n": len(route_ids),
            "accuracy": accuracy(route_truth, route_pred),
            "macro_f1": macro_f1(route_truth, route_pred),
            "ci": {
                "accuracy_low": route_accuracy_low,
                "accuracy_high": route_accuracy_high,
                "macro_f1_low": route_f1_low,
                "macro_f1_high": route_f1_high,
            },
        },
        "scope": {
            "n": len(scope_ids),
            "precision": scope_precision,
            "recall": scope_recall,
            "f1": scope_f1,
            "accuracy": accuracy(scope_truth, scope_pred),
            "ci": {
                "precision_low": precision_low,
                "precision_high": precision_high,
                "recall_low": recall_low,
                "recall_high": recall_high,
                "f1_low": scope_f1_low,
                "f1_high": scope_f1_high,
            },
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        results_dir / "route_predictions.jsonl",
        TaskName.ROUTE,
        [
            (item_id, {"intent": predicted})
            for item_id, predicted in zip(route_ids, route_pred, strict=True)
        ],
    )
    _write_predictions(
        results_dir / "scope_predictions.jsonl",
        TaskName.SCOPE,
        [
            (item_id, {"out_of_scope": predicted == "out_of_scope"})
            for item_id, predicted in zip(scope_ids, scope_pred, strict=True)
        ],
    )
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=RAW_DIR)
    parser.add_argument("--samples", type=Path, default=SAMPLE_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)
    try:
        metrics = run_baseline(
            data_root=args.data_root,
            samples_dir=args.samples,
            results_dir=args.results,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"baseline: {exc}\n")
        sys.stderr.write(
            "baseline: run `make fetch-data` first and keep data/samples/ committed.\n"
        )
        return 1
    sys.stdout.write(
        "baseline: T-ROUTE accuracy {accuracy:.4f}, macro-F1 {macro_f1:.4f}; "
        "T-SCOPE F1 {f1:.4f}\n".format(
            accuracy=metrics["route"]["accuracy"],
            macro_f1=metrics["route"]["macro_f1"],
            f1=metrics["scope"]["f1"],
        )
    )
    sys.stdout.write(f"baseline: wrote {args.results}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
