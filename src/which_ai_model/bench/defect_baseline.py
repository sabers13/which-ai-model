"""T-DEFECT baselines (``PLAN-v1.1`` §A4), both at cost 0.

* ``component_group``: the same TF-IDF (word 1-2-grams) + logistic-regression pipeline as
  the v1 baselines, trained on the committed ``data/defect/train.jsonl.gz``.
* ``safety_relevant``: the fixed keyword rule in ``data/defect/safety_keywords.txt``.

The result lands in ``results/baseline/defect.json``; the v1 baseline files are left
untouched. Everything here is deterministic: the same committed data gives the same
bytes, so the result can be rebuilt without a model call and without a socket.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from which_ai_model.bench.baseline import MODEL_DESCRIPTION, Baseline
from which_ai_model.bench.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_metric_ci,
    macro_f1,
    precision_recall_f1,
    wilson_ci,
)
from which_ai_model.bench.nhtsa import (
    DEFECT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    file_sha256,
)
from which_ai_model.contracts import SCHEMA_VERSION, ModelRef

__all__ = [
    "BASELINE_MODEL",
    "KEYWORDS_PATH",
    "RESULTS_DIR",
    "SEED",
    "compile_keywords",
    "keyword_flags",
    "load_items",
    "load_keywords",
    "main",
    "run_baseline",
]

RESULTS_DIR: Final = Path("results/baseline")
KEYWORDS_PATH: Final = DEFECT_DIR / "safety_keywords.txt"

BASELINE_MODEL: Final = ModelRef(connection_id="baseline", model_id="tfidf-logreg")

#: The spec's sampling seed; the bootstrap resamples use the same one so a rebuild is
#: byte-identical. ``tests/bench/test_defect_baseline.py`` checks it against the spec.
SEED: Final = 20261031

POSITIVE: Final = "safety_relevant"
NEGATIVE: Final = "not_safety_relevant"


def load_keywords(path: Path = KEYWORDS_PATH) -> tuple[str, ...]:
    """The fixed keyword list: one keyword per line, ``#`` comments, blank lines ignored."""
    keywords: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keywords.append(line.lower())
    if not keywords:
        msg = f"{path} lists no keywords; the safety baseline needs at least one"
        raise ValueError(msg)
    return tuple(keywords)


def compile_keywords(keywords: Sequence[str]) -> re.Pattern[str]:
    """One case-insensitive whole-word pattern for the whole list."""
    alternation = "|".join(re.escape(keyword) for keyword in keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def keyword_flags(texts: Sequence[str], keywords: Sequence[str]) -> list[bool]:
    """True when a narrative contains any keyword as a whole word or phrase."""
    pattern = compile_keywords(keywords)
    return [pattern.search(text) is not None for text in texts]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_items(path: Path) -> list[dict[str, Any]]:
    """Read a committed JSONL extract (plain or gzipped)."""
    if not path.is_file():
        msg = f"{path} is missing; rebuild it with `python -m which_ai_model.bench.nhtsa build`"
        raise FileNotFoundError(msg)
    return _read_jsonl(path)


def _wilson(successes: int, n: int) -> tuple[float, float]:
    return wilson_ci(successes, n) if n else (0.0, 1.0)


def run_baseline(
    *,
    data_dir: Path = DEFECT_DIR,
    results_dir: Path = RESULTS_DIR,
    keywords_path: Path = KEYWORDS_PATH,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train both baselines, write ``defect.json`` and return the metrics."""
    test_path = data_dir / TEST_PATH.name
    train_path = data_dir / TRAIN_PATH.name
    test = load_items(test_path)
    train = load_items(train_path)

    texts = [str(item["text"]) for item in test]
    group_truth = [str(item["component_group"]) for item in test]
    safety_truth = [bool(item["safety_relevant"]) for item in test]

    model = Baseline(random_state=seed).fit(
        [str(item["text"]) for item in train],
        [str(item["component_group"]) for item in train],
    )
    group_pred = model.predict(texts)

    keywords = load_keywords(keywords_path)
    safety_pred = keyword_flags(texts, keywords)

    group_correct = [
        1.0 if truth == predicted else 0.0
        for truth, predicted in zip(group_truth, group_pred, strict=True)
    ]
    accuracy_low, accuracy_high = bootstrap_ci(group_correct, seed=seed)
    f1_low, f1_high = bootstrap_metric_ci(group_truth, group_pred, metric=macro_f1, seed=seed)

    safety_true_labels = [POSITIVE if value else NEGATIVE for value in safety_truth]
    safety_pred_labels = [POSITIVE if value else NEGATIVE for value in safety_pred]
    precision, recall, f1 = precision_recall_f1(safety_true_labels, safety_pred_labels, POSITIVE)
    true_positive = sum(
        1
        for truth, predicted in zip(safety_true_labels, safety_pred_labels, strict=True)
        if truth == predicted == POSITIVE
    )
    predicted_positive = sum(1 for value in safety_pred if value)
    support = sum(1 for value in safety_truth if value)
    precision_low, precision_high = _wilson(true_positive, predicted_positive)
    recall_low, recall_high = _wilson(true_positive, support)

    group_counts = dict(sorted(Counter(group_truth).items()))
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "defect",
        "model": BASELINE_MODEL.model_dump(mode="json"),
        "description": MODEL_DESCRIPTION,
        "seed": seed,
        "cost_per_1k_usd": 0.0,
        "data": {
            "test_file": (data_dir / TEST_PATH.name).as_posix(),
            "train_file": (data_dir / TRAIN_PATH.name).as_posix(),
            "n_test": len(test),
            "n_train": len(train),
            "sample_hash": file_sha256(test_path),
            "groups": group_counts,
            "safety_positive": support,
        },
        "component_group": {
            "n": len(test),
            "macro_f1": macro_f1(group_truth, group_pred),
            "accuracy": accuracy(group_truth, group_pred),
            "coverage": 1.0,
            "accuracy_answered": accuracy(group_truth, group_pred),
            "ci": {
                "macro_f1_low": f1_low,
                "macro_f1_high": f1_high,
                "accuracy_low": accuracy_low,
                "accuracy_high": accuracy_high,
            },
        },
        "safety_relevant": {
            "n": len(test),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "predicted_positive": predicted_positive,
            "support": support,
            "rule": "keyword",
            "keywords_file": keywords_path.as_posix(),
            "keyword_count": len(keywords),
            "ci": {
                "precision_low": precision_low,
                "precision_high": precision_high,
                "recall_low": recall_low,
                "recall_high": recall_high,
            },
        },
        "schema_valid_rate": 1.0,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "defect.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFECT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--keywords", type=Path, default=KEYWORDS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        metrics = run_baseline(
            data_dir=args.data,
            results_dir=args.results,
            keywords_path=args.keywords,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"defect-baseline: {exc}\n")
        return 1
    sys.stdout.write(
        "defect-baseline: component-group macro-F1 {f1:.4f} (accuracy {accuracy:.4f}); "
        "safety recall {recall:.4f}, precision {precision:.4f}\n".format(
            f1=metrics["component_group"]["macro_f1"],
            accuracy=metrics["component_group"]["accuracy"],
            recall=metrics["safety_relevant"]["recall"],
            precision=metrics["safety_relevant"]["precision"],
        )
    )
    sys.stdout.write(f"defect-baseline: wrote {args.results / 'defect.json'}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
