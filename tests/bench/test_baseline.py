"""The TF-IDF baseline trains, predicts and writes results on a synthetic mini set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from which_ai_model.bench.baseline import (
    BASELINE_MODEL,
    Baseline,
    main,
    run_baseline,
)
from which_ai_model.bench.data import BANKING77_LABELS, CLINC150_LABELS
from which_ai_model.bench.sample import build_samples, sample_hash
from which_ai_model.contracts import Prediction


def test_baseline_learns_a_three_class_toy_set() -> None:
    texts = [
        "alpha alpha payment",
        "alpha alpha transfer",
        "beta beta card",
        "beta beta loan",
        "gamma gamma address",
        "gamma gamma phone",
    ]
    labels = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
    model = Baseline(random_state=7).fit(texts, labels)

    assert model.predict(["alpha payment", "beta card", "gamma phone"]) == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert model.predict([]) == []
    with pytest.raises(ValueError, match="labels"):
        Baseline().fit(["one"], [])
    with pytest.raises(ValueError, match="empty"):
        Baseline().fit([], [])


def _write_banking(root: Path, split: str, *, rows_per_intent: int) -> None:
    rows = [
        {"text": f"{intent} banking {index}", "label": index_label}
        for index_label, intent in enumerate(BANKING77_LABELS)
        for index in range(rows_per_intent)
    ]
    pd.DataFrame(rows).to_parquet(root / f"banking77_{split}.parquet")


def _write_clinc(root: Path, split: str, *, rows_per_intent: int, oos_rows: int) -> None:
    rows = [
        {"text": f"{intent} clinc {index}", "intent": index_label}
        for index_label, intent in enumerate(CLINC150_LABELS)
        for index in range(oos_rows if intent == "oos" else rows_per_intent)
    ]
    pd.DataFrame(rows).to_parquet(root / f"clinc150_plus_{split}.parquet")


@pytest.fixture
def mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    _write_banking(data_root, "train", rows_per_intent=3)
    _write_banking(data_root, "test", rows_per_intent=10)
    _write_clinc(data_root, "train", rows_per_intent=3, oos_rows=100)
    _write_clinc(data_root, "test", rows_per_intent=2, oos_rows=200)
    samples = tmp_path / "samples"
    build_samples(data_root=data_root, samples_dir=samples)
    return data_root, samples


def test_run_baseline_writes_predictions_and_metrics(
    tmp_path: Path,
    mini_dataset: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, samples = mini_dataset
    results = tmp_path / "results" / "baseline"

    metrics = run_baseline(
        data_root=data_root,
        samples_dir=samples,
        results_dir=results,
    )

    assert metrics["model"] == BASELINE_MODEL.model_dump(mode="json")
    assert metrics["sample_hash"] == sample_hash(samples)
    assert metrics["cost_per_1k_usd"] == 0.0
    assert metrics["route"]["n"] == 616
    assert metrics["scope"]["n"] == 300
    assert 0.0 <= metrics["route"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["route"]["macro_f1"] <= 1.0
    assert 0.0 <= metrics["scope"]["f1"] <= 1.0
    assert set(metrics["route"]["ci"]) == {
        "accuracy_low",
        "accuracy_high",
        "macro_f1_low",
        "macro_f1_high",
    }

    route_file = results / "route_predictions.jsonl"
    scope_file = results / "scope_predictions.jsonl"
    route_rows = [json.loads(line) for line in route_file.read_text(encoding="utf-8").splitlines()]
    scope_rows = [json.loads(line) for line in scope_file.read_text(encoding="utf-8").splitlines()]
    assert len(route_rows) == 616
    assert len(scope_rows) == 300
    first = Prediction.model_validate(route_rows[0])
    assert first.model == BASELINE_MODEL
    assert first.task.value == "T-ROUTE"
    assert set(first.output) == {"intent"}
    assert set(Prediction.model_validate(scope_rows[0]).output) == {"out_of_scope"}

    written = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    assert written == metrics

    assert (
        main(["--data-root", str(data_root), "--samples", str(samples), "--results", str(results)])
        == 0
    )
    printed = capsys.readouterr().out
    assert "T-ROUTE accuracy" in printed
    assert "T-SCOPE F1" in printed


def test_run_baseline_is_byte_identical_twice(
    tmp_path: Path,
    mini_dataset: tuple[Path, Path],
) -> None:
    data_root, samples = mini_dataset
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_baseline(data_root=data_root, samples_dir=samples, results_dir=first)
    run_baseline(data_root=data_root, samples_dir=samples, results_dir=second)

    for name in ("metrics.json", "route_predictions.jsonl", "scope_predictions.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_baseline_cli_explains_missing_data(tmp_path: Path) -> None:
    code = main(
        [
            "--data-root",
            str(tmp_path / "nowhere"),
            "--samples",
            str(tmp_path / "samples"),
            "--results",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
