"""The T-DEFECT baselines: the fixed keyword rule and the TF-IDF pipeline.

The full run needs the committed extracts, so the tests slice those files into
``tmp_path`` and run the same code path on the slice; nothing here downloads, calls a
model or reads the scout's snapshot folder.
"""

from __future__ import annotations

import gzip
import json
import tomllib
from pathlib import Path

import pytest

from which_ai_model.bench.defect_baseline import (
    SEED,
    compile_keywords,
    keyword_flags,
    load_items,
    load_keywords,
    main,
    run_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFECT_DIR = REPO_ROOT / "data" / "defect"


def test_keyword_rule_flags_every_family_it_covers() -> None:
    keywords = load_keywords(DEFECT_DIR / "safety_keywords.txt")
    assert keyword_flags(["The car caught fire on the motorway"], keywords) == [True]
    assert keyword_flags(["Smoke came out of the dashboard"], keywords) == [True]
    assert keyword_flags(["We crashed into the barrier"], keywords) == [True]
    assert keyword_flags(["The driver was injured and went to hospital"], keywords) == [True]
    assert keyword_flags(["Two deaths were reported"], keywords) == [True]
    assert keyword_flags(["The radio resets every morning"], keywords) == [False]


def test_keyword_rule_matches_whole_words_only_and_ignores_case() -> None:
    keywords = load_keywords(DEFECT_DIR / "safety_keywords.txt")
    assert keyword_flags(["CRASHED into a pole"], keywords) == [True]
    assert keyword_flags(["The firefly lamp is dim"], keywords) == [False]
    assert keyword_flags(["The injured worker left"], keywords) == [True]
    assert keyword_flags(["The uninjured passenger left"], keywords) == [False]


def test_keyword_list_is_fixed_plain_and_unique() -> None:
    keywords = load_keywords(DEFECT_DIR / "safety_keywords.txt")
    assert len(keywords) >= 50
    assert len(set(keywords)) == len(keywords)
    assert all(keyword == keyword.lower() for keyword in keywords)
    assert all(keyword.strip() for keyword in keywords)
    assert "crash" in keywords
    assert "fire" in keywords
    assert "injury" in keywords


def test_the_compiled_rule_is_one_pattern() -> None:
    pattern = compile_keywords(["fire", "crash"])
    assert pattern.search("a fire") is not None
    assert pattern.search("a crash") is not None
    assert pattern.search("a firework") is None


def test_the_bootstrap_seed_matches_the_spec() -> None:
    spec = tomllib.loads((DEFECT_DIR / "nhtsa-spec.toml").read_text(encoding="utf-8"))
    assert spec["sampling"]["seed"] == SEED


def _sliced_data(tmp_path: Path) -> Path:
    """A small, group-spread slice of the committed extracts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test = load_items(DEFECT_DIR / "test.jsonl")[::30]
    train = load_items(DEFECT_DIR / "train.jsonl.gz")[::250]
    (data_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in test) + "\n", encoding="utf-8"
    )
    with gzip.GzipFile(data_dir / "train.jsonl.gz", mode="wb", mtime=0) as handle:
        for item in train:
            handle.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
    return data_dir


def test_baseline_is_reproducible_on_the_committed_data(tmp_path: Path) -> None:
    data_dir = _sliced_data(tmp_path)
    first = run_baseline(data_dir=data_dir, results_dir=tmp_path / "results-1")
    second = run_baseline(data_dir=data_dir, results_dir=tmp_path / "results-2")
    assert first == second
    assert (tmp_path / "results-1" / "defect.json").read_bytes() == (
        tmp_path / "results-2" / "defect.json"
    ).read_bytes()


def test_baseline_reports_the_metrics_the_spec_names(tmp_path: Path) -> None:
    data_dir = _sliced_data(tmp_path)
    metrics = run_baseline(data_dir=data_dir, results_dir=tmp_path / "results")
    assert metrics["task"] == "defect"
    assert metrics["cost_per_1k_usd"] == 0.0
    assert metrics["schema_valid_rate"] == 1.0
    assert 0.0 <= metrics["component_group"]["macro_f1"] <= 1.0
    assert 0.0 <= metrics["component_group"]["accuracy"] <= 1.0
    assert metrics["component_group"]["coverage"] == 1.0
    assert 0.0 <= metrics["safety_relevant"]["recall"] <= 1.0
    assert 0.0 <= metrics["safety_relevant"]["precision"] <= 1.0
    assert metrics["data"]["n_test"] == metrics["component_group"]["n"]
    assert metrics["data"]["sample_hash"]
    ci = metrics["safety_relevant"]["ci"]
    assert ci["recall_low"] <= metrics["safety_relevant"]["recall"] <= ci["recall_high"]


def test_baseline_leaves_other_result_files_alone(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = results_dir / "metrics.json"
    sentinel.write_text("{}\n", encoding="utf-8")
    run_baseline(data_dir=_sliced_data(tmp_path), results_dir=results_dir)
    assert sentinel.read_text(encoding="utf-8") == "{}\n"
    assert (results_dir / "defect.json").is_file()


def test_baseline_cli_reports_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--data",
            str(tmp_path / "nothing"),
            "--results",
            str(tmp_path / "results"),
        ]
    )
    assert code == 1
    assert "missing" in capsys.readouterr().err
