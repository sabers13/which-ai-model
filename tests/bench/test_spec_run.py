"""Spec tasks through the runner: replay, score, card and prompt library end to end."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from which_ai_model.bench.run import (
    BenchStopError,
    Clock,
    ReplayIncompleteError,
    run,
)
from which_ai_model.bench.score import (
    build_recommendations,
    build_recommendations_with_specs,
    load_spec_cells,
)
from which_ai_model.report.cards import build_spec_card, write_card, write_prompt_library
from which_ai_model.task import load_spec

FIXTURE = Path("tests/fixtures/defect")
SPEC_PATH = FIXTURE / "spec.toml"
FIXED = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _clock() -> Clock:
    return Clock(monotonic=lambda: 0.0, now=lambda: FIXED, sleep=lambda _seconds: None)


def _spec(repo_root: Path):
    return load_spec(repo_root / SPEC_PATH)


def _run(
    repo_root: Path,
    work: Path,
    *,
    run_id: str = "fixture-replay",
    tasks: tuple[str, ...] = ("defect",),
):
    return run(
        "replay",
        config=repo_root / FIXTURE / "config",
        models_path=repo_root / FIXTURE / "models.lock.toml",
        tasks=tasks,
        specs=[_spec(repo_root)],
        recordings_dir=repo_root / FIXTURE / "recordings",
        runs_dir=work / "runs",
        reports_dir=work / "reports",
        recommendations_path=work / "recommendations.json",
        baseline_metrics=repo_root / "results" / "baseline" / "defect.json",
        run_id=run_id,
        root=repo_root,
        clock=_clock(),
    )


def _relative_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree(repo_root: Path, work: Path) -> dict[str, bytes]:
    files = _relative_files(work / "runs" / "fixture-replay")
    files["recommendations.json"] = (work / "recommendations.json").read_bytes()
    return files


def test_spec_replay_run_scores_writes_and_repeats_byte_identically(
    repo_root: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(repo_root, first)
    _run(repo_root, second)

    assert _tree(repo_root, first) == _tree(repo_root, second)

    run_dir = first / "runs" / "fixture-replay"
    payload = json.loads((run_dir / "cells.json").read_text(encoding="utf-8"))
    assert payload["cells"] == []
    assert {cell["task"] for cell in payload["spec_cells"]} == {"defect"}
    assert {cell["model"]["model_id"] for cell in payload["spec_cells"]} == {
        "fixture-accurate",
        "fixture-cheap",
    }
    cells = load_spec_cells(run_dir / "cells.json")
    accurate = next(cell for cell in cells if cell.model_id == "fixture-accurate")
    cheap = next(cell for cell in cells if cell.model_id == "fixture-cheap")
    assert accurate.primary_metric > 0.90
    assert cheap.primary_metric < 0.60
    assert accurate.metrics["schema_valid_rate"] == 1.0
    assert cheap.metrics["schema_valid_rate"] < 1.0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "fixture-replay"
    assert manifest["prompt_version"] == "d1"
    assert manifest["scorer_version"] == "s1"
    assert manifest["started_at"] == manifest["finished_at"] == "2026-11-01T12:00:00Z"

    predictions = run_dir / "predictions" / "defect-fixture-accurate.jsonl"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert rows[0]["task"] == "defect"
    assert set(rows[0]) == {"item_id", "task", "model", "output", "raw_key"}

    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 40
    assert {call["task"] for call in calls} == {"defect"}
    assert all(call["tool"] == "bench" for call in calls)
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))
    assert budget["total_estimated_usd"] > 0
    assert budget["total_spent_usd"] > 0


def test_card_and_prompt_library_are_built_from_the_replayed_run(
    repo_root: Path, tmp_path: Path
) -> None:
    _run(repo_root, tmp_path)
    run_dir = tmp_path / "runs" / "fixture-replay"
    spec = _spec(repo_root)
    cells = load_spec_cells(run_dir / "cells.json")
    card = build_spec_card(spec, cells)

    assert card.task == "defect"
    assert card.sample_size == 20
    assert card.most_accurate == "fixture-accurate"
    assert card.no_model_meets_floor is False
    assert len(card.models) == 2
    assert len(card.volumes) == 6

    markdown_path, json_path = write_card(card, out_dir=tmp_path / "cards")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Sort electrical complaints and flag safety")
    for heading in (
        "## The job",
        "## The data",
        "## Results",
        "## Recommendation",
        "## Gate rule",
        "## When to re-check",
        "## Cost at volume",
        "## Known limits",
    ):
        assert heading in markdown
    assert "`safety_relevant.recall` must stay at or above 0.80" in markdown
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["task"] == "defect"
    assert saved["gate"]["floors"][0]["metric"] == "safety_relevant.recall"

    library_md, library_json = write_prompt_library(card, spec, out_dir=tmp_path / "library")
    assert "## Output schema" in library_md.read_text(encoding="utf-8")
    entry = json.loads(library_json.read_text(encoding="utf-8"))
    assert entry["task"] == "defect"
    assert entry["prompt_version"] == "d1"
    assert entry["recommended_model"] == "fixture-accurate"
    # The weaker model is below the safety-recall floor, so it is never a fallback.
    assert entry["fallback_model"] is None
    assert "component_group" in entry["output_schema"]["properties"]


def test_recommendations_gain_the_spec_task_and_leave_v1_entries_alone(
    repo_root: Path, tmp_path: Path
) -> None:
    _run(repo_root, tmp_path)
    run_dir = tmp_path / "runs" / "fixture-replay"
    cells = load_spec_cells(run_dir / "cells.json")
    spec = _spec(repo_root)

    without_spec = build_recommendations([])
    combined = build_recommendations_with_specs([], cells, [spec])

    assert without_spec["tasks"] == {}
    assert set(combined["tasks"]) == {"defect"}
    entry = combined["tasks"]["defect"]
    assert entry["most_accurate"]["model"] == "fixture-accurate"
    assert entry["cheapest"]["model"] == "fixture-accurate"
    assert entry["no_model_meets_floor"] is False
    assert entry["gate_floors"][0]["metric"] == "safety_relevant.recall"
    assert [row["model"] for row in entry["good_enough"]] == ["fixture-accurate"]


def test_a_replay_miss_names_the_spec_task(repo_root: Path, tmp_path: Path) -> None:
    spec = _spec(repo_root)
    with pytest.raises(ReplayIncompleteError, match="defect"):
        run(
            "replay",
            config=repo_root / FIXTURE / "config",
            models_path=repo_root / FIXTURE / "models.lock.toml",
            tasks=["defect"],
            specs=[spec],
            recordings_dir=tmp_path / "no-recordings",
            runs_dir=tmp_path / "runs",
            reports_dir=tmp_path / "reports",
            recommendations_path=tmp_path / "recommendations.json",
            baseline_metrics=repo_root / "results" / "baseline" / "defect.json",
            run_id="missing",
            root=repo_root,
            clock=_clock(),
        )


def test_the_spec_budget_guard_stops_like_a_builtin_cell(repo_root: Path, tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "budget.toml").write_text(
        """
[budget]
window_hours = 5
cap_usd = 0.0000001

[limits.fixture-openai]
max_concurrency = 1
requests_per_minute = 600
""",
        encoding="utf-8",
    )
    prices = (repo_root / FIXTURE / "config" / "prices.toml").read_text(encoding="utf-8")
    (config / "prices.toml").write_text(
        prices.replace("input_per_mtok = 1.0", "input_per_mtok = 10.0").replace(
            "output_per_mtok = 2.0", "output_per_mtok = 20.0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchStopError, match="budget"):
        run(
            "replay",
            config=config,
            models_path=repo_root / FIXTURE / "models.lock.toml",
            tasks=["defect"],
            specs=[_spec(repo_root)],
            recordings_dir=repo_root / FIXTURE / "recordings",
            runs_dir=tmp_path / "runs",
            reports_dir=tmp_path / "reports",
            recommendations_path=tmp_path / "recommendations.json",
            baseline_metrics=repo_root / "results" / "baseline" / "defect.json",
            run_id="budget-stop",
            root=repo_root,
            clock=_clock(),
            spec_reduce_items=10,
        )
    assert (tmp_path / "reports" / "STOP.md").is_file()


def test_a_v1_only_run_keeps_the_v1_cells_and_manifest(repo_root: Path, tmp_path: Path) -> None:
    run(
        "replay",
        config=repo_root / "tests" / "fixtures" / "bench" / "config",
        models_path=repo_root / "tests" / "fixtures" / "bench" / "models.lock.toml",
        tasks=("T-ROUTE",),
        recordings_dir=repo_root / "tests" / "fixtures" / "bench" / "recordings",
        samples_dir=repo_root / "tests" / "fixtures" / "bench" / "samples",
        items_dir=repo_root / "tests" / "fixtures" / "bench" / "items",
        runs_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports",
        recommendations_path=tmp_path / "recommendations.json",
        baseline_metrics=repo_root / "tests" / "fixtures" / "bench" / "baseline-metrics.json",
        run_id="v1-only",
        clock=_clock(),
    )
    run_dir = tmp_path / "runs" / "v1-only"
    payload = json.loads((run_dir / "cells.json").read_text(encoding="utf-8"))
    assert "spec_cells" not in payload
    assert {cell["task"] for cell in payload["cells"]} == {"T-ROUTE"}
    assert load_spec_cells(run_dir / "cells.json") == []
    calls = (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert calls
    assert all(json.loads(line)["task"] == "T-ROUTE" for line in calls)


def test_the_cli_accepts_a_spec_id_but_not_an_unknown_ref() -> None:
    import typer

    from which_ai_model.bench.cli import _parse_tasks

    assert _parse_tasks("T-ROUTE,defect") == ("T-ROUTE", "defect")
    with pytest.raises(typer.BadParameter):
        _parse_tasks("Not A Task")


def test_build_card_finds_the_committed_spec_run(repo_root: Path, tmp_path: Path) -> None:
    from which_ai_model.report.cards import build_card

    _run(repo_root, tmp_path, run_id="live-v1.1")
    card = build_card(
        "defect",
        runs_dir=tmp_path / "runs",
        results_dir=repo_root / "results",
    )
    assert card.task == "defect"
    assert card.sample_size == 20
    assert card.baseline is not None
    assert card.baseline.is_baseline is True
    assert card.baseline.metrics["safety_relevant.recall"] >= 0.0
    assert card.most_accurate == "fixture-accurate"


def _budget_config(tmp_path: Path, *, ceilings: str = "", cap: str = "") -> Path:
    config = tmp_path / "config"
    config.mkdir()
    cap_line = f"cap_usd = {cap}\n" if cap else ""
    ceiling_block = f"\n[window.ceilings]\n{ceilings}" if ceilings else ""
    (config / "budget.toml").write_text(
        f"""
[budget]
window_hours = 5
{cap_line}{ceiling_block}
[limits.fixture-openai]
max_concurrency = 1
requests_per_minute = 600
""",
        encoding="utf-8",
    )
    prices = (Path("tests/fixtures/defect/config/prices.toml")).read_text(encoding="utf-8")
    (config / "prices.toml").write_text(prices, encoding="utf-8")
    return config


def test_the_budget_reduction_keeps_every_label(repo_root: Path, tmp_path: Path) -> None:
    """§A10 cuts the cell to 400 items; the cut must stay stratified by label."""
    config = _budget_config(
        tmp_path,
        ceilings="fixture-accurate = 0.03\nfixture-cheap = 0.015\n",
    )
    run(
        "replay",
        config=config,
        models_path=repo_root / FIXTURE / "models.lock.toml",
        tasks=["defect"],
        specs=[_spec(repo_root)],
        recordings_dir=repo_root / FIXTURE / "recordings",
        runs_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports",
        recommendations_path=tmp_path / "recommendations.json",
        baseline_metrics=repo_root / "results" / "baseline" / "defect.json",
        run_id="reduced",
        root=repo_root,
        clock=_clock(),
        spec_reduce_items=10,
    )
    truth = {
        row["id"]: row["component_group"]
        for row in (
            json.loads(line)
            for line in (repo_root / FIXTURE / "items.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    for slug in ("fixture-accurate", "fixture-cheap"):
        rows = tmp_path / "runs" / "reduced" / "predictions" / f"defect-{slug}.jsonl"
        predictions = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
        assert len(predictions) == 10
        assert {truth[row["item_id"]] for row in predictions} == {
            "battery_charging",
            "braking_system",
            "exterior_lighting",
        }
    deviations = (tmp_path / "reports" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert "reduced to 10 items" in deviations


def test_cap_usd_overrides_the_committed_budget(repo_root: Path, tmp_path: Path) -> None:
    with pytest.raises(BenchStopError, match="budget cap"):
        run(
            "replay",
            config=repo_root / FIXTURE / "config",
            models_path=repo_root / FIXTURE / "models.lock.toml",
            tasks=["defect"],
            specs=[_spec(repo_root)],
            recordings_dir=repo_root / FIXTURE / "recordings",
            runs_dir=tmp_path / "runs",
            reports_dir=tmp_path / "reports",
            recommendations_path=tmp_path / "recommendations.json",
            baseline_metrics=repo_root / "results" / "baseline" / "defect.json",
            run_id="capped",
            root=repo_root,
            clock=_clock(),
            cap_usd=0.0000001,
        )
    stop = (tmp_path / "reports" / "STOP.md").read_text(encoding="utf-8")
    assert "budget cap" in stop


def test_the_lock_can_record_a_provider_default_temperature() -> None:
    """A model that rejects ``temperature`` is recorded as provider-default (§A5)."""
    from which_ai_model.bench.run import load_models, run_settings_for

    specs = {spec.ref.model_id: spec for spec in load_models(Path("config/models.lock.toml"))}
    claude = run_settings_for(specs["claude-haiku-5-5"])
    gpt = run_settings_for(specs["gpt-6-luna"])
    assert claude.temperature == "provider-default"
    assert gpt.temperature == 0.0
    assert claude.max_output_tokens == 1024


def test_a_replayed_spec_cell_reports_unknown_latency(repo_root: Path, tmp_path: Path) -> None:
    """A replayed call has no measured latency; a zero must not read as "instant"."""
    _run(repo_root, tmp_path)
    cells = load_spec_cells(tmp_path / "runs" / "fixture-replay" / "cells.json")
    assert cells
    assert all(cell.latency_p50 is None and cell.latency_p95 is None for cell in cells)
