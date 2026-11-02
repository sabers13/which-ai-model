"""Runner, budget guard, resume and recommendation tests (fake models, fake clock)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from which_ai_model.bench.run import (
    BenchStopError,
    Clock,
    PolicyStopError,
    estimate_cell_cost,
    load_budget_config,
    load_models,
    parse_response,
    reduce_route_items,
    run,
)
from which_ai_model.bench.score import (
    build_recommendations,
    candidates_from_cells,
    load_cells,
)
from which_ai_model.contracts import CellResult, ModelRef, TaskName
from which_ai_model.core import load_prices


class FakeClock:
    """Deterministic monotonic and wall clock with an injected sleep."""

    def __init__(self, moment: datetime | None = None) -> None:
        self._monotonic = 0.0
        self._now = moment or datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._monotonic

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)

    def clock(self) -> Clock:
        return Clock(monotonic=self.monotonic, now=self.now, sleep=self.sleep)


def _write_config(
    root: Path,
    *,
    cap_usd: float | None = None,
    window_ceiling: float | None = None,
    peak_windows: str = "",
) -> Path:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    budget = ["[budget]", "window_hours = 5", ""]
    if cap_usd is not None:
        budget.append(f"cap_usd = {cap_usd}")
    if window_ceiling is not None:
        budget.extend(["[window.ceilings]", f"fake-a = {window_ceiling}"])
    budget.extend(["[limits.opencode-go]", "max_concurrency = 2", "requests_per_minute = 30"])
    if peak_windows:
        budget.append(peak_windows)
    (config / "budget.toml").write_text("\n".join(budget) + "\n", encoding="utf-8")
    (config / "prices.toml").write_text(
        """
[models."fake-a"]
input_per_mtok = 1.0
output_per_mtok = 10.0
currency = "USD"
""",
        encoding="utf-8",
    )
    lock = """
[[model]]
model_id = "fake-a"
connection_id = "fake"
protocol = "OPENAI_COMPATIBLE"
reasoning = "minimal"
max_tokens = 1024
"""
    (config / "models.lock.toml").write_text(lock, encoding="utf-8")
    return config


def _write_samples(root: Path) -> Path:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "route.txt").write_text("r1\nr2\nr3\nr4\n", encoding="utf-8")
    (samples / "scope.txt").write_text("s1\n", encoding="utf-8")
    (samples / "reply.txt").write_text("r1\n", encoding="utf-8")
    return samples


def _write_items(root: Path, *, per_intent: int = 2, intents: int = 2) -> Path:
    items_dir = root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, intent in enumerate(["card_not_working", "card_arrival"]):
        for position in range(per_intent):
            rows.append(
                {
                    "item_id": f"r{index * per_intent + position + 1}",
                    "text": f"message {index}-{position}",
                    "intent": intent,
                }
            )
    (items_dir / "route.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return items_dir


def _invoker_for(items_dir: Path):
    truth = {
        row["text"]: row["intent"]
        for row in (
            json.loads(line)
            for line in (items_dir / "route.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    calls: list[str] = []

    def invoke(_model: ModelRef, messages) -> tuple[dict[str, object], int]:
        text = str(messages[-1]["content"]).splitlines()[-1]
        calls.append(text)
        response = {
            "content": json.dumps({"intent": truth[text]}),
            "tool_calls": [],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
        return response, 7

    invoke.calls = calls  # type: ignore[attr-defined]
    return invoke


def _run_kwargs(root: Path, config: Path) -> dict[str, object]:
    return {
        "config": config,
        "models_path": config / "models.lock.toml",
        "tasks": [TaskName.ROUTE],
        "recordings_dir": root / "recordings",
        "samples_dir": root / "samples",
        "items_dir": root / "items",
        "runs_dir": root / "runs",
        "reports_dir": root / "reports",
        "recommendations_path": root / "recommendations.json",
        "baseline_metrics": root / "missing-baseline.json",
    }


def test_record_run_writes_outputs_and_resumes_without_repeating_calls(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    samples = _write_samples(tmp_path)
    items_dir = _write_items(tmp_path)
    kwargs = _run_kwargs(tmp_path, config)
    kwargs["samples_dir"] = samples
    invoker = _invoker_for(items_dir)
    clock = FakeClock()
    item_count = 4
    first = [0]

    def interrupting(model: ModelRef, messages) -> tuple[dict[str, object], int]:
        first[0] += 1
        if first[0] == 3:
            raise KeyboardInterrupt
        return invoker(model, messages)

    with pytest.raises(KeyboardInterrupt):
        run(
            "record",
            run_id="resume",
            invoker=interrupting,
            clock=clock.clock(),
            workers=1,
            **kwargs,
        )
    assert len(invoker.calls) == 2

    resumed_invoker = _invoker_for(items_dir)
    manifest = run(
        "record",
        run_id="resume",
        invoker=resumed_invoker,
        clock=clock.clock(),
        workers=1,
        **kwargs,
    )
    assert manifest.run_id == "resume"
    assert len(resumed_invoker.calls) == item_count - 2
    run_dir = tmp_path / "runs" / "resume"
    predictions = list((run_dir / "predictions").glob("*.jsonl"))
    assert len(predictions) == 1
    assert len(predictions[0].read_text(encoding="utf-8").splitlines()) == item_count
    assert len((run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == item_count
    recorded = list((tmp_path / "recordings").rglob("*.jsonl"))
    assert (
        sum(len(path.read_text(encoding="utf-8").splitlines()) for path in recorded) == item_count
    )
    cells = load_cells(run_dir / "cells.json")
    assert cells[0].n == item_count


def test_budget_reduction_cuts_route_and_logs_deviation(tmp_path: Path) -> None:
    items_dir = _write_items(tmp_path, per_intent=8, intents=2)
    prices = load_prices(_write_config(tmp_path, cap_usd=1.0) / "prices.toml")
    full = estimate_cell_cost(
        TaskName.ROUTE,
        [
            json.loads(line)
            for line in (items_dir / "route.jsonl").read_text(encoding="utf-8").splitlines()
        ],
        prices,
        model_id="fake-a",
    )
    reduced_items = reduce_route_items(
        [
            json.loads(line)
            for line in (items_dir / "route.jsonl").read_text(encoding="utf-8").splitlines()
        ],
        per_intent=5,
    )
    reduced = estimate_cell_cost(TaskName.ROUTE, reduced_items, prices, model_id="fake-a")
    assert full is not None
    assert reduced is not None
    cap = (full + reduced) / 2
    config = _write_config(tmp_path, cap_usd=cap)
    samples = _write_samples(tmp_path)
    kwargs = _run_kwargs(tmp_path, config)
    kwargs["samples_dir"] = samples
    invoker = _invoker_for(items_dir)

    manifest = run(
        "record",
        run_id="reduced",
        invoker=invoker,
        clock=FakeClock().clock(),
        workers=1,
        **kwargs,
    )
    assert manifest.run_id == "reduced"
    cells = load_cells(tmp_path / "runs" / "reduced" / "cells.json")
    assert cells[0].n == 10
    deviations = (tmp_path / "reports" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert "T-ROUTE reduced to 5 items per intent" in deviations
    budget = json.loads((tmp_path / "runs" / "reduced" / "budget.json").read_text(encoding="utf-8"))
    assert budget["cap_usd"] == cap
    assert budget["reductions"]


def test_budget_stop_when_even_reduced_cell_cannot_fit(tmp_path: Path) -> None:
    config = _write_config(tmp_path, cap_usd=1e-9)
    samples = _write_samples(tmp_path)
    items_dir = _write_items(tmp_path, per_intent=8, intents=2)
    kwargs = _run_kwargs(tmp_path, config)
    kwargs["samples_dir"] = samples
    invoker = _invoker_for(items_dir)

    with pytest.raises(BenchStopError):
        run(
            "record",
            run_id="stopped",
            invoker=invoker,
            clock=FakeClock().clock(),
            workers=1,
            **kwargs,
        )
    stop = (tmp_path / "reports" / "STOP.md").read_text(encoding="utf-8")
    assert "budget cap" in stop
    assert invoker.calls == []  # type: ignore[attr-defined]


def test_policy_stop_stops_without_retry(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    samples = _write_samples(tmp_path)
    _write_items(tmp_path)
    kwargs = _run_kwargs(tmp_path, config)
    kwargs["samples_dir"] = samples
    calls = [0]

    class AbuseError(RuntimeError):
        status_code = 403

    def invoke(_model: ModelRef, _messages) -> tuple[dict[str, object], int]:
        calls[0] += 1
        raise AbuseError("abuse detected for this account")

    with pytest.raises(PolicyStopError):
        run(
            "record",
            run_id="policy",
            invoker=invoke,
            clock=FakeClock().clock(),
            workers=1,
            **kwargs,
        )
    assert calls == [1]
    stop = (tmp_path / "reports" / "STOP.md").read_text(encoding="utf-8")
    assert "403" in stop
    assert "abuse" in stop.lower()


def test_deepseek_waits_out_peak_hours(tmp_path: Path) -> None:
    peak = (
        '[schedule."fake-a"]\npeak_windows_utc = ["01:00-04:00", "06:00-10:00"]\n'
        'peak_days = "mon-fri"'
    )
    config = _write_config(tmp_path, peak_windows=peak)
    samples = _write_samples(tmp_path)
    items_dir = _write_items(tmp_path)
    kwargs = _run_kwargs(tmp_path, config)
    kwargs["samples_dir"] = samples
    clock = FakeClock(datetime(2026, 10, 28, 2, 30, tzinfo=UTC))  # Wednesday, in peak
    invoker = _invoker_for(items_dir)

    run(
        "record",
        run_id="schedule",
        invoker=invoker,
        clock=clock.clock(),
        workers=1,
        **kwargs,
    )
    assert clock.sleeps
    assert clock.sleeps[0] == 5400.0
    budget = json.loads(
        (tmp_path / "runs" / "schedule" / "budget.json").read_text(encoding="utf-8")
    )
    assert any("off-peak" in wait for wait in budget["waits"])


def test_replay_run_is_byte_identical_twice(tmp_path: Path, repo_root: Path) -> None:
    fixture = repo_root / "tests" / "fixtures" / "bench"
    kwargs = {
        "config": fixture / "config",
        "models_path": fixture / "models.lock.toml",
        "tasks": [TaskName.ROUTE, TaskName.SCOPE, TaskName.REPLY, TaskName.DOCS],
        "recordings_dir": fixture / "recordings",
        "samples_dir": fixture / "samples",
        "items_dir": fixture / "items",
        "runs_dir": tmp_path / "runs",
        "reports_dir": tmp_path / "reports",
        "baseline_metrics": fixture / "baseline-metrics.json",
    }
    run(
        "replay",
        run_id="first",
        recommendations_path=tmp_path / "first.json",
        **kwargs,
    )
    run(
        "replay",
        run_id="second",
        recommendations_path=tmp_path / "second.json",
        **kwargs,
    )
    first = (tmp_path / "runs" / "first" / "cells.json").read_bytes()
    second = (tmp_path / "runs" / "second" / "cells.json").read_bytes()
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


def _cell(
    model_id: str, task: TaskName, primary: float, cost: float | None, latency: float
) -> CellResult:
    return CellResult(
        task=task,
        model=ModelRef(connection_id="c", model_id=model_id),
        n=10,
        primary_metric=primary,
        metrics={"schema_valid_rate": 1.0} if task is TaskName.REPLY else {},
        ci={},
        latency_p50=latency,
        latency_p95=latency,
        cost_per_1k=cost,
    )


def test_recommendations_good_enough_ties_and_unknown_cost() -> None:
    cells = [
        _cell("best", TaskName.ROUTE, 0.90, 1.0, 100.0),
        _cell("good-fast", TaskName.ROUTE, 0.88, 0.5, 50.0),
        _cell("poor", TaskName.ROUTE, 0.80, 0.1, 10.0),
        _cell("unknown-cost", TaskName.ROUTE, 0.88, None, 60.0),
        _cell("tie", TaskName.ROUTE, 0.88, 0.5, 50.0),
    ]
    recommendation = build_recommendations(
        [cell for cell in cells if cell.model.model_id != "tie"]
    )["tasks"]["T-ROUTE"]
    assert recommendation["cheapest"]["model"] == "good-fast"
    assert recommendation["most_accurate"]["model"] == "best"
    assert recommendation["fastest"]["model"] == "good-fast"
    labels = [row["model"] for row in recommendation["good_enough"]]
    assert "poor" not in labels

    tied = build_recommendations(
        [_cell("a", TaskName.ROUTE, 0.88, 0.5, 50.0), _cell("b", TaskName.ROUTE, 0.88, 0.5, 50.0)]
    )["tasks"]["T-ROUTE"]
    assert tied["most_accurate"]["model"] == "a"  # stable tie-break by label


def test_recommendations_include_baseline_zero_cost(tmp_path: Path) -> None:
    baseline_path = tmp_path / "metrics.json"
    baseline_path.write_text(
        json.dumps(
            {
                "route": {"macro_f1": 0.89, "accuracy": 0.89},
                "scope": {"f1": 0.5},
            }
        ),
        encoding="utf-8",
    )
    from which_ai_model.bench.score import baseline_candidates

    cells = [
        _cell("best", TaskName.ROUTE, 0.90, 1.0, 100.0),
        _cell("good", TaskName.ROUTE, 0.88, 0.5, 50.0),
    ]
    recommendation = build_recommendations(cells, baseline=baseline_candidates(baseline_path))[
        "tasks"
    ]["T-ROUTE"]
    assert recommendation["cheapest"]["baseline"] is True
    assert recommendation["cheapest"]["cost_per_1k"] == 0.0
    assert all(row["model"] != "tfidf-logreg" for row in [recommendation["fastest"]])


def test_budget_config_without_cap_and_strict_route_reduction() -> None:
    config = load_budget_config(Path("config/budget.toml"))
    assert config.cap_usd is None
    assert config.limits["opencode-go"].max_concurrency == 2
    assert config.limits["opencode-go"].requests_per_minute == 30
    assert config.window_ceilings["gemini-4-argon"] == 5.0
    # Captain order 2026-10-29: the DeepSeek off-peak schedule was removed for T10.
    # The parser is still covered by the fake-config peak-window test below.
    assert "deepseek-v4.1-flash" not in config.peak_windows
    assert reduce_route_items(
        [{"item_id": str(index), "intent": "a"} for index in range(8)], per_intent=5
    ) == [{"item_id": str(index), "intent": "a"} for index in range(5)]


def test_load_models_and_parse_response() -> None:
    specs = load_models(Path("tests/fixtures/bench/models.lock.toml"))
    assert [spec.ref.model_id for spec in specs] == ["fixture-accurate", "fixture-cheap"]

    from which_ai_model.contracts import RouteOut

    output, valid = parse_response(
        RouteOut, {"content": '{"intent": "card_arrival"}', "tool_calls": []}
    )
    assert valid
    assert output == {"intent": "card_arrival"}
    assert parse_response(RouteOut, {"content": "not json", "tool_calls": []}) == ({}, False)
    assert candidates_from_cells([]) == []
