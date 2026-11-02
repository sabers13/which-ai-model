#!/usr/bin/env python3
"""Rebuild ``results/`` deterministically from the committed live evidence (T10).

The live run is the only time this project makes paid calls. Its artifacts are kept
under ``recordings/_run/live/`` as immutable evidence; the per-response rows under
``recordings/<task>/`` are the replayable record. This script:

1. replays every recording into a throwaway run and checks that every model x task cell
   reproduces the live primary metrics (a missing recording fails here, never silently);
2. exports the live run (manifest, predictions, calls with real latency, cells, budget)
   into ``results/runs/live/`` byte for byte;
3. copies the secret-free doctor snapshot;
4. rebuilds ``results/recommendations.json`` from the exported cells and the baseline;
5. writes the deterministic trace sample, the two gate demonstrations and ``SUMMARY.md``;
6. writes ``results/MANIFEST.sha256`` over everything in ``results/``.

Nothing here reads ``.env`` or opens a socket. Re-running it after deleting ``results/``
produces the same bytes; that is the replay check in PLAN.md section 11.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from which_ai_model.bench.run import load_specs, rag_registry, run
from which_ai_model.bench.score import (
    SpecCellResult,
    baseline_candidates,
    build_recommendations_with_specs,
    load_cells,
    load_spec_cells,
    write_recommendations,
)
from which_ai_model.contracts import CellResult, RunManifest, TaskName, TaskSpec
from which_ai_model.gate import GateResult, evaluate_gate

ROOT = Path.cwd()
RESULTS = ROOT / "results"
LIVE_SOURCE = ROOT / "recordings" / "_run" / "live"
SPEC_RUN_ID = "live-v1.1"
SPEC_SOURCE = ROOT / "recordings" / "_run" / SPEC_RUN_ID
DOCTOR_SOURCE = ROOT / "recordings" / "_run" / "doctor.json"
BASELINE_METRICS = RESULTS / "baseline" / "metrics.json"
RUN_ID = "live"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


# --------------------------------------------------------------------------- replay


def verify_replay() -> None:
    """Replay every recording and require the live primary metrics to reproduce."""
    live_cells = load_cells(LIVE_SOURCE / "cells.json")
    with tempfile.TemporaryDirectory(prefix="wam-replay-") as tmp:
        work = Path(tmp)
        run(
            "replay",
            run_id="replay-check",
            registry=rag_registry(),
            runs_dir=work / "runs",
            reports_dir=work / "reports",
            recommendations_path=work / "recommendations.json",
            baseline_metrics=BASELINE_METRICS,
        )
        replay_cells = load_cells(work / "runs" / "replay-check" / "cells.json")
    live = {(cell.task, cell.model.connection_id, cell.model.model_id): cell for cell in live_cells}
    replayed = {
        (cell.task, cell.model.connection_id, cell.model.model_id): cell for cell in replay_cells
    }
    if set(live) != set(replayed):
        missing = sorted({(task.value, model) for task, _conn, model in set(live) - set(replayed)})
        extra = sorted({(task.value, model) for task, _conn, model in set(replayed) - set(live)})
        msg = f"replay cell set differs: missing={missing} extra={extra}"
        raise SystemExit(msg)
    for key, live_cell in sorted(live.items(), key=lambda item: item[0]):
        replay_cell = replayed[key]
        if live_cell.n != replay_cell.n or live_cell.primary_metric != replay_cell.primary_metric:
            msg = (
                f"replay differs for {key[0].value}/{key[2]}: "
                f"n {live_cell.n}->{replay_cell.n}, "
                f"primary {live_cell.primary_metric}->{replay_cell.primary_metric}"
            )
            raise SystemExit(msg)
    print(f"replay: {len(replay_cells)} cells reproduced from recordings")


# ------------------------------------------------------------------------ exports


def export_live() -> None:
    _copy_tree(LIVE_SOURCE, RESULTS / "runs" / RUN_ID)
    if DOCTOR_SOURCE.is_file():
        (RESULTS / "doctor.json").write_bytes(DOCTOR_SOURCE.read_bytes())


def _spec_ids_from_recordings() -> list[str]:
    """Spec ids with a committed recording directory and a committed spec file."""
    recorded: list[str] = []
    recordings = ROOT / "recordings"
    if not recordings.is_dir():
        return recorded
    for path in sorted(recordings.iterdir()):
        if not path.is_dir() or path.name.startswith("_") or path.name == "fresh":
            continue
        if (ROOT / "tasks" / f"{path.name}.toml").is_file():
            recorded.append(path.name)
    return recorded


def _recorded_span(spec_ids: Sequence[str]) -> tuple[str, str] | None:
    """The first and last ``recorded_at`` in the spec recordings, for a stable manifest."""
    stamps: list[str] = []
    for spec_id in spec_ids:
        directory = ROOT / "recordings" / spec_id
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                stamp = row.get("recorded_at") if isinstance(row, dict) else None
                if isinstance(stamp, str):
                    stamps.append(stamp)
    return (min(stamps), max(stamps)) if stamps else None


def _stabilise_manifest(path: Path, span: tuple[str, str]) -> None:
    document = _read_json(path)
    if not isinstance(document, dict):
        return
    document["started_at"], document["finished_at"] = span
    _write_json(path, document)


def export_spec_run() -> tuple[list[SpecCellResult], list[TaskSpec]]:
    """Export the v1.1 spec run, or replay its recordings when no evidence copy exists.

    The v1.1 run is committed under ``recordings/_run/live-v1.1`` exactly like the v1
    live run, so ``make all`` after deleting ``results/`` restores the same bytes. When
    only the raw spec recordings exist, a replay rebuilds the run and takes the manifest
    stamps from the rows, so the rebuild is still deterministic (latency is then 0, as
    in any replay).
    """
    target = RESULTS / "runs" / SPEC_RUN_ID
    if SPEC_SOURCE.is_dir():
        _copy_tree(SPEC_SOURCE, target)
    elif not (target / "cells.json").is_file():
        spec_ids = _spec_ids_from_recordings()
        if not spec_ids:
            return [], []
        with tempfile.TemporaryDirectory(prefix="wam-spec-replay-") as tmp:
            work = Path(tmp)
            run(
                "replay",
                run_id=SPEC_RUN_ID,
                tasks=tuple(spec_ids),
                recordings_dir=ROOT / "recordings",
                runs_dir=work / "runs",
                reports_dir=work / "reports",
                recommendations_path=work / "recommendations.json",
                baseline_metrics=BASELINE_METRICS,
                root=ROOT,
                tasks_dir=ROOT / "tasks",
            )
            _copy_tree(work / "runs" / SPEC_RUN_ID, target)
        span = _recorded_span(spec_ids)
        if span is not None:
            _stabilise_manifest(target / "manifest.json", span)
        print("results: spec run replayed from recordings (no recordings/_run copy)")
    cells_path = target / "cells.json"
    if not cells_path.is_file():
        return [], []
    spec_cells = load_spec_cells(cells_path)
    spec_ids_present = sorted({cell.task for cell in spec_cells})
    specs = load_specs(spec_ids_present, tasks_dir=ROOT / "tasks")
    return spec_cells, specs


def write_spec_gate_demo(spec_cells: Sequence[SpecCellResult], specs: Sequence[TaskSpec]) -> None:
    """The §A4 gate demonstration for T-DEFECT: the cheapest pick against the best model.

    Both cells come from the committed v1.1 run, so the pair is fixed by the results and
    the file is rebuilt deterministically from ``cells.json``. The outcome is recorded
    whatever it is: a cheapest model below the safety floor must FAIL, and saying so is
    the point of the demonstration.
    """
    from which_ai_model.gate import evaluate_spec_gate
    from which_ai_model.gate.spec import pick

    if not spec_cells or not specs:
        return
    spec = next((item for item in specs if item.id == "defect"), None)
    if spec is None:
        return
    rows = [cell for cell in spec_cells if cell.task == spec.id]
    if not rows:
        return
    by_id = {cell.model_id: cell for cell in rows}
    choice = pick(rows, floors=spec.gate.floors)
    most_accurate = (
        choice.most_accurate
        or min(rows, key=lambda cell: (-cell.primary_metric, cell.model_id)).model_id
    )
    cheapest = (
        choice.cheapest
        or min(
            rows,
            key=lambda cell: (
                cell.cost_per_1k is None,
                cell.cost_per_1k if cell.cost_per_1k is not None else 0.0,
                cell.model_id,
            ),
        ).model_id
    )
    selection = (
        "cheapest overall: no model met the safety floor"
        if choice.no_model_meets_floor
        else "cheapest good-enough model against the most accurate one"
    )
    result = evaluate_spec_gate(
        spec,
        {cheapest: by_id[most_accurate]},
        {cheapest: by_id[cheapest]},
        baseline_manifest=_read_json(RESULTS / "runs" / SPEC_RUN_ID / "manifest.json"),
        candidate_manifest=_read_json(RESULTS / "runs" / SPEC_RUN_ID / "manifest.json"),
    )
    payload = {
        "description": (
            "Regression gate on the T-DEFECT run (PLAN-v1.1 §A4): the cheapest model "
            "(candidate) against the most accurate one (baseline), both from the "
            "committed v1.1 run, with the spec's own thresholds and safety floor."
        ),
        "task": spec.id,
        "run": _rel(RESULTS / "runs" / SPEC_RUN_ID),
        "baseline_model": most_accurate,
        "candidate_model": cheapest,
        "selection": selection,
        "good_enough": list(choice.good_enough),
        "no_model_meets_floor": choice.no_model_meets_floor,
        "outcome": result.outcome.value,
        "rules": [rule.as_json() for rule in result.rules],
        "warnings": list(result.warnings),
        "reasons": list(result.reasons),
    }
    _write_json(RESULTS / "gate_demo_defect.json", payload)
    print(f"gate defect: {most_accurate} -> {cheapest} {result.outcome.value}")


def restore_page_screenshots() -> None:
    """Restore the committed page screenshots (T12 captures, not a build product).

    ``make all`` rebuilds every generated file, but a screenshot needs a browser and is
    captured once and committed as evidence. The evidence copy under
    ``recordings/_run/screenshots`` lets "delete ``results/`` and rebuild" restore them
    byte for byte, exactly like the live runs' evidence copies.
    """
    target = RESULTS / "page" / "screenshots"
    source = ROOT / "recordings" / "_run" / "screenshots"
    if target.is_dir() or not source.is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def export_recheck_demo() -> None:
    """Restore the committed §A6 re-check demo runs from their evidence copy.

    ``make recheck-demo`` in record mode records the fresh answers once and copies both
    run directories to ``recordings/_run/`` exactly like the live runs, so deleting
    ``results/`` and rebuilding restores the same bytes (including the wall-clock
    ``recorded_at`` stamps, which a replay would move).
    """
    for run_id in ("recheck-demo", "recheck-demo-accepted"):
        source = ROOT / "recordings" / "_run" / run_id
        if source.is_dir():
            _copy_tree(source, RESULTS / "runs" / run_id)


def rebuild_recommendations(
    spec_cells: Sequence[SpecCellResult] = (),
    specs: Sequence[TaskSpec] = (),
) -> None:
    cells = load_cells(RESULTS / "runs" / RUN_ID / "cells.json")
    payload = build_recommendations_with_specs(
        cells,
        spec_cells,
        specs,
        baseline=baseline_candidates(BASELINE_METRICS),
    )
    write_recommendations(RESULTS / "recommendations.json", payload)


def write_trace_sample(count: int = 50) -> None:
    rows = [
        json.loads(line)
        for line in (RESULTS / "runs" / RUN_ID / "calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) <= count:
        sample = rows
    else:
        step = (len(rows) - 1) / (count - 1)
        sample = [rows[round(index * step)] for index in range(count)]
    path = RESULTS / "traces_sample.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample), encoding="utf-8"
    )
    print(f"traces: {len(sample)} of {len(rows)} call rows")


# ---------------------------------------------------------------------------- gate


def _cheapest(cells: Sequence[CellResult], task: TaskName) -> CellResult | None:
    rows = [cell for cell in cells if cell.task is task]
    rows = [cell for cell in rows if cell.cost_per_1k is not None]
    return min(rows, key=lambda cell: (cell.cost_per_1k or 0.0, cell.model.model_id), default=None)


def write_gate_demo() -> None:
    """Cheapest model per task against the flagship, both from the live run."""
    cells = load_cells(RESULTS / "runs" / RUN_ID / "cells.json")
    run_path = RESULTS / "runs" / RUN_ID
    pairs: dict[tuple[str, str], GateResult] = {}
    chosen: dict[tuple[str, str], list[str]] = {}
    for task in TaskName:
        candidate = _cheapest(cells, task)
        if candidate is None:
            continue
        key = ("gemini-4-argon", candidate.model.model_id)
        chosen.setdefault(key, []).append(task.value)
        if key in pairs:
            continue
        pairs[key] = evaluate_gate(
            Path(_rel(run_path)), Path(_rel(run_path)), model_a=key[0], model_b=key[1]
        )
    pair_rows = []
    for (baseline, candidate), result in sorted(pairs.items()):
        pair_rows.append(
            {
                "baseline_model": baseline,
                "candidate_model": candidate,
                "chosen_for_tasks": sorted(chosen.get((baseline, candidate), [])),
                "outcome": result.outcome.value,
                "exit_code": result.exit_code,
                "reasons": list(result.reasons),
                "comparisons": [row.as_json() for row in result.comparisons],
            }
        )
    payload = {
        "description": (
            "Regression gate on the live run: for each task the cheapest model (candidate) "
            "against Gemini 4 Argon, the one flagship (baseline). The seven models are one "
            "per lab, so a per-lab cheapest-vs-flagship pair would compare a model with "
            "itself; the meaningful comparison is cheapest overall vs the flagship."
        ),
        "run": _rel(run_path),
        "pairs": pair_rows,
    }
    _write_json(RESULTS / "gate_demo.json", payload)
    print("gate: " + ", ".join(f"{row['candidate_model']}={row['outcome']}" for row in pair_rows))


def _fixture_run(work: Path, *, run_id: str) -> Path:
    """A deterministic 10-item replay run used only for the refusal demonstration."""
    from which_ai_model.bench.score import write_cells

    samples = work / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for name in ("route", "scope", "reply"):
        ids = (ROOT / "data" / "samples" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        (samples / f"{name}.txt").write_text(
            "\n".join([line.strip() for line in ids if line.strip()][:10]) + "\n",
            encoding="utf-8",
        )
    gold = (ROOT / "data" / "rag" / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    (work / "gold.jsonl").write_text("\n".join(gold[:10]) + "\n", encoding="utf-8")
    run(
        "replay",
        run_id=run_id,
        registry=rag_registry(),
        samples_dir=samples,
        docs_gold=work / "gold.jsonl",
        runs_dir=work / "runs",
        reports_dir=work / "reports",
        recommendations_path=work / "recommendations.json",
        baseline_metrics=BASELINE_METRICS,
    )
    source = work / "runs" / run_id
    target = RESULTS / "gate_refused" / ("a" if run_id.endswith("a") else "b")
    target.mkdir(parents=True, exist_ok=True)
    cells = load_cells(source / "cells.json")
    write_cells(target / "cells.json", cells)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    document = manifest.model_dump(mode="json")
    document["run_id"] = run_id
    document["started_at"] = "2026-10-29T00:00:00Z"
    document["finished_at"] = "2026-10-29T00:00:00Z"
    if run_id.endswith("b"):
        document["prompt_version"] = "v2-refusal-demo"
    _write_json(target / "manifest.json", document)
    return target


def write_gate_refused_demo() -> None:
    """Two 10-item runs that differ only in prompt version: the gate must refuse."""
    with tempfile.TemporaryDirectory(prefix="wam-gate-refused-") as tmp:
        work = Path(tmp)
        first = _fixture_run(work, run_id="gate-refused-a")
        second = _fixture_run(work, run_id="gate-refused-b")
        result = evaluate_gate(Path(_rel(first)), Path(_rel(second)))
    payload = {
        "description": (
            "Refusal demonstration: two 10-item fixture runs over the same recordings, "
            "where the candidate manifest names a different prompt version. The gate "
            "refuses before comparing metrics; a refusal is never a pass."
        ),
        "fixture_items_per_task": 10,
        "baseline": _rel(first),
        "candidate": _rel(second),
        "gate": result.as_json(),
    }
    _write_json(RESULTS / "gate_refused_demo.json", payload)
    print(f"gate refused demo: {result.outcome.value}")


# ------------------------------------------------------------------------- summary


def _fmt(value: float | None, digits: int = 4) -> str:
    return "unknown" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value * 100:.1f}%"


def _gap_text(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}%"


def _per_model_items(cells: Sequence[CellResult], task: TaskName) -> int:
    """The item count one model gets on ``task``; every model gets the same items."""
    counts = {cell.n for cell in cells if cell.task is task}
    if len(counts) != 1:
        msg = f"per-model item counts differ for {task.value}: {sorted(counts)}"
        raise SystemExit(msg)
    return counts.pop()


def write_summary() -> None:
    cells = load_cells(RESULTS / "runs" / RUN_ID / "cells.json")
    budget = _read_json(RESULTS / "runs" / RUN_ID / "budget.json")
    recommendations = _read_json(RESULTS / "recommendations.json")
    baseline = _read_json(BASELINE_METRICS)
    gate = _read_json(RESULTS / "gate_demo.json")
    inventory_before = _read_json(RESULTS / "inventory" / "before.json")
    inventory_after = _read_json(RESULTS / "inventory" / "after.json")
    manifest = _read_json(RESULTS / "runs" / RUN_ID / "manifest.json")

    by_task: dict[str, list[CellResult]] = {}
    for cell in cells:
        by_task.setdefault(cell.task.value, []).append(cell)
    lines: list[str] = [
        "# Results summary",
        "",
        "Every number below is read from a file in `results/`; the file is named next to it.",
        "The live run is the only one that made paid calls. `make all` replays the committed",
        "recordings and rebuilds this file byte for byte.",
        "",
        f"- Run date: {manifest['started_at']}",
        f"- Models: {len(manifest['models'])} (source: `results/runs/live/manifest.json`)",
        f"- Prompt/scorer versions: {manifest['prompt_version']} / {manifest['scorer_version']} "
        "(source: `results/runs/live/manifest.json`)",
        f"- Items per model: "
        f"{_per_model_items(cells, TaskName.ROUTE)} routing, "
        f"{_per_model_items(cells, TaskName.SCOPE)} scope, "
        f"{_per_model_items(cells, TaskName.REPLY)} reply, "
        f"{_per_model_items(cells, TaskName.DOCS)} docs "
        f"(7 models, {sum(cell.n for cell in cells):,} calls in total; "
        "source: `results/runs/live/cells.json`)",
        "",
        "Every cell also carries 95% intervals (2,000-resample bootstrap for accuracy and",
        "macro-F1, Wilson for proportions) in `results/runs/live/cells.json`; the tables",
        "below show point estimates only.",
        "",
        "## Headline numbers",
        "",
        f"- Total spend recorded by the runner: **${budget['total_spent_usd']:.2f}** "
        "(source: `results/runs/live/budget.json`)",
        f"- Gateway-measured spend where the gateway sent `usage.opencode.list_cost`: "
        f"**${budget['total_measured_covered_usd']:.2f}** over "
        f"{len(budget['measured_coverage_models'])} models "
        "(source: `results/runs/live/budget.json`)",
        f"- Price-table estimate for the same models: "
        f"**${budget['total_price_covered_usd']:.2f}**; measured gap "
        f"{_gap_text(budget['measured_gap_pct'])} "
        "(source: `results/runs/live/budget.json`)",
        f"- TF-IDF baseline: routing macro-F1 **{baseline['route']['macro_f1']:.4f}**, "
        f"scope F1 **{baseline['scope']['f1']:.4f}**, cost 0 "
        "(source: `results/baseline/metrics.json`)",
        f"- Inventory call sites: **{inventory_before['summary'].get('sites', 0)}** before, "
        f"**{inventory_after['summary'].get('sites', 0)}** after "
        "(source: `results/inventory/before.json`, `results/inventory/after.json`)",
        "",
        "## Per task",
        "",
    ]
    for task_name in ("T-ROUTE", "T-SCOPE", "T-REPLY", "T-DOCS"):
        rows = sorted(
            by_task.get(task_name, []), key=lambda cell: cell.primary_metric, reverse=True
        )
        if not rows:
            continue
        primary = {
            "T-ROUTE": "macro_f1",
            "T-SCOPE": "f1",
            "T-REPLY": "intent_correct",
            "T-DOCS": "hit_at_4",
        }[task_name]
        lines.extend(
            [
                f"### {task_name} — primary metric `{primary}`",
                "",
                "| Model | Primary | Cost / 1k items (USD) | Latency p50 (ms) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for cell in rows:
            lines.append(
                f"| {cell.model.model_id} | {cell.primary_metric:.4f} | "
                f"{_fmt(cell.cost_per_1k, 4)} | {_fmt(cell.latency_p50, 0)} |"
            )
        task_pick = recommendations["tasks"].get(task_name, {})
        for label in ("cheapest", "most_accurate", "fastest"):
            pick = task_pick.get(label)
            if pick:
                lines.append(
                    f"- {label}: **{pick['model']}** "
                    f"({_fmt(pick['primary_metric'])}, ${_fmt(pick['cost_per_1k'], 4)}/1k, "
                    f"p50 {_fmt(pick['latency_p50'], 0)} ms)"
                )
        lines.extend(
            ["", "(source: `results/runs/live/cells.json`, `results/recommendations.json`)", ""]
        )

    lines.extend(
        [
            "## Regression gate",
            "",
            "| Baseline (flagship) | Candidate (cheapest) | Outcome |",
            "| --- | --- | --- |",
        ]
    )
    for pair in gate["pairs"]:
        lines.append(
            f"| {pair['baseline_model']} | {pair['candidate_model']} | **{pair['outcome']}** |"
        )
    lines.extend(
        [
            "",
            "A `FAIL` is a real quality loss the gate blocks; a refusal is never a pass. "
            "The refusal demonstration (different prompt versions) is in "
            "`results/gate_refused_demo.json`.",
            "",
            "## Spend by model",
            "",
            "| Model | Runner spend (USD) | Measured (USD) | Price table (USD) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in sorted(budget["spent_usd_by_model"]):
        measured = budget["measured_usd_by_model"].get(model)
        price = budget["price_table_usd_by_model"].get(model)
        lines.append(
            f"| {model} | {budget['spent_usd_by_model'][model]:.6f} | "
            f"{'not sent' if measured is None else f'{measured:.6f}'} | "
            f"{'unknown' if price is None else f'{price:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "The gateway sends `usage.opencode.list_cost` on Chat Completions and Anthropic "
            "responses; the Responses and Gemini responses reached this project without it, "
            "so those rows fall back to the committed price table. Both are recorded and "
            "the gap above is computed only over models where both exist "
            "(source: `results/runs/live/budget.json`).",
            "",
            "## Limits",
            "",
            "- All seven models were reached through one gateway (OpenCode Go), each in its "
            "lab's own API format.",
            "- Tiers are mixed on purpose: Gemini 4 Argon is a flagship, the other six are "
            "small models.",
            "- Public datasets may be known to the models.",
            "- Prices retrieved 2026-10-29 from https://opencode.ai/docs/go/ "
            "(source: `config/prices.toml`).",
            "",
        ]
    )
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------ manifest


def write_manifest() -> None:
    lines: list[str] = []
    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {_rel(path)}")
    (RESULTS / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest: {len(lines)} files")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if "--manifest-only" in args:
        write_manifest()
        return 0
    if "--gate-only" in args:
        write_gate_demo()
        write_gate_refused_demo()
        return 0
    if not LIVE_SOURCE.is_dir():
        print(f"build_results: live evidence missing at {_rel(LIVE_SOURCE)}", file=sys.stderr)
        return 2
    verify_replay()
    export_live()
    restore_page_screenshots()
    spec_cells, specs = export_spec_run()
    rebuild_recommendations(spec_cells, specs)
    write_spec_gate_demo(spec_cells, specs)
    export_recheck_demo()
    write_trace_sample()
    write_gate_demo()
    write_gate_refused_demo()
    write_summary()
    write_manifest()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
