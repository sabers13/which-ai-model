"""``wam bench`` — the benchmark runner CLI (T08).

`wam bench --mode replay --models config/models.lock.toml --tasks T-ROUTE,T-SCOPE`
iterates every model x task x item, reuses recordings, and writes
``results/runs/<run_id>/{manifest.json,predictions/*.jsonl,calls.jsonl,cells.json}``
plus ``results/recommendations.json``.

The fixture flags (``--items``, ``--recordings``, ``--samples``, ``--config``) exist so
``make bench-replay`` runs end to end on the tiny committed fixture set without the
downloaded datasets. T10 uses the defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from which_ai_model.bench.run import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DOCS_GOLD,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SAMPLES_DIR,
    BenchStopError,
    ReplayIncompleteError,
    rag_registry,
    run,
)
from which_ai_model.contracts import SPEC_ID_PATTERN, TaskName, builtin_task

#: An empty sub-app so T00's loader does not skip this module; the root command is in
#: ``TOP_LEVEL_COMMANDS`` below.
app = typer.Typer(
    name="bench",
    help="Benchmark runner: replay or record every model x task x item.",
    no_args_is_help=False,
)


def _parse_tasks(value: str) -> tuple[str, ...]:
    """Accept built-in task names and spec ids (``--tasks defect``)."""
    tasks: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if builtin_task(item) is None and not re.fullmatch(SPEC_ID_PATTERN, item):
            allowed = ", ".join(task.value for task in TaskName)
            msg = f"unknown task {item!r}; choose from {allowed} or a spec id such as 'defect'"
            raise typer.BadParameter(msg)
        if item not in tasks:
            tasks.append(item)
    if not tasks:
        msg = "no tasks selected"
        raise typer.BadParameter(msg)
    return tuple(tasks)


def bench_command(
    mode: str = typer.Option("replay", "--mode", help="replay or record."),
    models: Path = typer.Option(
        Path("config/models.lock.toml"), "--models", help="Model lock file."
    ),
    tasks: str = typer.Option(
        "T-ROUTE,T-SCOPE,T-REPLY,T-DOCS",
        "--tasks",
        help="Comma-separated task names.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run directory name."),
    config: Path = typer.Option(Path("config"), "--config", help="Config directory."),
    recordings: Path = typer.Option(DEFAULT_RECORDINGS_DIR, "--recordings", help="Recording root."),
    samples: Path = typer.Option(DEFAULT_SAMPLES_DIR, "--samples", help="Sample id directory."),
    items: Path | None = typer.Option(
        None, "--items", help="Fixture item JSONL directory (instead of datasets)."
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root", help="Dataset directory."),
    docs_gold: Path = typer.Option(DEFAULT_DOCS_GOLD, "--docs-gold", help="T-DOCS gold file."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Run output root."),
    reports_dir: Path = typer.Option(Path("reports"), "--reports-dir", help="STOP/DEVIATIONS."),
    recommendations: Path = typer.Option(
        Path("results/recommendations.json"),
        "--recommendations",
        help="Recommendations output.",
    ),
    baseline: Path = typer.Option(
        Path("results/baseline/metrics.json"),
        "--baseline",
        help="TF-IDF baseline metrics to include in recommendations.",
    ),
    workers: int | None = typer.Option(
        None, "--workers", help="Per-cell worker count (default: budget concurrency)."
    ),
    reduce_per_intent: int = typer.Option(
        5, "--reduce-per-intent", help="T-ROUTE reduction size when the budget guard fires."
    ),
    rag_docs: bool = typer.Option(
        False,
        "--rag-docs",
        help="Use the real T-DOCS retrieval predictor (record/replay of the live run).",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Record into recordings/fresh/<run_id>/ and never reuse a saved answer (M4).",
    ),
    no_peak_wait: bool = typer.Option(
        False,
        "--no-peak-wait",
        help="Do not wait out the v1 off-peak windows (the v1.1 run, PLAN-v1.1 §A10).",
    ),
    cap_usd: float | None = typer.Option(
        None,
        "--cap-usd",
        help="Hard USD cap for this run (overrides config/budget.toml; the v1.1 run).",
    ),
) -> None:
    """Run every model x task x item and score the result."""
    try:
        manifest = run(
            mode=mode,
            config=config,
            run_id=run_id,
            models_path=models,
            tasks=_parse_tasks(tasks),
            recordings_dir=recordings,
            samples_dir=samples,
            data_root=data_root,
            items_dir=items,
            docs_gold=docs_gold,
            runs_dir=runs_dir,
            reports_dir=reports_dir,
            recommendations_path=recommendations,
            baseline_metrics=baseline,
            registry=rag_registry() if rag_docs else None,
            workers=workers,
            reduce_per_intent=reduce_per_intent,
            fresh=fresh,
            peak_wait=not no_peak_wait,
            cap_usd=cap_usd,
        )
    except (BenchStopError, ReplayIncompleteError) as exc:
        typer.secho(f"bench: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"bench: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"bench: {manifest.run_id} {manifest.prompt_version}/{manifest.scorer_version} "
        f"{len(manifest.models)} model(s)"
    )
    typer.echo(f"bench: wrote {runs_dir / manifest.run_id}")


#: Mounted at the root by ``cli.py``: ``wam bench``.
TOP_LEVEL_COMMANDS = {"bench": bench_command}
