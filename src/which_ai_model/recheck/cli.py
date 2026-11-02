"""``wam recheck`` — record fresh answers and gate them against an earlier run (M4).

The lane exports a Typer sub-app; ``which_ai_model.cli`` mounts it as ``wam recheck``.
The verdict uses the ``wam gate`` exit codes: PASS 0, FAIL 1, REFUSED 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from which_ai_model.bench.run import BenchStopError, ReplayIncompleteError
from which_ai_model.recheck import run_recheck

app = typer.Typer(
    name="recheck",
    help="Record fresh answers, then gate them against an earlier run.",
    no_args_is_help=True,
)


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        msg = "no tasks selected"
        raise typer.BadParameter(msg)
    return tasks


def _gate_result(report_path: Path) -> tuple[str, int, list[str], list[str]]:
    document: Any = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        msg = f"{report_path}: recheck report must be a JSON object"
        raise ValueError(msg)
    gate = document.get("gate")
    if not isinstance(gate, dict):
        msg = f"{report_path}: recheck report has no gate block"
        raise ValueError(msg)
    return (
        str(gate.get("outcome", "REFUSED")),
        int(gate.get("exit_code", 2)),
        [str(reason) for reason in gate.get("reasons", [])],
        [str(warning) for warning in gate.get("warnings", [])],
    )


@app.command("run")
def run_command(
    models: Path = typer.Option(
        Path("config/models.lock.toml"), "--models", help="Model lock file."
    ),
    tasks: str = typer.Option(
        ..., "--tasks", help="Comma-separated task refs (built-in names or spec ids)."
    ),
    against: Path = typer.Option(
        ..., "--against", help="Earlier run directory, e.g. results/runs/live."
    ),
    candidate: str | None = typer.Option(
        None, "--candidate", help="Model id to gate as the candidate."
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Model id to gate against (default: the run's own pick)."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Fresh run directory name."),
    accept_settings_mismatch: bool = typer.Option(
        False,
        "--accept-settings-mismatch",
        help="Gate anyway when sampling settings differ, with a WARN per difference.",
    ),
) -> None:
    """Record into recordings/fresh/<run_id>/, score, gate, and write recheck.json."""
    try:
        report_path = run_recheck(
            models=models,
            tasks=_parse_tasks(tasks),
            against=against,
            candidate=candidate,
            baseline=baseline,
            run_id=run_id,
            accept_settings_mismatch=accept_settings_mismatch,
        )
        outcome, exit_code, reasons, warnings = _gate_result(report_path)
    except (BenchStopError, ReplayIncompleteError, FileNotFoundError) as exc:
        typer.secho(f"recheck: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError) as exc:
        typer.secho(f"recheck: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"{outcome}: {report_path}")
    for reason in reasons:
        typer.echo(f"- {reason}")
    for warning in warnings:
        typer.echo(f"WARN: {warning}")
    raise typer.Exit(code=exit_code)
