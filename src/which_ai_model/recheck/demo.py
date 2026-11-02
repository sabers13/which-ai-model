"""The §A6 re-check demo: one model, fresh answers, both settings outcomes recorded.

Record mode (M7, with the key) makes one set of calls and stores them under
``recordings/fresh/recheck-demo/``. Replay mode reads those committed recordings and
makes no network call. Either way two reports are written:

* ``results/runs/recheck-demo/recheck.json`` — no flag, expected REFUSED;
* ``results/runs/recheck-demo-accepted/recheck.json`` — accepted, normal verdict + WARN.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer

from which_ai_model.bench.run import BenchStopError, ReplayIncompleteError
from which_ai_model.recheck import run_recheck

DEMO_MODEL = "gpt-6-luna"
DEMO_TASK = "T-SCOPE"
DEMO_RUN_ID = "recheck-demo"
DEMO_ACCEPTED_RUN_ID = "recheck-demo-accepted"
DEMO_AGAINST = Path("results/runs/live")


def run_demo(
    *,
    mode: str | None = None,
    models: Path = Path("config/models.lock.toml"),
    against: Path = DEMO_AGAINST,
    fresh_root_dir: Path = Path("recordings/fresh"),
    runs_dir: Path = Path("results/runs"),
) -> tuple[Path, Path]:
    """Record (or replay) the demo once, then write the refused and accepted reports."""
    resolved_mode = (mode or os.environ.get("WAM_MODE") or "replay").lower()
    first = run_recheck(
        models=models,
        tasks=[DEMO_TASK],
        against=against,
        candidate=DEMO_MODEL,
        run_id=DEMO_RUN_ID,
        accept_settings_mismatch=False,
        mode=resolved_mode,
        fresh_root_dir=fresh_root_dir,
        runs_dir=runs_dir,
        temperature=0.0,
    )
    accepted = run_recheck(
        models=models,
        tasks=[DEMO_TASK],
        against=against,
        candidate=DEMO_MODEL,
        run_id=DEMO_ACCEPTED_RUN_ID,
        accept_settings_mismatch=True,
        mode="replay",
        fresh_run=DEMO_RUN_ID,
        fresh_root_dir=fresh_root_dir,
        runs_dir=runs_dir,
        temperature=0.0,
    )
    return first, accepted


def _summary(report_path: Path) -> tuple[str, list[str], list[str]]:
    document: Any = json.loads(report_path.read_text(encoding="utf-8"))
    gate = document.get("gate", {}) if isinstance(document, dict) else {}
    gate = gate if isinstance(gate, dict) else {}
    return (
        str(gate.get("outcome", "REFUSED")),
        [str(reason) for reason in gate.get("reasons", [])],
        [str(warning) for warning in gate.get("warnings", [])],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("record", "replay"),
        default=None,
        help="record makes the calls (M7); replay reads committed fresh recordings.",
    )
    args = parser.parse_args(argv)
    try:
        reports = run_demo(mode=args.mode)
    except (BenchStopError, ReplayIncompleteError, OSError, ValueError) as exc:
        typer.secho(f"recheck-demo: {exc}", err=True)
        return 2
    for report in reports:
        outcome, reasons, warnings = _summary(report)
        typer.echo(f"{outcome}: {report}")
        for reason in reasons:
            typer.echo(f"- {reason}")
        for warning in warnings:
            typer.echo(f"WARN: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m` entry point
    raise SystemExit(main())
