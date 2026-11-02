"""``wam doctor`` — probe the configured connections and the benchmark model list.

The lane exports ``TOP_LEVEL_COMMANDS`` so the command is ``wam doctor``, not
``wam connect doctor``; T00's loader reads that mapping.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import typer

from which_ai_model.connect.doctor import collect_report, format_doctor, report_to_dict
from which_ai_model.connect.store import StoreError, load_connections

app = typer.Typer(
    name="doctor",
    help="Probe configured connections without inference and report model discovery.",
    no_args_is_help=False,
)


def _session_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"wam-doctor-{stamp}-{os.getpid()}"


@app.command("doctor")
def doctor_command(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    config: Path = typer.Option(
        Path("config/providers.toml"), "--config", help="Providers file to load."
    ),
    cache: Path = typer.Option(
        Path("config/models.catalog.json"), "--cache", help="Where discovery results are cached."
    ),
    timeout_s: float = typer.Option(20.0, "--timeout", help="Per-request timeout in seconds."),
) -> None:
    """Probe each connection, then list models once; never makes an inference call."""
    try:
        connections = load_connections(config)
    except StoreError as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=2) from None
    report = collect_report(
        connections, timeout_s=timeout_s, session_id=_session_id(), cache_path=cache
    )
    if json_output:
        typer.echo(json.dumps(report_to_dict(report), indent=2))
    else:
        typer.echo(format_doctor(report))


#: Mounted at the root by ``cli.py``: ``wam doctor``.
TOP_LEVEL_COMMANDS = {"doctor": doctor_command}
