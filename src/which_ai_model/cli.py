"""The `wam` command line.

Wave 0 ships the entry point and the command tree; later lanes own the command bodies.
A lane exposes its commands as a Typer sub-app in ``which_ai_model/<lane>/cli.py`` with a
module-level ``app`` object. This module discovers those sub-apps lazily, so a lane never
has to edit this file (and no lane collides with another in it).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Final

import typer

from which_ai_model import __version__

#: Lane sub-apps, in help order. Missing modules are skipped (Wave 0 has none yet).
LANE_CLI_MODULES: Final[tuple[str, ...]] = (
    "which_ai_model.connect.cli",
    "which_ai_model.core.cli",
    "which_ai_model.bench.cli",
    "which_ai_model.rag.cli",
    "which_ai_model.inventory.cli",
    "which_ai_model.tools.cli",
    "which_ai_model.gate.cli",
    "which_ai_model.report.cli",
    "which_ai_model.task.cli",
    "which_ai_model.recheck.cli",
)

app = typer.Typer(
    name="wam",
    help="Which AI model should each tool use — and how do you switch without rewriting everything?",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"which-ai-model {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Run a lane command; see `wam --help`."""


def _load_lane_apps() -> None:
    for module_name in LANE_CLI_MODULES:
        try:
            module: ModuleType = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        lane_app = getattr(module, "app", None)
        if not isinstance(lane_app, typer.Typer):
            continue
        top_level = getattr(module, "TOP_LEVEL_COMMANDS", None)
        if isinstance(top_level, dict):
            # A lane whose command belongs at the root (`wam doctor`) opts in.
            for command_name, callback in top_level.items():
                app.command(str(command_name))(callback)
            continue
        app.add_typer(lane_app, name=module_name.rsplit(".", 2)[-2])


@app.command()
def status() -> None:
    """Show what this scaffold knows about itself."""
    typer.echo(f"which-ai-model {__version__}")
    typer.echo("Wave 0 scaffold: contracts, Makefile, scanners. Lane commands arrive later.")


_load_lane_apps()

if __name__ == "__main__":  # pragma: no cover - convenience for `python -m`
    app()
