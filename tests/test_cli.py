"""The `wam` entry point exists and reports the scaffold version."""

from __future__ import annotations

from typer.testing import CliRunner

from which_ai_model import __version__
from which_ai_model.cli import app


def test_status_command() -> None:
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"which-ai-model {__version__}"
