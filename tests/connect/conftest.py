"""Test setup for the connect lane: no ambient keys, no ``.env`` leakage."""

from __future__ import annotations

from pathlib import Path

import pytest

CREDENTIAL_VARS = (
    "OPENCODE_GO_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "CUSTOM_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test starts keyless and with an empty, temporary ``.env`` path."""
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WAM_ENV_FILE", str(tmp_path / "absent.env"))
