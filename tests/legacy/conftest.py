"""Fixtures for the legacy ("before") tool tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"

#: Every legacy tool reads its own env vars; tests must never inherit a real one.
LEGACY_ENV = (
    "ROUTER_KEY",
    "ROUTER_MODEL",
    "ROUTER_BASE_URL",
    "ANTHROPIC_API_KEY",
    "DRAFTER_MODEL",
    "DRAFTER_BASE_URL",
    "GEMINI_API_KEY",
    "SCOPE_MODEL",
    "SCOPE_BASE_URL",
    "DOCS_KEY",
    "DOCS_MODEL",
    "DOCS_BASE_URL",
    "DOCS_GUIDE_PATH",
    "AGENT_KEY",
    "AGENT_MODEL",
    "AGENT_BASE_URL",
    "AGENT_MAX_TURNS",
)


@pytest.fixture(autouse=True)
def clean_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every legacy variable so a developer shell cannot change a test."""
    for name in LEGACY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def parity_items() -> list[dict[str, Any]]:
    """The 50 committed parity items, parsed from JSONL."""
    lines = (FIXTURE_DIR / "parity_items.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR
