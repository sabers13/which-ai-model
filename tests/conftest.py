"""Shared pytest setup.

Every test runs with the network blocked. The block is switched on by the pytest options
in ``pyproject.toml`` (``--disable-socket --allow-unix-socket``); pytest-socket re-arms it
before every test, unlike a one-off call at import time. Unix sockets stay allowed so
local IPC — Postgres over a unix socket, subprocess pipes — still works. Tests that must
exercise a transport use fakes or recorded responses; only T10 makes live calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root (the worktree this test session runs in)."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory for committed test fixtures."""
    return REPO_ROOT / "tests" / "fixtures"
