"""The client-side cutoff filters recorded scores without changing the benchmark."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_score_picker_against_recorded_results() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to check the browser's pure selection function")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, "tests/report/score_picker_checks.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
