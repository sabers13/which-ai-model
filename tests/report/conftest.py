"""Shared fixture for the page lane."""

from __future__ import annotations

from typing import Any

import pytest

from which_ai_model.report.page_data import build_page_data


@pytest.fixture(scope="session")
def page() -> dict[str, Any]:
    return build_page_data()
