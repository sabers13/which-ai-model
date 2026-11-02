"""The facts checker accepts a published rounded value, never a truncated one (M11, S6)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_facts.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_facts", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_published_value_must_be_the_rounded_form() -> None:
    module = _module()
    body = '{"total_spent_usd": 0.8936682499999982}'
    assert module._rounded_match("0.8937", body) is True
    assert module._rounded_match("0.8936", body) is False


def test_a_six_decimal_value_must_also_be_rounded() -> None:
    module = _module()
    body = '{"total_spent_usd": 0.8936682499999982}'
    assert module._rounded_match("0.893668", body) is True
    assert module._rounded_match("0.893669", body) is False


def test_non_decimal_needles_are_not_rounded() -> None:
    module = _module()
    assert module._rounded_match("29 October", "29 October 2026") is False
