"""Cost: price table when known, ``None`` when not, gateway measurement wins."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from which_ai_model.contracts import ModelRef
from which_ai_model.core.cost import cost_usd, load_prices, usage_cost_usd

PRICES = """
[models."deepseek-v4.1-flash"]
input_per_mtok = 0.15
output_per_mtok = 0.60
currency = "USD"
source_url = "https://opencode.ai/docs/go/"
retrieved = "2026-10-28"

[models."eur-model"]
input_per_mtok = 1.0
output_per_mtok = 2.0
currency = "EUR"
"""


def _write_prices(tmp_path: Path) -> Path:
    path = tmp_path / "prices.toml"
    path.write_text(PRICES, encoding="utf-8")
    return path


def test_known_price_is_computed_per_million(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost_usd("deepseek-v4.1-flash", 1_000_000, 1_000_000, prices) == 0.75
    assert cost_usd("deepseek-v4.1-flash", 1_000, 2_000, prices) == 0.00135


def test_unknown_model_or_missing_file_is_none(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost_usd("not-listed", 100, 100, prices) is None
    assert cost_usd("deepseek-v4.1-flash", None, 100, prices) is None
    assert cost_usd("deepseek-v4.1-flash", 100, 100, tmp_path / "missing.toml") is None
    assert load_prices(tmp_path / "missing.toml") == {}


def test_non_usd_currency_is_not_guessed(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost_usd("eur-model", 1_000_000, 0, prices) is None


def test_gateway_measurement_beats_the_table(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    model = ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash")
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000},
        response_metadata={"token_usage": {"opencode": {"list_cost": 0.42}}},
    )
    assert usage_cost_usd(message, model, prices=prices) == 0.42

    table_only = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000},
    )
    assert usage_cost_usd(table_only, model, prices=prices) == 0.15
