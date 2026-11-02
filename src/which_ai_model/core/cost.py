"""Cost from the committed price table, with ``None`` for anything unknown.

``config/prices.toml`` is written by T10 from the provider's published prices::

    [models."deepseek-v4.1-flash"]
    input_per_mtok = 0.15
    output_per_mtok = 0.60
    currency = "USD"
    source_url = "https://opencode.ai/docs/go/"
    retrieved = "2026-10-28"

The gateway's measured ``usage.opencode.list_cost`` is always preferred (see
:func:`usage_cost_usd`). A model without a verified price is ``None``, never a guess
(``PLAN.md`` §7).
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from which_ai_model.contracts import ModelRef

from .usage import extract_usage

__all__ = [
    "DEFAULT_PRICES_PATH",
    "Price",
    "cost_usd",
    "load_prices",
    "usage_cost_usd",
]

DEFAULT_PRICES_PATH = Path("config/prices.toml")


@dataclass(frozen=True)
class Price:
    """Published price per million tokens, in USD."""

    input_per_mtok: float
    output_per_mtok: float
    currency: str = "USD"
    source_url: str = ""
    retrieved: str = ""


def _price_from_row(row: Mapping[str, Any]) -> Price | None:
    input_price = row.get("input_per_mtok")
    output_price = row.get("output_per_mtok")
    if not isinstance(input_price, (int, float)) or not isinstance(output_price, (int, float)):
        return None
    if input_price < 0 or output_price < 0:
        return None
    currency = row.get("currency", "USD")
    return Price(
        input_per_mtok=float(input_price),
        output_per_mtok=float(output_price),
        currency=currency if isinstance(currency, str) else "USD",
        source_url=_as_str(row.get("source_url")),
        retrieved=_as_str(row.get("retrieved")),
    )


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def load_prices(path: Path | str = DEFAULT_PRICES_PATH) -> dict[str, Price]:
    """Load ``[models."<id>"]`` rows. A missing or unreadable file is empty."""
    price_path = Path(path)
    if not price_path.is_file():
        return {}
    with price_path.open("rb") as handle:
        data = tomllib.load(handle)
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    prices: dict[str, Price] = {}
    for model_id, row in models.items():
        if not isinstance(model_id, str) or not isinstance(row, dict):
            continue
        price = _price_from_row(row)
        if price is not None:
            prices[model_id] = price
    return prices


def cost_usd(
    model_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    prices: Mapping[str, Price] | Path | str | None = None,
) -> float | None:
    """Price one call from the table, or ``None`` when tokens or price are unknown."""
    if input_tokens is None or output_tokens is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    table: Mapping[str, Price]
    if prices is None:
        table = load_prices()
    elif isinstance(prices, (str, Path)):
        table = load_prices(prices)
    else:
        table = prices
    price = table.get(model_id)
    if price is None or price.currency != "USD":
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


def usage_cost_usd(
    usage: Any,
    model: ModelRef,
    *,
    prices: Mapping[str, Price] | Path | str | None = None,
) -> float | None:
    """Measured gateway cost when present, else the price table, else ``None``.

    ``usage`` may be an ``AIMessage``, a recording row's ``response`` or a raw usage
    dict; :func:`which_ai_model.core.usage.extract_usage` normalises all three.
    """
    measured = extract_usage(usage)
    if measured.measured_cost_usd is not None:
        return measured.measured_cost_usd
    return cost_usd(
        model.model_id,
        measured.input_tokens,
        measured.output_tokens,
        prices,
    )
