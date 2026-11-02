"""Cost at volume for a use-case card (``PLAN-v1.1`` §A8).

The arithmetic is deliberately small: one measured cost per 1,000 items, multiplied
linearly by the stated monthly volume. There are no savings claims, no volume tiers and
no other extrapolation.

``CardModelRow.cost_per_1k is None`` means the run could not measure a cost (the gateway
reported none and no price entry applied), so every row for that model stays ``None`` and
the card shows "unknown" instead of inventing a zero.
"""

from __future__ import annotations

from collections.abc import Sequence

from which_ai_model.contracts import CardData, VolumeRow

__all__ = ["DEFAULT_VOLUMES", "volume_rows"]

#: The three monthly volumes a card states (§A7 item 8).
DEFAULT_VOLUMES: tuple[int, ...] = (1_000, 10_000, 100_000)


def volume_rows(
    card_data: CardData,
    volumes: Sequence[int] = DEFAULT_VOLUMES,
) -> list[VolumeRow]:
    """One row per model and stated monthly volume, in card order then volume order.

    The cost-0 baseline is not repeated here: it is a separate row on the card and is
    never a model the prompt library recommends.
    """
    rows: list[VolumeRow] = []
    for model in card_data.models:
        for items_per_month in volumes:
            cost_usd = (
                None if model.cost_per_1k is None else model.cost_per_1k * items_per_month / 1000.0
            )
            rows.append(
                VolumeRow(
                    model_id=model.model_id,
                    items_per_month=items_per_month,
                    cost_usd=cost_usd,
                )
            )
    return rows
