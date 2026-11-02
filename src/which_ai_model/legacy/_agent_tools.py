"""The two fictional-bank tools used by the legacy L5 agent.

Kept in their own module so the loop in ``l5_agent.py`` stays small. Both tools read
committed, clearly fictional fixtures; there is no bank or customer data here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fee",
            "description": "Look up a published fee for a card type and action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {"type": "string", "enum": ["debit", "credit", "prepaid"]},
                    "action": {
                        "type": "string",
                        "enum": [
                            "cash_withdrawal",
                            "foreign_transaction",
                            "replacement_card",
                            "late_payment",
                        ],
                    },
                },
                "required": ["card_type", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Look up a published policy by topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "refunds",
                            "lost_card",
                            "disputes",
                            "overdraft",
                            "international_use",
                        ],
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def execute_tool(name: str, arguments: dict[str, Any], fees_path: Path, policies_path: Path) -> str:
    """Run one permitted tool and return its JSON text, or raise ``ValueError``."""
    if name == "get_fee":
        card_type = arguments.get("card_type")
        action = arguments.get("action")
        if not isinstance(card_type, str) or not isinstance(action, str):
            raise ValueError("get_fee needs string card_type and action")
        table = _load(fees_path)
        fee = table.get("fees", {}).get(card_type, {}).get(action)
        if not isinstance(fee, int | float):
            raise ValueError("no published fee for that card type and action")
        return json.dumps(
            {
                "card_type": card_type,
                "action": action,
                "fee_usd": fee,
                "currency": table.get("currency", "USD"),
            }
        )
    if name == "get_policy":
        topic = arguments.get("topic")
        if not isinstance(topic, str):
            raise ValueError("get_policy needs a string topic")
        policy = _load(policies_path).get("policies", {}).get(topic)
        if not isinstance(policy, str):
            raise ValueError("no published policy for that topic")
        return json.dumps({"topic": topic, "policy": policy})
    raise ValueError("tool is not permitted")
