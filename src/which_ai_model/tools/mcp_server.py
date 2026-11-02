"""The local MCP server: the two fictional-bank tools of L5, over stdio.

The legacy agent (``legacy/l5_agent.py``) carried these tools as hand-written JSON
schema and dispatched them itself. Here the same two tools are published once, by the
official ``mcp`` 2.x ``MCPServer``, and both the agent and any MCP client reach them the
same way. The data is the committed, clearly fictional fixtures the legacy tool reads.

Run it as a stdio server (what ``langchain.mcp.MCPAdapter`` does with a ``Path``):

    python -m which_ai_model.tools.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

__all__ = [
    "FIXTURE_DIR",
    "SERVER_NAME",
    "build_mcp_server",
    "fee_text",
    "main",
    "policy_text",
]

#: Repository root from ``src/which_ai_model/tools/mcp_server.py``.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: Same fixtures as the legacy L5 agent (``legacy/_common.py``).
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "legacy"
DEFAULT_FEES = FIXTURE_DIR / "fees.json"
DEFAULT_POLICIES = FIXTURE_DIR / "policies.json"

SERVER_NAME = "which-ai-model-fee-tools"
INSTRUCTIONS = (
    "Published fees and policies for a fictional example bank. Every number and policy "
    "in an answer must come from these two tools."
)


def _load(path: Path) -> dict[str, Any]:
    """Read one fixture table, or fail as a controllable tool error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolError(f"fixture {path.name} is not readable") from exc
    except ValueError as exc:
        raise ToolError(f"fixture {path.name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ToolError(f"fixture {path.name} is not a JSON object")
    return value


def fee_text(card_type: str, action: str, fees_path: Path | None = None) -> str:
    """The JSON text of one published fee; same keys and order as the legacy tool."""
    table = _load(fees_path or DEFAULT_FEES)
    fee = table.get("fees", {}).get(card_type, {}).get(action)
    if not isinstance(fee, int | float):
        raise ToolError("no published fee for that card type and action")
    return json.dumps(
        {
            "card_type": card_type,
            "action": action,
            "fee_usd": fee,
            "currency": table.get("currency", "USD"),
        }
    )


def policy_text(topic: str, policies_path: Path | None = None) -> str:
    """The JSON text of one published policy; same shape as the legacy tool."""
    table = _load(policies_path or DEFAULT_POLICIES)
    policy = table.get("policies", {}).get(topic)
    if not isinstance(policy, str):
        raise ToolError("no published policy for that topic")
    return json.dumps({"topic": topic, "policy": policy})


def build_mcp_server(
    *,
    fees_path: Path | None = None,
    policies_path: Path | None = None,
) -> MCPServer:
    """Build the stdio MCP server, optionally over other fixture files (tests do)."""
    fees = fees_path or DEFAULT_FEES
    policies = policies_path or DEFAULT_POLICIES
    server: MCPServer = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool()
    def get_fee(
        card_type: Literal["debit", "credit", "prepaid"],
        action: Literal[
            "cash_withdrawal",
            "foreign_transaction",
            "replacement_card",
            "late_payment",
        ],
    ) -> str:
        """Look up a published fee for a card type and action."""
        return fee_text(card_type, action, fees)

    @server.tool()
    def get_policy(
        topic: Literal["refunds", "lost_card", "disputes", "overdraft", "international_use"],
    ) -> str:
        """Look up a published policy by topic."""
        return policy_text(topic, policies)

    return server


def main() -> None:
    """Serve over stdio until the client closes the pipe."""
    build_mcp_server().run("stdio")


if __name__ == "__main__":
    main()
