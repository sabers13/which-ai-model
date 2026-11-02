"""L5 — tool-calling agent ("before").

Adapted from SaberOps ``src/saberops/workers/api_agent.py`` at commit ``72e04bd``
(https://github.com/sabers13/SaberOps/blob/72e04bd/src/saberops/workers/api_agent.py),
Apache-2.0, same author; see ``THIRD_PARTY.md``. The hand-written loop is kept: manual
``tools`` JSON, a hard turn limit, redirects refused and one tool message per call. Its
coding tools are replaced by two fictional-bank tools reading committed fixtures.
Configuration comes from ``AGENT_BASE_URL``, ``AGENT_MODEL`` and ``AGENT_KEY``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from which_ai_model.legacy._agent_tools import TOOL_DEFINITIONS, execute_tool
from which_ai_model.legacy._common import (
    LEGACY_FIXTURES,
    LegacyResult,
    elapsed_ms,
    env,
    error_text,
    result_dict,
)

DEFAULT_MAX_TURNS = 12
_PROMPT = (
    "Answer the customer's fee and policy question. Use the tools for every number and "
    "policy; never guess. Finish with one short paragraph that names the fee in USD."
)


def run_agent(
    question: str,
    *,
    transport: httpx.BaseTransport | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_turns: int | None = None,
    fees_path: Path | None = None,
    policies_path: Path | None = None,
    timeout_s: float = 60.0,
) -> LegacyResult:
    """Hand-written chat-completions loop with the two fee tools."""
    started = time.perf_counter()
    key = api_key if api_key is not None else env("AGENT_KEY")
    if not key:
        return LegacyResult("", [], elapsed_ms(started), "AGENT_KEY is not set")
    limit = max_turns or int(env("AGENT_MAX_TURNS", str(DEFAULT_MAX_TURNS)) or DEFAULT_MAX_TURNS)
    fees = fees_path or LEGACY_FIXTURES / "fees.json"
    policies = policies_path or LEGACY_FIXTURES / "policies.json"
    root = (base_url or env("AGENT_BASE_URL", "http://localhost:8080/v1") or "").rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": question},
    ]
    recorded: list[dict[str, Any]] = []
    try:
        with httpx.Client(transport=transport, timeout=timeout_s, follow_redirects=False) as client:
            for _ in range(limit):
                body = {
                    "model": model or env("AGENT_MODEL", "agent-small"),
                    "messages": messages,
                    "tools": TOOL_DEFINITIONS,
                    "stream": False,
                }
                response = client.post(f"{root}/chat/completions", headers=headers, json=body)
                if 300 <= response.status_code < 400:
                    return LegacyResult("", recorded, elapsed_ms(started), "API redirect refused")
                if response.status_code >= 400:
                    return LegacyResult(
                        "",
                        recorded,
                        elapsed_ms(started),
                        f"API request failed (HTTP {response.status_code})",
                    )
                data: Any = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if not isinstance(content, str):
                    return LegacyResult("", recorded, elapsed_ms(started), "unsupported content")
                calls = message.get("tool_calls")
                if not calls:
                    if choice.get("finish_reason") != "stop":
                        return LegacyResult(
                            "",
                            recorded,
                            elapsed_ms(started),
                            "API completion ended without a successful stop",
                        )
                    return LegacyResult(content, recorded, elapsed_ms(started), None)
                if not isinstance(calls, list) or len(calls) > 16:
                    return LegacyResult("", recorded, elapsed_ms(started), "too many tool calls")
                messages.append({"role": "assistant", "content": content, "tool_calls": calls})
                for call in calls:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                        if not isinstance(arguments, dict):
                            raise ValueError("tool arguments are not an object")
                        text = execute_tool(name, arguments, fees, policies)
                    except (OSError, ValueError):
                        text = "Tool failed or was denied. Check the permitted arguments."
                        arguments = {}
                    recorded.append({"name": name, "arguments": arguments, "result": text})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        return LegacyResult("", recorded, elapsed_ms(started), error_text(exc, key))
    return LegacyResult(
        "", recorded, elapsed_ms(started), f"API tool loop reached its {limit}-turn limit"
    )


def run(question: str, **kwargs: Any) -> dict[str, Any]:
    """``run(input) -> dict`` with the legacy fields; tool calls are recorded."""
    return result_dict(run_agent(question, **kwargs))
