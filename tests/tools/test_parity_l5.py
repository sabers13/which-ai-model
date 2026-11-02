"""T07b parity: the LangChain agent + MCP server against the legacy L5 loop.

Ten committed multi-step fixture questions are replayed through the new path with a
scripted model that emits exactly the tool calls the legacy fixture recorded. The run
must produce the same tool-call sequence, the same tool results (the MCP server reads
the same fixtures) and the same final numbers, now delivered as the ``AgentOut``
schema. Everything is in-process: the MCP server is passed to ``MCPAdapter`` as a
server object, so no subprocess, pipe or socket is opened.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ConfigDict, Field

from which_ai_model.contracts import Connection, ModelRef, Protocol
from which_ai_model.legacy._agent_tools import TOOL_DEFINITIONS, execute_tool
from which_ai_model.tools.agent import (
    DEFAULT_MAX_TURNS,
    AgentOut,
    AgentTurnLimitError,
    run_agent,
    trace_call_for,
)
from which_ai_model.tools.mcp_server import build_mcp_server, fee_text, policy_text

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"
ITEMS = [
    json.loads(line)
    for line in (FIXTURE_DIR / "parity_items.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
#: The ten fixture questions whose legacy run made at least one tool call.
MULTI_STEP = [item for item in ITEMS if item["responses"]["l5"]["tool_calls"]]
#: ToolStrategy names the structured-output tool after the schema class.
SCHEMA_TOOL = AgentOut.__name__

MODEL = ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash")
FALLBACK = ModelRef(connection_id="opencode-go-anthropic", model_id="deepseek-v4.1-flash")
CONNECTIONS = {
    "opencode-go-openai": Connection(
        id="opencode-go-openai",
        provider="opencode-go",
        protocol=Protocol.OPENAI_COMPATIBLE,
        base_url="https://opencode.ai/inference/go/openai/v1",
        credential_ref="env:OPENCODE_GO_API_KEY",
        preset="opencode-go",
    ),
    "opencode-go-anthropic": Connection(
        id="opencode-go-anthropic",
        provider="opencode-go",
        protocol=Protocol.ANTHROPIC,
        base_url="https://opencode.ai/inference/go/anthropic",
        credential_ref="env:OPENCODE_GO_API_KEY",
        preset="opencode-go",
    ),
}


class ScriptedToolModel(BaseChatModel):
    """Return the next scripted message; keep every message list it was given."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolModel:
        del kwargs
        for tool in tools:
            self.bound.append(str(getattr(tool, "name", tool)))
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.calls.append(list(messages))
        if not self.script:
            msg = "script exhausted"
            raise AssertionError(msg)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


class FakeBuilder:
    """Stand-in for ``connect.to_langchain.chat_model``; never reads a credential."""

    def __init__(self, models: dict[str, BaseChatModel]) -> None:
        self.models = models
        self.calls: list[tuple[Connection, str]] = []

    def __call__(self, connection: Connection, model_id: str, **options: Any) -> BaseChatModel:
        del options
        self.calls.append((connection, model_id))
        return self.models[connection.id]


def legacy_tool_text(name: str, arguments: dict[str, Any]) -> str:
    return execute_tool(name, arguments, FIXTURE_DIR / "fees.json", FIXTURE_DIR / "policies.json")


def expected_fees(calls: list[dict[str, Any]]) -> list[float]:
    """The USD amounts behind the legacy answer, in call order."""
    fees: list[float] = []
    for call in calls:
        if call["name"] == "get_fee":
            fees.append(json.loads(legacy_tool_text(call["name"], call["arguments"]))["fee_usd"])
    return fees


def script_for(item: dict[str, Any]) -> list[AIMessage]:
    """The tool calls the fixture recorded, then the schema answer it ended with."""
    response = item["responses"]["l5"]
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": call["name"],
                    "args": call["arguments"],
                    "id": f"call_{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index, call in enumerate(response["tool_calls"])
    ]
    script.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": SCHEMA_TOOL,
                    "args": {
                        "answer": response["content"],
                        "fees_used": expected_fees(response["tool_calls"]),
                    },
                    "id": "call_final",
                    "type": "tool_call",
                }
            ],
        )
    )
    return script


def tool_turns(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
    """Every non-schema tool call the model asked for, in order."""
    turns: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["name"] != SCHEMA_TOOL:
                turns.append((call["name"], call["args"]))
    return turns


def tool_results(messages: list[BaseMessage]) -> list[str]:
    """The text of every MCP tool result, in order."""
    texts: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.content, list):
            texts.extend(
                block["text"]
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif isinstance(message, ToolMessage) and isinstance(message.content, str):
            texts.append(message.content)
    return texts


def fixture_item(item_id: str) -> dict[str, Any]:
    return next(item for item in ITEMS if item["item_id"] == item_id)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("item", MULTI_STEP, ids=[item["item_id"] for item in MULTI_STEP])
def test_ten_multi_step_questions_match_the_legacy_loop(item: dict[str, Any]) -> None:
    model = ScriptedToolModel(script=script_for(item))
    expected_calls = item["responses"]["l5"]["tool_calls"]

    result = run_agent(item["message"], MODEL, model=model)

    assert isinstance(result, AgentOut)
    assert result.answer == item["responses"]["l5"]["content"]
    assert result.fees_used == expected_fees(expected_calls)
    final = model.calls[-1]
    assert tool_turns(final) == [(c["name"], c["arguments"]) for c in expected_calls]
    assert tool_results(final) == [
        legacy_tool_text(c["name"], c["arguments"]) for c in expected_calls
    ]


def test_the_ten_questions_actually_need_a_tool() -> None:
    assert len(MULTI_STEP) == 10
    assert all(len(item["responses"]["l5"]["tool_calls"]) >= 1 for item in MULTI_STEP)
    assert sum(1 for item in MULTI_STEP if len(item["responses"]["l5"]["tool_calls"]) > 1) >= 3


def test_a_question_without_tools_still_returns_the_schema() -> None:
    item = next(item for item in ITEMS if not item["responses"]["l5"]["tool_calls"])
    model = ScriptedToolModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": SCHEMA_TOOL,
                        "args": {"answer": item["responses"]["l5"]["content"], "fees_used": []},
                        "id": "call_final",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    result = run_agent(item["message"], MODEL, model=model)

    assert result is not None
    assert result.answer == item["responses"]["l5"]["content"]
    assert result.fees_used == []
    assert tool_turns(model.calls[-1]) == []


# ------------------------------------------------------- the shared-layer model path


def test_model_ref_runs_through_core_with_a_fallback() -> None:
    item = fixture_item("legacy-02")
    primary = ScriptedToolModel(script=script_for(item))
    fallback = ScriptedToolModel(script=[])
    builder = FakeBuilder({"opencode-go-openai": primary, "opencode-go-anthropic": fallback})

    result = run_agent(
        item["message"],
        MODEL,
        fallback=FALLBACK,
        connections=CONNECTIONS,
        builder=builder,
    )

    assert result is not None
    assert result.answer == item["responses"]["l5"]["content"]
    assert [connection.id for connection, _ in builder.calls] == [
        "opencode-go-openai",
        "opencode-go-anthropic",
    ]
    assert {model_id for _, model_id in builder.calls} == {"deepseek-v4.1-flash"}


def test_a_ready_model_cannot_take_a_fallback() -> None:
    model = ScriptedToolModel(script=[])

    with pytest.raises(ValueError, match="fallback belongs to a built model"):
        run_agent("anything", MODEL, model=model, fallback=FALLBACK)


def test_zero_turns_is_rejected_before_any_call() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        run_agent("anything", MODEL, model=ScriptedToolModel(script=[]), max_turns=0)


def looping_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_policy", "args": {"topic": "overdraft"}, "id": "c", "type": "tool_call"}
        ],
    )


def test_the_turn_limit_matches_the_legacy_default() -> None:
    small = ScriptedToolModel(script=[looping_call() for _ in range(20)])
    with pytest.raises(AgentTurnLimitError, match="2-turn limit"):
        run_agent("loop", MODEL, model=small, max_turns=2)
    assert len(small.calls) == 2

    legacy_limit = ScriptedToolModel(script=[looping_call() for _ in range(20)])
    with pytest.raises(AgentTurnLimitError, match="12-turn limit"):
        run_agent("loop", MODEL, model=legacy_limit)
    assert len(legacy_limit.calls) == DEFAULT_MAX_TURNS == 12


def test_a_plain_text_finish_is_not_a_schema_answer() -> None:
    model = ScriptedToolModel(script=[AIMessage(content="USD 2.50 without a tool call")])

    assert run_agent("What does a debit cash withdrawal cost?", MODEL, model=model) is None


def test_the_run_is_traced_without_prompt_text(tmp_path: Path) -> None:
    item = fixture_item("legacy-01")
    script = script_for(item)
    script[-1] = AIMessage(
        content="",
        tool_calls=script[-1].tool_calls,
        usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
    )
    model = ScriptedToolModel(script=script)
    call = trace_call_for(
        item["message"], MODEL, item_id="legacy-01", run_id="run-l5", root=tmp_path
    )
    path = tmp_path / "replay" / "run-l5.jsonl"

    result = run_agent(item["message"], MODEL, model=model, call=call)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert result is not None
    assert result.answer == item["responses"]["l5"]["content"]
    assert len(rows) == 2
    assert {row["tool"] for row in rows} == {"agent"}
    assert {row["run_id"] for row in rows} == {"run-l5"}
    assert {row["item_id"] for row in rows} == {"legacy-01"}
    assert {row["task"] for row in rows} == {"T-REPLY"}
    assert len({row["response_key"] for row in rows}) == 1
    assert len(rows[-1]["response_key"]) == 64
    assert rows[-1]["input_tokens"] == 11
    assert rows[-1]["output_tokens"] == 5
    assert rows[-1]["model"]["model_id"] == "deepseek-v4.1-flash"
    assert item["message"] not in path.read_text()


def test_an_untraced_call_writes_no_row(tmp_path: Path) -> None:
    item = fixture_item("legacy-01")

    run_agent(item["message"], MODEL, model=ScriptedToolModel(script=script_for(item)))

    assert not list(tmp_path.glob("**/*.jsonl"))


# ------------------------------------------------------------- the MCP server itself


def test_the_server_publishes_the_two_legacy_tools() -> None:
    server = build_mcp_server()

    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {"get_fee", "get_policy"}
    legacy = {
        entry["function"]["name"]: entry["function"]["parameters"] for entry in TOOL_DEFINITIONS
    }
    for name, definition in legacy.items():
        schema = tools[name].input_schema
        assert set(schema["properties"]) == set(definition["properties"])
        assert set(schema["required"]) == set(definition["required"])
        for field, spec in definition["properties"].items():
            assert schema["properties"][field]["enum"] == spec["enum"]


def test_every_fee_and_policy_matches_the_legacy_tool_text() -> None:
    server = build_mcp_server()
    fees = json.loads((FIXTURE_DIR / "fees.json").read_text(encoding="utf-8"))["fees"]
    policies = json.loads((FIXTURE_DIR / "policies.json").read_text(encoding="utf-8"))["policies"]

    for card_type, actions in fees.items():
        for action in actions:
            result = asyncio.run(
                server.call_tool("get_fee", {"card_type": card_type, "action": action})
            )
            text = result.content[0].text
            assert text == legacy_tool_text("get_fee", {"card_type": card_type, "action": action})
            assert json.loads(text)["fee_usd"] == actions[action]
            assert json.loads(text)["currency"] == "USD"
    for topic in policies:
        result = asyncio.run(server.call_tool("get_policy", {"topic": topic}))
        assert result.content[0].text == legacy_tool_text("get_policy", {"topic": topic})


def test_the_server_reads_other_fixture_files(tmp_path: Path) -> None:
    fees = tmp_path / "fees.json"
    policies = tmp_path / "policies.json"
    fees.write_text(
        json.dumps({"currency": "EUR", "fees": {"debit": {"cash_withdrawal": 1}}}),
        encoding="utf-8",
    )
    policies.write_text(json.dumps({"policies": {"refunds": "Test policy."}}), encoding="utf-8")
    server = build_mcp_server(fees_path=fees, policies_path=policies)

    fee = json.loads(
        asyncio.run(
            server.call_tool("get_fee", {"card_type": "debit", "action": "cash_withdrawal"})
        )
        .content[0]
        .text
    )
    policy = json.loads(
        asyncio.run(server.call_tool("get_policy", {"topic": "refunds"})).content[0].text
    )

    assert fee["fee_usd"] == 1
    assert fee["currency"] == "EUR"
    assert policy["policy"] == "Test policy."


def test_an_absent_fee_or_policy_is_a_tool_error(tmp_path: Path) -> None:
    fees = tmp_path / "fees.json"
    policies = tmp_path / "policies.json"
    fees.write_text(json.dumps({"fees": {"debit": {}}}), encoding="utf-8")
    policies.write_text(json.dumps({"policies": {}}), encoding="utf-8")
    server = build_mcp_server(fees_path=fees, policies_path=policies)

    with pytest.raises(ToolError, match="no published fee"):
        asyncio.run(
            server.call_tool("get_fee", {"card_type": "debit", "action": "cash_withdrawal"})
        )
    with pytest.raises(ToolError, match="no published policy"):
        asyncio.run(server.call_tool("get_policy", {"topic": "refunds"}))


def test_a_broken_fixture_is_a_tool_error_not_a_crash(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("[not json", encoding="utf-8")
    missing = tmp_path / "missing.json"

    with pytest.raises(ToolError, match="not valid JSON"):
        fee_text("debit", "cash_withdrawal", broken)
    with pytest.raises(ToolError, match="not readable"):
        policy_text("refunds", missing)


def test_an_argument_outside_the_enum_never_reaches_the_fixture() -> None:
    server = build_mcp_server()

    with pytest.raises(ToolError, match="validation error"):
        asyncio.run(server.call_tool("get_fee", {"card_type": "gold", "action": "cash_withdrawal"}))


# ------------------------------------------------------------------ the MCP wiring


def test_the_agent_loads_its_tools_from_mcp_not_by_hand() -> None:
    item = fixture_item("legacy-01")
    model = ScriptedToolModel(script=script_for(item))

    run_agent(item["message"], MODEL, model=model)

    assert {"get_fee", "get_policy"} <= set(model.bound)
    assert SCHEMA_TOOL in model.bound
    texts = tool_results(model.calls[-1])
    assert len(texts) == 1
    assert json.loads(texts[0])["fee_usd"] == 2.5


def test_the_server_runs_over_stdio_as_a_script(repo_root: Path) -> None:
    """The module is a stdio entry point: ``python -m ...`` must be a runnable target."""
    source = (repo_root / "src" / "which_ai_model" / "tools" / "mcp_server.py").read_text(
        encoding="utf-8"
    )

    assert 'if __name__ == "__main__":' in source
    assert 'run("stdio")' in source


def test_the_archived_adapter_is_never_used(repo_root: Path) -> None:
    import sys

    assert "langchain_mcp_adapters" not in sys.modules
    files = [
        repo_root / "pyproject.toml",
        repo_root / "src" / "which_ai_model" / "tools" / "agent.py",
        repo_root / "src" / "which_ai_model" / "tools" / "mcp_server.py",
    ]
    for path in files:
        assert "langchain-mcp-adapters" not in path.read_text(encoding="utf-8")
        assert "langchain_mcp_adapters" not in path.read_text(encoding="utf-8")
