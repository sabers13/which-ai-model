"""L5 "after": the fee agent as a LangChain v1 ``create_agent`` graph.

The legacy loop (``legacy/l5_agent.py``) hand-rolled everything: the ``tools`` JSON, the
assistant/tool message bookkeeping, the turn limit, the redirect guard and the error
strings. Here ``create_agent`` owns the loop, ``langchain.mcp.MCPAdapter`` loads the two
tools from :mod:`which_ai_model.tools.mcp_server`, and ``ToolStrategy`` makes the final
turn a schema (``AgentOut``) instead of free text that a caller has to parse.

The tool is laid out like the rest of ``tools/`` (``_template.py``): ``TOOL``, ``TASK``,
``PROMPT_VERSION`` and ``SYSTEM_PROMPT`` constants, ``run_agent(question, ref, ...)``
returning the contract or ``None`` when the model never answered through the schema
channel, and a ``TraceCall`` opening the run's trace row. Two things are specific to an
agent: the model is built with ``core.make_agent`` so retries and the fallback model wrap
the whole graph (``bind_tools`` would otherwise bypass them), and the graph is
async-only because MCP tools are (``arun_agent`` is the entry point, ``run_agent`` the
sync wrapper).

A ``BaseChatModel`` may be injected (replay mode, tests) exactly as in the other tools;
the fallback then belongs to the injected model.

L5 is a parity demo, not one of the four benchmark tasks, and the frozen ``TaskName``
contract has no L5 member: trace rows are therefore tagged ``T-REPLY`` with
``tool="agent"``, which still separates them from every benchmark row.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.mcp import MCPAdapter
from langchain_core.language_models import BaseChatModel
from langgraph.errors import GraphRecursionError
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from which_ai_model.contracts import Connection, ModelRef, TaskName
from which_ai_model.core import ModelBuilder, Price, make_agent
from which_ai_model.core.trace import CallTraceHandler

from ._template import TraceCall, new_trace_call, prompt_messages, response_key
from .mcp_server import build_mcp_server

__all__ = [
    "DEFAULT_MAX_TURNS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "TASK",
    "TOOL",
    "AgentOut",
    "AgentTurnLimitError",
    "arun_agent",
    "run_agent",
    "trace_call_for",
]

TOOL = "agent"
#: L5 is not a benchmark task; the frozen ``TaskName`` has no member for it.
TASK = TaskName.REPLY
#: Version of the prompt below, recorded in every trace row.
PROMPT_VERSION = "l5-agent-v1"
#: The legacy loop's default limit; kept equal so a "before/after" run stops alike.
DEFAULT_MAX_TURNS = 12

SYSTEM_PROMPT = (
    "Answer the customer's fee and policy question. Use the tools for every number and "
    "policy; never guess. Finish with one short paragraph that names the fee in USD, and "
    "report the fee amounts you looked up."
)

type Connections = Mapping[str, Connection] | Sequence[Connection] | None
type AdapterTarget = MCPServer | Path


class AgentOut(BaseModel):
    """The agent's final answer: the prose answer and the fee amounts behind it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    fees_used: list[float] = Field(default_factory=list)


class AgentTurnLimitError(RuntimeError):
    """The agent hit the legacy turn limit before it answered."""


def _connection_table(connections: Connections) -> dict[str, Connection]:
    """Resolve connections exactly like ``core.models.get_model`` does (that map is private)."""
    if connections is None:
        from which_ai_model.connect import load_connections

        return {connection.id: connection for connection in load_connections()}
    if isinstance(connections, Mapping):
        return dict(connections)
    return {connection.id: connection for connection in connections}


def _connection(table: Mapping[str, Connection], connection_id: str, *, role: str) -> Connection:
    try:
        return table[connection_id]
    except KeyError as exc:
        raise KeyError(f"unknown {role} connection {connection_id!r}") from exc


def _build_agent(
    ref: ModelRef,
    model: BaseChatModel | None,
    tools: Sequence[Any],
    *,
    fallback: ModelRef | None,
    connections: Connections,
    builder: ModelBuilder | None,
    session_id: str | None,
) -> Any:
    """``create_agent`` for an injected model, or built from ``ref`` by ``core``."""
    if model is not None:
        if fallback is not None:
            msg = "fallback belongs to a built model; pass ref only, or wrap the model yourself"
            raise ValueError(msg)
        return create_agent(
            model=model,
            tools=list(tools),
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(AgentOut, handle_errors=False),
        )
    table = _connection_table(connections)
    primary = _connection(table, ref.connection_id, role="primary")
    backup = (
        _connection(table, fallback.connection_id, role="fallback")
        if fallback is not None
        else None
    )
    return make_agent(
        primary,
        backup,
        ref.model_id,
        tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=AgentOut,
        builder=builder,
        session_id=session_id,
    )


def _trace_handler(call: TraceCall | None) -> CallTraceHandler | None:
    """One row per model call, built by ``core`` from the run's ``TraceCall``."""
    if call is None:
        return None
    return CallTraceHandler(
        call.writer,
        run_id=call.run_id,
        task=TASK,
        tool=TOOL,
        item_id=call.item_id,
        model=call.model,
        prompt_version=PROMPT_VERSION,
        response_key=call.response_key,
        prices=call.prices,
    )


async def arun_agent(
    question: str,
    ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
    fallback: ModelRef | None = None,
    server: AdapterTarget | None = None,
    connections: Connections = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    call: TraceCall | None = None,
    builder: ModelBuilder | None = None,
    session_id: str | None = None,
) -> AgentOut | None:
    """Answer one fee/policy question through the MCP tools and the agent graph.

    ``None`` means the model never answered through the schema channel (scored wrong,
    never retried); a run that keeps calling tools until the turn limit raises
    :class:`AgentTurnLimitError`.
    """
    if max_turns < 1:
        msg = f"max_turns must be >= 1; got {max_turns}"
        raise ValueError(msg)
    messages = prompt_messages(SYSTEM_PROMPT, question)
    # LangGraph counts one superstep per graph node, so one model turn is the model
    # node plus the tool node: 2 * max_turns - 1 supersteps allow exactly max_turns
    # model calls, the same limit the legacy loop enforced by hand.
    config: dict[str, Any] = {"recursion_limit": 2 * max_turns - 1}
    handler = _trace_handler(call)
    if handler is not None:
        config["callbacks"] = [handler]
    target: AdapterTarget = server if server is not None else build_mcp_server()

    async with MCPAdapter(target) as adapter:
        tools = await adapter.list_tools()
        agent = _build_agent(
            ref,
            model,
            tools,
            fallback=fallback,
            connections=connections,
            builder=builder,
            session_id=session_id,
        )
        try:
            result: Any = await agent.ainvoke({"messages": messages}, config=config)
        except GraphRecursionError as exc:
            raise AgentTurnLimitError(f"agent reached its {max_turns}-turn limit") from exc

    output = result.get("structured_response") if isinstance(result, dict) else None
    return output if isinstance(output, AgentOut) else None


def run_agent(
    question: str,
    ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
    fallback: ModelRef | None = None,
    server: AdapterTarget | None = None,
    connections: Connections = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    call: TraceCall | None = None,
    builder: ModelBuilder | None = None,
    session_id: str | None = None,
) -> AgentOut | None:
    """Synchronous :func:`arun_agent`; not callable from inside a running event loop."""
    return asyncio.run(
        arun_agent(
            question,
            ref,
            model=model,
            fallback=fallback,
            server=server,
            connections=connections,
            max_turns=max_turns,
            call=call,
            builder=builder,
            session_id=session_id,
        )
    )


def trace_call_for(
    question: str,
    ref: ModelRef,
    *,
    item_id: str | None = None,
    run_id: str | None = None,
    mode: str | None = None,
    root: Path | None = None,
    prices: Mapping[str, Price] | None = None,
) -> TraceCall:
    """Open the run's trace row for this question, like ``predict()`` does for L1-L3."""
    messages = prompt_messages(SYSTEM_PROMPT, question)
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return new_trace_call(
        item_id or digest,
        ref,
        response_key=response_key(ref.model_id, PROMPT_VERSION, messages),
        run_id=run_id,
        mode=mode,
        root=root,
        prices=prices,
    )
