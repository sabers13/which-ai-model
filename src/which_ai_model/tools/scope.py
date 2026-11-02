"""L3 "after": the scope checker as a tiny LangGraph — classify, then maybe escalate.

Before (``legacy/l3_scope.py``): the Gemini SDK and a substring search for "yes"/"no" in
free text, where the first "yes" anywhere wins. After: a ``classify`` node asks for
``ScopeOut`` through the schema channel, and a conditional edge sends an out-of-scope
message to an ``escalate`` node that returns a fixed handoff. The model call is traced
once; the graph itself makes no network call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from which_ai_model.contracts import ModelRef, Prediction, ScopeOut, TaskName

from ._template import (
    TraceCall,
    ask,
    chat_model,
    item_id,
    item_text,
    new_trace_call,
    prediction,
    prompt_messages,
    response_key,
)

__all__ = [
    "HANDOFF_MESSAGE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "TASK",
    "TOOL",
    "build_scope_graph",
    "check_scope",
    "predict",
]

TOOL = "scope"
TASK = TaskName.SCOPE
PROMPT_VERSION = "scope-v1"
SYSTEM_PROMPT = (
    "You check whether a customer message is within the supported scope of a retail-bank "
    "support assistant (cards, transfers, top-ups, fees and account access). Answer "
    "through the schema: out_of_scope is true only when the request is outside that scope."
)

#: What the escalate node returns: a fixed human handoff, never model-generated text.
HANDOFF_MESSAGE = (
    "This request is outside the supported banking scope, so I have passed it to a human "
    "colleague who can help."
)


class ScopeState(TypedDict, total=False):
    """Graph state: the input, the traced verdict, and the handoff when escalated."""

    message: str
    parsed: ScopeOut | None
    schema_valid: bool
    out_of_scope: bool
    reply: str | None


ScopeGraph = CompiledStateGraph[ScopeState, Any, ScopeState, ScopeState]


def _classify_node(
    model: BaseChatModel,
    call: TraceCall | None,
) -> Any:  # a graph node; langgraph's node protocol is not expressible as a plain Callable
    def classify(state: ScopeState) -> dict[str, Any]:
        answer = ask(
            model,
            ScopeOut,
            prompt_messages(SYSTEM_PROMPT, state["message"]),
            task=TASK,
            tool=TOOL,
            prompt_version=PROMPT_VERSION,
            call=call,
        )
        parsed: ScopeOut | None = answer.parsed if answer.schema_valid else None
        return {
            "parsed": parsed,
            "schema_valid": answer.schema_valid,
            "out_of_scope": parsed is not None and parsed.out_of_scope,
            "reply": None,
        }

    return classify


def _escalate_node(state: ScopeState) -> dict[str, Any]:
    del state
    return {"reply": HANDOFF_MESSAGE}


def _next_node(state: ScopeState) -> str:
    return "escalate" if state.get("out_of_scope") else END


def build_scope_graph(
    model: BaseChatModel,
    *,
    call: TraceCall | None = None,
) -> ScopeGraph:
    """``START -> classify -> (escalate -> END) | END``, one traced model call inside."""
    builder: StateGraph[ScopeState, Any, ScopeState, ScopeState] = StateGraph(ScopeState)
    builder.add_node("classify", _classify_node(model, call))
    builder.add_node("escalate", _escalate_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", _next_node, ["escalate", END])
    builder.add_edge("escalate", END)
    return builder.compile()


def check_scope(
    message: str,
    model: BaseChatModel,
    *,
    call: TraceCall | None = None,
) -> ScopeOut | None:
    """Run the graph once; ``None`` means the answer did not match ``ScopeOut``."""
    state = cast(ScopeState, build_scope_graph(model, call=call).invoke({"message": message}))
    if not state.get("schema_valid"):
        return None
    return state.get("parsed")


def predict(
    item: Mapping[str, Any],
    ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
) -> Prediction:
    """T-SCOPE benchmark callable (``Predictor``): one item, one model, one prediction."""
    message = item_text(item)
    messages = prompt_messages(SYSTEM_PROMPT, message)
    call = new_trace_call(
        item_id(item),
        ref,
        response_key=response_key(ref.model_id, PROMPT_VERSION, messages),
    )
    parsed = check_scope(message, chat_model(ref, model), call=call)
    return prediction(
        item,
        ref,
        TASK,
        parsed=parsed,
        schema_valid=parsed is not None,
        call=call,
    )
