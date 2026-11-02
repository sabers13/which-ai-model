"""L1 "after": the ticket router on the shared layer.

Before (``legacy/l1_router.py``): a raw ``httpx`` POST, a free-text reply, and a first-line
substring match against the 77 intents. After: one ``RouteOut`` through the schema
channel, one trace row per call, and not a line of provider code — the chat model is built
from a ``ModelRef`` by ``core``, so the same tool runs on any connection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from which_ai_model.contracts import ModelRef, Prediction, RouteOut, TaskName

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
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "TASK",
    "TOOL",
    "predict",
    "route_message",
]

TOOL = "router"
TASK = TaskName.ROUTE
PROMPT_VERSION = "route-v1"
SYSTEM_PROMPT = (
    "You route retail-bank support messages. Each message belongs to exactly one "
    "Banking77 intent. Answer through the schema with that intent's snake_case label "
    "(for example card_arrival or cash_withdrawal_charge). Do not explain."
)


def route_message(
    message: str,
    model: BaseChatModel,
    *,
    call: TraceCall | None = None,
) -> RouteOut | None:
    """One structured call; ``None`` means the answer did not match ``RouteOut``."""
    answer = ask(
        model,
        RouteOut,
        prompt_messages(SYSTEM_PROMPT, message),
        task=TASK,
        tool=TOOL,
        prompt_version=PROMPT_VERSION,
        call=call,
    )
    return answer.parsed if answer.schema_valid else None


def predict(
    item: Mapping[str, Any],
    ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
) -> Prediction:
    """T-ROUTE benchmark callable (``Predictor``): one item, one model, one prediction."""
    messages = prompt_messages(SYSTEM_PROMPT, item_text(item))
    call = new_trace_call(
        item_id(item),
        ref,
        response_key=response_key(ref.model_id, PROMPT_VERSION, messages),
    )
    answer = ask(
        chat_model(ref, model),
        RouteOut,
        messages,
        task=TASK,
        tool=TOOL,
        prompt_version=PROMPT_VERSION,
        call=call,
    )
    return prediction(
        item,
        ref,
        TASK,
        parsed=answer.parsed,
        schema_valid=answer.schema_valid,
        call=call,
    )
