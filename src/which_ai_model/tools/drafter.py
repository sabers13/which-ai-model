"""L2 "after": the reply drafter on the shared layer.

Before (``legacy/l2_drafter.py``): the Anthropic SDK, a free-text answer and a single
regex pulling the first ``{...}`` block out of it. After: one ``ReplyOut`` through the
schema channel — ``intent``, ``urgency`` and ``reply`` are validated fields, not a lucky
match — traced, with no provider code in this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from which_ai_model.contracts import ModelRef, Prediction, ReplyOut, TaskName

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
    "draft_reply",
    "predict",
]

TOOL = "drafter"
TASK = TaskName.REPLY
PROMPT_VERSION = "reply-v1"
SYSTEM_PROMPT = (
    "You are a concise retail-bank support assistant. Answer through the schema: the "
    "message's intent as a short snake_case label, its urgency as low, medium or high, "
    "and one short paragraph the customer can read as the reply."
)


def draft_reply(
    message: str,
    model: BaseChatModel,
    *,
    call: TraceCall | None = None,
) -> ReplyOut | None:
    """One structured call; ``None`` means the answer did not match ``ReplyOut``."""
    answer = ask(
        model,
        ReplyOut,
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
    """T-REPLY benchmark callable (``Predictor``): one item, one model, one prediction."""
    messages = prompt_messages(SYSTEM_PROMPT, item_text(item))
    call = new_trace_call(
        item_id(item),
        ref,
        response_key=response_key(ref.model_id, PROMPT_VERSION, messages),
    )
    answer = ask(
        chat_model(ref, model),
        ReplyOut,
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
