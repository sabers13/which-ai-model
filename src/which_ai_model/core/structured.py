"""Structured output through LangChain's schema strategies — never JSON in a prompt.

The model is asked for the schema through its native tool/JSON-schema channel
(``with_structured_output``). This module only normalises the result into::

    (parsed | None, schema_valid: bool, raw_content)

``schema_valid`` is ``False`` for an answer that did not satisfy the schema, including an
empty or length-truncated answer. Those are scored wrong, never raised and never retried
(``PLAN.md`` §7). A JSON-looking string in free text is not parsed: if the model did not
answer through the schema channel, the answer is invalid.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

__all__ = [
    "StructuredAnswer",
    "answer_message",
    "make_structured_model",
    "message_text",
    "structured",
]


#: A parsed answer, its validity, and the raw text (empty for an empty answer).
class StructuredAnswer(NamedTuple):
    parsed: Any
    schema_valid: bool
    raw_content: str | None


def make_structured_model(
    model: Any,
    schema: type[BaseModel],
    *,
    method: str = "function_calling",
) -> Runnable[Any, Any]:
    """``model.with_structured_output(schema, include_raw=True, method=...)``.

    ``include_raw`` is required for the trace: the raw message holds usage and finish
    reason. ``method="function_calling"`` is the benchmark decision (identical strategy
    for every model, ``docs/INTERFACES.md`` §3).
    """
    return cast(
        Runnable[Any, Any],
        model.with_structured_output(schema, include_raw=True, method=method),
    )


def message_text(message: Any) -> str | None:
    """Plain text of a message, or ``None`` when it carries no text."""
    if isinstance(message, str):
        return message
    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, str):
            return content
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
        return None
    return None


def _raw_from_error(exc: OutputParserException) -> Any:
    return getattr(exc, "llm_output", None)


def structured(
    model: BaseChatModel | Any,
    schema: type[BaseModel],
    request: Any = "",
    *,
    method: str = "function_calling",
    config: RunnableConfig | None = None,
) -> StructuredAnswer:
    """Call ``model`` for ``schema`` and return ``(parsed, schema_valid, raw_content)``.

    ``request`` is anything the model accepts as messages (a string, a list of messages
    or a prompt-value dict). A length-truncated or empty answer is schema-invalid — not
    an exception — with ``raw_content`` empty. ``config`` is the run configuration for
    this one call — tools pass their trace callbacks there, so a fallback model forwards
    them to whichever model actually answers.
    """
    runnable = make_structured_model(model, schema, method=method)
    try:
        result = runnable.invoke(request, config=config)
    except OutputParserException as exc:
        raw = _raw_from_error(exc)
        return StructuredAnswer(None, False, message_text(raw))

    if isinstance(result, dict):
        parsed = result.get("parsed")
        raw = result.get("raw")
        error = result.get("parsing_error")
    else:  # a model that ignored include_raw; treat as parsed if it is the schema
        parsed = result
        raw = None
        error = None

    valid = isinstance(parsed, schema) and error is None
    if not isinstance(parsed, schema):
        parsed = None
    return StructuredAnswer(parsed, valid, message_text(raw))


def answer_message(result: Any) -> AIMessage | None:
    """The raw message from a structured-output result, for finish-reason checks."""
    if isinstance(result, dict):
        raw = result.get("raw")
        return raw if isinstance(raw, AIMessage) else None
    return None
