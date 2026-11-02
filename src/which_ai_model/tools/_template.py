"""The frame every tool is built from — copy a tool, not a blank file.

A tool has four pieces, and only the last two are new work:

1. a pydantic schema from ``contracts`` (``RouteOut``, ``ReplyOut``, ``ScopeOut``, ...);
2. one versioned prompt string and ``prompt_messages()`` turning the input into messages;
3. an entry function ``tool(message, model) -> <Schema> | None`` — the parsed contract, or
   ``None`` when the model did not answer through the schema channel (scored wrong, never
   retried; ``PLAN.md`` §7);
4. ``predict(item, model_ref) -> Prediction`` for the benchmark runner.

Pieces 2 to 4 are all a tool writes. Everything a tool must not do itself — choosing a
provider class, reading a key, retrying, falling back, writing a recording key — lives in
``core``. The helpers here only supply the frame: a trace-context object, the one traced
structured call (``ask``) and the ``Prediction`` assembly. Copy the ``example`` block at
the bottom, rename it, and it is a new tool in about ten lines of its own code.

Trace files are one per run (``traces/replay/<run_id>.jsonl`` for replay runs,
``traces/live/...`` otherwise). ``WAM_RUN_ID`` and ``WAM_MODE`` name the run, exactly as
the ``Makefile`` exports them; outside a run both fall back to ``adhoc`` / ``replay`` so a
stray call still leaves an honest row. Neither variable is a credential and neither is
ever written into a row.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from which_ai_model.contracts import ModelRef, Prediction, RouteOut, TaskName
from which_ai_model.core import (
    CallTraceHandler,
    Price,
    StructuredAnswer,
    TraceWriter,
    get_model,
    recording_key,
    request_digest,
    structured,
    trace_path,
)

__all__ = [
    "ADHOC_RUN_ID",
    "TRACE_ROOT",
    "TraceCall",
    "ask",
    "item_id",
    "item_text",
    "new_trace_call",
    "prediction",
    "prompt_messages",
    "response_key",
]

#: Where trace files land unless a run points elsewhere.
TRACE_ROOT = Path("traces")
#: Run id used when ``WAM_RUN_ID`` is unset, so a call outside a run is still traced.
ADHOC_RUN_ID = "adhoc"


def prompt_messages(system_prompt: str, message: str) -> list[dict[str, str]]:
    """The two-message request every tool sends: fixed instructions, then the input."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]


def item_id(item: Mapping[str, Any]) -> str:
    """The item id a benchmark sample and a parity fixture both carry."""
    return str(item["item_id"])


def item_text(item: Mapping[str, Any]) -> str:
    """The input text: ``text`` in a benchmark sample, ``message`` in a parity fixture."""
    for key in ("text", "message"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    msg = "item carries neither 'text' nor 'message'"
    raise KeyError(msg)


def response_key(model_id: str, prompt_version: str, messages: Sequence[Any]) -> str:
    """``recordings/`` key for this request — the trace row and ``Prediction`` carry it."""
    return recording_key(model_id, prompt_version, request_digest(list(messages)))


@dataclass(frozen=True)
class TraceCall:
    """Identity of one tool call inside a run: what its trace row needs to say."""

    run_id: str
    item_id: str
    model: ModelRef
    response_key: str
    writer: TraceWriter
    prices: Mapping[str, Price] | None = None


def new_trace_call(
    item_id: str,
    ref: ModelRef,
    *,
    response_key: str,
    run_id: str | None = None,
    mode: str | None = None,
    root: Path | None = None,
    prices: Mapping[str, Price] | None = None,
) -> TraceCall:
    """Open this call's trace context: the run's file, the model and the recording key.

    ``run_id`` defaults to ``WAM_RUN_ID`` (``adhoc`` outside a run) and ``mode`` to
    ``WAM_MODE`` (``replay``), matching what the ``Makefile`` exports.
    """
    resolved_run = run_id or os.environ.get("WAM_RUN_ID") or ADHOC_RUN_ID
    resolved_mode = mode or os.environ.get("WAM_MODE") or "replay"
    writer = TraceWriter(trace_path(resolved_run, mode=resolved_mode, root=root or TRACE_ROOT))
    return TraceCall(
        run_id=resolved_run,
        item_id=item_id,
        model=ref,
        response_key=response_key,
        writer=writer,
        prices=prices,
    )


class _DeferredTraceHandler(CallTraceHandler):
    """Hold the row until the schema verdict is known, then write it once.

    ``CallTraceHandler`` writes on ``on_llm_end``, before the structured-output parser
    has run. The tool knows ``schema_valid`` a moment later, so the row is buffered by
    :func:`ask` and flushed with the real verdict. The row itself is still built by
    ``core`` — no field is assembled here.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._end: tuple[LLMResult, dict[str, Any]] | None = None
        self._error: tuple[BaseException, dict[str, Any]] | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._end = (response, kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._error = (error, kwargs)

    def flush(self, *, schema_valid: bool | None) -> None:
        """Write the buffered row with its verdict; a failed call keeps its error."""
        self.schema_valid = schema_valid
        if self._error is not None:
            error, kwargs = self._error
            super().on_llm_error(error, **kwargs)
        elif self._end is not None:
            response, kwargs = self._end
            super().on_llm_end(response, **kwargs)


def ask[OutT: BaseModel](
    model: BaseChatModel,
    schema: type[OutT],
    messages: Sequence[dict[str, str]],
    *,
    task: TaskName,
    tool: str,
    prompt_version: str,
    call: TraceCall | None = None,
) -> StructuredAnswer:
    """The tool's only model call: one schema, one trace row, no retry of its own.

    ``call`` carries the run identity; ``None`` means this invocation is outside a
    benchmark run (a unit test, a notebook) and writes no row. Retries and fallback are
    already inside ``model`` when it came from :func:`~which_ai_model.core.get_model`.
    """
    config: RunnableConfig | None = None
    handler: _DeferredTraceHandler | None = None
    if call is not None:
        handler = _DeferredTraceHandler(
            writer=call.writer,
            run_id=call.run_id,
            task=task,
            tool=tool,
            item_id=call.item_id,
            model=call.model,
            prompt_version=prompt_version,
            response_key=call.response_key,
            prices=call.prices,
        )
        config = {"callbacks": [handler]}
    try:
        answer = structured(model, schema, list(messages), config=config)
    except BaseException:
        if handler is not None:
            handler.flush(schema_valid=None)
        raise
    if handler is not None:
        handler.flush(schema_valid=answer.schema_valid)
    return answer


def prediction(
    item: Mapping[str, Any],
    ref: ModelRef,
    task: TaskName,
    *,
    parsed: Any,
    schema_valid: bool,
    call: TraceCall,
) -> Prediction:
    """The scored answer: the contract as JSON, or ``{}`` when the answer was invalid."""
    model = parsed if schema_valid else None
    output = model.model_dump() if isinstance(model, BaseModel) else {}
    return Prediction(
        item_id=item_id(item),
        task=task,
        model=ref,
        output=output,
        raw_key=call.response_key,
    )


def chat_model(ref: ModelRef, model: BaseChatModel | None) -> BaseChatModel:
    """The model to call: an injected one (tests, replay) or one built from ``ref``."""
    return model if model is not None else get_model(ref)


# ---------------------------------------------------------------------------------------
# The template itself. A real tool changes the schema, the prompt and these constants and
# keeps the two function bodies; ``tests/tools/test_parity_l1_l3.py`` runs this example so
# the frame cannot rot.
# ---------------------------------------------------------------------------------------

EXAMPLE_TOOL = "example"
EXAMPLE_TASK = TaskName.ROUTE
EXAMPLE_PROMPT_VERSION = "example-v1"
EXAMPLE_PROMPT = "Answer with one intent label. Do not explain."


def example_tool(
    message: str,
    model: BaseChatModel,
    *,
    call: TraceCall | None = None,
) -> RouteOut | None:
    """The entry function: input plus a built model in, parsed contract out."""
    answer = ask(
        model,
        RouteOut,
        prompt_messages(EXAMPLE_PROMPT, message),
        task=EXAMPLE_TASK,
        tool=EXAMPLE_TOOL,
        prompt_version=EXAMPLE_PROMPT_VERSION,
        call=call,
    )
    return answer.parsed if answer.schema_valid else None


def example_predict(
    item: Mapping[str, Any],
    ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
) -> Prediction:
    """The benchmark callable: builds the model, opens the trace row, scores the answer."""
    messages = prompt_messages(EXAMPLE_PROMPT, item_text(item))
    call = new_trace_call(
        item_id(item),
        ref,
        response_key=response_key(ref.model_id, EXAMPLE_PROMPT_VERSION, messages),
    )
    answer = ask(
        chat_model(ref, model),
        RouteOut,
        messages,
        task=EXAMPLE_TASK,
        tool=EXAMPLE_TOOL,
        prompt_version=EXAMPLE_PROMPT_VERSION,
        call=call,
    )
    return prediction(
        item,
        ref,
        EXAMPLE_TASK,
        parsed=answer.parsed,
        schema_valid=answer.schema_valid,
        call=call,
    )
