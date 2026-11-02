"""Always-on local JSONL tracing; LangSmith only when its key is set.

One :class:`~which_ai_model.contracts.CallRecord` per model call, written to
``traces/<run_id>.jsonl`` — replay runs under ``traces/replay/`` and live/record runs
under ``traces/live/`` (both gitignored). A record holds timing, tokens, cost, validity
and an error class; never a prompt, a header or a key.

LangSmith is an additional, optional sink. It is switched on only when
``LANGSMITH_API_KEY`` is present; the value is never read, logged or written here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult

from which_ai_model.contracts import CallRecord, ModelRef, TaskName

from .cost import Price, cost_usd
from .usage import extract_usage

__all__ = [
    "CallTraceHandler",
    "TraceWriter",
    "maybe_enable_langsmith",
    "trace_path",
]

DEFAULT_TRACE_ROOT = Path("traces")

_ERROR_MAX_CHARS = 200


class TraceWriter:
    """Append-only writer for one trace file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: CallRecord) -> None:
        """Append one record as a single JSON line."""
        line = json.dumps(record.model_dump(mode="json"), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def close(self) -> None:
        """Nothing is held open between rows; present for a uniform lifecycle."""

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def trace_path(
    run_id: str,
    *,
    mode: str = "live",
    root: Path | str = DEFAULT_TRACE_ROOT,
) -> Path:
    """``traces/replay/<run_id>.jsonl`` for replay, ``traces/live/...`` otherwise."""
    bucket = "replay" if mode == "replay" else "live"
    return Path(root) / bucket / f"{run_id}.jsonl"


def maybe_enable_langsmith(environ: MutableMapping[str, str] | None = None) -> bool:
    """Enable LangSmith tracing only when ``LANGSMITH_API_KEY`` is set.

    Returns whether it was enabled. The key value is only tested for presence; it is
    never copied into another variable, printed or logged.
    """
    env = os.environ if environ is None else environ
    if not env.get("LANGSMITH_API_KEY"):
        return False
    env["LANGSMITH_TRACING"] = "true"
    env["LANGCHAIN_TRACING_V2"] = "true"
    return True


def _first_generation(response: LLMResult) -> BaseMessage | None:
    try:
        generation = response.generations[0][0]
    except (IndexError, TypeError):
        return None
    message = getattr(generation, "message", None)
    return message if isinstance(message, BaseMessage) else None


class CallTraceHandler(BaseCallbackHandler):
    """Write one ``CallRecord`` per chat-model call.

    ``response_key`` may be a callable when the recording key is only known after the
    input is hashed. ``prices`` is the loaded price table; the gateway's measured cost
    always wins over it.
    """

    def __init__(
        self,
        writer: TraceWriter,
        *,
        run_id: str,
        task: TaskName | str,
        tool: str,
        item_id: str,
        model: ModelRef,
        prompt_version: str,
        response_key: str | Callable[[], str],
        prices: Mapping[str, Price] | None = None,
        schema_valid: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.writer = writer
        self.run_id = run_id
        self.task = TaskName(task) if isinstance(task, str) else task
        self.tool = tool
        self.item_id = item_id
        self.model = model
        self.prompt_version = prompt_version
        self.response_key = response_key
        self.prices = prices
        self.schema_valid = schema_valid
        self.clock = clock
        self._starts: dict[UUID, float] = {}
        self._lock = threading.Lock()

    def _start(self, run_id: UUID | None) -> None:
        if run_id is None:
            return
        with self._lock:
            self._starts[run_id] = self.clock()

    def _latency_ms(self, run_id: UUID | None) -> int:
        if run_id is None:
            return 0
        with self._lock:
            started = self._starts.pop(run_id, None)
        if started is None:
            return 0
        return max(0, int((self.clock() - started) * 1000))

    def _key(self) -> str:
        key = self.response_key() if callable(self.response_key) else self.response_key
        return key or "unrecorded"

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, **kwargs: Any) -> None:
        del serialized, messages
        self._start(kwargs.get("run_id"))

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        del serialized, prompts
        self._start(kwargs.get("run_id"))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        message = _first_generation(response)
        usage = extract_usage(message) if message is not None else extract_usage({})
        cost = usage.measured_cost_usd
        if cost is None:
            cost = cost_usd(
                self.model.model_id,
                usage.input_tokens,
                usage.output_tokens,
                self.prices,
            )
        self.writer.write(
            CallRecord(
                run_id=self.run_id,
                task=self.task,
                tool=self.tool,
                item_id=self.item_id,
                model=self.model,
                prompt_version=self.prompt_version,
                latency_ms=self._latency_ms(kwargs.get("run_id")),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                schema_valid=self.schema_valid,
                error=None,
                response_key=self._key(),
            )
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        name = type(error).__name__[:_ERROR_MAX_CHARS]
        self.writer.write(
            CallRecord(
                run_id=self.run_id,
                task=self.task,
                tool=self.tool,
                item_id=self.item_id,
                model=self.model,
                prompt_version=self.prompt_version,
                latency_ms=self._latency_ms(kwargs.get("run_id")),
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                schema_valid=None,
                error=name,
                response_key=self._key(),
            )
        )


def message_usage(message: AIMessage) -> dict[str, Any]:
    """Small helper for callers that want the normalised usage block."""
    return extract_usage(message).as_dict()
