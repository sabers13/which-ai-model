"""Record/replay cache keyed by sha256, with a replay miss that never hits the network.

Modes are ``live`` (call the provider, write nothing), ``record`` (call the provider and
append a row) and ``replay`` (serve recordings only). The key is::

    key = sha256(f"{model_id}\\n{prompt_version}\\n{request_digest}")
    request_digest = sha256(canonical JSON of the messages)

Rows live at ``recordings/<task>/<model_slug>.jsonl`` and are append-only. A row keeps
only ``content``, ``tool_calls`` and ``usage`` — never headers, keys or base URLs. In
replay mode a missing key raises :class:`ReplayMiss`; it never falls through to a model
(``PLAN.md`` §5).

A fresh run (``PLAN-v1.1`` §A6) uses :func:`fresh_recording_root` and a recorder with
``reuse=False``: it writes under ``recordings/fresh/<run_id>/`` and never serves a saved
answer from any other root. Replay of those fresh rows is still allowed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from .usage import extract_usage

__all__ = [
    "DEFAULT_FRESH_RECORDINGS_ROOT",
    "DEFAULT_RECORDINGS_ROOT",
    "Recorder",
    "RecordingMode",
    "ReplayChatModel",
    "ReplayMiss",
    "fresh_recording_root",
    "model_slug",
    "recording_key",
    "replay_model",
    "request_digest",
    "response_from_message",
    "scrub_response",
]

DEFAULT_RECORDINGS_ROOT = Path("recordings")
#: A fresh run (``PLAN-v1.1`` §A6) writes under this root, never the default one.
DEFAULT_FRESH_RECORDINGS_ROOT = Path("recordings") / "fresh"

_SLUG = re.compile(r"[^a-z0-9]+")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SECRET_KEYS = frozenset(
    {"headers", "authorization", "api_key", "apikey", "x-api-key", "x-goog-api-key"}
)


class RecordingMode(StrEnum):
    """How calls are served. ``record`` is live plus a written row."""

    LIVE = "live"
    REPLAY = "replay"
    RECORD = "record"


class ReplayMiss(LookupError):  # noqa: N818
    """A replay key has no recorded row. The caller must not call a model instead."""

    def __init__(self, key: str, root: Path) -> None:
        self.key = key
        self.root = root
        super().__init__(f"no recording for key {key} under {root}; replay never calls a model")


def _normalise_message(message: Any) -> Any:
    if isinstance(message, BaseMessage):
        row: dict[str, Any] = {"type": message.type, "content": message.content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            row["tool_calls"] = tool_calls
        return row
    if isinstance(message, tuple):
        return {"role": message[0], "content": message[1]}
    if isinstance(message, dict):
        return message
    return str(message)


def request_digest(messages: Any) -> str:
    """``sha256`` of the canonical JSON form of the request messages."""
    if isinstance(messages, (str, bytes)):
        payload: Any = messages.decode() if isinstance(messages, bytes) else messages
    elif isinstance(messages, (list, tuple)):
        payload = [_normalise_message(message) for message in messages]
    else:
        payload = _normalise_message(messages)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recording_key(model_id: str, prompt_version: str, digest: str) -> str:
    """The recording key: model id, prompt version and request digest, sha256."""
    material = f"{model_id}\n{prompt_version}\n{digest}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def model_slug(model_id: str) -> str:
    """A filesystem-safe directory name for a model id."""
    slug = _SLUG.sub("-", model_id.lower()).strip("-")
    return slug or "model"


def fresh_recording_root(
    run_id: str,
    *,
    root: Path | str = DEFAULT_FRESH_RECORDINGS_ROOT,
) -> Path:
    """``<root>/<run_id>``: the only recording root a fresh run may read or write.

    A fresh run must never reuse an answer from the committed evidence root, so the
    run id is validated as a single path component instead of being joined blindly.
    """
    if not _RUN_ID.fullmatch(run_id):
        msg = (
            f"invalid fresh run id {run_id!r}: use letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
        raise ValueError(msg)
    return Path(root) / run_id


def _scrub_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_mapping(item)
            for key, item in value.items()
            if isinstance(key, str) and key.lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_scrub_mapping(item) for item in value]
    return value


def scrub_response(response: dict[str, Any]) -> dict[str, Any]:
    """Keep ``content``, ``tool_calls`` and ``usage``; drop everything else."""
    tool_calls = response.get("tool_calls")
    usage = response.get("usage")
    return {
        "content": response.get("content"),
        "tool_calls": _scrub_mapping(tool_calls) if isinstance(tool_calls, list) else [],
        "usage": _scrub_mapping(usage) if isinstance(usage, dict) else {},
    }


def response_from_message(message: AIMessage) -> dict[str, Any]:
    """Build a recording response from a message that just came back from a model."""
    content_value: Any = message.content
    if content_value is None:
        content = None
    elif isinstance(content_value, str):
        content = content_value
    else:
        content = message.text or None
    tool_calls = [
        {
            "name": call.get("name"),
            "args": call.get("args", {}),
            "id": call.get("id"),
            "type": call.get("type", "tool_call"),
        }
        for call in message.tool_calls
    ]
    response_metadata = message.response_metadata or {}
    raw_usage = response_metadata.get("token_usage")
    if not isinstance(raw_usage, dict):
        # Anthropic keeps the provider usage (including the gateway's `opencode` block)
        # under "usage"; OpenAI keeps it under "token_usage".
        raw_usage = response_metadata.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    if not usage:
        usage = dict(message.usage_metadata or {})
    return scrub_response(
        {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
        }
    )


@dataclass(frozen=True)
class _Row:
    key: str
    model_id: str
    prompt_version: str
    request_digest: str
    response: dict[str, Any]
    recorded_at: str


class Recorder:
    """Append-only recordings under ``root``; lookup returns the last matching row."""

    def __init__(
        self,
        root: Path | str = DEFAULT_RECORDINGS_ROOT,
        mode: RecordingMode | str = RecordingMode.REPLAY,
        *,
        task: str | None = None,
        reuse: bool = True,
    ) -> None:
        self.root = Path(root)
        self.mode = RecordingMode(mode)
        self.task = task
        #: ``reuse=False`` is the fresh-run rule: in record mode an existing key is
        #: never served, so ``lookup`` returns ``None`` without reading any root.
        #: Replay still reads its own root (a fresh recording is replayable).
        self.reuse = reuse
        self._lock = threading.Lock()
        self._index: dict[str, tuple[str | None, dict[str, Any]]] | None = None

    def _path_for(self, task: str, model_id: str) -> Path:
        return self.root / task / f"{model_slug(model_id)}.jsonl"

    def record(
        self,
        key: str,
        model_id: str,
        prompt_version: str,
        digest: str,
        response: dict[str, Any],
        *,
        task: str | None = None,
        recorded_at: datetime | None = None,
    ) -> Path | None:
        """Append one row in ``record`` mode; a no-op in ``live`` mode.

        Recording is refused in ``replay`` mode: a replay run must never overwrite the
        evidence it is reading.
        """
        if self.mode is RecordingMode.LIVE:
            return None
        if self.mode is RecordingMode.REPLAY:
            msg = "cannot record while in replay mode"
            raise RuntimeError(msg)
        row_task = task or self.task
        if not row_task:
            msg = "record() needs a task (or Recorder(task=...))"
            raise ValueError(msg)
        moment = recorded_at or datetime.now(UTC)
        row = _Row(
            key=key,
            model_id=model_id,
            prompt_version=prompt_version,
            request_digest=digest,
            response=scrub_response(response),
            recorded_at=moment.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        )
        path = self._path_for(row_task, model_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_row_dict(row), sort_keys=True) + "\n")
            if self._index is not None:
                self._index[key] = (row_task, row.response)
        return path

    def record_message(
        self,
        key: str,
        model_id: str,
        prompt_version: str,
        digest: str,
        message: AIMessage,
        *,
        task: str | None = None,
    ) -> Path | None:
        """Convenience wrapper around :func:`response_from_message`."""
        return self.record(
            key,
            model_id,
            prompt_version,
            digest,
            response_from_message(message),
            task=task,
        )

    def lookup(self, key: str, *, task: str | None = None) -> dict[str, Any] | None:
        """Find the last recorded response for ``key``.

        Raises :class:`ReplayMiss` in replay mode. In ``live``/``record`` mode a miss is
        ``None`` so the caller knows it must make the call. A ``reuse=False`` recorder
        returns ``None`` before touching any root: a fresh run never serves a saved
        answer.
        """
        if not self.reuse and self.mode is not RecordingMode.REPLAY:
            return None
        scope = task or self.task
        entry = self._load_index().get(key)
        if entry is not None and (scope is None or entry[0] == scope):
            return entry[1]
        if self.mode is RecordingMode.REPLAY:
            raise ReplayMiss(key, self.root)
        return None

    def _load_index(self) -> dict[str, tuple[str | None, dict[str, Any]]]:
        """Read every recording once; later appends update the index in memory.

        Only the documented layout ``<root>/<task>/<model>.jsonl`` is read. A fresh
        re-check root lives under ``recordings/fresh/<run_id>/<task>/`` and its rows
        answer the same keys with newer answers, so a recursive walk would let them
        shadow the committed v1 rows and change a replay (``PLAN-v1.1`` §A6).
        """
        with self._lock:
            if self._index is not None:
                return self._index
            index: dict[str, tuple[str | None, dict[str, Any]]] = {}
            if self.root.is_dir():
                for path in sorted(self.root.glob("*/*.jsonl")):
                    row_task = path.parent.name
                    with path.open(encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(row, dict) or not isinstance(row.get("key"), str):
                                continue
                            response = row.get("response")
                            if isinstance(response, dict):
                                index[row["key"]] = (row_task, response)
            self._index = index
            return index

    def model(self, key: str, model_id: str, *, task: str | None = None) -> ReplayChatModel:
        """A chat model that serves the recorded response for ``key``."""
        response = self.lookup(key, task=task)
        if response is None:
            raise ReplayMiss(key, self.root)
        return ReplayChatModel(response=response, model_id=model_id, key=key)


def _row_dict(row: _Row) -> dict[str, Any]:
    return {
        "key": row.key,
        "model_id": row.model_id,
        "prompt_version": row.prompt_version,
        "request_digest": row.request_digest,
        "response": row.response,
        "recorded_at": row.recorded_at,
    }


class ReplayChatModel(BaseChatModel):
    """A chat model that replays one recorded response. It never opens a socket."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: dict[str, Any]
    model_id: str
    key: str

    @property
    def _llm_type(self) -> str:
        return "which-ai-model-replay"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ReplayChatModel:
        """The recorded answer already carries its tool call; binding changes nothing."""
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        content = self.response.get("content")
        tool_calls = self.response.get("tool_calls")
        usage = extract_usage(self.response)
        usage_metadata = None
        if usage.input_tokens is not None and usage.output_tokens is not None:
            usage_metadata = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens
                if usage.total_tokens is not None
                else usage.input_tokens + usage.output_tokens,
            }
        message = AIMessage(
            content=content if isinstance(content, str) else "",
            tool_calls=tool_calls if isinstance(tool_calls, list) else [],
            usage_metadata=usage_metadata,
            response_metadata={
                "model_name": self.model_id,
                "finish_reason": usage.finish_reason,
                "token_usage": self.response.get("usage", {}),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def replay_model(
    recorder: Recorder, key: str, model_id: str, *, task: str | None = None
) -> ReplayChatModel:
    """Look ``key`` up and return a replay model for it (raises :class:`ReplayMiss`)."""
    return recorder.model(key, model_id, task=task)
