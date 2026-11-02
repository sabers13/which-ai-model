"""The shared model layer: build a chat model, retry transport errors, fall back.

``make_chat_model`` is a thin policy wrapper around T01's
``connect.to_langchain.chat_model`` (the protocol → class mapping lives there): it
enforces a 1,024-token output floor (``PLAN.md`` §7), ``max_retries=0`` (retries belong
here, not in the provider SDK) and passes the task's session id through.

``get_model`` adds the two behaviours ``PLAN.md`` §5 asks for: up to five attempts with
exponential backoff and jitter, honouring ``Retry-After``, and a fallback model for
``RATE_LIMITED`` / ``ENDPOINT_UNREACHABLE`` only. A schema-invalid answer or a 400 is
never retried.
"""

from __future__ import annotations

import asyncio
import importlib
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from which_ai_model.contracts import Connection, ModelRef

from .errors import RETRYABLE_REASONS, classify_exception, retry_after_seconds

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "FallbackChatModel",
    "ModelBuilder",
    "get_model",
    "make_agent",
    "make_chat_model",
]

#: Every call allows at least this many output tokens (``PLAN.md`` §7).
DEFAULT_MAX_TOKENS = 1024
#: Attempts per model, including the first call.
MAX_ATTEMPTS = 5


class ModelBuilder(Protocol):
    """T01's ``connect.to_langchain.chat_model`` signature."""

    def __call__(self, connection: Connection, model_id: str, **options: Any) -> BaseChatModel: ...


def _load_builder() -> ModelBuilder:
    """Import T01's builder lazily so core stays importable before/without connect."""
    try:
        module = importlib.import_module("which_ai_model.connect.to_langchain")
    except ModuleNotFoundError as exc:  # pragma: no cover - lane integration guard
        msg = (
            "which_ai_model.connect.to_langchain is not available; T01 provides the "
            "protocol -> chat-model mapping. Tests pass a fake builder."
        )
        raise RuntimeError(msg) from exc
    builder = getattr(module, "chat_model", None)
    if builder is None:  # pragma: no cover - lane integration guard
        msg = "which_ai_model.connect.to_langchain.chat_model is missing"
        raise RuntimeError(msg)
    return cast(ModelBuilder, builder)


def make_chat_model(
    connection: Connection,
    model_id: str,
    *,
    reasoning: str = "minimal",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = 0.0,
    session_id: str | None = None,
    builder: ModelBuilder | None = None,
) -> BaseChatModel:
    """Build one provider chat model for a connection.

    ``max_tokens`` may not go below 1,024: providers that spend output tokens on
    reasoning can return nothing from a small budget (``PLAN.md`` §7, Muse Spark).
    ``temperature`` is sent as given; ``None`` leaves the parameter out where the
    provider rejects it (``PLAN.md`` §7, "temperature 0 where the API allows").
    """
    if max_tokens < DEFAULT_MAX_TOKENS:
        msg = f"max_tokens must be >= {DEFAULT_MAX_TOKENS}; got {max_tokens}"
        raise ValueError(msg)
    options: dict[str, Any] = {
        "reasoning": reasoning,
        "max_tokens": max_tokens,
        "max_retries": 0,
        "temperature": temperature,
    }
    if session_id is not None:
        options["session_id"] = session_id
    build = builder or _load_builder()
    return build(connection, model_id, **options)


def _connection_map(
    connections: Mapping[str, Connection] | Sequence[Connection] | None,
) -> dict[str, Connection]:
    if connections is None:
        module = importlib.import_module("which_ai_model.connect")
        loaded = cast(Sequence[Connection], module.load_connections())
        return {connection.id: connection for connection in loaded}
    if isinstance(connections, Mapping):
        return dict(connections)
    return {connection.id: connection for connection in connections}


class FallbackChatModel(BaseChatModel):
    """A chat model that retries transport errors and can fall back to a second model.

    Construct it with :func:`get_model`; ``primary`` and ``fallback`` are real provider
    chat models with their own base URL, credential reference and headers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: BaseChatModel
    fallback: BaseChatModel | None = None
    max_attempts: int = MAX_ATTEMPTS
    base_delay: float = 0.5
    max_delay: float = 30.0
    sleep_fn: Callable[[float], None] = Field(default=time.sleep, exclude=True)
    rng: random.Random = Field(default_factory=random.Random, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "which-ai-model-fallback"

    @property
    def models(self) -> list[BaseChatModel]:
        """Primary first, then the optional fallback."""
        return [self.primary] if self.fallback is None else [self.primary, self.fallback]

    def delay_for(self, attempt: int, error: BaseException) -> float:
        """Honour ``Retry-After``; otherwise exponential backoff with jitter."""
        asked = retry_after_seconds(error)
        if asked is not None:
            return max(asked, 0.0)
        raw = float(self.base_delay) * float(2**attempt)
        jittered = raw * float(self.rng.uniform(0.5, 1.5))
        return float(min(jittered, self.max_delay))

    def _call_models(self, call: Callable[[BaseChatModel], Any]) -> Any:
        last_error: BaseException | None = None
        for model in self.models:
            for attempt in range(self.max_attempts):
                try:
                    return call(model)
                except Exception as exc:
                    reason = classify_exception(exc)
                    if reason not in RETRYABLE_REASONS:
                        raise
                    last_error = exc
                    if attempt + 1 >= self.max_attempts:
                        break
                    self.sleep_fn(self.delay_for(attempt, exc))
        if last_error is None:  # pragma: no cover - models is never empty
            msg = "no model configured"
            raise RuntimeError(msg)
        raise last_error

    async def _acall_models(self, call: Callable[[BaseChatModel], Any]) -> Any:
        last_error: BaseException | None = None
        for model in self.models:
            for attempt in range(self.max_attempts):
                try:
                    return await call(model)
                except Exception as exc:
                    reason = classify_exception(exc)
                    if reason not in RETRYABLE_REASONS:
                        raise
                    last_error = exc
                    if attempt + 1 >= self.max_attempts:
                        break
                    await asyncio.sleep(self.delay_for(attempt, exc))
        if last_error is None:  # pragma: no cover - models is never empty
            msg = "no model configured"
            raise RuntimeError(msg)
        raise last_error

    def invoke(
        self,
        input: Any,  # noqa: A002 - matches BaseChatModel.invoke
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> AIMessage:
        return cast(
            AIMessage,
            self._call_models(
                lambda model: model.invoke(input, config=config, stop=stop, **kwargs)
            ),
        )

    async def ainvoke(
        self,
        input: Any,  # noqa: A002 - matches BaseChatModel.invoke
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> AIMessage:
        return cast(
            AIMessage,
            await self._acall_models(
                lambda model: model.ainvoke(input, config=config, stop=stop, **kwargs)
            ),
        )

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _ResilientStructured:
        """Keep retries/fallback for structured calls too."""
        return _ResilientStructured(self, schema, include_raw=include_raw, options=kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Tool binding is a property of the primary provider's wire format."""
        return self.primary.bind_tools(tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        message = self.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        message = await self.ainvoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _ResilientStructured:
    """Structured-output runnable with the same retry/fallback policy as the model."""

    def __init__(
        self,
        owner: FallbackChatModel,
        schema: Any,
        *,
        include_raw: bool,
        options: dict[str, Any],
    ) -> None:
        self.owner = owner
        self.schema = schema
        self.include_raw = include_raw
        self.options = options
        self._runnables: dict[int, Any] = {}

    def _runnable(self, model: BaseChatModel) -> Any:
        key = id(model)
        runnable = self._runnables.get(key)
        if runnable is None:
            runnable = model.with_structured_output(
                self.schema, include_raw=self.include_raw, **self.options
            )
            self._runnables[key] = runnable
        return runnable

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return self.owner._call_models(
            lambda model: self._runnable(model).invoke(input, config=config, **kwargs)
        )

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return await self.owner._acall_models(
            lambda model: self._runnable(model).ainvoke(input, config=config, **kwargs)
        )


def get_model(
    ref: ModelRef,
    *,
    fallback: ModelRef | None = None,
    connections: Mapping[str, Connection] | Sequence[Connection] | None = None,
    builder: ModelBuilder | None = None,
    reasoning: str = "minimal",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    session_id: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> FallbackChatModel:
    """Build a runnable for ``ref`` with retries and an optional fallback model.

    ``connections`` is normally omitted: T01's ``connect.load_connections`` is used.
    Tests pass a small mapping plus a fake ``builder``.
    """
    table = _connection_map(connections)
    try:
        primary_connection = table[ref.connection_id]
    except KeyError as exc:
        msg = f"unknown connection {ref.connection_id!r}"
        raise KeyError(msg) from exc
    primary = make_chat_model(
        primary_connection,
        ref.model_id,
        reasoning=reasoning,
        max_tokens=max_tokens,
        session_id=session_id,
        builder=builder,
    )
    backup: BaseChatModel | None = None
    if fallback is not None:
        try:
            fallback_connection = table[fallback.connection_id]
        except KeyError as exc:
            msg = f"unknown fallback connection {fallback.connection_id!r}"
            raise KeyError(msg) from exc
        backup = make_chat_model(
            fallback_connection,
            fallback.model_id,
            reasoning=reasoning,
            max_tokens=max_tokens,
            session_id=session_id,
            builder=builder,
        )
    return FallbackChatModel(
        primary=primary,
        fallback=backup,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        sleep_fn=sleep_fn,
        rng=rng or random.Random(),
    )


def _retry_transport_error(error: BaseException) -> bool:
    """Only ``RATE_LIMITED``/``ENDPOINT_UNREACHABLE`` may be retried (``PLAN.md`` §5)."""
    return classify_exception(error) in RETRYABLE_REASONS


def make_agent(
    primary: Connection,
    fallback: Connection | None,
    model_id: str,
    tools: Sequence[Any],
    *,
    system_prompt: str,
    response_format: Any,
    builder: ModelBuilder | None = None,
    session_id: str | None = None,
) -> Any:
    """``create_agent`` with fallback + retry middleware and no output self-repair."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ToolStrategy
    from pydantic import BaseModel

    primary_model = make_chat_model(primary, model_id, session_id=session_id, builder=builder)
    middleware: list[Any] = []
    if fallback is not None:
        middleware.append(
            ModelFallbackMiddleware(
                make_chat_model(fallback, model_id, session_id=session_id, builder=builder)
            )
        )
    middleware.append(
        ModelRetryMiddleware(
            max_retries=2,
            retry_on=_retry_transport_error,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=30.0,
            on_failure="error",
        )
    )
    structured: Any = response_format
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        structured = ToolStrategy(response_format, handle_errors=False)
    return create_agent(
        model=primary_model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=middleware,
        response_format=structured,
    )
