"""Test doubles for the core lane: no provider SDK, no network, no real model."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from which_ai_model.contracts import Connection


class ScriptedChatModel(BaseChatModel):
    """A chat model that returns/raises the next scripted item and records each call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[Any] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "test-scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        self.calls.append("call")
        if not self.script:
            msg = "script exhausted"
            raise AssertionError(msg)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        message = item if isinstance(item, AIMessage) else AIMessage(content=str(item))
        return ChatResult(generations=[ChatGeneration(message=message)])


class StructuredFakeModel(BaseChatModel):
    """A chat model whose reply is fixed; it accepts tool binding like a provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: Any = Field(default_factory=lambda: AIMessage(content=""))
    bound: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "test-structured"

    def bind_tools(self, tools: Any, **kwargs: Any) -> StructuredFakeModel:
        del kwargs
        self.bound.extend(tools if isinstance(tools, list) else [tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        if isinstance(self.reply, BaseException):
            raise self.reply
        return ChatResult(generations=[ChatGeneration(message=self.reply)])


class FakeBuilder:
    """A stand-in for ``connect.to_langchain.chat_model``."""

    def __init__(self, models: dict[str, BaseChatModel] | None = None) -> None:
        self.models = models or {}
        self.calls: list[tuple[Connection, str, dict[str, Any]]] = []

    def __call__(self, connection: Connection, model_id: str, **options: Any) -> BaseChatModel:
        self.calls.append((connection, model_id, options))
        model = self.models.get(connection.id)
        if model is None:
            return ScriptedChatModel(script=[])
        return model


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def answer_with_tool(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[tool_call(name, args, call_id)])
