"""An "after" tool: a LangChain chat model with structured output."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class RouteOut(BaseModel):
    intent: str


def route_message(message: str, model: BaseChatModel) -> RouteOut:
    structured = model.with_structured_output(RouteOut)
    return structured.invoke(message)
