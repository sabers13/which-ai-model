"""Structured output: schema-valid, schema-invalid, empty/truncated; no JSON-by-prompt."""

from __future__ import annotations

from core_fakes import StructuredFakeModel, answer_with_tool
from langchain_core.messages import AIMessage

from which_ai_model.contracts import RouteOut
from which_ai_model.core.structured import structured
from which_ai_model.core.usage import TRUNCATED, empty_answer_reason


def test_valid_tool_call_is_parsed() -> None:
    model = StructuredFakeModel(reply=answer_with_tool("RouteOut", {"intent": "card_arrival"}))
    parsed, valid, raw = structured(model, RouteOut, "my card never arrived")
    assert isinstance(parsed, RouteOut)
    assert parsed.intent == "card_arrival"
    assert valid is True
    assert raw == ""
    assert model.bound, "the schema must go through the tool channel, not a prompt"


def test_wrong_arguments_are_invalid_not_raised() -> None:
    model = StructuredFakeModel(reply=answer_with_tool("RouteOut", {"wrong": "field"}))
    parsed, valid, raw = structured(model, RouteOut, "hello")
    assert parsed is None
    assert valid is False
    assert raw == ""


def test_json_in_free_text_is_never_parsed() -> None:
    model = StructuredFakeModel(reply=AIMessage(content='{"intent": "card_arrival"}'))
    parsed, valid, raw = structured(model, RouteOut, "my card never arrived")
    assert parsed is None
    assert valid is False
    assert raw == '{"intent": "card_arrival"}'


def test_empty_length_answer_is_schema_invalid_not_an_exception() -> None:
    reply = AIMessage(content="", response_metadata={"finish_reason": "length"})
    model = StructuredFakeModel(reply=reply)
    parsed, valid, raw = structured(model, RouteOut, "hello")
    assert parsed is None
    assert valid is False
    assert raw == ""
    assert empty_answer_reason(reply) == TRUNCATED
