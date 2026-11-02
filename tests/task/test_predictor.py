"""The generic predictor: one prompt, one schema, the runner's call closure."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from which_ai_model.bench.run import Message, ModelAnswer
from which_ai_model.contracts import ModelRef
from which_ai_model.task import SpecError, SpecPredictor, load_spec

FIXTURE = Path("tests/fixtures/task")
MODEL = ModelRef(connection_id="fake", model_id="fake-model")


class FakeCalls:
    """A fake ``call`` that records messages and returns one canned response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.messages: list[Sequence[Message]] = []

    def __call__(self, messages: Sequence[Message]) -> ModelAnswer:
        self.messages.append(messages)
        return ModelAnswer(key="recorded-key", response=self.response, latency_ms=1)


def _predictor(repo_root: Path) -> SpecPredictor:
    return SpecPredictor(load_spec(repo_root / FIXTURE / "example.toml"), root=repo_root)


def _item() -> dict[str, Any]:
    return {
        "item_id": "item-01",
        "id": "item-01",
        "text": "My invoice was charged twice.",
    }


def test_predictor_renders_one_prompt_and_parses_the_tool_call(repo_root: Path) -> None:
    predictor = _predictor(repo_root)
    calls = FakeCalls(
        {"tool_calls": [{"args": {"queue": "billing", "urgent": True}}], "content": ""}
    )

    prediction = predictor.predict(_item(), MODEL, calls)

    assert prediction.item_id == "item-01"
    assert prediction.task == "example"
    assert prediction.model == MODEL
    assert prediction.output == {"queue": "billing", "urgent": True}
    assert prediction.raw_key == "recorded-key"
    assert predictor.prompt_version == "e1"
    assert len(calls.messages) == 1
    [message] = calls.messages[0]
    assert message["role"] == "user"
    assert "- billing: Invoices" in message["content"]
    assert "My invoice was charged twice." in message["content"]


def test_predictor_accepts_a_json_content_answer(repo_root: Path) -> None:
    predictor = _predictor(repo_root)
    calls = FakeCalls({"content": '{"queue": "technical", "urgent": false}'})
    prediction = predictor.predict(_item(), MODEL, calls)
    assert prediction.output == {"queue": "technical", "urgent": False}


@pytest.mark.parametrize(
    "response",
    [
        {"content": ""},
        {"content": "not json at all"},
        {"content": '{"queue": "sales", "urgent": false}'},
        {"tool_calls": [{"args": {"queue": "billing"}}]},
        {},
    ],
)
def test_unparseable_or_incomplete_answers_are_empty(
    repo_root: Path, response: dict[str, Any]
) -> None:
    predictor = _predictor(repo_root)
    prediction = predictor.predict(_item(), MODEL, FakeCalls(response))
    assert prediction.output == {}


def test_item_id_falls_back_to_the_spec_id_field(repo_root: Path) -> None:
    predictor = _predictor(repo_root)
    item = {"id": 42, "text": "A message"}
    prediction = predictor.predict(item, MODEL, FakeCalls({"content": "{}"}))
    assert prediction.item_id == "42"


def test_missing_text_is_a_plain_error(repo_root: Path) -> None:
    predictor = _predictor(repo_root)
    with pytest.raises(SpecError, match="no 'text' value"):
        predictor.predict({"item_id": "item-01"}, MODEL, FakeCalls({}))


def test_missing_item_id_is_a_plain_error(repo_root: Path) -> None:
    predictor = _predictor(repo_root)
    with pytest.raises(SpecError, match="no 'id' value"):
        predictor.predict({"text": "A message"}, MODEL, FakeCalls({}))
