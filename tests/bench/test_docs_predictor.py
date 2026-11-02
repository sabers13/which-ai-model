"""T-DOCS wiring: retrieval context and citation filtering through the runner's call."""

from __future__ import annotations

from typing import Any

from which_ai_model.bench.run import DocsPredictor
from which_ai_model.contracts import ModelRef
from which_ai_model.rag.corpus import Chunk, SearchHit


class FakeStore:
    def search(self, embedding: Any, *, k: int = 4) -> list[SearchHit]:
        del embedding
        return [
            SearchHit(chunk=Chunk(id="p1", text="first passage", source="a.md"), score=0.9),
            SearchHit(chunk=Chunk(id="p2", text="second passage", source="b.md"), score=0.8),
        ][:k]


def _call(content: str):
    def call(messages):
        del messages
        return type(
            "Answer",
            (),
            {"key": "key-1", "response": {"content": content, "tool_calls": [], "usage": {}}},
        )()

    return call


def test_docs_predictor_adds_retrieved_ids_and_drops_unknown_citations(monkeypatch) -> None:
    monkeypatch.setattr("which_ai_model.rag.embed.embed_texts", lambda texts: [[0.0, 1.0]])
    predictor = DocsPredictor(store=FakeStore(), k=2)
    prediction = predictor.predict(
        {"item_id": "q1", "question": "how?"},
        ModelRef(connection_id="c", model_id="m"),
        _call('{"answer": "a", "cited_ids": ["p1", "nope"], "abstained": false}'),
    )
    assert prediction.output["retrieved_ids"] == ["p1", "p2"]
    assert prediction.output["cited_ids"] == ["p1"]
    assert prediction.raw_key == "key-1"


def test_docs_predictor_keeps_retrieval_when_the_answer_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr("which_ai_model.rag.embed.embed_texts", lambda texts: [[0.0, 1.0]])
    predictor = DocsPredictor(store=FakeStore(), k=2)
    prediction = predictor.predict(
        {"item_id": "q2", "question": "how?"},
        ModelRef(connection_id="c", model_id="m"),
        _call("not json"),
    )
    assert prediction.output == {"retrieved_ids": ["p1", "p2"]}
