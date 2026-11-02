"""T-DOCS eval: retrieval-only hit@4, citation precision, abstain-correct."""

from __future__ import annotations

import pytest
from test_graph_support import (
    MappingEmbedder,
    ScriptedDocsModel,
    ScriptedStore,
    docs_tool_call,
    make_hit,
    one_hot,
)

from which_ai_model.contracts import DocsOut, ModelRef, Prediction, TaskName
from which_ai_model.rag.eval import (
    DOCS_METRIC_KEYS,
    evaluate_rag,
    hit_at_4,
    retrieval_rows,
    score_docs,
)

QUESTIONS = [
    {
        "id": "t-docs-001",
        "question": "first question?",
        "answerable": True,
        "gold_ids": ["guide/a#0"],
    },
    {
        "id": "t-docs-002",
        "question": "second question?",
        "answerable": True,
        "gold_ids": ["guide/b#0"],
    },
    {"id": "t-docs-003", "question": "third question?", "answerable": False, "gold_ids": []},
]

EMBEDDER = MappingEmbedder(
    {
        "first question?": one_hot(0),
        "second question?": one_hot(1),
        "third question?": one_hot(2),
    }
)
STORE = ScriptedStore(
    {
        tuple(one_hot(0)): [make_hit("guide/a#0"), make_hit("guide/c#0")],
        tuple(one_hot(1)): [make_hit("guide/c#0"), make_hit("guide/a#0")],
        tuple(one_hot(2)): [make_hit("guide/a#0")],
    }
)

MODEL_REF = ModelRef(connection_id="test", model_id="fake-docs")


def _prediction(item_id: str, output: dict[str, object], raw_key: str = "k" * 64) -> Prediction:
    return Prediction(
        item_id=item_id,
        task=TaskName.DOCS,
        model=MODEL_REF,
        output=output,
        raw_key=raw_key,
    )


def test_retrieval_rows_and_hit_at_4_are_model_free() -> None:
    rows = retrieval_rows(QUESTIONS, store=STORE, embedder=EMBEDDER)

    assert [row.item_id for row in rows] == ["t-docs-001", "t-docs-002", "t-docs-003"]
    assert rows[0].hit is True
    assert rows[1].hit is False  # gold b#0 is not in the top four
    assert rows[2].answerable is False
    assert rows[0].top_score == pytest.approx(0.9)
    assert hit_at_4(QUESTIONS, store=STORE, embedder=EMBEDDER) == pytest.approx(0.5)


def test_evaluate_rag_scores_three_metrics() -> None:
    model = ScriptedDocsModel(
        replies=[
            docs_tool_call("A.", ["guide/a#0"]),
            docs_tool_call("C.", ["guide/a#0"]),  # cited but not the gold id
            docs_tool_call("The guide does not cover it.", [], abstained=True),
        ]
    )

    metrics = evaluate_rag(QUESTIONS, store=STORE, model=model, embedder=EMBEDDER)

    assert metrics["hit_at_4"] == pytest.approx(0.5)
    assert metrics["citation_precision"] == pytest.approx(0.5)
    assert metrics["abstain_correct"] == pytest.approx(1.0)
    assert metrics["schema_valid_rate"] == pytest.approx(1.0)
    assert metrics["answerable_abstain_rate"] == pytest.approx(0.0)
    assert metrics["n"] == 3
    assert metrics["n_answerable"] == 2
    assert metrics["n_unanswerable"] == 1
    assert set(DOCS_METRIC_KEYS) <= set(metrics)
    assert len(model.calls) == 3


def test_evaluate_rag_counts_a_false_abstention_as_wrong() -> None:
    model = ScriptedDocsModel(
        replies=[
            docs_tool_call("No idea.", [], abstained=True),
            docs_tool_call("C.", ["guide/b#0"]),
            docs_tool_call("No idea.", [], abstained=True),
        ]
    )

    metrics = evaluate_rag(QUESTIONS, store=STORE, model=model, embedder=EMBEDDER)

    assert metrics["abstain_correct"] == pytest.approx(1.0)
    assert metrics["answerable_abstain_rate"] == pytest.approx(0.5)
    # The second citation is a gold id but it was not retrieved, so the graph drops it;
    # neither answerable item then earns citation precision.
    assert metrics["citation_precision"] == pytest.approx(0.0)


def test_score_docs_uses_predictions_and_optional_retrieval() -> None:
    predictions = [
        _prediction(
            "t-docs-001",
            DocsOut(answer="A.", cited_ids=["guide/a#0"], abstained=False).model_dump(mode="json"),
        ),
        _prediction(
            "t-docs-002",
            DocsOut(answer="C.", cited_ids=["guide/a#0"], abstained=False).model_dump(mode="json"),
        ),
        _prediction(
            "t-docs-003",
            DocsOut(answer="No.", cited_ids=[], abstained=True).model_dump(mode="json"),
        ),
    ]
    retrieved = {
        "t-docs-001": ["guide/a#0", "guide/c#0"],
        "t-docs-002": ["guide/c#0", "guide/a#0"],
        "t-docs-003": ["guide/a#0"],
    }

    metrics = score_docs(predictions, QUESTIONS, retrieved=retrieved)

    assert metrics["hit_at_4"] == pytest.approx(0.5)
    assert metrics["citation_precision"] == pytest.approx(0.5)
    assert metrics["abstain_correct"] == pytest.approx(1.0)
    assert metrics["schema_valid_rate"] == pytest.approx(1.0)


def test_score_docs_without_retrieval_omits_hit_at_4() -> None:
    metrics = score_docs([], QUESTIONS)
    assert "hit_at_4" not in metrics
    assert metrics["citation_precision"] == pytest.approx(0.0)
    assert metrics["abstain_correct"] == pytest.approx(0.0)
    assert metrics["schema_valid_rate"] == pytest.approx(0.0)


def test_score_docs_defaults_to_the_committed_gold_set() -> None:
    metrics = score_docs([])
    assert metrics["n"] == 40
    assert metrics["n_answerable"] == 35
    assert metrics["n_unanswerable"] == 5
    assert metrics["abstain_correct"] == pytest.approx(0.0)


def test_malformed_prediction_output_is_not_schema_valid() -> None:
    prediction = _prediction("t-docs-001", {"answer": 3, "cited_ids": "nope", "abstained": False})
    metrics = score_docs([prediction], QUESTIONS[:1])
    assert metrics["schema_valid_rate"] == pytest.approx(0.0)
    assert metrics["citation_precision"] == pytest.approx(0.0)


def test_hit_at_4_rejects_an_empty_question_set() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        hit_at_4([], store=STORE, embedder=EMBEDDER)


def test_hit_at_4_needs_an_answerable_question() -> None:
    with pytest.raises(ValueError, match="at least one answerable"):
        hit_at_4([QUESTIONS[2]], store=STORE, embedder=EMBEDDER)


def test_embedder_returning_the_wrong_count_is_rejected() -> None:
    class ShortEmbedder:
        def __call__(self, texts: list[str]) -> list[list[float]]:
            del texts
            return [one_hot(0)]

    with pytest.raises(ValueError, match="vectors for"):
        retrieval_rows(QUESTIONS, store=STORE, embedder=ShortEmbedder())
