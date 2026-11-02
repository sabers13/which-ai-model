"""The L4 graph: retrieve k=4, answer through DocsOut, validate citations, abstain."""

from __future__ import annotations

import pytest
from langgraph.graph.state import CompiledStateGraph
from test_graph_support import (
    CitingDocsModel,
    ScriptedDocsModel,
    docs_tool_call,
    make_chunk,
    make_store,
    token_embed,
)

from which_ai_model.contracts import DocsOut
from which_ai_model.rag.graph import (
    ABSTAIN_ANSWER,
    answer_question,
    answer_question_with_metadata,
    build_rag_graph,
)

CHUNKS = [
    make_chunk("guide/a#0", "alpha " * 20),
    make_chunk("guide/b#0", "beta " * 20),
    make_chunk("guide/c#0", "gamma " * 20),
    make_chunk("guide/d#0", "delta " * 20),
    make_chunk("guide/e#0", "epsilon " * 20),
    make_chunk("guide/f#0", "zeta " * 20),
]


def test_retrieves_four_passages_and_keeps_a_retrieved_citation() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(replies=[docs_tool_call("Use the schema.", ["guide/a#0"])])

    docs, metadata = answer_question_with_metadata(
        "alpha", store=store, model=model, embedder=token_embed
    )

    assert docs == DocsOut(answer="Use the schema.", cited_ids=["guide/a#0"], abstained=False)
    assert metadata.retrieved_ids[0] == "guide/a#0"
    assert len(metadata.retrieved_ids) == 4
    assert metadata.dropped_ids == ()
    assert metadata.schema_valid is True
    assert metadata.abstained is False
    assert metadata.top_score is not None
    assert metadata.top_score > 0.5
    assert metadata.request_digest is not None
    assert len(metadata.request_digest) == 64
    assert len(model.calls) == 1
    assert "[guide/a#0]" in str(model.calls[0][-1].content)


def test_cited_ids_outside_the_retrieved_set_are_dropped_and_counted() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(
        replies=[docs_tool_call("Answer.", ["guide/ghost#9", "guide/a#0", "guide/ghost#9"])]
    )

    docs, metadata = answer_question_with_metadata(
        "alpha", store=store, model=model, embedder=token_embed
    )

    assert docs.cited_ids == ["guide/a#0"]
    assert docs.abstained is False
    assert metadata.dropped_ids == ("guide/ghost#9",)


def test_no_passage_above_the_threshold_abstains_without_a_model_call() -> None:
    store = make_store([make_chunk("guide/b#0", "beta " * 20)])
    model = ScriptedDocsModel(replies=[])

    docs, metadata = answer_question_with_metadata(
        "alpha", store=store, model=model, embedder=token_embed, min_score=0.5
    )

    assert docs == DocsOut(answer=ABSTAIN_ANSWER, cited_ids=[], abstained=True)
    assert metadata.abstain_reason == "retrieval-threshold"
    assert metadata.retrieved_ids  # retrieval still happened; only the answer skipped
    assert model.calls == []


def test_empty_store_abstains() -> None:
    store = make_store([])
    model = ScriptedDocsModel(replies=[])

    docs, metadata = answer_question_with_metadata(
        "anything", store=store, model=model, embedder=token_embed
    )

    assert docs.abstained is True
    assert metadata.retrieved_ids == ()
    assert metadata.request_digest is None
    assert model.calls == []


def test_model_abstention_is_explicit_and_keeps_no_citations() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(
        replies=[
            docs_tool_call(
                "The guide does not cover shipping fees.",
                ["guide/a#0"],
                abstained=True,
            )
        ]
    )

    docs, metadata = answer_question_with_metadata(
        "alpha", store=store, model=model, embedder=token_embed
    )

    assert docs.abstained is True
    assert docs.cited_ids == ["guide/a#0"], "a retrieved citation is not invented"
    assert metadata.abstain_reason == "model-abstained"


def test_schema_invalid_answer_abstains_instead_of_raising() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(replies=['{"answer": "json in free text"}'])

    docs, metadata = answer_question_with_metadata(
        "alpha", store=store, model=model, embedder=token_embed
    )

    assert docs == DocsOut(answer="", cited_ids=[], abstained=True)
    assert metadata.schema_valid is False
    assert metadata.abstain_reason == "schema-invalid"
    assert metadata.request_digest is not None


def test_answer_question_returns_only_the_contract() -> None:
    store = make_store(CHUNKS)
    model = CitingDocsModel(answer="Retrieved answer.")

    docs = answer_question("alpha", store=store, model=model, embedder=token_embed)

    assert isinstance(docs, DocsOut)
    assert docs.answer == "Retrieved answer."
    assert docs.cited_ids == ["guide/a#0"]
    assert docs.abstained is False


def test_build_rag_graph_returns_a_compiled_graph() -> None:
    graph = build_rag_graph(make_store(CHUNKS), CitingDocsModel(), embedder=token_embed)
    assert isinstance(graph, CompiledStateGraph)


def test_build_rag_graph_rejects_bad_arguments() -> None:
    store = make_store(CHUNKS)
    model = CitingDocsModel()
    with pytest.raises(ValueError, match="k must be positive"):
        build_rag_graph(store, model, k=0)
    with pytest.raises(ValueError, match="min_score"):
        build_rag_graph(store, model, min_score=-0.1)
    with pytest.raises(ValueError, match="min_score"):
        build_rag_graph(store, model, min_score=1.1)
