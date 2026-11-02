"""The migrated L4 tool: DocsOut answers and the runner's Prediction contract."""

from __future__ import annotations

import os
from pathlib import Path

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

from which_ai_model.contracts import DocsOut, ModelRef, Prediction, TaskName
from which_ai_model.rag.graph import build_rag_graph
from which_ai_model.rag.store import PgVectorStore
from which_ai_model.tools.docs import answer_docs, build_in_memory_store, predict

MODEL_REF = ModelRef(connection_id="test", model_id="fake-docs")
CHUNKS = [
    make_chunk("guide/a#0", "alpha " * 20),
    make_chunk("guide/b#0", "beta " * 20),
]


def test_predict_returns_a_scored_docs_prediction() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(replies=[docs_tool_call("Use the schema.", ["guide/a#0"])])

    prediction = predict(
        {"id": "t-docs-001", "question": "alpha"},
        MODEL_REF,
        model=model,
        store=store,
        embedder=token_embed,
    )

    assert isinstance(prediction, Prediction)
    assert prediction.item_id == "t-docs-001"
    assert prediction.task is TaskName.DOCS
    assert prediction.model == MODEL_REF
    assert prediction.output["answer"] == "Use the schema."
    assert prediction.output["cited_ids"] == ["guide/a#0"]
    assert prediction.output["abstained"] is False
    assert len(prediction.raw_key) == 64


def test_predict_accepts_runner_item_and_message_keys() -> None:
    store = make_store(CHUNKS)
    model = CitingDocsModel(answer="Answer.")

    by_item = predict(
        {"item_id": "item-7", "docs_question": "alpha"},
        MODEL_REF,
        model=model,
        store=store,
        embedder=token_embed,
    )
    assert by_item.item_id == "item-7"
    assert by_item.output["cited_ids"] == ["guide/a#0"]

    by_message = predict(
        {"item_id": "item-8", "message": "alpha"},
        MODEL_REF,
        model=model,
        store=store,
        embedder=token_embed,
    )
    assert by_message.item_id == "item-8"


def test_predict_string_item_gets_a_stable_id() -> None:
    store = make_store(CHUNKS)
    prediction = predict(
        "alpha", MODEL_REF, model=CitingDocsModel(), store=store, embedder=token_embed
    )
    again = predict("alpha", MODEL_REF, model=CitingDocsModel(), store=store, embedder=token_embed)
    assert prediction.item_id.startswith("docs-")
    assert prediction.item_id == again.item_id


def test_predict_rejects_an_item_without_a_question() -> None:
    with pytest.raises(KeyError, match="no question"):
        predict({"id": "x"}, MODEL_REF, model=CitingDocsModel(), store=make_store(CHUNKS))


def test_answer_docs_returns_the_contract_and_validates_citations() -> None:
    store = make_store(CHUNKS)
    model = ScriptedDocsModel(replies=[docs_tool_call("Answer.", ["guide/ghost#9", "guide/a#0"])])

    docs = answer_docs("alpha", model, store=store, embedder=token_embed)

    assert docs == DocsOut(answer="Answer.", cited_ids=["guide/a#0"], abstained=False)


def test_build_in_memory_store_indexes_markdown_pages(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("# Page\n\n## Alpha\n\nalpha alpha alpha\n\n## Beta\n\nbeta beta\n", "utf-8")

    store = build_in_memory_store(tmp_path, embedder=token_embed)
    hits = store.search(token_embed(["alpha"])[0], k=2)

    assert hits[0].chunk.id == "guide/page#1"
    assert hits[0].chunk.source == "page.md"


def test_pgvector_store_is_accepted_by_the_graph_without_connecting() -> None:
    graph = build_rag_graph(
        PgVectorStore("postgresql://unused/unused"),
        CitingDocsModel(),
        embedder=token_embed,
    )
    assert isinstance(graph, CompiledStateGraph)


_NEEDS_POSTGRES = pytest.mark.skipif(
    os.environ.get("WAM_TEST_POSTGRES") != "1",
    reason="needs a live pgvector service (WAM_TEST_POSTGRES=1)",
)


@_NEEDS_POSTGRES
def test_pgvector_graph_path_answers_like_the_in_memory_store() -> None:
    dsn = os.environ.get("WAM_PGVECTOR_DSN", "postgresql://wam:wam@localhost:5433/wam")
    store = PgVectorStore(dsn, table="wam_test_graph_chunks")
    store.ensure_schema(dimensions=64)
    chunks = [make_chunk("guide/pg#0", "alpha " * 20), make_chunk("guide/pg#1", "beta " * 20)]
    store.upsert(chunks, token_embed([chunk.text for chunk in chunks]))

    docs = answer_docs("alpha", CitingDocsModel(), store=store, embedder=token_embed)

    assert docs.cited_ids == ["guide/pg#0"]
    assert docs.abstained is False
