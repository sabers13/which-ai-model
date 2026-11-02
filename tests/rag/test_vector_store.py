"""Tests for the vector stores.

The in-memory store (pure numpy) carries the offline behaviour: ranking,
replacement, and edge cases. Postgres round-trips run only when
``WAM_TEST_POSTGRES=1`` with a reachable pgvector service, since the
default suite runs with sockets blocked.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from which_ai_model.rag.corpus import Chunk, SearchHit
from which_ai_model.rag.store import InMemoryStore, PgVectorStore, VectorStore


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(id=chunk_id, text=text, source="s.md")


def _one_hot(dimensions: int, hot: int) -> list[float]:
    return [1.0 if i == hot else 0.0 for i in range(dimensions)]


def test_in_memory_store_satisfies_the_store_interface() -> None:
    assert isinstance(InMemoryStore(), VectorStore)
    # A subclass of PgVectorStore, so lanes typed on PgVectorStore accept it.
    assert isinstance(InMemoryStore(), PgVectorStore)


def test_in_memory_search_ranks_by_cosine_similarity() -> None:
    store = InMemoryStore()
    store.ensure_schema(dimensions=3)
    store.upsert(
        [_chunk("guide/a#0"), _chunk("guide/a#1"), _chunk("guide/a#2")],
        [_one_hot(3, 0), _one_hot(3, 1), [1.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=3)
    assert [hit.chunk.id for hit in hits] == ["guide/a#0", "guide/a#2", "guide/a#1"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score > hits[2].score
    assert all(isinstance(hit, SearchHit) for hit in hits)


def test_in_memory_search_normalises_magnitudes() -> None:
    store = InMemoryStore()
    store.ensure_schema(dimensions=2)
    store.upsert([_chunk("guide/a#0")], [[100.0, 0.0]])
    hits = store.search([0.5, 0.0], k=1)
    assert hits[0].score == pytest.approx(1.0)


def test_in_memory_search_empty_store_and_zero_k() -> None:
    store = InMemoryStore()
    store.ensure_schema(dimensions=2)
    assert store.search([1.0, 0.0], k=4) == []
    store.upsert([_chunk("guide/a#0")], [[1.0, 0.0]])
    assert store.search([1.0, 0.0], k=0) == []
    assert store.search([0.0, 0.0], k=4) == []


def test_in_memory_upsert_replaces_by_id() -> None:
    store = InMemoryStore()
    store.ensure_schema(dimensions=2)
    store.upsert([_chunk("guide/a#0", "old")], [[1.0, 0.0]])
    store.upsert([_chunk("guide/a#0", "new")], [[0.0, 1.0]])
    hits = store.search([0.0, 1.0], k=4)
    assert len(hits) == 1
    assert hits[0].chunk.text == "new"
    assert hits[0].score == pytest.approx(1.0)


def test_in_memory_upsert_validates_shapes() -> None:
    store = InMemoryStore()
    store.ensure_schema(dimensions=2)
    with pytest.raises(ValueError, match="chunks but"):
        store.upsert([_chunk("guide/a#0")], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="dimensions"):
        store.upsert([_chunk("guide/a#0")], [[1.0, 0.0, 0.0]])


def test_in_memory_search_rejects_negative_k() -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError, match="k must be non-negative"):
        store.search([1.0], k=-1)


def test_stores_reject_non_identifier_tables() -> None:
    with pytest.raises(ValueError, match="table"):
        InMemoryStore(table="wam_chunks; DROP TABLE wam_chunks;")
    with pytest.raises(ValueError, match="table"):
        PgVectorStore("postgresql://localhost/x", table="has space")


def test_pg_store_validates_without_a_server() -> None:
    store = PgVectorStore("postgresql://localhost:1/none", table="wam_chunks")
    with pytest.raises(ValueError, match="chunks but"):
        store.upsert([_chunk("guide/a#0")], [])
    with pytest.raises(ValueError, match="k must be non-negative"):
        store.search([1.0], k=-1)
    with pytest.raises(ValueError, match="dimensions"):
        store.ensure_schema(dimensions=0)


_NEEDS_POSTGRES = pytest.mark.skipif(
    os.environ.get("WAM_TEST_POSTGRES") != "1",
    reason="needs a live pgvector service (WAM_TEST_POSTGRES=1)",
)


@_NEEDS_POSTGRES
def test_pg_roundtrip_upsert_and_search() -> None:
    dsn = os.environ.get("WAM_PGVECTOR_DSN", "postgresql://wam:wam@localhost:5433/wam")
    store = PgVectorStore(dsn, table="wam_test_chunks")
    assert isinstance(store, VectorStore)
    store.ensure_schema(dimensions=3)
    chunks: Sequence[Chunk] = [_chunk("guide/t#0", "alpha"), _chunk("guide/t#1", "beta")]
    store.upsert(chunks, [_one_hot(3, 0), _one_hot(3, 1)])
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert [hit.chunk.id for hit in hits] == ["guide/t#0", "guide/t#1"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    # Upsert replaces the text while keeping the id stable.
    store.upsert([_chunk("guide/t#0", "alpha v2")], [_one_hot(3, 0)])
    assert store.search([1.0, 0.0, 0.0], k=1)[0].chunk.text == "alpha v2"
