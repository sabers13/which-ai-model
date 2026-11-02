"""Vector stores: Postgres + pgvector, and a pure-numpy in-memory twin.

:class:`PgVectorStore` persists the indexed corpus for ``make rag-index``
and the live docs flow. :class:`InMemoryStore` implements the same three
methods over numpy so tests and replay builds (``make all``) never need
Postgres. Both satisfy the :class:`VectorStore` protocol.
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from which_ai_model.rag.corpus import Chunk, SearchHit

__all__ = ["InMemoryStore", "PgVectorStore", "VectorStore", "load_gold"]

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_table(table: str) -> str:
    if not _TABLE_RE.match(table):
        msg = f"table must be a plain SQL identifier, got {table!r}"
        raise ValueError(msg)
    return table


@runtime_checkable
class VectorStore(Protocol):
    """The three methods every passage store implements."""

    def ensure_schema(self, *, dimensions: int = 384) -> None: ...
    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...
    def search(self, embedding: Sequence[float], *, k: int = 4) -> list[SearchHit]: ...


class PgVectorStore:
    """Passage store on Postgres + pgvector, searched by cosine distance."""

    def __init__(self, dsn: str, *, table: str = "wam_chunks") -> None:
        self._dsn = dsn
        self._table = _check_table(table)

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def ensure_schema(self, *, dimensions: int = 384) -> None:
        """Create the extension, table, and a cosine HNSW index (ivfflat fallback)."""
        if dimensions <= 0:
            msg = f"dimensions must be positive, got {dimensions}"
            raise ValueError(msg)
        import psycopg
        from pgvector.psycopg import register_vector

        table = self._table
        with self._connect() as conn:
            # The extension must exist before the client registers the type.
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id TEXT PRIMARY KEY, "
                "text TEXT NOT NULL, "
                "source TEXT NOT NULL, "
                f"embedding vector({dimensions}))",
            )
            try:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw "
                    f"ON {table} USING hnsw (embedding vector_cosine_ops)",
                )
            except psycopg.errors.Error:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_embedding_ivfflat "
                    f"ON {table} USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 100)",
                )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or replace passages with their embeddings, matched by id."""
        rows = list(chunks)
        vectors = [list(map(float, row)) for row in embeddings]
        if len(rows) != len(vectors):
            msg = f"{len(rows)} chunks but {len(vectors)} embeddings"
            raise ValueError(msg)
        if not rows:
            return
        table = self._table
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {table} (id, text, source, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, "
                "source = EXCLUDED.source, "
                "embedding = EXCLUDED.embedding",
                [
                    (chunk.id, chunk.text, chunk.source, vector)
                    for chunk, vector in zip(rows, vectors, strict=True)
                ],
            )

    def search(self, embedding: Sequence[float], *, k: int = 4) -> list[SearchHit]:
        """Return the ``k`` closest passages by cosine similarity, best first."""
        if k < 0:
            msg = f"k must be non-negative, got {k}"
            raise ValueError(msg)
        if k == 0:
            return []
        vector = list(map(float, embedding))
        table = self._table
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, text, source, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, k),
            ).fetchall()
        return [
            SearchHit(chunk=Chunk(id=row[0], text=row[1], source=row[2]), score=float(row[3]))
            for row in rows
        ]


class InMemoryStore(PgVectorStore):
    """Pure-numpy store with the same interface as :class:`PgVectorStore`.

    Vectors are normalised on write and on query, so ranking by dot product
    matches the cosine ranking of the Postgres store. The ``dsn`` argument
    is accepted and ignored so the two stores stay interchangeable wherever
    a :class:`PgVectorStore` is expected; nothing is persisted.
    """

    def __init__(self, dsn: str = "", *, table: str = "wam_chunks") -> None:
        super().__init__(dsn, table=table)
        self._dimensions: int | None = None
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def table(self) -> str:
        """The logical table name (mirrors the Postgres table)."""
        return self._table

    def ensure_schema(self, *, dimensions: int = 384) -> None:
        """Record the expected embedding width; clears nothing."""
        if dimensions <= 0:
            msg = f"dimensions must be positive, got {dimensions}"
            raise ValueError(msg)
        self._dimensions = dimensions

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or replace passages with their embeddings, matched by id."""
        rows = list(chunks)
        vectors = list(embeddings)
        if len(rows) != len(vectors):
            msg = f"{len(rows)} chunks but {len(vectors)} embeddings"
            raise ValueError(msg)
        for chunk, embedding in zip(rows, vectors, strict=True):
            vector = np.asarray(list(map(float, embedding)), dtype=np.float64)
            if self._dimensions is not None and vector.shape != (self._dimensions,):
                msg = (
                    f"chunk {chunk.id}: expected {self._dimensions} dimensions, "
                    f"got {vector.shape[0] if vector.ndim == 1 else vector.shape}"
                )
                raise ValueError(msg)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector

    def search(self, embedding: Sequence[float], *, k: int = 4) -> list[SearchHit]:
        """Return the ``k`` closest passages by cosine similarity, best first."""
        if k < 0:
            msg = f"k must be non-negative, got {k}"
            raise ValueError(msg)
        if k == 0 or not self._chunks:
            return []
        query = np.asarray(list(map(float, embedding)), dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scored = [(chunk_id, float(query @ vector)) for chunk_id, vector in self._vectors.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchHit(chunk=self._chunks[chunk_id], score=score) for chunk_id, score in scored[:k]
        ]


def load_gold(path: Path = Path("data/rag/gold.jsonl")) -> list[dict[str, object]]:
    """Read the hand-written T-DOCS gold questions (answerable + abstain)."""
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = _json.loads(line)
            if not isinstance(row, dict):
                msg = f"{path}: every gold row must be a JSON object"
                raise ValueError(msg)
            rows.append(row)
    return rows
