"""L4 after: retrieval-grounded docs answers with validated citations.

The migrated docs helper replaces L4's whole-guide prompt with the T09 RAG graph:
retrieve four passages, answer through :class:`~which_ai_model.contracts.DocsOut`, keep
only cited ids that were retrieved. :func:`predict` is the task predictor the benchmark
runner calls; :func:`answer_docs` is the ``docs/INTERFACES.md`` §8 entry point.

Two stores work unchanged: the default in-memory store built from ``docs/guide`` (replay,
tests, ``make all``) and a :class:`~which_ai_model.rag.store.PgVectorStore` built by
``make rag-index`` (live). Callers select one with ``store=``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from which_ai_model.contracts import DocsOut, ModelRef, Prediction, TaskName
from which_ai_model.core import get_model
from which_ai_model.core.record import recording_key, request_digest
from which_ai_model.rag.corpus import chunk_documents, load_corpus
from which_ai_model.rag.embed import EMBED_DIMENSIONS, embed_texts
from which_ai_model.rag.graph import (
    DEFAULT_K,
    DEFAULT_MIN_SCORE,
    DOCS_PROMPT_VERSION,
    answer_question_with_metadata,
)
from which_ai_model.rag.store import InMemoryStore, VectorStore

__all__ = ["GUIDE_ROOT", "answer_docs", "build_in_memory_store", "predict"]

#: The RAG corpus (``PLAN.md`` §6).
GUIDE_ROOT = Path("docs/guide")

Embedder = Callable[[Sequence[str]], list[list[float]]]

_default_store: InMemoryStore | None = None


def build_in_memory_store(
    root: Path = GUIDE_ROOT,
    *,
    embedder: Embedder = embed_texts,
) -> InMemoryStore:
    """Chunk and embed the guide into a :class:`~which_ai_model.rag.store.InMemoryStore`."""
    chunks = chunk_documents(load_corpus(root))
    vectors = embedder([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        msg = f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        raise ValueError(msg)
    store = InMemoryStore()
    store.ensure_schema(dimensions=len(vectors[0]) if vectors else EMBED_DIMENSIONS)
    store.upsert(chunks, vectors)
    return store


def default_store() -> InMemoryStore:
    """The process-wide in-memory guide index, built once with the real embedder."""
    global _default_store
    if _default_store is None:
        _default_store = build_in_memory_store()
    return _default_store


def answer_docs(
    question: str,
    model: BaseChatModel,
    *,
    store: VectorStore | None = None,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> DocsOut:
    """Answer one docs question from retrieved passages, with validated citations."""
    docs, _metadata = answer_question_with_metadata(
        question,
        store=store if store is not None else default_store(),
        model=model,
        k=k,
        min_score=min_score,
        embedder=embedder,
    )
    return docs


def _question(item: Mapping[str, object] | str) -> str:
    if isinstance(item, str):
        return item
    for name in ("question", "docs_question", "message"):
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value
    msg = "docs item has no question/docs_question/message field"
    raise KeyError(msg)


def _item_id(item: Mapping[str, object] | str, question: str) -> str:
    if not isinstance(item, str):
        for name in ("id", "item_id"):
            value = item.get(name)
            if isinstance(value, str) and value:
                return value
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"docs-{digest}"


def predict(
    item: Mapping[str, object] | str,
    model_ref: ModelRef,
    *,
    model: BaseChatModel | None = None,
    store: VectorStore | None = None,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> Prediction:
    """One T-DOCS prediction for the runner.

    ``model`` is an injection point for replay and tests; when it is omitted the model
    is built from ``model_ref`` through T02's retrying/fallback layer (live wiring).
    """
    question = _question(item)
    docs, metadata = answer_question_with_metadata(
        question,
        store=store if store is not None else default_store(),
        model=model if model is not None else get_model(model_ref),
        k=k,
        min_score=min_score,
        embedder=embedder,
    )
    digest = metadata.request_digest
    if digest is None:
        digest = request_digest({"question": question})
    raw_key = recording_key(model_ref.model_id, DOCS_PROMPT_VERSION, digest)
    return Prediction(
        item_id=_item_id(item, question),
        task=TaskName.DOCS,
        model=model_ref,
        output=docs.model_dump(mode="json"),
        raw_key=raw_key,
    )
