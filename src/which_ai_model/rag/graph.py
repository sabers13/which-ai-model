"""The L4 RAG graph: retrieve (k=4) → answer → validate citations.

The graph never asks the model for free text that it then parses for citations. The
answer schema is :class:`~which_ai_model.contracts.DocsOut`, delivered through the
model's structured-output channel; the citation step then keeps only ids that were in
the retrieved passage set, drops the rest and records them. A question with no passage
above ``min_score`` abstains before any model call, and a model that sets
``abstained=True`` (or breaks the schema) also abstains explicitly.

``min_score`` is a cosine-similarity floor on the best hit. ``0.0`` keeps retrieval
permissive: every passage the store returns is eligible, and the model decides whether
the passage answers the question. A positive floor is opt-in for a stricter run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from which_ai_model.contracts import DocsOut
from which_ai_model.core.record import request_digest
from which_ai_model.core.structured import structured
from which_ai_model.rag.corpus import SearchHit
from which_ai_model.rag.embed import embed_texts
from which_ai_model.rag.store import VectorStore

__all__ = [
    "ABSTAIN_ANSWER",
    "DEFAULT_K",
    "DEFAULT_MIN_SCORE",
    "DOCS_PROMPT_VERSION",
    "RagMetadata",
    "RagState",
    "answer_question",
    "answer_question_with_metadata",
    "build_rag_graph",
]

#: Retrieval depth fixed by ``PLAN.md`` §7 (hit@4).
DEFAULT_K = 4
#: Cosine floor on the best hit; ``0.0`` disables the retrieval-only abstention.
DEFAULT_MIN_SCORE = 0.0
#: Prompt version for recordings and the scorer's comparability check.
DOCS_PROMPT_VERSION = "docs-v1"
#: Answer used when retrieval itself finds nothing relevant.
ABSTAIN_ANSWER = "No passage in the guide is relevant enough to answer this question."

Embedder = Callable[[Sequence[str]], list[list[float]]]

_SYSTEM_PROMPT = (
    "You are the documentation helper for this repository. Answer the question using "
    "only the numbered passages given to you. Cite the passage ids exactly as written "
    "in square brackets, e.g. guide/call-model-schema#1. Never invent an id: if the "
    "passages do not answer the question, set abstained=true, leave cited_ids empty "
    "and say briefly that the guide does not cover it."
)


class RagState(TypedDict, total=False):
    """The graph state: question in, retrieved passages and a validated answer out."""

    question: str
    hits: list[SearchHit]
    answer: str
    cited_ids: list[str]
    dropped_ids: list[str]
    abstained: bool
    schema_valid: bool
    abstain_reason: str | None
    request_digest: str | None


@dataclass(frozen=True)
class RagMetadata:
    """What the graph saw and what it had to drop, for eval and parity reporting."""

    retrieved_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    top_score: float | None
    schema_valid: bool
    abstained: bool
    abstain_reason: str | None
    request_digest: str | None


def _passage_block(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(f"[{hit.chunk.id}] {hit.chunk.text}" for hit in hits)


def _answer_messages(question: str, hits: Sequence[SearchHit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nPassages (cite only these ids):\n\n{_passage_block(hits)}"
            ),
        },
    ]


def _retrieve(question: str, store: VectorStore, embedder: Embedder, k: int) -> list[SearchHit]:
    vectors = embedder([question])
    if not vectors:
        return []
    return list(store.search(vectors[0], k=k))


def _retrieved_relevant(hits: Sequence[SearchHit], min_score: float) -> bool:
    return bool(hits) and hits[0].score > min_score


def _retrieve_node(
    state: RagState, store: VectorStore, embedder: Embedder, k: int
) -> dict[str, Any]:
    hits = _retrieve(state.get("question", ""), store, embedder, k)
    return {"hits": hits}


def _answer_node(state: RagState, model: BaseChatModel) -> dict[str, Any]:
    hits = state.get("hits", [])
    messages = _answer_messages(state.get("question", ""), hits)
    digest = request_digest(messages)
    parsed, valid, _raw = structured(model, DocsOut, messages)
    if not valid or parsed is None:
        return {
            "answer": "",
            "cited_ids": [],
            "abstained": True,
            "schema_valid": False,
            "abstain_reason": "schema-invalid",
            "request_digest": digest,
        }
    return {
        "answer": parsed.answer,
        "cited_ids": list(parsed.cited_ids),
        "abstained": parsed.abstained,
        "schema_valid": True,
        "abstain_reason": "model-abstained" if parsed.abstained else None,
        "request_digest": digest,
    }


def _cite_node(state: RagState) -> dict[str, Any]:
    allowed = {hit.chunk.id for hit in state.get("hits", [])}
    kept: list[str] = []
    dropped: list[str] = []
    for chunk_id in state.get("cited_ids", []):
        if chunk_id in allowed:
            if chunk_id not in kept:
                kept.append(chunk_id)
        elif chunk_id not in dropped:
            dropped.append(chunk_id)
    return {"cited_ids": kept, "dropped_ids": dropped}


def _abstain_node(state: RagState) -> dict[str, Any]:
    reason = state.get("abstain_reason") or "retrieval-threshold"
    return {
        "answer": ABSTAIN_ANSWER,
        "cited_ids": [],
        "abstained": True,
        "schema_valid": False,
        "abstain_reason": reason,
    }


def build_rag_graph(
    store: VectorStore,
    model: BaseChatModel,
    *,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> CompiledStateGraph[RagState, Any, RagState, RagState]:
    """Compile retrieve → (answer → cite | abstain) for one store and model."""
    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)
    if not 0.0 <= min_score <= 1.0:
        msg = f"min_score must be a cosine similarity in [0, 1], got {min_score}"
        raise ValueError(msg)

    builder: StateGraph[RagState, Any, RagState, RagState] = StateGraph(RagState)

    def retrieve_node(state: RagState) -> dict[str, Any]:
        return _retrieve_node(state, store, embedder, k)

    def answer_node(state: RagState) -> dict[str, Any]:
        return _answer_node(state, model)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("cite", _cite_node)
    builder.add_node("abstain", _abstain_node)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        lambda state: (
            "answer" if _retrieved_relevant(state.get("hits", []), min_score) else "abstain"
        ),
    )
    builder.add_edge("answer", "cite")
    builder.add_edge("cite", END)
    builder.add_edge("abstain", END)
    return builder.compile()


def answer_question_with_metadata(
    question: str,
    *,
    store: VectorStore,
    model: BaseChatModel,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> tuple[DocsOut, RagMetadata]:
    """Run the graph once and return the answer plus what retrieval/citation did."""
    graph = build_rag_graph(store, model, k=k, min_score=min_score, embedder=embedder)
    state: RagState = cast(RagState, graph.invoke({"question": question}))
    hits = state.get("hits", [])
    docs = DocsOut(
        answer=state.get("answer", ""),
        cited_ids=list(state.get("cited_ids", [])),
        abstained=bool(state.get("abstained", False)),
    )
    metadata = RagMetadata(
        retrieved_ids=tuple(hit.chunk.id for hit in hits),
        dropped_ids=tuple(state.get("dropped_ids", [])),
        top_score=hits[0].score if hits else None,
        schema_valid=bool(state.get("schema_valid", False)),
        abstained=docs.abstained,
        abstain_reason=state.get("abstain_reason"),
        request_digest=state.get("request_digest"),
    )
    return docs, metadata


def answer_question(
    question: str,
    *,
    store: VectorStore,
    model: BaseChatModel,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> DocsOut:
    """The public T-DOCS answer: citations are validated, abstention is explicit."""
    docs, _metadata = answer_question_with_metadata(
        question,
        store=store,
        model=model,
        k=k,
        min_score=min_score,
        embedder=embedder,
    )
    return docs
