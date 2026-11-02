"""T-DOCS evaluation: retrieval hit@4, citation precision and abstain-correct.

Two layers, because one metric is free and the others need answers:

- :func:`hit_at_4` and :func:`retrieval_rows` are **model-free**: embed the question,
  search the store, compare the top four ids with the gold ids. This is the number the
  lane report quotes for the real guide.
- :func:`evaluate_rag` runs the graph and adds citation precision and abstain-correct;
  :func:`score_docs` computes the same two metrics from committed ``Prediction`` rows,
  which is what the T08 scorer can call without a store.

Citation precision uses ``PLAN.md`` §7's rule (``cited ids ⊆ gold``, at least one
citation) and is averaged over **answerable** questions. Abstain-correct is averaged
over the **unanswerable** questions. ``hit_at_4`` is averaged over answerable questions
only: an unanswerable row has no gold passage to hit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from which_ai_model.bench.metrics import citation_precision, hit_at_k
from which_ai_model.contracts import Prediction
from which_ai_model.rag.corpus import SearchHit
from which_ai_model.rag.embed import embed_texts
from which_ai_model.rag.graph import (
    DEFAULT_K,
    DEFAULT_MIN_SCORE,
    answer_question_with_metadata,
)
from which_ai_model.rag.store import VectorStore, load_gold

__all__ = [
    "DOCS_METRIC_KEYS",
    "RetrievalRow",
    "evaluate_rag",
    "hit_at_4",
    "retrieval_rows",
    "score_docs",
]

Embedder = Callable[[Sequence[str]], list[list[float]]]

#: The metric keys a T-DOCS cell reports. ``hit_at_4`` is the primary metric.
DOCS_METRIC_KEYS = (
    "hit_at_4",
    "citation_precision",
    "abstain_correct",
    "schema_valid_rate",
)


@dataclass(frozen=True)
class RetrievalRow:
    """One question's ranked ids and whether any gold id made the top ``k``."""

    item_id: str
    question: str
    ranked_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]
    answerable: bool
    top_score: float | None
    hit: bool


def _question(row: Mapping[str, object]) -> str:
    value = row.get("question")
    if not isinstance(value, str) or not value.strip():
        msg = f"gold row {row.get('id')!r} has no question"
        raise ValueError(msg)
    return value


def _item_id(row: Mapping[str, object], position: int) -> str:
    value = row.get("id", row.get("item_id"))
    if isinstance(value, str) and value:
        return value
    return f"t-docs-{position + 1:03d}"


def _gold_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    value = row.get("gold_ids")
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        msg = f"gold row {row.get('id')!r} has non-list gold_ids"
        raise ValueError(msg)
    return tuple(str(item_id) for item_id in value)


def _answerable(row: Mapping[str, object], gold_ids: Sequence[str]) -> bool:
    value = row.get("answerable")
    if isinstance(value, bool):
        return value
    return bool(gold_ids)


def retrieval_rows(
    questions: Sequence[Mapping[str, object]],
    *,
    store: VectorStore,
    k: int = DEFAULT_K,
    embedder: Embedder = embed_texts,
) -> list[RetrievalRow]:
    """Rank passages for every question once; no model is involved."""
    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)
    if not questions:
        msg = "retrieval needs at least one question"
        raise ValueError(msg)
    texts = [_question(row) for row in questions]
    vectors = embedder(texts)
    if len(vectors) != len(texts):
        msg = f"embedder returned {len(vectors)} vectors for {len(texts)} questions"
        raise ValueError(msg)
    rows: list[RetrievalRow] = []
    for position, (row, text, vector) in enumerate(zip(questions, texts, vectors, strict=True)):
        hits: list[SearchHit] = list(store.search(vector, k=k))
        gold_ids = _gold_ids(row)
        ranked = tuple(hit.chunk.id for hit in hits)
        answerable = _answerable(row, gold_ids)
        rows.append(
            RetrievalRow(
                item_id=_item_id(row, position),
                question=text,
                ranked_ids=ranked,
                gold_ids=gold_ids,
                answerable=answerable,
                top_score=hits[0].score if hits else None,
                hit=bool(answerable and set(ranked[:k]) & set(gold_ids)),
            )
        )
    return rows


def hit_at_4(
    questions: Sequence[Mapping[str, object]],
    *,
    store: VectorStore,
    k: int = DEFAULT_K,
    embedder: Embedder = embed_texts,
) -> float:
    """Retrieval-only hit@k over answerable questions; the model is never called."""
    rows = retrieval_rows(questions, store=store, k=k, embedder=embedder)
    answerable = [row for row in rows if row.answerable]
    if not answerable:
        msg = "hit@4 needs at least one answerable question"
        raise ValueError(msg)
    return hit_at_k(
        [row.ranked_ids for row in answerable], [row.gold_ids for row in answerable], k=k
    )


@dataclass(frozen=True)
class _AnswerRow:
    item_id: str
    cited_ids: tuple[str, ...]
    abstained: bool
    schema_valid: bool


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(
    questions: Sequence[Mapping[str, object]],
    answers: Mapping[str, _AnswerRow],
    retrieved: Mapping[str, Sequence[str]] | None,
    *,
    k: int = DEFAULT_K,
) -> dict[str, float]:
    answerable_citation: list[float] = []
    answerable_abstain: list[float] = []
    answerable_hits: list[float] = []
    abstain_correct: list[float] = []
    schema_valid: list[float] = []
    n_answerable = 0
    n_unanswerable = 0

    for position, row in enumerate(questions):
        item_id = _item_id(row, position)
        gold_ids = _gold_ids(row)
        answerable = _answerable(row, gold_ids)
        answer = answers.get(item_id)
        cited = () if answer is None else answer.cited_ids
        abstained = False if answer is None else answer.abstained
        valid = False if answer is None else answer.schema_valid
        schema_valid.append(1.0 if valid else 0.0)
        if answerable:
            n_answerable += 1
            answerable_citation.append(citation_precision(cited, gold_ids))
            answerable_abstain.append(1.0 if abstained else 0.0)
            if retrieved is not None:
                ranked = retrieved.get(item_id, ())
                hit = bool(set(list(ranked)[:k]) & set(gold_ids))
                answerable_hits.append(1.0 if hit else 0.0)
        else:
            n_unanswerable += 1
            abstain_correct.append(1.0 if abstained else 0.0)

    metrics = {
        "n": float(len(questions)),
        "n_answerable": float(n_answerable),
        "n_unanswerable": float(n_unanswerable),
        "citation_precision": _mean(answerable_citation),
        "abstain_correct": _mean(abstain_correct),
        "answerable_abstain_rate": _mean(answerable_abstain),
        "schema_valid_rate": _mean(schema_valid),
    }
    if retrieved is not None:
        metrics["hit_at_4"] = _mean(answerable_hits)
    return metrics


def evaluate_rag(
    questions: Sequence[Mapping[str, object]],
    *,
    store: VectorStore,
    model: BaseChatModel,
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
    embedder: Embedder = embed_texts,
) -> dict[str, float]:
    """Run the graph on every gold question and return the T-DOCS metrics."""
    rows = retrieval_rows(questions, store=store, k=k, embedder=embedder)
    answers: dict[str, _AnswerRow] = {}
    for row in rows:
        docs, metadata = answer_question_with_metadata(
            row.question,
            store=store,
            model=model,
            k=k,
            min_score=min_score,
            embedder=embedder,
        )
        answers[row.item_id] = _AnswerRow(
            item_id=row.item_id,
            cited_ids=tuple(docs.cited_ids),
            abstained=docs.abstained,
            schema_valid=metadata.schema_valid,
        )
    retrieved = {row.item_id: row.ranked_ids for row in rows}
    return _score(questions, answers, retrieved, k=k)


def _docs_from_output(output: Mapping[str, object]) -> tuple[tuple[str, ...], bool, bool]:
    raw_cited = output.get("cited_ids", [])
    cited = tuple(str(item_id) for item_id in raw_cited) if isinstance(raw_cited, list) else ()
    abstained = output.get("abstained") is True
    schema_valid = (
        "answer" in output
        and isinstance(output.get("answer"), str)
        and isinstance(output.get("cited_ids"), list)
        and isinstance(output.get("abstained"), bool)
    )
    return cited, abstained, schema_valid


def score_docs(
    predictions: Sequence[Prediction],
    questions: Sequence[Mapping[str, object]] | None = None,
    *,
    retrieved: Mapping[str, Sequence[str]] | None = None,
    k: int = DEFAULT_K,
    gold_path: Path = Path("data/rag/gold.jsonl"),
) -> dict[str, float]:
    """Score committed T-DOCS predictions without a store or a model.

    ``questions`` defaults to the committed gold set, so a task scorer can call
    ``score_docs(predictions)`` with the same shape as the other scoring functions.
    Pass ``retrieved`` (item id → ranked passage ids) when the run also needs hit@4;
    a run that did not persist retrieval can still report citation precision and
    abstain-correct from the predictions alone.
    """
    rows = load_gold(gold_path) if questions is None else questions
    by_id: dict[str, _AnswerRow] = {}
    for prediction in predictions:
        cited, abstained, schema_valid = _docs_from_output(prediction.output)
        by_id[prediction.item_id] = _AnswerRow(
            item_id=prediction.item_id,
            cited_ids=cited,
            abstained=abstained,
            schema_valid=schema_valid,
        )
    return _score(rows, by_id, retrieved, k=k)
