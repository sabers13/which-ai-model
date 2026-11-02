"""Tests for the T-DOCS gold set: shape, coverage, and no verbatim copies."""

from __future__ import annotations

import re
from pathlib import Path

from which_ai_model.rag.corpus import chunk_documents, load_corpus
from which_ai_model.rag.store import load_gold

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO_ROOT / "data" / "rag" / "gold.jsonl"

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _word_windows(words: list[str], size: int) -> set[str]:
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def test_gold_has_40_questions_with_5_abstentions() -> None:
    rows = load_gold(GOLD_PATH)
    assert len(rows) == 40
    ids = [row["id"] for row in rows]
    assert ids == [f"t-docs-{n:03d}" for n in range(1, 41)], "ids must be unique and ordered"
    answerable = [row for row in rows if row["answerable"] is True]
    abstentions = [row for row in rows if row["answerable"] is False]
    assert len(answerable) == 35
    assert len(abstentions) == 5
    for row in answerable:
        gold_ids = row["gold_ids"]
        assert isinstance(gold_ids, list), row["id"]
        assert 1 <= len(gold_ids) <= 3, row["id"]
    for row in abstentions:
        assert row["gold_ids"] == [], row["id"]


def test_gold_ids_exist_in_the_corpus() -> None:
    chunk_ids = {chunk.id for chunk in chunk_documents(load_corpus(REPO_ROOT / "docs" / "guide"))}
    rows = load_gold(GOLD_PATH)
    for row in rows:
        for gold_id in row["gold_ids"]:
            assert gold_id in chunk_ids, f"{row['id']} cites unknown {gold_id}"
    cited = {gold_id for row in rows for gold_id in row["gold_ids"]}
    assert len(cited) >= 20, "gold should spread across the guide, not one page"


def test_gold_questions_are_well_formed() -> None:
    rows = load_gold(GOLD_PATH)
    for row in rows:
        question = row["question"]
        assert isinstance(question, str), row["id"]
        assert len(question.split()) >= 8, row["id"]
        assert question.strip().endswith("?"), row["id"]


def test_gold_questions_copy_no_passage_verbatim() -> None:
    chunks = chunk_documents(load_corpus(REPO_ROOT / "docs" / "guide"))
    passage_windows: set[str] = set()
    for chunk in chunks:
        passage_windows |= _word_windows(_normalise(chunk.text).split(), 10)
    rows = load_gold(GOLD_PATH)
    for row in rows:
        words = _normalise(str(row["question"])).split()
        assert _normalise(str(row["question"])) not in {
            _normalise(chunk.text) for chunk in chunks
        }, row["id"]
        overlap = _word_windows(words, 10) & passage_windows
        assert not overlap, f"{row['id']} copies a passage verbatim: {sorted(overlap)[:1]}"
