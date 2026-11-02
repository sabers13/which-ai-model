"""Tests for the docs corpus: heading split, stable ids, char chunking."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from which_ai_model.rag.corpus import Chunk, chunk_documents, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_ROOT = REPO_ROOT / "docs" / "guide"

ID_RE = re.compile(r"^guide/[a-z0-9-]+#\d+$")


def test_load_corpus_splits_guide_by_heading() -> None:
    chunks = load_corpus(GUIDE_ROOT)
    assert len(chunks) >= 8, "the guide needs enough passages to retrieve from"
    ids = [chunk.id for chunk in chunks]
    assert len(set(ids)) == len(ids), "passage ids must be unique"
    for chunk in chunks:
        assert ID_RE.match(chunk.id), f"unstable-looking id: {chunk.id}"
        assert chunk.text.strip(), f"empty passage: {chunk.id}"
        assert chunk.source.endswith(".md"), chunk.source


def test_load_corpus_is_deterministic() -> None:
    first = load_corpus(GUIDE_ROOT)
    second = load_corpus(GUIDE_ROOT)
    assert first == second


def test_split_page_keeps_subheadings_inside_section(tmp_path: Path) -> None:
    page = tmp_path / "demo.md"
    page.write_text(
        "# Title\n\nIntro paragraph here.\n\n## First\n\nBody one.\n\n### Detail\n\n"
        "Still section one.\n\n## Second\n\nBody two.\n",
        encoding="utf-8",
    )
    chunks = load_corpus(tmp_path)
    assert [chunk.id for chunk in chunks] == [
        "guide/demo#0",
        "guide/demo#1",
        "guide/demo#2",
    ]
    assert "### Detail" in chunks[1].text
    assert "Still section one." in chunks[1].text
    assert chunks[0].source == "demo.md"


def test_split_page_skips_empty_sections(tmp_path: Path) -> None:
    page = tmp_path / "empty.md"
    page.write_text("# Title\n\n## Blank\n\n## Filled\n\nWords here.\n", encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert [chunk.id for chunk in chunks] == ["guide/empty#0", "guide/empty#2"]
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_documents_passes_short_chunks_through() -> None:
    chunks = [Chunk(id="guide/x#0", text="short text", source="x.md")]
    assert chunk_documents(chunks) == chunks


def test_chunk_documents_splits_long_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunk = Chunk(id="guide/x#1", text=text, source="x.md")
    parts = chunk_documents([chunk], max_chars=200, overlap=50)
    assert len(parts) > 1
    assert all(len(part.text) <= 200 for part in parts)
    assert [part.id for part in parts] == [f"guide/x#1-{n}" for n in range(len(parts))]
    assert all(part.source == "x.md" for part in parts)
    rejoined = " ".join(part.text for part in parts)
    for word in ("word0", "word100", "word199"):
        assert word in rejoined


def test_chunk_documents_rejects_bad_arguments() -> None:
    chunk = Chunk(id="guide/x#0", text="text", source="x.md")
    with pytest.raises(ValueError, match="max_chars"):
        chunk_documents([chunk], max_chars=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents([chunk], max_chars=100, overlap=100)
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents([chunk], max_chars=100, overlap=-1)


def test_chunk_documents_empty_input() -> None:
    assert chunk_documents([]) == []
