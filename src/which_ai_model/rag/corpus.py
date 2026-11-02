"""Docs corpus: guide pages split into passages by heading.

Each ``docs/guide/*.md`` page becomes one passage per ``##`` section (the
title plus intro before the first section is passage ``#0``); deeper
``###`` headings stay inside their section. Passage ids are stable —
``guide/<file-stem>#<n>`` with ``n`` counting sections in file order — so
gold answers and citations survive a rebuild. Longer passages are split on
character count by :func:`chunk_documents`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Chunk", "SearchHit", "chunk_documents", "load_corpus"]

#: A Markdown ATX heading, e.g. ``## Tracing``.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage: stable id, text, and source file."""

    id: str
    text: str
    source: str


@dataclass(frozen=True)
class SearchHit:
    """A chunk returned by search with its similarity score (higher is better)."""

    chunk: Chunk
    score: float


def load_corpus(root: Path = Path("docs/guide")) -> list[Chunk]:
    """Split every ``*.md`` page under ``root`` into passages by heading."""
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        chunks.extend(_split_page(path, root))
    return chunks


def _is_section_heading(line: str) -> bool:
    match = _HEADING_RE.match(line)
    return match is not None and len(match.group(1)) == 2


def _split_page(path: Path, root: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[list[str]] = [[]]
    for line in lines:
        if _is_section_heading(line):
            sections.append([line])
        else:
            sections[-1].append(line)
    chunks: list[Chunk] = []
    for number, section in enumerate(sections):
        body = section[1:] if section and _is_section_heading(section[0]) else section
        if not any(line.strip() for line in body):
            continue
        text = "\n".join(section).strip()
        chunks.append(
            Chunk(
                id=f"guide/{path.stem}#{number}",
                text=text,
                source=str(path.relative_to(root)),
            )
        )
    return chunks


def chunk_documents(
    chunks: Sequence[Chunk],
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Split chunks longer than ``max_chars`` into overlapping windows.

    Short chunks pass through unchanged with their ids intact. A long chunk
    becomes windows of at most ``max_chars`` characters cut at word
    boundaries, each sharing ``overlap`` characters with the next, with
    stable child ids ``<parent-id>-<n>``.
    """
    if max_chars <= 0:
        msg = f"max_chars must be positive, got {max_chars}"
        raise ValueError(msg)
    if overlap < 0 or overlap >= max_chars:
        msg = f"overlap must satisfy 0 <= overlap < max_chars, got {overlap}"
        raise ValueError(msg)
    out: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= max_chars:
            out.append(chunk)
            continue
        out.extend(_split_long_chunk(chunk, max_chars=max_chars, overlap=overlap))
    return out


def _split_long_chunk(chunk: Chunk, *, max_chars: int, overlap: int) -> list[Chunk]:
    step = max_chars - overlap
    parts: list[Chunk] = []
    start = 0
    number = 0
    text = chunk.text
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(Chunk(id=f"{chunk.id}-{number}", text=part, source=chunk.source))
            number += 1
        if end >= len(text):
            break
        start += step
    return parts
