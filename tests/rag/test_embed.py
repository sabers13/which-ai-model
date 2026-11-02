"""Tests for the embedding wrapper. The real model is never loaded here."""

from __future__ import annotations

import re

import numpy as np
import pytest

from which_ai_model.rag import embed


class FakeModel:
    """Deterministic fake: hashed bag-of-words vectors, L2-normalised."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        **kwargs: object,
    ) -> np.ndarray:
        assert batch_size > 0
        assert kwargs.get("normalize_embeddings") is True
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimensions))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, hash(word) % self.dimensions] += 1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    fake = FakeModel()
    monkeypatch.setattr(embed, "_get_model", lambda: fake)
    monkeypatch.setattr(embed, "_model", None)
    return fake


def test_embed_texts_returns_normalised_vectors(fake_model: FakeModel) -> None:
    vectors = embed.embed_texts(["hello world", "another sentence here"])
    assert fake_model.calls == [["hello world", "another sentence here"]]
    assert len(vectors) == 2
    assert all(len(vector) == embed.EMBED_DIMENSIONS for vector in vectors)
    for vector in vectors:
        assert isinstance(vector, list)
        assert all(isinstance(value, float) for value in vector)
        assert abs(float(np.linalg.norm(np.asarray(vector))) - 1.0) < 1e-6


def test_embed_texts_empty_input_returns_empty_list(fake_model: FakeModel) -> None:
    assert embed.embed_texts([]) == []
    assert fake_model.calls == []


def test_embed_texts_rejects_non_positive_batch_size(fake_model: FakeModel) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed.embed_texts(["text"], batch_size=0)


def test_embed_texts_accepts_any_sequence(fake_model: FakeModel) -> None:
    vectors = embed.embed_texts(text for text in ["one two", "three four"])
    assert len(vectors) == 2


def test_embedding_coordinates_are_pinned() -> None:
    assert embed.EMBED_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert re.fullmatch(r"[0-9a-f]{40}", embed.EMBED_MODEL_REVISION), (
        "revision must be a pinned 40-char commit hash"
    )
    assert embed.EMBED_DIMENSIONS == 384


def test_embed_texts_forwards_batch_size(fake_model: FakeModel) -> None:
    embed.embed_texts(["a b c", "d e f"], batch_size=1)
    assert fake_model.calls == [["a b c", "d e f"]]
