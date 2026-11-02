"""Embeddings: ``BAAI/bge-small-en-v1.5`` on CPU, normalised.

The model name and revision are pinned below so every index build embeds
with identical weights. The model is loaded lazily (and only here), so
importing this module never touches the network; tests replace
:func:`_get_model` with a fake embedder instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "EMBED_DIMENSIONS",
    "EMBED_MODEL_NAME",
    "EMBED_MODEL_REVISION",
    "embed_texts",
]

#: Embedding model (MIT licence) used for the docs corpus.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
#: Exact commit of the model repository every build must use.
EMBED_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
#: Output dimensions of bge-small-en-v1.5; the pgvector column width.
EMBED_DIMENSIONS = 384

_model: Any = None


def _get_model() -> Any:
    """Load (once) the pinned sentence-transformers model on CPU.

    Tests monkeypatch this function with a fake; production code must not
    call it anywhere else.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            EMBED_MODEL_NAME,
            revision=EMBED_MODEL_REVISION,
            device="cpu",
            trust_remote_code=False,
        )
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed ``texts`` with the pinned model; vectors are L2-normalised."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    items = list(texts)
    if not items:
        return []
    model = _get_model()
    vectors = model.encode(
        items,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(value) for value in row] for row in vectors]
