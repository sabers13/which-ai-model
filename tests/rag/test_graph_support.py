"""Shared test doubles for the T09 graph, eval and parity tests.

No provider SDK, no network, no real embedder. Vectors are deterministic token counts
so retrieval order is stable across runs and machines.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from which_ai_model.rag.corpus import Chunk, SearchHit
from which_ai_model.rag.store import InMemoryStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DIMENSIONS = 64


def token_embed(texts: Sequence[str]) -> list[list[float]]:
    """A deterministic bag-of-tokens embedder: shared words score higher."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for token in _TOKEN_RE.findall(text.lower()):
            vector[zlib.crc32(token.encode("utf-8")) % DIMENSIONS] += 1.0
        vectors.append(vector)
    return vectors


def make_chunk(chunk_id: str, text: str, source: str = "page.md") -> Chunk:
    return Chunk(id=chunk_id, text=text, source=source)


def make_store(chunks: Sequence[Chunk]) -> InMemoryStore:
    store = InMemoryStore()
    vectors = token_embed([chunk.text for chunk in chunks])
    store.ensure_schema(dimensions=DIMENSIONS)
    store.upsert(chunks, vectors)
    return store


def make_hit(chunk_id: str, score: float = 0.9, text: str = "passage text") -> SearchHit:
    return SearchHit(chunk=make_chunk(chunk_id, text), score=score)


class ScriptedStore(InMemoryStore):
    """A store whose search result is fixed per query vector (no ranking surprises)."""

    def __init__(self, rankings: dict[tuple[float, ...], list[SearchHit]]) -> None:
        super().__init__()
        self._rankings = rankings
        self.queries: list[tuple[float, ...]] = []

    def search(self, embedding: Sequence[float], *, k: int = 4) -> list[SearchHit]:
        key = tuple(float(value) for value in embedding)
        self.queries.append(key)
        return list(self._rankings.get(key, []))[:k]


class MappingEmbedder:
    """Maps exact question text to a vector, so a ScriptedStore knows the question."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]


def one_hot(position: int, dimensions: int = 8) -> list[float]:
    vector = [0.0] * dimensions
    vector[position] = 1.0
    return vector


def docs_tool_call(
    answer: str = "Answer from the retrieved passage.",
    cited_ids: Sequence[str] = (),
    abstained: bool = False,
) -> AIMessage:
    args = {
        "answer": answer,
        "cited_ids": list(cited_ids),
        "abstained": abstained,
    }
    return AIMessage(
        content="",
        tool_calls=[{"name": "DocsOut", "args": args, "id": "call_1", "type": "tool_call"}],
    )


class ScriptedDocsModel(BaseChatModel):
    """Replies with queued structured answers, invalid text, or an exception."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    replies: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "test-docs-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedDocsModel:
        del kwargs
        self.bound.extend(tools if isinstance(tools, list) else [tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.calls.append(messages)
        if not self.replies:
            msg = "docs model script exhausted"
            raise AssertionError(msg)
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        message = item if isinstance(item, AIMessage) else AIMessage(content=str(item))
        return ChatResult(generations=[ChatGeneration(message=message)])


class CitingDocsModel(BaseChatModel):
    """Cites the first passage id it finds in the prompt, like a well-behaved model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str = "Answer from the retrieved passage."
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "test-docs-citing"

    def bind_tools(self, tools: Any, **kwargs: Any) -> CitingDocsModel:
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.calls.append(messages)
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        found = re.findall(r"\[(guide/[^\]\s]+)\]", prompt)
        message = docs_tool_call(
            answer=self.answer,
            cited_ids=found[:1],
            abstained=not found,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
