"""L4 parity: the migrated tool answers all 10 fixture questions and cites a passage.

L4 is the documented exception to byte-for-byte parity (``PLAN.md`` §9): retrieval
changes the answer, so the check is "answers and cites at least one passage". The
legacy tool still answers, with no citation channel at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from test_graph_support import CitingDocsModel, make_store, token_embed

from which_ai_model.legacy.l4_docs import docs_answer
from which_ai_model.rag.corpus import chunk_documents, load_corpus
from which_ai_model.rag.graph import answer_question_with_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_PATH = REPO_ROOT / "tests" / "fixtures" / "legacy" / "parity_items.jsonl"
GUIDE_ROOT = REPO_ROOT / "docs" / "guide"


def _docs_items() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PARITY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("docs_question")]


def _legacy_client(content: str) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "docs-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_ten_fixture_questions_are_present() -> None:
    assert len(_docs_items()) == 10


def test_migrated_tool_answers_and_cites_on_all_ten_fixtures() -> None:
    store = make_store(chunk_documents(load_corpus(GUIDE_ROOT)))
    model = CitingDocsModel(answer="Answer grounded in the guide.")
    rows = _docs_items()

    for row in rows:
        question = row["docs_question"]
        docs, metadata = answer_question_with_metadata(
            question, store=store, model=model, embedder=token_embed
        )
        assert docs.answer.strip(), row["item_id"]
        assert docs.abstained is False, row["item_id"]
        assert len(docs.cited_ids) >= 1, row["item_id"]
        assert set(docs.cited_ids) <= set(metadata.retrieved_ids), row["item_id"]
        assert metadata.dropped_ids == (), row["item_id"]
        assert metadata.schema_valid is True, row["item_id"]


def test_legacy_tool_answers_without_a_citation_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCS_KEY", "test-docs-key")
    monkeypatch.delenv("DOCS_GUIDE_PATH", raising=False)

    for row in _docs_items():
        content = row["responses"]["l4"]["content"]
        result = docs_answer(
            row["docs_question"],
            httpx_client=_legacy_client(content),
            guide_path=GUIDE_ROOT,
        )

        assert result.error is None, row["item_id"]
        assert result.content == content, row["item_id"]
        assert result.tool_calls == [], row["item_id"]
        assert not hasattr(result, "cited_ids"), "legacy L4 has no citation channel"
