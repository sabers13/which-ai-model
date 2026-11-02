"""L4 docs helper: one OpenAI-compatible call with the whole guide pasted in."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from which_ai_model.legacy.l4_docs import docs_answer, guide_text, run


def openai_client(handler: object) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))  # type: ignore[arg-type]


def completion_payload(text: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "docs-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_pastes_the_fallback_guide_when_docs_guide_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCS_KEY", "test-docs-key")
    monkeypatch.setenv("DOCS_MODEL", "docs-test")
    monkeypatch.setenv("DOCS_BASE_URL", "https://docs.test/v1")
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_payload("Add it to config/providers.toml."))

    result = docs_answer(
        "How do I add a connection?",
        httpx_client=openai_client(handler),
        guide_path=tmp_path / "absent",
    )

    assert result.error is None
    assert result.content == "Add it to config/providers.toml."
    assert str(seen[0].url) == "https://docs.test/v1/chat/completions"
    prompt = json.loads(seen[0].content)["messages"][0]["content"]
    assert "Connecting a provider" in prompt
    assert "How do I add a connection?" in prompt


def test_runtime_guide_reads_every_markdown_file_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "b.md").write_text("BETA SECTION", encoding="utf-8")
    (tmp_path / "a.md").write_text("ALPHA SECTION", encoding="utf-8")
    monkeypatch.setenv("DOCS_GUIDE_PATH", str(tmp_path))

    guide = guide_text()

    assert guide.index("ALPHA SECTION") < guide.index("BETA SECTION")
    assert "Connecting a provider" not in guide


def test_runtime_guide_path_is_used_by_the_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "guide.md").write_text("RUNTIME GUIDE MARKER", encoding="utf-8")
    monkeypatch.setenv("DOCS_KEY", "test-docs-key")
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_payload("ok"))

    result = run("question", httpx_client=openai_client(handler), guide_path=tmp_path)

    assert result["content"] == "ok"
    prompt = json.loads(seen[0].content)["messages"][0]["content"]
    assert "RUNTIME GUIDE MARKER" in prompt


def test_api_error_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCS_KEY", "test-docs-key")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "bad key", "type": "auth"}})

    result = docs_answer("hello", httpx_client=openai_client(handler))
    assert result.error is not None
    assert "test-docs-key" not in result.error


def test_missing_key_is_an_error() -> None:
    assert docs_answer("hello").error == "DOCS_KEY is not set"
