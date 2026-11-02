"""Prompts are one per task, versioned and identical for every model."""

from __future__ import annotations

from which_ai_model.bench.prompts import (
    PROMPT_VERSION,
    PROMPTS,
    intent_names,
    prompt_for,
)
from which_ai_model.contracts import TaskName


def test_one_versioned_prompt_per_task() -> None:
    assert PROMPT_VERSION == "v1"
    assert set(PROMPTS) == set(TaskName)
    for task in TaskName:
        prompt = prompt_for(task)
        assert prompt.task is task
        assert prompt.system.strip()


def test_routing_prompt_lists_all_77_intents() -> None:
    names = intent_names()
    assert len(names) == 77
    assert len(set(names)) == 77
    system = prompt_for(TaskName.ROUTE).system
    for name in names:
        assert f"- {name}" in system


def test_messages_are_identical_for_every_model() -> None:
    item = {"item_id": "x", "text": "My card was swallowed by the ATM."}
    first = prompt_for(TaskName.ROUTE).messages(item)
    second = prompt_for(TaskName.ROUTE).messages(item)
    assert first == second
    assert [message["role"] for message in first] == ["system", "user"]
    assert item["text"] in first[1]["content"]


def test_docs_prompt_uses_the_question_field() -> None:
    messages = prompt_for(TaskName.DOCS).messages(
        {"item_id": "t-docs-001", "question": "Where is the key kept?"}
    )
    assert "Where is the key kept?" in messages[1]["content"]
