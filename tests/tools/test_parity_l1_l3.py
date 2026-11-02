"""L1-L3 before/after parity, tracing and the tool template — all offline.

The 50 committed fixtures are the same items ``tests/legacy/test_parity_fixtures.py``
replays. Here the fake model skips the wire and answers through the schema with the
payload that carries the same meaning as the fixture's legacy free text: the router's
first-line label, the drafter's regexed JSON object, the scope checker's yes/no verdict.
Parity means the migrated tool returns the legacy label and the legacy parsed fields for
every item, with no network and no provider code.

Every model call also writes exactly one ``CallRecord``; the trace tests read the row
back and check the run, task, tool, prompt version and schema verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from which_ai_model.contracts import ModelRef, Prediction, TaskName
from which_ai_model.legacy.l1_router import parse_label
from which_ai_model.legacy.l2_drafter import extract_json
from which_ai_model.legacy.l3_scope import OUT_OF_SCOPE, parse_scope
from which_ai_model.tools import _template, drafter, router, scope

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "legacy" / "parity_items.jsonl"
REF = ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash")


def _tool_name(tool: Any) -> str:
    """The name LangChain put on the schema's function-calling tool."""
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name"))
        return str(tool.get("name"))
    return str(getattr(tool, "__name__", tool))


class PayloadModel(BaseChatModel):
    """A chat model that answers the bound schema with ``payload`` — nothing else.

    ``with_structured_output`` binds the schema as a tool, so ``bind_tools`` sees its
    name; the reply is one tool call for that name carrying the payload. No socket, no
    provider class, no key.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    tool_name: str = "unknown"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "test-payload"

    def bind_tools(self, tools: Any, **kwargs: Any) -> PayloadModel:
        del kwargs
        items = tools if isinstance(tools, list) else [tools]
        self.tool_name = _tool_name(items[0])
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        self.calls += 1
        call = {"name": self.tool_name, "args": self.payload, "id": "call_1", "type": "tool_call"}
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
        )


class BrokenModel(BaseChatModel):
    """A chat model that raises, to prove a failed call still leaves a trace row."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "test-broken"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BrokenModel:
        del tools, kwargs
        return self

    def _generate(self, messages: list[Any], **kwargs: Any) -> ChatResult:
        del messages, kwargs
        raise RuntimeError("transport is down")


@pytest.fixture(scope="module")
def parity_items() -> list[dict[str, Any]]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(autouse=True)
def offline_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep trace rows out of the repository, and name the run like the Makefile does."""
    monkeypatch.setattr(_template, "TRACE_ROOT", tmp_path)
    monkeypatch.setenv("WAM_RUN_ID", "parity")
    monkeypatch.setenv("WAM_MODE", "replay")
    return tmp_path


def _legacy_route(item: dict[str, Any]) -> str | None:
    return parse_label(item["responses"]["l1"]["content"])


def _legacy_reply(item: dict[str, Any]) -> dict[str, Any] | None:
    return extract_json(item["responses"]["l2"]["content"])


def _legacy_scope(item: dict[str, Any]) -> bool:
    return parse_scope(item["responses"]["l3"]["content"]) == OUT_OF_SCOPE


def test_fixture_set_is_the_committed_50(parity_items: list[dict[str, Any]]) -> None:
    assert len(parity_items) == 50
    assert len({item["item_id"] for item in parity_items}) == 50
    assert all(_legacy_route(item) for item in parity_items)
    assert all(_legacy_reply(item) for item in parity_items)


def test_l1_route_parity(parity_items: list[dict[str, Any]]) -> None:
    model = PayloadModel()
    differences: list[str] = []
    for item in parity_items:
        before = _legacy_route(item)
        model.payload = {"intent": before}
        after = router.route_message(item["message"], model)
        if after is None:
            differences.append(f"{item['item_id']}: legacy={before!r} after=schema-invalid")
        elif after.intent != before:
            differences.append(f"{item['item_id']}: legacy={before!r} after={after.intent!r}")
    assert differences == [], f"{len(differences)} L1 differences: {differences}"
    assert model.calls == 50


def test_l2_reply_parity(parity_items: list[dict[str, Any]]) -> None:
    model = PayloadModel()
    differences: list[str] = []
    for item in parity_items:
        before = _legacy_reply(item)
        model.payload = dict(before or {})
        after = drafter.draft_reply(item["message"], model)
        if after is None:
            differences.append(f"{item['item_id']}: legacy={before!r} after=schema-invalid")
        elif after.model_dump() != before:
            differences.append(f"{item['item_id']}: legacy={before!r} after={after.model_dump()!r}")
    assert differences == [], f"{len(differences)} L2 differences: {differences}"
    assert model.calls == 50


def test_l3_scope_parity(parity_items: list[dict[str, Any]]) -> None:
    model = PayloadModel()
    differences: list[str] = []
    for item in parity_items:
        before = _legacy_scope(item)
        model.payload = {"out_of_scope": before}
        after = scope.check_scope(item["message"], model)
        if after is None:
            differences.append(f"{item['item_id']}: legacy={before!r} after=schema-invalid")
        elif after.out_of_scope != before:
            differences.append(f"{item['item_id']}: legacy={before!r} after={after.out_of_scope!r}")
    assert differences == [], f"{len(differences)} L3 differences: {differences}"


def test_scope_graph_escalates_with_the_fixed_handoff() -> None:
    out_of_scope = PayloadModel(payload={"out_of_scope": True})
    graph = scope.build_scope_graph(out_of_scope)
    state = graph.invoke({"message": "Can you recommend a holiday destination?"})
    assert state["out_of_scope"] is True
    assert state["reply"] == scope.HANDOFF_MESSAGE

    in_scope = PayloadModel(payload={"out_of_scope": False})
    state = scope.build_scope_graph(in_scope).invoke({"message": "Where is my card?"})
    assert state["out_of_scope"] is False
    assert state["reply"] is None


def test_schema_invalid_answer_is_wrong_but_not_raised() -> None:
    model = PayloadModel(payload={"unexpected": "field"})
    assert router.route_message("Where is my card?", model) is None
    assert drafter.draft_reply("Where is my card?", model) is None
    assert scope.check_scope("Where is my card?", model) is None


def test_predict_returns_a_scored_prediction_and_one_trace_row(
    offline_trace_dir: Path, parity_items: list[dict[str, Any]]
) -> None:
    item = {**parity_items[0], "item_id": "fixture-1"}
    model = PayloadModel(payload={"intent": _legacy_route(parity_items[0])})

    prediction = router.predict(item, REF, model=model)

    assert isinstance(prediction, Prediction)
    assert prediction.item_id == "fixture-1"
    assert prediction.task is TaskName.ROUTE
    assert prediction.model == REF
    assert prediction.output == {"intent": _legacy_route(parity_items[0])}
    assert len(prediction.raw_key) == 64

    rows = _trace_rows(offline_trace_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "parity"
    assert row["task"] == "T-ROUTE"
    assert row["tool"] == "router"
    assert row["item_id"] == "fixture-1"
    assert row["prompt_version"] == router.PROMPT_VERSION
    assert row["schema_valid"] is True
    assert row["response_key"] == prediction.raw_key


def test_every_tool_predicts_and_traces(offline_trace_dir: Path) -> None:
    item = {"item_id": "sample-7", "text": "I was charged twice for a top-up."}
    cases: list[tuple[Any, dict[str, Any], TaskName]] = [
        (router, {"intent": "top_up_failed"}, TaskName.ROUTE),
        (
            drafter,
            {"intent": "top_up_failed", "urgency": "high", "reply": "We are on it."},
            TaskName.REPLY,
        ),
        (scope, {"out_of_scope": False}, TaskName.SCOPE),
    ]
    for module, payload, task in cases:
        prediction = module.predict(item, REF, model=PayloadModel(payload=payload))
        assert prediction.task is task
        assert prediction.output == payload
    rows = _trace_rows(offline_trace_dir)
    assert [row["tool"] for row in rows] == ["router", "drafter", "scope"]
    assert [row["task"] for row in rows] == ["T-ROUTE", "T-REPLY", "T-SCOPE"]
    assert all(row["schema_valid"] is True for row in rows)
    assert len({row["response_key"] for row in rows}) == 3


def test_invalid_prediction_carries_no_fields(offline_trace_dir: Path) -> None:
    item = {"item_id": "sample-8", "text": "hello"}
    prediction = drafter.predict(item, REF, model=PayloadModel(payload={"intent": "hello"}))
    assert prediction.output == {}
    row = _trace_rows(offline_trace_dir)[0]
    assert row["schema_valid"] is False


def test_failed_call_writes_an_error_row(offline_trace_dir: Path) -> None:
    item = {"item_id": "sample-9", "text": "hello"}
    with pytest.raises(RuntimeError, match="transport is down"):
        router.predict(item, REF, model=BrokenModel())
    row = _trace_rows(offline_trace_dir)[0]
    assert row["error"] == "RuntimeError"
    assert row["schema_valid"] is None


def test_template_example_is_a_working_tool(offline_trace_dir: Path) -> None:
    item = {"item_id": "template-1", "message": "Where is my card?"}
    prediction = _template.example_predict(
        item, REF, model=PayloadModel(payload={"intent": "card_arrival"})
    )
    assert prediction.output == {"intent": "card_arrival"}
    assert _trace_rows(offline_trace_dir)[0]["tool"] == _template.EXAMPLE_TOOL


def _trace_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "replay" / "parity.jsonl"
    assert path.exists(), f"no trace file at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
