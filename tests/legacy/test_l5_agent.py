"""L5 agent: the hand-written tool loop over two fictional-bank tools."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from _helpers import chat_completion, scripted, tool_call

from which_ai_model.legacy._agent_tools import TOOL_DEFINITIONS, execute_tool
from which_ai_model.legacy.l5_agent import run, run_agent


def configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_KEY", "test-agent-key")
    monkeypatch.setenv("AGENT_MODEL", "agent-test")
    monkeypatch.setenv("AGENT_BASE_URL", "https://agent.test/v1")


def test_two_step_fee_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    responses = [
        chat_completion(
            calls=[
                tool_call("call_1", "get_fee", {"card_type": "debit", "action": "cash_withdrawal"})
            ]
        ),
        chat_completion("A debit card cash withdrawal costs USD 2.50."),
    ]
    transport, requests = scripted(responses)

    result = run_agent("What does a debit cash withdrawal cost?", transport=transport)

    assert result.error is None
    assert result.content == "A debit card cash withdrawal costs USD 2.50."
    assert [call["name"] for call in result.tool_calls] == ["get_fee"]
    assert json.loads(result.tool_calls[0]["result"])["fee_usd"] == 2.5
    assert str(requests[0].url) == "https://agent.test/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "agent-test"
    assert payload["tools"] == TOOL_DEFINITIONS
    assert payload["messages"][0]["role"] == "system"
    second = json.loads(requests[1].content)["messages"]
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "call_1"
    assert "2.5" in second[-1]["content"]


def test_run_returns_the_legacy_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    transport, _ = scripted([chat_completion("No fee applies.")])

    result = run("Is there a fee?", transport=transport)

    assert set(result) == {"content", "tool_calls", "latency_ms", "error"}
    assert result["content"] == "No fee applies."
    assert result["error"] is None


def test_multi_step_policy_then_fee(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    responses = [
        chat_completion(calls=[tool_call("call_a", "get_policy", {"topic": "lost_card"})]),
        chat_completion(
            calls=[
                tool_call(
                    "call_b",
                    "get_fee",
                    {"card_type": "prepaid", "action": "replacement_card"},
                )
            ]
        ),
        chat_completion("Report it; a prepaid replacement costs USD 5.00."),
    ]
    transport, requests = scripted(responses)

    result = run_agent("I lost my prepaid card.", transport=transport)

    assert result.error is None
    assert [call["name"] for call in result.tool_calls] == ["get_policy", "get_fee"]
    roles = [message["role"] for message in json.loads(requests[-1].content)["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]


def test_fixture_tools_are_clearly_fictional(fixture_dir: Path) -> None:
    fees = fixture_dir / "fees.json"
    policies = fixture_dir / "policies.json"
    fee = json.loads(
        execute_tool("get_fee", {"card_type": "credit", "action": "late_payment"}, fees, policies)
    )
    policy = json.loads(execute_tool("get_policy", {"topic": "refunds"}, fees, policies))
    assert fee["fee_usd"] == 25.0
    assert fee["currency"] == "USD"
    assert "five business days" in policy["policy"]
    with pytest.raises(ValueError, match="not permitted"):
        execute_tool("run_command", {"argv": ["ls"]}, fees, policies)


def test_turn_limit_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    call = tool_call("call_x", "get_policy", {"topic": "overdraft"})
    transport, requests = scripted([chat_completion(calls=[call]), chat_completion(calls=[call])])

    result = run_agent("loop", transport=transport, max_turns=2)

    assert result.error == "API tool loop reached its 2-turn limit"
    assert len(requests) == 2
    assert len(result.tool_calls) == 2


def test_redirect_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "https://elsewhere.test"})
    )

    result = run_agent("hello", transport=transport)

    assert result.error == "API redirect refused"


def test_unpermitted_tool_is_denied_and_the_loop_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    responses = [
        chat_completion(calls=[tool_call("call_1", "run_command", {"argv": ["ls"]})]),
        chat_completion("I cannot run commands."),
    ]
    transport, _ = scripted(responses)

    result = run_agent("list files", transport=transport)

    assert result.error is None
    assert "denied" in result.tool_calls[0]["result"]


def test_non_stop_finish_reason_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    transport, _ = scripted([chat_completion("cut off", finish="length")])

    result = run_agent("hello", transport=transport)

    assert result.error == "API completion ended without a successful stop"


def test_missing_key_and_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_agent("hello").error == "AGENT_KEY is not set"
    configure(monkeypatch)
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    result = run_agent("hello", transport=transport)
    assert result.error == "API request failed (HTTP 500)"
