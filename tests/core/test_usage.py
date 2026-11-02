"""Usage reading: reasoning tokens, the gateway block and measured cost, finish reasons."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from which_ai_model.core.usage import (
    EMPTY,
    TRUNCATED,
    empty_answer_reason,
    extract_usage,
    measured_cost_usd,
)


def test_reads_langchain_usage_metadata_and_reasoning() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 812,
            "output_tokens": 44,
            "total_tokens": 856,
            "output_token_details": {"reasoning": 12},
        },
    )
    usage = extract_usage(message)
    assert usage.input_tokens == 812
    assert usage.output_tokens == 44
    assert usage.total_tokens == 856
    assert usage.reasoning_tokens == 12
    assert usage.measured_cost_usd is None


def test_reads_gateway_opencode_block_and_measured_cost() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "opencode": {
                    "list_cost": 0.000123,
                    "allowance": "go",
                    "billing_source": "subscription",
                    "tokens": {"input_tokens": 10, "output_tokens": 5},
                },
            }
        },
    )
    usage = extract_usage(message)
    assert usage.measured_cost_usd == 0.000123
    assert usage.gateway["billing_source"] == "subscription"
    assert measured_cost_usd(message) == 0.000123


def test_reads_recording_response_and_raw_provider_usage() -> None:
    response = {
        "content": "ok",
        "tool_calls": [],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 3},
            "opencode": {"list_cost": 0.5},
        },
    }
    usage = extract_usage(response)
    assert (usage.input_tokens, usage.output_tokens) == (20, 8)
    assert usage.reasoning_tokens == 3
    assert usage.measured_cost_usd == 0.5


def test_unknown_usage_is_none_not_zero() -> None:
    usage = extract_usage({"content": "ok", "tool_calls": [], "usage": {}})
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.measured_cost_usd is None


def test_empty_and_truncated_answers_are_reasons_not_exceptions() -> None:
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    empty = AIMessage(content="")
    assert empty_answer_reason(truncated) == TRUNCATED
    assert empty_answer_reason(empty) == EMPTY
    assert empty_answer_reason(AIMessage(content="a real answer")) is None
    tool_only = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    assert empty_answer_reason(tool_only) is None
