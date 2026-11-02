"""Fixtures for the core tests: two gateway-shaped connections, nothing live."""

from __future__ import annotations

import pytest

from which_ai_model.contracts import Connection, Protocol


@pytest.fixture
def connection() -> Connection:
    return Connection(
        id="opencode-go-openai",
        provider="opencode-go",
        protocol=Protocol.OPENAI_COMPATIBLE,
        base_url="https://opencode.ai/inference/go/openai/v1",
        credential_ref="env:OPENCODE_GO_API_KEY",
        preset="opencode-go",
    )


@pytest.fixture
def fallback_connection() -> Connection:
    return Connection(
        id="opencode-go-anthropic",
        provider="opencode-go",
        protocol=Protocol.ANTHROPIC,
        base_url="https://opencode.ai/inference/go/anthropic",
        credential_ref="env:OPENCODE_GO_API_KEY",
        preset="opencode-go",
    )
