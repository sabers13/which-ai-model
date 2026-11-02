"""Map a :class:`Connection` + model id to a LangChain chat model.

T02 wraps this; keeping the mapping here means the wire protocol, custom base URL,
fixed headers and the lowest reasoning setting live in one place. Construction is
offline — no request leaves this module.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from which_ai_model.connect.credentials import resolve_credential
from which_ai_model.connect.presets import connection_headers
from which_ai_model.contracts import Connection, Protocol

__all__ = ["chat_model"]


def chat_model(
    connection: Connection,
    model_id: str,
    *,
    session_id: str | None = None,
    reasoning: str | None = "minimal",
    max_tokens: int = 1024,
    temperature: float | None = 0.0,
    timeout_s: float = 120.0,
    **opts: Any,
) -> BaseChatModel:
    """Build the provider class for ``connection.protocol``.

    - ``OPENAI_COMPATIBLE`` -> ``ChatOpenAI`` (Chat Completions)
    - ``OPENAI_RESPONSES`` -> ``ChatOpenAI(use_responses_api=True)``
    - ``ANTHROPIC`` -> ``ChatAnthropic`` (no extended thinking by default)
    - ``GEMINI`` -> ``ChatGoogleGenerativeAI`` (lowest thinking level)

    ``temperature`` defaults to ``0.0`` on every protocol (``PLAN.md`` §7). Pass
    ``None`` to leave the parameter out where a provider rejects it.
    """
    credential = resolve_credential(connection)
    api_key = credential.reveal() if credential is not None else "not-needed"
    headers = connection_headers(connection, session_id)
    base_url = connection.base_url

    if connection.protocol in (Protocol.OPENAI_COMPATIBLE, Protocol.OPENAI_RESPONSES):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model_id,
            "api_key": api_key,
            "base_url": base_url,
            "default_headers": headers,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "timeout": timeout_s,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning and connection.provider != "local":
            kwargs["reasoning_effort"] = reasoning
        if connection.protocol is Protocol.OPENAI_RESPONSES:
            kwargs["use_responses_api"] = True
            kwargs["output_version"] = "responses/v1"
        kwargs.update(opts)
        return ChatOpenAI(**kwargs)

    if connection.protocol is Protocol.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model_id,
            "api_key": api_key,
            "base_url": base_url,
            "default_headers": headers,
            "max_tokens": max(1024, max_tokens),
            "max_retries": 0,
            "timeout": timeout_s,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if opts.get("thinking") is None:
            opts.pop("thinking", None)  # lowest level: no extended thinking request
        kwargs.update(opts)
        return ChatAnthropic(**kwargs)

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {
        "model": model_id,
        "api_key": api_key,
        "base_url": base_url,
        "additional_headers": headers,
        "max_tokens": max_tokens,
        "max_retries": 0,
        "timeout": timeout_s,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning:
        kwargs["thinking_level"] = reasoning
    kwargs.update(opts)
    return ChatGoogleGenerativeAI(**kwargs)
