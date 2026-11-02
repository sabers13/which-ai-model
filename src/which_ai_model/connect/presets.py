"""Known connection presets.

A preset is a named group of ``Connection`` rows. It stores ``env:NAME`` handles only;
the value is resolved at call time in :mod:`which_ai_model.connect.credentials`.
"""

from __future__ import annotations

from dataclasses import dataclass

from which_ai_model.contracts import Connection, Protocol

__all__ = [
    "BENCHMARK_MODELS",
    "GATEWAY_MODELS_URL",
    "USER_AGENT",
    "Preset",
    "all_presets",
    "connection_headers",
    "default_connections",
    "preset_connections",
]

#: One honest identifier for every request this project makes.
USER_AGENT = "which-ai-model/1.0 (+https://github.com/sabers13/which-ai-model)"

#: The gateway lists every model on one endpoint, whatever the wire format of a call.
GATEWAY_MODELS_URL = "https://opencode.ai/inference/go/v1/models"

#: The seven models fixed in ``PLAN.md`` §7, in plan order.
BENCHMARK_MODELS: tuple[str, ...] = (
    "claude-haiku-5-5",
    "gpt-6-luna",
    "gemini-4-argon",
    "muse-spark-1.3-contributor",
    "deepseek-v4.1-flash",
    "mimo-v2.6-flash",
    "glm-5.3-flash",
)

_SESSION_HEADER = "x-opencode-session"
_GATEWAY = "opencode-go"
_GATEWAY_REF = "env:OPENCODE_GO_API_KEY"


@dataclass(frozen=True)
class Preset:
    """A provider preset: its display summary and its connections."""

    name: str
    summary: str
    connections: tuple[Connection, ...]


def _connection(
    identifier: str,
    provider: str,
    protocol: Protocol,
    base_url: str | None,
    credential_ref: str | None,
    preset: str,
) -> Connection:
    return Connection(
        id=identifier,
        provider=provider,
        protocol=protocol,
        base_url=base_url,
        credential_ref=credential_ref,
        preset=preset,
    )


#: (id, protocol, base URL) — one connection per wire format, one shared key.
_GATEWAY_ROWS: tuple[tuple[str, Protocol, str], ...] = (
    (
        "opencode-go-openai",
        Protocol.OPENAI_COMPATIBLE,
        "https://opencode.ai/inference/go/openai/v1",
    ),
    (
        "opencode-go-responses",
        Protocol.OPENAI_RESPONSES,
        "https://opencode.ai/inference/go/openai/v1",
    ),
    ("opencode-go-anthropic", Protocol.ANTHROPIC, "https://opencode.ai/inference/go/anthropic"),
    # Root only: google-genai appends ``v1beta/models/<model>:<method>`` itself.
    ("opencode-go-google", Protocol.GEMINI, "https://opencode.ai/inference/go/google/"),
)

_GATEWAY_CONNECTIONS = tuple(
    _connection(identifier, _GATEWAY, protocol, base_url, _GATEWAY_REF, _GATEWAY)
    for identifier, protocol, base_url in _GATEWAY_ROWS
)

#: (name, summary, id, provider, protocol, base URL, credential ref)
_OTHER_ROWS: tuple[tuple[str, str, str, str, Protocol, str | None, str | None], ...] = (
    (
        "openai",
        "First-party OpenAI, Chat Completions.",
        "openai",
        "openai",
        Protocol.OPENAI_COMPATIBLE,
        "https://api.openai.com/v1",
        "env:OPENAI_API_KEY",
    ),
    (
        "anthropic",
        "First-party Anthropic Messages API (the SDK appends /v1/messages).",
        "anthropic",
        "anthropic",
        Protocol.ANTHROPIC,
        "https://api.anthropic.com",
        "env:ANTHROPIC_API_KEY",
    ),
    (
        "gemini",
        "First-party Google GenAI (root base; the SDK appends v1beta).",
        "gemini",
        "google",
        Protocol.GEMINI,
        "https://generativelanguage.googleapis.com",
        "env:GEMINI_API_KEY",
    ),
    (
        "openrouter",
        "OpenRouter aggregator, OpenAI-compatible.",
        "openrouter",
        "openrouter",
        Protocol.OPENAI_COMPATIBLE,
        "https://openrouter.ai/api/v1",
        "env:OPENROUTER_API_KEY",
    ),
    (
        "local-openai-compatible",
        "A local OpenAI-compatible server; no key.",
        "local-openai-compatible",
        "local",
        Protocol.OPENAI_COMPATIBLE,
        "http://localhost:11434/v1",
        None,
    ),
    (
        "custom",
        "Template for your own endpoint; fill id, provider, protocol and base_url.",
        "custom",
        "custom",
        Protocol.OPENAI_COMPATIBLE,
        None,
        "env:CUSTOM_API_KEY",
    ),
)

_PRESETS: tuple[Preset, ...] = (
    Preset(_GATEWAY, "One gateway key, one connection per wire protocol.", _GATEWAY_CONNECTIONS),
    *(
        Preset(
            name,
            summary,
            (_connection(identifier, provider, protocol, base_url, credential_ref, preset=name),),
        )
        for name, summary, identifier, provider, protocol, base_url, credential_ref in _OTHER_ROWS
    ),
)


def all_presets() -> dict[str, Preset]:
    """Every known preset, keyed by name, in display order."""
    return {preset.name: preset for preset in _PRESETS}


def preset_connections(name: str) -> list[Connection]:
    """The connections of one preset; ``KeyError`` for an unknown name."""
    return list(all_presets()[name].connections)


def default_connections(include_local: bool = False) -> list[Connection]:
    """Every preset's connections, gateway first.

    ``local-openai-compatible`` is opt-in: it is a keyless localhost endpoint, so it is
    not part of the file a fresh checkout probes by default.
    """
    return [
        connection
        for preset in _PRESETS
        if include_local or preset.name != "local-openai-compatible"
        for connection in preset.connections
    ]


def connection_headers(connection: Connection, session_id: str | None = None) -> dict[str, str]:
    """Fixed headers for one request. Only the gateway asks for a session header."""
    headers = {"User-Agent": USER_AGENT}
    if connection.preset == _GATEWAY:
        headers[_SESSION_HEADER] = session_id or "wam-no-session"
    return headers
