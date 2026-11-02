"""One cheap, non-inference request per connection.

Every protocol exposes a model-list endpoint, so a probe never spends a token. Outcomes
use only the frozen :class:`~which_ai_model.contracts.ProbeReason` vocabulary and
``detail`` is scrubbed of keys before it leaves this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from which_ai_model.connect.credentials import resolve_credential
from which_ai_model.connect.presets import GATEWAY_MODELS_URL, connection_headers
from which_ai_model.contracts import Connection, ProbeReason, ProbeResult, Protocol

__all__ = ["ProbeOutcome", "check_connection", "models_url", "probe_connection"]


@dataclass(frozen=True)
class ProbeOutcome:
    """A probe result plus the model ids it happened to see (empty on failure)."""

    connection_id: str
    reason: ProbeReason
    latency_ms: int
    detail: str | None = None
    model_ids: tuple[str, ...] = ()

    def result(self) -> ProbeResult:
        return ProbeResult(
            connection_id=self.connection_id,
            reason=self.reason,
            latency_ms=self.latency_ms,
            detail=self.detail,
        )


def models_url(connection: Connection) -> str:
    """The model-list URL for a connection.

    The gateway quirk (verified 2026-10-28) wins over every protocol rule: all four
    gateway connections list on the shared ``/inference/go/v1/models`` path, while
    ``/openai/v1/models`` answers 404.
    """
    if connection.preset == "opencode-go":
        return GATEWAY_MODELS_URL
    if connection.base_url is None:
        raise ValueError("base_url is required for a probe")
    base = connection.base_url.rstrip("/")
    if connection.protocol is Protocol.ANTHROPIC:
        return base + ("/models" if base.endswith("/v1") else "/v1/models")
    if connection.protocol is Protocol.GEMINI:
        return base + ("/models" if base.endswith("/v1beta") else "/v1beta/models")
    return base + "/models"


def _auth_headers(connection: Connection, key: str | None) -> dict[str, str]:
    if key is None:
        return {}
    if connection.protocol is Protocol.ANTHROPIC:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if connection.protocol is Protocol.GEMINI:
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


def _parse_ids(payload: Any) -> tuple[str, ...] | None:
    """Model ids from an OpenAI/Anthropic ``data`` list or a Gemini ``models`` list."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("data", payload.get("models"))
    if not isinstance(raw, list):
        return None
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("id", item.get("name"))
        if isinstance(value, str) and value:
            ids.append(value.removeprefix("models/"))
    return tuple(dict.fromkeys(ids))


def _reason_for_status(status: int) -> ProbeReason:
    if 200 <= status < 300:
        return ProbeReason.OK
    if status in (401, 403):
        return ProbeReason.CREDENTIAL_REJECTED
    if status == 429:
        return ProbeReason.RATE_LIMITED
    if status in (400, 404):
        return ProbeReason.CONFIG_INVALID
    if status == 405:
        return ProbeReason.PROTOCOL_UNSUPPORTED
    if status >= 500:
        return ProbeReason.ENDPOINT_UNREACHABLE
    return ProbeReason.UNKNOWN


def check_connection(
    connection: Connection,
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ProbeOutcome:
    """Probe one connection without inference; never raises for a failed probe."""
    credential = resolve_credential(connection)
    if connection.credential_env is not None and credential is None:
        return ProbeOutcome(connection.id, ProbeReason.CREDENTIAL_MISSING, 0, "env var not set")
    if connection.base_url is None and connection.preset != "opencode-go":
        return ProbeOutcome(connection.id, ProbeReason.CONFIG_INVALID, 0, "base_url missing")

    key = credential.reveal() if credential is not None else None
    headers = dict(connection_headers(connection, session_id))
    headers.update(_auth_headers(connection, key))
    try:
        url = models_url(connection)
    except ValueError:
        return ProbeOutcome(connection.id, ProbeReason.CONFIG_INVALID, 0, "base_url invalid")

    started = time.perf_counter()
    try:
        with httpx.Client(
            transport=transport,
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.get(url, headers=headers)
    except httpx.TimeoutException:
        return ProbeOutcome(
            connection.id,
            ProbeReason.ENDPOINT_UNREACHABLE,
            _elapsed_ms(started),
            "timeout",
        )
    except httpx.HTTPError as exc:
        return ProbeOutcome(
            connection.id,
            ProbeReason.ENDPOINT_UNREACHABLE,
            _elapsed_ms(started),
            type(exc).__name__,
        )

    latency = _elapsed_ms(started)
    reason = _reason_for_status(response.status_code)
    if reason is not ProbeReason.OK:
        return ProbeOutcome(connection.id, reason, latency, f"HTTP {response.status_code}")
    try:
        ids = _parse_ids(response.json())
    except ValueError:
        return ProbeOutcome(connection.id, ProbeReason.CONFIG_INVALID, latency, "invalid JSON")
    if ids is None:
        return ProbeOutcome(
            connection.id,
            ProbeReason.CONFIG_INVALID,
            latency,
            "model list missing",
        )
    return ProbeOutcome(connection.id, ProbeReason.OK, latency, None, ids)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def probe_connection(
    connection: Connection,
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ProbeResult:
    """The public probe used by ``doctor`` and tests."""
    return check_connection(
        connection, timeout_s=timeout_s, session_id=session_id, transport=transport
    ).result()
