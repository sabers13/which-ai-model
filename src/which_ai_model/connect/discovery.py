"""Tri-state model discovery with a small on-disk catalog.

``unknown`` is the answer whenever the listing could not be read; it never decays to
``no``. The catalog caches the last observation so ``doctor`` can show what was true
before, clearly marked stale, instead of pretending a model disappeared.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from which_ai_model.connect.presets import BENCHMARK_MODELS
from which_ai_model.connect.probe import check_connection
from which_ai_model.contracts import Connection, DiscoveredModel, ProbeReason, TriState

__all__ = [
    "Catalog",
    "DiscoveryOutcome",
    "check_discovery",
    "discover_models",
    "load_catalog",
]

CACHE_PATH = Path("config/models.catalog.json")
MAX_AGE_S = 24 * 60 * 60


@dataclass(frozen=True)
class DiscoveryOutcome:
    """One listing attempt: the probe reason plus a state for every benchmark model."""

    connection_id: str
    reason: ProbeReason
    latency_ms: int
    detail: str | None
    model_count: int
    observed_at: datetime
    states: Mapping[str, TriState] = field(default_factory=dict)
    model_ids: tuple[str, ...] = ()


def check_discovery(
    connection: Connection,
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> DiscoveryOutcome:
    """List models once and project the benchmark ids to yes/no/unknown."""
    probe = check_connection(
        connection, timeout_s=timeout_s, session_id=session_id, transport=transport
    )
    observed = now or datetime.now(UTC)
    listed = set(probe.model_ids)
    if probe.reason is ProbeReason.OK:
        states = {
            model: TriState.YES if model in listed else TriState.NO for model in BENCHMARK_MODELS
        }
    else:
        states = dict.fromkeys(BENCHMARK_MODELS, TriState.UNKNOWN)
    return DiscoveryOutcome(
        connection_id=connection.id,
        reason=probe.reason,
        latency_ms=probe.latency_ms,
        detail=probe.detail,
        model_count=len(listed),
        observed_at=observed,
        states=states,
        model_ids=probe.model_ids,
    )


def discover_models(
    connection: Connection,
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> list[DiscoveredModel]:
    """One :class:`DiscoveredModel` per benchmark model, tri-state."""
    outcome = check_discovery(
        connection, timeout_s=timeout_s, session_id=session_id, transport=transport, now=now
    )
    return [
        DiscoveredModel(
            connection_id=connection.id,
            model_id=model,
            listed=state,
            observed_at=outcome.observed_at,
        )
        for model, state in outcome.states.items()
    ]


@dataclass
class CachedConnection:
    """Last known states for one connection."""

    observed_at: datetime
    model_count: int
    states: dict[str, TriState]


class Catalog:
    """The cached catalog. Corrupt or missing files simply start empty."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self.path = path
        self.connections: dict[str, CachedConnection] = {}

    def record(self, outcome: DiscoveryOutcome) -> None:
        if outcome.reason is not ProbeReason.OK:
            return
        self.connections[outcome.connection_id] = CachedConnection(
            observed_at=outcome.observed_at,
            model_count=outcome.model_count,
            states=dict(outcome.states),
        )

    def lookup(
        self, connection_id: str, *, max_age_s: int = MAX_AGE_S, now: datetime | None = None
    ) -> tuple[CachedConnection, bool] | None:
        """The cached entry and whether it is older than ``max_age_s``."""
        entry = self.connections.get(connection_id)
        if entry is None:
            return None
        age = ((now or datetime.now(UTC)) - entry.observed_at).total_seconds()
        return entry, age > max_age_s

    def save(self) -> None:
        document = {
            "schema_version": 1,
            "connections": {
                connection_id: {
                    "observed_at": entry.observed_at.isoformat(),
                    "model_count": entry.model_count,
                    "states": {model: state.value for model, state in entry.states.items()},
                }
                for connection_id, entry in sorted(self.connections.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def load_catalog(path: Path = CACHE_PATH) -> Catalog:
    """Load the catalog; any unreadable or malformed entry is dropped, never fatal."""
    catalog = Catalog(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return catalog
    rows = document.get("connections") if isinstance(document, dict) else None
    if not isinstance(rows, dict):
        return catalog
    for connection_id, row in rows.items():
        if not isinstance(row, dict):
            continue
        observed_at = _parse_time(row.get("observed_at"))
        states = row.get("states")
        if observed_at is None or not isinstance(states, dict):
            continue
        clean: dict[str, TriState] = {}
        for model, state in states.items():
            try:
                clean[str(model)] = TriState(state)
            except ValueError:
                continue
        count = row.get("model_count")
        catalog.connections[str(connection_id)] = CachedConnection(
            observed_at=observed_at,
            model_count=count if isinstance(count, int) and count >= 0 else 0,
            states=clean,
        )
    return catalog
