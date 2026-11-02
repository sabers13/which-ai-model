"""``wam doctor``: probe every connection and report it without ever printing a key.

The table has one row per connection and a compact tri-state column for the seven
benchmark models. Subscription CLIs are reported from ``shutil.which`` only: they are
information ("installed here"), never a model backend for this project.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from which_ai_model.connect.credentials import resolve_credential
from which_ai_model.connect.discovery import (
    CACHE_PATH,
    Catalog,
    check_discovery,
    load_catalog,
)
from which_ai_model.connect.presets import BENCHMARK_MODELS
from which_ai_model.connect.probe import probe_connection
from which_ai_model.contracts import Connection, ProbeReason, ProbeResult, TriState

__all__ = [
    "SUBSCRIPTION_CLIS",
    "ConnectionReport",
    "DoctorReport",
    "collect_report",
    "doctor",
    "format_doctor",
    "report_to_dict",
]

#: Reported as installed-or-not; never executed and never used for model calls.
SUBSCRIPTION_CLIS: tuple[str, ...] = ("claude", "codex", "opencode", "gemini", "agy")
_CLI_NOTE = "subscription CLI — not used for model calls"

_STATUS_MARK = {TriState.YES: "Y", TriState.NO: "N", TriState.UNKNOWN: "?"}


@dataclass(frozen=True)
class ConnectionReport:
    """One doctor row: probe outcome plus the model matrix for a connection."""

    connection: Connection
    key_found: bool | None
    reason: ProbeReason
    latency_ms: int
    detail: str | None
    model_count: int | None
    states: Mapping[str, TriState]
    stale: bool


@dataclass(frozen=True)
class DoctorReport:
    """Everything `wam doctor` knows after one pass."""

    generated_at: datetime
    session_id: str
    connections: tuple[ConnectionReport, ...]
    subscription_clis: tuple[tuple[str, bool], ...]


def doctor(
    connections: Sequence[Connection],
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[ProbeResult]:
    """Probe only (the interface-contract shape); the CLI collects the richer report."""
    return [
        probe_connection(
            connection, timeout_s=timeout_s, session_id=session_id, transport=transport
        )
        for connection in connections
    ]


def collect_report(
    connections: Sequence[Connection],
    *,
    timeout_s: float = 20.0,
    session_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
    cache_path: Path | None = CACHE_PATH,
    which: Callable[[str], str | None] = shutil.which,
    now: datetime | None = None,
) -> DoctorReport:
    """Probe, list models, merge the cache, and report installed subscription CLIs."""
    generated_at = now or datetime.now(UTC)
    catalog = load_catalog(cache_path) if cache_path is not None else Catalog(Path("unused"))
    rows: list[ConnectionReport] = []
    for connection in connections:
        outcome = check_discovery(
            connection,
            timeout_s=timeout_s,
            session_id=session_id,
            transport=transport,
            now=generated_at,
        )
        if outcome.reason is ProbeReason.OK:
            catalog.record(outcome)
            count: int | None = outcome.model_count
            states = outcome.states
            stale = False
        else:
            cached = catalog.lookup(connection.id, now=generated_at)
            if cached is not None:
                entry, _aged = cached
                count = entry.model_count
                states = {
                    model: entry.states.get(model, TriState.UNKNOWN) for model in BENCHMARK_MODELS
                }
                stale = True
            else:
                count = None
                states = outcome.states
                stale = False
        key_found = None if connection.credential_env is None else _has_key(connection)
        rows.append(
            ConnectionReport(
                connection=connection,
                key_found=key_found,
                reason=outcome.reason,
                latency_ms=outcome.latency_ms,
                detail=outcome.detail,
                model_count=count,
                states=states,
                stale=stale,
            )
        )
    if cache_path is not None:
        with contextlib.suppress(OSError):
            catalog.save()
    return DoctorReport(
        generated_at=generated_at,
        session_id=session_id or "",
        connections=tuple(rows),
        subscription_clis=tuple((name, which(name) is not None) for name in SUBSCRIPTION_CLIS),
    )


def _has_key(connection: Connection) -> bool:
    return resolve_credential(connection) is not None


def _key_label(key_found: bool | None) -> str:
    if key_found is None:
        return "keyless"
    return "found" if key_found else "MISSING"


def format_doctor(report: DoctorReport | Sequence[ProbeResult]) -> str:
    """A plain-text table (no terminal control codes) for the CLI and tests."""
    if not isinstance(report, DoctorReport):
        table = Table(title="wam doctor", header_style="bold")
        for column in ("connection", "reason", "ms", "detail"):
            table.add_column(column)
        for result in report:
            table.add_row(
                result.connection_id,
                result.reason.value,
                str(result.latency_ms),
                result.detail or "",
            )
        return _render(table)

    table = Table(title="wam doctor", header_style="bold")
    for column in (
        "connection",
        "protocol",
        "key",
        "probe",
        "ms",
        "models",
        "Benchmark models (Y/N/?)*",
    ):
        table.add_column(column, overflow="fold")
    for row in report.connections:
        marks = "".join(
            _STATUS_MARK[row.states.get(model, TriState.UNKNOWN)] for model in BENCHMARK_MODELS
        )
        models = (
            "-"
            if row.model_count is None
            else str(row.model_count) + (" stale" if row.stale else "")
        )
        table.add_row(
            row.connection.id,
            row.connection.protocol.value,
            _key_label(row.key_found),
            row.reason.value,
            str(row.latency_ms),
            models,
            marks,
        )
    lines = [
        _render(table),
        "",
        "* model order: " + ", ".join(BENCHMARK_MODELS),
        "  Y yes · N no · ? unknown (unknown never becomes no)",
        "",
        "Subscription CLIs (information only — not used for model calls):",
    ]
    lines.extend(
        f"  {name}: {'installed' if installed else 'not installed'}"
        for name, installed in report.subscription_clis
    )
    return "\n".join(lines)


def _render(table: Table) -> str:
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=False, width=200, color_system=None).print(table)
    return buffer.getvalue().rstrip("\n")


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    """The JSON shape printed by ``wam doctor --json``. No keys, no headers."""
    return {
        "generated_at": report.generated_at.isoformat(),
        "session_id": report.session_id,
        "benchmark_models": list(BENCHMARK_MODELS),
        "connections": [
            {
                "id": row.connection.id,
                "provider": row.connection.provider,
                "protocol": row.connection.protocol.value,
                "base_url": row.connection.base_url,
                "key": _key_label(row.key_found),
                "reason": row.reason.value,
                "latency_ms": row.latency_ms,
                "detail": row.detail,
                "model_count": row.model_count,
                "models": {
                    model: row.states.get(model, TriState.UNKNOWN).value
                    for model in BENCHMARK_MODELS
                },
                "stale": row.stale,
            }
            for row in report.connections
        ],
        "subscription_clis": [
            {"name": name, "installed": installed, "note": _CLI_NOTE}
            for name, installed in report.subscription_clis
        ],
    }
