"""Doctor: table, JSON shape, subscription CLIs as information, cache fallback."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from which_ai_model.cli import app
from which_ai_model.connect.doctor import (
    SUBSCRIPTION_CLIS,
    collect_report,
    doctor,
    format_doctor,
    report_to_dict,
)
from which_ai_model.connect.presets import default_connections, preset_connections
from which_ai_model.connect.store import save_connections

SECRET = "sk-" + "test-not-a-real-key-1234567890"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def _transport(status: int = 200, payload: object | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload if payload is not None else {"data": []})

    return httpx.MockTransport(handler)


def test_no_keys_reports_missing_and_never_crashes(tmp_path: Path) -> None:
    report = collect_report(
        default_connections(),
        session_id="run-1",
        cache_path=tmp_path / "catalog.json",
        which=lambda name: None,
        now=NOW,
    )
    assert all(row.reason.value == "CREDENTIAL_MISSING" for row in report.connections)
    assert all(row.key_found is False for row in report.connections)
    assert all(set(row.states.values()) == {"unknown"} for row in report.connections)
    assert [name for name, _ in report.subscription_clis] == list(SUBSCRIPTION_CLIS)


def test_json_shape_is_serialisable_and_secret_free(tmp_path: Path) -> None:
    report = collect_report(
        default_connections(),
        cache_path=tmp_path / "catalog.json",
        which=lambda name: None,
        now=NOW,
    )
    payload = report_to_dict(report)
    text = json.dumps(payload)
    assert SECRET not in text
    assert set(payload) == {
        "generated_at",
        "session_id",
        "benchmark_models",
        "connections",
        "subscription_clis",
    }
    assert len(payload["benchmark_models"]) == 7
    assert all(
        item["note"] == "subscription CLI — not used for model calls"
        for item in payload["subscription_clis"]
    )
    assert set(payload["connections"][0]) >= {"id", "protocol", "key", "reason", "models", "stale"}


def test_keyless_local_connection_reports_no_key_not_missing(tmp_path: Path) -> None:
    local = preset_connections("local-openai-compatible")
    report = collect_report(
        local,
        cache_path=tmp_path / "catalog.json",
        transport=_transport(200, {"data": [{"id": "llama"}]}),
        which=lambda name: None,
        now=NOW,
    )
    row = report.connections[0]
    assert row.key_found is None
    assert row.reason.value == "OK"
    assert row.model_count == 1
    assert row.states["gpt-6-luna"].value == "no"


def test_failed_refresh_serves_the_cache_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "catalog.json"
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    connections = preset_connections("openai")
    first = collect_report(
        connections,
        cache_path=cache,
        transport=_transport(200, {"data": [{"id": "gpt-6-luna"}]}),
        which=lambda name: None,
        now=NOW,
    )
    assert first.connections[0].stale is False
    assert first.connections[0].model_count == 1

    monkeypatch.delenv("OPENAI_API_KEY")
    second = collect_report(connections, cache_path=cache, which=lambda name: None, now=NOW)
    row = second.connections[0]
    assert row.reason.value == "CREDENTIAL_MISSING"
    assert row.stale is True
    assert row.model_count == 1
    assert row.states["gpt-6-luna"].value == "yes"


def test_subscription_clis_are_looked_up_but_never_run(tmp_path: Path) -> None:
    seen: list[str] = []

    def which(name: str) -> str | None:
        seen.append(name)
        return f"/usr/local/bin/{name}" if name == "claude" else None

    report = collect_report(
        preset_connections("openai"),
        cache_path=tmp_path / "catalog.json",
        which=which,
        now=NOW,
    )
    assert seen == list(SUBSCRIPTION_CLIS)
    assert dict(report.subscription_clis)["claude"] is True
    assert "claude: installed" in format_doctor(report)


def test_doctor_returns_probe_results() -> None:
    results = doctor(preset_connections("openai"))
    assert len(results) == 1
    assert results[0].reason.value == "CREDENTIAL_MISSING"


def test_table_names_every_column_and_model(tmp_path: Path) -> None:
    report = collect_report(
        default_connections()[:2], cache_path=tmp_path / "catalog.json", which=lambda name: None
    )
    table = format_doctor(report)
    assert "wam doctor" in table
    assert "opencode-go-openai" in table
    assert "Benchmark models" in table
    assert "unknown never becomes no" in table


def test_cli_json_works_with_no_keys(tmp_path: Path) -> None:
    config = tmp_path / "providers.toml"
    save_connections(default_connections(), config)
    result = CliRunner().invoke(
        app, ["doctor", "--json", "--config", str(config), "--cache", str(tmp_path / "c.json")]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["reason"] for row in payload["connections"]] == ["CREDENTIAL_MISSING"] * len(
        default_connections()
    )
    assert all(row["key"] == "MISSING" for row in payload["connections"])


def test_cli_bad_config_exits_two(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["doctor", "--config", str(tmp_path / "absent.toml")])
    assert result.exit_code == 2
    assert "not found" in result.output
