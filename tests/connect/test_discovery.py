"""Discovery is tri-state; the cache keeps last-known truth and marks it stale."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from which_ai_model.connect.discovery import (
    BENCHMARK_MODELS,
    Catalog,
    check_discovery,
    discover_models,
    load_catalog,
)
from which_ai_model.connect.presets import preset_connections
from which_ai_model.contracts import Connection, TriState

SECRET = "sk-" + "test-not-a-real-key-1234567890"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def _openai() -> Connection:
    return preset_connections("openai")[0]


def _transport(status: int, payload: object) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_listed_and_absent_models_are_yes_and_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    transport = _transport(200, {"data": [{"id": "gpt-6-luna"}, {"id": "other-model"}]})
    outcome = check_discovery(_openai(), transport=transport, now=NOW)
    assert outcome.reason.value == "OK"
    assert outcome.model_count == 2
    assert outcome.states["gpt-6-luna"] is TriState.YES
    assert outcome.states["claude-haiku-5-5"] is TriState.NO


def test_failed_listing_is_unknown_never_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    outcome = check_discovery(_openai(), transport=_transport(401, {"error": "no"}), now=NOW)
    assert set(outcome.states.values()) == {TriState.UNKNOWN}


def test_missing_key_is_unknown_without_a_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("unexpected request")

    outcome = check_discovery(_openai(), transport=httpx.MockTransport(handler), now=NOW)
    assert set(outcome.states.values()) == {TriState.UNKNOWN}
    assert outcome.reason.value == "CREDENTIAL_MISSING"


def test_discover_models_returns_the_seven_benchmark_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    models = discover_models(
        _openai(), transport=_transport(200, {"data": [{"id": "gpt-6-luna"}]}), now=NOW
    )
    assert [model.model_id for model in models] == list(BENCHMARK_MODELS)
    assert all(model.observed_at == NOW for model in models)
    assert models[1].listed is TriState.YES


def test_catalog_round_trip_and_staleness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    path = tmp_path / "models.catalog.json"
    catalog = Catalog(path)
    catalog.record(
        check_discovery(
            _openai(), transport=_transport(200, {"data": [{"id": "gpt-6-luna"}]}), now=NOW
        )
    )
    catalog.save()

    loaded = load_catalog(path)
    entry = loaded.lookup("openai", now=NOW + timedelta(hours=1))
    assert entry is not None
    cached, stale = entry
    assert cached.states["gpt-6-luna"] is TriState.YES
    assert cached.model_count == 1
    assert stale is False
    late = loaded.lookup("openai", now=NOW + timedelta(days=2))
    assert late is not None
    assert late[1] is True


def test_failed_refresh_does_not_erase_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    path = tmp_path / "models.catalog.json"
    catalog = Catalog(path)
    catalog.record(
        check_discovery(
            _openai(), transport=_transport(200, {"data": [{"id": "gpt-6-luna"}]}), now=NOW
        )
    )
    failed = check_discovery(
        _openai(), transport=_transport(500, {}), now=NOW + timedelta(minutes=5)
    )
    catalog.record(failed)
    assert catalog.lookup("openai") is not None
    assert catalog.connections["openai"].states["gpt-6-luna"] is TriState.YES


def test_corrupt_or_missing_catalog_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "models.catalog.json"
    assert load_catalog(path).connections == {}
    path.write_text("{not json", encoding="utf-8")
    assert load_catalog(path).connections == {}
    path.write_text('{"connections": {"x": {"observed_at": "nope"}}}', encoding="utf-8")
    assert load_catalog(path).connections == {}
