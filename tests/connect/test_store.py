"""providers.toml holds handles, never values, and refuses secret-looking input."""

from __future__ import annotations

from pathlib import Path

import pytest

from which_ai_model.connect.presets import default_connections
from which_ai_model.connect.store import (
    StoreError,
    connections_to_toml,
    load_connections,
    save_connections,
)

HEADER = "[connection]\n"


def _row(identifier: str = "demo", **extra: str) -> str:
    lines = [
        f'id = "{identifier}"',
        'provider = "demo"',
        'protocol = "OPENAI_COMPATIBLE"',
        'preset = "custom"',
    ]
    lines.extend(f'{key} = "{value}"' for key, value in extra.items())
    return "[[connection]]\n" + "\n".join(lines) + "\n"


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    connections = default_connections(include_local=True)
    save_connections(connections, path)
    assert load_connections(path) == connections


def test_committed_file_matches_the_presets(repo_root: Path) -> None:
    loaded = load_connections(repo_root / "config" / "providers.toml")
    assert loaded == default_connections()
    assert all(connection.credential_env for connection in loaded)


def test_saved_file_never_contains_a_resolved_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "very-secret-1234567890")
    text = connections_to_toml(default_connections())
    assert "very-secret-1234567890" not in text
    assert "env:OPENAI_API_KEY" in text


def test_a_secret_field_is_refused_without_echoing_the_value(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text(_row(api_key="sk-" + "live-abcdefghijkl"), encoding="utf-8")
    with pytest.raises(StoreError) as caught:
        load_connections(path)
    assert "live-abcdefghijkl" not in str(caught.value)


def test_a_secret_looking_handle_is_refused_without_echoing_it(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text(_row(credential_ref="sk-" + "live-abcdefghijkl"), encoding="utf-8")
    with pytest.raises(StoreError) as caught:
        load_connections(path)
    assert "live-abcdefghijkl" not in str(caught.value)


def test_unknown_fields_and_top_level_keys_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text(_row(note="hello"), encoding="utf-8")
    with pytest.raises(StoreError, match="unexpected fields"):
        load_connections(path)

    path.write_text("version = 1\n" + _row(), encoding="utf-8")
    with pytest.raises(StoreError, match="top-level"):
        load_connections(path)


def test_duplicate_ids_and_missing_files_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text(_row("demo") + _row("demo"), encoding="utf-8")
    with pytest.raises(StoreError, match="duplicate"):
        load_connections(path)
    with pytest.raises(StoreError, match="not found"):
        load_connections(tmp_path / "absent.toml")


def test_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(StoreError, match="no"):
        load_connections(path)


def test_broken_toml_reports_the_parser_without_echoing_content(tmp_path: Path) -> None:
    path = tmp_path / "providers.toml"
    path.write_text(HEADER + 'id = "sk-' + 'oops-not-a-real-key"\n', encoding="utf-8")
    with pytest.raises(StoreError) as caught:
        load_connections(path)
    assert "oops-not-a-real-key" not in str(caught.value)
