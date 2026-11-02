"""Credential handles resolve at call time and print as ``***``."""

from __future__ import annotations

from pathlib import Path

import pytest

from which_ai_model.connect.credentials import (
    Credential,
    read_env_file,
    resolve_credential,
    resolve_credential_value,
)
from which_ai_model.connect.presets import preset_connections
from which_ai_model.contracts import Connection

SECRET = "sk-" + "test-not-a-real-key-1234567890"


def _openai() -> Connection:
    return preset_connections("openai")[0]


def test_environment_wins_and_the_value_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    credential = resolve_credential(_openai())
    assert isinstance(credential, Credential)
    assert credential is not None
    assert credential.reveal() == SECRET
    assert str(credential) == "***"
    assert SECRET not in repr(credential)
    assert credential.env_name == "OPENAI_API_KEY"


def test_env_file_is_a_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"# comment\nexport OPENAI_API_KEY='{SECRET}'\n", encoding="utf-8")
    monkeypatch.setenv("WAM_ENV_FILE", str(env_file))
    credential = resolve_credential(_openai())
    assert credential is not None
    assert credential.reveal() == SECRET
    assert read_env_file(env_file) == {"OPENAI_API_KEY": SECRET}


def test_process_environment_wins_over_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("WAM_ENV_FILE", str(env_file))
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    credential = resolve_credential(_openai())
    assert credential is not None
    assert credential.reveal() == "from-env"


def test_missing_and_empty_values_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_credential(_openai()) is None
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert resolve_credential(_openai()) is None
    assert resolve_credential_value(_openai()) is None


def test_keyless_connections_resolve_to_none() -> None:
    local = preset_connections("local-openai-compatible")[0]
    assert resolve_credential(local) is None


def test_raw_value_helper_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert resolve_credential_value(_openai()) == SECRET


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "absent.env") == {}
