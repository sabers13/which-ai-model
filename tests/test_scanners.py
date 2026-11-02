"""Unit tests for the two repository scanners, using temporary files only."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wam_scripts_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def forbidden() -> ModuleType:
    return _load_script("scan_forbidden")


@pytest.fixture(scope="module")
def secrets() -> ModuleType:
    return _load_script("scan_secrets")


def test_load_hashes_ignores_comments_and_blank_lines(
    forbidden: ModuleType, tmp_path: Path
) -> None:
    digest = "a" * 64
    path = tmp_path / "hashes.txt"
    path.write_text(f"# a comment\n\n{digest}\n", encoding="utf-8")
    assert forbidden.load_hashes(path) == {digest}


def test_load_hashes_rejects_non_sha256(forbidden: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "hashes.txt"
    path.write_text("not-a-hash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a sha256"):
        forbidden.load_hashes(path)


def test_token_hashes_splits_dotted_and_dashed_runs(forbidden: ModuleType) -> None:
    tokens = {token for token, _ in forbidden.token_hashes("Jane.Doe met jane-doe")}
    assert {"jane.doe", "jane", "doe"} <= tokens


def test_find_forbidden_lines_reports_line_not_token(forbidden: ModuleType) -> None:
    hashes = {forbidden.hashlib.sha256(b"acme").hexdigest()}
    hits = forbidden.find_forbidden_lines("clean line\nAcme Corp\n", hashes)
    assert len(hits) == 1
    assert hits[0][0] == 2
    assert "acme" not in hits[0][1]


def test_scan_checks_a_file(forbidden: ModuleType, tmp_path: Path) -> None:
    hashes = {forbidden.hashlib.sha256(b"acme").hexdigest()}
    target = tmp_path / "note.txt"
    target.write_text("Acme\n", encoding="utf-8")
    findings = forbidden.scan(tmp_path, hashes, commits=0, files=[target])
    assert findings == ["note.txt:1: forbidden token (sha256 " + next(iter(hashes))[:12] + "…)"]


def test_scan_reads_commit_messages(forbidden: ModuleType, tmp_path: Path) -> None:
    hashes = {forbidden.hashlib.sha256(b"acme").hexdigest()}
    target = tmp_path / "readme.txt"
    target.write_text("clean\n", encoding="utf-8")
    try:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@example.invalid"], cwd=tmp_path, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "readme.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "mention Acme"], cwd=tmp_path, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    findings = forbidden.scan(tmp_path, hashes, commits=5, files=[target])
    assert len(findings) == 1
    assert findings[0].startswith("commit ")


def test_secret_patterns(secrets: ModuleType) -> None:
    assert secrets.scan_text('key = "sk-' + "a" * 32 + '"') == [(1, "openai-style key")]
    assert secrets.scan_text('api_key = "' + "b" * 40 + '"') == [(1, "secret assignment")]
    assert secrets.scan_text("token: '" + "c" * 40 + "'") == [(1, "secret assignment")]


def test_placeholders_and_env_refs_are_safe(secrets: ModuleType) -> None:
    assert secrets.scan_text('api_key = "sk-..."') == []
    assert secrets.scan_text('credential_ref = "env:OPENCODE_GO_API_KEY"') == []
    assert secrets.scan_text('api_key = "test-key-not-a-secret"') == []
    assert secrets.scan_text('password = os.environ["WAM_PASSWORD"]') == []


def test_secret_scan_never_prints_the_value(secrets: ModuleType, tmp_path: Path) -> None:
    value = "sk-" + "d" * 32
    target = tmp_path / "leak.txt"
    target.write_text(f"key = {value}\n", encoding="utf-8")
    findings = secrets.scan(tmp_path, files=[target])
    assert findings == ["leak.txt:1: possible openai-style key (value not shown)"]
    assert value not in "\n".join(findings)
