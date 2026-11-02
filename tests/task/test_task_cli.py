"""`wam task check` and `wam task new` say what is wrong and what they wrote."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from which_ai_model.task import load_spec
from which_ai_model.task.cli import app

FIXTURE = Path("tests/fixtures/task")
runner = CliRunner()


def test_check_prints_ok_for_the_fixture(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.chdir(repo_root)
    result = runner.invoke(app, ["check", str(FIXTURE / "example.toml")])
    assert result.exit_code == 0
    assert "OK: 'example' is valid" in result.output
    assert "queue (label)" in result.output
    assert "urgent (bool)" in result.output


def test_check_lists_each_problem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path
) -> None:
    import shutil

    root = tmp_path
    destination = root / FIXTURE
    shutil.copytree(fixtures_dir / "task", destination)
    rows = (destination / "example.jsonl").read_text(encoding="utf-8").splitlines()
    rows[0] = rows[0].replace('"queue": "billing"', '"queue": "sales"')
    (destination / "example.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (destination / "example.prompt.txt").write_text("No placeholders here.", encoding="utf-8")

    monkeypatch.chdir(root)
    result = runner.invoke(app, ["check", str(FIXTURE / "example.toml")])
    assert result.exit_code == 1
    assert "problems to fix" in result.output
    assert "1." in result.output
    assert "every truth label must be one of the known ids" in result.output
    assert "{text}" in result.output


def test_check_reports_a_missing_spec(tmp_path: Path) -> None:
    result = runner.invoke(app, ["check", str(tmp_path / "nothing.toml")])
    assert result.exit_code == 1
    assert "Cannot read the spec file" in result.output


def test_new_writes_a_loadable_skeleton(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    result = runner.invoke(app, ["new", "my-task", "--tasks-dir", str(tasks)])
    assert result.exit_code == 0
    spec_path = tasks / "my-task.toml"
    prompt_path = tasks / "my-task.prompt.txt"
    assert spec_path.is_file()
    assert prompt_path.is_file()
    assert "Wrote" in result.output

    spec = load_spec(spec_path)
    assert spec.id == "my-task"
    assert spec.outputs[0].name == "label"
    assert "{labels}" in prompt_path.read_text(encoding="utf-8")
    assert "{text}" in prompt_path.read_text(encoding="utf-8")


def test_new_refuses_to_overwrite(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    assert runner.invoke(app, ["new", "once", "--tasks-dir", str(tasks)]).exit_code == 0
    second = runner.invoke(app, ["new", "once", "--tasks-dir", str(tasks)])
    assert second.exit_code == 1
    assert "already exists" in second.output


def test_new_rejects_a_bad_id(tmp_path: Path) -> None:
    result = runner.invoke(app, ["new", "Not Valid", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "not a usable task id" in result.output
