"""Spec loader and validator: valid spec, plain-language problems (``PLAN-v1.1`` §A3)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from which_ai_model.contracts import BaselineSpec, TaskSpec
from which_ai_model.task import SpecError, load_labels, load_spec, validate_spec

FIXTURE = Path("tests/fixtures/task")


@pytest.fixture
def task_root(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A copy of the committed fixture under a temporary repository root."""
    destination = tmp_path / FIXTURE
    shutil.copytree(fixtures_dir / "task", destination)
    return tmp_path


def _spec(task_root: Path) -> TaskSpec:
    return load_spec(task_root / FIXTURE / "example.toml")


def test_valid_spec_loads_and_validates(task_root: Path) -> None:
    spec = _spec(task_root)
    assert spec.id == "example"
    assert [output.name for output in spec.outputs] == ["queue", "urgent"]
    assert spec.outputs[0].allow_unclear is True
    assert validate_spec(spec, root=task_root) == []
    assert load_labels(spec, root=task_root) == {
        "billing": "Invoices, payments, refunds and plan changes",
        "technical": "Errors, outages, setup and anything that needs a fix",
        "account": "Sign-in, passwords, profile details and access rights",
    }


def test_broken_toml_is_one_plain_message(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text('id = "broken"\ntitle = \n', encoding="utf-8")
    with pytest.raises(SpecError, match="not valid TOML"):
        load_spec(path)


def test_unknown_field_is_rejected_with_its_location(tmp_path: Path) -> None:
    path = tmp_path / "wrong.toml"
    path.write_text(
        'id = "wrong"\ntitle = "t"\ndescription = "d"\nprompt_version = "w1"\n'
        'scorer_version = "s1"\n[data]\nfile = "x.jsonl"\n[prompt]\n'
        'template_file = "x.txt"\n[[outputs]]\nname = "label"\nkind = "label"\n'
        'truth_field = "label"\n[metrics]\nprimary = "label.macro_f1"\n',
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="labels_file"):
        load_spec(path)


def test_missing_data_file_is_named(task_root: Path) -> None:
    (task_root / FIXTURE / "example.jsonl").unlink()
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("data file" in problem and "missing" in problem for problem in problems)


def test_template_problems_are_explained(task_root: Path) -> None:
    template = task_root / FIXTURE / "example.prompt.txt"
    template.write_text('Only the text: {text} and JSON {"queue": "billing"}\n', encoding="utf-8")
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("needs the placeholder {labels}" in problem for problem in problems)


def test_empty_template_problem(task_root: Path) -> None:
    (task_root / FIXTURE / "example.prompt.txt").unlink()
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("prompt template" in problem and "missing" in problem for problem in problems)


def _rows(task_root: Path) -> list[dict[str, Any]]:
    path = task_root / FIXTURE / "example.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_rows(task_root: Path, rows: list[dict[str, Any]]) -> None:
    path = task_root / FIXTURE / "example.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_duplicate_ids_are_reported(task_root: Path) -> None:
    rows = _rows(task_root)
    rows[1]["id"] = rows[0]["id"]
    _write_rows(task_root, rows)
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("share the id" in problem for problem in problems)


def test_unknown_truth_label_is_reported(task_root: Path) -> None:
    rows = _rows(task_root)
    rows[2]["queue"] = "sales"
    _write_rows(task_root, rows)
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("queues.toml" in problem for problem in problems)
    assert any("every truth label must be one of the known ids" in problem for problem in problems)


def test_bool_truth_field_needs_a_real_boolean(task_root: Path) -> None:
    rows = _rows(task_root)
    rows[0]["urgent"] = "yes"
    _write_rows(task_root, rows)
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("wrong type" in problem for problem in problems)


def test_missing_text_is_reported(task_root: Path) -> None:
    rows = _rows(task_root)
    rows[0].pop("text")
    _write_rows(task_root, rows)
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("'text'" in problem for problem in problems)


def test_reserved_unclear_label_is_rejected(task_root: Path) -> None:
    labels = task_root / FIXTURE / "queues.toml"
    labels.write_text(
        '[[groups]]\nid = "billing"\ndescription = "Billing"\n'
        '[[groups]]\nid = "unclear"\ndescription = "Reserved"\n',
        encoding="utf-8",
    )
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("reserved for abstentions" in problem for problem in problems)


def test_labels_table_shape_is_accepted(task_root: Path) -> None:
    labels = task_root / FIXTURE / "queues.toml"
    labels.write_text(
        '[labels]\nbilling = "Invoices and payments"\ntechnical = "Errors and outages"\n',
        encoding="utf-8",
    )
    assert load_labels(_spec(task_root), root=task_root) == {
        "billing": "Invoices and payments",
        "technical": "Errors and outages",
    }
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("every truth label must be one of the known ids" in problem for problem in problems)


def test_labels_file_without_a_container_is_explained(task_root: Path) -> None:
    (task_root / FIXTURE / "queues.toml").write_text('billing = "Billing"\n', encoding="utf-8")
    problems = validate_spec(_spec(task_root), root=task_root)
    assert any("lists no labels" in problem for problem in problems)


def test_missing_baseline_file_is_reported(task_root: Path) -> None:
    spec = _spec(task_root).model_copy(
        update={"baseline": {"queue": BaselineSpec(kind="keyword", keywords_file="nope.txt")}}
    )
    problems = validate_spec(spec, root=task_root)
    assert any("nope.txt" in problem and "missing" in problem for problem in problems)
