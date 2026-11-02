"""The dynamic output model accepts every allowed answer and rejects the rest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import pytest
from pydantic import ValidationError

from which_ai_model.contracts import OutputKind, TaskSpec
from which_ai_model.task import build_output_model, load_spec

FIXTURE = Path("tests/fixtures/task")


def _spec(repo_root: Path) -> TaskSpec:
    return load_spec(repo_root / FIXTURE / "example.toml")


def test_schema_fields_and_literal(repo_root: Path) -> None:
    model = build_output_model(_spec(repo_root), root=repo_root)
    assert set(model.model_fields) == {"queue", "urgent"}
    label = model.model_fields["queue"].annotation
    assert get_args(label) == ("billing", "technical", "account", "unclear")
    assert model.model_fields["urgent"].annotation is bool


def test_schema_accepts_allowed_answers(repo_root: Path) -> None:
    model = build_output_model(_spec(repo_root), root=repo_root)
    parsed = model.model_validate({"queue": "billing", "urgent": True})
    assert parsed.model_dump() == {"queue": "billing", "urgent": True}
    assert model.model_validate({"queue": "unclear", "urgent": False}).queue == "unclear"


def test_schema_rejects_an_unknown_label(repo_root: Path) -> None:
    model = build_output_model(_spec(repo_root), root=repo_root)
    with pytest.raises(ValidationError, match="queue"):
        model.model_validate({"queue": "sales", "urgent": False})


def test_schema_rejects_a_missing_field_and_an_extra_one(repo_root: Path) -> None:
    model = build_output_model(_spec(repo_root), root=repo_root)
    with pytest.raises(ValidationError, match="urgent"):
        model.model_validate({"queue": "billing"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({"queue": "billing", "urgent": False, "note": "hi"})


def test_unclear_is_rejected_when_the_spec_does_not_allow_it(repo_root: Path) -> None:
    spec = _spec(repo_root)
    strict = spec.model_copy(
        update={
            "outputs": [
                spec.outputs[0].model_copy(update={"allow_unclear": False}),
                spec.outputs[1],
            ]
        }
    )
    model = build_output_model(strict, root=repo_root)
    label = model.model_fields["queue"].annotation
    assert get_args(label) == ("billing", "technical", "account")
    with pytest.raises(ValidationError, match="queue"):
        model.model_validate({"queue": "unclear", "urgent": False})


def test_bool_only_spec_has_no_literal(repo_root: Path) -> None:
    spec = _spec(repo_root)
    only_bool = spec.model_copy(
        update={"outputs": [output for output in spec.outputs if output.kind is OutputKind.BOOL]}
    )
    model = build_output_model(only_bool, root=repo_root)
    assert set(model.model_fields) == {"urgent"}
    assert model.model_fields["urgent"].annotation is bool


def test_literal_type_is_exported_for_readers(repo_root: Path) -> None:
    # A guard against a runtime-built Literal quietly becoming a plain ``str``.
    model = build_output_model(_spec(repo_root), root=repo_root)
    annotation = model.model_fields["queue"].annotation
    assert isinstance(annotation, type(Literal["x"]))
