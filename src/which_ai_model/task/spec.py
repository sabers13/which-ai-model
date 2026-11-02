"""Load, validate and render a spec task (``PLAN-v1.1`` §A3).

A spec task is one TOML file plus its files: a labelled data file, a prompt template,
one labels file per label output and optional baseline sources. Nothing here calls a
model: the loader reads bytes and reports plain-language problems a non-programmer can
act on, and the renderer fills the one prompt every model receives.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from which_ai_model.contracts import (
    SPEC_ID_PATTERN,
    OutputKind,
    TaskSpec,
)

__all__ = [
    "SpecError",
    "build_output_model",
    "load_labels",
    "load_spec",
    "render_prompt",
    "validate_spec",
    "write_skeleton",
]

#: The answer that means "I cannot tell". It is never a real label id.
UNCLEAR = "unclear"

#: The two placeholders every prompt template must carry.
_PLACEHOLDERS = ("labels", "text")

#: Containers a labels file may use. ``groups`` is the canonical one (the T-DEFECT
#: file is ``groups.toml``); ``group`` is the singular table the NHTSA snapshot's
#: ``groups.toml`` uses (``[[group]]``); the others keep a hand-written file friendly.
_LABEL_CONTAINERS = ("groups", "group", "labels", "categories")

_MAX_EXAMPLES = 3


class SpecError(ValueError):
    """A spec or one of its files cannot be used. The message is plain language."""


def _read_toml(path: Path, *, what: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise SpecError(f"Cannot read the {what} file {path}: {reason}") from exc
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SpecError(f"The {what} file {path} is not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"The {what} file {path} is not valid TOML: {exc}") from exc


def _plain_validation_error(path: Path, exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "spec"
        lines.append(f"  - {where}: {error['msg']}")
    return f"{path} is not a valid task spec:\n" + "\n".join(lines)


def load_spec(path: Path) -> TaskSpec:
    """Read one TOML spec. Raises :class:`SpecError` with a plain-language message."""
    document = _read_toml(path, what="spec")
    try:
        return TaskSpec.model_validate(document)
    except ValidationError as exc:
        raise SpecError(_plain_validation_error(path, exc)) from exc


def _entry(
    entry: Any,
    where: str,
    path: Path,
) -> tuple[str, str] | str:
    """One ``[[groups]]`` entry as ``(id, description)``, or a problem sentence."""
    if not isinstance(entry, dict):
        return f"{path}: {where} must be a table with id and description"
    label_id = entry.get("id")
    description = entry.get("description")
    if not isinstance(label_id, str) or not label_id.strip():
        return f"{path}: {where} has no id; every label needs a short id"
    if not isinstance(description, str) or not description.strip():
        return f"{path}: label {label_id!r} has no one-line description"
    if label_id == UNCLEAR:
        return (
            f"{path}: the label id 'unclear' is reserved for abstentions; "
            "choose another id for this label"
        )
    return label_id, description


def _label_entries(path: Path, document: Mapping[str, Any]) -> dict[str, str]:
    """Label id -> description from the shapes a ``groups.toml`` can reasonably use."""
    entries: list[tuple[str, str]] = []
    problems: list[str] = []
    found = False
    for key in _LABEL_CONTAINERS:
        value = document.get(key)
        if value is None:
            continue
        found = True
        if isinstance(value, list):
            for index, entry in enumerate(value):
                parsed = _entry(entry, f"{key}[{index}]", path)
                if isinstance(parsed, str):
                    problems.append(parsed)
                else:
                    entries.append(parsed)
        elif isinstance(value, dict):
            for label_id, body in value.items():
                if isinstance(body, str) and body.strip():
                    entries.append((label_id, body))
                elif isinstance(body, dict):
                    description = body.get("description")
                    if isinstance(description, str) and description.strip():
                        entries.append((label_id, description))
                    else:
                        problems.append(f"{path}: label {label_id!r} has no one-line description")
                else:
                    problems.append(f"{path}: label {label_id!r} needs a one-line description")
        else:
            problems.append(f"{path}: [{key}] must be a table or a list of tables")

    if not found:
        problems.append(
            f"{path} lists no labels: add [[groups]] entries with an id and a description "
            '(or a [labels] table of id = "description")'
        )
    seen: dict[str, str] = {}
    for label_id, description in entries:
        if label_id in seen:
            problems.append(f"{path}: label {label_id!r} appears more than once")
        seen[label_id] = description
    if problems:
        raise SpecError("\n".join(problems))
    return seen


def _file_labels(path: Path) -> dict[str, str]:
    return _label_entries(path, _read_toml(path, what="labels"))


def load_labels(spec: TaskSpec, *, root: Path = Path()) -> dict[str, str]:
    """Label id -> one-line description, from every ``labels_file`` in the spec.

    The prompt template has one ``{labels}`` placeholder, so two label outputs share
    the union of their label lists (the usual spec has one label output).
    """
    labels: dict[str, str] = {}
    for output in spec.outputs:
        if output.kind is not OutputKind.LABEL or output.labels_file is None:
            continue
        for label_id, description in _file_labels(root / output.labels_file).items():
            previous = labels.get(label_id)
            if previous is not None and previous != description:
                raise SpecError(
                    f"The label {label_id!r} is described in two ways: {previous!r} and "
                    f"{description!r}. Use one description for one id."
                )
            labels[label_id] = description
    return labels


def _labels_by_output(spec: TaskSpec, root: Path) -> dict[str, dict[str, str]]:
    by_output: dict[str, dict[str, str]] = {}
    for output in spec.outputs:
        if output.kind is not OutputKind.LABEL or output.labels_file is None:
            continue
        by_output[output.name] = _file_labels(root / output.labels_file)
    return by_output


def _check_template(path: Path) -> list[str]:
    if not path.is_file():
        return [f"The prompt template {path} is missing; create it or fix prompt.template_file."]
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return [f"Cannot read the prompt template {path}: {reason}"]
    problems: list[str] = []
    for required in _PLACEHOLDERS:
        if "{" + required + "}" not in template:
            problems.append(
                f"The prompt template {path} needs the placeholder {{{required}}}; "
                "the tool fills it before every call."
            )
    return problems


def _read_rows(path: Path) -> tuple[list[Any], list[str]]:
    """Each non-empty JSONL line, or a problem when the file is missing and unreadable."""
    if not path.is_file():
        return [], [
            f"The data file {path} is missing; put one JSON object per line there or fix data.file."
        ]
    rows: list[Any] = []
    problems: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    problems.append(f"{path} line {number} is not valid JSON: {exc}")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return [], [f"Cannot read the data file {path}: {reason}"]
    if not rows and not problems:
        problems.append(f"The data file {path} is empty; it needs at least one item.")
    return rows, problems


def _scalar_id(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def _validate_rows(spec: TaskSpec, rows: list[Any], root: Path) -> list[str]:
    problems: list[str] = []
    label_sets: dict[str, set[str]] = {}
    for output in spec.outputs:
        if output.kind is OutputKind.LABEL and output.labels_file is not None:
            try:
                label_sets[output.name] = set(_file_labels(root / output.labels_file))
            except SpecError as exc:
                problems.extend(str(exc).splitlines())

    seen_ids: dict[str, int] = {}
    bad_fields: dict[str, int] = {}
    bad_labels: dict[str, int] = {}
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            problems.append(
                f"Item {position} in {spec.data.file} is not a JSON object; "
                "each line must be one object."
            )
            continue
        item_id = _scalar_id(row.get(spec.data.id_field))
        if item_id is None:
            problems.append(
                f"Item {position} in {spec.data.file} has no {spec.data.id_field!r} value; "
                "every item needs a unique id."
            )
        elif item_id in seen_ids:
            problems.append(
                f"Items {seen_ids[item_id]} and {position} in {spec.data.file} share the id "
                f"{item_id!r}; ids must be unique."
            )
        else:
            seen_ids[item_id] = position

        text = row.get(spec.data.text_field)
        if not isinstance(text, str) or not text.strip():
            bad_fields[spec.data.text_field] = bad_fields.get(spec.data.text_field, 0) + 1

        for output in spec.outputs:
            truth = row.get(output.truth_field)
            if output.kind is OutputKind.LABEL:
                known = label_sets.get(output.name)
                if not isinstance(truth, str):
                    bad_fields[output.truth_field] = bad_fields.get(output.truth_field, 0) + 1
                elif known is not None and truth not in known:
                    bad_labels[output.name] = bad_labels.get(output.name, 0) + 1
            elif not isinstance(truth, bool):
                bad_fields[output.truth_field] = bad_fields.get(output.truth_field, 0) + 1

    for field, count in bad_fields.items():
        problems.append(
            f"{count} item(s) in {spec.data.file} have no usable {field!r} value "
            "or it has the wrong type (a label is text, a flag is true or false)."
        )
    for name, count in bad_labels.items():
        output = next(output for output in spec.outputs if output.name == name)
        problems.append(
            f"{count} item(s) in {spec.data.file} use a {output.truth_field!r} value that is "
            f"not in {output.labels_file}; every truth label must be one of the known ids."
        )
    return problems


def validate_spec(spec: TaskSpec, *, root: Path = Path()) -> list[str]:
    """Plain-language problems with the spec and its files; empty means valid."""
    problems: list[str] = []
    problems.extend(_check_template(root / spec.prompt.template_file))

    label_ids: set[str] = set()
    for output in spec.outputs:
        if output.kind is not OutputKind.LABEL or output.labels_file is None:
            continue
        try:
            label_ids.update(_file_labels(root / output.labels_file))
        except SpecError as exc:
            problems.extend(str(exc).splitlines())

    rows, row_problems = _read_rows(root / spec.data.file)
    problems.extend(row_problems)
    if rows:
        problems.extend(_validate_rows(spec, rows, root))

    for name, baseline in spec.baseline.items():
        for field in ("train_file", "keywords_file"):
            value = getattr(baseline, field)
            if value is not None and not (root / value).is_file():
                problems.append(
                    f"The baseline for output {name!r} points at {value}, which is missing."
                )

    return list(dict.fromkeys(problems))


def render_prompt(
    spec: TaskSpec,
    labels: Mapping[str, str],
    text: str,
    *,
    root: Path = Path(),
) -> str:
    """The template with ``{labels}`` (one ``id: description`` line each) and ``{text}``."""
    path = root / spec.prompt.template_file
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise SpecError(f"Cannot read the prompt template {path}: {reason}") from exc
    block = (
        "\n".join(f"- {label_id}: {description}" for label_id, description in labels.items())
        or "- (no labels)"
    )
    # Plain replacement, not ``str.format``: a prompt almost always contains JSON
    # braces, and a spec author should not have to double them.
    return template.replace("{labels}", block).replace("{text}", text)


def _model_name(task_id: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", task_id) if part]
    return "".join(part.capitalize() for part in parts) + "Out"


def _model_from_labels(
    spec: TaskSpec,
    labels_by_output: Mapping[str, Mapping[str, str]],
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for output in spec.outputs:
        if output.kind is OutputKind.LABEL:
            ids = tuple(labels_by_output[output.name])
            values = (*ids, UNCLEAR) if output.allow_unclear else ids
            fields[output.name] = (Literal[values], ...)
        else:
            fields[output.name] = (bool, ...)
    return cast(
        "type[BaseModel]",
        create_model(_model_name(spec.id), __config__=ConfigDict(extra="forbid"), **fields),
    )


def build_output_model(spec: TaskSpec, *, root: Path = Path()) -> type[BaseModel]:
    """The pydantic output model: labels become ``Literal`` ids, flags become ``bool``.

    ``root`` is where the spec's files live (the repository root by default); it was
    added as an optional keyword so the predictor can build the same model from any
    working directory.
    """
    return _model_from_labels(spec, _labels_by_output(spec, root))


_SKELETON_SPEC = """\
# A task spec. `wam task check tasks/{id}.toml` tells you what is missing.
# One TOML file plus a labelled data file is the whole task; no Python needed.
id = "{id}"
title = "One line: what the task is called"
description = "One sentence: the job a person would hand over."
prompt_version = "{short_version}"
scorer_version = "s1"

[data]
# One JSON object per line. Every item needs a unique id and some text.
file = "data/{id}/test.jsonl"
id_field = "id"
text_field = "text"

[prompt]
# Placeholders: {{labels}} becomes the list below and {{text}} the item's text.
template_file = "tasks/{id}.prompt.txt"

[[outputs]]
# The field the model must fill in. Change the name, the labels and the truth field.
name = "label"
kind = "label"
labels_file = "data/{id}/labels.toml"
# Set to true to let the model answer "unclear"; it then counts as wrong.
allow_unclear = false
truth_field = "label"

# A second, yes/no output looks like this (delete the # signs to use it):
# [[outputs]]
# name = "flag"
# kind = "bool"
# truth_field = "flag"

[metrics]
primary = "label.macro_f1"   # unclear counts as wrong
report = [
  "label.accuracy",
  "label.coverage",
  "label.accuracy_answered",
  "schema_valid_rate",
]

[gate]
# A new run fails when the primary metric drops by more than this many points.
primary_drop_points = 2.0
# A run below this share of schema-valid answers fails.
schema_valid_floor = 0.98

# Optional: a baseline that costs nothing. `keywords_file` or `train_file` must exist.
# [baseline.label]
# kind = "keyword"
# keywords_file = "data/{id}/keywords.txt"

# Optional: a floor a run may never fall below, even when the primary metric passes.
# [[gate.floors]]
# metric = "flag.recall"
# min = 0.90
# max_drop_points = 2.0
"""

_SKELETON_PROMPT = """\
You sort each item into exactly one category.

Categories (use the id exactly as written, including hyphens):
{labels}

Item:
{text}

Reply with one JSON object and nothing else: {{"label": "<one category id>"}}
"""


def write_skeleton(task_id: str, *, tasks_dir: Path = Path("tasks")) -> tuple[Path, Path]:
    """Write ``<tasks_dir>/<id>.toml`` and ``<tasks_dir>/<id>.prompt.txt``.

    A commented skeleton with the placeholders already in the right places. It still
    needs the data and labels files that its comments name.
    """
    if re.fullmatch(SPEC_ID_PATTERN, task_id) is None:
        raise SpecError(
            f"{task_id!r} is not a usable task id. Use lower-case letters and digits, "
            "with single hyphens between words (for example 'defect' or 'my-task')."
        )
    tasks_dir.mkdir(parents=True, exist_ok=True)
    spec_path = tasks_dir / f"{task_id}.toml"
    prompt_path = tasks_dir / f"{task_id}.prompt.txt"
    for path in (spec_path, prompt_path):
        if path.exists():
            raise SpecError(f"{path} already exists. Choose another id, or edit that file instead.")
    short_version = task_id[:1] + "1"
    spec_path.write_text(
        _SKELETON_SPEC.format(id=task_id, short_version=short_version), encoding="utf-8"
    )
    prompt_path.write_text(_SKELETON_PROMPT, encoding="utf-8")
    return spec_path, prompt_path
