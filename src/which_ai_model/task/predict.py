"""The generic predictor for a spec task: render, call, parse (``PLAN-v1.1`` §A3).

This is the same path as the built-in predictors: the runner owns record, replay,
pacing and cost and hands the predictor a ``call`` closure; the predictor builds the
exact messages, calls it, and parses the answer against the schema its spec defines.
No model is named here and no prompt differs between models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from which_ai_model.bench.run import Call, Message, parse_response
from which_ai_model.contracts import ModelRef, TaskSpec
from which_ai_model.task.spec import (
    SpecError,
    build_output_model,
    load_labels,
    render_prompt,
)

__all__ = ["SpecPrediction", "SpecPredictor"]


@dataclass(frozen=True)
class SpecPrediction:
    """One spec answer. ``Prediction`` is unchanged and stays built-in-only."""

    item_id: str
    task: str
    model: ModelRef
    output: dict[str, Any]
    raw_key: str


class SpecPredictor:
    """One spec task, served to the runner beside the built-in predictors."""

    task: str
    prompt_version: str

    def __init__(self, spec: TaskSpec, *, root: Path = Path()) -> None:
        self.spec = spec
        self.task = spec.id
        self.prompt_version = spec.prompt_version
        self._root = root
        self._labels = load_labels(spec, root=root)
        self._template_path = root / spec.prompt.template_file
        if not self._template_path.is_file():
            raise SpecError(
                f"The prompt template {self._template_path} is missing; "
                "create it or fix prompt.template_file."
            )
        self._schema: type[BaseModel] = build_output_model(spec, root=root)

    @property
    def schema(self) -> type[BaseModel]:
        """The output model built from this spec's outputs."""
        return self._schema

    def messages(self, item: Mapping[str, Any]) -> list[Message]:
        """The exact messages for one item. No model input, so every model matches."""
        text = item.get(self.spec.data.text_field)
        if not isinstance(text, str) or not text.strip():
            raise SpecError(
                f"Item {item.get(self.spec.data.id_field)!r} has no "
                f"{self.spec.data.text_field!r} value to send."
            )
        prompt = render_prompt(self.spec, self._labels, text, root=self._root)
        return [{"role": "user", "content": prompt}]

    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> SpecPrediction:
        """Call the model through the runner and parse its structured answer.

        An empty, cut-off or unparseable answer becomes ``{}``: the scorer counts it as
        schema-invalid and wrong, and the trace records the runner's error. It is never
        retried.
        """
        item_id = item.get("item_id", item.get(self.spec.data.id_field))
        if item_id is None or str(item_id).strip() == "":
            raise SpecError(
                f"Item in {self.spec.data.file} has no {self.spec.data.id_field!r} value to score."
            )
        answer = call(self.messages(item))
        output, _valid = parse_response(self._schema, answer.response)
        return SpecPrediction(
            item_id=str(item_id),
            task=self.spec.id,
            model=model,
            output=output,
            raw_key=answer.key,
        )
