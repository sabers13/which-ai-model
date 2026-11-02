"""Spec tasks (``PLAN-v1.1`` §A3): a task is one TOML file plus a labelled data file.

The loader and validator read the spec and its files and report problems in plain
language; the predictor renders the one prompt every model receives and parses the
structured answer; the scorer turns one model x task cell into metrics and CIs.
"""

from __future__ import annotations

from which_ai_model.task.predict import SpecPrediction, SpecPredictor
from which_ai_model.task.score import score_spec
from which_ai_model.task.spec import (
    SpecError,
    build_output_model,
    load_labels,
    load_spec,
    render_prompt,
    validate_spec,
    write_skeleton,
)

__all__ = [
    "SpecError",
    "SpecPrediction",
    "SpecPredictor",
    "build_output_model",
    "load_labels",
    "load_spec",
    "render_prompt",
    "score_spec",
    "validate_spec",
    "write_skeleton",
]
