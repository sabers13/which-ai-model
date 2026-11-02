"""Prompt-library export for a use-case card (``PLAN-v1.1`` §A9).

One card becomes one ready-to-use entry: the task and when to use it, the recommended
model and a fallback, the exact prompt with its version, the output schema, the
guardrails and the re-check trigger.

The recommended model is the cheapest *good-enough* model under the card's picker rule
(§A4: within 3.0 points of the best primary metric, never below a gate floor); the
fallback is the next good-enough model by cost. Unknown costs sort last, as they do in
the gate, and ties fall back to the better primary metric and then the model id, so the
export is deterministic.

The spec prompt and output schema are read from the repository root, because a
``TaskSpec`` stores file paths relative to it. No model is called and nothing is
recomputed from a live result.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from which_ai_model.bench.prompts import prompt_for
from which_ai_model.contracts import (
    CardData,
    CardModelRow,
    PromptLibraryEntry,
    TaskName,
    TaskSpec,
)
from which_ai_model.gate.spec import pick
from which_ai_model.task.spec import build_output_model

__all__ = ["export_prompt_library"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _Cell:
    """A card row reshaped for the gate picker, which only reads these fields."""

    model_id: str
    primary_metric: float
    metrics: Mapping[str, float]
    cost_per_1k: float | None
    latency_p50: float | None


def export_prompt_library(
    card_data: CardData, spec_or_builtin: TaskSpec | TaskName
) -> tuple[str, str]:
    """``(markdown, json_text)`` for ``results/prompt-library/<task>.{md,json}``."""
    if isinstance(spec_or_builtin, TaskSpec):
        task = spec_or_builtin.id
        prompt = _spec_prompt(spec_or_builtin.prompt.template_file)
        prompt_version = spec_or_builtin.prompt_version
        schema = build_output_model(spec_or_builtin, root=_REPO_ROOT).model_json_schema()
    else:
        builtin = TaskName(spec_or_builtin)
        task = builtin.value
        builtin_prompt = prompt_for(builtin)
        prompt = f"{builtin_prompt.system}\n\n{builtin_prompt.user_template}"
        prompt_version = card_data.prompt_version
        schema = builtin_prompt.schema.model_json_schema()

    if task != card_data.task:
        raise ValueError(
            f"card task {card_data.task!r} does not match the prompt for {task!r}; "
            "export the library entry for the task the card describes"
        )

    recommended, fallback, no_floor = _pick_models(card_data)
    entry = PromptLibraryEntry(
        task=card_data.task,
        title=card_data.title,
        when_to_use=card_data.job,
        prompt=prompt,
        prompt_version=prompt_version,
        output_schema=schema,
        recommended_model=recommended,
        fallback_model=fallback,
        guardrails=_guardrails(card_data),
        recheck_trigger=card_data.recheck,
    )
    return _markdown(entry, card_data, no_floor=no_floor), _json_text(entry)


def _spec_prompt(template_file: str) -> str:
    """The exact committed prompt template, resolved from the repository root."""
    path = Path(template_file)
    if not path.is_absolute() and (_REPO_ROOT / path).is_file():
        path = _REPO_ROOT / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise FileNotFoundError(
            f"The prompt template {path} is missing; create it or fix prompt.template_file "
            f"({reason})."
        ) from exc


def _cost_key(row: CardModelRow) -> tuple[bool, float, float, str]:
    """Unknown cost last, then cheaper, then more accurate, then the model id."""
    return (
        row.cost_per_1k is None,
        row.cost_per_1k if row.cost_per_1k is not None else 0.0,
        -row.primary_metric,
        row.model_id,
    )


def _good_enough(card_data: CardData) -> list[CardModelRow]:
    """The card's models that are good enough, ordered by the cost rule above."""
    if card_data.no_model_meets_floor:
        return []
    cells = [
        _Cell(
            model_id=row.model_id,
            primary_metric=row.primary_metric,
            metrics=row.metrics,
            cost_per_1k=row.cost_per_1k,
            latency_p50=row.latency_p50,
        )
        for row in card_data.models
    ]
    picked = pick(cells, floors=card_data.gate.floors)
    if picked.no_model_meets_floor:
        return []
    good = set(picked.good_enough)
    rows = [row for row in card_data.models if row.model_id in good]
    return sorted(rows, key=_cost_key)


def _recommended_and_fallback(
    ordered: list[CardModelRow],
) -> tuple[str | None, str | None]:
    if not ordered:
        return None, None
    recommended = ordered[0].model_id
    fallback = ordered[1].model_id if len(ordered) > 1 else None
    return recommended, fallback


def _pick_models(card_data: CardData) -> tuple[str | None, str | None, bool]:
    """The two model ids and whether the card has no model above its gate floors."""
    ordered = _good_enough(card_data)
    recommended, fallback = _recommended_and_fallback(ordered)
    no_floor = card_data.no_model_meets_floor or (
        bool(card_data.models) and bool(card_data.gate.floors) and not ordered
    )
    return recommended, fallback, no_floor


def _guardrails(card_data: CardData) -> list[str]:
    """The §A9 guardrails, with the card's own gate floors spelled out."""
    lines = [
        "An `unclear` or abstained answer goes to a human; it is scored as wrong, "
        "so abstaining is never free.",
    ]
    for floor in card_data.gate.floors:
        if floor.min is not None:
            text = f"Gate floor: `{floor.metric}` must stay at or above {floor.min:.2f}"
        else:
            text = f"Gate floor: `{floor.metric}`"
        if floor.max_drop_points is not None:
            text += (
                f" and may not drop by more than {floor.max_drop_points:g} points "
                "against the baseline"
            )
        lines.append(text + ".")
    lines.append(
        f"The schema-valid rate must stay at or above {card_data.gate.schema_valid_floor:.0%}."
    )
    lines.append(
        "Never auto-close, auto-send or auto-approve a model answer; a person makes the final call."
    )
    return lines


def _usd(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _cost_text(row: CardModelRow | None) -> str:
    if row is None or row.cost_per_1k is None:
        return "cost per 1,000 items: unknown"
    return f"{_usd(row.cost_per_1k)} per 1,000 items"


def _fence(text: str) -> str:
    """A backtick fence longer than any run inside ``text``."""
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _markdown(entry: PromptLibraryEntry, card_data: CardData, *, no_floor: bool) -> str:
    rows = {row.model_id: row for row in card_data.models}
    lines = [
        f"# {entry.title}",
        "",
        f"**Task:** `{entry.task}`",
        "",
        "## Task and when to use",
        "",
        entry.when_to_use,
        "",
        "## Recommended model",
        "",
    ]
    if entry.recommended_model is None:
        if no_floor:
            lines.append(
                "No model meets this task's gate floors yet. Keep the current baseline "
                "and a human check until one does."
            )
        else:
            lines.append("No model result is committed for this task yet.")
    else:
        lines.append(
            f"- **Recommended:** `{entry.recommended_model}` — "
            f"{_cost_text(rows.get(entry.recommended_model))}."
        )
        if entry.fallback_model is not None:
            lines.append(
                f"- **Fallback:** `{entry.fallback_model}` — "
                f"{_cost_text(rows.get(entry.fallback_model))}."
            )
        else:
            lines.append(
                "- **Fallback:** none yet; only one model is good enough at this sample size."
            )
    rule = card_data.picker_rule.rstrip(".")
    lines += [
        "",
        f"The rule that chose them: {rule}.",
        "",
        f"## Prompt (version `{entry.prompt_version}`)",
        "",
    ]
    fence = _fence(entry.prompt)
    lines += [fence, entry.prompt.rstrip("\n"), fence, ""]
    lines += [
        "## Output schema",
        "",
        "```json",
        json.dumps(entry.output_schema, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Guardrails",
        "",
    ]
    lines += [f"- {guardrail}" for guardrail in entry.guardrails]
    lines += [
        "",
        "## Re-check",
        "",
        entry.recheck_trigger,
        "",
    ]
    return "\n".join(lines)


def _json_text(entry: PromptLibraryEntry) -> str:
    payload: dict[str, Any] = entry.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
