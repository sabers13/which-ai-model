"""Build ``results/page/page-data.json`` from committed evidence.

The page never reads a result at render time. This module copies the values it needs
from committed files, keeps the source path next to every measurement, and adds the
small amount of curated text that the page shows as prose. Nothing here touches the
network, the clock or a random seed, so the same checkout always produces the same
bytes.

Source rules used by the tests:

* every measured value names a file under ``results/`` that exists in the repository;
* curated content (the example message, code excerpts and line counts) names the file
  it came from, and the tests check that the file exists and that the line counts
  match the committed source inventory.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from which_ai_model.gate import GATE_PRIMARY_DROP_POINTS, GATE_SCHEMA_VALID_FLOOR
from which_ai_model.rag.corpus import load_corpus

__all__ = [
    "BUILTIN_TASK_ORDER",
    "PAGE_DATA_PATH",
    "TASK_ORDER",
    "build_page_data",
    "render_json",
    "write_page_data",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENT_PATH = Path("web/content.json")
PAGE_DATA_PATH = Path("results/page/page-data.json")

#: Every task the project has evidence for, in page order. The four built-in tasks keep
#: their v1 sections; the spec task is rendered by the v1.1 section only.
TASK_ORDER = ("T-ROUTE", "T-SCOPE", "T-REPLY", "T-DOCS", "T-DEFECT")

#: The four v1 tasks, in the order their v1 sections render them.
BUILTIN_TASK_ORDER = TASK_ORDER[:4]

TASK_META: dict[str, dict[str, str]] = {
    "T-ROUTE": {
        "name": "Route a message",
        "short": "Routing",
        "question": "Which of 77 banking intents is this message?",
        "metric_label": "macro-F1",
    },
    "T-SCOPE": {
        "name": "Check the scope",
        "short": "Scope",
        "question": "Is this message in scope for the support team, or out of scope?",
        "metric_label": "out-of-scope F1",
    },
    "T-REPLY": {
        "name": "Draft a structured reply",
        "short": "Replies",
        "question": "Return a valid intent, urgency and reply as one typed object.",
        "metric_label": "intent-correct",
    },
    "T-DOCS": {
        "name": "Answer from the guide",
        "short": "Docs",
        "question": "Answer a question about this stack and cite the passages used.",
        "metric_label": "retrieval hit@4",
    },
    "T-DEFECT": {
        "name": "Sort electrical complaints",
        "short": "Complaints",
        "question": "Sort a vehicle complaint into a component group and flag the safety-relevant ones.",
        "metric_label": "group macro-F1",
    },
}

#: The task cards in the v1.1 section's "five cards" list, with their choice labels.
#: The card files keep the spec id (``defect``) for a task the page calls ``T-DEFECT``.
CARD_TASKS = TASK_ORDER
CARD_FILES = {task: task for task in BUILTIN_TASK_ORDER} | {"T-DEFECT": "defect"}
CARD_PICK_ORDER = ("cheapest", "most_accurate", "fastest")

PRIORITIES = ("cheapest", "most_accurate", "fastest")

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _read_json(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _read_jsonl(root: Path, relative: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (root / relative).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{relative}: expected a JSON object per line")
            rows.append(payload)
    return rows


def _metric(value: float | int | str, display: str, source: str) -> dict[str, Any]:
    return {"value": value, "display": display, "source": source}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _per_1k(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _money(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.001:
        return f"${value:.5f}"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _ms(value: float) -> str:
    return f"{value:,.0f} ms"


def _count(value: int) -> str:
    return f"{value:,}"


def _long_date(iso: str) -> str:
    parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"{parsed.day} {_MONTHS[parsed.month - 1]} {parsed.year}"


def _slug(model_id: str) -> str:
    return model_id.replace(".", "-")


def _load_models(root: Path) -> dict[str, dict[str, Any]]:
    payload = tomllib.loads((root / "config/models.lock.toml").read_text(encoding="utf-8"))
    models: dict[str, dict[str, Any]] = {}
    for entry in payload.get("model", []):
        models[str(entry["model_id"])] = entry
    return models


def _excerpt(text: str, limit: int = 520) -> str:
    """Plain-text excerpt: drop Markdown heading and backtick marks, keep the words."""
    lines = [re.sub(r"^#+\s*", "", line) for line in text.splitlines()]
    collapsed = re.sub(r"\s+", " ", "\n".join(lines)).strip().replace("`", "")
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("; "))
    if boundary > limit // 2:
        return cut[: boundary + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def _example(
    root: Path, content: dict[str, Any], models: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    spec = content["example"]
    item_id = str(spec["item_id"])
    task = str(spec["task"])
    truth = str(spec["true_label"])
    source = str(spec["sample_source"])

    calls: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(root, "results/runs/live/calls.jsonl"):
        if row.get("task") == task and row.get("item_id") == item_id:
            model_id = str(row["model"]["model_id"])
            calls[model_id] = row

    rows: list[dict[str, Any]] = []
    for model_id in sorted(calls):
        prediction_file = f"results/runs/live/predictions/{task}-{_slug(model_id)}.jsonl"
        answers = {str(row["item_id"]): row for row in _read_jsonl(root, prediction_file)}
        output = answers.get(item_id, {}).get("output", {})
        answer = str(output.get("intent", "")) if isinstance(output, dict) else ""
        call = calls[model_id]
        rows.append(
            {
                "model": model_id,
                "name": str(models[model_id]["display_name"]),
                "answer": answer or "no valid answer",
                "answer_display": answer.replace("_", " ") if answer else "no valid answer",
                "correct": answer == truth,
                "latency": _metric(
                    float(call["latency_ms"]),
                    _ms(float(call["latency_ms"])),
                    "results/runs/live/calls.jsonl",
                ),
                "cost": _metric(
                    float(call["cost_usd"]),
                    _money(float(call["cost_usd"])),
                    "results/runs/live/calls.jsonl",
                ),
                "schema_valid": bool(call["schema_valid"]),
            }
        )

    correct = sum(1 for row in rows if row["correct"])
    return {
        "item_id": item_id,
        "task": task,
        "task_name": TASK_META[task]["name"],
        "text": spec["text"],
        "true_label": truth,
        "true_label_display": truth.replace("_", " "),
        "dataset": spec["dataset"],
        "dataset_url": spec["dataset_url"],
        "source": source,
        "rows": rows,
        "correct": correct,
        "total": len(rows),
    }


def _tasks(
    root: Path,
    models: dict[str, dict[str, Any]],
    recommendations: dict[str, Any],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_task_model = {(str(cell["task"]), str(cell["model"]["model_id"])): cell for cell in cells}
    tasks: list[dict[str, Any]] = []
    for task in BUILTIN_TASK_ORDER:
        rec = recommendations["tasks"][task]
        picks: dict[str, Any] = {}
        for priority in PRIORITIES:
            pick = rec[priority]
            model_id = str(pick["model"])
            picks[priority] = {
                "model": model_id,
                "name": str(models[model_id]["display_name"]),
                "metric": _metric(
                    float(pick["primary_metric"]),
                    _pct(float(pick["primary_metric"])),
                    "results/recommendations.json",
                ),
                "cost": _metric(
                    float(pick["cost_per_1k"]),
                    _per_1k(float(pick["cost_per_1k"])),
                    "results/recommendations.json",
                ),
                "latency": _metric(
                    float(pick["latency_p50"]),
                    _ms(float(pick["latency_p50"])),
                    "results/recommendations.json",
                ),
                "good_enough": len(rec["good_enough"]),
            }

        rows: list[dict[str, Any]] = []
        for model_id in sorted(models):
            key = (task, model_id)
            if key not in by_task_model:
                continue
            cell = by_task_model[key]
            rows.append(
                {
                    "model": model_id,
                    "name": str(models[model_id]["display_name"]),
                    "metric": _metric(
                        float(cell["primary_metric"]),
                        _pct(float(cell["primary_metric"])),
                        "results/runs/live/cells.json",
                    ),
                    "cost": _metric(
                        float(cell["cost_per_1k"]),
                        _per_1k(float(cell["cost_per_1k"])),
                        "results/runs/live/cells.json",
                    ),
                    "latency": _metric(
                        float(cell["latency_p50"]),
                        _ms(float(cell["latency_p50"])),
                        "results/runs/live/cells.json",
                    ),
                }
            )
        rows.sort(key=lambda row: float(row["metric"]["value"]), reverse=True)

        tasks.append(
            {
                "id": task,
                **TASK_META[task],
                "picks": picks,
                "good_enough": _metric(
                    float(recommendations["good_enough_points"]),
                    f"{recommendations['good_enough_points']:g}",
                    "results/recommendations.json",
                ),
                "best": _metric(
                    float(rec["best_primary"]),
                    _pct(float(rec["best_primary"])),
                    "results/recommendations.json",
                ),
                "rows": rows,
            }
        )
    return tasks


def _docs(
    root: Path,
    content: dict[str, Any],
    models: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold = {str(row["id"]): row for row in _read_jsonl(root, "data/rag/gold.jsonl")}
    chunks = {chunk.id: chunk for chunk in load_corpus(root / "docs/guide")}
    by_task_model = {(str(cell["task"]), str(cell["model"]["model_id"])): cell for cell in cells}
    examples: list[dict[str, Any]] = []
    for spec in content["docs_examples"]:
        item_id = str(spec["item_id"])
        model_id = str(spec["model"])
        prediction_file = f"results/runs/live/predictions/T-DOCS-{_slug(model_id)}.jsonl"
        rows = {str(row["item_id"]): row for row in _read_jsonl(root, prediction_file)}
        prediction = rows[item_id]["output"]
        cited = [str(value) for value in prediction.get("cited_ids", [])]
        passages = []
        for chunk_id in cited:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            passages.append(
                {
                    "id": chunk_id,
                    "label": chunk_id.replace("guide/", "").replace("#", " · passage "),
                    "text": _excerpt(chunk.text),
                    "source": f"docs/guide/{Path(chunk.source).name}",
                }
            )
        cell = by_task_model[("T-DOCS", model_id)]
        examples.append(
            {
                "item_id": item_id,
                "question": str(gold[item_id]["question"]),
                "answer": str(prediction.get("answer", "")),
                "model": model_id,
                "model_name": str(models[model_id]["display_name"]),
                "passages": passages,
                "hit_at_4": _metric(
                    float(cell["metrics"]["hit_at_4"]),
                    _pct(float(cell["metrics"]["hit_at_4"])),
                    "results/runs/live/cells.json",
                ),
                "answer_source": prediction_file,
                "question_source": "data/rag/gold.jsonl",
            }
        )
    return examples


def _inventory(root: Path) -> dict[str, Any]:
    before = _read_json(root, "results/inventory/before.json")
    after = _read_json(root, "results/inventory/after.json")

    def rows(payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "path": str(site["path"]),
                "tool": str(site["tool"]),
                "provider": str(site["provider"]),
                "call_style": str(site["call_style"]),
                "structured_output": str(site["structured_output"]),
            }
            for site in payload["sites"]
        ]

    return {
        "before": rows(before),
        "after": rows(after),
        "before_count": _metric(
            int(before["summary"]["sites"]),
            str(before["summary"]["sites"]),
            "results/inventory/before.json",
        ),
        "after_count": _metric(
            int(after["summary"]["sites"]),
            str(after["summary"]["sites"]),
            "results/inventory/after.json",
        ),
        "before_source": "results/inventory/before.json",
        "after_source": "results/inventory/after.json",
    }


def _gate(root: Path, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    demo = _read_json(root, "results/gate_demo.json")
    refused = _read_json(root, "results/gate_refused_demo.json")
    pairs = []
    for pair in demo["pairs"]:
        failures = []
        comparisons = {str(item["task"]): item for item in pair["comparisons"]}
        for comparison in pair["comparisons"]:
            for failure in comparison["failures"]:
                failures.append(str(failure))
        candidate = str(pair["candidate_model"])
        route = comparisons.get("T-ROUTE", {})
        reply = comparisons.get("T-REPLY", {})
        route_drop = max(float(route.get("primary_delta", 0.0)), 0.0)
        reply_drop = max(float(reply.get("primary_delta", 0.0)), 0.0)
        schema = reply.get("schema_valid_candidate")
        pairs.append(
            {
                "baseline": str(pair["baseline_model"]),
                "baseline_name": str(models[str(pair["baseline_model"])]["display_name"]),
                "candidate": candidate,
                "candidate_name": str(models[candidate]["display_name"]),
                "outcome": str(pair["outcome"]),
                "reasons": failures,
                "tasks": [str(value) for value in pair["chosen_for_tasks"]],
                "route_drop": _metric(
                    round(route_drop * 100, 2),
                    f"{route_drop * 100:.2f}",
                    "results/gate_demo.json",
                ),
                "reply_drop": _metric(
                    round(reply_drop * 100, 2),
                    f"{reply_drop * 100:.2f}",
                    "results/gate_demo.json",
                ),
                "schema_valid": _metric(
                    float(schema) if schema is not None else 0.0,
                    _pct(float(schema)) if schema is not None else "n/a",
                    "results/gate_demo.json",
                ),
            }
        )
    return {
        "pairs": pairs,
        "blocked_count": _metric(
            sum(1 for pair in pairs if pair["outcome"] == "FAIL"),
            str(len(pairs)),
            "results/gate_demo.json",
        ),
        "limit": _metric(
            GATE_PRIMARY_DROP_POINTS,
            f"{GATE_PRIMARY_DROP_POINTS:.1f}",
            "results/gate_demo.json",
        ),
        "schema_floor": _metric(
            GATE_SCHEMA_VALID_FLOOR,
            _pct(GATE_SCHEMA_VALID_FLOOR),
            "results/gate_demo.json",
        ),
        "source": "results/gate_demo.json",
        "refusal": {
            "outcome": str(refused["gate"]["outcome"]),
            "reason": str(refused["gate"]["reasons"][0]),
            "source": "results/gate_refused_demo.json",
        },
    }


def _before_after(root: Path, content: dict[str, Any]) -> dict[str, Any]:
    def excerpt(spec: dict[str, Any]) -> dict[str, Any]:
        path = root / str(spec["file"])
        lines = path.read_text(encoding="utf-8").splitlines()
        start = int(spec["start"])
        end = int(spec["end"])
        return {
            "file": str(spec["file"]),
            "language": str(spec["language"]),
            "start": start,
            "end": end,
            "text": "\n".join(lines[start - 1 : end]),
        }

    line_counts = []
    for row in content["line_counts"]:
        line_counts.append(
            {
                "tool": str(row["tool"]),
                "before_file": str(row["before_file"]),
                "after_file": str(row["after_file"]),
                "before": _metric(int(row["before"]), str(row["before"]), str(row["source"])),
                "after": _metric(int(row["after"]), str(row["after"]), str(row["source"])),
                "source": str(row["source"]),
            }
        )
    return {
        "before": excerpt(content["code"]["before"]),
        "after": excerpt(content["code"]["after"]),
        "line_counts": line_counts,
        "max_lines": max(
            max(int(row["before"]), int(row["after"])) for row in content["line_counts"]
        ),
    }


def _ci_display(interval: list[float]) -> str:
    low, high = (float(value) for value in interval)
    return f"{low:.4f} to {high:.4f}"


def _v11(
    root: Path,
    content: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The v1.1 section: the T-DEFECT card, re-check, volume and the five cards.

    Every number is copied from the committed card, the committed run or the committed
    re-check and gate reports. The browser calculator does the same linear arithmetic as
    the card, so the page and the card cannot disagree.
    """
    spec = content["v11"]
    card_source = "results/cards/defect.json"
    cell_source = "results/runs/live-v1.1/cells.json"
    baseline_source = "results/baseline/defect.json"
    gate_source = "results/gate_demo_defect.json"
    refused_source = "results/runs/recheck-demo/recheck.json"
    accepted_source = "results/runs/recheck-demo-accepted/recheck.json"

    card = _read_json(root, card_source)
    spec_cells = _read_json(root, cell_source)["spec_cells"]
    by_model = {str(cell["model"]["model_id"]): cell for cell in spec_cells}

    def name(model_id: str) -> str:
        entry = models.get(model_id)
        return str(entry["display_name"]) if entry else model_id

    def lane(
        payload: dict[str, Any],
        *,
        baseline: bool,
        source: str,
        items: int | None,
    ) -> dict[str, Any]:
        metrics = payload["metrics"]
        primary = float(payload["primary_metric"])
        latency = payload.get("latency_p50")
        coverage = float(metrics["component_group.coverage"])
        answered = float(metrics["component_group.accuracy_answered"])
        safety = float(metrics["safety_relevant.recall"])
        cost = float(payload["cost_per_1k"])
        return {
            "model": str(payload["model_id"]),
            "name": spec["baseline_label"] if baseline else name(str(payload["model_id"])),
            "baseline": baseline,
            "items": items,
            "primary": _metric(
                round(primary, 4),
                f"{primary:.4f} (95% CI {_ci_display(payload['primary_ci'])})",
                source,
            ),
            "coverage": _metric(round(coverage, 4), _pct(coverage), source),
            "answered": _metric(round(answered, 4), _pct(answered), source),
            "safety": _metric(round(safety, 4), _pct(safety), source),
            "cost": _metric(round(cost, 4), _per_1k(cost), source),
            "latency": {
                "value": None if latency is None else round(float(latency), 1),
                "display": "—" if latency is None else _ms(float(latency)),
                "source": source,
            },
        }

    rows = [
        lane(
            model,
            baseline=False,
            source=card_source,
            items=int(by_model[str(model["model_id"])]["n"]),
        )
        for model in card["models"]
    ]
    baseline = lane(
        card["baseline"],
        baseline=True,
        source=baseline_source,
        items=int(card["sample_size"]),
    )
    table = [*rows, baseline]

    floor_min = float(card["gate"]["floors"][0]["min"])
    floor_met = all(float(row["safety"]["value"]) >= floor_min for row in rows)
    baseline_met = float(baseline["safety"]["value"]) >= floor_min
    if floor_met:
        floor_sentence = (
            f"All {len(rows)} models cleared the {_pct(floor_min)} floor. "
            f"The keyword baseline reached {baseline['safety']['display']}, below it."
        )
    else:
        below = [str(row["name"]) for row in rows if float(row["safety"]["value"]) < floor_min]
        floor_sentence = (
            f"{len(below)} of {len(rows)} models fell below the {_pct(floor_min)} floor: "
            f"{', '.join(below)}. The keyword baseline reached {baseline['safety']['display']}."
        )

    coverages = [float(row["coverage"]["value"]) for row in rows]
    answered = [float(row["answered"]["value"]) for row in rows]
    coverage_sentence = (
        "Every model may answer unclear, and unclear counts as wrong in the group score. "
        f"Coverage ran from {_pct(min(coverages))} to {_pct(max(coverages))}; on the items "
        f"a model did answer, {_pct(min(answered))} to {_pct(max(answered))} were sorted correctly."
    )

    picks = []
    for key in CARD_PICK_ORDER:
        model_id = card.get(key)
        if not model_id:
            continue
        picks.append(
            {
                "key": key,
                "label": spec["cards_pick_labels"][key],
                "model": str(model_id),
                "name": name(str(model_id)),
            }
        )

    gate = _read_json(root, gate_source)
    rules = {str(rule["name"]): rule for rule in gate["rules"]}
    primary_drop = float(rules["primary_drop"]["value"])
    floor_value = float(rules["floor_min:safety_relevant.recall"]["value"])
    floor_drop = float(rules["floor_drop:safety_relevant.recall"]["value"])
    gate_demo = {
        "outcome": str(gate["outcome"]),
        "candidate": name(str(gate["candidate_model"])),
        "baseline": name(str(gate["baseline_model"])),
        "primary_drop": _metric(round(primary_drop, 2), f"{primary_drop:.2f}", gate_source),
        "primary_limit": _metric(
            float(card["gate"]["primary_drop_points"]),
            f"{float(card['gate']['primary_drop_points']):.1f}",
            card_source,
        ),
        "floor": _metric(round(floor_value, 4), _pct(floor_value), gate_source),
        "floor_drop": _metric(round(floor_drop, 2), f"{floor_drop:.2f}", gate_source),
        "floor_min": _metric(floor_min, _pct(floor_min), card_source),
        "source": gate_source,
    }

    refused = _read_json(root, refused_source)
    accepted = _read_json(root, accepted_source)
    accepted_pair = accepted["gate"]["pairs"][0]
    comparison = accepted_pair["comparisons"][0]
    baseline_primary = float(comparison["baseline_primary"])
    candidate_primary = float(comparison["candidate_primary"])
    demo = {
        "task": str(accepted["tasks"][0]),
        "candidate": name(str(accepted["settings"]["pairs"][0]["candidate_model"])),
        "refused": {
            "outcome": str(refused["gate"]["outcome"]),
            "differences": [str(item) for item in refused["settings"]["pairs"][0]["differences"]],
            "source": refused_source,
        },
        "accepted": {
            "outcome": str(accepted["gate"]["outcome"]),
            "warnings": [str(item) for item in accepted_pair["warnings"]],
            "warning_count": _metric(
                len(accepted_pair["warnings"]),
                str(len(accepted_pair["warnings"])),
                accepted_source,
            ),
            "baseline": _metric(
                round(baseline_primary, 4), _pct(baseline_primary), accepted_source
            ),
            "candidate": _metric(
                round(candidate_primary, 4), _pct(candidate_primary), accepted_source
            ),
            "source": accepted_source,
        },
    }

    volumes = card["volumes"]
    headers = [{"items": items, "display": f"{items:,}"} for items in (1_000, 10_000, 100_000)]
    by_volume: dict[str, list[dict[str, Any]]] = {}
    for entry in volumes:
        by_volume.setdefault(str(entry["model_id"]), []).append(entry)
    volume_rows = []
    for model in card["models"]:
        model_id = str(model["model_id"])
        entries = sorted(by_volume.get(model_id, []), key=lambda row: int(row["items_per_month"]))
        volume_rows.append(
            {
                "model": model_id,
                "name": name(model_id),
                "cost": _metric(
                    round(float(model["cost_per_1k"]), 4),
                    _per_1k(float(model["cost_per_1k"])),
                    card_source,
                ),
                # The exact measured cost, so the browser arithmetic cannot drift from
                # the card's 10,000-item column through a rounded display value.
                "per_1k": float(model["cost_per_1k"]),
                "volumes": [
                    {
                        "items": int(entry["items_per_month"]),
                        "display": f"{int(entry['items_per_month']):,}",
                        "amount": _metric(
                            round(float(entry["cost_usd"]), 4),
                            _money(float(entry["cost_usd"])),
                            card_source,
                        ),
                    }
                    for entry in entries
                ],
            }
        )

    cards = []
    for task in CARD_TASKS:
        source = f"results/cards/{CARD_FILES[task]}.json"
        payload = _read_json(root, source)
        task_picks = []
        for key in CARD_PICK_ORDER:
            model_id = payload.get(key)
            if not model_id:
                continue
            task_picks.append(
                {
                    "key": key,
                    "label": spec["cards_pick_labels"][key],
                    "model": str(model_id),
                    "name": name(str(model_id)),
                }
            )
        cards.append(
            {
                "task": task,
                "title": str(payload["title"]),
                "job": str(payload["job"]),
                "sample": _metric(
                    int(payload["sample_size"]),
                    f"{int(payload['sample_size']):,}",
                    source,
                ),
                "picks": task_picks,
                "source": source,
            }
        )

    return {
        "kicker": spec["kicker"],
        "title": spec["title"],
        "lead": spec["lead"],
        "defect": {
            "kicker": spec["defect_kicker"],
            "title": str(card["title"]),
            "job": str(card["job"]),
            "data": str(card["data_source"]),
            "licence": str(card["licence"]),
            "sample_note": spec["defect_sample_note"],
            "source_note": spec["defect_source_note"],
            "metric_label": TASK_META["T-DEFECT"]["metric_label"],
            "columns": {
                "model": spec["col_model"],
                "primary": spec["col_primary"],
                "coverage": spec["col_coverage"],
                "answered": spec["col_answered"],
                "safety": spec["col_safety"],
                "cost": spec["col_cost"],
                "latency": spec["col_latency"],
            },
            "baseline_label": spec["baseline_label"],
            "rows": table,
            "sample": _metric(
                int(card["sample_size"]),
                f"{int(card['sample_size']):,}",
                card_source,
            ),
            "floor": {
                "title": spec["floor_title"],
                "value": _metric(floor_min, _pct(floor_min), card_source),
                "met": floor_met,
                "baseline_met": baseline_met,
                "sentence": floor_sentence,
                "rule_prefix": spec["floor_rule_prefix"],
                "rule_suffix": spec["floor_rule_suffix"],
            },
            "coverage": {
                "title": spec["coverage_title"],
                "sentence": coverage_sentence,
            },
            "picks": picks,
            "gate_demo": gate_demo,
            "source": card_source,
        },
        "recheck": {
            "title": spec["recheck_title"],
            "lead": spec["recheck_lead"],
            "steps": spec["steps"],
            "demo_title": spec["demo_title"],
            "demo_lead": spec["demo_lead"],
            "demo_refused_label": spec["demo_refused_label"],
            "demo_accepted_label": spec["demo_accepted_label"],
            "demo_note": spec["demo_note"],
            "demo": demo,
        },
        "volume": {
            "title": spec["volume_title"],
            "lead": spec["volume_lead"],
            "label": spec["volume_label"],
            "hint": spec["volume_hint"],
            "no_js": spec["volume_no_js"],
            "live_column": spec["volume_live_col"],
            "price_note": spec["volume_price_note"],
            "unknown": spec["volume_unknown"],
            "default_items": 10_000,
            "headers": headers,
            "rows": volume_rows,
        },
        "cards": {
            "title": spec["cards_title"],
            "lead": spec["cards_lead"],
            "entries": cards,
        },
    }


def build_page_data(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return the page data for the repository at ``root`` without writing anything."""
    content = _read_json(root, str(CONTENT_PATH))
    manifest = _read_json(root, "results/runs/live/manifest.json")
    cells_doc = _read_json(root, "results/runs/live/cells.json")
    budget = _read_json(root, "results/runs/live/budget.json")
    recommendations = _read_json(root, "results/recommendations.json")
    baseline = _read_json(root, "results/baseline/metrics.json")
    models = _load_models(root)
    cells: list[dict[str, Any]] = cells_doc["cells"]

    model_rows = []
    for entry in manifest["models"]:
        model_id = str(entry["model_id"])
        model_rows.append(
            {
                "model": model_id,
                "name": str(models[model_id]["display_name"]),
                "lab": str(models[model_id]["lab"]),
                "tier": str(models[model_id]["tier"]),
                "connection": str(models[model_id]["connection_id"]),
                "source": "results/runs/live/manifest.json",
            }
        )

    tasks = _tasks(root, models, recommendations, cells)
    example = _example(root, content, models)
    docs = _docs(root, content, models, cells)
    inventory = _inventory(root)
    gate = _gate(root, models)
    before_after = _before_after(root, content)
    v11 = _v11(root, content, models)

    n_by_task = {str(cell["task"]): int(cell["n"]) for cell in cells}
    messages_per_model = sum(n_by_task[task] for task in BUILTIN_TASK_ORDER)
    recorded_answers = sum(int(cell["n"]) for cell in cells)
    best_route = max(float(cell["primary_metric"]) for cell in cells if cell["task"] == "T-ROUTE")
    best_scope = max(float(cell["primary_metric"]) for cell in cells if cell["task"] == "T-SCOPE")
    cheapest_picks = [
        {
            "task": task["short"],
            "task_id": task["id"],
            "model": task["picks"]["cheapest"]["name"],
            "metric": task["picks"]["cheapest"]["metric"],
            "cost": task["picks"]["cheapest"]["cost"],
        }
        for task in tasks
    ]

    run_date = _metric(
        str(manifest["started_at"]),
        _long_date(str(manifest["started_at"])),
        "results/runs/live/manifest.json",
    )
    numbers = [
        {
            "id": "tools-moved",
            "label": "AI tools moved onto one foundation",
            "value": _metric(
                int(inventory["before_count"]["value"]),
                str(inventory["before_count"]["value"]),
                "results/inventory/before.json",
            ),
            "note": "Five different wirings before, two shared call sites after.",
        },
        {
            "id": "messages-per-model",
            "label": "Messages and questions per model",
            "value": _metric(
                messages_per_model, _count(messages_per_model), "results/runs/live/cells.json"
            ),
            "note": "Routing, scope, reply and docs items, each answered by all seven models.",
        },
        {
            "id": "recorded-answers",
            "label": "Recorded model answers",
            "value": _metric(
                recorded_answers, _count(recorded_answers), "results/runs/live/calls.jsonl"
            ),
            "note": "One trace row per item and model. The reply sample is a subset of the routing sample.",
        },
        {
            "id": "models",
            "label": "Models compared",
            "value": _metric(
                len(model_rows), str(len(model_rows)), "results/runs/live/manifest.json"
            ),
            "note": "One flagship and six small models, all through one gateway.",
        },
        {
            "id": "spend",
            "label": "Total benchmark spend",
            "value": _metric(
                float(budget["total_spent_usd"]),
                _usd(float(budget["total_spent_usd"])),
                "results/runs/live/budget.json",
            ),
            "note": "The only paid step in the project. make all replays it for free.",
        },
        {
            "id": "gate-blocked",
            "label": "Cheaper candidates blocked by the gate",
            "value": _metric(
                gate["blocked_count"]["value"],
                gate["blocked_count"]["display"],
                "results/gate_demo.json",
            ),
            "note": "Both cheaper candidates lost quality. A blocked swap is a result, not a failure.",
        },
    ]

    baseline_block = {
        "route": _metric(
            float(baseline["route"]["macro_f1"]),
            _pct(float(baseline["route"]["macro_f1"])),
            "results/baseline/metrics.json",
        ),
        "scope": _metric(
            float(baseline["scope"]["f1"]),
            _pct(float(baseline["scope"]["f1"])),
            "results/baseline/metrics.json",
        ),
        "best_route": _metric(best_route, _pct(best_route), "results/runs/live/cells.json"),
        "best_scope": _metric(best_scope, _pct(best_scope), "results/runs/live/cells.json"),
        "cost": _metric(0.0, "$0", "results/baseline/metrics.json"),
    }

    return {
        "schema_version": 1,
        "site": content["site"],
        "hero": content["hero"],
        "numbers": {
            **content["numbers"],
            "cards": numbers,
            "best_route": _metric(best_route, _pct(best_route), "results/runs/live/cells.json"),
            "best_scope": _metric(best_scope, _pct(best_scope), "results/runs/live/cells.json"),
            "baseline": baseline_block,
            "cheapest_picks": cheapest_picks,
        },
        "picker": content["picker"],
        "tasks": tasks,
        "v11": v11,
        "example": example,
        "before_after": {**content["before_after"], **before_after},
        "inventory": inventory,
        "gate": {
            **content["gate"],
            "pairs": gate["pairs"],
            "blocked_count": gate["blocked_count"],
            "limit": gate["limit"],
            "schema_floor": gate["schema_floor"],
            "refusal": gate["refusal"],
        },
        "docs": {
            **content["docs"],
            "examples": docs,
            "metric": docs[0]["hit_at_4"] if docs else None,
            "answerable": _metric(
                float(cells_doc["cells"][0]["metrics"]["n_answerable"]),
                f"{cells_doc['cells'][0]['metrics']['n_answerable']:g}",
                "results/runs/live/cells.json",
            ),
            "unanswerable": _metric(
                float(cells_doc["cells"][0]["metrics"]["n_unanswerable"]),
                f"{cells_doc['cells'][0]['metrics']['n_unanswerable']:g}",
                "results/runs/live/cells.json",
            ),
        },
        "limits": {**content["limits"], "entries": content["limits"]["items"]},
        "technical": {
            **content["technical"],
            "run_date": run_date,
            "spend": next(card["value"] for card in numbers if card["id"] == "spend"),
            "measured": _metric(
                float(budget["total_measured_usd"]),
                _usd(float(budget["total_measured_usd"])),
                "results/runs/live/budget.json",
            ),
            "models": model_rows,
        },
        "footer": content["footer"],
        "sources": [
            "results/runs/live/manifest.json",
            "results/runs/live/cells.json",
            "results/runs/live/calls.jsonl",
            "results/runs/live/budget.json",
            "results/recommendations.json",
            "results/baseline/metrics.json",
            "results/inventory/before.json",
            "results/inventory/after.json",
            "results/gate_demo.json",
            "results/gate_refused_demo.json",
            "results/runs/live-v1.1/manifest.json",
            "results/runs/live-v1.1/cells.json",
            "results/baseline/defect.json",
            "results/gate_demo_defect.json",
            "results/runs/recheck-demo/recheck.json",
            "results/runs/recheck-demo-accepted/recheck.json",
            "results/cards/T-ROUTE.json",
            "results/cards/T-SCOPE.json",
            "results/cards/T-REPLY.json",
            "results/cards/T-DOCS.json",
            "results/cards/defect.json",
        ],
    }


def render_json(payload: dict[str, Any]) -> str:
    """Serialise page data deterministically (sorted keys, fixed indentation)."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_page_data(root: Path = REPO_ROOT, output: Path | None = None) -> Path:
    """Write ``results/page/page-data.json`` and return its path."""
    payload = build_page_data(root)
    destination = output if output is not None else root / PAGE_DATA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_json(payload), encoding="utf-8")
    return destination


def main() -> None:  # pragma: no cover - module entry point
    path = write_page_data()
    sys.stdout.write(f"page-data: wrote {path.relative_to(REPO_ROOT).as_posix()}\n")


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
