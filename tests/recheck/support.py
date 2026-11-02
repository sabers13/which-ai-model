"""Tiny offline fixtures for the re-check tests: one fake model, one T-SCOPE sample."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from which_ai_model.contracts import ModelRef

MODEL_ID = "fake-a"
CONNECTION_ID = "fake"


def write_config(root: Path) -> Path:
    """A config directory with one fake model, its price and generous pacing."""
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "budget.toml").write_text(
        "[budget]\nwindow_hours = 5\n\n"
        "[limits.opencode-go]\nmax_concurrency = 2\nrequests_per_minute = 30\n",
        encoding="utf-8",
    )
    (config / "prices.toml").write_text(
        '[models."fake-a"]\ninput_per_mtok = 1.0\noutput_per_mtok = 10.0\ncurrency = "USD"\n',
        encoding="utf-8",
    )
    (config / "models.lock.toml").write_text(
        "[[model]]\n"
        'model_id = "fake-a"\n'
        'connection_id = "fake"\n'
        'protocol = "OPENAI_COMPATIBLE"\n'
        'reasoning = "minimal"\n'
        "max_tokens = 1024\n",
        encoding="utf-8",
    )
    return config


def write_samples(root: Path) -> Path:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for name in ("route", "scope", "reply"):
        (samples / f"{name}.txt").write_text("s1\ns2\ns3\ns4\n", encoding="utf-8")
    return samples


def write_scope_items(root: Path, *, count: int = 4) -> Path:
    items_dir = root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": f"s{index}",
            "text": f"message {index}",
            "out_of_scope": index % 2 == 0,
        }
        for index in range(count)
    ]
    (items_dir / "scope.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return items_dir


def scope_invoker(
    items_dir: Path,
) -> Callable[[ModelRef, Sequence[dict[str, str]]], tuple[dict[str, Any], int]]:
    """A fake model that answers every T-SCOPE message correctly."""
    truth = {
        str(row["text"]): bool(row["out_of_scope"])
        for row in (
            json.loads(line)
            for line in (items_dir / "scope.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    calls: list[str] = []

    def invoke(_model: ModelRef, messages: Sequence[dict[str, str]]) -> tuple[dict[str, Any], int]:
        text = str(messages[-1]["content"]).splitlines()[-1]
        calls.append(text)
        response: dict[str, Any] = {
            "content": json.dumps({"out_of_scope": truth[text]}),
            "tool_calls": [],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
        return response, 7

    invoke.calls = calls  # type: ignore[attr-defined]
    return invoke


def run_kwargs(root: Path, config: Path) -> dict[str, Any]:
    return {
        "config": config,
        "models_path": config / "models.lock.toml",
        "recordings_dir": root / "recordings",
        "samples_dir": root / "samples",
        "items_dir": root / "items",
        "runs_dir": root / "runs",
        "reports_dir": root / "reports",
        "recommendations_path": root / "recommendations.json",
        "baseline_metrics": root / "missing-baseline.json",
    }


def recheck_kwargs(root: Path, config: Path) -> dict[str, Any]:
    return {
        "models": config / "models.lock.toml",
        "config": config,
        "tasks": ["T-SCOPE"],
        "samples_dir": root / "samples",
        "items_dir": root / "items",
        "runs_dir": root / "runs",
        "fresh_root_dir": root / "recordings" / "fresh",
        "reports_dir": root / "reports",
        "baseline_metrics": root / "missing-baseline.json",
    }
