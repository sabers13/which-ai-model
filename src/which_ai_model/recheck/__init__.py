"""Re-check runs (``PLAN-v1.1`` §A6): fresh answers, gated against an earlier run.

A re-check records **new** answers under ``recordings/fresh/<run_id>/`` (never reading
the committed evidence root), scores them with the ordinary runner, and gates the fresh
run against an earlier run. The verdict, the settings comparison and both run paths land
in ``results/runs/<run_id>/recheck.json``.

The gate is the M3 interface (``docs/INTERFACES-v1.1.md`` §3). Until the M3 module
lands, the local fallbacks below implement the same shapes: ``check_settings`` and the
ordinary v1 gate for built-in tasks. A caller can inject another gate for tests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from which_ai_model import gate as gate_module
from which_ai_model.bench.run import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DOCS_GOLD,
    DEFAULT_RUNS_DIR,
    DEFAULT_SAMPLES_DIR,
    Clock,
    Predictor,
    load_models,
    run,
    run_settings_for,
)
from which_ai_model.bench.score import load_cells, load_spec_cells, write_cells
from which_ai_model.contracts import (
    ModelRef,
    RunSettings,
    TaskName,
    TaskRef,
    builtin_task,
    task_key,
)
from which_ai_model.core.record import DEFAULT_FRESH_RECORDINGS_ROOT, fresh_recording_root
from which_ai_model.gate import GateOutcome, evaluate_gate

__all__ = [
    "GateHook",
    "GateRequest",
    "RecheckVerdict",
    "SettingsCheck",
    "evaluate_default_gate",
    "fresh_root",
    "run_recheck",
]

#: The runner's call shape, repeated locally so this module need not import a private alias.
type Invoker = Callable[[ModelRef, Sequence[dict[str, str]]], tuple[dict[str, Any], int]]


@dataclass(frozen=True)
class SettingsCheck:
    """The M3 ``SettingsCheck`` shape: comparable, or refused with one reason each."""

    comparable: bool
    differences: tuple[str, ...] = ()
    accepted: bool = False

    @property
    def refused(self) -> bool:
        return not self.comparable

    @property
    def warnings(self) -> tuple[str, ...]:
        """Accepted differences are warnings, never silent."""
        return self.differences if self.comparable and self.accepted else ()


@dataclass(frozen=True)
class RecheckVerdict:
    """One gate verdict in the ``wam gate`` vocabulary."""

    outcome: GateOutcome
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    comparisons: tuple[Mapping[str, Any], ...] = ()
    rules: tuple[Mapping[str, Any], ...] = ()

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "outcome": self.outcome.value,
            "exit_code": self.exit_code,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "comparisons": [dict(row) for row in self.comparisons],
        }
        if self.rules:
            payload["rules"] = [dict(row) for row in self.rules]
        return payload


@dataclass(frozen=True)
class GateRequest:
    """Everything the gate hook needs; a fake gate implements the same call."""

    baseline: Path
    candidate: Path
    tasks: tuple[str, ...]
    baseline_model: str
    candidate_model: str
    settings: SettingsCheck


type GateHook = Callable[[GateRequest], RecheckVerdict]


def fresh_root(run_id: str, *, root: Path = DEFAULT_FRESH_RECORDINGS_ROOT) -> Path:
    """``<root>/<run_id>``; the only root a fresh run may read or write."""
    return fresh_recording_root(run_id, root=root)


# ------------------------------------------------------------------ settings


def _fallback_check_settings(
    baseline: Mapping[str, RunSettings],
    candidate: Mapping[str, RunSettings],
    *,
    accept_mismatch: bool,
) -> SettingsCheck:
    """The M3 rule until M3 lands: a difference is refused unless accepted."""
    differences: list[str] = []
    for model_id in sorted(set(baseline) | set(candidate)):
        old = baseline.get(model_id, RunSettings.unrecorded())
        new = candidate.get(model_id, RunSettings.unrecorded())
        for field in ("temperature", "reasoning", "max_output_tokens"):
            old_value = getattr(old, field)
            new_value = getattr(new, field)
            if old_value == new_value or "not-recorded" in (old_value, new_value):
                continue
            differences.append(
                f"{model_id}: {field} {new_value!r} (fresh) vs {old_value!r} (earlier)"
            )
    accepted = bool(differences) and accept_mismatch
    return SettingsCheck(
        comparable=accept_mismatch or not differences,
        differences=tuple(differences),
        accepted=accepted,
    )


def _check_settings(
    baseline: Mapping[str, RunSettings],
    candidate: Mapping[str, RunSettings],
    *,
    accept_mismatch: bool,
) -> SettingsCheck:
    checker = getattr(gate_module, "check_settings", None)
    if callable(checker):  # M3 landed: its rule and wording win.
        result = checker(baseline, candidate, accept_mismatch=accept_mismatch)
        differences = tuple(str(item) for item in result.differences)
        comparable = bool(result.comparable)
        return SettingsCheck(
            comparable=comparable,
            differences=differences,
            accepted=bool(differences) and comparable and accept_mismatch,
        )
    return _fallback_check_settings(baseline, candidate, accept_mismatch=accept_mismatch)


def _manifest_settings(document: Mapping[str, Any]) -> dict[str, RunSettings]:
    raw = document.get("settings")
    if not isinstance(raw, dict):
        return {}
    settings: dict[str, RunSettings] = {}
    for model_id, body in raw.items():
        if isinstance(model_id, str) and isinstance(body, dict):
            settings[model_id] = RunSettings.model_validate(body)
    return settings


def _settings_for(document: Mapping[str, Any], model: str) -> RunSettings:
    settings = _manifest_settings(document)
    plain_id = model.rsplit("/", 1)[-1]
    return settings.get(plain_id, settings.get(model, RunSettings.unrecorded()))


# ---------------------------------------------------------------------- gate


def _filtered_run(source: Path, target: Path, tasks: Sequence[str]) -> None:
    """Copy a run's manifest and only the selected tasks' cells into ``target``.

    ``tasks`` holds built-in names and/or spec ids; spec rows are carried under their
    own key so the spec gate can score them.
    """
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "manifest.json", target / "manifest.json")
    selected = set(tasks)
    cells = [cell for cell in load_cells(source / "cells.json") if cell.task.value in selected]
    spec_cells = [cell for cell in load_spec_cells(source / "cells.json") if cell.task in selected]
    write_cells(target / "cells.json", cells, spec_cells=spec_cells)


def _v1_gate(request: GateRequest) -> RecheckVerdict:
    """The ordinary v1 gate, limited to the tasks the re-check actually recorded."""
    with tempfile.TemporaryDirectory(prefix="wam-recheck-gate-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        candidate = root / "candidate"
        _filtered_run(request.baseline, baseline, request.tasks)
        _filtered_run(request.candidate, candidate, request.tasks)
        result = evaluate_gate(
            baseline,
            candidate,
            model_a=request.baseline_model,
            model_b=request.candidate_model,
        )
    return RecheckVerdict(
        outcome=result.outcome,
        reasons=tuple(result.reasons),
        warnings=request.settings.warnings,
        comparisons=tuple(comparison.as_json() for comparison in result.comparisons),
        rules=tuple(result.rules),
    )


def _spec_gate(request: GateRequest) -> RecheckVerdict:
    """§A5 gate for the spec ids in a re-check, scored from ``spec_cells``."""
    from which_ai_model.gate.spec import evaluate_spec_runs

    with tempfile.TemporaryDirectory(prefix="wam-recheck-spec-gate-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        candidate = root / "candidate"
        _filtered_run(request.baseline, baseline, request.tasks)
        _filtered_run(request.candidate, candidate, request.tasks)
        result = evaluate_spec_runs(
            baseline,
            candidate,
            model_a=request.baseline_model,
            model_b=request.candidate_model,
        )
    return RecheckVerdict(
        outcome=result.outcome,
        reasons=tuple(result.reasons),
        warnings=tuple(result.warnings) or request.settings.warnings,
        rules=tuple(rule.as_json() for rule in result.rules),
    )


def _merge_verdicts(verdicts: Sequence[RecheckVerdict]) -> RecheckVerdict:
    order = {GateOutcome.PASS: 0, GateOutcome.FAIL: 1, GateOutcome.REFUSED: 2}
    outcome = max((verdict.outcome for verdict in verdicts), key=order.__getitem__)
    return RecheckVerdict(
        outcome=outcome,
        reasons=tuple(reason for verdict in verdicts for reason in verdict.reasons),
        warnings=tuple(warning for verdict in verdicts for warning in verdict.warnings),
        comparisons=tuple(comparison for verdict in verdicts for comparison in verdict.comparisons),
        rules=tuple(rule for verdict in verdicts for rule in verdict.rules),
    )


def evaluate_default_gate(request: GateRequest) -> RecheckVerdict:
    """Settings first, then each selected task through its own gate.

    Built-in tasks use the v1 gate and spec ids use the §A5 gate, scored from
    ``spec_cells``; a re-check may carry both. A settings difference is REFUSED unless
    accepted.
    """
    if request.settings.refused:
        reasons = tuple(
            f"sampling settings differ for {request.candidate_model} vs "
            f"{request.baseline_model}: {difference}"
            for difference in request.settings.differences
        )
        return RecheckVerdict(GateOutcome.REFUSED, reasons=reasons or ("settings differ",))
    builtin = tuple(ref for ref in request.tasks if builtin_task(ref) is not None)
    specs = tuple(ref for ref in request.tasks if builtin_task(ref) is None)
    verdicts: list[RecheckVerdict] = []
    if builtin:
        verdicts.append(
            _v1_gate(
                GateRequest(
                    baseline=request.baseline,
                    candidate=request.candidate,
                    tasks=builtin,
                    baseline_model=request.baseline_model,
                    candidate_model=request.candidate_model,
                    settings=request.settings,
                )
            )
        )
    if specs:
        verdicts.append(
            _spec_gate(
                GateRequest(
                    baseline=request.baseline,
                    candidate=request.candidate,
                    tasks=specs,
                    baseline_model=request.baseline_model,
                    candidate_model=request.candidate_model,
                    settings=request.settings,
                )
            )
        )
    if not verdicts:
        return RecheckVerdict(GateOutcome.REFUSED, reasons=("no tasks selected",))
    return _merge_verdicts(verdicts)


# --------------------------------------------------------------------- report


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    document = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        msg = f"{run_dir}: manifest.json must be a JSON object"
        raise ValueError(msg)
    return document


def _model_id(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _outcome_of(verdicts: Sequence[RecheckVerdict]) -> GateOutcome:
    order = {GateOutcome.PASS: 0, GateOutcome.FAIL: 1, GateOutcome.REFUSED: 2}
    if not verdicts:
        return GateOutcome.REFUSED
    return max((verdict.outcome for verdict in verdicts), key=order.__getitem__)


def _pair_label(baseline_model: str, candidate_model: str) -> str:
    if baseline_model == candidate_model:
        return candidate_model
    return f"{candidate_model} vs {baseline_model}"


def run_recheck(
    *,
    models: Path,
    tasks: Sequence[TaskRef],
    against: Path,
    candidate: str | None = None,
    baseline: str | None = None,
    run_id: str | None = None,
    accept_settings_mismatch: bool = False,
    mode: str = "record",
    fresh_run: str | None = None,
    fresh_root_dir: Path = DEFAULT_FRESH_RECORDINGS_ROOT,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    config: Path = Path("config"),
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    items_dir: Path | None = None,
    docs_gold: Path = DEFAULT_DOCS_GOLD,
    reports_dir: Path = Path("reports"),
    baseline_metrics: Path = Path("results/baseline/metrics.json"),
    registry: Mapping[TaskName, Predictor] | None = None,
    invoker: Invoker | None = None,
    gate: GateHook | None = None,
    clock: Clock | None = None,
    workers: int | None = None,
    temperature: float = 0.0,
) -> Path:
    """Fresh record -> score -> gate -> ``results/runs/<run_id>/recheck.json``.

    ``mode="record"`` makes the calls; ``mode="replay"`` scores committed fresh
    recordings (the demo's replay path) without a network call. ``fresh_run`` selects
    which fresh recording root a replay reads (default: ``run_id``).
    """
    if mode not in {"record", "replay"}:
        msg = f"unsupported recheck mode {mode!r}; use record or replay"
        raise ValueError(msg)
    resolved_run_id = run_id or f"recheck-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    selected_tasks: list[str] = []
    for ref in tasks:
        key = task_key(ref)
        if key not in selected_tasks:
            selected_tasks.append(key)
    if not selected_tasks:
        msg = "recheck: no tasks selected"
        raise ValueError(msg)

    recorded_run = fresh_run or resolved_run_id
    recordings = fresh_root(recorded_run, root=fresh_root_dir)
    if mode == "replay" and not recordings.is_dir():
        msg = (
            f"recheck: no fresh recordings at {recordings}; record them first "
            "(M7 records the committed demo)"
        )
        raise FileNotFoundError(msg)

    candidate_id = _model_id(candidate) if candidate is not None else None
    selected_models = [candidate_id] if candidate_id is not None else None
    settings = {
        spec.ref.model_id: run_settings_for(spec, temperature=temperature)
        for spec in load_models(models, model_ids=selected_models)
    }
    run(
        mode=mode,
        config=config,
        run_id=resolved_run_id,
        models_path=models,
        tasks=tuple(selected_tasks),
        recordings_dir=recordings,
        samples_dir=samples_dir,
        data_root=data_root,
        items_dir=items_dir,
        docs_gold=docs_gold,
        runs_dir=runs_dir,
        reports_dir=reports_dir,
        recommendations_path=Path(runs_dir) / resolved_run_id / "recommendations.json",
        baseline_metrics=baseline_metrics,
        registry=registry,
        invoker=invoker,
        clock=clock,
        workers=workers,
        model_ids=selected_models,
        fresh=mode == "record",
        fresh_root=fresh_root_dir,
        settings=settings,
        lock_sha256=hashlib.sha256(Path(models).read_bytes()).hexdigest(),
        temperature=temperature,
    )

    run_dir = Path(runs_dir) / resolved_run_id
    if not run_dir.is_dir():  # pragma: no cover - run() always creates it
        msg = f"recheck: runner wrote no run directory at {run_dir}"
        raise FileNotFoundError(msg)
    candidate_manifest = _read_manifest(run_dir)
    baseline_manifest = _read_manifest(Path(against))
    candidate_cells = load_cells(run_dir / "cells.json")
    candidate_spec_cells = load_spec_cells(run_dir / "cells.json")
    candidate_models = sorted(
        {cell.model.model_id for cell in candidate_cells}
        | {cell.model_id for cell in candidate_spec_cells}
    )
    if candidate_id is not None and candidate_id not in candidate_models:
        msg = f"recheck: the fresh run has no cell for {candidate_id!r}"
        raise ValueError(msg)
    pairs: list[tuple[str, str]] = []
    if candidate_id is not None:
        pairs.append((baseline or candidate_id, candidate_id))
    else:
        pairs.extend((baseline or model_id, model_id) for model_id in candidate_models)
    if not pairs:
        msg = f"recheck: no candidate model cell in {run_dir}"
        raise ValueError(msg)

    verdicts: list[RecheckVerdict] = []
    settings_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    hook = gate or evaluate_default_gate
    for baseline_model, candidate_model in pairs:
        # Key both sides by the candidate id: a renamed model still compares one pair.
        baseline_settings = {
            candidate_model: _settings_for(baseline_manifest, baseline_model),
        }
        candidate_settings = {
            candidate_model: _settings_for(candidate_manifest, candidate_model),
        }
        check = _check_settings(
            baseline_settings,
            candidate_settings,
            accept_mismatch=accept_settings_mismatch,
        )
        request = GateRequest(
            baseline=Path(against),
            candidate=run_dir,
            tasks=tuple(selected_tasks),
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            settings=check,
        )
        verdict = hook(request)
        verdicts.append(verdict)
        settings_rows.append(
            {
                "candidate_model": candidate_model,
                "baseline_model": baseline_model,
                "comparable": check.comparable,
                "accepted": check.accepted,
                "differences": list(check.differences),
            }
        )
        gate_row: dict[str, Any] = {
            "pair": _pair_label(baseline_model, candidate_model),
        }
        gate_row.update(verdict.as_json())
        gate_rows.append(gate_row)

    outcome = _outcome_of(verdicts)
    reasons = [
        f"{row['pair']}: {reason}"
        for row, verdict in zip(gate_rows, verdicts, strict=True)
        for reason in verdict.reasons
    ]
    warnings = [
        f"{row['pair']}: {warning}"
        for row, verdict in zip(gate_rows, verdicts, strict=True)
        for warning in verdict.warnings
    ]
    payload = {
        "run_id": resolved_run_id,
        "mode": mode,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "against": str(against),
        "candidate_run": str(run_dir),
        "recordings_root": str(recordings),
        "tasks": list(selected_tasks),
        "settings": {
            "accept_mismatch": accept_settings_mismatch,
            "pairs": settings_rows,
        },
        "gate": {
            "outcome": outcome.value,
            "exit_code": outcome.exit_code,
            "reasons": reasons,
            "warnings": warnings,
            "pairs": gate_rows,
        },
    }
    report_path = run_dir / "recheck.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path
