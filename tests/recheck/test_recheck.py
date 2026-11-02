"""Re-check end to end: fresh record, score, gate, recheck.json (fake model and gate)."""

from __future__ import annotations

import json
from pathlib import Path

from support import (
    MODEL_ID,
    recheck_kwargs,
    run_kwargs,
    scope_invoker,
    write_config,
    write_samples,
    write_scope_items,
)

from which_ai_model.bench.run import run
from which_ai_model.contracts import TaskName
from which_ai_model.gate import GateOutcome
from which_ai_model.recheck import GateRequest, RecheckVerdict, run_recheck


def _baseline(tmp_path: Path, config: Path) -> Path:
    write_samples(tmp_path)
    items_dir = write_scope_items(tmp_path)
    run(
        "record",
        run_id="baseline",
        tasks=[TaskName.SCOPE],
        invoker=scope_invoker(items_dir),
        workers=1,
        **run_kwargs(tmp_path, config),
    )
    return tmp_path / "runs" / "baseline"


def _strip_settings(run_dir: Path) -> None:
    """Make the baseline look like a v1 manifest: no recorded settings."""
    path = run_dir / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("settings", None)
    document.pop("lock_sha256", None)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_recheck_records_scores_and_calls_the_gate(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    baseline = _baseline(tmp_path, config)
    items_dir = write_scope_items(tmp_path)
    invoker = scope_invoker(items_dir)
    seen: list[GateRequest] = []

    def fake_gate(request: GateRequest) -> RecheckVerdict:
        seen.append(request)
        return RecheckVerdict(GateOutcome.PASS, comparisons=({"fake": True},))

    report_path = run_recheck(
        against=baseline,
        candidate=MODEL_ID,
        run_id="recheck-e2e",
        invoker=invoker,
        workers=1,
        gate=fake_gate,
        **recheck_kwargs(tmp_path, config),
    )

    assert report_path == tmp_path / "runs" / "recheck-e2e" / "recheck.json"
    document = json.loads(report_path.read_text(encoding="utf-8"))
    assert document["gate"]["outcome"] == "PASS"
    assert document["tasks"] == ["T-SCOPE"]
    assert document["candidate_run"] == str(tmp_path / "runs" / "recheck-e2e")
    assert document["recordings_root"] == str(tmp_path / "recordings" / "fresh" / "recheck-e2e")
    assert document["settings"]["pairs"][0]["comparable"] is True
    assert len(invoker.calls) == 4  # type: ignore[attr-defined]
    assert seen
    assert seen[0].candidate_model == MODEL_ID
    assert seen[0].tasks == (TaskName.SCOPE,)
    assert seen[0].settings.comparable is True
    fresh_rows = tmp_path / "recordings" / "fresh" / "recheck-e2e" / "T-SCOPE" / f"{MODEL_ID}.jsonl"
    assert len(fresh_rows.read_text(encoding="utf-8").splitlines()) == 4
    assert (tmp_path / "runs" / "recheck-e2e" / "cells.json").is_file()


def test_default_gate_refuses_a_settings_change_until_accepted(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    baseline = _baseline(tmp_path, config)
    _strip_settings(baseline)
    items_dir = write_scope_items(tmp_path)
    kwargs = recheck_kwargs(tmp_path, config)

    refused_path = run_recheck(
        against=baseline,
        candidate=MODEL_ID,
        run_id="recheck-refused",
        invoker=scope_invoker(items_dir),
        workers=1,
        **kwargs,
    )
    refused = json.loads(refused_path.read_text(encoding="utf-8"))
    assert refused["gate"]["outcome"] == "REFUSED"
    assert refused["gate"]["exit_code"] == 2
    assert any("temperature" in reason for reason in refused["gate"]["reasons"])
    assert refused["settings"]["pairs"][0]["differences"]

    accepted_path = run_recheck(
        against=baseline,
        candidate=MODEL_ID,
        run_id="recheck-accepted",
        accept_settings_mismatch=True,
        mode="replay",
        fresh_run="recheck-refused",
        **kwargs,
    )
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    assert accepted["gate"]["outcome"] == "PASS"
    assert accepted["gate"]["warnings"]
    assert any("temperature" in warning for warning in accepted["gate"]["warnings"])
    assert accepted["settings"]["pairs"][0]["accepted"] is True
