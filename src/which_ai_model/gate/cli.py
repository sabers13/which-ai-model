"""``wam gate`` — the regression gate CLI (``PLAN.md`` §7).

The lane exports ``TOP_LEVEL_COMMANDS`` so the command is ``wam gate ...``, not
``wam gate gate ...``; T00's loader reads that mapping.
"""

from __future__ import annotations

from pathlib import Path

import typer

from which_ai_model.gate import evaluate_gate, write_report

app = typer.Typer(
    name="gate",
    help="Regression gate: PASS, FAIL or REFUSED.",
    no_args_is_help=False,
)


def gate_command(
    baseline: Path = typer.Option(..., "--baseline", help="Baseline run directory."),
    candidate: Path = typer.Option(..., "--candidate", help="Candidate run directory."),
    model_a: str | None = typer.Option(
        None, "--model-a", help="Baseline model id (or connection/model)."
    ),
    model_b: str | None = typer.Option(
        None, "--model-b", help="Candidate model id (or connection/model)."
    ),
    json_path: Path | None = typer.Option(
        None, "--json", help="Write the full JSON report to this path."
    ),
) -> None:
    """Compare two runs: PASS (0), FAIL (1) or REFUSED (2). Comparability comes first."""
    result = evaluate_gate(baseline, candidate, model_a=model_a, model_b=model_b)
    if json_path is not None:
        write_report(json_path, result)
    lines = [f"{result.outcome.value}: {result.baseline} -> {result.candidate}"]
    if result.reasons:
        lines.extend(f"- {reason}" for reason in result.reasons)
    elif result.rules:
        passed = sum(1 for rule in result.rules if rule.get("passed"))
        lines.append(f"- {passed}/{len(result.rules)} spec rule(s) passed")
    else:
        lines.append(
            f"- {len(result.comparisons)} comparison(s); no primary drop over "
            "2.0 points and T-REPLY schema-valid at or above 98%"
        )
    lines.extend(f"WARN: {warning}" for warning in result.warnings)
    typer.echo("\n".join(lines))
    raise typer.Exit(code=result.exit_code)


#: Mounted at the root by ``cli.py``: ``wam gate``.
TOP_LEVEL_COMMANDS = {"gate": gate_command}
