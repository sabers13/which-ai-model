"""``wam task`` — validate and scaffold spec tasks (``PLAN-v1.1`` §A3).

`wam task check <spec>` reads the spec and its files and prints each problem in plain
language; `wam task new <id>` writes a commented skeleton spec and prompt template.
"""

from __future__ import annotations

from pathlib import Path

import typer

from which_ai_model.task.spec import (
    SpecError,
    load_spec,
    validate_spec,
    write_skeleton,
)

app = typer.Typer(
    name="task",
    help="Validate and scaffold a task spec (one TOML file plus labelled data).",
    no_args_is_help=True,
)


@app.command("check")
def check_command(
    spec: Path = typer.Argument(..., help="Path to the task spec, e.g. tasks/defect.toml."),
) -> None:
    """Validate a spec and its data: ids, labels, files and prompt placeholders."""
    try:
        task_spec = load_spec(spec)
    except SpecError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    problems = validate_spec(task_spec)
    if problems:
        typer.echo(
            f"{spec} has {len(problems)} problem{'' if len(problems) == 1 else 's'} to fix:",
            err=True,
        )
        for number, problem in enumerate(problems, start=1):
            typer.echo(f"  {number}. {problem}", err=True)
        raise typer.Exit(code=1)

    outputs = ", ".join(f"{output.name} ({output.kind.value})" for output in task_spec.outputs)
    typer.echo(
        f"OK: {task_spec.id!r} is valid. Outputs: {outputs}. "
        f"Prompt version {task_spec.prompt_version}, scorer {task_spec.scorer_version}."
    )


@app.command("new")
def new_command(
    task_id: str = typer.Argument(..., help="New spec id, e.g. 'defect'."),
    tasks_dir: Path = typer.Option(Path("tasks"), help="Directory to write the spec into."),
) -> None:
    """Write a commented skeleton spec and prompt template for a new task."""
    try:
        spec_path, prompt_path = write_skeleton(task_id, tasks_dir=tasks_dir)
    except SpecError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {spec_path}")
    typer.echo(f"Wrote {prompt_path}")
    typer.echo("Next: add the data and labels files the comments name, then run")
    typer.echo(f"  wam task check {spec_path}")
