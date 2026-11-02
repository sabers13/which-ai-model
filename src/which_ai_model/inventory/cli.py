"""`wam inventory <path> [--json out] [--md out]`."""

from __future__ import annotations

from pathlib import Path

import typer

from which_ai_model.inventory import scan, write_json, write_markdown

app = typer.Typer(
    help="Scan Python source for LLM call sites without importing it.",
    no_args_is_help=True,
    add_completion=False,
    # This lane is a single command (`wam inventory <path> [options]`); allow options
    # after the positional argument, which Click groups disable by default.
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def inventory(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Python file or directory to scan.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json",
        metavar="PATH",
        help="Write the full inventory as JSON.",
    ),
    md_out: Path | None = typer.Option(
        None,
        "--md",
        metavar="PATH",
        help="Write a readable Markdown report.",
    ),
    note: list[str] | None = typer.Option(
        None,
        "--note",
        metavar="TEXT",
        help="A known blind spot to print under the Markdown table (repeatable).",
    ),
) -> None:
    """Report provider, call style, model, credentials, structured output and inputs."""
    result = scan(path)
    for site in result.sites:
        model = "—" if site.model is None else site.model
        if site.model_source == "env":
            model = f"env:{model}"
        inputs = ", ".join(site.inputs) if site.inputs else "—"
        typer.echo(
            f"{site.path}:{site.line}: {site.tool} | {site.provider} | {site.call_style} | "
            f"{model} | {site.structured_output} | inputs: {inputs}"
        )
    typer.echo(f"{len(result.sites)} call site(s) in {result.summary.get('modules', 0)} module(s)")
    if json_out is not None:
        write_json(result, json_out)
        typer.echo(f"wrote {json_out}")
    if md_out is not None:
        write_markdown(result, md_out, notes=tuple(note or ()))
        typer.echo(f"wrote {md_out}")
