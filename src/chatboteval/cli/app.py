"""Typer CLI application for chatboteval."""

from __future__ import annotations

import typer

from chatboteval.api import get_version

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the installed chatboteval version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(get_version())
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)