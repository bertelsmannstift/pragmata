"""Typer CLI application for chatboteval."""

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
    """Run the chatboteval CLI."""
    if version:
        typer.echo(get_version())
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
