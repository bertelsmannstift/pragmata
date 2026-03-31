"""Typer CLI application for pragmata."""

import typer

from pragmata.api import get_version
from pragmata.cli.commands.querygen import querygen_app

app = typer.Typer(add_completion=False)
app.add_typer(querygen_app, name="querygen")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the installed pragmata version and exit.",
        is_eager=True,
    ),
) -> None:
    """Run the pragmata CLI."""
    if version:
        typer.echo(get_version())
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
