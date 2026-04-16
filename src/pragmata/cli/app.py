"""Typer CLI application for pragmata."""

import logging

import typer

from pragmata.api import get_version
from pragmata.cli.commands.querygen import querygen_app

app = typer.Typer(add_completion=False)
app.add_typer(querygen_app, name="querygen")


def _configure_logging(verbosity: int) -> None:
    """Set up root logging for CLI usage. Verbosity: 0=WARNING, 1=INFO, 2+=DEBUG."""
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbosity, 2)]
    logging.basicConfig(
        format="%(levelname)s | %(name)s | %(message)s",
        level=level,
    )

    if level <= logging.INFO:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the installed pragmata version and exit.",
        is_eager=True,
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    ),
) -> None:
    """Run the pragmata CLI."""
    _configure_logging(verbose)
    if version:
        typer.echo(get_version())
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
