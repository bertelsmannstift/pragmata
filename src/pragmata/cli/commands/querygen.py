"""CLI commands for synthetic query generation."""

import typer

from pragmata import querygen
from pragmata.api import UNSET
from pragmata.cli.parsing import parse_cli_value

querygen_app = typer.Typer(help="Synthetic query generation commands.")


@querygen_app.command("gen-queries")
def gen_queries_command(
    domains: str | None = typer.Option(
        None, "--domains", help="Domain(s) of the query context. Accepts a string or JSON list."
    ),
    roles: str | None = typer.Option(
        None, "--roles", help="Role(s) or perspective to simulate. Accepts a string or JSON list."
    ),
    languages: str | None = typer.Option(
        None, "--languages", help="Language(s) for generated queries. Accepts a string or JSON list."
    ),
    topics: str | None = typer.Option(None, "--topics", help="Topic(s) to cover. Accepts a string or JSON list."),
    intents: str | None = typer.Option(
        None, "--intents", help="Intent(s) of the query. Accepts a string or JSON list."
    ),
    tasks: str | None = typer.Option(
        None, "--tasks", help="Task type(s) for the query. Accepts a string or JSON list."
    ),
    disallowed_topics: str | None = typer.Option(
        None,
        "--disallowed-topics",
        help="Topic(s) to exclude. Accepts a JSON list of strings.",
    ),
    difficulty: str | None = typer.Option(
        None, "--difficulty", help="Difficulty level(s) for the query. Accepts a string or JSON list."
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Requested format(s) for the query. Accepts a string or JSON list."
    ),
    base_dir: str | None = typer.Option(None, "--base-dir", help="Workspace base directory. Accepts a path string."),
    config_path: str | None = typer.Option(
        None, "--config-path", help="Path to the config file. Accepts a path string."
    ),
    n_queries: int | None = typer.Option(
        None, "--n-queries", help="Number of queries to generate. Accepts an integer."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Identifier for the run. Accepts a string."),
    model_provider: str | None = typer.Option(
        None, "--model-provider", help="Model provider to use. Accepts a string."
    ),
    planning_model: str | None = typer.Option(
        None, "--planning-model", help="Model identifier for the planning stage. Accepts a string."
    ),
    realization_model: str | None = typer.Option(
        None,
        "--realization-model",
        help="Model identifier for the realization stage. Accepts a string.",
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for the provider endpoint. Accepts a URL string"
    ),
    model_kwargs: str | None = typer.Option(
        None,
        "--model-kwargs",
        help="dditional model keyword arguments. Accepts a JSON object.",
    ),
) -> None:
    """Prepare a synthetic query generation run."""
    result = querygen.gen_queries(
        domains=parse_cli_value(domains),
        roles=parse_cli_value(roles),
        languages=parse_cli_value(languages),
        topics=parse_cli_value(topics),
        intents=parse_cli_value(intents),
        tasks=parse_cli_value(tasks),
        disallowed_topics=parse_cli_value(disallowed_topics),
        difficulty=parse_cli_value(difficulty),
        formats=parse_cli_value(formats),
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config_path is None else config_path,
        n_queries=UNSET if n_queries is None else n_queries,
        run_id=UNSET if run_id is None else run_id,
        model_provider=UNSET if model_provider is None else model_provider,
        planning_model=UNSET if planning_model is None else planning_model,
        realization_model=UNSET if realization_model is None else realization_model,
        base_url=UNSET if base_url is None else base_url,
        model_kwargs=parse_cli_value(model_kwargs),
    )

    typer.echo("Synthetic query generation run prepared.")
    typer.echo(f"run_id: {result.settings.run_id}")
    typer.echo(f"run_directory: {result.paths.run_dir}")
    typer.echo(f"synthetic_queries_csv: {result.paths.synthetic_queries_csv}")
    typer.echo(f"synthetic_queries_meta_json: {result.paths.synthetic_queries_meta_json}")
