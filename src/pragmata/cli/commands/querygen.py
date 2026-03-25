"""CLI commands for synthetic query generation."""

import typer

from pragmata import querygen
from pragmata.cli.parsing import parse_optional_cli_value

app = typer.Typer(help="Synthetic query generation commands.")


@app.command("gen-queries")
def gen_queries_command(
    domains: str | None = typer.Option(None, "--domains", help="Domain specification as text or JSON list."),
    roles: str | None = typer.Option(None, "--roles", help="Role specification."),
    languages: str | None = typer.Option(None, "--languages", help="Language specification."),
    topics: str | None = typer.Option(None, "--topics", help="Topic specification."),
    intents: str | None = typer.Option(None, "--intents", help="Intent specification."),
    tasks: str | None = typer.Option(None, "--tasks", help="Task specification."),
    disallowed_topics: str | None = typer.Option(
        None,
        "--disallowed-topics",
        help="Disallowed topics as a JSON list of strings.",
    ),
    difficulty: str | None = typer.Option(None, "--difficulty", help="Difficulty specification."),
    formats: str | None = typer.Option(None, "--formats", help="Format specification."),
    base_dir: str | None = typer.Option(None, "--base-dir", help="Workspace base directory."),
    config_path: str | None = typer.Option(None, "--config-path", help="Path to config file."),
    n_queries: str | None = typer.Option(None, "--n-queries", help="Number of queries."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run identifier."),
    model_provider: str | None = typer.Option(None, "--model-provider", help="Model provider."),
    planning_model: str | None = typer.Option(None, "--planning-model", help="Planning model."),
    realization_model: str | None = typer.Option(
        None,
        "--realization-model",
        help="Realization model.",
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Provider base URL."),
    model_kwargs: str | None = typer.Option(
        None,
        "--model-kwargs",
        help="Additional model keyword arguments as a JSON object.",),
) -> None:
    """Prepare a synthetic query generation run."""
    result = querygen.gen_queries(
        domains=parse_optional_cli_value(domains),
        roles=parse_optional_cli_value(roles),
        languages=parse_optional_cli_value(languages),
        topics=parse_optional_cli_value(topics),
        intents=parse_optional_cli_value(intents),
        tasks=parse_optional_cli_value(tasks),
        disallowed_topics=parse_optional_cli_value(disallowed_topics),
        difficulty=parse_optional_cli_value(difficulty),
        formats=parse_optional_cli_value(formats),
        base_dir=parse_optional_cli_value(base_dir),
        config_path=parse_optional_cli_value(config_path),
        n_queries=parse_optional_cli_value(n_queries),
        run_id=parse_optional_cli_value(run_id),
        model_provider=parse_optional_cli_value(model_provider),
        planning_model=parse_optional_cli_value(planning_model),
        realization_model=parse_optional_cli_value(realization_model),
        base_url=parse_optional_cli_value(base_url),
        model_kwargs=parse_optional_cli_value(model_kwargs),
    )

    typer.echo("Synthetic query generation run prepared.")
    typer.echo(f"run_id: {result.settings.run_id}")
    typer.echo(f"run_directory: {result.paths.run_dir}")
    typer.echo(f"synthetic_queries_csv: {result.paths.synthetic_queries_csv}")
    typer.echo(f"synthetic_queries_meta_json: {result.paths.synthetic_queries_meta_json}")