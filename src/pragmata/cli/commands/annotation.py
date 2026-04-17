"""CLI commands for annotation pipeline."""

import typer

from pragmata.api import UNSET
from pragmata.cli.parsing import parse_tasks, parse_user_specs

annotation_app = typer.Typer(help="Annotation pipeline commands.")

_api_url_opt = typer.Option(None, "--api-url", help="Argilla server URL. Falls back to ARGILLA_API_URL env var.")
_api_key_opt = typer.Option(None, "--api-key", help="Argilla API key. Falls back to ARGILLA_API_KEY env var.")
_prefix_opt = typer.Option(
    None, "--prefix", help="Prefix prepended to workspace and dataset names. Defaults to config file value or ''."
)
_base_dir_opt = typer.Option(
    None, "--base-dir", help="Workspace base directory for run artifacts. Defaults to current working directory."
)
_config_opt = typer.Option(None, "--config", help="Path to YAML config file for annotation settings.")


@annotation_app.command("setup")
def setup_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    prefix: str | None = _prefix_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    min_submitted: int | None = typer.Option(
        None, "--min-submitted", help="Minimum submitted annotations required per record before it is complete."
    ),
    users_json: str | None = typer.Option(
        None,
        "--users",
        help="Path to a JSON file containing user specs. Each entry needs 'username' and 'role' (owner/annotator);"
        " 'workspaces' and 'password' are optional.",
    ),
) -> None:
    """Create Argilla workspaces, datasets, and (optionally) user accounts."""
    from pragmata import annotation

    result = annotation.setup(
        parse_user_specs(users_json),
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        workspace_prefix=UNSET if prefix is None else prefix,
        min_submitted=UNSET if min_submitted is None else min_submitted,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
    )
    typer.echo(f"Workspaces created: {len(result.created_workspaces)}, skipped: {len(result.skipped_workspaces)}")
    typer.echo(f"Datasets created: {len(result.created_datasets)}, skipped: {len(result.skipped_datasets)}")
    typer.echo(f"Users created: {len(result.created_users)}, skipped: {len(result.skipped_users)}")


@annotation_app.command("teardown")
def teardown_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    prefix: str | None = _prefix_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
) -> None:
    """Remove Argilla workspaces and datasets."""
    from pragmata import annotation

    annotation.teardown(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        workspace_prefix=UNSET if prefix is None else prefix,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
    )
    typer.echo("Teardown complete.")


@annotation_app.command("import")
def import_command(
    records: str = typer.Argument(..., help="Path to records file (JSON, JSONL, or CSV)."),
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    prefix: str | None = _prefix_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    format: str | None = typer.Option(
        None, "--format", help="File format override (json, jsonl, csv). Auto-detected by default."
    ),
) -> None:
    """Validate and import records into annotation datasets."""
    from pragmata import annotation

    result = annotation.import_records(
        records,
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        format=format or "auto",
        workspace_prefix=UNSET if prefix is None else prefix,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
    )
    typer.echo(f"Total records: {result.total_records}")
    for ds, count in result.dataset_counts.items():
        typer.echo(f"  {ds}: {count}")
    if result.errors:
        typer.echo(f"Validation errors: {len(result.errors)}", err=True)
        raise typer.Exit(code=1)


@annotation_app.command("export")
def export_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    prefix: str | None = _prefix_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    export_id: str | None = typer.Option(None, "--export-id", help="Export run identifier. Auto-generated if omitted."),
    tasks: str | None = typer.Option(
        None, "--tasks", help="Comma-separated tasks to export (retrieval,grounding,generation)."
    ),
) -> None:
    """Fetch submitted annotations and write flat CSVs per task."""
    from pragmata import annotation

    result = annotation.export_annotations(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        export_id=UNSET if export_id is None else export_id,
        base_dir=UNSET if base_dir is None else base_dir,
        tasks=parse_tasks(tasks),
        workspace_prefix=UNSET if prefix is None else prefix,
        config_path=UNSET if config is None else config,
    )
    for task_name, count in result.row_counts.items():
        typer.echo(f"{task_name}: {count} rows")
    for path in result.files.values():
        typer.echo(f"  {path}")


@annotation_app.command("iaa")
def iaa_command(
    export_id: str = typer.Argument(..., help="Export run identifier whose CSVs to analyse."),
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    prefix: str | None = _prefix_opt,
    tasks: str | None = typer.Option(
        None, "--tasks", help="Comma-separated tasks to analyse (retrieval,grounding,generation)."
    ),
    n_resamples: int = typer.Option(1000, "--n-resamples", help="Bootstrap iterations for confidence intervals."),
    ci: float = typer.Option(0.95, "--ci", help="Confidence level (e.g. 0.95)."),
    seed: int | None = typer.Option(None, "--seed", help="RNG seed for reproducible bootstrap."),
) -> None:
    """Compute inter-annotator agreement from exported CSVs."""
    from pragmata import annotation

    report = annotation.compute_iaa(
        export_id,
        base_dir=UNSET if base_dir is None else base_dir,
        tasks=parse_tasks(tasks),
        workspace_prefix=UNSET if prefix is None else prefix,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
        config_path=UNSET if config is None else config,
    )
    for task_agreement in report.tasks:
        typer.echo(f"\n{task_agreement.task}:")
        for label in task_agreement.labels:
            if label.alpha is None:
                typer.echo(f"  {label.label}: n/a (insufficient overlap)")
            else:
                typer.echo(f"  {label.label}: alpha={label.alpha:.3f} [{label.ci_lower:.3f}, {label.ci_upper:.3f}]")
