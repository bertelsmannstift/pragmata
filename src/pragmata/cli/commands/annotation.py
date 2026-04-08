"""CLI commands for annotation pipeline."""

from pathlib import Path

import argilla as rg
import typer

from pragmata import annotation
from pragmata.api import UNSET
from pragmata.core.schemas.annotation_task import Task

annotation_app = typer.Typer(help="Annotation pipeline commands.")


def _client(api_url: str | None, api_key: str | None) -> rg.Argilla:
    """Build an Argilla client from explicit args or environment defaults."""
    kwargs: dict = {}
    if api_url is not None:
        kwargs["api_url"] = api_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    return rg.Argilla(**kwargs)


def _parse_tasks(raw: str | None) -> list[Task] | None:
    """Parse comma-separated task names, or None for all tasks."""
    if raw is None:
        return None
    return [Task(t.strip()) for t in raw.split(",")]


# -- shared options ----------------------------------------------------------

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
    import json

    users = None
    if users_json is not None:
        raw = json.loads(Path(users_json).read_text())
        users = [annotation.UserSpec(**u) for u in raw]

    result = annotation.setup(
        _client(api_url, api_key),
        users,
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
    annotation.teardown(
        _client(api_url, api_key),
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
    result = annotation.import_records(
        _client(api_url, api_key),
        records,
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
    result = annotation.export_annotations(
        _client(api_url, api_key),
        export_id=UNSET if export_id is None else export_id,
        base_dir=UNSET if base_dir is None else base_dir,
        tasks=_parse_tasks(tasks),
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
    report = annotation.compute_iaa(
        export_id,
        base_dir=UNSET if base_dir is None else base_dir,
        tasks=_parse_tasks(tasks),
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
