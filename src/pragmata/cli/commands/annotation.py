"""CLI commands for annotation pipeline."""

import typer

from pragmata.api import UNSET
from pragmata.cli.parsing import parse_locale, parse_tasks, parse_user_specs

annotation_app = typer.Typer(help="Annotation pipeline commands.")

_api_url_opt = typer.Option(None, "--api-url", help="Argilla server URL. Falls back to ARGILLA_API_URL env var.")
_api_key_opt = typer.Option(None, "--api-key", help="Argilla API key. Falls back to ARGILLA_API_KEY env var.")
_dataset_id_opt = typer.Option(
    None, "--dataset-id", help="Suffix appended to dataset names for run scoping (e.g. 'run1')."
)
_base_dir_opt = typer.Option(
    None, "--base-dir", help="Workspace base directory for run artifacts. Defaults to current working directory."
)
_config_opt = typer.Option(None, "--config", help="Path to YAML config file for annotation settings.")


@annotation_app.command("setup")
def setup_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    users_json: str | None = typer.Option(
        None,
        "--users",
        help="Path to a JSON file containing user specs. Each entry needs 'username' and 'role' (owner/annotator);"
        " 'workspaces' and 'password' are optional.",
    ),
) -> None:
    """Create Argilla workspaces and (optionally) user accounts.

    Datasets are created automatically on import, not here. Per-task overlap
    (production and calibration ``min_submitted``) is configured via
    ``workspaces`` in the YAML config (``--config``).
    """
    from pragmata import annotation

    result = annotation.setup(
        parse_user_specs(users_json),
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
    )
    typer.echo(f"Workspaces created: {len(result.created_workspaces)}, skipped: {len(result.skipped_workspaces)}")
    typer.echo(f"Users created: {len(result.created_users)}, skipped: {len(result.skipped_users)}")
    if result.generated_passwords:
        typer.echo("\nGenerated passwords (shown once — record them now):")
        for username, password in result.generated_passwords.items():
            typer.echo(f"  {username}: {password}")


@annotation_app.command("teardown")
def teardown_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    dataset_id: str | None = _dataset_id_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
) -> None:
    """Remove Argilla datasets and (optionally) workspaces.

    With --dataset-id, only datasets matching that suffix are deleted.
    Without it, all default datasets and workspaces are removed.
    """
    from pragmata import annotation

    annotation.teardown(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        dataset_id=UNSET if dataset_id is None else dataset_id,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
    )
    typer.echo("Teardown complete.")


@annotation_app.command("import")
def import_command(
    records: str = typer.Argument(..., help="Path to records file (JSON, JSONL, or CSV)."),
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    dataset_id: str | None = _dataset_id_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    format: str | None = typer.Option(
        None, "--format", help="File format override (json, jsonl, csv). Auto-detected by default."
    ),
    calibration_fraction: float | None = typer.Option(
        None,
        "--calibration-fraction",
        help="Deployment-level fraction of annotation items routed to the calibration "
        "dataset for this batch (inherited by workspaces/tasks unless overridden in "
        "YAML config). Falls through to YAML config and built-in default (0.1) when "
        "omitted; set to 0.0 for production-only batches.",
    ),
    calibration_max_records: int | None = typer.Option(
        None,
        "--calibration-max-records",
        help="Deployment-level absolute cap on calibration annotation items per task "
        "(inherited by workspaces/tasks unless overridden in YAML config). Smaller of "
        "(fraction × N_items, cap) wins. Existing assignments are never demoted. "
        "Cap unit is the annotation item: chunks for retrieval, records for grounding "
        "and generation. Omit to leave uncapped.",
    ),
    no_calibration: bool = typer.Option(
        False,
        "--no-calibration",
        help="Disable calibration entirely for this batch: sets calibration_min_submitted=None "
        "and calibration_fraction=0.0. Cannot be combined with --calibration-fraction > 0 "
        "or --calibration-max-records.",
    ),
    calibration_partition_seed: int | None = typer.Option(
        None,
        "--calibration-partition-seed",
        help="Deterministic seed for assigning new annotation items to calibration vs "
        "production. Existing assignments are locked by the partition manifest.",
    ),
    locale: str | None = typer.Option(
        None,
        "--locale",
        help="Deployment-level UI locale for Argilla dataset titles/questions/guidelines "
        "(en, de). Inherits to workspaces/tasks unless they carve out a value in YAML. "
        "Falls back to config, then 'en'.",
    ),
    locale_catalog_dir: str | None = typer.Option(
        None,
        "--locale-catalog",
        help="Directory of user-provided locale YAML files layered over bundled "
        "catalogs (user wins on stem collision). Must exist if set. "
        "Falls back to config.",
    ),
) -> None:
    """Validate and import records into annotation datasets.

    Datasets are auto-created if they don't exist. Records are partitioned
    deterministically into calibration vs production buckets; the partition
    is locked across re-imports via a sidecar manifest scoped to ``dataset_id``.
    """
    from pragmata import annotation

    if no_calibration and calibration_fraction is not None and calibration_fraction > 0:
        typer.echo(
            "Error: --no-calibration cannot be combined with --calibration-fraction > 0.",
            err=True,
        )
        raise typer.Exit(code=2)
    if no_calibration and calibration_max_records is not None:
        typer.echo(
            "Error: --no-calibration cannot be combined with --calibration-max-records.",
            err=True,
        )
        raise typer.Exit(code=2)

    result = annotation.import_records(
        records,
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        format=format or "auto",
        dataset_id=UNSET if dataset_id is None else dataset_id,
        base_dir=UNSET if base_dir is None else base_dir,
        config_path=UNSET if config is None else config,
        calibration_fraction=(
            0.0 if no_calibration else UNSET if calibration_fraction is None else calibration_fraction
        ),
        calibration_max_records=UNSET if calibration_max_records is None else calibration_max_records,
        calibration_min_submitted=None if no_calibration else UNSET,
        calibration_partition_seed=UNSET if calibration_partition_seed is None else calibration_partition_seed,
        locale=parse_locale(locale) or UNSET,
        locale_catalog_dir=UNSET if locale_catalog_dir is None else locale_catalog_dir,
    )
    typer.echo(f"Total records: {result.total_records}")
    for task, cal_n in result.calibration_count.items():
        prod_n = result.production_count.get(task, 0)
        cfg_fraction = result.calibration_fraction.get(task, 0.0)
        realised = result.realised_calibration_fraction.get(task, 0.0)
        cap = result.calibration_max_records.get(task)
        cap_str = f", cap={cap}" if cap is not None else ""
        typer.echo(
            f"  {task.value}: calibration={cal_n}, production={prod_n} "
            f"(configured={cfg_fraction:.3f}, realised={realised:.3f}{cap_str})"
        )
    for ds, count in result.dataset_counts.items():
        typer.echo(f"  {ds}: {count}")
    if result.errors:
        typer.echo(f"Validation errors: {len(result.errors)}", err=True)
        raise typer.Exit(code=1)


@annotation_app.command("export")
def export_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    dataset_id: str | None = _dataset_id_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    export_id: str | None = typer.Option(None, "--export-id", help="Export run identifier. Auto-generated if omitted."),
    tasks: str | None = typer.Option(
        None, "--tasks", help="Comma-separated tasks to export (retrieval,grounding,generation)."
    ),
    include_discarded: bool = typer.Option(
        False,
        "--include-discarded",
        help="Include responses the annotator discarded. Off by default to keep eval pipelines clean.",
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
        dataset_id=UNSET if dataset_id is None else dataset_id,
        config_path=UNSET if config is None else config,
        include_discarded=UNSET if not include_discarded else True,
    )
    for task_name, count in result.row_counts.items():
        typer.echo(f"{task_name}: {count} rows")
    for path in result.files.values():
        typer.echo(f"  {path}")


@annotation_app.command("status")
def status_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    dataset_id: str | None = _dataset_id_opt,
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
    tag_incomplete: bool = typer.Option(
        False,
        "--tag-incomplete",
        help=(
            "Stamp a 'needs_completion' metadata tag on incomplete unresolved retrieval chunks (and "
            "idempotently clear stale tags). The tag is visible to annotators in the Argilla UI for filtering."
        ),
    ),
) -> None:
    """Report live retrieval panel-completeness across prod + cal datasets."""
    from pragmata import annotation

    report = annotation.report_status(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        base_dir=UNSET if base_dir is None else base_dir,
        dataset_id=UNSET if dataset_id is None else dataset_id,
        config_path=UNSET if config is None else config,
        tag_incomplete=tag_incomplete,
    )
    h = report.headline
    typer.echo(f"records: total={h.total} completed={h.completed} pending={h.pending}")
    pct = (100.0 * report.n_complete / report.n_panels) if report.n_panels else 0.0
    typer.echo(
        f"panels: {report.n_panels} ({report.n_complete} complete = {pct:.1f}%, "
        f"{report.n_distribution_satisfied} distribution-satisfied)"
    )
    if report.n_integrity_warnings:
        typer.echo(f"integrity warnings: {report.n_integrity_warnings} panel(s) (records != n_retrieved_chunks)")
        typer.echo(
            "  note: panel_complete here uses live record-count K; the export sidecar uses the metadata K, "
            "so flagged panels may read panel_complete=False in retrieval.csv even if shown complete above."
        )
    if report.n_orphans_skipped:
        typer.echo(f"orphans skipped: {report.n_orphans_skipped} record(s) with empty record_uuid")
    if report.tag_result is not None:
        tr = report.tag_result
        typer.echo(f"tag-incomplete: tagged={tr.n_tagged} cleared={tr.n_cleared} already_tagged={tr.n_already_tagged}")


@annotation_app.command("iaa")
def iaa_command(
    export_id: str = typer.Argument(..., help="Export run identifier whose CSVs to analyse."),
    base_dir: str | None = _base_dir_opt,
    config: str | None = _config_opt,
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


@annotation_app.command("incomplete")
def incomplete_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    workspace: str | None = typer.Option(
        None, "--workspace", help="Only datasets in this Argilla workspace. Default: all workspaces."
    ),
    task: str | None = typer.Option(
        None, "--task", help="Only this task: retrieval | grounding | generation. Default: all tasks."
    ),
    tag: bool = typer.Option(
        False,
        "--tag",
        help=(
            "Live write: stamp 'needs_completion' on the listed records (and clear stale tags) so "
            "annotators can filter straight to them in the Argilla UI. Off by default (read-only)."
        ),
    ),
) -> None:
    """List records still needed to complete their bundle; --tag steers annotators to them.

    A bundle is all Argilla records sharing a record_uuid (K chunk-records for
    retrieval, a single record for generation/grounding). Read-only unless --tag.
    """
    from pragmata import annotation

    report = annotation.report_incomplete(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        workspace=workspace,
        task=task,
        tag=tag,
    )
    if not report.bundles:
        typer.echo("no partially-complete query-bundles in scope")
    else:
        for b in report.bundles:
            typer.echo(
                f"{b.workspace}/{b.dataset}  {b.record_uuid}  "
                f"{b.n_submitted}/{b.n_records} done, {len(b.missing_record_ids)} to finish"
            )
        typer.echo("")
        typer.echo(
            f"TOTAL: {report.n_bundles} incomplete query-bundle(s) in {report.n_domains} domain(s) "
            f"({', '.join(report.tasks)}), {report.n_records} record(s) need completion"
        )
        if not report.tagged:
            typer.echo(
                "Run 'pragmata annotation incomplete --tag' to flag these records so "
                "annotators can filter to them in the Argilla UI."
            )
    if report.tagged:
        typer.echo(
            f"tagged={report.n_tagged} cleared={report.n_cleared} already_tagged={report.n_already_tagged}"
        )
