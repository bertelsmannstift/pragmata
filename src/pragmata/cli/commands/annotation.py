"""CLI commands for annotation pipeline."""

from collections.abc import Sequence

import typer

from pragmata.api import UNSET
from pragmata.cli.parsing import (
    parse_annotator_ids,
    parse_datetime,
    parse_locale,
    parse_tasks,
    parse_user_specs,
)

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
    calibration_max_items: int | None = typer.Option(
        None,
        "--calibration-max-items",
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
        "or --calibration-max-items.",
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
    if no_calibration and calibration_max_items is not None:
        typer.echo(
            "Error: --no-calibration cannot be combined with --calibration-max-items.",
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
        calibration_max_items=UNSET if calibration_max_items is None else calibration_max_items,
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
        cap = result.calibration_max_items.get(task)
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


def _render_table(headers: list[str], rows: Sequence[tuple[str, ...]], aligns: str) -> list[str]:
    """Aligned fixed-width table as a list of lines.

    ``aligns`` is one char per column: 'l' (left) or 'r' (right).
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(
            cell.rjust(widths[i]) if aligns[i] == "r" else cell.ljust(widths[i]) for i, cell in enumerate(cells)
        ).rstrip()

    return [_fmt(tuple(headers)), *[_fmt(r) for r in rows]]


@annotation_app.command("status")
def status_command(
    api_url: str | None = _api_url_opt,
    api_key: str | None = _api_key_opt,
    workspace: str | None = typer.Option(
        None, "--workspace", help="Only datasets in this Argilla workspace. Default: all workspaces."
    ),
    by_workspace: bool = typer.Option(False, "--by-workspace", help="Add a per-workspace progress breakdown."),
    by_dataset: bool = typer.Option(False, "--by-dataset", help="Add a per-dataset progress breakdown."),
    tag_partial_panels: bool = typer.Option(
        False,
        "--tag-partial-panels",
        help=(
            "Live write: stamp 'needs_completion' on the unresolved chunks of PARTIAL panels "
            "(some but not all chunks annotated) and clear stale tags, so annotators can filter "
            "straight to them in the Argilla UI. Off by default (read-only)."
        ),
    ),
) -> None:
    """Report live annotation progress across all tasks, plus retrieval panel-completeness.

    Config-free: walks every Argilla dataset. Record progress (total / completed)
    is shown per task; the retrieval row also carries panel-completeness. Add
    --by-workspace / --by-dataset for finer breakdowns. With --tag-partial-panels,
    also stamps the 'needs_completion' advisory tag on partial panels' unresolved
    chunks (a live write).
    """
    from pragmata import annotation

    report = annotation.report_status(
        api_url=UNSET if api_url is None else api_url,
        api_key=UNSET if api_key is None else api_key,
        workspace=workspace,
        tag_partial_panels=tag_partial_panels,
    )

    def _num(n: int) -> str:
        return f"{n:,}"

    def _pct(done: int, total: int) -> str:
        return f"{round(100 * done / total)}%" if total else "0%"

    prog = report.progress
    assert prog is not None  # report_status always attaches the all-task progress
    g = prog.grand
    typer.echo(
        f"records: {_num(g.total)} total · {_num(g.completed)} completed "
        f"({_pct(g.completed, g.total)}) · {_num(g.pending)} pending"
    )
    typer.echo("")

    # By task (default). The retrieval row carries panel-completeness; other
    # tasks are single-record and have no panels ("-").
    task_rows: list[tuple[str, ...]] = []
    for row in prog.by_task:
        if row.task == "retrieval":
            panels = (_num(report.n_panels), _num(report.n_complete), _num(report.n_overlap_satisfied))
        else:
            panels = ("–", "–", "–")
        task_rows.append((row.label, _num(row.total), _num(row.completed), _pct(row.completed, row.total), *panels))
    for line in _render_table(
        ["TASK", "TOTAL", "COMPLETED", "%", "PANELS", "PANEL-COMPL", "OVERLAP"], task_rows, "lrrrrrr"
    ):
        typer.echo(line)

    if by_workspace:
        typer.echo("")
        ws_rows = [
            (row.label, row.task, _num(row.total), _num(row.completed), _pct(row.completed, row.total))
            for row in prog.by_workspace
        ]
        for line in _render_table(["WORKSPACE", "TASK", "TOTAL", "COMPL", "%"], ws_rows, "llrrr"):
            typer.echo(line)

    if by_dataset:
        typer.echo("")
        ds_rows = [
            (row.label, _num(row.total), _num(row.completed), _pct(row.completed, row.total)) for row in prog.by_dataset
        ]
        for line in _render_table(["DATASET", "TOTAL", "COMPL", "%"], ds_rows, "lrrr"):
            typer.echo(line)

    if report.n_integrity_warnings:
        typer.echo("")
        typer.echo(f"integrity warnings: {report.n_integrity_warnings} panel(s) (records != n_retrieved_chunks)")
        typer.echo(
            "  note: panel_complete uses live record-count K; the export sidecar uses the metadata K, "
            "so flagged panels may read panel_complete=False in retrieval.csv even if shown complete above."
        )
    if report.n_orphans_skipped:
        typer.echo(f"orphans skipped: {report.n_orphans_skipped} record(s) with empty record_uuid")
    if report.tag_result is not None:
        tr = report.tag_result
        typer.echo(
            f"tag-partial-panels: tagged={tr.n_tagged} cleared={tr.n_cleared} already_tagged={tr.n_already_tagged}"
        )


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
    after: str | None = typer.Option(
        None,
        "--after",
        help="Keep only annotations submitted on or after this ISO 8601 date or datetime "
        "(e.g. 2026-05-01 or 2026-05-01T00:00:00; a date is treated as midnight).",
    ),
    before: str | None = typer.Option(
        None,
        "--before",
        help="Keep only annotations submitted before this ISO 8601 date or datetime.",
    ),
    exclude_annotators: str | None = typer.Option(
        None,
        "--exclude-annotators",
        help="Comma-separated annotator IDs to drop from the analysis.",
    ),
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
        after=parse_datetime(after),
        before=parse_datetime(before),
        exclude_annotators=parse_annotator_ids(exclude_annotators),
        config_path=UNSET if config is None else config,
    )
    for task_agreement in report.tasks:
        typer.echo(f"\n{task_agreement.task}:")
        for label in task_agreement.labels:
            if label.alpha is None:
                typer.echo(f"  {label.label}: n/a (insufficient overlap)")
            else:
                typer.echo(f"  {label.label}: alpha={label.alpha:.3f} [{label.ci_lower:.3f}, {label.ci_upper:.3f}]")
