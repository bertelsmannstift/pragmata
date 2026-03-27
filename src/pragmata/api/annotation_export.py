"""Annotation export API — fetch submitted responses and write flat CSVs per task."""

from datetime import UTC, datetime
from pathlib import Path

import argilla as rg

from pragmata.core.annotation.export_fetcher import AnnotationModel, build_user_lookup, fetch_task
from pragmata.core.annotation.export_helpers import ExportResult, write_export_csv
from pragmata.core.paths.annotation_paths import AnnotationExportPaths, resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

_TASK_CSV_ATTR = {
    Task.RETRIEVAL: "retrieval_annotation_csv",
    Task.GROUNDING: "grounding_annotation_csv",
    Task.GENERATION: "generation_annotation_csv",
}


def _run_export(
    client: rg.Argilla,
    settings: AnnotationSettings,
    paths: AnnotationExportPaths,
    tasks: list[Task],
) -> ExportResult:
    """Fetch all tasks, write CSVs atomically, return ExportResult."""
    user_lookup = build_user_lookup(client)

    task_rows: dict[Task, list[tuple[AnnotationModel, list[str]]]] = {}
    for task in tasks:
        task_rows[task] = fetch_task(client, settings, task, user_lookup)

    task_paths = {task: getattr(paths, _TASK_CSV_ATTR[task]) for task in tasks}

    written: list[Path] = []
    try:
        for task in tasks:
            write_export_csv(task_rows[task], task_paths[task], task)
            written.append(task_paths[task])
    except Exception:
        for p in written:
            p.unlink(missing_ok=True)
        raise

    row_counts = {task: len(task_rows[task]) for task in tasks}

    constraint_summary: dict[str, int] = {}
    for task in tasks:
        for _, violations in task_rows[task]:
            for v in violations:
                constraint_summary[v] = constraint_summary.get(v, 0) + 1

    return ExportResult(
        paths=paths,
        files=task_paths,
        row_counts=row_counts,
        constraint_summary=constraint_summary,
    )


def export_annotations(
    client: rg.Argilla,
    workspace: WorkspacePaths,
    *,
    export_id: str | Unset = UNSET,
    tasks: list[Task] | None = None,
    workspace_prefix: str | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> ExportResult:
    """Fetch submitted annotations from Argilla and write flat CSVs per task.

    Queries each task dataset for submitted-only responses, groups by annotator,
    applies constraint validation, and writes atomic CSVs.

    Args:
        client: Connected Argilla client.
        workspace: Workspace path bundle.
        export_id: Unique identifier for this export run. Auto-generated from
            prefix + ISO timestamp if not supplied.
        tasks: Tasks to export. Defaults to all three tasks.
        workspace_prefix: Prefix used when the environment was created.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ExportResult with file paths, row counts, and constraint summary.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix},
    )

    if export_id is UNSET:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        prefix = settings.workspace_prefix
        resolved_id = f"{prefix}_{ts}" if prefix else ts
    else:
        resolved_id = str(export_id)

    paths = resolve_export_paths(workspace=workspace, export_id=resolved_id).ensure_dirs()
    resolved_tasks = tasks if tasks is not None else list(Task)

    return _run_export(client, settings, paths, resolved_tasks)
