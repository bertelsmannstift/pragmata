"""Annotation export API — fetch submitted responses and write flat CSVs per task."""

from datetime import UTC, datetime
from pathlib import Path

import argilla as rg

from pragmata.core.annotation.export_helpers import ExportResult, run_export
from pragmata.core.paths.annotation_paths import resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file


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

    return run_export(client, settings, paths, resolved_tasks)
