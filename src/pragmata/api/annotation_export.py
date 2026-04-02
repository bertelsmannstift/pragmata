"""Annotation export API — fetch submitted responses and write flat CSVs per task."""

import logging
from pathlib import Path

import argilla as rg

from pragmata.api._error_log import error_log
from pragmata.core.annotation.export_runner import ExportResult, resolve_export_id, run_export
from pragmata.core.paths.annotation_paths import resolve_annotation_paths, resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

logger = logging.getLogger(__name__)


def export_annotations(
    client: rg.Argilla,
    *,
    export_id: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    tasks: list[Task] | None = None,
    workspace_prefix: str | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> ExportResult:
    """Fetch submitted annotations from Argilla and write flat CSVs per task.

    Queries each task dataset for submitted-only responses, groups by annotator,
    applies constraint validation, and writes atomic CSVs.

    Args:
        client: Connected Argilla client.
        export_id: Unique identifier for this export run. Auto-generated from
            prefix + ISO timestamp if not supplied.
        base_dir: Workspace base directory for run artifacts. Defaults to cwd.
        tasks: Tasks to export. Defaults to all three tasks.
        workspace_prefix: Prefix used when the environment was created.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ExportResult with file paths, row counts, and constraint summary.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix, "base_dir": base_dir},
    )
    resolved_id = resolve_export_id(settings, export_id if isinstance(export_id, str) else None)
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    annotation_paths = resolve_annotation_paths(workspace=workspace).ensure_dirs()
    export_paths = resolve_export_paths(workspace=workspace, export_id=resolved_id).ensure_dirs()
    resolved_tasks = tasks if tasks is not None else list(Task)

    with error_log(annotation_paths.tool_root):
        result = run_export(client, settings, export_paths, resolved_tasks)

    logger.info(
        "Export complete: %d task(s), %d total rows",
        len(result.row_counts),
        sum(result.row_counts.values()),
    )
    return result
