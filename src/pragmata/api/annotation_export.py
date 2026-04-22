"""Annotation export API — fetch submitted responses and write flat CSVs per task."""

import logging
import os
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.export_runner import ExportResult, resolve_export_id, run_export
from pragmata.core.paths.annotation_paths import resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


def export_annotations(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    export_id: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    tasks: list[Task] | None = None,
    dataset_id: str | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    include_discarded: bool = False,
) -> ExportResult:
    """Fetch annotations from Argilla and write flat CSVs per task.

    By default, queries each task dataset for submitted-only responses. Set
    ``include_discarded=True`` to also include responses the annotator
    discarded; their label columns are null and constraint validation is
    skipped, but the row carries ``discard_reason`` and ``discard_notes``.

    Credential resolution:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``argilla.api_url``)
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        export_id: Unique identifier for this export run. Auto-generated from
            dataset_id + ISO timestamp if not supplied.
        base_dir: Workspace base directory for run artifacts. Defaults to cwd.
        tasks: Tasks to export. Defaults to all three tasks.
        dataset_id: Suffix identifying which datasets to export from.
        config_path: Path to YAML config file for settings resolution.
        include_discarded: If True, include discarded responses alongside
            submitted ones. Defaults to False to avoid polluting downstream
            evaluation pipelines.

    Returns:
        ExportResult with file paths, row counts, and constraint summary.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env={"argilla": {"api_url": os.environ.get("ARGILLA_API_URL")}} if os.environ.get("ARGILLA_API_URL") else None,
        overrides={
            "argilla": {"api_url": api_url},
            "dataset_id": dataset_id,
            "base_dir": base_dir,
        },
    )
    api_key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(settings.argilla.api_url, api_key)
    resolved_id = resolve_export_id(settings, export_id if isinstance(export_id, str) else None)
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    export_paths = resolve_export_paths(workspace=workspace, export_id=resolved_id).ensure_dirs()
    resolved_tasks = tasks if tasks is not None else list(Task)

    with error_log(export_paths.tool_root):
        result = run_export(client, settings, export_paths, resolved_tasks, include_discarded=include_discarded)

    logger.info(
        "Export complete: %d task(s), %d total rows",
        len(result.row_counts),
        sum(result.row_counts.values()),
    )
    return result
