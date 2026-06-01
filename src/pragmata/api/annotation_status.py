"""Annotation status API - live per-panel completeness report.

Pure read. Mirrors ``annotation_export``'s credential resolution. The
optional ``--tag-incomplete`` advisory write (which stamps records for
annotator UI filtering) ships in a follow-up PR so the read path can land
without any Argilla mutation surface.
"""

import logging
import os
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.panel_status import (
    StatusReport,
    _build_report,
    _collect_records,
)
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


def report_status(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    dataset_id: str | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> StatusReport:
    """Fetch live retrieval panel status from Argilla.

    Credential resolution mirrors ``export_annotations``:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``argilla.api_url``)
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        base_dir: Workspace base directory for error-log artifacts.
        dataset_id: Suffix identifying which datasets to inspect.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ``StatusReport`` with per-panel facts and headline totals.
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
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    # error_log appends to a file under tool_root; ensure the dir exists so
    # the FileHandler doesn't itself crash on first error-write.
    tool_root = workspace.tool_root("annotation")
    tool_root.mkdir(parents=True, exist_ok=True)

    with error_log(tool_root):
        collected = _collect_records(client, settings)
        report = _build_report(collected, settings)

    logger.info(
        "Status: %d panels, %d complete (%.0f%%), %d distribution-satisfied, %d integrity warnings",
        report.n_panels,
        report.n_complete,
        100.0 * report.n_complete / report.n_panels if report.n_panels else 0.0,
        report.n_distribution_satisfied,
        report.n_integrity_warnings,
    )
    return report
