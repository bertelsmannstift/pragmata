"""Annotation status API - live per-panel completeness report.

Pure read and config-free: resolves Argilla credentials, then walks the live
retrieval datasets (optionally narrowed by workspace) with no local topology
config. The optional ``--tag-partial-panels`` advisory write ships in a
follow-up PR so the read path carries no Argilla mutation surface.
"""

import logging
import os

from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.panel_status import StatusReport, compute_panel_status, compute_task_progress
from pragmata.core.settings.settings_base import UNSET, Unset, resolve_api_key

logger = logging.getLogger(__name__)


def report_status(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    workspace: str | None = None,
) -> StatusReport:
    """Fetch live retrieval panel status from Argilla (config-free).

    Credential resolution (config-free):
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        workspace: If set, only datasets in this Argilla workspace.

    Returns:
        ``StatusReport`` with the all-task ``progress`` summary plus the
        retrieval per-panel facts.
    """
    url = api_url if isinstance(api_url, str) else os.environ.get("ARGILLA_API_URL")
    key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(url, key)
    progress = compute_task_progress(client, workspace=workspace)
    report = compute_panel_status(client, workspace=workspace).with_progress(progress)
    logger.info(
        "Status: %d panels, %d complete (%.0f%%), %d overlap-satisfied, %d integrity warnings",
        report.n_panels,
        report.n_complete,
        100.0 * report.n_complete / report.n_panels if report.n_panels else 0.0,
        report.n_overlap_satisfied,
        report.n_integrity_warnings,
    )
    return report
