"""Annotation status API - live per-panel completeness report + optional tag.

Config-free: resolves Argilla credentials, then walks the live retrieval
datasets (optionally narrowed by workspace) with no local topology config.
``tag_partial_panels=True`` additionally stamps the ``needs_completion``
advisory tag on partial panels' unresolved chunks - the one live-write surface,
sharing the same single walk as the read path.
"""

import logging
import os

from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.panel_status import (
    StatusReport,
    _apply_tags,
    _build_report,
    _collect_records,
)
from pragmata.core.settings.settings_base import UNSET, Unset, resolve_api_key

logger = logging.getLogger(__name__)


def report_status(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    workspace: str | None = None,
    tag_partial_panels: bool = False,
) -> StatusReport:
    """Fetch live retrieval panel status from Argilla (config-free).

    Credential resolution (config-free):
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        workspace: If set, only datasets in this Argilla workspace.
        tag_partial_panels: If True, stamp ``needs_completion`` on partial
            panels' unresolved chunks (and clear stale tags). Opt-in live write.

    Returns:
        ``StatusReport`` with per-panel facts and headline totals, plus an
        optional ``tag_result`` populated when ``tag_partial_panels=True``.
    """
    url = api_url if isinstance(api_url, str) else os.environ.get("ARGILLA_API_URL")
    key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(url, key)

    # One walk shared between the read report and the optional tag write.
    collected = _collect_records(client, workspace=workspace)
    report = _build_report(collected)
    if tag_partial_panels:
        report = report.with_tag_result(_apply_tags(collected))

    logger.info(
        "Status: %d panels, %d complete (%.0f%%), %d distribution-satisfied, %d integrity warnings",
        report.n_panels,
        report.n_complete,
        100.0 * report.n_complete / report.n_panels if report.n_panels else 0.0,
        report.n_distribution_satisfied,
        report.n_integrity_warnings,
    )
    return report
