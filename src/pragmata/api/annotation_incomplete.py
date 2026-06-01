"""Annotation incomplete API - records still needed to finish each bundle.

Read-only and config-free: resolves Argilla credentials, then walks datasets
(optionally narrowed by workspace/task) for bundles that aren't complete. With
``tag=True`` it additionally stamps the ``needs_completion`` advisory tag - a
live write.
"""

import os

from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.incomplete import IncompleteReport, find_incomplete
from pragmata.core.settings.settings_base import UNSET, Unset, resolve_api_key


def report_incomplete(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    workspace: str | None = None,
    task: str | None = None,
    tag: bool = False,
) -> IncompleteReport:
    """Find bundles still needing annotation; optionally tag the missing records.

    Credential resolution mirrors ``report_status``:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        workspace: If set, only datasets in this workspace.
        task: If set, only datasets for this task (``retrieval``/``grounding``/``generation``).
        tag: If True, stamp ``needs_completion`` on the unresolved records (live write).

    Returns:
        ``IncompleteReport`` listing incomplete bundles + their missing records.
    """
    url = api_url if isinstance(api_url, str) else os.environ.get("ARGILLA_API_URL")
    key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(url, key)
    return find_incomplete(client, workspace=workspace, task=task, tag=tag)
