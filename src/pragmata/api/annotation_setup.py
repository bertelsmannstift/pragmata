"""Argilla annotation setup API — thin orchestration over core/ implementation."""

import logging
import os
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.setup import SetupResult, provision_users, setup_workspaces, teardown_resources
from pragmata.core.paths.annotation_paths import resolve_annotation_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


def setup(
    users: list[UserSpec] | None = None,
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    min_submitted: int | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> SetupResult:
    """Create the Argilla annotation environment idempotently.

    Creates workspaces for all three annotation tasks (retrieval, grounding,
    generation). Optionally provisions user accounts and assigns them to
    workspaces. Existing resources are skipped.

    Datasets are not created here — they are auto-created on import,
    scoped by dataset_id.

    Settings are resolved from config file and/or keyword overrides. Omitted
    values fall through to config-file defaults, then built-in defaults.

    Credential resolution:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``argilla.api_url``)
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        users: User accounts to provision. Pass None to skip user setup.
        api_url: Argilla server URL.
        api_key: Argilla API key.
        min_submitted: Minimum annotations required per record.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        SetupResult tracking created/skipped workspaces and users.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env={"argilla": {"api_url": os.environ.get("ARGILLA_API_URL")}} if os.environ.get("ARGILLA_API_URL") else None,
        overrides={
            "argilla": {"api_url": api_url},
            "min_submitted": min_submitted,
            "base_dir": base_dir,
        },
    )
    api_key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(settings.argilla.api_url, api_key)
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    paths = resolve_annotation_paths(workspace=workspace).ensure_dirs()
    with error_log(paths.tool_root):
        ws_result = setup_workspaces(client, settings)
        user_result = provision_users(client, users or [], settings)
    merged = ws_result.merge(user_result)
    logger.info(
        "Setup complete: %d workspaces, %d users created",
        len(merged.created_workspaces),
        len(merged.created_users),
    )
    return merged


def teardown(
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    dataset_id: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> None:
    """Remove the Argilla annotation environment.

    See :func:`pragmata.core.annotation.setup.teardown_resources` for deletion
    semantics (dataset_id scoping, ordering, idempotency).

    Credential resolution:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``argilla.api_url``)
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)

    Args:
        api_url: Argilla server URL.
        api_key: Argilla API key.
        dataset_id: Suffix identifying which datasets to delete. Empty tears
            down workspaces too.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.
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
    paths = resolve_annotation_paths(workspace=workspace).ensure_dirs()
    logger.info("Starting teardown (dataset_id=%r)", settings.dataset_id)
    with error_log(paths.tool_root):
        teardown_resources(client, settings)
    logger.info("Teardown complete")
