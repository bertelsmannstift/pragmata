"""Argilla annotation setup API — thin orchestration over core/ implementation."""

import logging
from pathlib import Path

import argilla as rg

from pragmata.api._error_log import error_log
from pragmata.core.annotation.setup import SetupResult, provision_users, setup_workspaces, teardown_resources
from pragmata.core.paths.annotation_paths import resolve_annotation_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

logger = logging.getLogger(__name__)


def setup(
    client: rg.Argilla,
    users: list[UserSpec] | None = None,
    *,
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

    Args:
        client: Connected Argilla client instance.
        users: User accounts to provision. Pass None to skip user setup.
        min_submitted: Minimum annotations required per record.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        SetupResult tracking created/skipped workspaces and users.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"min_submitted": min_submitted, "base_dir": base_dir},
    )
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
    client: rg.Argilla,
    *,
    dataset_id: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> None:
    """Remove the Argilla annotation environment.

    When dataset_id is set, only datasets matching that suffix are deleted
    and workspaces are left intact. When dataset_id is empty, all default
    datasets and workspaces are deleted.

    Deletes datasets first (Argilla requires empty workspaces), then
    workspaces. Missing resources are silently skipped. User accounts
    are not touched.

    Args:
        client: Connected Argilla client instance.
        dataset_id: Suffix identifying which datasets to delete.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"dataset_id": dataset_id, "base_dir": base_dir},
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    paths = resolve_annotation_paths(workspace=workspace).ensure_dirs()
    logger.info("Starting teardown (dataset_id=%r)", settings.dataset_id)
    with error_log(paths.tool_root):
        teardown_resources(client, settings)
    logger.info("Teardown complete")
