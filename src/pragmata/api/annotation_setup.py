"""Argilla annotation setup API — thin orchestration over core/ implementation."""

import logging
from pathlib import Path

import argilla as rg

from pragmata.api._error_log import error_log
from pragmata.core.annotation.setup import SetupResult, provision_users, setup_datasets, teardown_resources
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

logger = logging.getLogger(__name__)


def setup(
    client: rg.Argilla,
    users: list[UserSpec] | None = None,
    *,
    workspace_prefix: str | Unset = UNSET,
    min_submitted: int | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> SetupResult:
    """Create the full Argilla annotation environment idempotently.

    Creates workspaces and datasets for all three annotation tasks (retrieval,
    grounding, generation). Optionally provisions user accounts and assigns
    them to workspaces. Existing resources are skipped.

    Settings are resolved from config file and/or keyword overrides. Omitted
    values fall through to config-file defaults, then built-in defaults.

    Args:
        client: Connected Argilla client instance.
        users: User accounts to provision. Pass None to skip user setup.
        workspace_prefix: Prefix prepended to workspace and dataset names.
        min_submitted: Minimum annotations required per record.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        SetupResult tracking created/skipped workspaces, datasets, and users.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={
            "workspace_prefix": workspace_prefix,
            "min_submitted": min_submitted,
            "base_dir": base_dir,
        },
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    with error_log(workspace.tool_root("annotation")):
        ds_result = setup_datasets(client, settings)
        user_result = provision_users(client, users or [], settings)
    merged = ds_result.merge(user_result)
    logger.info(
        "Setup complete: %d workspaces, %d datasets, %d users created",
        len(merged.created_workspaces),
        len(merged.created_datasets),
        len(merged.created_users),
    )
    return merged


def teardown(
    client: rg.Argilla,
    *,
    workspace_prefix: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> None:
    """Remove the Argilla annotation environment.

    Deletes datasets first (Argilla requires empty workspaces), then
    workspaces. Missing resources are silently skipped. User accounts
    are not touched.

    Args:
        client: Connected Argilla client instance.
        workspace_prefix: Prefix used when the environment was created.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix, "base_dir": base_dir},
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    logger.info("Starting teardown (prefix=%r)", settings.workspace_prefix)
    with error_log(workspace.tool_root("annotation")):
        teardown_resources(client, settings)
    logger.info("Teardown complete")
