"""Argilla annotation setup API — thin orchestration over core/ implementation."""

from pathlib import Path

import argilla as rg

from pragmata.core.annotation.setup import SetupResult, provision_users, setup_datasets, teardown_resources
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file


def setup(
    client: rg.Argilla,
    users: list[UserSpec] | None = None,
    *,
    workspace_prefix: str | Unset = UNSET,
    min_submitted: int | Unset = UNSET,
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
        config_path: Path to YAML config file for settings resolution.

    Returns:
        SetupResult tracking created/skipped workspaces, datasets, and users.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={
            "workspace_prefix": workspace_prefix,
            "min_submitted": min_submitted,
        },
    )
    ds_result = setup_datasets(client, settings)
    user_result = provision_users(client, users or [], settings)
    return ds_result.merge(user_result)


def teardown(
    client: rg.Argilla,
    *,
    workspace_prefix: str | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> None:
    """Remove the Argilla annotation environment.

    Deletes datasets first (Argilla requires empty workspaces), then
    workspaces. Missing resources are silently skipped. User accounts
    are not touched.

    Args:
        client: Connected Argilla client instance.
        workspace_prefix: Prefix used when the environment was created.
        config_path: Path to YAML config file for settings resolution.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix},
    )
    teardown_resources(client, settings)
