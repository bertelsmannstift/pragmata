"""Annotation setup implementation — workspace and user provisioning.

All Argilla SDK interaction for setup/teardown lives here. The api/ layer
resolves settings and delegates to these functions.
"""

import logging
from dataclasses import dataclass, field

import argilla as rg

from pragmata.core.annotation.argilla_ops import apply_suffix, create_user, create_workspace
from pragmata.core.annotation.argilla_task_definitions import DATASET_NAMES
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec

logger = logging.getLogger(__name__)


@dataclass
class SetupResult:
    """Tracks resources created and skipped during a setup or provision call.

    Attributes:
        created_workspaces: Names of newly created workspaces.
        skipped_workspaces: Names of workspaces that already existed.
        created_users: Usernames of newly created accounts.
        skipped_users: Usernames of accounts that already existed.
        generated_passwords: Mapping of username to auto-generated password
            for newly created accounts (only present when no password was
            supplied in the UserSpec).
    """

    created_workspaces: list[str] = field(default_factory=list)
    skipped_workspaces: list[str] = field(default_factory=list)
    created_users: list[str] = field(default_factory=list)
    skipped_users: list[str] = field(default_factory=list)
    generated_passwords: dict[str, str] = field(default_factory=dict)

    def merge(self, other: "SetupResult") -> "SetupResult":
        """Combine two results (e.g. workspace setup + user provisioning)."""
        return SetupResult(
            created_workspaces=self.created_workspaces + other.created_workspaces,
            skipped_workspaces=self.skipped_workspaces + other.skipped_workspaces,
            created_users=self.created_users + other.created_users,
            skipped_users=self.skipped_users + other.skipped_users,
            generated_passwords={**self.generated_passwords, **other.generated_passwords},
        )


def setup_workspaces(
    client: rg.Argilla,
    settings: AnnotationSettings,
) -> SetupResult:
    """Create all workspaces idempotently per settings topology."""
    result = SetupResult()
    for ws_base in settings.workspace_dataset_map:
        workspace, created = create_workspace(client, ws_base)
        (result.created_workspaces if created else result.skipped_workspaces).append(ws_base)
    return result


def provision_users(
    client: rg.Argilla,
    users: list[UserSpec],
    settings: AnnotationSettings,
) -> SetupResult:
    """Create user accounts and assign to workspaces idempotently."""
    result = SetupResult()

    for spec in users:
        user, generated_pw, created = create_user(client, spec)
        (result.created_users if created else result.skipped_users).append(spec.username)
        if generated_pw is not None:
            result.generated_passwords[spec.username] = generated_pw

        for ws_base in spec.workspaces:
            workspace = client.workspaces(ws_base)
            if workspace is None:
                logger.warning("Workspace %r not found when assigning user %r", ws_base, spec.username)
            elif user not in workspace.users:
                workspace.add_user(user)

    return result


def teardown_resources(
    client: rg.Argilla,
    settings: AnnotationSettings,
) -> None:
    """Delete datasets and (optionally) workspaces for the annotation environment.

    When dataset_id is set, only datasets matching that suffix are deleted and
    workspaces are left intact. When dataset_id is empty, all default datasets
    and workspaces are deleted.

    Ordering: datasets first (Argilla requires workspace to be empty before deletion).
    Missing resources are silently skipped. User accounts are not touched.
    """
    for ws_base, tasks in settings.workspace_dataset_map.items():
        workspace = client.workspaces(ws_base)
        if workspace is None:
            logger.info("Workspace %r not found — skipping", ws_base)
            continue

        for task in tasks:
            ds_name = apply_suffix(DATASET_NAMES[task], settings.dataset_id)
            dataset = client.datasets(ds_name, workspace=ws_base)
            if dataset is not None:
                dataset.delete()
                logger.info("Deleted dataset %r from workspace %r", ds_name, ws_base)

        if not settings.dataset_id:
            for user in list(workspace.users):
                workspace.remove_user(user)
            workspace.delete()
            logger.info("Deleted workspace %r", ws_base)
