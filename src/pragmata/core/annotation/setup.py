"""Annotation setup implementation — workspace, dataset, and user provisioning.

All Argilla SDK interaction for setup/teardown lives here. The api/ layer
resolves settings and delegates to these functions.
"""

import logging
from dataclasses import dataclass, field

import argilla as rg

from pragmata.core.annotation.argilla_ops import apply_prefix, create_dataset, create_user, create_workspace
from pragmata.core.annotation.argilla_task_definitions import DATASET_NAMES, build_task_settings
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec

logger = logging.getLogger(__name__)


@dataclass
class SetupResult:
    """Tracks resources created and skipped during a setup or provision call.

    Attributes:
        created_workspaces: Names of newly created workspaces.
        skipped_workspaces: Names of workspaces that already existed.
        created_datasets: Names of newly created datasets.
        skipped_datasets: Names of datasets that already existed.
        created_users: Usernames of newly created accounts.
        skipped_users: Usernames of accounts that already existed.
        generated_passwords: Mapping of username to auto-generated password
            for newly created accounts (only present when no password was
            supplied in the UserSpec).
    """

    created_workspaces: list[str] = field(default_factory=list)
    skipped_workspaces: list[str] = field(default_factory=list)
    created_datasets: list[str] = field(default_factory=list)
    skipped_datasets: list[str] = field(default_factory=list)
    created_users: list[str] = field(default_factory=list)
    skipped_users: list[str] = field(default_factory=list)
    generated_passwords: dict[str, str] = field(default_factory=dict)

    def merge(self, other: "SetupResult") -> "SetupResult":
        """Combine two results (e.g. dataset setup + user provisioning)."""
        return SetupResult(
            created_workspaces=self.created_workspaces + other.created_workspaces,
            skipped_workspaces=self.skipped_workspaces + other.skipped_workspaces,
            created_datasets=self.created_datasets + other.created_datasets,
            skipped_datasets=self.skipped_datasets + other.skipped_datasets,
            created_users=self.created_users + other.created_users,
            skipped_users=self.skipped_users + other.skipped_users,
            generated_passwords={**self.generated_passwords, **other.generated_passwords},
        )


def setup_datasets(
    client: rg.Argilla,
    settings: AnnotationSettings,
) -> SetupResult:
    """Create all workspaces and datasets idempotently per settings topology."""
    result = SetupResult()
    task_settings_map = build_task_settings()

    for ws_base, tasks in settings.workspace_dataset_map.items():
        ws_name = apply_prefix(settings.workspace_prefix, ws_base)
        workspace, created = create_workspace(client, ws_name)
        (result.created_workspaces if created else result.skipped_workspaces).append(ws_name)

        for task in tasks:
            ds_base = DATASET_NAMES[task]
            ds_name = apply_prefix(settings.workspace_prefix, ds_base)
            base_settings = task_settings_map[task]

            # Never mutate the cached constant — construct fresh Settings
            task_settings = rg.Settings(
                fields=base_settings.fields,
                questions=base_settings.questions,
                metadata=base_settings.metadata,
                guidelines=base_settings.guidelines,
                distribution=rg.TaskDistribution(min_submitted=settings.min_submitted),
            )
            _, ds_created = create_dataset(client, ds_name, ws_name, task_settings)
            (result.created_datasets if ds_created else result.skipped_datasets).append(ds_name)

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
            ws_name = apply_prefix(settings.workspace_prefix, ws_base)
            workspace = client.workspaces(ws_name)
            if workspace is None:
                logger.warning("Workspace %r not found when assigning user %r", ws_name, spec.username)
            elif user not in workspace.users:
                workspace.add_user(user)

    return result


def teardown_resources(
    client: rg.Argilla,
    settings: AnnotationSettings,
) -> None:
    """Delete datasets and workspaces for the annotation environment.

    Ordering: datasets first (Argilla requires workspace to be empty before deletion).
    Missing resources are silently skipped. User accounts are not touched.
    """
    for ws_base, tasks in settings.workspace_dataset_map.items():
        ws_name = apply_prefix(settings.workspace_prefix, ws_base)
        workspace = client.workspaces(ws_name)
        if workspace is None:
            logger.info("Workspace %r not found — skipping", ws_name)
            continue

        for task in tasks:
            ds_base = DATASET_NAMES[task]
            ds_name = apply_prefix(settings.workspace_prefix, ds_base)
            dataset = client.datasets(ds_name, workspace=ws_name)
            if dataset is not None:
                dataset.delete()
                logger.info("Deleted dataset %r from workspace %r", ds_name, ws_name)

        workspace.delete()
        logger.info("Deleted workspace %r", ws_name)
