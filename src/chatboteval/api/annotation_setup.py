"""Argilla annotation setup API — workspaces, datasets, users, teardown.

Public API:
    setup_datasets(client, settings=None) -> SetupResult
    provision_users(client, users, settings=None) -> SetupResult
    setup(client, settings=None, users=None) -> SetupResult
    teardown(client, settings=None, *, include_users=False) -> None
"""

import logging
import secrets
import string
from dataclasses import dataclass, field

import argilla as rg

from chatboteval.api.annotation_task_config import DATASET_NAMES, TASK_SETTINGS
from chatboteval.core.settings.annotation_settings import AnnotationSetupSettings, UserSpec

logger = logging.getLogger(__name__)

_PASSWORD_CHARS = string.ascii_letters + string.digits + "!@#$%"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SetupResult:
    """Tracks resources created and skipped during a setup or provision call."""

    created_workspaces: list[str] = field(default_factory=list)
    skipped_workspaces: list[str] = field(default_factory=list)
    created_datasets: list[str] = field(default_factory=list)
    skipped_datasets: list[str] = field(default_factory=list)
    created_users: list[str] = field(default_factory=list)
    skipped_users: list[str] = field(default_factory=list)
    generated_passwords: dict[str, str] = field(default_factory=dict)

    def _merge(self, other: "SetupResult") -> "SetupResult":
        return SetupResult(
            created_workspaces=self.created_workspaces + other.created_workspaces,
            skipped_workspaces=self.skipped_workspaces + other.skipped_workspaces,
            created_datasets=self.created_datasets + other.created_datasets,
            skipped_datasets=self.skipped_datasets + other.skipped_datasets,
            created_users=self.created_users + other.created_users,
            skipped_users=self.skipped_users + other.skipped_users,
            generated_passwords={**self.generated_passwords, **other.generated_passwords},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_prefix(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _generate_password(length: int = 16) -> str:
    return "".join(secrets.choice(_PASSWORD_CHARS) for _ in range(length))


def _create_workspace(client: rg.Argilla, name: str) -> tuple[rg.Workspace, bool]:
    existing = client.workspaces(name)
    if existing is not None:
        logger.warning("Workspace %r already exists — skipping", name)
        return existing, False
    ws = rg.Workspace(name=name, client=client)
    ws.create()
    return ws, True


def _create_dataset(
    client: rg.Argilla,
    name: str,
    workspace: str,
    settings: rg.Settings,
) -> tuple[rg.Dataset, bool]:
    existing = client.datasets(name, workspace=workspace)
    if existing is not None:
        logger.warning("Dataset %r in workspace %r already exists — skipping", name, workspace)
        return existing, False
    ds = rg.Dataset(name=name, workspace=workspace, settings=settings, client=client)
    ds.create()
    return ds, True


def _create_user(
    client: rg.Argilla,
    spec: UserSpec,
    prefix: str,
) -> tuple[rg.User, str | None, bool]:
    existing = client.users(spec.username)
    if existing is not None:
        logger.warning("User %r already exists — skipping", spec.username)
        return existing, None, False
    password = spec.password if spec.password is not None else _generate_password()
    user = rg.User(username=spec.username, role=spec.role, password=password, client=client)
    user.create()
    generated = password if spec.password is None else None
    return user, generated, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_datasets(
    client: rg.Argilla,
    settings: AnnotationSetupSettings | None = None,
) -> SetupResult:
    """Create all workspaces and datasets idempotently per settings topology."""
    if settings is None:
        settings = AnnotationSetupSettings()

    result = SetupResult()

    for ws_base, tasks in settings.workspace_dataset_map.items():
        ws_name = _apply_prefix(settings.workspace_prefix, ws_base)
        workspace, created = _create_workspace(client, ws_name)
        (result.created_workspaces if created else result.skipped_workspaces).append(ws_name)

        for task in tasks:
            ds_base = DATASET_NAMES[task]
            ds_name = _apply_prefix(settings.workspace_prefix, ds_base)
            base_settings = TASK_SETTINGS[task]

            # Never mutate the module-level constant — construct fresh Settings
            task_settings = rg.Settings(
                fields=base_settings.fields,
                questions=base_settings.questions,
                guidelines=base_settings.guidelines,
                distribution=rg.TaskDistribution(min_submitted=settings.min_submitted),
            )
            _, ds_created = _create_dataset(client, ds_name, ws_name, task_settings)
            (result.created_datasets if ds_created else result.skipped_datasets).append(ds_name)

    return result


def provision_users(
    client: rg.Argilla,
    users: list[UserSpec],
    settings: AnnotationSetupSettings | None = None,
) -> SetupResult:
    """Create user accounts and assign to workspaces idempotently."""
    if settings is None:
        settings = AnnotationSetupSettings()

    result = SetupResult()

    for spec in users:
        user, generated_pw, created = _create_user(client, spec, settings.workspace_prefix)
        (result.created_users if created else result.skipped_users).append(spec.username)
        if generated_pw is not None:
            result.generated_passwords[spec.username] = generated_pw

        if created:
            for ws_base in spec.workspaces:
                ws_name = _apply_prefix(settings.workspace_prefix, ws_base)
                workspace = client.workspaces(ws_name)
                if workspace is not None:
                    workspace.add_user(user)
                else:
                    logger.warning("Workspace %r not found when assigning user %r", ws_name, spec.username)

    return result


def setup(
    client: rg.Argilla,
    settings: AnnotationSetupSettings | None = None,
    users: list[UserSpec] | None = None,
) -> SetupResult:
    """Orchestrate full annotation setup: workspaces, datasets, users."""
    ds_result = setup_datasets(client, settings)
    user_result = provision_users(client, users or [], settings)
    return ds_result._merge(user_result)


def teardown(
    client: rg.Argilla,
    settings: AnnotationSetupSettings | None = None,
    *,
    include_users: bool = False,
) -> None:
    """Delete datasets, workspaces, and optionally users.

    Ordering: datasets first (Argilla requires workspace to be empty before deletion).
    Missing resources are silently skipped.
    """
    if settings is None:
        settings = AnnotationSetupSettings()

    deleted_user_ids: set = set()

    for ws_base, tasks in settings.workspace_dataset_map.items():
        ws_name = _apply_prefix(settings.workspace_prefix, ws_base)
        workspace = client.workspaces(ws_name)
        if workspace is None:
            logger.info("Workspace %r not found — skipping", ws_name)
            continue

        for task in tasks:
            ds_base = DATASET_NAMES[task]
            ds_name = _apply_prefix(settings.workspace_prefix, ds_base)
            dataset = client.datasets(ds_name, workspace=ws_name)
            if dataset is not None:
                dataset.delete()
                logger.info("Deleted dataset %r from workspace %r", ds_name, ws_name)

        if include_users:
            for user in list(workspace.users):
                deleted_user_ids.add(user.id)

        if workspace is not None:
            workspace.delete()
            logger.info("Deleted workspace %r", ws_name)

    if include_users:
        for user_id in deleted_user_ids:
            user = client.users(id=user_id)
            if user is not None:
                logger.info("Deleted user %r", user.username)
                user.delete()
