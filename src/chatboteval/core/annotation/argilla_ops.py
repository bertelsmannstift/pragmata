"""Low-level Argilla SDK operations — idempotent create/lookup wrappers."""

import logging
import secrets
import string

import argilla as rg

from chatboteval.core.settings.annotation_settings import UserSpec

logger = logging.getLogger(__name__)

_PASSWORD_CHARS = string.ascii_letters + string.digits + "!@#$%"


def apply_prefix(prefix: str, name: str) -> str:
    """Prepend prefix with underscore separator; return name unchanged if prefix is empty."""
    return f"{prefix}_{name}" if prefix else name


def generate_password(length: int = 16) -> str:
    """Random password from alphanumeric + special characters."""
    return "".join(secrets.choice(_PASSWORD_CHARS) for _ in range(length))


def create_workspace(client: rg.Argilla, name: str) -> tuple[rg.Workspace, bool]:
    """Idempotent workspace creation. Returns (workspace, was_created)."""
    existing = client.workspaces(name)
    if existing is not None:
        logger.info("Workspace %r already exists — skipping", name)
        return existing, False
    ws = rg.Workspace(name=name, client=client)
    ws.create()
    return ws, True


def create_dataset(
    client: rg.Argilla,
    name: str,
    workspace: str,
    settings: rg.Settings,
) -> tuple[rg.Dataset, bool]:
    """Idempotent dataset creation. Returns (dataset, was_created)."""
    existing = client.datasets(name, workspace=workspace)
    if existing is not None:
        logger.info("Dataset %r in workspace %r already exists — skipping", name, workspace)
        return existing, False
    ds = rg.Dataset(name=name, workspace=workspace, settings=settings, client=client)
    ds.create()
    return ds, True


def create_user(
    client: rg.Argilla,
    spec: UserSpec,
) -> tuple[rg.User, str | None, bool]:
    """Idempotent user creation. Returns (user, generated_password_or_None, was_created)."""
    existing = client.users(spec.username)
    if existing is not None:
        logger.info("User %r already exists — skipping", spec.username)
        return existing, None, False
    password = spec.password if spec.password is not None else generate_password()
    user = rg.User(username=spec.username, role=spec.role, password=password, client=client)
    user.create()
    generated = password if spec.password is None else None
    return user, generated, True
