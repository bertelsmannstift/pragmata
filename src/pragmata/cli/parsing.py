"""Parsing helpers for CLI option values."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from pragmata.api import UNSET

if TYPE_CHECKING:
    from pragmata.annotation import Locale, Task, UserSpec


def parse_cli_value(value: str | None) -> Any:
    """Normalize a CLI option value into an API-ready value.

    Omitted CLI values are converted to the UNSET sentinel.
    Plain strings are returned unchanged.
    JSON-encoded lists of strings, lists of objects, and objects are decoded
    and returned as native Python values.
    """
    if value is None:
        return UNSET

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value

    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed

    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed

    if isinstance(parsed, dict):
        return parsed

    return value


def parse_tasks(raw: str | None) -> "list[Task] | None":
    """Parse a comma-separated task list (e.g. ``retrieval,grounding``).

    Returns None when no tasks are supplied, leaving task selection to
    the downstream default.
    """
    from pragmata.annotation import Task

    if raw is None:
        return None
    return [Task(item.strip()) for item in raw.split(",")]


def parse_locale(raw: str | None) -> "Locale | None":
    """Normalise a locale string (e.g. ``en``, ``de``).

    Returns None when no locale is supplied. Does not validate catalog
    membership.
    """
    if raw is None:
        return None
    return raw.strip()


def parse_datetime(raw: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string (e.g. ``2026-05-01T00:00:00``).

    Returns None when no value is supplied, leaving filter selection to
    the downstream default. Raises ``typer.BadParameter`` on invalid input
    so the CLI exits with a usage error rather than a traceback.
    """
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid ISO datetime: {raw!r} ({exc})") from exc


def parse_annotator_ids(raw: str | None) -> list[str] | None:
    """Parse a comma-separated list of annotator IDs (e.g. ``alice,bob``).

    Returns None when no IDs are supplied, leaving filter selection to
    the downstream default.
    """
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_user_specs(path: str | None) -> "list[UserSpec] | None":
    """Load annotator user specs from a JSON file.

    The file must contain a list of dicts; each dict is forwarded to
    ``UserSpec(**entry)``. Returns None when ``path`` is None.
    """
    from pragmata.annotation import UserSpec

    if path is None:
        return None
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [UserSpec(**entry) for entry in raw]
