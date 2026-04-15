"""Parsing helpers for CLI option values."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pragmata.api import UNSET

if TYPE_CHECKING:
    from pragmata.annotation import Task, UserSpec


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


def parse_user_specs(path: str | None) -> "list[UserSpec] | None":
    """Load annotator user specs from a JSON file.

    The file must contain a list of dicts; each dict is forwarded to
    ``UserSpec(**entry)``. Returns None when ``path`` is None.
    """
    from pragmata.annotation import UserSpec

    if path is None:
        return None
    raw = json.loads(Path(path).read_text())
    return [UserSpec(**entry) for entry in raw]
