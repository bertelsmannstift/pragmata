"""Parsing helpers for CLI option values."""

import json

from pragmata.api import UNSET


def parse_cli_value(value: str | None) -> object:
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
