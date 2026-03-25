"""Parsing helpers for CLI option values."""

import json

from pragmata.api.querygen import UNSET


def parse_optional_cli_value(value: str | None) -> object:
    """Convert an optional CLI string value into an API-ready value.

    Omitted CLI values are converted to the UNSET sentinel.
    Plain strings are returned unchanged.
    JSON-encoded list[str] and list[dict[str, object]] values are decoded.
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