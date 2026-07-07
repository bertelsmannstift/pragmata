"""Shared helpers for CLI unit tests."""

import re

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color escapes from CLI output."""
    return ANSI_ESCAPE_RE.sub("", text)
