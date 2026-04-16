"""Argilla client construction helper (argilla import deferred to call time)."""

from typing import TYPE_CHECKING

from pragmata.core.settings.settings_base import resolve_api_key

if TYPE_CHECKING:
    import argilla as rg


def resolve_argilla_client(api_url: str | None, api_key: str | None) -> "rg.Argilla":
    """Build an Argilla client from resolved credentials.

    ``api_key`` falls back to ``ARGILLA_API_KEY`` via ``resolve_api_key("argilla")``
    when None. ``api_url`` is passed through as-is; when None the Argilla SDK
    falls back to its own default (``ARGILLA_API_URL`` or the bundled localhost).

    argilla is imported lazily so this helper can be referenced from modules
    that must stay importable without the ``annotation`` extra installed.
    """
    import argilla as rg

    resolved_key = api_key if api_key is not None else resolve_api_key("argilla")
    kwargs: dict[str, str] = {"api_key": resolved_key}
    if api_url is not None:
        kwargs["api_url"] = api_url
    return rg.Argilla(**kwargs)
