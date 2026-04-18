"""Argilla client construction helper (argilla import deferred to call time)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argilla as rg


def resolve_argilla_client(api_url: str | None, api_key: str) -> "rg.Argilla":
    """Build an Argilla client from resolved credentials.

    Credentials are resolved by the caller (API layer); this helper only
    constructs the SDK client. ``argilla`` is imported lazily so this module
    stays referenceable without the ``annotation`` extra installed.
    """
    import argilla as rg

    kwargs: dict[str, str] = {"api_key": api_key}
    if api_url is not None:
        kwargs["api_url"] = api_url
    return rg.Argilla(**kwargs)
