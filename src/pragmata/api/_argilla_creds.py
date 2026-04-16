"""Resolve Argilla api_url/api_key at the API boundary.

Mirrors the querygen pattern (``api/querygen.py`` calls ``resolve_api_key()``
directly): the API layer owns secret resolution; ``core/`` helpers receive
already-resolved values.

Resolution order:
- ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``settings.argilla.api_url``)
- ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (never from config)
"""

import os

from pragmata.core.settings.settings_base import UNSET, Unset, resolve_api_key


def resolve_api_url_override(api_url: str | Unset) -> str | Unset:
    """Return the api_url override to merge into AnnotationSettings.resolve().

    ``UNSET`` leaves resolution to the config layer (``settings.argilla.api_url``).
    """
    if isinstance(api_url, str):
        return api_url
    env_url = os.environ.get("ARGILLA_API_URL")
    if env_url is not None and env_url.strip():
        return env_url
    return UNSET


def resolve_api_key_override(api_key: str | Unset) -> str:
    """Return a resolved api_key string; raises MissingSecretError if unavailable."""
    return api_key if isinstance(api_key, str) else resolve_api_key("argilla")
