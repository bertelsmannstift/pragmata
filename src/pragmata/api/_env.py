"""Env-derived settings layer shared across annotation API entry points.

Every env var the annotation system reads is listed explicitly here so
`grep os.environ.get` in this package surfaces the full coupling. The
layer is wired into ``AnnotationSettings.resolve(env=...)`` at each API
entry point — see :mod:`pragmata.api.annotation_setup`,
:mod:`pragmata.api.annotation_import`, :mod:`pragmata.api.annotation_export`.
"""

import os
from typing import Any


def annotation_env_layer() -> dict[str, Any] | None:
    """Build the env-derived settings layer for annotation APIs.

    Returns ``None`` (not an empty dict) when no relevant env vars are set,
    matching the ``env=`` parameter convention of ``ResolveSettings.resolve``.
    """
    layer: dict[str, Any] = {}
    if api_url := os.environ.get("ARGILLA_API_URL"):
        layer["argilla"] = {"api_url": api_url}
    return layer or None
