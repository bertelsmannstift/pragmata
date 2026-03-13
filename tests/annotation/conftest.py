"""Pytest configuration for annotation unit tests.

Patches the Argilla client before schema modules are imported so that
rg.Settings objects can be constructed without a live Argilla server.
"""

from unittest.mock import MagicMock

import argilla as rg

# Patch before any test module imports schemas.py (which constructs rg.Settings
# at module level). This avoids needing a live Argilla server for unit tests.
if rg.Argilla._default_client is None:
    rg.Argilla._default_client = MagicMock()
