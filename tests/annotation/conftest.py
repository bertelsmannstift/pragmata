"""Pytest configuration for annotation tests.

Module-level patch (not a fixture) because build_task_settings() is called
at collection time in test modules. Fixtures run too late.
"""

from unittest.mock import MagicMock

import argilla as rg

if rg.Argilla._default_client is None:
    rg.Argilla._default_client = MagicMock()
