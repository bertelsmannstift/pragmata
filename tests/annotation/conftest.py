"""Pytest configuration for annotation tests.

Module-level patch (not a fixture) because build_task_settings() is called
at collection time in test modules. Fixtures run too late.

Guarded import: argilla is an optional dependency (annotation extra).
"""

try:
    import argilla as rg
except ModuleNotFoundError:
    pass
else:
    if rg.Argilla._default_client is None:
        from unittest.mock import MagicMock

        rg.Argilla._default_client = MagicMock()
