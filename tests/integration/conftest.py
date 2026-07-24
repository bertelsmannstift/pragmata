"""Shared fixtures for annotation integration tests.

Every annotation integration module talks to the same Argilla stack under test
and isolates its filesystem state per test, so ``client`` and ``base_dir`` live
here rather than being re-derived (and drifting) in each module.
"""

from pathlib import Path

import argilla as rg
import pytest

from tests.conftest import argilla_api_key, argilla_api_url


@pytest.fixture(scope="module")
def client(annotation_stack_status) -> rg.Argilla:
    """Argilla client for the stack under test; skips the module if it isn't ready."""
    if not annotation_stack_status.ready:
        pytest.skip(annotation_stack_status.skip_reason or "annotation stack not ready")
    return rg.Argilla(api_url=argilla_api_url(), api_key=argilla_api_key())


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    """Per-test workspace so each test owns its partition manifest.

    Without this, tests sharing ``base_dir=cwd`` also share one persistent
    manifest; an earlier default-fraction import would lock records into
    calibration and skew later ``calibration_fraction=0.0`` assertions.
    """
    return tmp_path
