"""Shared pytest configuration and fixtures."""

import shutil

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers.

    - registers integration marker programmatically.
    - belt-and-suspenders w/ pyproject.toml registration
    - = ensures no "unknown marker" warnings regardless of how pytest invoked
    """
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring a live Docker/Argilla stack",
    )


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.fixture(autouse=True)
def _skip_without_docker(request: pytest.FixtureRequest) -> None:
    """Skip integration-marked tests when Docker CLI is not on PATH."""
    if request.node.get_closest_marker("integration") and not _docker_available():
        pytest.skip("Docker CLI not available")
