"""Shared pytest configuration and fixtures."""

import shutil
import socket

import pytest

ARGILLA_DEFAULT_HOST = "localhost"
ARGILLA_DEFAULT_PORT = 6900
_CONNECT_TIMEOUT_S = 2


def _docker_available() -> bool:
    """Check whether Docker CLI is on PATH."""
    return shutil.which("docker") is not None


def _argilla_reachable() -> bool:
    """Check whether the Argilla server is accepting connections."""
    try:
        with socket.create_connection((ARGILLA_DEFAULT_HOST, ARGILLA_DEFAULT_PORT), timeout=_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _require_integration_stack(request: pytest.FixtureRequest) -> None:
    """Skip annotation interface integration tests when prerequisites are missing.

    Checks (in order):
    1. Docker CLI is on PATH
    2. Argilla server is reachable on localhost:6900

    Fails fast with a clear message rather than hanging on connection timeouts.
    """
    if not request.node.get_closest_marker("annotation"):
        return
    if not _docker_available():
        pytest.skip("Docker CLI not available")
    if not _argilla_reachable():
        pytest.skip(f"Argilla not reachable at {ARGILLA_DEFAULT_HOST}:{ARGILLA_DEFAULT_PORT}")
