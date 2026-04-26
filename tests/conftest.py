"""Shared pytest configuration and fixtures."""

import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

ARGILLA_DEFAULT_HOST = "localhost"
ARGILLA_DEFAULT_PORT = 6900
ARGILLA_DEFAULT_API_KEY = "argilla.apikey"
_CONNECT_TIMEOUT_S = 2
_HTTP_TIMEOUT_S = 3


@dataclass(frozen=True)
class AnnotationStackStatus:
    """Result of progressive preflight checks for the annotation stack."""

    docker_cli: bool
    docker_daemon: bool
    argilla_tcp: bool
    argilla_api: bool

    @property
    def ready(self) -> bool:
        return self.docker_cli and self.docker_daemon and self.argilla_tcp and self.argilla_api

    @property
    def skip_reason(self) -> str | None:
        if not self.docker_cli:
            return "Docker CLI not on PATH"
        if not self.docker_daemon:
            return "Docker daemon not running (try: open -a Docker)"
        if not self.argilla_tcp:
            return f"Argilla not reachable at {ARGILLA_DEFAULT_HOST}:{ARGILLA_DEFAULT_PORT} (try: make docker-up)"
        if not self.argilla_api:
            return f"Argilla API not responding at http://{ARGILLA_DEFAULT_HOST}:{ARGILLA_DEFAULT_PORT}"
        return None


_STACK_STATUS_KEY: pytest.StashKey[AnnotationStackStatus] = pytest.StashKey()


def _docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def _docker_daemon_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=_HTTP_TIMEOUT_S,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _argilla_tcp_open() -> bool:
    try:
        with socket.create_connection((ARGILLA_DEFAULT_HOST, ARGILLA_DEFAULT_PORT), timeout=_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _argilla_api_healthy() -> bool:
    try:
        req = urllib.request.Request(
            f"http://{ARGILLA_DEFAULT_HOST}:{ARGILLA_DEFAULT_PORT}/api/v1/me",
            headers={"X-Argilla-Api-Key": ARGILLA_DEFAULT_API_KEY},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def _check_annotation_stack() -> AnnotationStackStatus:
    docker_cli = _docker_cli_available()
    docker_daemon = docker_cli and _docker_daemon_running()
    argilla_tcp = docker_daemon and _argilla_tcp_open()
    argilla_api = argilla_tcp and _argilla_api_healthy()
    return AnnotationStackStatus(
        docker_cli=docker_cli,
        docker_daemon=docker_daemon,
        argilla_tcp=argilla_tcp,
        argilla_api=argilla_api,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Run annotation stack preflight once at session start, cache on config."""
    config.stash[_STACK_STATUS_KEY] = _check_annotation_stack()


def _stack_layer_lines(status: AnnotationStackStatus) -> list[tuple[str, bool]]:
    return [
        ("docker-cli", status.docker_cli),
        ("daemon", status.docker_daemon),
        ("argilla-tcp", status.argilla_tcp),
        ("argilla-api", status.argilla_api),
    ]


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Print annotation stack status once at session start."""
    status = config.stash[_STACK_STATUS_KEY]
    marks = " ".join(f"{name} [{'ok' if ok else '--'}]" for name, ok in _stack_layer_lines(status))
    line = f">>> annotation stack: {marks}"
    if not status.ready:
        line += f"  (skipping integration: {status.skip_reason})"
    return [line]


@pytest.fixture(scope="session")
def annotation_stack_status(pytestconfig: pytest.Config) -> AnnotationStackStatus:
    """Session-wide preflight result for the annotation stack."""
    return pytestconfig.stash[_STACK_STATUS_KEY]


@pytest.fixture(autouse=True)
def _require_annotation_stack(
    request: pytest.FixtureRequest,
    annotation_stack_status: AnnotationStackStatus,
) -> None:
    """Skip annotation-marked tests when the stack is not ready."""
    if not request.node.get_closest_marker("annotation"):
        return
    if not annotation_stack_status.ready:
        pytest.skip(annotation_stack_status.skip_reason or "annotation stack not ready")
