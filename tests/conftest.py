"""Shared pytest configuration and fixtures."""

import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from urllib.parse import urlsplit

import pytest

ARGILLA_DEFAULT_HOST = "localhost"
ARGILLA_DEFAULT_PORT = 6900
ARGILLA_DEFAULT_API_KEY = "argilla.apikey"
_CONNECT_TIMEOUT_S = 2
_HTTP_TIMEOUT_S = 3


def argilla_api_url() -> str:
    """Base URL of the Argilla stack under test.

    Defaults to the live dev stack on ``localhost:6900``; override via
    ``PRAGMATA_TEST_ARGILLA_URL`` to point integration tests at an ephemeral
    isolated stack (see ``make test-integration``) so destructive
    setup/teardown never touches a shared or live deployment.
    """
    return os.environ.get("PRAGMATA_TEST_ARGILLA_URL", f"http://{ARGILLA_DEFAULT_HOST}:{ARGILLA_DEFAULT_PORT}")


def argilla_api_key() -> str:
    """API key for the Argilla stack under test (override via ``PRAGMATA_TEST_ARGILLA_API_KEY``)."""
    return os.environ.get("PRAGMATA_TEST_ARGILLA_API_KEY", ARGILLA_DEFAULT_API_KEY)


def _argilla_host_port() -> tuple[str, int]:
    parts = urlsplit(argilla_api_url())
    return parts.hostname or ARGILLA_DEFAULT_HOST, parts.port or ARGILLA_DEFAULT_PORT


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
            return "Docker daemon not running (start Docker, then retry)"
        if not self.argilla_tcp:
            host, port = _argilla_host_port()
            return f"Argilla not reachable at {host}:{port} (try: make docker-up)"
        if not self.argilla_api:
            return f"Argilla API not responding at {argilla_api_url()}"
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
    host, port = _argilla_host_port()
    try:
        with socket.create_connection((host, port), timeout=_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _argilla_api_healthy() -> bool:
    try:
        req = urllib.request.Request(
            f"{argilla_api_url()}/api/v1/me",
            headers={"X-Argilla-Api-Key": argilla_api_key()},
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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail-closed guard for destructive annotation integration tests.

    Runs ``trylast`` so pytest's own ``-m`` marker filtering happens first: on a
    unit run the annotation items are already gone and this is a silent no-op;
    only a run that actually selects them (e.g. ``-m integration``) triggers the
    deselection + warning below.

    The annotation integration suite runs destructive setup/teardown (workspace
    and user deletion) against whatever ``argilla_api_url()`` resolves to, which
    defaults to the shared dev stack on :6900. To make it impossible to wipe a
    live or shared stack by accident, these tests are *deselected at collection
    time* unless ``PRAGMATA_TEST_ARGILLA_URL`` is explicitly set.

    ``make test-integration`` sets it to a throwaway isolated stack; a bare
    ``pytest`` run without it collects zero annotation tests, so no fixture at
    any scope is instantiated and no teardown ever fires. Deselection — rather
    than a skip marker — is what guarantees the module-scoped ``clean_environment``
    fixtures never run.
    """
    if os.environ.get("PRAGMATA_TEST_ARGILLA_URL"):
        return
    selected, deselected = [], []
    for item in items:
        (deselected if item.get_closest_marker("annotation") else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        warnings.warn(
            f"Deselected {len(deselected)} annotation integration test(s): PRAGMATA_TEST_ARGILLA_URL "
            "is unset, so they will not run against the default Argilla stack on :6900. Use "
            "`make test-integration` (ephemeral isolated stack) or set PRAGMATA_TEST_ARGILLA_URL "
            "to a disposable target.",
            stacklevel=2,
        )


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
