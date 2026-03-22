"""Structural tests for the Docker Compose dev stack (no Docker required)."""

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEPLOY_DIR = ROOT / "deploy" / "annotation"
COMPOSE_PATH = DEPLOY_DIR / "docker-compose.dev.yml"
ENV_EXAMPLE_PATH = DEPLOY_DIR / ".env.dev.example"
MAKEFILE_PATH = ROOT / "Makefile"

EXPECTED_SERVICES = {"argilla", "worker", "postgres", "elasticsearch", "redis"}
HEALTHCHECK_SERVICES = {"argilla", "postgres", "elasticsearch"}
NAMED_VOLUME_SERVICES = {"argilla", "postgres", "elasticsearch", "redis"}
REQUIRED_ENV_VARS = {"ARGILLA_PORT", "ARGILLA_USERNAME", "ARGILLA_PASSWORD", "ARGILLA_API_KEY", "POSTGRES_PASSWORD"}
EXPECTED_MAKE_TARGETS = {
    "docker-up",
    "docker-down",
    "docker-stop",
    "docker-logs",
    "docker-status",
    "test-stack",
    "test",
    "test-integration",
    "test-all",
}


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text())


def test_compose_file_valid_yaml() -> None:
    """docker-compose.dev.yml parses as valid YAML with a services key."""
    data = _load_compose()
    assert "services" in data


def test_compose_defines_expected_services() -> None:
    """All 5 required services are defined."""
    services = set(_load_compose()["services"])
    assert services == EXPECTED_SERVICES


def test_compose_services_have_healthchecks() -> None:
    """Critical services define healthcheck blocks."""
    services = _load_compose()["services"]
    for name in HEALTHCHECK_SERVICES:
        assert "healthcheck" in services[name], f"{name} missing healthcheck"


def test_compose_services_use_named_volumes() -> None:
    """Stateful services mount named volumes (not anonymous)."""
    data = _load_compose()
    top_level_volumes = set(data.get("volumes", {}))
    services = data["services"]
    for name in NAMED_VOLUME_SERVICES:
        svc_volumes = services[name].get("volumes", [])
        assert svc_volumes, f"{name} has no volumes"
        for vol in svc_volumes:
            vol_name = vol.split(":")[0]
            assert vol_name in top_level_volumes, f"{name} volume '{vol_name}' not in top-level named volumes"


def test_env_example_exists_with_required_vars() -> None:
    """.env.dev.example contains all required environment variables."""
    content = ENV_EXAMPLE_PATH.read_text()
    defined_vars = {line.split("=")[0] for line in content.splitlines() if "=" in line and not line.startswith("#")}
    missing = REQUIRED_ENV_VARS - defined_vars
    assert not missing, f"Missing env vars: {missing}"


def test_makefile_defines_expected_targets() -> None:
    """Makefile contains all expected phony targets."""
    content = MAKEFILE_PATH.read_text()
    # Match target definitions (word followed by colon at start of line)
    defined_targets = set(re.findall(r"^(\S+):", content, re.MULTILINE))
    missing = EXPECTED_MAKE_TARGETS - defined_targets
    assert not missing, f"Missing Makefile targets: {missing}"
