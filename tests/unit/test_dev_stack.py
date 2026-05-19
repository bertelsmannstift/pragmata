"""Structural tests for the annotation Docker Compose stack (no Docker required).

After the lifecycle migration (docs/design/annotation-stack-lifecycle.md §1),
the stack lives in two files:
- ``src/pragmata/annotation/docker-compose.yml`` — shipped as package data,
  the production-first baseline. This file is the structural source of truth.
- ``deploy/annotation/docker-compose.dev.override.yml`` — contributor-only
  override layered on top by the Makefile's ``docker-*`` targets.

These tests assert structural invariants on the shipped file. A separate
packaging smoke test (``tests/packaging/test_packaged_compose.py``)
asserts the shipped file actually lands in the built wheel.
"""

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHIPPED_COMPOSE_PATH = ROOT / "src" / "pragmata" / "annotation" / "docker-compose.yml"
DEV_OVERRIDE_PATH = ROOT / "deploy" / "annotation" / "docker-compose.dev.override.yml"
ENV_EXAMPLE_PATH = ROOT / "deploy" / "annotation" / ".env.dev.example"
MAKEFILE_PATH = ROOT / "Makefile"

EXPECTED_SERVICES = {"argilla", "worker", "postgres", "elasticsearch", "redis"}
HEALTHCHECK_SERVICES = {"argilla", "postgres", "elasticsearch"}
NAMED_VOLUME_SERVICES = {"argilla", "postgres", "elasticsearch", "redis"}
EXPECTED_VOLUME_NAMES = {
    "argilladata": "pragmata_annotation_argilladata",
    "postgresdata": "pragmata_annotation_postgresdata",
    "elasticdata": "pragmata_annotation_elasticdata",
    "redisdata": "pragmata_annotation_redisdata",
}
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


def _load_shipped_compose() -> dict:
    return yaml.safe_load(SHIPPED_COMPOSE_PATH.read_text())


def _load_dev_override() -> dict:
    return yaml.safe_load(DEV_OVERRIDE_PATH.read_text())


def test_shipped_compose_valid_yaml() -> None:
    """Shipped docker-compose.yml parses as valid YAML with a services key."""
    data = _load_shipped_compose()
    assert "services" in data


def test_shipped_compose_defines_expected_services() -> None:
    """All 5 required services are defined in the shipped compose."""
    services = set(_load_shipped_compose()["services"])
    assert services == EXPECTED_SERVICES


def test_shipped_compose_services_have_healthchecks() -> None:
    """Critical services define healthcheck blocks in the shipped compose."""
    services = _load_shipped_compose()["services"]
    for name in HEALTHCHECK_SERVICES:
        assert "healthcheck" in services[name], f"{name} missing healthcheck"


def test_shipped_compose_services_use_named_volumes() -> None:
    """Stateful services mount top-level named volumes (not anonymous)."""
    data = _load_shipped_compose()
    top_level_volumes = set(data.get("volumes", {}))
    services = data["services"]
    for name in NAMED_VOLUME_SERVICES:
        svc_volumes = services[name].get("volumes", [])
        assert svc_volumes, f"{name} has no volumes"
        for vol in svc_volumes:
            vol_name = vol.split(":")[0]
            assert vol_name in top_level_volumes, f"{name} volume '{vol_name}' not in top-level named volumes"


def test_shipped_compose_volume_names_use_pragmata_annotation_prefix() -> None:
    """Top-level volumes carry deterministic pragmata_annotation_* names per design §3.2."""
    volumes = _load_shipped_compose().get("volumes", {})
    for key, expected_name in EXPECTED_VOLUME_NAMES.items():
        assert key in volumes, f"top-level volume {key} missing"
        assert volumes[key].get("name") == expected_name, (
            f"volume {key} has name {volumes[key].get('name')!r}, expected {expected_name!r}"
        )


def test_shipped_compose_loopback_only_port_binding() -> None:
    """Argilla port binds to 127.0.0.1 in the shipped compose (multi-annotator requires reverse proxy)."""
    argilla_ports = _load_shipped_compose()["services"]["argilla"].get("ports", [])
    assert any("127.0.0.1:" in p for p in argilla_ports), (
        "shipped compose must bind argilla to 127.0.0.1 only — see lifecycle doc §4.3"
    )


def test_dev_override_valid_yaml() -> None:
    """Dev override parses as valid YAML."""
    data = _load_dev_override()
    assert isinstance(data, dict)
    assert "services" in data


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
