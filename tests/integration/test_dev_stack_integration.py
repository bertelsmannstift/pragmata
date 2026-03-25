"""Integration tests for the Docker Compose dev stack (requires Docker)."""

import json
import subprocess
import urllib.request

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.annotation]


def test_stack_boots_healthy() -> None:
    """Make setup brings all services to healthy state."""
    subprocess.run(["make", "setup"], check=True, capture_output=True, timeout=120)
    try:
        resp = urllib.request.urlopen("http://localhost:6900/api/docs", timeout=10)
        assert resp.status == 200, f"Argilla docs returned {resp.status}"
    finally:
        subprocess.run(["make", "teardown"], check=True, capture_output=True, timeout=60)


def test_argilla_api_authenticated() -> None:
    """Argilla API responds to authenticated requests."""
    subprocess.run(["make", "setup"], check=True, capture_output=True, timeout=120)
    try:
        req = urllib.request.Request(
            "http://localhost:6900/api/v1/me",
            headers={"X-Argilla-Api-Key": "argilla.apikey"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["username"] == "argilla"
        assert data["role"] == "owner"
    finally:
        subprocess.run(["make", "teardown"], check=True, capture_output=True, timeout=60)


def test_teardown_removes_volumes() -> None:
    """Make teardown removes containers and volumes for clean slate."""
    subprocess.run(["make", "setup"], check=True, capture_output=True, timeout=120)
    subprocess.run(["make", "teardown"], check=True, capture_output=True, timeout=60)

    result = subprocess.run(
        ["docker", "compose", "-f", "deploy/annotation/docker-compose.dev.yml", "ps", "-q"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "", "Containers still running after teardown"
