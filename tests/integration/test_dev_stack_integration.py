"""Integration tests for the Argilla dev stack."""

import json
import urllib.request

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.annotation]


def test_argilla_api_authenticated_as_owner() -> None:
    """Argilla API accepts the configured API key and returns the owner account."""
    req = urllib.request.Request(
        "http://localhost:6900/api/v1/me",
        headers={"X-Argilla-Api-Key": "argilla.apikey"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
    assert data["username"] == "argilla"
    assert data["role"] == "owner"
