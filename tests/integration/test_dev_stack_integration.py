"""Integration tests for the Argilla dev stack.

Requires the stack to be running (make docker-up). The annotation-stack
preflight in tests/conftest.py gates collection, so these tests skip
cleanly when the stack is unavailable.
"""

import json
import urllib.request

import pytest

from tests.conftest import argilla_api_key, argilla_api_url

pytestmark = [pytest.mark.integration, pytest.mark.annotation]


def test_argilla_api_authenticated_as_owner() -> None:
    """Argilla API accepts the configured API key and returns the owner account."""
    req = urllib.request.Request(
        f"{argilla_api_url()}/api/v1/me",
        headers={"X-Argilla-Api-Key": argilla_api_key()},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
    assert data["username"] == "argilla"
    assert data["role"] == "owner"
