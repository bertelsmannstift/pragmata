"""Integration tests for annotation setup against a live Argilla server.

Run with: pytest tests/integration/test_annotation_setup.py -m "integration and annotation" -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

import argilla as rg
import pytest

from pragmata.core.annotation.setup import (
    SetupResult,
    provision_users,
    setup_workspaces,
    teardown_resources,
)
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec

pytestmark = [pytest.mark.integration, pytest.mark.annotation]

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_TEST_USER = "test_annotator_integration"

_DEFAULT_SETTINGS = AnnotationSettings()


@pytest.fixture(scope="module")
def client() -> rg.Argilla:
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


@pytest.fixture(autouse=True, scope="module")
def clean_slate(client: rg.Argilla):
    """Tear down before and after the full module so tests start and end clean."""
    teardown_resources(client, _DEFAULT_SETTINGS)
    yield
    teardown_resources(client, _DEFAULT_SETTINGS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_creates_workspaces(client: rg.Argilla) -> None:
    teardown_resources(client, _DEFAULT_SETTINGS)

    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert isinstance(result, SetupResult)

    # 3 workspaces (one per task)
    assert client.workspaces("retrieval") is not None
    assert client.workspaces("grounding") is not None
    assert client.workspaces("generation") is not None

    # Result accounting
    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert result.skipped_workspaces == []


def test_idempotent_rerun(client: rg.Argilla) -> None:
    # Workspaces already exist from prior test — re-run should skip all
    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert result.created_workspaces == []
    assert set(result.skipped_workspaces) == {"retrieval", "grounding", "generation"}


def test_user_provisioning(client: rg.Argilla) -> None:
    result = provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    assert _TEST_USER in result.created_users
    assert _TEST_USER in result.generated_passwords
    assert len(result.generated_passwords[_TEST_USER]) == 16

    # User exists in Argilla
    user = client.users(_TEST_USER)
    assert user is not None


def test_user_workspace_reconciliation_on_rerun(client: rg.Argilla) -> None:
    """Rerunning provision_users assigns existing user to newly-requested workspace."""
    # First run: user in retrieval only
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )
    # Second run: user now also in grounding
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval", "grounding"])],
        _DEFAULT_SETTINGS,
    )
    ws_grounding = client.workspaces("grounding")
    assert ws_grounding is not None
    user = client.users(_TEST_USER)
    assert user in ws_grounding.users


def test_teardown_retains_user_accounts(client: rg.Argilla) -> None:
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    teardown_resources(client, _DEFAULT_SETTINGS)

    # Workspaces gone
    assert client.workspaces("retrieval") is None
    assert client.workspaces("grounding") is None
    assert client.workspaces("generation") is None

    # User still exists
    assert client.users(_TEST_USER) is not None


def test_rerun_after_teardown(client: rg.Argilla) -> None:
    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert result.skipped_workspaces == []


def test_scoped_teardown_preserves_workspaces(client: rg.Argilla) -> None:
    """Teardown with dataset_id only deletes matching datasets, not workspaces."""
    teardown_resources(client, _DEFAULT_SETTINGS)
    setup_workspaces(client, _DEFAULT_SETTINGS)

    # Scoped teardown with a dataset_id should not delete workspaces
    scoped_settings = AnnotationSettings(dataset_id="run1")
    teardown_resources(client, scoped_settings)

    # Workspaces still exist
    assert client.workspaces("retrieval") is not None
    assert client.workspaces("grounding") is not None
    assert client.workspaces("generation") is not None
