"""Integration tests for annotation setup against a live Argilla server.

Run with: pytest tests/integration/test_annotation_setup.py -m "integration and annotation" -v
Requires: Argilla stack running on localhost:6900 (make docker-up).

Each test runs against a clean Argilla state — workspaces, datasets, and the
test user are reset before and after every test. Tests are independently
runnable (`pytest -k <name>`) and order-independent.
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
from tests.integration._argilla_cleanup import purge_workspace_datasets

pytestmark = [pytest.mark.integration, pytest.mark.annotation]

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_TEST_USER = "test_annotator_integration"

_DEFAULT_SETTINGS = AnnotationSettings()


@pytest.fixture(scope="module")
def client(annotation_stack_status) -> rg.Argilla:
    if not annotation_stack_status.ready:
        pytest.skip(annotation_stack_status.skip_reason or "annotation stack not ready")
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


def _purge_test_user(client: rg.Argilla) -> None:
    user = client.users(_TEST_USER)
    if user is not None:
        user.delete()


@pytest.fixture(autouse=True)
def clean_slate(client: rg.Argilla):
    """Reset Argilla state (workspaces, datasets, test user) around every test.

    Orphan datasets from earlier runs are purged before teardown — Argilla
    blocks workspace deletion while any dataset is linked. teardown_resources
    intentionally leaves users intact in production, so we purge the test
    user explicitly to keep tests independent.
    """
    for ws_base in _DEFAULT_SETTINGS.workspace_dataset_map:
        purge_workspace_datasets(client, ws_base)
    teardown_resources(client, _DEFAULT_SETTINGS)
    _purge_test_user(client)
    yield
    for ws_base in _DEFAULT_SETTINGS.workspace_dataset_map:
        purge_workspace_datasets(client, ws_base)
    teardown_resources(client, _DEFAULT_SETTINGS)
    _purge_test_user(client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_creates_workspaces(client: rg.Argilla) -> None:
    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert isinstance(result, SetupResult)
    assert client.workspaces("retrieval") is not None
    assert client.workspaces("grounding") is not None
    assert client.workspaces("generation") is not None
    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert result.skipped_workspaces == []


def test_idempotent_rerun(client: rg.Argilla) -> None:
    setup_workspaces(client, _DEFAULT_SETTINGS)

    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert result.created_workspaces == []
    assert set(result.skipped_workspaces) == {"retrieval", "grounding", "generation"}


def test_user_provisioning(client: rg.Argilla) -> None:
    setup_workspaces(client, _DEFAULT_SETTINGS)

    result = provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    assert _TEST_USER in result.created_users
    assert _TEST_USER in result.generated_passwords
    assert len(result.generated_passwords[_TEST_USER]) == 16
    assert client.users(_TEST_USER) is not None


def test_user_workspace_reconciliation_on_rerun(client: rg.Argilla) -> None:
    """Rerunning provision_users assigns existing user to newly-requested workspace."""
    setup_workspaces(client, _DEFAULT_SETTINGS)
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

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
    setup_workspaces(client, _DEFAULT_SETTINGS)
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    teardown_resources(client, _DEFAULT_SETTINGS)

    assert client.workspaces("retrieval") is None
    assert client.workspaces("grounding") is None
    assert client.workspaces("generation") is None
    assert client.users(_TEST_USER) is not None


def test_rerun_after_teardown(client: rg.Argilla) -> None:
    setup_workspaces(client, _DEFAULT_SETTINGS)
    teardown_resources(client, _DEFAULT_SETTINGS)

    result = setup_workspaces(client, _DEFAULT_SETTINGS)

    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert result.skipped_workspaces == []


def test_scoped_teardown_preserves_workspaces(client: rg.Argilla) -> None:
    """Teardown with dataset_id only deletes matching datasets, not workspaces."""
    setup_workspaces(client, _DEFAULT_SETTINGS)

    scoped_settings = AnnotationSettings(dataset_id="run1")
    teardown_resources(client, scoped_settings)

    assert client.workspaces("retrieval") is not None
    assert client.workspaces("grounding") is not None
    assert client.workspaces("generation") is not None
