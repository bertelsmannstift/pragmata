"""Integration tests for annotation setup against a live Argilla server.

Run with: pytest tests/annotation/test_setup_integration.py -m integration -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

import argilla as rg
import pytest

from chatboteval.core.annotation.setup import (
    SetupResult,
    provision_users,
    setup_datasets,
    teardown_resources,
)
from chatboteval.core.settings.annotation_settings import AnnotationSettings, UserSpec

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
    teardown_resources(client, _DEFAULT_SETTINGS, include_users=True)
    yield
    teardown_resources(client, _DEFAULT_SETTINGS, include_users=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_setup_creates_workspaces_and_datasets(client: rg.Argilla) -> None:
    teardown_resources(client, _DEFAULT_SETTINGS, include_users=True)

    result = setup_datasets(client, _DEFAULT_SETTINGS)

    assert isinstance(result, SetupResult)

    # 3 workspaces (one per task)
    assert client.workspaces("retrieval") is not None
    assert client.workspaces("grounding") is not None
    assert client.workspaces("generation") is not None

    # 3 datasets
    assert client.datasets("task_retrieval", workspace="retrieval") is not None
    assert client.datasets("task_grounding", workspace="grounding") is not None
    assert client.datasets("task_generation", workspace="generation") is not None

    # Result accounting
    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert result.skipped_workspaces == []
    assert set(result.created_datasets) == {
        "task_retrieval",
        "task_grounding",
        "task_generation",
    }
    assert result.skipped_datasets == []


@pytest.mark.integration
def test_dataset_field_and_question_counts(client: rg.Argilla) -> None:
    """Verify datasets have the expected schema shape from annotation-interface.md."""
    # Retrieval: 3 fields (query, chunk, generated_answer), 4 questions (3 label + 1 text)
    ds1 = client.datasets("task_retrieval", workspace="retrieval")
    assert ds1 is not None
    assert len(ds1.settings.fields) == 3
    assert len(ds1.settings.questions) == 4

    # Grounding: 3 fields (answer, context_set, query), 6 questions (5 label + 1 text)
    ds2 = client.datasets("task_grounding", workspace="grounding")
    assert ds2 is not None
    assert len(ds2.settings.fields) == 3
    assert len(ds2.settings.questions) == 6

    # Generation: 3 fields (query, answer, context_set), 6 questions (5 label + 1 text)
    ds3 = client.datasets("task_generation", workspace="generation")
    assert ds3 is not None
    assert len(ds3.settings.fields) == 3
    assert len(ds3.settings.questions) == 6


@pytest.mark.integration
def test_dataset_min_submitted(client: rg.Argilla) -> None:
    ds1 = client.datasets("task_retrieval", workspace="retrieval")
    assert ds1 is not None
    assert ds1.settings.distribution.min_submitted == 1


@pytest.mark.integration
def test_idempotent_rerun(client: rg.Argilla) -> None:
    # Datasets already exist from prior test — re-run should skip all
    result = setup_datasets(client, _DEFAULT_SETTINGS)

    assert result.created_workspaces == []
    assert result.created_datasets == []
    assert set(result.skipped_workspaces) == {"retrieval", "grounding", "generation"}
    assert set(result.skipped_datasets) == {
        "task_retrieval",
        "task_grounding",
        "task_generation",
    }


@pytest.mark.integration
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


@pytest.mark.integration
def test_teardown_without_users_retains_accounts(client: rg.Argilla) -> None:
    # Ensure user exists first
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    teardown_resources(client, _DEFAULT_SETTINGS, include_users=False)

    # Workspaces and datasets gone
    assert client.workspaces("retrieval") is None
    assert client.workspaces("grounding") is None
    assert client.workspaces("generation") is None
    assert client.datasets("task_retrieval", workspace="retrieval") is None

    # User still exists
    assert client.users(_TEST_USER) is not None


@pytest.mark.integration
def test_teardown_with_users_deletes_accounts(client: rg.Argilla) -> None:
    # Re-setup for this test
    setup_datasets(client, _DEFAULT_SETTINGS)
    provision_users(
        client,
        [UserSpec(username=_TEST_USER, role="annotator", workspaces=["retrieval"])],
        _DEFAULT_SETTINGS,
    )

    teardown_resources(client, _DEFAULT_SETTINGS, include_users=True)

    # Workspaces, datasets, and user all gone
    assert client.workspaces("retrieval") is None
    assert client.workspaces("grounding") is None
    assert client.workspaces("generation") is None
    assert client.users(_TEST_USER) is None


@pytest.mark.integration
def test_rerun_after_teardown(client: rg.Argilla) -> None:
    # Clean slate (teardown already ran above)
    result = setup_datasets(client, _DEFAULT_SETTINGS)

    assert set(result.created_workspaces) == {"retrieval", "grounding", "generation"}
    assert set(result.created_datasets) == {
        "task_retrieval",
        "task_grounding",
        "task_generation",
    }
    assert result.skipped_workspaces == []
    assert result.skipped_datasets == []


@pytest.mark.integration
def test_prefix_support(client: rg.Argilla) -> None:
    prefixed_settings = AnnotationSettings(workspace_prefix="test")
    teardown_resources(client, prefixed_settings, include_users=True)

    result = setup_datasets(client, prefixed_settings)

    assert client.workspaces("test_retrieval") is not None
    assert client.workspaces("test_grounding") is not None
    assert client.workspaces("test_generation") is not None
    assert client.datasets("test_task_retrieval", workspace="test_retrieval") is not None
    assert client.datasets("test_task_grounding", workspace="test_grounding") is not None
    assert client.datasets("test_task_generation", workspace="test_generation") is not None

    assert set(result.created_workspaces) == {"test_retrieval", "test_grounding", "test_generation"}
    assert set(result.created_datasets) == {
        "test_task_retrieval",
        "test_task_grounding",
        "test_task_generation",
    }

    # Cleanup prefixed resources
    teardown_resources(client, prefixed_settings, include_users=True)
