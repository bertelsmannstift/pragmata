"""Integration tests for annotation export against a live Argilla server.

Run with: pytest tests/annotation/test_export_integration.py -m integration -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

import csv
from uuid import UUID

import argilla as rg
import pytest

from pragmata.api.annotation_export import export_annotations
from pragmata.api.annotation_import import import_records
from pragmata.api.annotation_setup import teardown
from pragmata.core.annotation.export_helpers import ExportResult
from pragmata.core.annotation.setup import setup_datasets
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_PREFIX = "testexport"

_SETTINGS = AnnotationSettings(workspace_prefix=_PREFIX)

_RAW_RECORD = {
    "query": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "chunks": [
        {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "text": "Paris is the capital of France."},
    ],
    "context_set": "ctx-001",
    "language": "en",
}


@pytest.fixture(scope="module")
def client() -> rg.Argilla:
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> WorkspacePaths:
    return WorkspacePaths.from_base_dir(tmp_path_factory.mktemp("export_test"))


@pytest.fixture(autouse=True, scope="module")
def clean_environment(client: rg.Argilla) -> None:
    """Tear down and re-setup prefixed environment before/after all tests."""
    teardown(client, workspace_prefix=_PREFIX)
    setup_datasets(client, _SETTINGS)
    yield  # type: ignore[misc]
    teardown(client, workspace_prefix=_PREFIX)


def _submit_response(
    client: rg.Argilla,
    dataset_name: str,
    workspace_name: str,
    answers: dict[str, str],
    user_id: UUID,
) -> None:
    """Submit a response to the first record in the dataset."""
    ds = client.datasets(dataset_name, workspace=workspace_name)
    records = list(ds.records(with_responses=True))
    if not records:
        return
    record = records[0]
    responses = [rg.Response(question_name=q, value=v, user_id=user_id, status="submitted") for q, v in answers.items()]
    record.responses = responses
    ds.records.log([record])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_export_result_type(client: rg.Argilla, workspace: WorkspacePaths) -> None:
    """export_annotations returns an ExportResult."""
    import_records(client, [_RAW_RECORD], workspace_prefix=_PREFIX)
    result = export_annotations(client, workspace, export_id="type-check", workspace_prefix=_PREFIX)
    assert isinstance(result, ExportResult)


@pytest.mark.integration
def test_three_csv_files_produced(client: rg.Argilla, workspace: WorkspacePaths) -> None:
    """Export produces one CSV per task."""
    import_records(client, [_RAW_RECORD], workspace_prefix=_PREFIX)
    result = export_annotations(client, workspace, export_id="three-files", workspace_prefix=_PREFIX)
    assert set(result.files.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}
    for path in result.files.values():
        assert path.exists()


@pytest.mark.integration
def test_csv_has_correct_schema_columns(client: rg.Argilla, workspace: WorkspacePaths) -> None:
    """Exported CSV has expected columns including constraint columns."""
    import_records(client, [_RAW_RECORD], workspace_prefix=_PREFIX)
    result = export_annotations(client, workspace, export_id="schema-check", workspace_prefix=_PREFIX)

    retrieval_path = result.files[Task.RETRIEVAL]
    with retrieval_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    expected_base = ["record_uuid", "annotator_id", "task", "language", "inserted_at", "created_at", "record_status"]
    for col in expected_base:
        assert col in fieldnames, f"Missing column: {col}"
    assert "constraint_violated" in fieldnames
    assert "constraint_details" in fieldnames


@pytest.mark.integration
def test_empty_export_produces_headers_only(client: rg.Argilla, workspace: WorkspacePaths) -> None:
    """When no responses are submitted, CSVs have headers only."""
    # Fresh teardown/setup to ensure no submitted responses
    teardown(client, workspace_prefix=_PREFIX)
    setup_datasets(client, _SETTINGS)
    import_records(client, [_RAW_RECORD], workspace_prefix=_PREFIX)

    result = export_annotations(client, workspace, export_id="empty-check", workspace_prefix=_PREFIX)
    for task, count in result.row_counts.items():
        assert count == 0, f"Expected 0 rows for {task}, got {count}"
    for path in result.files.values():
        rows = list(csv.DictReader(path.open()))
        assert rows == []
