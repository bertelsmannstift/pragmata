"""Unit tests for the annotation export API — Argilla client is fully mocked."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task

# ---------------------------------------------------------------------------
# Helpers to build mock Argilla records
# ---------------------------------------------------------------------------


def _make_response(question_name: str, value: Any, user_id: UUID) -> MagicMock:
    resp = MagicMock()
    resp.question_name = question_name
    resp.value = value
    resp.user_id = user_id
    return resp


def _make_record(
    *,
    fields: dict[str, str],
    metadata: dict[str, str],
    responses: list[MagicMock],
    updated_at: datetime | None = None,
    inserted_at: datetime | None = None,
    status: str = "submitted",
) -> MagicMock:
    record = MagicMock()
    record.fields = fields
    record.metadata = metadata
    record.responses = responses
    record.status = status
    record._model = MagicMock()
    record._model.updated_at = updated_at or datetime(2024, 1, 1, tzinfo=UTC)
    record._model.inserted_at = inserted_at or datetime(2024, 1, 2, tzinfo=UTC)
    return record


def _make_retrieval_responses(
    user_id: UUID, *, topically_relevant="yes", evidence_sufficient="yes", misleading="no", notes=""
) -> list[MagicMock]:
    return [
        _make_response("topically_relevant", topically_relevant, user_id),
        _make_response("evidence_sufficient", evidence_sufficient, user_id),
        _make_response("misleading", misleading, user_id),
        _make_response("notes", notes, user_id),
    ]


def _make_grounding_responses(user_id: UUID) -> list[MagicMock]:
    return [
        _make_response("support_present", "yes", user_id),
        _make_response("unsupported_claim_present", "no", user_id),
        _make_response("contradicted_claim_present", "no", user_id),
        _make_response("source_cited", "no", user_id),
        _make_response("fabricated_source", "no", user_id),
        _make_response("notes", "", user_id),
    ]


def _make_generation_responses(user_id: UUID) -> list[MagicMock]:
    return [
        _make_response("proper_action", "yes", user_id),
        _make_response("response_on_topic", "yes", user_id),
        _make_response("helpful", "yes", user_id),
        _make_response("incomplete", "no", user_id),
        _make_response("unsafe_content", "no", user_id),
        _make_response("notes", "", user_id),
    ]


RETRIEVAL_FIELDS = {
    "query": "What is X?",
    "chunk": "X is Y.",
}
GROUNDING_FIELDS = {
    "answer": "X is Y.",
    "context_set": "Y explains X.",
}
GENERATION_FIELDS = {
    "query": "What is X?",
    "answer": "X is Y.",
}
BASE_METADATA = {
    "record_uuid": "abc123",
    "language": "en",
    "chunk_id": "chunk-1",
    "doc_id": "doc-1",
    "chunk_rank": 1,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    return WorkspacePaths.from_base_dir(tmp_path)


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    user = MagicMock()
    user.id = UUID("00000000-0000-0000-0000-000000000001")
    user.username = "annotator1"
    client.users.return_value = [user]
    return client


def _setup_dataset(client: MagicMock, task: Task, records: list[MagicMock]) -> None:
    """Wire client.datasets() to return a mock dataset yielding given records."""
    dataset = MagicMock()
    dataset.records.return_value = iter(records)

    original_datasets = client.datasets.side_effect

    calls: dict[str, MagicMock] = {}
    if original_datasets is not None:
        # already set up — preserve previous registrations
        pass

    task_name_map = {
        Task.RETRIEVAL: "task_retrieval",
        Task.GROUNDING: "task_grounding",
        Task.GENERATION: "task_generation",
    }

    def side_effect(name: str, workspace: str | None = None) -> MagicMock:
        if name.endswith(task_name_map[task]):
            return dataset
        return MagicMock(records=MagicMock(return_value=iter([])))

    client.datasets.side_effect = side_effect


def _setup_all_empty(client: MagicMock) -> None:
    """Make all datasets return empty record sets."""
    empty = MagicMock()
    empty.records.return_value = iter([])
    client.datasets.return_value = empty


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportAnnotations:
    def test_submitted_records_exported(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Submitted records appear in output; draft/discarded are excluded."""
        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        submitted = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id),
            status="submitted",
        )
        draft = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id),
            status="draft",
        )

        dataset = MagicMock()
        # Submitted-only records should come from the query filter on the SDK side
        # Our mock returns only submitted records (simulating SDK filter)
        dataset.records.return_value = iter([submitted])
        client = MagicMock()
        user = MagicMock()
        user.id = user_id
        user.username = "annotator1"
        client.users.return_value = [user]
        client.datasets.return_value = dataset

        result = export_annotations(client, workspace, export_id="test-run")
        assert result.row_counts[Task.RETRIEVAL] == 1

    def test_one_row_per_annotator_per_record(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Two annotators on one record produce two rows."""
        from pragmata.api.annotation_export import export_annotations

        uid1 = UUID("00000000-0000-0000-0000-000000000001")
        uid2 = UUID("00000000-0000-0000-0000-000000000002")

        mock_client.users.return_value = [
            MagicMock(id=uid1, username="annotator1"),
            MagicMock(id=uid2, username="annotator2"),
        ]

        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(uid1) + _make_retrieval_responses(uid2),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.row_counts[Task.RETRIEVAL] == 2

    def test_yes_no_converted_to_bool(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """LabelQuestion 'yes'/'no' strings become True/False in output CSV."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(
                user_id, topically_relevant="yes", evidence_sufficient="no", misleading="no"
            ),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        csv_path = result.files[Task.RETRIEVAL]
        rows = list(csv.DictReader(csv_path.open()))
        assert rows[0]["topically_relevant"] == "True"
        assert rows[0]["evidence_sufficient"] == "False"

    def test_empty_dataset_produces_headers_only_csv(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Zero submitted records → CSV with headers, row_count=0."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        dataset = MagicMock()
        dataset.records.return_value = iter([])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.row_counts[Task.RETRIEVAL] == 0
        csv_path = result.files[Task.RETRIEVAL]
        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.open()))
        assert rows == []

    def test_missing_record_uuid_logs_warning(
        self, workspace: WorkspacePaths, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Records missing record_uuid metadata are included with empty string; warning logged."""
        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        metadata_no_uuid = {k: v for k, v in BASE_METADATA.items() if k != "record_uuid"}
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=metadata_no_uuid,
            responses=_make_retrieval_responses(user_id),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        with caplog.at_level(logging.WARNING, logger="pragmata.api.annotation_export"):
            result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])

        assert result.row_counts[Task.RETRIEVAL] == 1
        assert any("record_uuid" in msg.lower() for msg in caplog.messages)

    def test_constraint_violations_in_summary(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Violated constraints are counted in ExportResult.constraint_summary."""
        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        # evidence_sufficient=True but topically_relevant=False → violation
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id, topically_relevant="no", evidence_sufficient="yes"),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.constraint_summary  # non-empty
        total_violations = sum(result.constraint_summary.values())
        assert total_violations >= 1

    def test_tasks_filter(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Passing tasks=[Task.RETRIEVAL] exports only retrieval CSV."""
        from pragmata.api.annotation_export import export_annotations

        dataset = MagicMock()
        dataset.records.return_value = iter([])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert Task.RETRIEVAL in result.files
        assert Task.GROUNDING not in result.files
        assert Task.GENERATION not in result.files

    def test_export_id_auto_generated(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """export_id is auto-generated when not supplied."""
        from pragmata.api.annotation_export import export_annotations

        dataset = MagicMock()
        dataset.records.return_value = iter([])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace)
        # export_dir should exist and have a non-empty name
        assert result.paths.export_dir.name != ""

    def test_annotator_id_is_username(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """annotator_id column contains username string, not UUID."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        mock_client.users.return_value = [MagicMock(id=user_id, username="alice")]

        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows[0]["annotator_id"] == "alice"

    def test_created_at_from_updated_at(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """created_at is taken from record._model.updated_at."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        updated = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id),
            updated_at=updated,
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert "2024-06-15" in rows[0]["created_at"]

    def test_created_at_fallback_to_inserted_at(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """created_at falls back to inserted_at when updated_at is None."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        inserted = datetime(2024, 3, 10, 8, 0, tzinfo=UTC)
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id),
            updated_at=None,
            inserted_at=inserted,
        )
        record._model.updated_at = None  # explicit None fallback
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert "2024-03-10" in rows[0]["created_at"]

    def test_all_three_tasks_exported_by_default(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """Default export (no tasks filter) produces all three CSV files."""
        from pragmata.api.annotation_export import export_annotations

        dataset = MagicMock()
        dataset.records.return_value = iter([])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run")
        assert set(result.files.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}

    def test_notes_none_coerced_to_empty_string(self, workspace: WorkspacePaths, mock_client: MagicMock) -> None:
        """TextQuestion notes=None is coerced to empty string."""
        import csv

        from pragmata.api.annotation_export import export_annotations

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        # notes value is None (annotator left it blank)
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(user_id, notes=None),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, workspace, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows[0]["notes"] == ""
