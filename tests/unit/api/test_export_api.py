"""Unit tests for the annotation export API — Argilla client is fully mocked."""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

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

_UID1 = UUID("00000000-0000-0000-0000-000000000001")
_UID2 = UUID("00000000-0000-0000-0000-000000000002")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    user = MagicMock()
    user.id = _UID1
    user.username = "annotator1"
    client.users.list.return_value = [user]
    dataset = MagicMock()
    dataset.records.return_value = iter([])
    client.datasets.return_value = dataset
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportAnnotations:
    def test_submitted_records_exported(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        submitted = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(_UID1),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([submitted])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run")
        assert result.row_counts[Task.RETRIEVAL] == 1

    def test_one_row_per_annotator_per_record(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        mock_client.users.list.return_value = [
            MagicMock(id=_UID1, username="annotator1"),
            MagicMock(id=_UID2, username="annotator2"),
        ]
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(_UID1) + _make_retrieval_responses(_UID2),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.row_counts[Task.RETRIEVAL] == 2

    def test_empty_dataset_produces_headers_only_csv(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.row_counts[Task.RETRIEVAL] == 0
        assert result.files[Task.RETRIEVAL].exists()
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows == []

    def test_missing_record_uuid_logs_warning(
        self, tmp_path: Path, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from pragmata.api.annotation_export import export_annotations

        metadata_no_uuid = {k: v for k, v in BASE_METADATA.items() if k != "record_uuid"}
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=metadata_no_uuid,
            responses=_make_retrieval_responses(_UID1),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.export_fetcher"):
            result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])

        assert result.row_counts[Task.RETRIEVAL] == 1
        assert any("record_uuid" in msg.lower() for msg in caplog.messages)

    def test_tasks_filter(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert Task.RETRIEVAL in result.files
        assert Task.GROUNDING not in result.files
        assert Task.GENERATION not in result.files

    def test_export_id_auto_generated(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        before = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        result = export_annotations(mock_client, base_dir=tmp_path)
        after = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        export_id = result.paths.export_dir.name
        assert before <= export_id <= after

    def test_export_id_auto_generated_with_dataset_id(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        result = export_annotations(mock_client, base_dir=tmp_path, dataset_id="run1")
        export_id = result.paths.export_dir.name
        assert export_id.startswith("run1_")

    def test_annotator_id_is_username(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        mock_client.users.list.return_value = [MagicMock(id=_UID1, username="alice")]
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(_UID1),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows[0]["annotator_id"] == "alice"

    def test_created_at_from_updated_at(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        updated = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(_UID1),
            updated_at=updated,
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert "2024-06-15" in rows[0]["created_at"]

    def test_created_at_fallback_to_inserted_at(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        inserted = datetime(2024, 3, 10, 8, 0, tzinfo=UTC)
        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_make_retrieval_responses(_UID1),
            updated_at=None,
            inserted_at=inserted,
        )
        record._model.updated_at = None
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert "2024-03-10" in rows[0]["created_at"]

    def test_all_three_tasks_exported_by_default(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        def _fresh_dataset(*_args, **_kwargs) -> MagicMock:
            ds = MagicMock()
            ds.records.return_value = iter([])
            return ds

        mock_client.datasets.side_effect = _fresh_dataset

        result = export_annotations(mock_client, base_dir=tmp_path, export_id="test-run")

        assert set(result.files.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}
        called_names = {call.args[0] for call in mock_client.datasets.call_args_list}
        assert called_names == {"retrieval", "grounding", "generation"}
