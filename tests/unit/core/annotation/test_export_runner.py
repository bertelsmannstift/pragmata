"""Unit tests for annotation export orchestration, CSV writer, and ExportResult."""

import csv
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from pragmata.core.annotation.export_runner import (
    TASK_EXPORT_ROW,
    ExportResult,
    write_export_csv,
)
from pragmata.core.csv_io import read_csv
from pragmata.core.paths.annotation_paths import AnnotationExportPaths
from pragmata.core.schemas.annotation_export import (
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task

_NOW = datetime.now()

_BASE = {
    "record_uuid": "abc123",
    "annotator_id": "user1",
    "language": "en",
    "inserted_at": _NOW,
    "created_at": _NOW,
    "record_status": "submitted",
}

_UID1 = UUID("00000000-0000-0000-0000-000000000001")


def _retrieval(**kwargs) -> RetrievalAnnotation:
    defaults = {
        "query": "q",
        "chunk": "c",
        "chunk_id": "cid",
        "doc_id": "did",
        "chunk_rank": 1,
        "topically_relevant": True,
        "evidence_sufficient": False,
        "misleading": False,
    }
    return RetrievalAnnotation.model_validate({**_BASE, **defaults, **kwargs})


def _grounding(**kwargs) -> GroundingAnnotation:
    defaults = {
        "answer": "a",
        "context_set": "ctx",
        "support_present": True,
        "unsupported_claim_present": False,
        "contradicted_claim_present": False,
        "source_cited": True,
        "fabricated_source": False,
    }
    return GroundingAnnotation.model_validate({**_BASE, **defaults, **kwargs})


def _make_response(question_name: str, value: object, user_id: UUID) -> MagicMock:
    resp = MagicMock()
    resp.question_name = question_name
    resp.value = value
    resp.user_id = user_id
    return resp


def _make_record(
    *,
    fields: dict[str, str],
    metadata: dict[str, object],
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


RETRIEVAL_FIELDS = {"query": "What is X?", "chunk": "X is Y."}
BASE_METADATA = {
    "record_uuid": "abc123",
    "language": "en",
    "chunk_id": "chunk-1",
    "doc_id": "doc-1",
    "chunk_rank": 1,
}


def _retrieval_responses(
    user_id: UUID, *, topically_relevant="yes", evidence_sufficient="yes", misleading="no", notes=""
) -> list[MagicMock]:
    return [
        _make_response("topically_relevant", topically_relevant, user_id),
        _make_response("evidence_sufficient", evidence_sufficient, user_id),
        _make_response("misleading", misleading, user_id),
        _make_response("notes", notes, user_id),
    ]


# ---------------------------------------------------------------------------
# write_export_csv
# ---------------------------------------------------------------------------


class TestWriteExportCsv:
    def test_empty_rows_writes_header_only(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        write_export_csv([], out, Task.RETRIEVAL)
        assert out.exists()
        with out.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 1  # header only
        assert "constraint_violated" in rows[0]
        assert "constraint_details" in rows[0]

    def test_header_includes_schema_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        write_export_csv([], out, Task.RETRIEVAL)
        with out.open() as f:
            header = next(csv.reader(f))
        retrieval_fields = list(RetrievalAnnotation.model_fields.keys())
        for field in retrieval_fields:
            assert field in header

    def test_writes_row_with_no_violations(self, tmp_path: Path) -> None:
        row = _retrieval()
        out = tmp_path / "out.csv"
        write_export_csv([(row, [])], out, Task.RETRIEVAL)
        with out.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 1
        assert data[0]["constraint_violated"] == "false"
        assert data[0]["constraint_details"] == ""

    def test_writes_row_with_violations(self, tmp_path: Path) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=True)
        violations = ["retrieval: evidence_sufficient=True but topically_relevant=False"]
        out = tmp_path / "out.csv"
        write_export_csv([(row, violations)], out, Task.RETRIEVAL)
        with out.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert data[0]["constraint_violated"] == "true"
        assert data[0]["constraint_details"] == violations[0]

    def test_multiple_violations_semicolon_joined(self, tmp_path: Path) -> None:
        row = _retrieval()
        violations = ["violation A", "violation B"]
        out = tmp_path / "out.csv"
        write_export_csv([(row, violations)], out, Task.RETRIEVAL)
        with out.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert data[0]["constraint_details"] == "violation A;violation B"

    def test_bool_values_serialised_lowercase(self, tmp_path: Path) -> None:
        row = _retrieval(topically_relevant=True, evidence_sufficient=False)
        out = tmp_path / "out.csv"
        write_export_csv([(row, [])], out, Task.RETRIEVAL)
        with out.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert data[0]["topically_relevant"] == "true"
        assert data[0]["evidence_sufficient"] == "false"

    def test_none_language_serialised_empty(self, tmp_path: Path) -> None:
        row = RetrievalAnnotation.model_validate(
            {
                **_BASE,
                "language": None,
                "query": "q",
                "chunk": "c",
                "chunk_id": "cid",
                "doc_id": "did",
                "chunk_rank": 1,
                "topically_relevant": True,
                "evidence_sufficient": False,
                "misleading": False,
            }
        )
        out = tmp_path / "out.csv"
        write_export_csv([(row, [])], out, Task.RETRIEVAL)
        with out.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert data[0]["language"] == ""

    def test_atomic_write_no_tmp_on_success(self, tmp_path: Path) -> None:
        row = _retrieval()
        out = tmp_path / "out.csv"
        write_export_csv([(row, [])], out, Task.RETRIEVAL)
        assert not out.with_suffix(".tmp").exists()
        assert out.exists()

    def test_grounding_task_schema(self, tmp_path: Path) -> None:
        row = _grounding()
        out = tmp_path / "out.csv"
        write_export_csv([(row, [])], out, Task.GROUNDING)
        with out.open() as f:
            header = next(csv.reader(f))
        grounding_fields = list(GroundingAnnotation.model_fields.keys())
        for field in grounding_fields:
            assert field in header


# ---------------------------------------------------------------------------
# Export-row schema round-trip via csv_io
# ---------------------------------------------------------------------------


class TestExportRowRoundTrip:
    def test_csv_header_matches_export_row_schema(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        write_export_csv([(_retrieval(), [])], out, Task.RETRIEVAL)
        with out.open() as f:
            header = next(csv.reader(f))
        assert header == list(TASK_EXPORT_ROW[Task.RETRIEVAL].model_fields.keys())

    def test_csv_roundtrips_via_read_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        write_export_csv(
            [(_retrieval(), []), (_retrieval(chunk_id="c2"), ["rule_a", "rule_b"])],
            out,
            Task.RETRIEVAL,
        )
        rows = read_csv(out, TASK_EXPORT_ROW[Task.RETRIEVAL])
        assert len(rows) == 2
        assert rows[0].constraint_violated is False
        assert rows[0].constraint_details == ""
        assert rows[1].constraint_violated is True
        assert rows[1].constraint_details == "rule_a;rule_b"


# ---------------------------------------------------------------------------
# ExportResult
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_constructable(self, tmp_path: Path) -> None:
        paths = AnnotationExportPaths(
            export_dir=tmp_path,
            tool_root=tmp_path,
            retrieval_annotation_csv=tmp_path / "retrieval.csv",
            grounding_annotation_csv=tmp_path / "grounding.csv",
            generation_annotation_csv=tmp_path / "generation.csv",
        )
        result = ExportResult(
            paths=paths,
            files={Task.RETRIEVAL: tmp_path / "retrieval.csv"},
            row_counts={Task.RETRIEVAL: 5},
            constraint_summary={"some_rule": 2},
        )
        assert result.paths is paths
        assert result.row_counts[Task.RETRIEVAL] == 5

    def test_frozen(self, tmp_path: Path) -> None:
        paths = AnnotationExportPaths(
            export_dir=tmp_path,
            tool_root=tmp_path,
            retrieval_annotation_csv=tmp_path / "retrieval.csv",
            grounding_annotation_csv=tmp_path / "grounding.csv",
            generation_annotation_csv=tmp_path / "generation.csv",
        )
        result = ExportResult(paths=paths, files={}, row_counts={}, constraint_summary={})
        with pytest.raises((AttributeError, TypeError)):
            result.row_counts = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core export behaviour (moved from test_export_api)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    user = MagicMock()
    user.id = _UID1
    user.username = "annotator1"
    client.users.list.return_value = [user]
    dataset = MagicMock()
    dataset.records.return_value = iter([])
    client.datasets.return_value = dataset

    import pragmata.api.annotation_export as export_module

    monkeypatch.setattr(export_module, "resolve_argilla_client", lambda api_url, api_key: client)
    return client


class TestYesNoConversion:
    def test_yes_no_converted_to_bool(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_retrieval_responses(_UID1, topically_relevant="yes", evidence_sufficient="no"),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows[0]["topically_relevant"] == "true"
        assert rows[0]["evidence_sufficient"] == "false"


class TestConstraintViolations:
    def test_constraint_violations_in_summary(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_retrieval_responses(_UID1, topically_relevant="no", evidence_sufficient="yes"),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        assert result.constraint_summary
        assert sum(result.constraint_summary.values()) >= 1


class TestNotesCoercion:
    def test_notes_none_coerced_to_empty_string(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_export import export_annotations

        record = _make_record(
            fields=RETRIEVAL_FIELDS,
            metadata=BASE_METADATA,
            responses=_retrieval_responses(_UID1, notes=None),
        )
        dataset = MagicMock()
        dataset.records.return_value = iter([record])
        mock_client.datasets.return_value = dataset

        result = export_annotations(base_dir=tmp_path, export_id="test-run", tasks=[Task.RETRIEVAL])
        rows = list(csv.DictReader(result.files[Task.RETRIEVAL].open()))
        assert rows[0]["notes"] == ""
