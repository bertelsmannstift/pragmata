"""Unit tests for annotation export CSV writer and ExportResult type."""

import csv
from datetime import datetime
from pathlib import Path

import pytest

from pragmata.core.annotation.export_helpers import ExportResult, write_export_csv
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
        violations = ["evidence_sufficient=True but topically_relevant=False"]
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
        row = _retrieval()
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
        tmp_file = out.with_suffix(".tmp")
        assert not tmp_file.exists()
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
# ExportResult
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_constructable(self, tmp_path: Path) -> None:
        paths = AnnotationExportPaths(
            export_dir=tmp_path,
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
            retrieval_annotation_csv=tmp_path / "retrieval.csv",
            grounding_annotation_csv=tmp_path / "grounding.csv",
            generation_annotation_csv=tmp_path / "generation.csv",
        )
        result = ExportResult(
            paths=paths,
            files={},
            row_counts={},
            constraint_summary={},
        )
        with pytest.raises((AttributeError, TypeError)):
            result.row_counts = {}  # type: ignore[misc]
