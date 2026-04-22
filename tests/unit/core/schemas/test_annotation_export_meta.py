"""Unit tests for AnnotationExportMeta schema."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_export_meta import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task

_NOW = datetime(2026, 4, 22, 15, 30, tzinfo=UTC)


def _meta(**overrides) -> AnnotationExportMeta:
    base = {
        "export_id": "run-1",
        "created_at": _NOW,
        "dataset_id": "pilot",
        "tasks": [Task.RETRIEVAL],
        "include_discarded": False,
        "row_counts": {Task.RETRIEVAL: 3},
        "n_annotators": {Task.RETRIEVAL: 2},
        "constraint_summary": {},
    }
    return AnnotationExportMeta(**{**base, **overrides})


class TestAnnotationExportMeta:
    def test_constructable_with_minimal_fields(self) -> None:
        meta = _meta()
        assert meta.export_id == "run-1"
        assert meta.tasks == [Task.RETRIEVAL]

    def test_dataset_id_can_be_none(self) -> None:
        meta = _meta(dataset_id=None)
        assert meta.dataset_id is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AnnotationExportMeta(
                export_id="run-1",
                created_at=_NOW,
                dataset_id=None,
                tasks=[],
                include_discarded=False,
                row_counts={},
                n_annotators={},
                constraint_summary={},
                surprise="bad",  # type: ignore[call-arg]
            )

    def test_row_counts_keys_must_match_tasks(self) -> None:
        with pytest.raises(ValidationError, match="row_counts"):
            _meta(row_counts={Task.RETRIEVAL: 1, Task.GROUNDING: 1})

    def test_n_annotators_keys_must_match_tasks(self) -> None:
        with pytest.raises(ValidationError, match="n_annotators"):
            _meta(n_annotators={Task.RETRIEVAL: 1, Task.GROUNDING: 1})

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _meta(row_counts={Task.RETRIEVAL: -1})

    def test_serialises_iso_8601_datetime(self) -> None:
        dumped = _meta().model_dump(mode="json")
        assert dumped["created_at"] == "2026-04-22T15:30:00Z"

    def test_round_trips_via_json(self) -> None:
        original = _meta(
            tasks=[Task.RETRIEVAL, Task.GROUNDING],
            row_counts={Task.RETRIEVAL: 3, Task.GROUNDING: 2},
            n_annotators={Task.RETRIEVAL: 2, Task.GROUNDING: 1},
            constraint_summary={"some_rule": 1},
        )
        restored = AnnotationExportMeta.model_validate(original.model_dump(mode="json"))
        assert restored == original
