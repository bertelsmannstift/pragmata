"""Unit tests for the IAA runner (orchestration + report writing)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pragmata.core.annotation.export_runner import TASK_EXPORT_ROW, write_export_csv
from pragmata.core.annotation.iaa_runner import run_iaa
from pragmata.core.paths.annotation_paths import (
    AnnotationExportPaths,
    IaaPaths,
    resolve_iaa_paths,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.iaa_report import IaaReport

_BASE_FIELDS = {
    "language": "en",
    "inserted_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "record_status": "submitted",
    "response_status": "submitted",
}

_RETRIEVAL_DEFAULTS = {
    "query": "q",
    "chunk": "c",
    "chunk_id": "cid",
    "doc_id": "did",
    "chunk_rank": 1,
    "notes": "",
}

_GENERATION_DEFAULTS = {
    "query": "q",
    "answer": "a",
    "notes": "",
}

_GROUNDING_DEFAULTS = {
    "answer": "a",
    "context_set": "ctx",
    "notes": "",
}


def _make_annotation(
    record_uuid: str,
    annotator_id: str,
    task: Task,
    label_values: dict[str, bool],
    created_at: datetime = datetime(2026, 4, 1, tzinfo=timezone.utc),
):
    if task == Task.RETRIEVAL:
        extra = _RETRIEVAL_DEFAULTS
    elif task == Task.GENERATION:
        extra = _GENERATION_DEFAULTS
    else:
        extra = _GROUNDING_DEFAULTS
    schema_cls = TASK_EXPORT_ROW[task].__bases__[0]  # annotation (not export-row) class
    return schema_cls.model_validate(
        {
            **_BASE_FIELDS,
            **extra,
            "record_uuid": record_uuid,
            "annotator_id": annotator_id,
            "created_at": created_at,
        }
        | label_values
    )


def _write_csv(path: Path, task: Task, rows: list[tuple]) -> None:
    """Write an export CSV using the real write_export_csv function."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_export_csv(rows, path, task)


def _make_row(
    record_uuid: str,
    annotator_id: str,
    task: Task,
    label_values: dict[str, bool],
    created_at: datetime = datetime(2026, 4, 1, tzinfo=timezone.utc),
) -> tuple:
    annotation = _make_annotation(record_uuid, annotator_id, task, label_values, created_at)
    return (annotation, [])


@pytest.fixture()
def export_dir(tmp_path: Path) -> AnnotationExportPaths:
    export = tmp_path / "exports" / "test_export"
    return AnnotationExportPaths(
        export_dir=export,
        tool_root=tmp_path,
        retrieval_annotation_csv=export / "retrieval.csv",
        grounding_annotation_csv=export / "grounding.csv",
        generation_annotation_csv=export / "generation.csv",
    )


@pytest.fixture()
def iaa_dir(export_dir: AnnotationExportPaths) -> IaaPaths:
    return resolve_iaa_paths(export_paths=export_dir).ensure_dirs()


class TestRunIaa:
    """Tests for the IAA runner."""

    def test_basic_retrieval_report(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {
            "topically_relevant": True,
            "evidence_sufficient": True,
            "misleading": False,
        }
        labels_b = {
            "topically_relevant": True,
            "evidence_sufficient": False,
            "misleading": False,
        }
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels_b),
            _make_row("r3", "ann1", Task.RETRIEVAL, labels_b),
            _make_row("r3", "ann2", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        assert len(report.tasks) == 1
        task_result = report.tasks[0]
        assert task_result.task == Task.RETRIEVAL
        assert len(task_result.labels) == 3
        assert all(la.n_items == 3 for la in task_result.labels)
        assert all(la.n_annotators == 2 for la in task_result.labels)

    def test_report_written_to_disk(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        assert iaa_dir.report.exists()
        data = json.loads(iaa_dir.report.read_text(encoding="utf-8"))
        parsed = IaaReport.model_validate(data)
        assert parsed.export_id == "test_export"

    def test_pairwise_kappa_included(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels_a = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        labels_b = {"topically_relevant": False, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels_a),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels_a),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels_a),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels_b),
            _make_row("r3", "ann1", Task.RETRIEVAL, labels_b),
            _make_row("r3", "ann2", Task.RETRIEVAL, labels_a),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        pairs = report.tasks[0].pairwise_kappa
        assert len(pairs) == 1
        assert pairs[0].annotator_a == "ann1"
        assert pairs[0].annotator_b == "ann2"
        assert pairs[0].n_shared_items == 3

    def test_missing_csv_skipped(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)
        assert len(report.tasks) == 0

    def test_empty_csv_skipped(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, [])
        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)
        assert len(report.tasks) == 0

    def test_multiple_tasks(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        ret_labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        gen_labels = {
            "proper_action": True,
            "response_on_topic": True,
            "helpful": True,
            "incomplete": False,
            "unsafe_content": False,
        }
        _write_csv(
            export_dir.retrieval_annotation_csv,
            Task.RETRIEVAL,
            [
                _make_row("r1", "ann1", Task.RETRIEVAL, ret_labels),
                _make_row("r1", "ann2", Task.RETRIEVAL, ret_labels),
            ],
        )
        _write_csv(
            export_dir.generation_annotation_csv,
            Task.GENERATION,
            [
                _make_row("r1", "ann1", Task.GENERATION, gen_labels),
                _make_row("r1", "ann2", Task.GENERATION, gen_labels),
            ],
        )

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL, Task.GENERATION], n_resamples=50, seed=42)
        assert len(report.tasks) == 2
        assert {t.task for t in report.tasks} == {Task.RETRIEVAL, Task.GENERATION}

    def test_three_annotators(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        labels_diff = {"topically_relevant": False, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels),
            _make_row("r1", "ann3", Task.RETRIEVAL, labels_diff),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels_diff),
            _make_row("r2", "ann3", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        assert report.tasks[0].labels[0].n_annotators == 3
        # 3 annotators -> 3 pairs
        assert len(report.tasks[0].pairwise_kappa) == 3

    def test_exclude_annotator(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels),
            _make_row("r1", "ann3", Task.RETRIEVAL, labels),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels),
            _make_row("r2", "ann3", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(
            export_dir,
            iaa_dir,
            [Task.RETRIEVAL],
            n_resamples=50,
            seed=42,
            exclude_annotators=["ann3"],
        )

        assert report.tasks[0].labels[0].n_annotators == 2

    def test_filter_by_date(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        labels_diff = {"topically_relevant": False, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels, created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels, created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels_diff, created_at=datetime(2026, 4, 1, tzinfo=timezone.utc)),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels_diff, created_at=datetime(2026, 4, 1, tzinfo=timezone.utc)),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(
            export_dir,
            iaa_dir,
            [Task.RETRIEVAL],
            n_resamples=50,
            seed=42,
            after=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )

        # Only r2 (April) should be included.
        assert report.tasks[0].labels[0].n_items == 1

    def test_pairwise_kappa_omitted_no_shared_items(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        """Pairs with no shared items are silently omitted from pairwise_kappa."""
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        rows = [
            # ann1 annotates r1 only, ann2 annotates r2 only — no overlap
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        assert report.tasks[0].pairwise_kappa == []

    def test_pairwise_kappa_omitted_all_nan(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        """Pairs where all per-label kappas are NaN (constant labels) are omitted."""
        # Both annotators agree perfectly on every label -> kappa is undefined (NaN)
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(export_dir, iaa_dir, [Task.RETRIEVAL], n_resamples=50, seed=42)

        # Perfect constant agreement -> cohen_kappa returns NaN for each label -> pair dropped
        assert report.tasks[0].pairwise_kappa == []

    def test_filter_before_date(self, export_dir: AnnotationExportPaths, iaa_dir: IaaPaths):
        labels = {"topically_relevant": True, "evidence_sufficient": True, "misleading": False}
        rows = [
            _make_row("r1", "ann1", Task.RETRIEVAL, labels, created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _make_row("r1", "ann2", Task.RETRIEVAL, labels, created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _make_row("r2", "ann1", Task.RETRIEVAL, labels, created_at=datetime(2026, 4, 1, tzinfo=timezone.utc)),
            _make_row("r2", "ann2", Task.RETRIEVAL, labels, created_at=datetime(2026, 4, 1, tzinfo=timezone.utc)),
        ]
        _write_csv(export_dir.retrieval_annotation_csv, Task.RETRIEVAL, rows)

        report = run_iaa(
            export_dir,
            iaa_dir,
            [Task.RETRIEVAL],
            n_resamples=50,
            seed=42,
            before=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )

        assert report.tasks[0].labels[0].n_items == 1
