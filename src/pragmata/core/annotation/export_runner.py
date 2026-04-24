"""Annotation export orchestration: fetch from Argilla, write CSVs, return ExportResult."""

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import argilla as rg

from pragmata.core.annotation.export_fetcher import AnnotationModel, build_user_lookup, fetch_task
from pragmata.core.csv_io import _to_csv_value
from pragmata.core.paths.annotation_paths import AnnotationExportPaths
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GenerationExportRow,
    GroundingAnnotation,
    GroundingExportRow,
    RetrievalAnnotation,
    RetrievalExportRow,
)
from pragmata.core.schemas.annotation_export_meta import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pragmata.core.settings.annotation_settings import AnnotationSettings

TASK_CSV_ATTR = {
    Task.RETRIEVAL: "retrieval_annotation_csv",
    Task.GROUNDING: "grounding_annotation_csv",
    Task.GENERATION: "generation_annotation_csv",
}

TASK_ANNOTATION_SCHEMA: dict[
    Task, type[RetrievalAnnotation] | type[GroundingAnnotation] | type[GenerationAnnotation]
] = {
    Task.RETRIEVAL: RetrievalAnnotation,
    Task.GROUNDING: GroundingAnnotation,
    Task.GENERATION: GenerationAnnotation,
}

TASK_EXPORT_ROW: dict[Task, type[RetrievalExportRow] | type[GroundingExportRow] | type[GenerationExportRow]] = {
    Task.RETRIEVAL: RetrievalExportRow,
    Task.GROUNDING: GroundingExportRow,
    Task.GENERATION: GenerationExportRow,
}


@dataclass(frozen=True)
class ExportResult:
    """Result of a completed annotation export.

    Attributes:
        paths: Path bundle used for this export.
        files: Mapping of task to written CSV file path.
        row_counts: Number of rows written per task.
        constraint_summary: Violation count per rule name (namespaced by task).
        n_annotators: Count of distinct annotators per task.
    """

    paths: AnnotationExportPaths
    files: dict[Task, Path]
    row_counts: dict[Task, int]
    constraint_summary: dict[str, int]
    n_annotators: dict[Task, int]


def write_export_csv(
    rows: list[tuple[RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation, list[str]]],
    path: Path,
    task: Task,
) -> None:
    """Write annotation rows to a CSV using the task's export-row schema.

    The schema (``TASK_EXPORT_ROW[task]``) models the full on-disk format,
    including ``constraint_violated`` and ``constraint_details``. Writes
    atomically via a .tmp file; cleans up on failure. Always writes the
    header row even when rows is empty.

    Args:
        rows: List of (annotation, violations) tuples.
        path: Final output path.
        task: Task type — determines the export-row schema.
    """
    row_cls = TASK_EXPORT_ROW[task]
    headers = list(row_cls.model_fields.keys())
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for annotation, violations in rows:
                export_row = row_cls(
                    **annotation.model_dump(),
                    constraint_violated=bool(violations),
                    constraint_details=";".join(violations),
                )
                raw = export_row.model_dump(mode="json")
                writer.writerow({k: _to_csv_value(raw[k]) for k in headers})
        tmp.rename(path)
        logger.info("Wrote %d rows to %s", len(rows), path)
    except Exception:
        logger.error("Failed writing CSV %s — cleaning up temp file", path)
        tmp.unlink(missing_ok=True)
        raise


def assemble_export_meta(
    export_id: str,
    dataset_id: str | None,
    tasks: list[Task],
    include_discarded: bool,
    row_counts: dict[Task, int],
    n_annotators: dict[Task, int],
    constraint_summary: dict[str, int],
) -> AnnotationExportMeta:
    """Assemble run-level provenance metadata for an annotation export.

    Args:
        export_id: Stable export identifier.
        dataset_id: Dataset suffix scoping the Argilla datasets, if any.
        tasks: Tasks that were exported.
        include_discarded: Whether discarded responses were included.
        row_counts: Rows written per task.
        n_annotators: Distinct annotator count per task.
        constraint_summary: Constraint violation count per rule name.

    Returns:
        Provenance sidecar with an internally stamped creation time.
    """
    return AnnotationExportMeta(
        export_id=export_id,
        created_at=datetime.now(UTC),
        dataset_id=dataset_id,
        tasks=tasks,
        include_discarded=include_discarded,
        row_counts=row_counts,
        n_annotators=n_annotators,
        constraint_summary=constraint_summary,
    )


def write_export_meta(meta: AnnotationExportMeta, path: Path) -> None:
    """Write the export provenance sidecar to disk as JSON."""
    path.write_text(
        json.dumps(meta.model_dump(mode="json")),
        encoding="utf-8",
    )


def run_export(
    client: rg.Argilla,
    settings: "AnnotationSettings",
    paths: AnnotationExportPaths,
    tasks: list[Task],
    *,
    include_discarded: bool,
) -> ExportResult:
    """Fetch all tasks, write CSVs and provenance sidecar, return ExportResult."""
    dataset_id = settings.dataset_id or None

    if not tasks:
        return ExportResult(paths=paths, files={}, row_counts={}, constraint_summary={}, n_annotators={})

    user_lookup = build_user_lookup(client)

    task_rows: dict[Task, list[tuple[AnnotationModel, list[str]]]] = {}
    for task in tasks:
        task_rows[task] = fetch_task(client, settings, task, user_lookup, include_discarded=include_discarded)

    task_paths = {task: getattr(paths, TASK_CSV_ATTR[task]) for task in tasks}

    written: list[Path] = []
    try:
        for task in tasks:
            write_export_csv(task_rows[task], task_paths[task], task)
            written.append(task_paths[task])
    except Exception:
        logger.error("Export failed — rolling back %d written file(s)", len(written))
        for p in written:
            p.unlink(missing_ok=True)
        raise

    row_counts = {task: len(task_rows[task]) for task in tasks}
    n_annotators = {task: len({row.annotator_id for row, _ in task_rows[task]}) for task in tasks}

    constraint_summary: dict[str, int] = {}
    for task in tasks:
        for _, violations in task_rows[task]:
            for v in violations:
                constraint_summary[v] = constraint_summary.get(v, 0) + 1

    meta = assemble_export_meta(
        export_id=paths.export_dir.name,
        dataset_id=dataset_id,
        tasks=tasks,
        include_discarded=include_discarded,
        row_counts=row_counts,
        n_annotators=n_annotators,
        constraint_summary=constraint_summary,
    )
    write_export_meta(meta, paths.export_meta_json)

    return ExportResult(
        paths=paths,
        files=task_paths,
        row_counts=row_counts,
        constraint_summary=constraint_summary,
        n_annotators=n_annotators,
    )


def resolve_export_id(settings: "AnnotationSettings", export_id: str | None) -> str:
    """Derive a run identifier from an explicit value or generate one from dataset_id + timestamp."""
    if export_id is not None:
        return export_id
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    dataset_id = settings.dataset_id
    return f"{dataset_id}_{ts}" if dataset_id else ts
