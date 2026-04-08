"""Annotation export orchestration: fetch from Argilla, write CSVs, return ExportResult."""

import csv
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
    AnnotationBase,
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pragmata.core.settings.annotation_settings import AnnotationSettings

TASK_CSV_ATTR = {
    Task.RETRIEVAL: "retrieval_annotation_csv",
    Task.GROUNDING: "grounding_annotation_csv",
    Task.GENERATION: "generation_annotation_csv",
}

TASK_SCHEMA: dict[Task, type[AnnotationBase]] = {
    Task.RETRIEVAL: RetrievalAnnotation,
    Task.GROUNDING: GroundingAnnotation,
    Task.GENERATION: GenerationAnnotation,
}


@dataclass(frozen=True)
class ExportResult:
    """Result of a completed annotation export.

    Attributes:
        paths: Path bundle used for this export.
        files: Mapping of task to written CSV file path.
        row_counts: Number of rows written per task.
        constraint_summary: Violation count per rule name (namespaced by task).
    """

    paths: AnnotationExportPaths
    files: dict[Task, Path]
    row_counts: dict[Task, int]
    constraint_summary: dict[str, int]


def write_export_csv(
    rows: list[tuple[RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation, list[str]]],
    path: Path,
    task: Task,
) -> None:
    """Write annotation rows to a CSV file with constraint columns appended.

    Writes atomically via a .tmp file; cleans up on failure. Always writes
    the header row even when rows is empty.

    Args:
        rows: List of (annotation, violations) tuples.
        path: Final output path.
        task: Task type — determines schema for header derivation.
    """
    schema_cls = TASK_SCHEMA[task]
    headers = list(schema_cls.model_fields.keys()) + ["constraint_violated", "constraint_details"]
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for annotation, violations in rows:
                raw = annotation.model_dump(mode="json")
                row = {k: _to_csv_value(raw[k]) for k in schema_cls.model_fields}
                row["constraint_violated"] = "true" if violations else "false"
                row["constraint_details"] = ";".join(violations)
                writer.writerow(row)
        tmp.rename(path)
        logger.info("Wrote %d rows to %s", len(rows), path)
    except Exception:
        logger.error("Failed writing CSV %s — cleaning up temp file", path)
        tmp.unlink(missing_ok=True)
        raise


def run_export(
    client: rg.Argilla,
    settings: "AnnotationSettings",
    paths: AnnotationExportPaths,
    tasks: list[Task],
) -> ExportResult:
    """Fetch all tasks, write CSVs atomically, return ExportResult."""
    if not tasks:
        return ExportResult(paths=paths, files={}, row_counts={}, constraint_summary={})

    user_lookup = build_user_lookup(client)

    task_rows: dict[Task, list[tuple[AnnotationModel, list[str]]]] = {}
    for task in tasks:
        task_rows[task] = fetch_task(client, settings, task, user_lookup)

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

    constraint_summary: dict[str, int] = {}
    for task in tasks:
        for _, violations in task_rows[task]:
            for v in violations:
                constraint_summary[v] = constraint_summary.get(v, 0) + 1

    return ExportResult(
        paths=paths,
        files=task_paths,
        row_counts=row_counts,
        constraint_summary=constraint_summary,
    )


def resolve_export_id(settings: "AnnotationSettings", export_id: str | None) -> str:
    """Derive a run identifier from an explicit value or generate one from prefix + timestamp."""
    if export_id is not None:
        return export_id
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    prefix = settings.workspace_prefix
    return f"{prefix}_{ts}" if prefix else ts
