"""Export CSV writer with post-hoc constraint columns and ExportResult type."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pragmata.core.csv_io import _to_csv_value
from pragmata.core.schemas.annotation_export import (
    AnnotationBase,
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task

if TYPE_CHECKING:
    from pragmata.core.paths.annotation_paths import AnnotationExportPaths

_TASK_SCHEMA: dict[Task, type[AnnotationBase]] = {
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
        constraint_summary: Violation count per rule name.
    """

    paths: "AnnotationExportPaths"
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
    schema_cls = _TASK_SCHEMA[task]
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
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
