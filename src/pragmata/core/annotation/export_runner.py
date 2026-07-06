"""Annotation export orchestration: fetch from Argilla, write CSVs, return ExportResult."""

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import argilla as rg

from pragmata.core.annotation.completeness import (
    CompletenessReport,
    PanelCompleteness,
    compute_completeness_from_records,
)
from pragmata.core.annotation.export_fetcher import (
    AnnotationModel,
    build_user_lookup,
    fetch_retrieval_from_records,
    fetch_task,
    walk_retrieval_records,
)
from pragmata.core.annotation.logical_constraints import LogicalConstraint
from pragmata.core.csv_io import _to_csv_value
from pragmata.core.paths.annotation_paths import AnnotationExportPaths
from pragmata.core.schemas.annotation_export import (
    AnnotationExportMeta,
    CompletenessSummary,
    GenerationAnnotation,
    GenerationExportRow,
    GroundingAnnotation,
    GroundingExportRow,
    RetrievalAnnotation,
    RetrievalExportRow,
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
        constraint_summary: Violation count per ``constraint_id`` (the stable
            short identifier defined in :mod:`logical_constraints`).
        n_annotators: Count of distinct annotators per task.
        completeness: Retrieval panel-completeness report (``None`` when
            retrieval was not in the exported task set).
    """

    paths: AnnotationExportPaths
    files: dict[Task, Path]
    row_counts: dict[Task, int]
    constraint_summary: dict[str, int]
    n_annotators: dict[Task, int]
    completeness: CompletenessReport | None = None


def write_export_csv(
    rows: list[tuple[RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation, list[LogicalConstraint]]],
    path: Path,
    task: Task,
    *,
    completeness: dict[str, PanelCompleteness] | None = None,
) -> None:
    """Write annotation rows to a CSV using the task's export-row schema.

    The schema (``TASK_EXPORT_ROW[task]``) models the full on-disk format,
    including ``constraint_violated`` / ``constraint_details`` (all tasks)
    and ``panel_complete`` / ``n_annotated_chunks`` (retrieval only). Writes
    atomically via a .tmp file; cleans up on failure. Always writes the
    header row even when rows is empty.

    Args:
        rows: List of (annotation, violations) tuples. Each violation is a
            ``LogicalConstraint``; the CSV ``constraint_details`` column is
            generated from ``LogicalConstraint.violation_string()`` and stays
            in the legacy verbose ``";"``-joined format for human readers.
        path: Final output path.
        task: Task type — determines the export-row schema.
        completeness: Optional per-``record_uuid`` panel-completeness map.
            Only used for retrieval; ignored otherwise. Rows whose
            ``record_uuid`` is absent fall back to the schema defaults
            (``False`` / ``0``).
    """
    row_cls = TASK_EXPORT_ROW[task]
    headers = list(row_cls.model_fields.keys())
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for annotation, violations in rows:
                extras: dict[str, Any] = {}
                if task == Task.RETRIEVAL and completeness is not None:
                    pc = completeness.get(annotation.record_uuid)
                    if pc is not None:
                        extras["panel_complete"] = pc.panel_complete
                        extras["n_annotated_chunks"] = pc.n_annotated_chunks
                        extras["n_submitted_chunks"] = pc.n_submitted_chunks
                        extras["n_discarded_chunks"] = pc.n_discarded_chunks
                        extras["n_records_seen"] = pc.n_records_seen
                export_row = row_cls(
                    **annotation.model_dump(),
                    constraint_violated=bool(violations),
                    constraint_details=";".join(c.violation_string() for c in violations),
                    **extras,
                )
                raw = export_row.model_dump(mode="json")
                writer.writerow({k: _to_csv_value(raw[k]) for k in headers})
        tmp.rename(path)
        logger.info("Wrote %d rows to %s", len(rows), path)
    except Exception:
        logger.error("Failed writing CSV %s — cleaning up temp file", path)
        tmp.unlink(missing_ok=True)
        raise


def _resolve_calibration_enabled(settings: "AnnotationSettings", tasks: list[Task]) -> dict[Task, bool]:
    """Per-task calibration topology lookup from settings.

    Raises if any requested task is absent from the topology, mirroring the
    import-side validation. Silently defaulting missing tasks to False would
    hide a config bug.
    """
    flags: dict[Task, bool] = {}
    for ws_name, ws_settings in settings.workspaces.items():
        for task in ws_settings.tasks:
            if task in tasks:
                flags[task] = settings.resolved_task(ws_name, task).calibration_min_submitted is not None
    missing = [task.value for task in tasks if task not in flags]
    if missing:
        raise ValueError(
            f"tasks {sorted(missing)} not present in workspaces topology; "
            "add the task to the topology or remove it from the export request."
        )
    return flags


def assemble_export_meta(
    export_id: str,
    dataset_id: str | None,
    tasks: list[Task],
    include_discarded: bool,
    row_counts: dict[Task, int],
    n_annotators: dict[Task, int],
    calibration_enabled: dict[Task, bool],
    constraint_summary: dict[str, int],
    completeness_summary: CompletenessSummary | None = None,
    completeness_status: str = "not_requested",
) -> AnnotationExportMeta:
    """Assemble run-level provenance metadata for an annotation export.

    Args:
        export_id: Stable export identifier.
        dataset_id: Dataset suffix scoping the Argilla datasets, if any.
        tasks: Tasks that were exported.
        include_discarded: Whether discarded responses were included.
        row_counts: Rows written per task.
        n_annotators: Distinct annotator count per task.
        calibration_enabled: Whether the topology declared a calibration
            dataset for each task.
        constraint_summary: Violation count keyed by ``constraint_id``.
        completeness_summary: Retrieval panel-completeness aggregates;
            ``None`` unless retrieval was exported and computed successfully.
        completeness_status: ``"ok"`` / ``"failed"`` / ``"not_requested"``.
            Disambiguates ``completeness_summary=None`` between "retrieval
            not in tasks" and "compute_completeness raised".

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
        calibration_enabled=calibration_enabled,
        constraint_summary=constraint_summary,
        completeness_summary=completeness_summary,
        completeness_status=completeness_status,  # type: ignore[arg-type]
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
        meta = assemble_export_meta(
            export_id=paths.export_dir.name,
            dataset_id=dataset_id,
            tasks=[],
            include_discarded=include_discarded,
            row_counts={},
            n_annotators={},
            calibration_enabled={},
            constraint_summary={},
            completeness_status="not_requested",
        )
        write_export_meta(meta, paths.export_meta_json)
        return ExportResult(paths=paths, files={}, row_counts={}, constraint_summary={}, n_annotators={})

    user_lookup = build_user_lookup(client)

    # Single retrieval walk shared between the row-emission fetch and the
    # completeness aggregator. fetch_task for non-retrieval tasks walks
    # their own datasets as before; only retrieval was doing two scrolls.
    retrieval_snapshots = walk_retrieval_records(client, settings) if Task.RETRIEVAL in tasks else None

    task_rows: dict[Task, list[tuple[AnnotationModel, list[LogicalConstraint]]]] = {}
    for task in tasks:
        if task == Task.RETRIEVAL and retrieval_snapshots is not None:
            task_rows[task] = fetch_retrieval_from_records(
                retrieval_snapshots, user_lookup, include_discarded=include_discarded
            )
        else:
            task_rows[task] = fetch_task(client, settings, task, user_lookup, include_discarded=include_discarded)

    completeness_report: CompletenessReport | None = None
    completeness_status = "not_requested"
    if retrieval_snapshots is not None:
        try:
            completeness_report = compute_completeness_from_records(retrieval_snapshots)
            completeness_status = "ok"
        except Exception:
            # Completeness is derived (a sidecar summary + extra CSV columns);
            # an aggregation error must not lose an already-fetched export.
            # ``completeness_status="failed"`` distinguishes this case from
            # the "retrieval not requested" path so consumers reading the
            # sidecar can tell them apart.
            logger.exception("compute_completeness failed; continuing without panel-completeness sidecar")
            completeness_status = "failed"

    task_paths = {task: getattr(paths, TASK_CSV_ATTR[task]) for task in tasks}

    written: list[Path] = []
    try:
        for task in tasks:
            completeness_map = completeness_report.by_uuid if (completeness_report and task == Task.RETRIEVAL) else None
            write_export_csv(task_rows[task], task_paths[task], task, completeness=completeness_map)
            written.append(task_paths[task])
    except Exception:
        logger.error("Export failed — rolling back %d written file(s)", len(written))
        for p in written:
            p.unlink(missing_ok=True)
        raise

    row_counts = {task: len(task_rows[task]) for task in tasks}
    n_annotators = {task: len({row.annotator_id for row, _ in task_rows[task]}) for task in tasks}
    calibration_enabled = _resolve_calibration_enabled(settings, tasks)

    constraint_summary: dict[str, int] = {}
    for task in tasks:
        for _, violations in task_rows[task]:
            for c in violations:
                constraint_summary[c.constraint_id] = constraint_summary.get(c.constraint_id, 0) + 1

    meta = assemble_export_meta(
        export_id=paths.export_dir.name,
        dataset_id=dataset_id,
        tasks=tasks,
        include_discarded=include_discarded,
        row_counts=row_counts,
        n_annotators=n_annotators,
        calibration_enabled=calibration_enabled,
        constraint_summary=constraint_summary,
        completeness_summary=completeness_report.summary if completeness_report else None,
        completeness_status=completeness_status,
    )
    write_export_meta(meta, paths.export_meta_json)

    return ExportResult(
        paths=paths,
        files=task_paths,
        row_counts=row_counts,
        constraint_summary=constraint_summary,
        n_annotators=n_annotators,
        completeness=completeness_report,
    )


def resolve_export_id(settings: "AnnotationSettings", export_id: str | None) -> str:
    """Derive a run identifier from an explicit value or generate one from dataset_id + timestamp."""
    if export_id is not None:
        return export_id
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    dataset_id = settings.dataset_id
    return f"{dataset_id}_{ts}" if dataset_id else ts
