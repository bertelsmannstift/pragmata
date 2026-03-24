"""Annotation export API — fetch submitted annotations from Argilla and write CSVs."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import argilla as rg

from pragmata.core.annotation.argilla_ops import apply_prefix
from pragmata.core.annotation.argilla_task_definitions import DATASET_NAMES
from pragmata.core.annotation.constraints import CONSTRAINT_CHECKERS
from pragmata.core.annotation.export_helpers import ExportResult, write_export_csv
from pragmata.core.paths.annotation_paths import resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, load_config_file

logger = logging.getLogger(__name__)

AnnotationModel = RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation

_TASK_CSV_PATH = {
    Task.RETRIEVAL: "retrieval_csv",
    Task.GROUNDING: "grounding_csv",
    Task.GENERATION: "generation_csv",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_lookup(client: rg.Argilla) -> dict[UUID, str]:
    return {u.id: u.username for u in client.users()}


def _to_bool(value: str) -> bool:
    return value == "yes"


def _group_responses_by_user(record: object) -> dict[UUID, dict[str, str]]:
    """Group record.responses by user_id → {question_name: value}."""
    grouped: dict[UUID, dict[str, str]] = {}
    for resp in record.responses:  # type: ignore[attr-defined]
        uid: UUID = resp.user_id
        grouped.setdefault(uid, {})[resp.question_name] = resp.value
    return grouped


def _fetch_task(
    client: rg.Argilla,
    settings: AnnotationSettings,
    task: Task,
    user_lookup: dict[UUID, str],
) -> list[tuple[AnnotationModel, list[str]]]:
    """Fetch submitted records for task, build typed annotation rows + violations."""
    dataset_name = apply_prefix(settings.workspace_prefix, DATASET_NAMES[task])

    # Find workspace that owns this task's dataset
    workspace_name: str | None = None
    for ws_base, tasks in settings.workspace_dataset_map.items():
        if task in tasks:
            workspace_name = apply_prefix(settings.workspace_prefix, ws_base)
            break

    dataset = client.datasets(dataset_name, workspace=workspace_name)
    query = rg.Query(filter=rg.Filter([("response.status", "==", "submitted")]))

    rows: list[tuple[AnnotationModel, list[str]]] = []
    missing_uuid_count = 0

    for record in dataset.records(query, with_responses=True):
        record_uuid: str = record.metadata.get("record_uuid", "")
        if not record_uuid:
            missing_uuid_count += 1

        created_at: datetime = record._model.updated_at or record._model.inserted_at
        inserted_at: datetime = record._model.inserted_at
        language: str | None = record.metadata.get("language")
        record_status: str = record.status

        grouped = _group_responses_by_user(record)
        for user_id, answers in grouped.items():
            annotator_id = user_lookup.get(user_id, str(user_id))
            notes = answers.get("notes") or ""

            base = dict(
                record_uuid=record_uuid,
                annotator_id=annotator_id,
                language=language,
                inserted_at=inserted_at,
                created_at=created_at,
                record_status=record_status,
                notes=notes,
            )

            if task == Task.RETRIEVAL:
                row: AnnotationModel = RetrievalAnnotation(
                    **base,  # type: ignore[arg-type]
                    query=record.fields["query"],
                    chunk=record.fields["chunk"],
                    chunk_id=record.metadata.get("chunk_id", ""),
                    doc_id=record.metadata.get("doc_id", ""),
                    chunk_rank=record.metadata.get("chunk_rank", 0),
                    topically_relevant=_to_bool(answers["topically_relevant"]),
                    evidence_sufficient=_to_bool(answers["evidence_sufficient"]),
                    misleading=_to_bool(answers["misleading"]),
                )
            elif task == Task.GROUNDING:
                row = GroundingAnnotation(
                    **base,  # type: ignore[arg-type]
                    answer=record.fields["answer"],
                    context_set=record.fields["context_set"],
                    support_present=_to_bool(answers["support_present"]),
                    unsupported_claim_present=_to_bool(answers["unsupported_claim_present"]),
                    contradicted_claim_present=_to_bool(answers["contradicted_claim_present"]),
                    source_cited=_to_bool(answers["source_cited"]),
                    fabricated_source=_to_bool(answers["fabricated_source"]),
                )
            else:  # GENERATION
                row = GenerationAnnotation(
                    **base,  # type: ignore[arg-type]
                    query=record.fields["query"],
                    answer=record.fields["answer"],
                    proper_action=_to_bool(answers["proper_action"]),
                    response_on_topic=_to_bool(answers["response_on_topic"]),
                    helpful=_to_bool(answers["helpful"]),
                    incomplete=_to_bool(answers["incomplete"]),
                    unsafe_content=_to_bool(answers["unsafe_content"]),
                )

            violations = CONSTRAINT_CHECKERS[task](row)
            rows.append((row, violations))

    if missing_uuid_count:
        logger.warning(
            "task=%s: %d record(s) missing record_uuid metadata — included with empty string",
            task.value,
            missing_uuid_count,
        )

    return rows


def _run_export(
    client: rg.Argilla,
    settings: AnnotationSettings,
    paths: object,
    tasks: list[Task],
) -> ExportResult:
    """Orchestrate: fetch all tasks, write CSVs atomically, return ExportResult."""
    user_lookup = _build_user_lookup(client)

    task_rows: dict[Task, list[tuple[AnnotationModel, list[str]]]] = {}
    for task in tasks:
        task_rows[task] = _fetch_task(client, settings, task, user_lookup)

    # Build file mapping using AnnotationExportPaths attributes
    task_csv_attr = {
        Task.RETRIEVAL: "retrieval_csv",
        Task.GROUNDING: "grounding_csv",
        Task.GENERATION: "generation_csv",
    }
    task_paths = {task: getattr(paths, task_csv_attr[task]) for task in tasks}

    # Atomic: track written temps; clean up all on any failure
    written: list[Path] = []
    try:
        for task in tasks:
            write_export_csv(task_rows[task], task_paths[task], task)
            written.append(task_paths[task])
    except Exception:
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
        paths=paths,  # type: ignore[arg-type]
        files=task_paths,
        row_counts=row_counts,
        constraint_summary=constraint_summary,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_annotations(
    client: rg.Argilla,
    workspace: WorkspacePaths,
    *,
    export_id: str | object = UNSET,
    tasks: list[Task] | None = None,
    workspace_prefix: str | object = UNSET,
    config_path: str | Path | object = UNSET,
) -> ExportResult:
    """Fetch submitted annotations from Argilla and write flat CSVs per task.

    Queries each task dataset for submitted-only responses, groups by annotator,
    applies constraint validation, and writes atomic CSVs. Output paths are
    rooted at workspace.tool_root("annotation") / "exports" / export_id.

    Args:
        client: Connected Argilla client.
        workspace: Workspace path bundle.
        export_id: Unique identifier for this export run. Auto-generated from
            prefix + ISO timestamp if not supplied.
        tasks: Tasks to export. Defaults to all three tasks.
        workspace_prefix: Prefix used when the environment was created.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ExportResult with file paths, row counts, and constraint summary.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(cast("str | Path", config_path)) if config_path is not UNSET else None,
        overrides={"workspace_prefix": workspace_prefix},
    )

    if export_id is UNSET:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        prefix = settings.workspace_prefix
        resolved_id = f"{prefix}_{ts}" if prefix else ts
    else:
        resolved_id = cast(str, export_id)

    paths = resolve_export_paths(workspace=workspace, export_id=resolved_id).ensure_dirs()
    resolved_tasks = tasks if tasks is not None else list(Task)

    return _run_export(client, settings, paths, resolved_tasks)
