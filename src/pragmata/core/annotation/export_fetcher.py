"""Fetch submitted annotations from Argilla and build typed export rows.

Handles Argilla SDK interaction (dataset queries, response grouping) and
model construction. The api/ layer resolves settings and delegates here.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import argilla as rg

from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.annotation.export_constraint_checks import CONSTRAINT_CHECKERS
from pragmata.core.annotation.logical_constraints import LogicalConstraint
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)

# A chunk-record's annotation is "terminal" (no longer pending) when at least
# one of its responses has one of these statuses. Discarded counts as covered
# for metrics purposes — it's an explicit decision, not a hole. Shared by
# fetch_task's status-filter callers, completeness, and panel_status so the
# three never drift on the metric definition.
TERMINAL_STATUSES = frozenset({"submitted", "discarded"})

AnnotationModel = RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation


def build_user_lookup(client: rg.Argilla) -> dict[UUID, str]:
    """Map Argilla user IDs to usernames."""
    return {u.id: u.username for u in client.users.list()}


def _to_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "yes"


def _group_responses_by_user(record: rg.Record) -> dict[UUID, tuple[str, dict[str, str]]]:
    """Group record.responses by user_id -> (response_status, {question_name: value})."""
    grouped: dict[UUID, tuple[str, dict[str, str]]] = {}
    for resp in record.responses:
        uid: UUID = resp.user_id
        if uid not in grouped:
            grouped[uid] = (resp.status, {})
        grouped[uid][1][resp.question_name] = resp.value
    return grouped


def _build_row(
    task: Task,
    *,
    base: dict,
    answers: dict[str, str],
    fields: dict[str, str],
    metadata: dict,
) -> AnnotationModel:
    """Build a typed annotation model from Argilla record data."""
    if task == Task.RETRIEVAL:
        return RetrievalAnnotation(
            **base,
            query=fields["query"],
            chunk=fields["chunk"],
            chunk_id=metadata.get("chunk_id", ""),
            doc_id=metadata.get("doc_id", ""),
            chunk_rank=metadata.get("chunk_rank", 0),
            n_retrieved_chunks=metadata.get("n_retrieved_chunks", 0),
            topically_relevant=_to_bool(answers.get("topically_relevant")),
            evidence_sufficient=_to_bool(answers.get("evidence_sufficient")),
            misleading=_to_bool(answers.get("misleading")),
        )
    if task == Task.GROUNDING:
        return GroundingAnnotation(
            **base,
            answer=fields["answer"],
            context_set=fields["context_set"],
            support_present=_to_bool(answers.get("support_present")),
            unsupported_claim_present=_to_bool(answers.get("unsupported_claim_present")),
            contradicted_claim_present=_to_bool(answers.get("contradicted_claim_present")),
            source_cited=_to_bool(answers.get("source_cited")),
            fabricated_source=_to_bool(answers.get("fabricated_source")),
        )
    return GenerationAnnotation(
        **base,
        query=fields["query"],
        answer=fields["answer"],
        proper_action=_to_bool(answers.get("proper_action")),
        response_on_topic=_to_bool(answers.get("response_on_topic")),
        helpful=_to_bool(answers.get("helpful")),
        incomplete=_to_bool(answers.get("incomplete")),
        unsafe_content=_to_bool(answers.get("unsafe_content")),
    )


@dataclass(frozen=True)
class RetrievalRecordSnapshot:
    """One retrieval chunk-record + its per-purpose tagging + response summary.

    The single output of ``walk_retrieval_records`` so the export pipeline's
    row-emission pass and the completeness aggregator can both consume the
    same in-memory record set without independently scrolling the Argilla
    retrieval datasets.

    ``response_user_pairs`` is the per-user grouping (one entry per
    annotator with their (status, answers)), needed for typed-row
    construction; ``has_submitted`` / ``has_discarded`` are the aggregate
    flags completeness needs. ``n_retrieved_chunks_metadata`` is 0 when the
    record has no such metadata (pre-backfill state).
    """

    record: rg.Record
    record_uuid: str
    chunk_id: str
    calibration: bool
    n_retrieved_chunks_metadata: int
    response_user_pairs: list[tuple[UUID, str, dict[str, str]]]
    has_submitted: bool
    has_discarded: bool


def walk_retrieval_records(client: rg.Argilla, settings: AnnotationSettings) -> list[RetrievalRecordSnapshot]:
    """One walk across prod + cal retrieval datasets, no response.status filter.

    Used by both ``compute_completeness`` (needs every record including
    no-response ones for the integrity check) and ``run_export``'s
    retrieval row-emission path (filters in Python at row-construction
    time). When both consumers run in the same export, this lets
    ``run_export`` collect once and pass the result to both - one Argilla
    scroll per dataset instead of two.
    """
    workspace_name, purposes = resolve_task_purposes(settings, Task.RETRIEVAL)
    snapshots: list[RetrievalRecordSnapshot] = []
    for calibration in purposes:
        ds_name = dataset_name(Task.RETRIEVAL, calibration=calibration, dataset_id=settings.dataset_id)
        dataset = client.datasets(ds_name, workspace=workspace_name)
        if dataset is None:
            continue
        for record in dataset.records(with_responses=True):
            grouped = _group_responses_by_user(record)
            user_pairs = [(uid, status, answers) for uid, (status, answers) in grouped.items()]
            statuses = {s for _, s, _ in user_pairs}
            snapshots.append(
                RetrievalRecordSnapshot(
                    record=record,
                    record_uuid=record.metadata.get("record_uuid", ""),
                    chunk_id=record.metadata.get("chunk_id", ""),
                    calibration=calibration,
                    n_retrieved_chunks_metadata=int(record.metadata.get("n_retrieved_chunks") or 0),
                    response_user_pairs=user_pairs,
                    has_submitted="submitted" in statuses,
                    has_discarded="discarded" in statuses,
                )
            )
    return snapshots


def fetch_retrieval_from_records(
    snapshots: list[RetrievalRecordSnapshot],
    user_lookup: dict[UUID, str],
    *,
    include_discarded: bool,
) -> list[tuple[AnnotationModel, list[LogicalConstraint]]]:
    """Project pre-walked retrieval snapshots into typed rows.

    The status filter is applied in Python (per user-response) since the
    walk skipped the query-level filter to feed the completeness pass too.
    Mirrors ``fetch_task``'s retrieval branch but reuses ``_build_row`` so
    typed-row construction has one source of truth.
    """
    accepted = {"submitted", "discarded"} if include_discarded else {"submitted"}
    check_constraints = CONSTRAINT_CHECKERS[Task.RETRIEVAL]
    rows: list[tuple[AnnotationModel, list[LogicalConstraint]]] = []
    missing_uuid_count = 0
    for snap in snapshots:
        if not snap.record_uuid:
            missing_uuid_count += 1
            # Mirror fetch_task: keep going, the row still gets emitted with
            # an empty record_uuid (the export uses this signal too).
        record = snap.record
        created_at: datetime = record._model.updated_at or record._model.inserted_at
        inserted_at: datetime = record._model.inserted_at
        language: str | None = record.metadata.get("language")
        record_status: str = record.status
        for user_id, response_status, answers in snap.response_user_pairs:
            if response_status not in accepted:
                continue
            base = {
                "record_uuid": snap.record_uuid,
                "annotator_id": user_lookup.get(user_id, str(user_id)),
                "language": language,
                "calibration": snap.calibration,
                "inserted_at": inserted_at,
                "created_at": created_at,
                "record_status": record_status,
                "response_status": response_status,
                "discard_reason": answers.get("discard_reason"),
                "discard_notes": answers.get("discard_notes"),
                "notes": answers.get("notes") or "",
            }
            row = _build_row(Task.RETRIEVAL, base=base, answers=answers, fields=record.fields, metadata=record.metadata)
            violations = check_constraints(row) if response_status == "submitted" else []
            rows.append((row, violations))
    if missing_uuid_count:
        logger.warning(
            "task=retrieval: %d record(s) missing record_uuid metadata — included with empty string",
            missing_uuid_count,
        )
    return rows


def resolve_task_purposes(settings: AnnotationSettings, task: Task) -> tuple[str | None, list[bool]]:
    """Topology lookup: workspace owning the task, and which purposes to fetch.

    Returns ``(workspace_name, purposes)`` where purposes is ``[False]`` for
    production-only or ``[False, True]`` when the task has a calibration
    dataset declared. Used by both ``fetch_task`` and the completeness /
    status passes so they walk identical dataset sets.
    """
    workspace_name: str | None = None
    for ws_base, ws_settings in settings.workspaces.items():
        if task in ws_settings.tasks:
            workspace_name = ws_base
            break
    purposes: list[bool] = [False]
    if workspace_name is not None:
        if settings.resolved_task(workspace_name, task).calibration_min_submitted is not None:
            purposes.append(True)
    return workspace_name, purposes


def fetch_task(
    client: rg.Argilla,
    settings: AnnotationSettings,
    task: Task,
    user_lookup: dict[UUID, str],
    *,
    include_discarded: bool,
) -> list[tuple[AnnotationModel, list[LogicalConstraint]]]:
    """Fetch records for a task across calibration and production datasets.

    Iterates the production dataset (always present) and the calibration
    dataset (when topology declares one for this task). Each row is tagged
    with ``calibration: bool`` so downstream consumers can filter.

    By default returns only submitted responses; pass ``include_discarded=True``
    to also include responses the annotator discarded.
    """
    workspace_name, purposes = resolve_task_purposes(settings, task)

    statuses = ["submitted", "discarded"] if include_discarded else ["submitted"]
    query = rg.Query(filter=rg.Filter([("response.status", "in", statuses)]))
    check_constraints = CONSTRAINT_CHECKERS[task]

    rows: list[tuple[AnnotationModel, list[LogicalConstraint]]] = []
    missing_uuid_count = 0

    for calibration in purposes:
        ds_name = dataset_name(task, calibration=calibration, dataset_id=settings.dataset_id)
        dataset = client.datasets(ds_name, workspace=workspace_name)
        if dataset is None:
            # Calibration dataset may not yet exist if no records were ever
            # routed to it (lazy creation). Production should always exist
            # post-import; if missing, fall through silently to match prior
            # "skip empty CSV" behaviour.
            continue

        for record in dataset.records(query, with_responses=True):
            record_uuid: str = record.metadata.get("record_uuid", "")
            if not record_uuid:
                missing_uuid_count += 1

            created_at: datetime = record._model.updated_at or record._model.inserted_at
            inserted_at: datetime = record._model.inserted_at
            language: str | None = record.metadata.get("language")
            record_status: str = record.status

            grouped = _group_responses_by_user(record)
            for user_id, (response_status, answers) in grouped.items():
                base = {
                    "record_uuid": record_uuid,
                    "annotator_id": user_lookup.get(user_id, str(user_id)),
                    "language": language,
                    "calibration": calibration,
                    "inserted_at": inserted_at,
                    "created_at": created_at,
                    "record_status": record_status,
                    "response_status": response_status,
                    "discard_reason": answers.get("discard_reason"),
                    "discard_notes": answers.get("discard_notes"),
                    "notes": answers.get("notes") or "",
                }

                row = _build_row(task, base=base, answers=answers, fields=record.fields, metadata=record.metadata)
                violations = check_constraints(row) if response_status == "submitted" else []
                rows.append((row, violations))

    if missing_uuid_count:
        logger.warning(
            "task=%s: %d record(s) missing record_uuid metadata — included with empty string",
            task.value,
            missing_uuid_count,
        )

    return rows
