"""Fetch submitted annotations from Argilla and build typed export rows.

Handles Argilla SDK interaction (dataset queries, response grouping) and
model construction. The api/ layer resolves settings and delegates here.
"""

import logging
from datetime import datetime
from uuid import UUID

import argilla as rg

from pragmata.core.annotation.argilla_ops import apply_prefix
from pragmata.core.annotation.argilla_task_definitions import DATASET_NAMES
from pragmata.core.annotation.constraints import CONSTRAINT_CHECKERS
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)

AnnotationModel = RetrievalAnnotation | GroundingAnnotation | GenerationAnnotation


def build_user_lookup(client: rg.Argilla) -> dict[UUID, str]:
    """Map Argilla user IDs to usernames."""
    return {u.id: u.username for u in client.users()}


def _to_bool(value: str) -> bool:
    return value == "yes"


def _group_responses_by_user(record: object) -> dict[UUID, dict[str, str]]:
    """Group record.responses by user_id -> {question_name: value}."""
    grouped: dict[UUID, dict[str, str]] = {}
    for resp in record.responses:  # type: ignore[attr-defined]
        uid: UUID = resp.user_id
        grouped.setdefault(uid, {})[resp.question_name] = resp.value
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
            topically_relevant=_to_bool(answers["topically_relevant"]),
            evidence_sufficient=_to_bool(answers["evidence_sufficient"]),
            misleading=_to_bool(answers["misleading"]),
        )
    if task == Task.GROUNDING:
        return GroundingAnnotation(
            **base,
            answer=fields["answer"],
            context_set=fields["context_set"],
            support_present=_to_bool(answers["support_present"]),
            unsupported_claim_present=_to_bool(answers["unsupported_claim_present"]),
            contradicted_claim_present=_to_bool(answers["contradicted_claim_present"]),
            source_cited=_to_bool(answers["source_cited"]),
            fabricated_source=_to_bool(answers["fabricated_source"]),
        )
    return GenerationAnnotation(
        **base,
        query=fields["query"],
        answer=fields["answer"],
        proper_action=_to_bool(answers["proper_action"]),
        response_on_topic=_to_bool(answers["response_on_topic"]),
        helpful=_to_bool(answers["helpful"]),
        incomplete=_to_bool(answers["incomplete"]),
        unsafe_content=_to_bool(answers["unsafe_content"]),
    )


def fetch_task(
    client: rg.Argilla,
    settings: AnnotationSettings,
    task: Task,
    user_lookup: dict[UUID, str],
) -> list[tuple[AnnotationModel, list[str]]]:
    """Fetch submitted records for a task, build typed rows with constraint violations."""
    dataset_name = apply_prefix(settings.workspace_prefix, DATASET_NAMES[task])

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
            base = {
                "record_uuid": record_uuid,
                "annotator_id": user_lookup.get(user_id, str(user_id)),
                "language": language,
                "inserted_at": inserted_at,
                "created_at": created_at,
                "record_status": record_status,
                "notes": answers.get("notes") or "",
            }

            row = _build_row(task, base=base, answers=answers, fields=record.fields, metadata=record.metadata)
            violations = CONSTRAINT_CHECKERS[task](row)
            rows.append((row, violations))

    if missing_uuid_count:
        logger.warning(
            "task=%s: %d record(s) missing record_uuid metadata — included with empty string",
            task.value,
            missing_uuid_count,
        )

    return rows
