"""Annotation import API — validate and fan out canonical records to Argilla datasets.

Public API:
    validate_records(records) -> ValidationResult
    import_records(client, records, settings=None) -> ImportResult
"""

import logging
import uuid
from dataclasses import dataclass

import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling

from chatboteval.api.annotation_setup import _apply_prefix
from chatboteval.api.annotation_task_config import DATASET_NAMES
from chatboteval.core.schemas.annotation_import import QueryResponsePair
from chatboteval.core.schemas.annotation_task import Task
from chatboteval.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordError:
    """Validation failure for a single input record."""

    index: int
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validate_records(): typed pairs and per-index errors."""

    valid: list[QueryResponsePair]
    errors: list[RecordError]


@dataclass(frozen=True)
class ImportResult:
    """Outcome of import_records(): counts per dataset and overall totals."""

    total_records: int
    imported_records: int
    skipped_records: int
    dataset_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Internal record builders
# ---------------------------------------------------------------------------


def _build_retrieval_records(pair: QueryResponsePair, record_uuid: str) -> list[rg.Record]:
    records = []
    for i, chunk in enumerate(pair.chunks):
        metadata: dict = {
            "record_uuid": record_uuid,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_rank": chunk.chunk_rank,
        }
        if pair.language is not None:
            metadata["language"] = pair.language
        records.append(
            rg.Record(
                id=f"ret-{record_uuid}-{i}",
                fields={
                    "query": pair.query,
                    "chunk": chunk.text,
                    "generated_answer": {"text": pair.answer},
                },
                metadata=metadata,
            )
        )
    return records


def _build_grounding_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gnd-{record_uuid}",
        fields={
            "answer": pair.answer,
            "context_set": pair.context_set,
            "query": {"text": pair.query},
        },
        metadata=metadata,
    )


def _build_generation_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gen-{record_uuid}",
        fields={
            "query": pair.query,
            "answer": pair.answer,
            "context_set": {"text": pair.context_set},
        },
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_records(records: list[dict]) -> ValidationResult:
    """Validate raw dicts against QueryResponsePair schema. No I/O."""
    valid: list[QueryResponsePair] = []
    errors: list[RecordError] = []
    for i, raw in enumerate(records):
        try:
            valid.append(QueryResponsePair.model_validate(raw))
        except Exception as exc:
            errors.append(RecordError(index=i, detail=str(exc)))
    return ValidationResult(valid=valid, errors=errors)


def import_records(
    client: rg.Argilla,
    records: list[QueryResponsePair],
    settings: AnnotationSettings | None = None,
) -> ImportResult:
    """Fan out validated records to all three Argilla datasets."""
    if settings is None:
        settings = AnnotationSettings()

    prefix = settings.workspace_prefix

    # Build inverse map task -> ws_base from workspace_dataset_map
    task_to_ws: dict[Task, str] = {}
    for ws_base, tasks in settings.workspace_dataset_map.items():
        for task in tasks:
            task_to_ws[task] = ws_base

    # Accumulate rg.Record objects per task
    batches: dict[Task, list[rg.Record]] = {task: [] for task in Task}

    for pair in records:
        record_uuid = str(uuid.uuid4())
        batches[Task.RETRIEVAL].extend(_build_retrieval_records(pair, record_uuid))
        batches[Task.GROUNDING].append(_build_grounding_record(pair, record_uuid))
        batches[Task.GENERATION].append(_build_generation_record(pair, record_uuid))

    dataset_counts: dict[str, int] = {}

    for task, rg_records in batches.items():
        if not rg_records:
            continue
        ws_base = task_to_ws.get(task)
        if ws_base is None:
            logger.warning("Task %r not in workspace_dataset_map — skipping", task)
            continue

        ws_name = _apply_prefix(prefix, ws_base)
        ds_name = _apply_prefix(prefix, DATASET_NAMES[task])

        dataset = client.datasets(ds_name, workspace=ws_name)
        if dataset is None:
            raise RuntimeError(f"Dataset {ds_name!r} in workspace {ws_name!r} not found. Run setup_datasets() first.")

        dataset.records.log(rg_records, on_error=RecordErrorHandling.WARN)
        dataset_counts[ds_name] = len(rg_records)
        logger.info("Logged %d records to dataset %r", len(rg_records), ds_name)

    total = len(records)
    imported = total  # all validated records are passed to Argilla (errors were pre-filtered)
    return ImportResult(
        total_records=total,
        imported_records=imported,
        skipped_records=total - imported,
        dataset_counts=dataset_counts,
    )
