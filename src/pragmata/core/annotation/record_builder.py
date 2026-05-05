"""Annotation import implementation — validation, record building, and fan-out.

Validates raw dicts against the canonical schema, builds Argilla Record
objects from typed QueryResponsePair inputs, and logs them to the
appropriate datasets. The api/ layer resolves settings and delegates here.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling  # no public re-export in argilla v2; pinned to ==2.6.0

from pragmata.core.annotation.argilla_ops import apply_suffix, create_dataset
from pragmata.core.annotation.argilla_task_definitions import DATASET_NAMES, build_task_settings
from pragmata.core.schemas.annotation_import import QueryResponsePair
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings, TaskOverlap

logger = logging.getLogger(__name__)

# Static placeholder — the discard_flow CustomField template reads no record
# data, but Argilla still requires the field to be present on every record.
_DISCARD_FLOW_FIELD = {"discard_flow": {"text": ""}}


# ---------------------------------------------------------------------------
# Validation
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


def validate_records(records: list[dict[str, Any]]) -> ValidationResult:
    """Validate raw dicts against the canonical QueryResponsePair schema.

    Pure validation — no Argilla I/O.

    Args:
        records: Raw dictionaries to validate against QueryResponsePair.

    Returns:
        ValidationResult with successfully parsed pairs and per-index errors.
    """
    valid: list[QueryResponsePair] = []
    errors: list[RecordError] = []
    for i, raw in enumerate(records):
        try:
            valid.append(QueryResponsePair.model_validate(raw))
        except Exception as exc:
            errors.append(RecordError(index=i, detail=str(exc)))
    return ValidationResult(valid=valid, errors=errors)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def derive_record_uuid(pair: QueryResponsePair) -> str:
    """SHA-256 digest of canonical content fields — stable across calls for identical pairs."""
    # chunk_ids sorted for order invariance — same chunks in any order produce the same UUID
    chunk_ids = "|".join(sorted(c.chunk_id for c in pair.chunks))
    canonical = f"{pair.query}\x00{pair.answer}\x00{pair.context_set}\x00{chunk_ids}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_retrieval_records(pair: QueryResponsePair, record_uuid: str) -> list[rg.Record]:
    """One Argilla record per chunk, with shared query and generated answer."""
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
                    **_DISCARD_FLOW_FIELD,
                },
                metadata=metadata,
            )
        )
    return records


def build_grounding_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    """Single Argilla record for grounding evaluation."""
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gnd-{record_uuid}",
        fields={
            "answer": pair.answer,
            "context_set": pair.context_set,
            "query": {"text": pair.query},
            **_DISCARD_FLOW_FIELD,
        },
        metadata=metadata,
    )


def build_generation_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    """Single Argilla record for generation evaluation."""
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gen-{record_uuid}",
        fields={
            "query": pair.query,
            "answer": pair.answer,
            "context_set": {"text": pair.context_set},
            **_DISCARD_FLOW_FIELD,
        },
        metadata=metadata,
    )


def _invert_workspace_map(
    workspace_dataset_map: dict[str, dict[Task, TaskOverlap]],
) -> dict[Task, tuple[str, TaskOverlap]]:
    """Invert workspace_dataset_map to task → (workspace_base, overlap)."""
    task_to_ws: dict[Task, tuple[str, TaskOverlap]] = {}
    for ws_base, task_overlaps in workspace_dataset_map.items():
        for task, overlap in task_overlaps.items():
            task_to_ws[task] = (ws_base, overlap)
    return task_to_ws


def _build_batches(records: list[QueryResponsePair]) -> dict[Task, list[rg.Record]]:
    """Build Argilla records per task from canonical input pairs."""
    batches: dict[Task, list[rg.Record]] = {task: [] for task in Task}
    for pair in records:
        record_uuid = derive_record_uuid(pair)
        batches[Task.RETRIEVAL].extend(build_retrieval_records(pair, record_uuid))
        batches[Task.GROUNDING].append(build_grounding_record(pair, record_uuid))
        batches[Task.GENERATION].append(build_generation_record(pair, record_uuid))
    return batches


def fan_out_records(
    client: rg.Argilla,
    records: list[QueryResponsePair],
    settings: AnnotationSettings,
) -> dict[str, int]:
    """Build and log Argilla records to all three datasets.

    Datasets are created on-the-fly if they don't exist (idempotent).
    Workspaces must already exist (call setup() first).

    Returns counts of records submitted per dataset (not confirmed — individual
    record failures are logged as warnings by Argilla but not reflected in counts).
    """
    task_to_ws = _invert_workspace_map(settings.workspace_dataset_map)
    task_settings_map = build_task_settings()
    batches = _build_batches(records)

    dataset_counts: dict[str, int] = {}

    for task, rg_records in batches.items():
        if not rg_records:
            continue
        binding = task_to_ws.get(task)
        if binding is None:
            logger.warning("Task %r not in workspace_dataset_map — skipping", task)
            continue
        ws_base, overlap = binding

        ds_name = apply_suffix(DATASET_NAMES[task], settings.dataset_id)

        workspace = client.workspaces(ws_base)
        if workspace is None:
            raise RuntimeError(f"Workspace {ws_base!r} not found. Run setup() first.")

        base_settings = task_settings_map[task]
        task_cfg = rg.Settings(
            fields=base_settings.fields,
            questions=base_settings.questions,
            metadata=base_settings.metadata,
            guidelines=base_settings.guidelines,
            distribution=rg.TaskDistribution(min_submitted=overlap.production_min_submitted),
        )
        dataset, ds_created = create_dataset(client, ds_name, ws_base, task_cfg)
        if ds_created:
            logger.info("Auto-created dataset %r in workspace %r", ds_name, ws_base)

        dataset.records.log(rg_records, on_error=RecordErrorHandling.WARN)
        dataset_counts[ds_name] = len(rg_records)
        logger.info("Logged %d records to dataset %r", len(rg_records), ds_name)

    return dataset_counts
