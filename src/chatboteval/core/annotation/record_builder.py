"""Annotation import implementation — record building and fan-out logic.

Builds Argilla Record objects from canonical QueryResponsePair inputs and
logs them to the appropriate datasets. The api/ layer resolves settings
and delegates here.
"""

import hashlib
import logging

import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling  # no public re-export in argilla v2; pinned to ==2.6.0

from chatboteval.core.annotation.argilla_ops import apply_prefix
from chatboteval.core.annotation.argilla_settings import DATASET_NAMES
from chatboteval.core.schemas.annotation_import import QueryResponsePair
from chatboteval.core.schemas.annotation_task import Task
from chatboteval.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)


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
        },
        metadata=metadata,
    )


def _invert_workspace_map(workspace_dataset_map: dict[str, list[Task]]) -> dict[Task, str]:
    """Invert workspace_dataset_map to task → workspace_base lookup."""
    task_to_ws: dict[Task, str] = {}
    for ws_base, tasks in workspace_dataset_map.items():
        for task in tasks:
            task_to_ws[task] = ws_base
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

    Returns counts of records submitted per dataset (not confirmed — individual
    record failures are logged as warnings by Argilla but not reflected in counts).
    """
    prefix = settings.workspace_prefix
    task_to_ws = _invert_workspace_map(settings.workspace_dataset_map)
    batches = _build_batches(records)

    dataset_counts: dict[str, int] = {}

    for task, rg_records in batches.items():
        if not rg_records:
            continue
        ws_base = task_to_ws.get(task)
        if ws_base is None:
            logger.warning("Task %r not in workspace_dataset_map — skipping", task)
            continue

        ws_name = apply_prefix(prefix, ws_base)
        ds_name = apply_prefix(prefix, DATASET_NAMES[task])

        dataset = client.datasets(ds_name, workspace=ws_name)
        if dataset is None:
            raise RuntimeError(f"Dataset {ds_name!r} in workspace {ws_name!r} not found. Run setup() first.")

        dataset.records.log(rg_records, on_error=RecordErrorHandling.WARN)
        dataset_counts[ds_name] = len(rg_records)
        logger.info("Logged %d records to dataset %r", len(rg_records), ds_name)

    return dataset_counts
