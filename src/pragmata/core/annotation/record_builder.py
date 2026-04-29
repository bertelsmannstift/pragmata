"""Annotation import implementation — validation, record building, and fan-out.

Validates raw dicts against the canonical schema, builds Argilla Record
objects from typed QueryResponsePair inputs, partitions them into calibration
vs production buckets per task, and logs them to the matching Argilla
datasets. The api/ layer resolves settings and delegates here.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling  # no public re-export in argilla v2; pinned to ==2.6.0

from pragmata.core.annotation.argilla_ops import create_dataset
from pragmata.core.annotation.argilla_task_definitions import build_task_settings, dataset_name
from pragmata.core.schemas.annotation_import import (
    PartitionManifest,
    PartitionManifestEntry,
    QueryResponsePair,
)
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


# ---------------------------------------------------------------------------
# Partition logic
# ---------------------------------------------------------------------------


def _bucket_calibration(record_uuid: str, fraction: float, seed: int) -> bool:
    """Deterministic per-record assignment: hash(seed || uuid) < fraction * 2^32."""
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}\x00{record_uuid}".encode()).hexdigest()[:8]
    return int(digest, 16) < int(fraction * (2**32))


# ---------------------------------------------------------------------------
# Partition manifest IO
# ---------------------------------------------------------------------------


def load_partition_manifest(path: Path, *, dataset_id: str, partition_seed: int) -> PartitionManifest:
    """Load an existing manifest from disk or create an empty one."""
    if path.exists():
        return PartitionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    return PartitionManifest(
        dataset_id=dataset_id,
        created_at=now,
        updated_at=now,
        partition_seed=partition_seed,
        assignments={},
    )


def write_partition_manifest(path: Path, manifest: PartitionManifest) -> None:
    """Write manifest atomically (write-tmp-then-rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def assign_partitions(
    pairs: list[QueryResponsePair],
    *,
    manifest: PartitionManifest,
    fraction: float,
    import_id: str,
) -> dict[str, bool]:
    """Resolve each pair's calibration assignment, mutating the manifest in place.

    Existing records keep their prior assignment (re-import safety). New
    records are bucketed via ``_bucket_calibration`` using the manifest's
    stored seed and the current import's fraction; the assignment is recorded
    with import-time provenance.

    Returns a record_uuid -> calibration map covering every input pair.
    """
    assignments: dict[str, bool] = {}
    now = datetime.now(timezone.utc)
    for pair in pairs:
        rid = derive_record_uuid(pair)
        existing = manifest.assignments.get(rid)
        if existing is not None:
            assignments[rid] = existing.calibration
            continue
        is_cal = _bucket_calibration(rid, fraction, manifest.partition_seed)
        manifest.assignments[rid] = PartitionManifestEntry(
            calibration=is_cal,
            import_id=import_id,
            calibration_fraction_at_import=fraction,
            assigned_at=now,
        )
        assignments[rid] = is_cal
    manifest.updated_at = now
    return assignments


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def _build_batches(
    records: list[QueryResponsePair],
    assignments: dict[str, bool],
) -> dict[tuple[Task, bool], list[rg.Record]]:
    """Build Argilla records keyed by (task, calibration) bucket."""
    batches: dict[tuple[Task, bool], list[rg.Record]] = {
        (task, is_cal): [] for task in Task for is_cal in (False, True)
    }
    for pair in records:
        record_uuid = derive_record_uuid(pair)
        is_cal = assignments[record_uuid]
        batches[(Task.RETRIEVAL, is_cal)].extend(build_retrieval_records(pair, record_uuid))
        batches[(Task.GROUNDING, is_cal)].append(build_grounding_record(pair, record_uuid))
        batches[(Task.GENERATION, is_cal)].append(build_generation_record(pair, record_uuid))
    return batches


def _ensure_dataset(
    client: rg.Argilla,
    *,
    task: Task,
    calibration: bool,
    min_submitted: int,
    ws_base: str,
    dataset_id: str,
    task_settings_map: dict[Task, rg.Settings],
) -> rg.Dataset:
    """Resolve or create the Argilla dataset for a (task, purpose) pair."""
    ds_name = dataset_name(task, calibration=calibration, dataset_id=dataset_id)
    workspace = client.workspaces(ws_base)
    if workspace is None:
        raise RuntimeError(f"Workspace {ws_base!r} not found. Run setup() first.")
    base_settings = task_settings_map[task]
    task_cfg = rg.Settings(
        fields=base_settings.fields,
        questions=base_settings.questions,
        metadata=base_settings.metadata,
        guidelines=base_settings.guidelines,
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )
    dataset, ds_created = create_dataset(client, ds_name, ws_base, task_cfg)
    if ds_created:
        logger.info("Auto-created dataset %r in workspace %r", ds_name, ws_base)
    return dataset


def fan_out_records(
    client: rg.Argilla,
    records: list[QueryResponsePair],
    settings: AnnotationSettings,
    *,
    assignments: dict[str, bool],
) -> tuple[dict[str, int], int, int]:
    """Build and log Argilla records to per-purpose datasets.

    Datasets are created on-the-fly if they don't exist (idempotent).
    Workspaces must already exist (call setup() first).

    Args:
        client: Argilla client.
        records: Validated input pairs.
        settings: Annotation settings (topology, dataset_id).
        assignments: Per-record calibration assignment from
            ``assign_partitions``. The caller has already pinned each record
            to a bucket via the manifest.

    Returns:
        ``(dataset_counts, calibration_count, production_count)``.
    """
    task_to_ws = _invert_workspace_map(settings.workspace_dataset_map)
    task_settings_map = build_task_settings()
    batches = _build_batches(records, assignments)

    dataset_counts: dict[str, int] = {}
    calibration_count = sum(1 for is_cal in assignments.values() if is_cal)
    production_count = len(assignments) - calibration_count

    for (task, calibration), rg_records in batches.items():
        if not rg_records:
            continue
        binding = task_to_ws.get(task)
        if binding is None:
            logger.warning("Task %r not in workspace_dataset_map — skipping", task)
            continue
        ws_base, overlap = binding
        if calibration and overlap.calibration_min_submitted is None:
            # Defensive: should not happen because assign_partitions only
            # assigns calibration when topology supports it. Surface as an
            # error rather than silently route to production.
            raise RuntimeError(f"Task {task.value} has calibration records assigned but topology disables calibration")
        min_submitted = overlap.calibration_min_submitted if calibration else overlap.production_min_submitted
        # mypy: narrowed by the guard above
        assert min_submitted is not None
        dataset = _ensure_dataset(
            client,
            task=task,
            calibration=calibration,
            min_submitted=min_submitted,
            ws_base=ws_base,
            dataset_id=settings.dataset_id,
            task_settings_map=task_settings_map,
        )
        dataset.records.log(rg_records, on_error=RecordErrorHandling.WARN)
        ds_name = dataset.name
        dataset_counts[ds_name] = len(rg_records)
        logger.info("Logged %d records to dataset %r", len(rg_records), ds_name)

    return dataset_counts, calibration_count, production_count
