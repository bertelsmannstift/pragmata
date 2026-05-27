"""Boundary schemas for canonical annotation import records and partition state."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.types import NonEmptyStr, SafePathSegment


class Chunk(BaseModel):
    """Single retrieved chunk within a query-response pair.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        doc_id: Identifier of the source document.
        chunk_rank: 1-based rank indicating retrieval position.
        text: Content of the retrieved chunk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    chunk_rank: int = Field(ge=1)
    text: NonEmptyStr


class QueryResponsePair(BaseModel):
    """Canonical import record pairing a query with its response and chunks.

    Each pair fans out to records in the retrieval, grounding, and generation
    annotation datasets.

    Attributes:
        query: The user query.
        answer: The system-generated response.
        chunks: Retrieved chunks used to produce the answer (min 1).
        context_set: Identifier grouping chunks into a retrieval context.
        language: Optional ISO language code (e.g. ``"de"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: NonEmptyStr
    answer: NonEmptyStr
    chunks: list[Chunk] = Field(min_length=1)
    context_set: NonEmptyStr
    language: str | None = None


class PartitionManifestEntry(BaseModel):
    """One record's per-task / per-chunk calibration assignment with import provenance.

    Calibration is partitioned at the annotation-item granularity, which differs by
    task. Grounding and generation produce one annotation item per ``record_uuid``
    (``grounding_generation_calibration`` is keyed by task). Retrieval produces one
    annotation item per chunk (``retrieval_chunk_calibration`` is keyed by chunk_id).

    Attributes:
        grounding_generation_calibration: Per-task calibration flag for the two
            tasks that have one annotation item per ``record_uuid``.
        retrieval_chunk_calibration: Per-chunk calibration flag for retrieval,
            keyed by ``chunk_id``. Chunks not present in the manifest at fan-out
            time default to production.
        import_id: Identifier of the import call that produced this assignment.
        calibration_fraction_at_import: Per-task fraction in force at that
            import call.
        calibration_max_records_at_import: Per-task absolute cap in force at
            that import call (``None`` = uncapped).
        assigned_at: When the assignment was made.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounding_generation_calibration: dict[Task, bool]
    retrieval_chunk_calibration: dict[str, bool] = Field(default_factory=dict)
    import_id: NonEmptyStr
    calibration_fraction_at_import: dict[Task, float]
    calibration_max_records_at_import: dict[Task, int | None] = Field(default_factory=dict)
    assigned_at: datetime

    @model_validator(mode="after")
    def _check_fraction_range(self) -> Self:
        for task, fraction in self.calibration_fraction_at_import.items():
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"calibration_fraction_at_import[{task.value}]={fraction} must be in [0.0, 1.0]")
        for task, cap in self.calibration_max_records_at_import.items():
            if cap is not None and cap < 1:
                raise ValueError(
                    f"calibration_max_records_at_import[{task.value}]={cap} must be a positive integer or None"
                )
        return self


class PartitionManifest(BaseModel):
    """Persistent record_uuid -> assignment map scoped to one ``dataset_id``.

    Locks calibration vs production assignments across re-imports so growing
    or repeated batches never reshuffle records between Argilla datasets.

    Attributes:
        dataset_id: Topology scope this manifest belongs to (empty string ok).
        created_at: First write timestamp.
        updated_at: Most recent write timestamp.
        partition_seed: Hash seed used for new-record assignments. Stored at
            manifest level - changing seed mid-scope only affects new records.
        assignments: Map of record_uuid -> PartitionManifestEntry.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: SafePathSegment
    created_at: datetime
    updated_at: datetime
    partition_seed: int
    assignments: dict[str, PartitionManifestEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_timestamps(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at")
        return self
