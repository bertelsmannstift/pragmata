"""Boundary schemas for canonical annotation import records and partition state."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from pragmata.core.types import NonEmptyStr


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
    """One record's calibration vs production assignment with import provenance.

    Attributes:
        calibration: True if assigned to the calibration dataset, else production.
        import_id: Identifier of the import call that produced this assignment.
        calibration_fraction_at_import: The fraction in force at that import call.
        assigned_at: When the assignment was made.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibration: bool
    import_id: str
    calibration_fraction_at_import: float = Field(ge=0.0, le=1.0)
    assigned_at: datetime


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

    dataset_id: str
    created_at: datetime
    updated_at: datetime
    partition_seed: int
    assignments: dict[str, PartitionManifestEntry] = Field(default_factory=dict)
