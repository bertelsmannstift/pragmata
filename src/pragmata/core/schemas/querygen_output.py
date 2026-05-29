"""Output contract for synthetic query generation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_summary import PlanningSummaryState
from pragmata.core.types import NonEmptyStr


class SyntheticQueryRow(BaseModel):
    """Schema for one synthetic query."""

    model_config = ConfigDict(extra="forbid")

    query_id: NonEmptyStr
    query: NonEmptyStr
    domain: str | None = None
    role: str | None = None
    language: str | None = None
    topic: str | None = None
    intent: str | None = None
    task: str | None = None
    difficulty: str | None = None
    format: str | None = None


class SyntheticQueriesMeta(BaseModel):
    """Schema for synthetic query dataset-level metadata."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    n_requested_queries: PositiveInt
    n_returned_queries: NonNegativeInt
    model_provider: str
    planning_model: str
    realization_model: str

    @model_validator(mode="after")
    def validate_query_counts(self) -> "SyntheticQueriesMeta":
        """Enforce logical consistency between requested and returned counts."""
        if self.n_returned_queries > self.n_requested_queries:
            raise ValueError("n_returned_queries must be less than or equal to n_requested_queries")
        return self


class PlanningSummaryArtifact(BaseModel):
    """Schema for persisted planning-memory metadata and state."""

    model_config = ConfigDict(extra="forbid")

    spec_fingerprint: NonEmptyStr
    source_run_id: NonEmptyStr
    created_at: datetime
    state: PlanningSummaryState


class PlanningBatchArtifact(BaseModel):
    """Persisted result of one Stage 1 planning batch.

    Written atomically to ``<run_dir>/planning_batches/batch_NNNN.json`` after
    each successful Stage 1 batch. Lets a rerun of the same ``source_run_id``
    skip already-planned batches when the header fields and ``candidate_ids``
    still match.
    """

    model_config = ConfigDict(extra="forbid")

    spec_fingerprint: NonEmptyStr
    pragmata_version: NonEmptyStr
    llm_fingerprint: NonEmptyStr
    source_run_id: NonEmptyStr
    n_queries: PositiveInt
    batch_size: PositiveInt
    batch_idx: NonNegativeInt
    enable_planning_memory: bool
    candidate_ids: list[NonEmptyStr]
    blueprints: list[QueryBlueprint]
    planning_summary_state: PlanningSummaryState | None
    created_at: datetime

    @model_validator(mode="after")
    def validate_blueprint_count(self) -> "PlanningBatchArtifact":
        """Enforce a 1:1 mapping between batch candidate IDs and blueprints."""
        if len(self.candidate_ids) != len(self.blueprints):
            raise ValueError("candidate_ids and blueprints must have equal length")
        return self
