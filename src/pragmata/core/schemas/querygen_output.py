"""Output contract for synthetic query generation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator

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
