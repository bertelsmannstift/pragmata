"""Output contract for synthetic query generation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, PositiveInt

from chatboteval.core.types import NonEmptyStr


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
    n_queries: PositiveInt
    model_provider: str
    planning_model: str
    realization_model: str
