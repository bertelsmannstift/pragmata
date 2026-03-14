"""Boundary schemas for canonical annotation import records."""

from pydantic import BaseModel, ConfigDict, Field

from chatboteval.core.types import NonEmptyStr


class Chunk(BaseModel):
    """Single retrieved chunk within a query-response pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    chunk_rank: int = Field(ge=1)
    text: NonEmptyStr


class QueryResponsePair(BaseModel):
    """Canonical import record pairing a query with its response and chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: NonEmptyStr
    answer: NonEmptyStr
    chunks: list[Chunk] = Field(min_length=1)
    context_set: NonEmptyStr
    language: str | None = None
