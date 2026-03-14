"""Boundary schemas for canonical annotation import records."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _non_empty(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be empty or whitespace-only")
    return v


class Chunk(BaseModel):
    """Single retrieved chunk within a query-response pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    chunk_rank: int = Field(ge=1)
    text: str

    @field_validator("chunk_id", "doc_id", "text", mode="before")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _non_empty(v)


class QueryResponsePair(BaseModel):
    """Canonical import record pairing a query with its response and chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    answer: str
    chunks: list[Chunk] = Field(min_length=1)
    context_set: str
    language: str | None = None

    @field_validator("query", "answer", "context_set", mode="before")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _non_empty(v)
