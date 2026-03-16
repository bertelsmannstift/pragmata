"""Structured output contracts for LLM stage 2 query realization."""

from pydantic import BaseModel, ConfigDict, Field


class RealizedQuery(BaseModel):
    """One realized query aligned to a stage 2 candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        description="Candidate identifier preserved from the stage 2 input blueprint."
    )
    query: str = Field(
        description="Realized user query text for the stage 2 candidate."
    )


class RealizedQueryList(BaseModel):
    """Collection of realized queries aligned to stage 2 candidates."""

    model_config = ConfigDict(extra="forbid")

    queries: list[RealizedQuery] = Field(
        description="Realized queries aligned one-to-one with the stage 2 candidates."
    )
