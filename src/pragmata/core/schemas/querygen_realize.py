"""Structured output contracts for LLM stage 2 query realization."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model


class RealizedQuery(BaseModel):
    """One realized query aligned to a stage 2 candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(description="Candidate identifier preserved from the stage 2 input blueprint.")
    query: str = Field(description="Realized user query text for the stage 2 candidate.")


class RealizedQueryList(BaseModel):
    """Collection of realized queries aligned to stage 2 candidates."""

    model_config = ConfigDict(extra="forbid")

    queries: list[RealizedQuery] = Field(description="Realized queries aligned one-to-one with the stage 2 candidates.")


def make_realized_query_list_schema(
    expected_length: int,
) -> type[RealizedQueryList]:
    """Build a realization output schema constrained to one exact batch length."""
    realized_query_list_type = Annotated[
        list[RealizedQuery],
        Field(
            min_length=expected_length,
            max_length=expected_length,
            description=f"{expected_length} realized queries aligned one-to-one with the stage 2 candidates.",
        ),
    ]

    return create_model(
        f"RealizedQueryListLen{expected_length}",
        __base__=RealizedQueryList,
        queries=(realized_query_list_type, ...),
    )
