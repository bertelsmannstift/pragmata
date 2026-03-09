"""Structured output contracts for LLM stage 1 query planning."""

from pydantic import BaseModel, ConfigDict, Field


class QueryBlueprint(BaseModel):
    """Single structured candidate blueprint."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="Stable UUID assigned to this candidate query.")
    domain: str = Field(..., description="The setting or subject area assigned to this candidate query.")
    role: str = Field(..., description="The specific persona or perspective assigned to this candidate query.")
    language: str = Field(..., description="The language assigned to this candidate query.")
    topic: str = Field(..., description="The concrete subject matter assigned to this candidate query.")
    intent: str = Field(..., description="The underlying user goal assigned to this candidate query.")
    task: str = Field(..., description="The type of information-processing task assigned to this candidate query.")
    difficulty: str | None = Field(default=None, description="The complexity level assigned to this candidate query.")
    format: str | None = Field(default=None, description="The expected answer format assigned to this candidate query.")
    user_scenario: str = Field(..., min_length=1, max_length=200,
                               description="The realistic user context assigned to this candidate query.")
    information_need: str = Field(..., min_length=1, max_length=200,
                                  description="The specific user information need assigned to this candidate query.")


class QueryBlueprintList(BaseModel):
    """Wrapper for the full list of structured candidate query blueprints."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[QueryBlueprint] = Field(..., description="List of structured candidate query blueprints.")
