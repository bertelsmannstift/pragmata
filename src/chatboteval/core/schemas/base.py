"""Boundary contract base types. Changes are breaking — update dependents before modifying."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Task(StrEnum):
    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    GENERATION = "generation"


class ContractModel(BaseModel):
    """Frozen, strict base for all boundary contract schemas."""

    model_config = ConfigDict(frozen=True, extra="forbid")
