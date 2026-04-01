"""Shared type aliases and typing utilities for core modules."""

from typing import Annotated, Protocol, TypeVar

from pydantic import BaseModel, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

M = TypeVar("M", bound=BaseModel)


class HasCandidateId(Protocol):
    """Structural type for objects carrying a candidate ID."""

    candidate_id: str


CandidateItemT = TypeVar("CandidateItemT", bound=HasCandidateId)
