"""Shared type aliases and typing utilities for core modules."""

from typing import Annotated, Protocol, TypeVar

from pydantic import AfterValidator, BaseModel, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def _validate_safe_path_segment(value: str) -> str:
    if value == "":
        return value
    if value != value.strip():
        raise ValueError("must not have surrounding whitespace")
    if "/" in value or "\\" in value:
        raise ValueError("must not contain path separators")
    if ".." in value:
        raise ValueError("must not contain '..'")
    return value


SafePathSegment = Annotated[str, AfterValidator(_validate_safe_path_segment)]

M = TypeVar("M", bound=BaseModel)


class HasCandidateId(Protocol):
    """Structural type for objects carrying a candidate ID."""

    candidate_id: str


CandidateItemT = TypeVar("CandidateItemT", bound=HasCandidateId)
