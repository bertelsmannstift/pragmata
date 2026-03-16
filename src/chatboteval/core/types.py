"""Shared type aliases and typing utilities for core modules."""

from typing import Annotated, TypeVar

from pydantic import BaseModel, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

M = TypeVar("M", bound=BaseModel)
