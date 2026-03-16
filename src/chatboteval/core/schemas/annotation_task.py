"""Annotation task type enum, shared across import and export schemas."""

from enum import StrEnum


class Task(StrEnum):
    """Annotation task types."""

    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    GENERATION = "generation"
