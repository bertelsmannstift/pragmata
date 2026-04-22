"""Annotation task type enum, shared across import and export schemas."""

from enum import StrEnum


class Task(StrEnum):
    """Annotation task types."""

    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    GENERATION = "generation"


class DiscardReason(StrEnum):
    """Reasons an annotator may discard a record outright."""

    LOW_QUALITY_QUERY = "low_quality_query"
    DUPLICATE = "duplicate"
    UNCLEAR = "unclear"
    BEYOND_DOMAIN_KNOWLEDGE = "beyond_domain_knowledge"
