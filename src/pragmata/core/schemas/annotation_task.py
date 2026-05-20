"""Annotation task type enum, shared across import and export schemas."""

from enum import StrEnum


class Task(StrEnum):
    """Annotation task types."""

    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    GENERATION = "generation"


class DiscardReason(StrEnum):
    """Reasons an annotator may discard a record outright."""

    INVALID_OR_UNREALISTIC = "invalid_or_unrealistic"
    UNCLEAR = "unclear"
    OUTSIDE_REVIEWER_EXPERTISE = "outside_reviewer_expertise"


class Locale(StrEnum):
    """Supported UI locales for Argilla dataset titles, questions, and guidelines.

    Locale only affects display strings — field/question ``name=`` identities and
    label values remain stable across locales so exports merge cleanly.
    """

    EN = "en"
    DE = "de"
