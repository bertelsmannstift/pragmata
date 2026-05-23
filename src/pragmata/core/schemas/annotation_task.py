"""Annotation task type enum, shared across import and export schemas."""

from enum import StrEnum

type Locale = str
"""Locale code for Argilla dataset display strings.

Open string (e.g. ``"en"``, ``"de"``, ``"pt-BR"``). Validated at catalog
lookup via :func:`pragmata.core.annotation.locales.registry.get_catalog` —
a deployment adds a locale by dropping ``<code>.yaml`` into
``core/annotation/locales/`` with no Python edit required.

Display strings come from the per-locale catalog; field/question ``name=``
identities and label values are locale-invariant so exports merge cleanly
across locales.
"""


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
