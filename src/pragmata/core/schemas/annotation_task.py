"""Annotation task type enum, shared across import and export schemas."""

from enum import StrEnum
from typing import Literal


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


type FieldRenderMode = Literal["plain", "markdown"]
"""Rendering mode for a TextField in the Argilla UI.

``markdown`` enables the markdown renderer, which also handles inline raw HTML
(useful when content comes from a pipeline that emits HTML). ``plain`` (default)
renders escaped text. Configurable per field via
:class:`pragmata.core.settings.annotation_settings.AnnotationSettings.field_render_mode`.
"""


TEXT_FIELDS: dict[Task, frozenset[str]] = {
    Task.RETRIEVAL: frozenset({"query", "chunk"}),
    Task.GROUNDING: frozenset({"answer", "context_set"}),
    Task.GENERATION: frozenset({"query", "answer"}),
}
"""Registry of ``rg.TextField`` names per task.

Source of truth for which fields are eligible for
:attr:`AnnotationSettings.field_render_mode` configuration. Drift against
the actual ``rg.TextField(name=...)`` calls in
:mod:`pragmata.core.annotation.argilla_task_definitions` is caught by
``TestTextFieldsRegistryMatchesActualTextFields`` in the test suite.
"""
