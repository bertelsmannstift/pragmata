"""Post-hoc constraint validation for exported annotation rows.

Thin compatibility shim over :mod:`logical_constraints` — the constraint
definitions themselves are the single source of truth and are also consumed
by the annotator-time UI widget. See :mod:`logical_constraints`.
"""

from typing import Callable

from pragmata.core.annotation.logical_constraints import LOGICAL_CONSTRAINTS, LogicalConstraint
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task


def _evaluate(task: Task, row: object) -> list[str]:
    return [c.violation_string() for c in LOGICAL_CONSTRAINTS[task] if c.violated_by(row)]


def check_retrieval(row: RetrievalAnnotation) -> list[str]:
    """Return constraint violation strings for a retrieval annotation row."""
    return _evaluate(Task.RETRIEVAL, row)


def check_grounding(row: GroundingAnnotation) -> list[str]:
    """Return constraint violation strings for a grounding annotation row."""
    return _evaluate(Task.GROUNDING, row)


def check_generation(row: GenerationAnnotation) -> list[str]:
    """Return constraint violation strings for a generation annotation row."""
    return _evaluate(Task.GENERATION, row)


CONSTRAINT_CHECKERS: dict[Task, Callable] = {
    Task.RETRIEVAL: check_retrieval,
    Task.GROUNDING: check_grounding,
    Task.GENERATION: check_generation,
}


def violated_constraints(task: Task, row: object) -> list[LogicalConstraint]:
    """Return the constraints a row violates (the structured form, for callers that need severity)."""
    return [c for c in LOGICAL_CONSTRAINTS[task] if c.violated_by(row)]
