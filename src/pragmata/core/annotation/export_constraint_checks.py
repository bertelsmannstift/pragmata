"""Export-time constraint validation for submitted annotation rows.

Logical constraints are validated twice:

1. **Annotation time** (live UI): the constraints widget
   (``constraints_field.html``, wired up in :mod:`argilla_task_definitions`)
   evaluates each rule against the annotator's currently-selected answers
   and shows a warn/block banner. This is the enforcement layer.
2. **Export time** (post-hoc audit): this module re-evaluates the same rules
   on every submitted row when building the export. Violations are recorded
   in ``constraint_summary`` and ``constraint_details``. This is the audit
   layer that catches anything the UI did not enforce (warn-severity rules,
   client-side JS failures, etc.).

Both halves consume the same definitions in :mod:`logical_constraints`
(the SSOT), so they cannot drift.
"""

from typing import Callable

from pragmata.core.annotation.logical_constraints import LOGICAL_CONSTRAINTS, LogicalConstraint
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task


def _evaluate(task: Task, row: object) -> list[LogicalConstraint]:
    return [c for c in LOGICAL_CONSTRAINTS[task] if c.violated_by(row)]


def check_retrieval(row: RetrievalAnnotation) -> list[LogicalConstraint]:
    """Return violated logical constraints for a retrieval annotation row."""
    return _evaluate(Task.RETRIEVAL, row)


def check_grounding(row: GroundingAnnotation) -> list[LogicalConstraint]:
    """Return violated logical constraints for a grounding annotation row."""
    return _evaluate(Task.GROUNDING, row)


def check_generation(row: GenerationAnnotation) -> list[LogicalConstraint]:
    """Return violated logical constraints for a generation annotation row."""
    return _evaluate(Task.GENERATION, row)


CONSTRAINT_CHECKERS: dict[Task, Callable] = {
    Task.RETRIEVAL: check_retrieval,
    Task.GROUNDING: check_grounding,
    Task.GENERATION: check_generation,
}
