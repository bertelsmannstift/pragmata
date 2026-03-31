"""Post-hoc constraint validation for exported annotation rows."""

from typing import Callable

from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task


def check_retrieval(row: RetrievalAnnotation) -> list[str]:
    """Return constraint violations for a retrieval annotation row.

    Rules:
    1. evidence_sufficient=True requires topically_relevant=True.
    2. evidence_sufficient=True is incompatible with misleading=True.
    """
    violations: list[str] = []
    if row.evidence_sufficient and not row.topically_relevant:
        violations.append("retrieval: evidence_sufficient=True but topically_relevant=False")
    if row.evidence_sufficient and row.misleading:
        violations.append("retrieval: evidence_sufficient=True but misleading=True")
    return violations


def check_grounding(row: GroundingAnnotation) -> list[str]:
    """Return constraint violations for a grounding annotation row.

    Rules:
    1. contradicted_claim_present=True requires unsupported_claim_present=True.
    2. fabricated_source=True requires source_cited=True.
    """
    violations: list[str] = []
    if row.contradicted_claim_present and not row.unsupported_claim_present:
        violations.append("grounding: contradicted_claim_present=True but unsupported_claim_present=False")
    if row.fabricated_source and not row.source_cited:
        violations.append("grounding: fabricated_source=True but source_cited=False")
    return violations


def check_generation(row: GenerationAnnotation) -> list[str]:  # noqa: ARG001
    """Return constraint violations for a generation annotation row (none defined)."""
    return []


CONSTRAINT_CHECKERS: dict[Task, Callable] = {
    Task.RETRIEVAL: check_retrieval,
    Task.GROUNDING: check_grounding,
    Task.GENERATION: check_generation,
}
