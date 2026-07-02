"""Declarative logical constraints (SSOT for export-time validation checks).

A logical constraint is a binary implication on label values: when one question
is answered with a specific yes/no value, another question is constrained to a
specific yes/no value. Centralising the rules avoids scattered conditionals in
the export code.
"""

from dataclasses import dataclass
from typing import Literal

from pragmata.core.schemas.annotation_task import Task

Severity = Literal["warn", "block"]


@dataclass(frozen=True)
class LogicalConstraint:
    """Binary implication: ``<when_question>=<when_value>`` requires ``<then_question>=<then_value>``.

    ``constraint_id`` is a stable short identifier for the rule.
    """

    task: Task
    constraint_id: str
    when_question: str
    when_value: bool
    then_question: str
    then_value: bool

    def applies(self, row: object) -> bool:
        """Return True if the antecedent matches (i.e. the constraint is relevant to this row)."""
        return getattr(row, self.when_question, None) == self.when_value

    def violated_by(self, row: object) -> bool:
        """Return True if the antecedent matches but the consequent is not satisfied."""
        if not self.applies(row):
            return False
        return getattr(row, self.then_question, None) != self.then_value

    def violation_string(self) -> str:
        """Violation string used in export CSV/constraint_summary."""
        return (
            f"{self.task.value}: {self.when_question}={self.when_value} but {self.then_question}={not self.then_value}"
        )


LOGICAL_CONSTRAINTS: dict[Task, list[LogicalConstraint]] = {
    Task.RETRIEVAL: [
        LogicalConstraint(
            task=Task.RETRIEVAL,
            constraint_id="evidence_requires_relevance",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="topically_relevant",
            then_value=True,
        ),
        LogicalConstraint(
            task=Task.RETRIEVAL,
            constraint_id="evidence_excludes_misleading",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="misleading",
            then_value=False,
        ),
    ],
    Task.GROUNDING: [
        LogicalConstraint(
            task=Task.GROUNDING,
            constraint_id="contradiction_requires_unsupported",
            when_question="contradicted_claim_present",
            when_value=True,
            then_question="unsupported_claim_present",
            then_value=True,
        ),
        LogicalConstraint(
            task=Task.GROUNDING,
            constraint_id="fabricated_requires_cited",
            when_question="fabricated_source",
            when_value=True,
            then_question="source_cited",
            then_value=True,
        ),
    ],
    Task.GENERATION: [],
}


def _build_constraint_by_id(
    catalogue: dict[Task, list[LogicalConstraint]] = LOGICAL_CONSTRAINTS,
) -> dict[str, LogicalConstraint]:
    """By-id lookup over the flattened catalogue.

    Used by the settings layer to validate that override maps reference known
    constraint_ids. Raises eagerly on a duplicate constraint_id rather than
    silently letting the last one win, which would otherwise hide the earlier
    constraint from validation and export checks.
    """
    by_id: dict[str, LogicalConstraint] = {}
    for constraints in catalogue.values():
        for constraint in constraints:
            if constraint.constraint_id in by_id:
                raise ValueError(f"duplicate constraint_id {constraint.constraint_id!r} in constraint catalogue")
            by_id[constraint.constraint_id] = constraint
    return by_id


CONSTRAINT_BY_ID: dict[str, LogicalConstraint] = _build_constraint_by_id()
