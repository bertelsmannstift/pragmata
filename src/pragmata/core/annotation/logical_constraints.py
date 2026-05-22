"""Declarative logical constraints (SSOT for export-time checks AND the annotator-time UI widget).

A logical constraint is a binary implication on label values: when one question
is answered with a specific yes/no value, another question is constrained to a
specific yes/no value. The same definitions drive:

1. Export-time validation (``constraints.check_*`` returns violation strings)
2. Annotation-time UI warnings/blocks (constraints serialised via
   ``to_widget_payload()`` into ``constraints_field.html``)

Keeping both consumers off one definition guarantees they cannot drift.
"""

from dataclasses import dataclass
from typing import Literal

from pragmata.core.schemas.annotation_task import Task

Severity = Literal["warn", "block"]


@dataclass(frozen=True)
class LogicalConstraint:
    """Binary implication: ``<when_question>=<when_value>`` requires ``<then_question>=<then_value>``.

    ``constraint_id`` is a stable short identifier for the rule; ``message``
    is the annotator-facing explanation rendered by the UI widget. Severity
    is not a property of the rule itself, it is a configurable runtime
    setting (see ``AnnotationSettings.constraint_severity``).
    """

    task: Task
    constraint_id: str
    when_question: str
    when_value: bool
    then_question: str
    then_value: bool
    message: str

    def applies(self, row: object) -> bool:
        """Return True if the antecedent matches (i.e. the constraint is relevant to this row)."""
        return getattr(row, self.when_question, None) == self.when_value

    def violated_by(self, row: object) -> bool:
        """Return True if the antecedent matches but the consequent is not satisfied."""
        if not self.applies(row):
            return False
        return getattr(row, self.then_question, None) != self.then_value

    def violation_string(self) -> str:
        """Human-readable violation string rendered into the per-row CSV ``constraint_details`` column."""
        return (
            f"{self.task.value}: {self.when_question}={self.when_value} but {self.then_question}={not self.then_value}"
        )

    def to_widget_payload(self, severity: Severity) -> dict[str, str]:
        """Serialisable constraint shape for the JS widget; uses string yes/no values.

        ``severity`` is the resolved severity for this constraint
        (from ``AnnotationSettings.constraint_severity``).
        """
        return {
            "constraint_id": self.constraint_id,
            "when_question": self.when_question,
            "when_value": "yes" if self.when_value else "no",
            "then_question": self.then_question,
            "then_value": "yes" if self.then_value else "no",
            "severity": severity,
            "message": self.message,
        }


LOGICAL_CONSTRAINTS: dict[Task, list[LogicalConstraint]] = {
    Task.RETRIEVAL: [
        LogicalConstraint(
            task=Task.RETRIEVAL,
            constraint_id="evidence_requires_relevance",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="topically_relevant",
            then_value=True,
            message=(
                "If a passage provides sufficient evidence to answer the query, it must also be topically relevant."
            ),
        ),
        LogicalConstraint(
            task=Task.RETRIEVAL,
            constraint_id="evidence_excludes_misleading",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="misleading",
            then_value=False,
            message=(
                "A passage marked as providing sufficient evidence is usually not also "
                "misleading. Double-check this combination. Keep it only if the passage "
                "genuinely supports the answer while still being plausibly misleading."
            ),
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
            message=(
                "A contradicted claim is by definition not supported by the context, "
                "so 'unsupported claim present' must also be yes."
            ),
        ),
        LogicalConstraint(
            task=Task.GROUNDING,
            constraint_id="fabricated_requires_cited",
            when_question="fabricated_source",
            when_value=True,
            then_question="source_cited",
            then_value=True,
            message=(
                "A fabricated source can only exist if the answer cites a source, so 'source cited' must also be yes."
            ),
        ),
    ],
    Task.GENERATION: [],
}

# By-id lookup over the flattened catalogue, used by the settings layer to
# validate that override maps reference known constraint_ids.
CONSTRAINT_BY_ID: dict[str, LogicalConstraint] = {
    c.constraint_id: c for cs in LOGICAL_CONSTRAINTS.values() for c in cs
}
