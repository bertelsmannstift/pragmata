"""Unit tests for declarative logical constraints.

The Python ``check_*`` helpers and the JS annotator-time widget both consume
``LOGICAL_CONSTRAINTS``; these tests guard that single source of truth.
"""

import json

import pytest

from pragmata.core.annotation.logical_constraints import (
    LOGICAL_CONSTRAINTS,
    LogicalConstraint,
)
from pragmata.core.schemas.annotation_task import Task

# ---------------------------------------------------------------------------
# Catalogue shape
# ---------------------------------------------------------------------------


class TestCatalogue:
    def test_all_tasks_present(self):
        assert set(LOGICAL_CONSTRAINTS) == set(Task)

    def test_generation_has_no_constraints(self):
        assert LOGICAL_CONSTRAINTS[Task.GENERATION] == []

    def test_retrieval_constraint_ids_unique(self):
        ids = [c.constraint_id for c in LOGICAL_CONSTRAINTS[Task.RETRIEVAL]]
        assert len(ids) == len(set(ids))

    def test_grounding_constraint_ids_unique(self):
        ids = [c.constraint_id for c in LOGICAL_CONSTRAINTS[Task.GROUNDING]]
        assert len(ids) == len(set(ids))

    def test_every_constraint_has_message(self):
        for constraints in LOGICAL_CONSTRAINTS.values():
            for c in constraints:
                assert c.message and len(c.message) > 10


# ---------------------------------------------------------------------------
# LogicalConstraint semantics
# ---------------------------------------------------------------------------


class _Row:
    """Minimal stand-in for a Pydantic annotation row — exposes attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def implication_constraint():
    return LogicalConstraint(
        task=Task.RETRIEVAL,
        constraint_id="ev_req_rel",
        when_question="evidence_sufficient",
        when_value=True,
        then_question="topically_relevant",
        then_value=True,
        severity="block",
        message="If evidence is sufficient the chunk must also be relevant.",
    )


class TestConstraintSemantics:
    def test_applies_when_antecedent_matches(self, implication_constraint):
        row = _Row(evidence_sufficient=True, topically_relevant=False)
        assert implication_constraint.applies(row) is True

    def test_does_not_apply_when_antecedent_false(self, implication_constraint):
        row = _Row(evidence_sufficient=False, topically_relevant=False)
        assert implication_constraint.applies(row) is False

    def test_violated_when_consequent_fails(self, implication_constraint):
        row = _Row(evidence_sufficient=True, topically_relevant=False)
        assert implication_constraint.violated_by(row) is True

    def test_not_violated_when_consequent_holds(self, implication_constraint):
        row = _Row(evidence_sufficient=True, topically_relevant=True)
        assert implication_constraint.violated_by(row) is False

    def test_not_violated_when_antecedent_fails(self, implication_constraint):
        row = _Row(evidence_sufficient=False, topically_relevant=False)
        assert implication_constraint.violated_by(row) is False

    def test_violation_string_format(self, implication_constraint):
        s = implication_constraint.violation_string()
        assert s == "retrieval: evidence_sufficient=True but topically_relevant=False"


# ---------------------------------------------------------------------------
# JS widget payload
# ---------------------------------------------------------------------------


class TestWidgetPayload:
    def test_payload_uses_string_yes_no(self, implication_constraint):
        payload = implication_constraint.to_widget_payload()
        assert payload["when_value"] == "yes"
        assert payload["then_value"] == "yes"

    def test_payload_serialises_no_correctly(self):
        constraint = LogicalConstraint(
            task=Task.RETRIEVAL,
            constraint_id="x",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="misleading",
            then_value=False,
            severity="warn",
            message="...",
        )
        assert constraint.to_widget_payload()["then_value"] == "no"

    def test_payload_carries_severity_and_message(self, implication_constraint):
        payload = implication_constraint.to_widget_payload()
        assert payload["severity"] == "block"
        assert payload["message"].startswith("If evidence")

    def test_payload_is_json_serialisable(self):
        for constraints in LOGICAL_CONSTRAINTS.values():
            json.dumps([c.to_widget_payload() for c in constraints])
