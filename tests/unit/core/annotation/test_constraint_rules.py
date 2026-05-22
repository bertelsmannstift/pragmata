"""Unit tests for declarative constraint rules.

The Python ``check_*`` helpers and the JS annotator-time widget both consume
``CONSTRAINT_RULES``; these tests guard that single source of truth.
"""

import json

import pytest

from pragmata.core.annotation.constraint_rules import (
    CONSTRAINT_RULES,
    ConstraintRule,
)
from pragmata.core.schemas.annotation_task import Task

# ---------------------------------------------------------------------------
# Catalogue shape
# ---------------------------------------------------------------------------


class TestCatalogue:
    def test_all_tasks_present(self):
        assert set(CONSTRAINT_RULES) == set(Task)

    def test_generation_has_no_rules(self):
        assert CONSTRAINT_RULES[Task.GENERATION] == []

    def test_retrieval_rule_ids_unique(self):
        ids = [r.rule_id for r in CONSTRAINT_RULES[Task.RETRIEVAL]]
        assert len(ids) == len(set(ids))

    def test_grounding_rule_ids_unique(self):
        ids = [r.rule_id for r in CONSTRAINT_RULES[Task.GROUNDING]]
        assert len(ids) == len(set(ids))

    def test_every_rule_has_message(self):
        for rules in CONSTRAINT_RULES.values():
            for r in rules:
                assert r.message and len(r.message) > 10


# ---------------------------------------------------------------------------
# ConstraintRule semantics
# ---------------------------------------------------------------------------


class _Row:
    """Minimal stand-in for a Pydantic annotation row — exposes attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def implication_rule():
    return ConstraintRule(
        task=Task.RETRIEVAL,
        rule_id="ev_req_rel",
        when_question="evidence_sufficient",
        when_value=True,
        then_question="topically_relevant",
        then_value=True,
        severity="block",
        message="If evidence is sufficient the chunk must also be relevant.",
    )


class TestRuleSemantics:
    def test_applies_when_antecedent_matches(self, implication_rule):
        row = _Row(evidence_sufficient=True, topically_relevant=False)
        assert implication_rule.applies(row) is True

    def test_does_not_apply_when_antecedent_false(self, implication_rule):
        row = _Row(evidence_sufficient=False, topically_relevant=False)
        assert implication_rule.applies(row) is False

    def test_violated_when_consequent_fails(self, implication_rule):
        row = _Row(evidence_sufficient=True, topically_relevant=False)
        assert implication_rule.violated_by(row) is True

    def test_not_violated_when_consequent_holds(self, implication_rule):
        row = _Row(evidence_sufficient=True, topically_relevant=True)
        assert implication_rule.violated_by(row) is False

    def test_not_violated_when_antecedent_fails(self, implication_rule):
        row = _Row(evidence_sufficient=False, topically_relevant=False)
        assert implication_rule.violated_by(row) is False

    def test_violation_string_format(self, implication_rule):
        s = implication_rule.violation_string()
        assert s == "retrieval: evidence_sufficient=True but topically_relevant=False"


# ---------------------------------------------------------------------------
# JS widget payload
# ---------------------------------------------------------------------------


class TestWidgetPayload:
    def test_payload_uses_string_yes_no(self, implication_rule):
        payload = implication_rule.to_widget_payload()
        assert payload["when_value"] == "yes"
        assert payload["then_value"] == "yes"

    def test_payload_serialises_no_correctly(self):
        rule = ConstraintRule(
            task=Task.RETRIEVAL,
            rule_id="x",
            when_question="evidence_sufficient",
            when_value=True,
            then_question="misleading",
            then_value=False,
            severity="warn",
            message="...",
        )
        assert rule.to_widget_payload()["then_value"] == "no"

    def test_payload_carries_severity_and_message(self, implication_rule):
        payload = implication_rule.to_widget_payload()
        assert payload["severity"] == "block"
        assert payload["message"].startswith("If evidence")

    def test_payload_is_json_serialisable(self):
        for rules in CONSTRAINT_RULES.values():
            json.dumps([r.to_widget_payload() for r in rules])
