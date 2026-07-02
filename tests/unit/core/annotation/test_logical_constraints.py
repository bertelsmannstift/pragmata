"""Unit tests for declarative logical constraints.

The Python ``check_*`` helpers consume ``LOGICAL_CONSTRAINTS``; these tests
guard that single source of truth.
"""

from typing import get_args

import pytest

from pragmata.core.annotation.logical_constraints import (
    CONSTRAINT_BY_ID,
    LOGICAL_CONSTRAINTS,
    LogicalConstraint,
    _build_constraint_by_id,
)
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task

# Each task's constraints are evaluated against rows of this annotation model.
TASK_MODELS = {
    Task.RETRIEVAL: RetrievalAnnotation,
    Task.GROUNDING: GroundingAnnotation,
    Task.GENERATION: GenerationAnnotation,
}

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


class TestConstraintById:
    def test_covers_every_catalogue_entry(self):
        all_ids = {c.constraint_id for constraints in LOGICAL_CONSTRAINTS.values() for c in constraints}
        assert set(CONSTRAINT_BY_ID) == all_ids

    def test_duplicate_constraint_id_across_tasks_raises(self):
        duplicated = {
            Task.RETRIEVAL: [
                LogicalConstraint(
                    task=Task.RETRIEVAL,
                    constraint_id="dup",
                    when_question="a",
                    when_value=True,
                    then_question="b",
                    then_value=True,
                )
            ],
            Task.GROUNDING: [
                LogicalConstraint(
                    task=Task.GROUNDING,
                    constraint_id="dup",
                    when_question="c",
                    when_value=True,
                    then_question="d",
                    then_value=True,
                )
            ],
        }
        with pytest.raises(ValueError, match=r"duplicate constraint_id 'dup'"):
            _build_constraint_by_id(duplicated)


# ---------------------------------------------------------------------------
# Catalogue ⇆ annotation protocol contract
#
# ``violated_by`` reads questions via ``getattr(row, q, None)``, so a typo in a
# question name silently never-applies instead of erroring. Pin every catalogue
# entry to a real boolean field on its task's annotation model so a typo or an
# accidental rule change fails here rather than going unnoticed downstream.
# ---------------------------------------------------------------------------


def _accepts_bool(annotation) -> bool:
    """True if a Pydantic field annotation permits ``bool`` (incl. ``bool | None``)."""
    return annotation is bool or bool in get_args(annotation)


class TestCatalogueProtocolContract:
    @pytest.mark.parametrize(
        ("constraint", "question"),
        [
            (c, q)
            for constraints in LOGICAL_CONSTRAINTS.values()
            for c in constraints
            for q in (c.when_question, c.then_question)
        ],
        ids=lambda v: v if isinstance(v, str) else v.constraint_id,
    )
    def test_question_is_a_bool_field_on_its_annotation_model(self, constraint, question):
        model = TASK_MODELS[constraint.task]
        field = model.model_fields.get(question)
        assert field is not None, f"'{question}' is not a field on {model.__name__}"
        assert _accepts_bool(field.annotation), f"'{question}' on {model.__name__} is not a boolean field"

    def test_constraint_task_matches_catalogue_key(self):
        # The dataclass ``task`` must agree with the dict key it is filed under.
        for task, constraints in LOGICAL_CONSTRAINTS.items():
            for c in constraints:
                assert c.task is task, f"{c.constraint_id} filed under {task} but tagged {c.task}"


# ---------------------------------------------------------------------------
# LogicalConstraint semantics
# ---------------------------------------------------------------------------


class _Row:
    """Minimal stand-in for a Pydantic annotation row; exposes attribute access."""

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
