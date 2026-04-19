"""Unit tests for annotation constraint validation."""

from datetime import datetime, timezone

from pragmata.core.annotation.constraints import (
    CONSTRAINT_CHECKERS,
    check_generation,
    check_grounding,
    check_retrieval,
)
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task

_NOW = datetime.now(tz=timezone.utc)

_BASE = {
    "record_uuid": "abc123",
    "annotator_id": "user1",
    "language": "en",
    "inserted_at": _NOW,
    "created_at": _NOW,
    "record_status": "submitted",
    "response_status": "submitted",
}


def _retrieval(**kwargs) -> RetrievalAnnotation:
    defaults = {
        "query": "q",
        "chunk": "c",
        "chunk_id": "cid",
        "doc_id": "did",
        "chunk_rank": 1,
        "topically_relevant": True,
        "evidence_sufficient": False,
        "misleading": False,
    }
    return RetrievalAnnotation.model_validate({**_BASE, **defaults, **kwargs})


def _grounding(**kwargs) -> GroundingAnnotation:
    defaults = {
        "answer": "a",
        "context_set": "ctx",
        "support_present": True,
        "unsupported_claim_present": False,
        "contradicted_claim_present": False,
        "source_cited": True,
        "fabricated_source": False,
    }
    return GroundingAnnotation.model_validate({**_BASE, **defaults, **kwargs})


def _generation(**kwargs) -> GenerationAnnotation:
    defaults = {
        "query": "q",
        "answer": "a",
        "proper_action": True,
        "response_on_topic": True,
        "helpful": True,
        "incomplete": False,
        "unsafe_content": False,
    }
    return GenerationAnnotation.model_validate({**_BASE, **defaults, **kwargs})


# ---------------------------------------------------------------------------
# check_retrieval
# ---------------------------------------------------------------------------


class TestCheckRetrieval:
    def test_rule1_evidence_sufficient_not_relevant(self) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=True)
        violations = check_retrieval(row)
        assert len(violations) == 1

    def test_rule2_evidence_sufficient_and_misleading(self) -> None:
        row = _retrieval(evidence_sufficient=True, misleading=True, topically_relevant=True)
        violations = check_retrieval(row)
        assert len(violations) == 1

    def test_both_violations_cooccur(self) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=True, misleading=True)
        violations = check_retrieval(row)
        assert len(violations) == 2

    def test_valid_all_false(self) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=False, misleading=False)
        assert check_retrieval(row) == []

    def test_valid_relevant_and_sufficient_not_misleading(self) -> None:
        row = _retrieval(topically_relevant=True, evidence_sufficient=True, misleading=False)
        assert check_retrieval(row) == []

    def test_returns_strings(self) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=True)
        violations = check_retrieval(row)
        assert all(isinstance(v, str) for v in violations)


# ---------------------------------------------------------------------------
# check_grounding
# ---------------------------------------------------------------------------


class TestCheckGrounding:
    def test_rule1_contradicted_without_unsupported(self) -> None:
        row = _grounding(contradicted_claim_present=True, unsupported_claim_present=False)
        violations = check_grounding(row)
        assert len(violations) == 1

    def test_rule2_fabricated_without_source_cited(self) -> None:
        row = _grounding(fabricated_source=True, source_cited=False)
        violations = check_grounding(row)
        assert len(violations) == 1

    def test_valid_combination(self) -> None:
        row = _grounding(
            contradicted_claim_present=False,
            unsupported_claim_present=False,
            fabricated_source=False,
            source_cited=True,
        )
        assert check_grounding(row) == []

    def test_returns_strings(self) -> None:
        row = _grounding(contradicted_claim_present=True, unsupported_claim_present=False)
        violations = check_grounding(row)
        assert all(isinstance(v, str) for v in violations)


# ---------------------------------------------------------------------------
# check_generation
# ---------------------------------------------------------------------------


class TestCheckGeneration:
    def test_always_returns_empty(self) -> None:
        row = _generation(
            proper_action=False, response_on_topic=False, helpful=False, incomplete=True, unsafe_content=True
        )
        assert check_generation(row) == []

    def test_empty_for_all_true(self) -> None:
        row = _generation()
        assert check_generation(row) == []


# ---------------------------------------------------------------------------
# CONSTRAINT_CHECKERS dispatch
# ---------------------------------------------------------------------------


class TestConstraintCheckers:
    def test_has_all_tasks(self) -> None:
        assert Task.RETRIEVAL in CONSTRAINT_CHECKERS
        assert Task.GROUNDING in CONSTRAINT_CHECKERS
        assert Task.GENERATION in CONSTRAINT_CHECKERS

    def test_retrieval_checker_dispatches(self) -> None:
        row = _retrieval(topically_relevant=False, evidence_sufficient=True)
        result = CONSTRAINT_CHECKERS[Task.RETRIEVAL](row)
        assert len(result) == 1

    def test_generation_checker_dispatches(self) -> None:
        row = _generation()
        assert CONSTRAINT_CHECKERS[Task.GENERATION](row) == []
