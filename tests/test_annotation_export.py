"""Tests for annotation export schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chatboteval.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
    Task,
)

NOW = datetime.now(tz=timezone.utc)


def test_task_values():
    """Task enum members have expected string values."""
    assert Task.RETRIEVAL == "retrieval"
    assert Task.GROUNDING == "grounding"
    assert Task.GENERATION == "generation"


def test_task_from_string():
    """Task can be constructed from its string value."""
    assert Task("retrieval") is Task.RETRIEVAL


def test_task_invalid_raises():
    """Invalid string raises ValueError."""
    with pytest.raises(ValueError):
        Task("invalid")


def test_task_has_three_members():
    """Task enum has exactly three members."""
    assert len(list(Task)) == 3


@pytest.fixture()
def base_fields():
    """Minimal valid AnnotationBase fields."""
    return {
        "record_uuid": "uuid-1",
        "annotator_id": "ann-1",
        "task": Task.RETRIEVAL,
        "language": "en",
        "inserted_at": NOW,
        "created_at": NOW,
        "record_status": "submitted",
    }


@pytest.fixture()
def valid_retrieval(base_fields):
    """Valid retrieval annotation fields."""
    return {
        **base_fields,
        "task": Task.RETRIEVAL,
        "input_query": "Q?",
        "chunk": "Some chunk text.",
        "chunk_id": "c1",
        "doc_id": "d1",
        "chunk_rank": 1,
        "topically_relevant": True,
        "evidence_sufficient": False,
        "misleading": False,
    }


@pytest.fixture()
def valid_grounding(base_fields):
    """Valid grounding annotation fields."""
    return {
        **base_fields,
        "task": Task.GROUNDING,
        "answer": "The answer.",
        "context_set": "ctx-001",
        "support_present": True,
        "unsupported_claim_present": False,
        "contradicted_claim_present": False,
        "source_cited": True,
        "fabricated_source": False,
    }


@pytest.fixture()
def valid_generation(base_fields):
    """Valid generation annotation fields."""
    return {
        **base_fields,
        "task": Task.GENERATION,
        "query": "Q?",
        "answer": "A.",
        "proper_action": True,
        "response_on_topic": True,
        "helpful": True,
        "incomplete": False,
        "unsafe_content": False,
    }


def test_retrieval_constructs(valid_retrieval):
    """Retrieval annotation constructs from valid fields."""
    r = RetrievalAnnotation(**valid_retrieval)
    assert r.chunk_id == "c1"
    assert r.notes == ""


def test_grounding_constructs(valid_grounding):
    """Grounding annotation constructs from valid fields."""
    g = GroundingAnnotation(**valid_grounding)
    assert g.context_set == "ctx-001"
    assert g.notes == ""


def test_generation_constructs(valid_generation):
    """Generation annotation constructs from valid fields."""
    g = GenerationAnnotation(**valid_generation)
    assert g.query == "Q?"
    assert g.notes == ""


def test_notes_default_empty(valid_retrieval):
    """Notes field defaults to empty string."""
    r = RetrievalAnnotation(**valid_retrieval)
    assert r.notes == ""


def test_notes_explicit(valid_retrieval):
    """Explicit notes value is preserved."""
    valid_retrieval["notes"] = "comment"
    r = RetrievalAnnotation(**valid_retrieval)
    assert r.notes == "comment"


def test_annotation_base_field_order(base_fields):
    """AnnotationBase fields appear before task-specific fields."""
    keys = list(RetrievalAnnotation.model_fields.keys())
    base_keys = ["record_uuid", "annotator_id", "task", "language", "inserted_at", "created_at", "record_status"]
    assert keys[: len(base_keys)] == base_keys


def test_retrieval_bool_labels(valid_retrieval):
    """Retrieval label fields are booleans."""
    r = RetrievalAnnotation(**valid_retrieval)
    for f in ("topically_relevant", "evidence_sufficient", "misleading"):
        assert isinstance(getattr(r, f), bool)


def test_grounding_bool_labels(valid_grounding):
    """Grounding label fields are booleans."""
    g = GroundingAnnotation(**valid_grounding)
    for f in (
        "support_present",
        "unsupported_claim_present",
        "contradicted_claim_present",
        "source_cited",
        "fabricated_source",
    ):
        assert isinstance(getattr(g, f), bool)


def test_generation_bool_labels(valid_generation):
    """Generation label fields are booleans."""
    g = GenerationAnnotation(**valid_generation)
    for f in ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"):
        assert isinstance(getattr(g, f), bool)


def test_retrieval_frozen(valid_retrieval):
    """Retrieval annotation is immutable."""
    r = RetrievalAnnotation(**valid_retrieval)
    with pytest.raises(ValidationError):
        r.chunk_id = "new"


def test_grounding_frozen(valid_grounding):
    """Grounding annotation is immutable."""
    g = GroundingAnnotation(**valid_grounding)
    with pytest.raises(ValidationError):
        g.answer = "new"


def test_retrieval_extra_rejected(valid_retrieval):
    """Retrieval annotation rejects extra fields."""
    valid_retrieval["unknown"] = "x"
    with pytest.raises(ValidationError):
        RetrievalAnnotation(**valid_retrieval)


def test_grounding_extra_rejected(valid_grounding):
    """Grounding annotation rejects extra fields."""
    valid_grounding["unknown"] = "x"
    with pytest.raises(ValidationError):
        GroundingAnnotation(**valid_grounding)


def test_generation_extra_rejected(valid_generation):
    """Generation annotation rejects extra fields."""
    valid_generation["unknown"] = "x"
    with pytest.raises(ValidationError):
        GenerationAnnotation(**valid_generation)
