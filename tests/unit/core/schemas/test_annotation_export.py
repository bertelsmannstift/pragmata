"""Tests for annotation export schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GenerationExportRow,
    GroundingAnnotation,
    GroundingExportRow,
    RetrievalAnnotation,
    RetrievalExportRow,
)
from pragmata.core.schemas.annotation_task import Task

NOW = datetime.now(tz=timezone.utc)


@pytest.fixture()
def base_fields():
    """Minimal valid AnnotationBase fields."""
    return {
        "record_uuid": "uuid-1",
        "annotator_id": "ann-1",
        "language": "en",
        "inserted_at": NOW,
        "created_at": NOW,
        "record_status": "submitted",
        "response_status": "submitted",
    }


@pytest.fixture()
def valid_retrieval(base_fields):
    """Valid retrieval annotation fields."""
    return {
        **base_fields,
        "query": "Q?",
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
    assert r.task == Task.RETRIEVAL
    assert r.notes == ""


def test_grounding_constructs(valid_grounding):
    """Grounding annotation constructs from valid fields."""
    g = GroundingAnnotation(**valid_grounding)
    assert g.context_set == "ctx-001"
    assert g.task == Task.GROUNDING
    assert g.notes == ""


def test_generation_constructs(valid_generation):
    """Generation annotation constructs from valid fields."""
    g = GenerationAnnotation(**valid_generation)
    assert g.query == "Q?"
    assert g.task == Task.GENERATION
    assert g.notes == ""


def test_retrieval_rejects_wrong_task(valid_retrieval):
    """RetrievalAnnotation rejects non-retrieval task values."""
    valid_retrieval["task"] = Task.GROUNDING
    with pytest.raises(ValidationError):
        RetrievalAnnotation(**valid_retrieval)


def test_grounding_rejects_wrong_task(valid_grounding):
    """GroundingAnnotation rejects non-grounding task values."""
    valid_grounding["task"] = Task.RETRIEVAL
    with pytest.raises(ValidationError):
        GroundingAnnotation(**valid_grounding)


def test_generation_rejects_wrong_task(valid_generation):
    """GenerationAnnotation rejects non-generation task values."""
    valid_generation["task"] = Task.GROUNDING
    with pytest.raises(ValidationError):
        GenerationAnnotation(**valid_generation)


def test_notes_default_empty(valid_retrieval):
    """Notes field defaults to empty string."""
    r = RetrievalAnnotation(**valid_retrieval)
    assert r.notes == ""


def test_notes_explicit(valid_retrieval):
    """Explicit notes value is preserved."""
    valid_retrieval["notes"] = "comment"
    r = RetrievalAnnotation(**valid_retrieval)
    assert r.notes == "comment"


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


def test_retrieval_export_row_constructs(valid_retrieval):
    """Retrieval export row constructs from annotation fields plus constraint metadata."""
    row = RetrievalExportRow(**valid_retrieval, constraint_violated=False)
    assert row.chunk_id == "c1"
    assert row.constraint_violated is False
    assert row.constraint_details == ""


def test_grounding_export_row_constructs(valid_grounding):
    row = GroundingExportRow(**valid_grounding, constraint_violated=True, constraint_details="rule_a;rule_b")
    assert row.constraint_violated is True
    assert row.constraint_details == "rule_a;rule_b"


def test_generation_export_row_constructs(valid_generation):
    row = GenerationExportRow(**valid_generation, constraint_violated=False)
    assert row.constraint_details == ""


def test_export_row_frozen(valid_retrieval):
    row = RetrievalExportRow(**valid_retrieval, constraint_violated=False)
    with pytest.raises(ValidationError):
        row.constraint_violated = True


def test_export_row_extra_rejected(valid_retrieval):
    with pytest.raises(ValidationError):
        RetrievalExportRow(**valid_retrieval, constraint_violated=False, unknown="x")


def test_export_row_requires_constraint_violated(valid_retrieval):
    with pytest.raises(ValidationError):
        RetrievalExportRow(**valid_retrieval)


@pytest.mark.parametrize(
    ("row_cls", "annotation_cls"),
    [
        (RetrievalExportRow, RetrievalAnnotation),
        (GroundingExportRow, GroundingAnnotation),
        (GenerationExportRow, GenerationAnnotation),
    ],
)
def test_export_row_field_order(row_cls, annotation_cls):
    """Export row fields are annotation fields followed by the two constraint columns, in order."""
    fields = list(row_cls.model_fields.keys())
    expected = list(annotation_cls.model_fields.keys()) + ["constraint_violated", "constraint_details"]
    assert fields == expected


def test_response_status_rejects_unknown(valid_retrieval):
    """response_status is restricted to submitted/discarded."""
    valid_retrieval["response_status"] = "draft"
    with pytest.raises(ValidationError):
        RetrievalAnnotation(**valid_retrieval)


@pytest.mark.parametrize(
    ("cls_name", "fields_fixture", "label"),
    [
        ("RetrievalAnnotation", "valid_retrieval", "topically_relevant"),
        ("GroundingAnnotation", "valid_grounding", "support_present"),
        ("GenerationAnnotation", "valid_generation", "proper_action"),
    ],
)
def test_submitted_rejects_none_label(cls_name, fields_fixture, label, request):
    """Submitted annotations must have all task-specific bool labels populated."""
    cls = {
        "RetrievalAnnotation": RetrievalAnnotation,
        "GroundingAnnotation": GroundingAnnotation,
        "GenerationAnnotation": GenerationAnnotation,
    }[cls_name]
    fields = request.getfixturevalue(fields_fixture).copy()
    fields[label] = None
    with pytest.raises(ValidationError, match="missing required label"):
        cls(**fields)


@pytest.mark.parametrize(
    ("cls_name", "fields_fixture", "labels"),
    [
        ("RetrievalAnnotation", "valid_retrieval", ("topically_relevant", "evidence_sufficient", "misleading")),
        (
            "GroundingAnnotation",
            "valid_grounding",
            (
                "support_present",
                "unsupported_claim_present",
                "contradicted_claim_present",
                "source_cited",
                "fabricated_source",
            ),
        ),
        (
            "GenerationAnnotation",
            "valid_generation",
            ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"),
        ),
    ],
)
def test_discarded_allows_none_labels(cls_name, fields_fixture, labels, request):
    """Discarded annotations may have all task-specific labels as None."""
    cls = {
        "RetrievalAnnotation": RetrievalAnnotation,
        "GroundingAnnotation": GroundingAnnotation,
        "GenerationAnnotation": GenerationAnnotation,
    }[cls_name]
    fields = request.getfixturevalue(fields_fixture).copy()
    fields["response_status"] = "discarded"
    fields["discard_reason"] = "duplicate"
    for label in labels:
        fields[label] = None
    instance = cls(**fields)
    assert instance.response_status == "discarded"
    for label in labels:
        assert getattr(instance, label) is None
