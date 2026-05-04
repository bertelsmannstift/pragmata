"""Tests for annotation import schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_import import (
    Chunk,
    PartitionManifest,
    PartitionManifestEntry,
    QueryResponsePair,
)


@pytest.fixture()
def valid_chunk():
    """Minimal valid Chunk fields."""
    return {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "text": "Some text."}


@pytest.fixture()
def valid_qrp(valid_chunk):
    """Minimal valid QueryResponsePair fields."""
    return {
        "query": "What is X?",
        "answer": "X is Y.",
        "chunks": [valid_chunk],
        "context_set": "ctx-001",
        "language": "en",
    }


def test_valid_qrp_passes(valid_qrp):
    """Valid QueryResponsePair constructs successfully."""
    qrp = QueryResponsePair(**valid_qrp)
    assert qrp.query == "What is X?"


def test_language_none_accepted(valid_qrp):
    """Explicit None is accepted for language."""
    valid_qrp["language"] = None
    qrp = QueryResponsePair(**valid_qrp)
    assert qrp.language is None


def test_language_omitted_accepted(valid_qrp):
    """Omitting language defaults to None."""
    del valid_qrp["language"]
    qrp = QueryResponsePair(**valid_qrp)
    assert qrp.language is None


@pytest.mark.parametrize("field", ["query", "answer", "context_set"])
def test_empty_string_rejected(valid_qrp, field):
    """Empty string is rejected for required text fields."""
    valid_qrp[field] = ""
    with pytest.raises(ValidationError):
        QueryResponsePair(**valid_qrp)


@pytest.mark.parametrize("field", ["query", "answer", "context_set"])
def test_whitespace_only_rejected(valid_qrp, field):
    """Whitespace-only string is rejected for required text fields."""
    valid_qrp[field] = "   "
    with pytest.raises(ValidationError):
        QueryResponsePair(**valid_qrp)


def test_empty_chunks_rejected(valid_qrp):
    """Empty chunks list is rejected."""
    valid_qrp["chunks"] = []
    with pytest.raises(ValidationError):
        QueryResponsePair(**valid_qrp)


def test_chunk_rank_zero_rejected(valid_chunk):
    """Chunk rank of zero is rejected."""
    valid_chunk["chunk_rank"] = 0
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk)


@pytest.mark.parametrize("field", ["chunk_id", "doc_id", "text"])
def test_empty_chunk_string_rejected(valid_chunk, field):
    """Empty string is rejected for chunk text fields."""
    valid_chunk[field] = ""
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk)


@pytest.mark.parametrize("field", ["chunk_id", "doc_id", "text"])
def test_whitespace_chunk_string_rejected(valid_chunk, field):
    """Whitespace-only string is rejected for chunk text fields."""
    valid_chunk[field] = "  "
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk)


def test_qrp_frozen(valid_qrp):
    """QueryResponsePair is immutable."""
    qrp = QueryResponsePair(**valid_qrp)
    with pytest.raises(ValidationError):
        qrp.query = "new"


def test_chunk_frozen(valid_chunk):
    """Chunk is immutable."""
    c = Chunk(**valid_chunk)
    with pytest.raises(ValidationError):
        c.text = "new"


def test_extra_field_on_qrp_rejected(valid_qrp):
    """Extra fields on QueryResponsePair are rejected."""
    valid_qrp["unknown"] = "x"
    with pytest.raises(ValidationError):
        QueryResponsePair(**valid_qrp)


def test_extra_field_on_chunk_rejected(valid_chunk):
    """Extra fields on Chunk are rejected."""
    valid_chunk["unknown"] = "x"
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk)


# ---------------------------------------------------------------------------
# PartitionManifestEntry / PartitionManifest
# ---------------------------------------------------------------------------


_ENTRY_NOW = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def valid_entry_kwargs():
    return {
        "calibration": True,
        "import_id": "imp1",
        "calibration_fraction_at_import": 0.1,
        "assigned_at": _ENTRY_NOW,
    }


@pytest.fixture()
def valid_manifest_kwargs():
    return {
        "dataset_id": "run1",
        "created_at": _ENTRY_NOW,
        "updated_at": _ENTRY_NOW,
        "partition_seed": 0,
    }


class TestPartitionManifestEntry:
    def test_constructs(self, valid_entry_kwargs):
        entry = PartitionManifestEntry(**valid_entry_kwargs)
        assert entry.calibration is True
        assert entry.import_id == "imp1"
        assert entry.calibration_fraction_at_import == 0.1

    def test_fraction_above_one_rejected(self, valid_entry_kwargs):
        valid_entry_kwargs["calibration_fraction_at_import"] = 1.5
        with pytest.raises(ValidationError):
            PartitionManifestEntry(**valid_entry_kwargs)

    def test_fraction_negative_rejected(self, valid_entry_kwargs):
        valid_entry_kwargs["calibration_fraction_at_import"] = -0.1
        with pytest.raises(ValidationError):
            PartitionManifestEntry(**valid_entry_kwargs)

    def test_extra_fields_rejected(self, valid_entry_kwargs):
        valid_entry_kwargs["unknown"] = "x"
        with pytest.raises(ValidationError):
            PartitionManifestEntry(**valid_entry_kwargs)

    def test_frozen(self, valid_entry_kwargs):
        entry = PartitionManifestEntry(**valid_entry_kwargs)
        with pytest.raises(ValidationError):
            entry.calibration = False  # type: ignore[misc]


class TestPartitionManifest:
    def test_empty_assignments_default(self, valid_manifest_kwargs):
        manifest = PartitionManifest(**valid_manifest_kwargs)
        assert manifest.assignments == {}

    def test_assignments_round_trip(self, valid_manifest_kwargs, valid_entry_kwargs):
        entry = PartitionManifestEntry(**valid_entry_kwargs)
        manifest = PartitionManifest(**valid_manifest_kwargs, assignments={"uuid-1": entry})
        restored = PartitionManifest.model_validate_json(manifest.model_dump_json())
        assert restored == manifest

    def test_extra_fields_rejected(self, valid_manifest_kwargs):
        with pytest.raises(ValidationError):
            PartitionManifest(**valid_manifest_kwargs, surprise="bad")  # type: ignore[call-arg]

    def test_dataset_id_can_be_empty_string(self, valid_manifest_kwargs):
        valid_manifest_kwargs["dataset_id"] = ""
        manifest = PartitionManifest(**valid_manifest_kwargs)
        assert manifest.dataset_id == ""
