"""Unit tests for annotation import — validation logic and record construction.

No Argilla server required. Tests exercise pure Python logic only.
"""

import argilla as rg

from chatboteval.api.annotation_import import (
    RecordError,
    ValidationResult,
    _build_generation_record,
    _build_grounding_record,
    _build_retrieval_records,
    _derive_record_uuid,
    validate_records,
)
from chatboteval.core.schemas.annotation_import import QueryResponsePair

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_raw(*, language: str | None = "de") -> dict:
    return {
        "query": "What is the capital?",
        "answer": "Berlin is the capital.",
        "chunks": [
            {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "text": "Berlin is a city."},
            {"chunk_id": "c2", "doc_id": "d1", "chunk_rank": 2, "text": "It is the capital."},
        ],
        "context_set": "ctx-001",
        "language": language,
    }


def _invalid_raw() -> dict:
    return {"query": "Missing required fields"}


def _make_pair(*, language: str | None = "de") -> QueryResponsePair:
    return QueryResponsePair.model_validate(_valid_raw(language=language))


_UUID = "test-uuid-1234"


# ---------------------------------------------------------------------------
# validate_records
# ---------------------------------------------------------------------------


class TestValidateRecords:
    def test_all_valid(self) -> None:
        result = validate_records([_valid_raw(), _valid_raw()])
        assert isinstance(result, ValidationResult)
        assert len(result.valid) == 2
        assert result.errors == []

    def test_all_invalid(self) -> None:
        result = validate_records([_invalid_raw(), _invalid_raw()])
        assert result.valid == []
        assert len(result.errors) == 2

    def test_mixed(self) -> None:
        result = validate_records([_valid_raw(), _invalid_raw(), _valid_raw()])
        assert len(result.valid) == 2
        assert len(result.errors) == 1
        assert result.errors[0].index == 1

    def test_empty_list(self) -> None:
        result = validate_records([])
        assert result.valid == []
        assert result.errors == []

    def test_per_index_errors(self) -> None:
        raws = [_invalid_raw(), _valid_raw(), _invalid_raw()]
        result = validate_records(raws)
        indices = [e.index for e in result.errors]
        assert indices == [0, 2]

    def test_error_has_detail(self) -> None:
        result = validate_records([_invalid_raw()])
        assert isinstance(result.errors[0], RecordError)
        assert result.errors[0].detail  # non-empty string

    def test_valid_records_are_typed(self) -> None:
        result = validate_records([_valid_raw()])
        assert isinstance(result.valid[0], QueryResponsePair)

    def test_empty_string_fields_are_invalid(self) -> None:
        raw = _valid_raw()
        raw["query"] = "   "
        result = validate_records([raw])
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# _build_retrieval_records
# ---------------------------------------------------------------------------


class TestBuildRetrievalRecords:
    def test_one_record_per_chunk(self) -> None:
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        assert len(records) == len(pair.chunks)

    def test_record_ids_use_enumerate_index(self) -> None:
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        assert records[0].id == f"ret-{_UUID}-0"
        assert records[1].id == f"ret-{_UUID}-1"

    def test_fields_query_chunk_generated_answer(self) -> None:
        pair = _make_pair()
        rec = _build_retrieval_records(pair, _UUID)[0]
        assert rec.fields["query"] == pair.query
        assert rec.fields["chunk"] == pair.chunks[0].text
        assert rec.fields["generated_answer"] == {"text": pair.answer}
        assert "answer" not in rec.fields  # must NOT use "answer"

    def test_metadata_record_uuid(self) -> None:
        pair = _make_pair()
        for rec in _build_retrieval_records(pair, _UUID):
            assert rec.metadata["record_uuid"] == _UUID

    def test_metadata_chunk_fields(self) -> None:
        pair = _make_pair()
        rec = _build_retrieval_records(pair, _UUID)[0]
        chunk = pair.chunks[0]
        assert rec.metadata["chunk_id"] == chunk.chunk_id
        assert rec.metadata["doc_id"] == chunk.doc_id
        assert rec.metadata["chunk_rank"] == chunk.chunk_rank  # from chunk, not enumerate

    def test_chunk_rank_from_chunk_not_enumerate(self) -> None:
        """chunk_rank must come from chunk.chunk_rank, not the loop index."""
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        for i, (rec, chunk) in enumerate(zip(records, pair.chunks)):
            assert rec.metadata["chunk_rank"] == chunk.chunk_rank
            # Confirm the test is meaningful: chunk_rank != enumerate index (rank starts at 1)
            if chunk.chunk_rank != i:
                assert rec.metadata["chunk_rank"] != i

    def test_language_present_when_set(self) -> None:
        pair = _make_pair(language="de")
        rec = _build_retrieval_records(pair, _UUID)[0]
        assert rec.metadata["language"] == "de"

    def test_language_omitted_when_none(self) -> None:
        pair = _make_pair(language=None)
        for rec in _build_retrieval_records(pair, _UUID):
            assert "language" not in rec.metadata


# ---------------------------------------------------------------------------
# _build_grounding_record
# ---------------------------------------------------------------------------


class TestBuildGroundingRecord:
    def test_returns_single_record(self) -> None:
        pair = _make_pair()
        rec = _build_grounding_record(pair, _UUID)
        assert isinstance(rec, rg.Record)

    def test_record_id(self) -> None:
        rec = _build_grounding_record(_make_pair(), _UUID)
        assert rec.id == f"gnd-{_UUID}"

    def test_fields(self) -> None:
        pair = _make_pair()
        rec = _build_grounding_record(pair, _UUID)
        assert rec.fields["answer"] == pair.answer
        assert rec.fields["context_set"] == pair.context_set
        assert rec.fields["query"] == {"text": pair.query}

    def test_metadata_record_uuid(self) -> None:
        rec = _build_grounding_record(_make_pair(), _UUID)
        assert rec.metadata["record_uuid"] == _UUID

    def test_language_present_when_set(self) -> None:
        rec = _build_grounding_record(_make_pair(language="en"), _UUID)
        assert rec.metadata["language"] == "en"

    def test_language_omitted_when_none(self) -> None:
        rec = _build_grounding_record(_make_pair(language=None), _UUID)
        assert "language" not in rec.metadata


# ---------------------------------------------------------------------------
# _build_generation_record
# ---------------------------------------------------------------------------


class TestDeriveRecordUuid:
    def test_deterministic(self) -> None:
        pair = _make_pair()
        assert _derive_record_uuid(pair) == _derive_record_uuid(pair)

    def test_different_content_different_uuid(self) -> None:
        pair_a = _make_pair()
        pair_b = QueryResponsePair.model_validate({**_valid_raw(), "query": "Different query?"})
        assert _derive_record_uuid(pair_a) != _derive_record_uuid(pair_b)

    def test_chunk_order_independent(self) -> None:
        raw = _valid_raw()
        pair_a = QueryResponsePair.model_validate(raw)
        raw_reversed = {**raw, "chunks": list(reversed(raw["chunks"]))}
        pair_b = QueryResponsePair.model_validate(raw_reversed)
        assert _derive_record_uuid(pair_a) == _derive_record_uuid(pair_b)

    def test_returns_hex_string(self) -> None:
        result = _derive_record_uuid(_make_pair())
        assert isinstance(result, str)
        assert len(result) > 0
        # SHA-256 hex digest is 64 chars
        assert all(c in "0123456789abcdef" for c in result)


class TestBuildGenerationRecord:
    def test_returns_single_record(self) -> None:
        rec = _build_generation_record(_make_pair(), _UUID)
        assert isinstance(rec, rg.Record)

    def test_record_id(self) -> None:
        rec = _build_generation_record(_make_pair(), _UUID)
        assert rec.id == f"gen-{_UUID}"

    def test_fields(self) -> None:
        pair = _make_pair()
        rec = _build_generation_record(pair, _UUID)
        assert rec.fields["query"] == pair.query
        assert rec.fields["answer"] == pair.answer
        assert rec.fields["context_set"] == {"text": pair.context_set}

    def test_metadata_record_uuid(self) -> None:
        rec = _build_generation_record(_make_pair(), _UUID)
        assert rec.metadata["record_uuid"] == _UUID

    def test_language_present_when_set(self) -> None:
        rec = _build_generation_record(_make_pair(language="fr"), _UUID)
        assert rec.metadata["language"] == "fr"

    def test_language_omitted_when_none(self) -> None:
        rec = _build_generation_record(_make_pair(language=None), _UUID)
        assert "language" not in rec.metadata
