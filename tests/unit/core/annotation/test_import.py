"""Unit tests for annotation import — record construction and UUID derivation.

No Argilla server required. Tests exercise pure Python logic only.
"""

import argilla as rg

from pragmata.core.annotation.record_builder import (
    build_generation_record,
    build_grounding_record,
    build_retrieval_records,
    derive_record_uuid,
)
from pragmata.core.schemas.annotation_import import QueryResponsePair

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


def _make_pair(*, language: str | None = "de") -> QueryResponsePair:
    return QueryResponsePair.model_validate(_valid_raw(language=language))


_UUID = "test-uuid-1234"


# ---------------------------------------------------------------------------
# derive_record_uuid
# ---------------------------------------------------------------------------


class TestDeriveRecordUuid:
    def test_deterministic(self) -> None:
        pair = _make_pair()
        assert derive_record_uuid(pair) == derive_record_uuid(pair)

    def test_different_content_different_uuid(self) -> None:
        pair_a = _make_pair()
        pair_b = QueryResponsePair.model_validate({**_valid_raw(), "query": "Different query?"})
        assert derive_record_uuid(pair_a) != derive_record_uuid(pair_b)

    def test_chunk_order_independent(self) -> None:
        raw = _valid_raw()
        pair_a = QueryResponsePair.model_validate(raw)
        raw_reversed = {**raw, "chunks": list(reversed(raw["chunks"]))}
        pair_b = QueryResponsePair.model_validate(raw_reversed)
        assert derive_record_uuid(pair_a) == derive_record_uuid(pair_b)

    def test_returns_hex_string(self) -> None:
        result = derive_record_uuid(_make_pair())
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# build_retrieval_records
# ---------------------------------------------------------------------------


class TestBuildRetrievalRecords:
    def test_one_record_per_chunk(self) -> None:
        pair = _make_pair()
        records = build_retrieval_records(pair, _UUID)
        assert len(records) == len(pair.chunks)

    def test_record_ids_use_enumerate_index(self) -> None:
        pair = _make_pair()
        records = build_retrieval_records(pair, _UUID)
        assert records[0].id == f"ret-{_UUID}-0"
        assert records[1].id == f"ret-{_UUID}-1"

    def test_fields_query_chunk_generated_answer(self) -> None:
        pair = _make_pair()
        rec = build_retrieval_records(pair, _UUID)[0]
        assert rec.fields["query"] == pair.query
        assert rec.fields["chunk"] == pair.chunks[0].text
        assert rec.fields["generated_answer"] == {"text": pair.answer}
        assert "answer" not in rec.fields  # must NOT use "answer"

    def test_metadata_record_uuid(self) -> None:
        pair = _make_pair()
        for rec in build_retrieval_records(pair, _UUID):
            assert rec.metadata["record_uuid"] == _UUID

    def test_metadata_chunk_fields(self) -> None:
        pair = _make_pair()
        rec = build_retrieval_records(pair, _UUID)[0]
        chunk = pair.chunks[0]
        assert rec.metadata["chunk_id"] == chunk.chunk_id
        assert rec.metadata["doc_id"] == chunk.doc_id
        assert rec.metadata["chunk_rank"] == chunk.chunk_rank

    def test_chunk_rank_from_chunk_not_enumerate(self) -> None:
        """chunk_rank must come from chunk.chunk_rank, not the loop index."""
        pair = _make_pair()
        records = build_retrieval_records(pair, _UUID)
        for i, (rec, chunk) in enumerate(zip(records, pair.chunks)):
            assert rec.metadata["chunk_rank"] == chunk.chunk_rank
            if chunk.chunk_rank != i:
                assert rec.metadata["chunk_rank"] != i

    def test_language_present_when_set(self) -> None:
        pair = _make_pair(language="de")
        rec = build_retrieval_records(pair, _UUID)[0]
        assert rec.metadata["language"] == "de"

    def test_language_omitted_when_none(self) -> None:
        pair = _make_pair(language=None)
        for rec in build_retrieval_records(pair, _UUID):
            assert "language" not in rec.metadata


# ---------------------------------------------------------------------------
# build_grounding_record
# ---------------------------------------------------------------------------


class TestBuildGroundingRecord:
    def test_returns_single_record(self) -> None:
        pair = _make_pair()
        rec = build_grounding_record(pair, _UUID)
        assert isinstance(rec, rg.Record)

    def test_record_id(self) -> None:
        rec = build_grounding_record(_make_pair(), _UUID)
        assert rec.id == f"gnd-{_UUID}"

    def test_fields(self) -> None:
        pair = _make_pair()
        rec = build_grounding_record(pair, _UUID)
        assert rec.fields["answer"] == pair.answer
        assert rec.fields["context_set"] == pair.context_set
        assert rec.fields["query"] == {"text": pair.query}

    def test_metadata_record_uuid(self) -> None:
        rec = build_grounding_record(_make_pair(), _UUID)
        assert rec.metadata["record_uuid"] == _UUID

    def test_language_present_when_set(self) -> None:
        rec = build_grounding_record(_make_pair(language="en"), _UUID)
        assert rec.metadata["language"] == "en"

    def test_language_omitted_when_none(self) -> None:
        rec = build_grounding_record(_make_pair(language=None), _UUID)
        assert "language" not in rec.metadata


# ---------------------------------------------------------------------------
# build_generation_record
# ---------------------------------------------------------------------------


class TestBuildGenerationRecord:
    def test_returns_single_record(self) -> None:
        rec = build_generation_record(_make_pair(), _UUID)
        assert isinstance(rec, rg.Record)

    def test_record_id(self) -> None:
        rec = build_generation_record(_make_pair(), _UUID)
        assert rec.id == f"gen-{_UUID}"

    def test_fields(self) -> None:
        pair = _make_pair()
        rec = build_generation_record(pair, _UUID)
        assert rec.fields["query"] == pair.query
        assert rec.fields["answer"] == pair.answer
        assert rec.fields["context_set"] == {"text": pair.context_set}

    def test_metadata_record_uuid(self) -> None:
        rec = build_generation_record(_make_pair(), _UUID)
        assert rec.metadata["record_uuid"] == _UUID

    def test_language_present_when_set(self) -> None:
        rec = build_generation_record(_make_pair(language="fr"), _UUID)
        assert rec.metadata["language"] == "fr"

    def test_language_omitted_when_none(self) -> None:
        rec = build_generation_record(_make_pair(language=None), _UUID)
        assert "language" not in rec.metadata
