"""Unit tests for annotation import — record construction and UUID derivation.

No Argilla server required. Tests exercise pure Python logic only.
"""

import argilla as rg

from pragmata.core.annotation.record_builder import (
    _chunk_id_digest,
    build_generation_record,
    build_grounding_record,
    build_retrieval_record_for_chunk,
    derive_record_uuid,
)
from pragmata.core.schemas.annotation_import import QueryResponsePair


def _build_retrieval_records(pair: QueryResponsePair, record_uuid: str) -> list:
    """Mirror the production fan-out loop (record_builder._build_batches)."""
    return [build_retrieval_record_for_chunk(pair, record_uuid, chunk) for chunk in pair.chunks]


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
# _build_retrieval_records
# ---------------------------------------------------------------------------


class TestBuildRetrievalRecords:
    def test_one_record_per_chunk(self) -> None:
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        assert len(records) == len(pair.chunks)

    def test_record_ids_use_chunk_digest(self) -> None:
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        assert records[0].id == f"ret-{_UUID}-{_chunk_id_digest('c1')}"
        assert records[1].id == f"ret-{_UUID}-{_chunk_id_digest('c2')}"

    def test_record_ids_stable_under_chunk_reorder(self) -> None:
        raw = _valid_raw()
        pair = QueryResponsePair.model_validate(raw)
        pair_reordered = QueryResponsePair.model_validate({**raw, "chunks": list(reversed(raw["chunks"]))})
        ids = {r.metadata["chunk_id"]: r.id for r in _build_retrieval_records(pair, _UUID)}
        ids_reordered = {r.metadata["chunk_id"]: r.id for r in _build_retrieval_records(pair_reordered, _UUID)}
        assert ids == ids_reordered

    def test_fields_query_chunk_generated_answer(self) -> None:
        pair = _make_pair()
        rec = _build_retrieval_records(pair, _UUID)[0]
        assert rec.fields["query"] == pair.query
        assert rec.fields["chunk"] == pair.chunks[0].text
        assert rec.fields["generated_answer"] == {"text": pair.answer}
        assert rec.fields["discard_flow"] == {"text": ""}
        assert rec.fields["constraints_panel"] == {"text": ""}
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
        assert rec.metadata["chunk_rank"] == chunk.chunk_rank

    def test_n_retrieved_chunks_stamped_on_every_chunk_record(self) -> None:
        """K = len(pair.chunks) stamped uniformly on every chunk-record of the panel."""
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        k = len(pair.chunks)
        assert k >= 2
        for rec in records:
            assert rec.metadata["n_retrieved_chunks"] == k

    def test_chunk_rank_from_chunk_not_enumerate(self) -> None:
        """chunk_rank must come from chunk.chunk_rank, not the loop index."""
        pair = _make_pair()
        records = _build_retrieval_records(pair, _UUID)
        for i, (rec, chunk) in enumerate(zip(records, pair.chunks)):
            assert rec.metadata["chunk_rank"] == chunk.chunk_rank
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
        assert rec.fields["discard_flow"] == {"text": ""}
        assert rec.fields["constraints_panel"] == {"text": ""}

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
        assert rec.fields["discard_flow"] == {"text": ""}
        assert rec.fields["constraints_panel"] == {"text": ""}

    def test_metadata_record_uuid(self) -> None:
        rec = build_generation_record(_make_pair(), _UUID)
        assert rec.metadata["record_uuid"] == _UUID

    def test_language_present_when_set(self) -> None:
        rec = build_generation_record(_make_pair(language="fr"), _UUID)
        assert rec.metadata["language"] == "fr"

    def test_language_omitted_when_none(self) -> None:
        rec = build_generation_record(_make_pair(language=None), _UUID)
        assert "language" not in rec.metadata
