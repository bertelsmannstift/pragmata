"""Unit tests for core/annotation/export_fetcher — Argilla client is fully mocked."""

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from pragmata.core.annotation.export_fetcher import build_user_lookup, fetch_task
from pragmata.core.schemas.annotation_export import (
    GenerationAnnotation,
    GroundingAnnotation,
    RetrievalAnnotation,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_UID1 = UUID("00000000-0000-0000-0000-000000000001")
_UID2 = UUID("00000000-0000-0000-0000-000000000002")
_SETTINGS = AnnotationSettings()


def _make_response(question_name: str, value: Any, user_id: UUID, status: str = "submitted") -> MagicMock:
    resp = MagicMock()
    resp.question_name = question_name
    resp.value = value
    resp.user_id = user_id
    resp.status = status
    return resp


def _make_record(
    *,
    fields: dict[str, str],
    metadata: dict,
    responses: list[MagicMock],
    updated_at: datetime | None = None,
    inserted_at: datetime | None = None,
    status: str = "submitted",
) -> MagicMock:
    record = MagicMock()
    record.fields = fields
    record.metadata = metadata
    record.responses = responses
    record.status = status
    record._model = MagicMock()
    record._model.updated_at = updated_at or datetime(2024, 1, 1, tzinfo=UTC)
    record._model.inserted_at = inserted_at or datetime(2024, 1, 2, tzinfo=UTC)
    return record


def _retrieval_responses(
    user_id: UUID,
    *,
    topically_relevant="yes",
    evidence_sufficient="yes",
    misleading="no",
    notes="",
) -> list[MagicMock]:
    return [
        _make_response("topically_relevant", topically_relevant, user_id),
        _make_response("evidence_sufficient", evidence_sufficient, user_id),
        _make_response("misleading", misleading, user_id),
        _make_response("notes", notes, user_id),
    ]


def _grounding_responses(user_id: UUID) -> list[MagicMock]:
    return [
        _make_response("support_present", "yes", user_id),
        _make_response("unsupported_claim_present", "no", user_id),
        _make_response("contradicted_claim_present", "no", user_id),
        _make_response("source_cited", "no", user_id),
        _make_response("fabricated_source", "no", user_id),
        _make_response("notes", "", user_id),
    ]


def _generation_responses(user_id: UUID) -> list[MagicMock]:
    return [
        _make_response("proper_action", "yes", user_id),
        _make_response("response_on_topic", "yes", user_id),
        _make_response("helpful", "yes", user_id),
        _make_response("incomplete", "no", user_id),
        _make_response("unsafe_content", "no", user_id),
        _make_response("notes", "", user_id),
    ]


_RETRIEVAL_FIELDS = {"query": "What is X?", "chunk": "X is Y."}
_GROUNDING_FIELDS = {"answer": "X is Y.", "context_set": "Y explains X."}
_GENERATION_FIELDS = {"query": "What is X?", "answer": "X is Y."}
_BASE_METADATA = {
    "record_uuid": "abc123",
    "language": "en",
    "chunk_id": "chunk-1",
    "doc_id": "doc-1",
    "chunk_rank": 1,
}


def _mock_client_with_records(records: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    dataset = MagicMock()
    dataset.records.return_value = iter(records)
    client.datasets.return_value = dataset
    return client


# ---------------------------------------------------------------------------
# build_user_lookup
# ---------------------------------------------------------------------------


class TestBuildUserLookup:
    def test_maps_user_ids_to_usernames(self) -> None:
        client = MagicMock()
        u1 = MagicMock(id=_UID1, username="alice")
        u2 = MagicMock(id=_UID2, username="bob")
        client.users.list.return_value = [u1, u2]

        assert build_user_lookup(client) == {_UID1: "alice", _UID2: "bob"}

    def test_empty_users(self) -> None:
        client = MagicMock()
        client.users.list.return_value = []
        assert build_user_lookup(client) == {}


# ---------------------------------------------------------------------------
# fetch_task
# ---------------------------------------------------------------------------


class TestFetchTask:
    def test_retrieval_record_builds_typed_model(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        assert len(rows) == 1
        model, violations = rows[0]
        assert isinstance(model, RetrievalAnnotation)
        assert model.query == "What is X?"
        assert model.topically_relevant is True
        assert model.annotator_id == "alice"

    def test_grounding_record_builds_typed_model(self) -> None:
        record = _make_record(
            fields=_GROUNDING_FIELDS,
            metadata=_BASE_METADATA,
            responses=_grounding_responses(_UID1),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.GROUNDING, {_UID1: "alice"}, include_discarded=False)

        assert len(rows) == 1
        model, _ = rows[0]
        assert isinstance(model, GroundingAnnotation)
        assert model.support_present is True

    def test_generation_record_builds_typed_model(self) -> None:
        record = _make_record(
            fields=_GENERATION_FIELDS,
            metadata=_BASE_METADATA,
            responses=_generation_responses(_UID1),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.GENERATION, {_UID1: "alice"}, include_discarded=False)

        assert len(rows) == 1
        model, _ = rows[0]
        assert isinstance(model, GenerationAnnotation)
        assert model.helpful is True

    def test_two_annotators_produce_two_rows(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1) + _retrieval_responses(_UID2),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice", _UID2: "bob"}, include_discarded=False)

        assert len(rows) == 2
        annotators = {r[0].annotator_id for r in rows}
        assert annotators == {"alice", "bob"}

    def test_yes_no_converted_to_bool(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1, topically_relevant="yes", evidence_sufficient="no"),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        model, _ = rows[0]
        assert model.topically_relevant is True
        assert model.evidence_sufficient is False

    def test_constraint_violations_attached(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1, topically_relevant="no", evidence_sufficient="yes"),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        _, violations = rows[0]
        assert len(violations) >= 1

    def test_empty_dataset_returns_empty(self) -> None:
        client = _mock_client_with_records([])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)
        assert rows == []

    def test_missing_uuid_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        metadata = {k: v for k, v in _BASE_METADATA.items() if k != "record_uuid"}
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=metadata,
            responses=_retrieval_responses(_UID1),
        )
        client = _mock_client_with_records([record])

        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.export_fetcher"):
            rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        assert len(rows) == 1
        assert rows[0][0].record_uuid == ""
        assert any("record_uuid" in msg.lower() for msg in caplog.messages)

    def test_unknown_user_id_falls_back_to_uuid_string(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {}, include_discarded=False)  # empty lookup

        assert rows[0][0].annotator_id == str(_UID1)

    def test_created_at_prefers_updated_at(self) -> None:
        updated = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1),
            updated_at=updated,
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        assert rows[0][0].created_at == updated

    def test_created_at_falls_back_to_inserted_at(self) -> None:
        inserted = datetime(2024, 3, 10, 8, 0, tzinfo=UTC)
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1),
            inserted_at=inserted,
        )
        record._model.updated_at = None
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        assert rows[0][0].created_at == inserted

    def test_notes_none_coerced_to_empty_string(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1, notes=None),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        assert rows[0][0].notes == ""

    def test_discarded_response_included(self) -> None:
        responses = [
            _make_response("discard_reason", "invalid_or_unrealistic", _UID1, status="discarded"),
            _make_response("discard_notes", "same as Q42", _UID1, status="discarded"),
        ]
        record = _make_record(fields=_RETRIEVAL_FIELDS, metadata=_BASE_METADATA, responses=responses)
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=True)

        assert len(rows) == 1
        model, violations = rows[0]
        assert model.response_status == "discarded"
        assert model.topically_relevant is None
        assert violations == []

    def test_discard_reason_propagated(self) -> None:
        responses = [
            _make_response("discard_reason", "invalid_or_unrealistic", _UID1, status="discarded"),
        ]
        record = _make_record(fields=_RETRIEVAL_FIELDS, metadata=_BASE_METADATA, responses=responses)
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=True)

        assert rows[0][0].discard_reason == "invalid_or_unrealistic"

    def test_discard_notes_propagated(self) -> None:
        responses = [
            _make_response("discard_reason", "unclear", _UID1, status="discarded"),
            _make_response("discard_notes", "query is ambiguous", _UID1, status="discarded"),
        ]
        record = _make_record(fields=_RETRIEVAL_FIELDS, metadata=_BASE_METADATA, responses=responses)
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=True)

        assert rows[0][0].discard_notes == "query is ambiguous"

    def test_discard_fields_default_when_absent(self) -> None:
        record = _make_record(
            fields=_RETRIEVAL_FIELDS,
            metadata=_BASE_METADATA,
            responses=_retrieval_responses(_UID1),
        )
        client = _mock_client_with_records([record])
        rows = fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        model, _ = rows[0]
        assert model.response_status == "submitted"
        assert model.discard_reason is None
        assert model.discard_notes is None

    def test_submitted_only_query_when_include_discarded_false(self) -> None:
        client = _mock_client_with_records([])
        fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=False)

        query = client.datasets.return_value.records.call_args.args[0]
        assert query.filter.conditions == [("response.status", "in", ["submitted"])]

    def test_include_discarded_query_covers_both_statuses(self) -> None:
        client = _mock_client_with_records([])
        fetch_task(client, _SETTINGS, Task.RETRIEVAL, {_UID1: "alice"}, include_discarded=True)

        query = client.datasets.return_value.records.call_args.args[0]
        assert query.filter.conditions == [("response.status", "in", ["submitted", "discarded"])]
