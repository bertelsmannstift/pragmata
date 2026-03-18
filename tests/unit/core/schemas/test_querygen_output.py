"""Unit tests for the synthetic query generation output contract."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_output import SyntheticQueriesMeta, SyntheticQueryRow


@pytest.fixture()
def valid_row_payload() -> dict[str, str | None]:
    """Return a minimal valid synthetic query row payload."""
    return {
        "query_id": "q_001",
        "query": "How do I apply for housing support?",
        "domain": "public_policy",
        "role": "citizen",
        "language": "en",
        "topic": "housing",
        "intent": "information",
        "task": "eligibility",
        "difficulty": "easy",
        "format": "steps",
    }


@pytest.fixture()
def valid_meta_payload() -> dict[str, object]:
    """Return a valid synthetic query metadata payload."""
    return {
        "run_id": "run_20260309_001",
        "created_at": "2026-03-09T10:30:00Z",
        "n_queries": 25,
        "model_provider": "openai",
        "planning_model": "gpt-4.1-mini",
        "realization_model": "gpt-4.1-mini",
    }


def test_synthetic_query_row_accepts_valid_payload(valid_row_payload: dict[str, str | None]) -> None:
    """SyntheticQueryRow validates a complete row payload."""
    row = SyntheticQueryRow.model_validate(valid_row_payload)
    assert row.query_id == "q_001"
    assert row.query == "How do I apply for housing support?"
    assert row.language == "en"


@pytest.mark.parametrize("field_name", ["query_id", "query"])
def test_synthetic_query_row_rejects_blank_required_fields(
    valid_row_payload: dict[str, str | None], field_name: str
) -> None:
    """Required row fields reject blank strings."""
    payload = dict(valid_row_payload)
    payload[field_name] = "   "
    with pytest.raises(ValidationError):
        SyntheticQueryRow.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["domain", "role", "language", "topic", "intent", "task", "difficulty", "format"],
)
def test_synthetic_query_row_optional_fields_accept_none(
    valid_row_payload: dict[str, str | None], field_name: str
) -> None:
    """Optional metadata fields accept None."""
    payload = dict(valid_row_payload)
    payload[field_name] = None
    row = SyntheticQueryRow.model_validate(payload)
    assert getattr(row, field_name) is None


def test_synthetic_query_row_rejects_extra_keys(valid_row_payload: dict[str, str | None]) -> None:
    """SyntheticQueryRow rejects unexpected fields."""
    payload = dict(valid_row_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        SyntheticQueryRow.model_validate(payload)


def test_synthetic_query_row_column_order_matches_model_field_order() -> None:
    """CSV column order is derived from model_fields keys."""
    assert list(SyntheticQueryRow.model_fields.keys()) == [
        "query_id",
        "query",
        "domain",
        "role",
        "language",
        "topic",
        "intent",
        "task",
        "difficulty",
        "format",
    ]


def test_synthetic_queries_meta_accepts_valid_payload(valid_meta_payload: dict[str, object]) -> None:
    """SyntheticQueriesMeta validates a complete metadata payload."""
    meta = SyntheticQueriesMeta.model_validate(valid_meta_payload)
    assert meta.run_id == "run_20260309_001"
    assert meta.created_at == datetime(2026, 3, 9, 10, 30, tzinfo=timezone.utc)
    assert meta.n_queries == 25
    assert meta.model_provider == "openai"
    assert meta.planning_model == "gpt-4.1-mini"
    assert meta.realization_model == "gpt-4.1-mini"


def test_synthetic_queries_meta_rejects_non_positive_n_queries(valid_meta_payload: dict[str, object]) -> None:
    """n_queries must be strictly positive."""
    payload = dict(valid_meta_payload)
    payload["n_queries"] = 0
    with pytest.raises(ValidationError):
        SyntheticQueriesMeta.model_validate(payload)


def test_synthetic_queries_meta_rejects_extra_keys(valid_meta_payload: dict[str, object]) -> None:
    """SyntheticQueriesMeta rejects unexpected fields."""
    payload = dict(valid_meta_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        SyntheticQueriesMeta.model_validate(payload)
