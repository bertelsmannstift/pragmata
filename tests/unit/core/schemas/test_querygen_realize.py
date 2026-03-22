"""Unit tests for the stage 2 query realization structured output contract."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_realize import RealizedQuery, RealizedQueryList


@pytest.fixture()
def valid_realized_query_payload() -> dict[str, str]:
    """Return a valid realized query payload."""
    return {
        "candidate_id": "cand_001",
        "query": "How do I apply for housing support in Berlin?",
    }


@pytest.fixture()
def valid_realized_query_list_payload() -> dict[str, list[dict[str, str]]]:
    """Return a valid realized query list payload."""
    return {
        "queries": [
            {
                "candidate_id": "cand_001",
                "query": "How do I apply for housing support in Berlin?",
            },
            {
                "candidate_id": "cand_002",
                "query": "What documents do I need to apply for housing support?",
            },
        ]
    }


def test_realized_query_accepts_valid_payload(valid_realized_query_payload: dict[str, str]) -> None:
    """RealizedQuery validates a complete payload."""
    realized_query = RealizedQuery.model_validate(valid_realized_query_payload)
    assert realized_query.candidate_id == "cand_001"
    assert realized_query.query == "How do I apply for housing support in Berlin?"


def test_realized_query_rejects_extra_keys(valid_realized_query_payload: dict[str, str]) -> None:
    """RealizedQuery rejects unexpected fields."""
    payload = dict(valid_realized_query_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        RealizedQuery.model_validate(payload)


def test_realized_query_list_accepts_valid_payload(
    valid_realized_query_list_payload: dict[str, list[dict[str, str]]],
) -> None:
    """RealizedQueryList validates a complete payload."""
    realized_query_list = RealizedQueryList.model_validate(valid_realized_query_list_payload)
    assert len(realized_query_list.queries) == 2
    assert realized_query_list.queries[0].candidate_id == "cand_001"
    assert realized_query_list.queries[1].query == "What documents do I need to apply for housing support?"


def test_realized_query_list_rejects_extra_keys(
    valid_realized_query_list_payload: dict[str, list[dict[str, str]]],
) -> None:
    """RealizedQueryList rejects unexpected fields."""
    payload = dict(valid_realized_query_list_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        RealizedQueryList.model_validate(payload)


def test_realized_query_field_descriptions_are_defined() -> None:
    """RealizedQuery fields include explicit descriptions."""
    assert (
        RealizedQuery.model_fields["candidate_id"].description
        == "Candidate identifier preserved from the stage 2 input blueprint."
    )
    assert RealizedQuery.model_fields["query"].description == "Realized user query text for the stage 2 candidate."


def test_realized_query_list_field_descriptions_are_defined() -> None:
    """RealizedQueryList fields include explicit descriptions."""
    assert (
        RealizedQueryList.model_fields["queries"].description
        == "Realized queries aligned one-to-one with the stage 2 candidates."
    )


def test_realized_query_list_rejects_nested_items_with_extra_keys(
    valid_realized_query_list_payload: dict[str, list[dict[str, str]]],
) -> None:
    """RealizedQueryList rejects nested query items with unexpected fields."""
    payload = {"queries": [dict(valid_realized_query_list_payload["queries"][0], unexpected="boom")]}
    with pytest.raises(ValidationError):
        RealizedQueryList.model_validate(payload)


def test_realized_query_list_rejects_nested_items_missing_required_fields() -> None:
    """RealizedQueryList rejects nested query items missing required fields."""
    payload = {
        "queries": [
            {
                "candidate_id": "cand_001",
            }
        ]
    }
    with pytest.raises(ValidationError):
        RealizedQueryList.model_validate(payload)
