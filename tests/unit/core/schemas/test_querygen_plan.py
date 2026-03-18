"""Tests for the output contracts for LLM stage 1 query planning."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_plan import QueryBlueprint, QueryBlueprintList


@pytest.fixture()
def base_payload() -> dict[str, object]:
    """Valid QueryBlueprint payload for a publication-search use case."""
    return {
        "candidate_id": "candidate-001",
        "domain": "education policy",
        "role": "policy analyst",
        "language": "en",
        "topic": "teacher shortages",
        "intent": "find evidence",
        "task": "literature search",
        "difficulty": "medium",
        "format": "short list of publications",
        "user_scenario": ("I am preparing a briefing and need sources on teacher shortages in public schools."),
        "information_need": ("Find relevant publications on the causes of teacher shortages in rural areas."),
    }


def test_query_blueprint_accepts_valid_payload(base_payload: dict[str, object]) -> None:
    """Schema accepts a valid blueprint payload."""
    blueprint = QueryBlueprint.model_validate(base_payload)

    assert blueprint.candidate_id == "candidate-001"
    assert blueprint.domain == "education policy"
    assert blueprint.topic == "teacher shortages"
    assert blueprint.intent == "find evidence"
    assert blueprint.task == "literature search"


def test_query_blueprint_allows_optional_fields_to_be_none(
    base_payload: dict[str, object],
) -> None:
    """Schema allows optional difficulty and format fields to be None."""
    payload = {**base_payload, "difficulty": None, "format": None}

    blueprint = QueryBlueprint.model_validate(payload)

    assert blueprint.difficulty is None
    assert blueprint.format is None


@pytest.mark.parametrize("field_name", ["user_scenario", "information_need"])
def test_query_blueprint_rejects_empty_guidance_text(
    base_payload: dict[str, object],
    field_name: str,
) -> None:
    """Schema rejects empty guidance fields."""
    payload = {**base_payload, field_name: ""}

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprint.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field_name,)
    assert errors[0]["type"] == "string_too_short"


@pytest.mark.parametrize("field_name", ["user_scenario", "information_need"])
def test_query_blueprint_rejects_guidance_text_longer_than_200_chars(
    base_payload: dict[str, object],
    field_name: str,
) -> None:
    """Schema rejects guidance fields longer than 200 characters."""
    payload = {**base_payload, field_name: "x" * 201}

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprint.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field_name,)
    assert errors[0]["type"] == "string_too_long"


def test_query_blueprint_rejects_extra_fields(base_payload: dict[str, object]) -> None:
    """Schema rejects unexpected blueprint fields."""
    payload = {**base_payload, "unexpected": "value"}

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprint.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("unexpected",)
    assert errors[0]["type"] == "extra_forbidden"


def test_query_blueprint_list_accepts_valid_candidates(
    base_payload: dict[str, object],
) -> None:
    """Schema accepts a valid list of blueprint payloads."""
    second_payload = {
        **base_payload,
        "candidate_id": "candidate-002",
        "domain": "labour market policy",
        "topic": "minimum wage effects",
        "information_need": ("Find publications on employment effects of minimum wage increases."),
    }

    result = QueryBlueprintList.model_validate({"candidates": [base_payload, second_payload]})

    assert len(result.candidates) == 2
    assert result.candidates[1].domain == "labour market policy"
    assert result.candidates[1].topic == "minimum wage effects"


def test_query_blueprint_list_rejects_extra_fields() -> None:
    """Schema rejects unexpected wrapper fields."""
    payload = {"candidates": [], "unexpected": "value"}

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprintList.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("unexpected",)
    assert errors[0]["type"] == "extra_forbidden"


def test_query_blueprint_list_rejects_invalid_nested_candidate(
    base_payload: dict[str, object],
) -> None:
    """Schema surfaces nested candidate validation errors."""
    invalid_payload = {**base_payload, "candidate_id": "candidate-002", "user_scenario": "x" * 201}

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprintList.model_validate({"candidates": [base_payload, invalid_payload]})

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("candidates", 1, "user_scenario")
    assert errors[0]["type"] == "string_too_long"


def test_query_blueprint_fields_define_descriptions() -> None:
    """Blueprint fields expose descriptions for structured LLM output."""
    for field in QueryBlueprint.model_fields.values():
        assert field.description
        assert field.description.strip()


def test_query_blueprint_list_fields_define_descriptions() -> None:
    """Wrapper fields expose descriptions for structured LLM output."""
    for field in QueryBlueprintList.model_fields.values():
        assert field.description
        assert field.description.strip()


def test_query_blueprint_rejects_missing_required_field(
    base_payload: dict[str, object],
) -> None:
    """Schema rejects payloads missing a required field."""
    payload = dict(base_payload)
    payload.pop("candidate_id")

    with pytest.raises(ValidationError) as exc_info:
        QueryBlueprint.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("candidate_id",)
    assert errors[0]["type"] == "missing"
