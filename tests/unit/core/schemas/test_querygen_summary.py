"""Tests for the output contracts for LLM stage 1 planning-memory summary."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_summary import PlanningSummaryState


@pytest.fixture()
def base_payload() -> dict[str, str]:
    """Valid PlanningSummaryState payload for a planning-memory use case."""
    return {
        "redundancy_patterns": (
            "Recurring candidate blueprints focus on first-time eligibility checks for housing "
            "support with very similar applicant scenarios and information needs."
        ),
        "diversification_targets": (
            "Favor spec-compatible candidate blueprints about appeals, document preparation, "
            "deadline clarification, and follow-up procedures with distinct scenarios and information needs."
        ),
        "coverage_notes": (
            "Basic housing-support eligibility scenarios have already appeared and should not be "
            "revisited too closely in the next planning batch."
        ),
    }


def test_planning_summary_state_accepts_valid_payload(base_payload: dict[str, str]) -> None:
    """Schema accepts a valid planning summary payload."""
    summary = PlanningSummaryState.model_validate(base_payload)

    assert summary.redundancy_patterns.startswith("Recurring candidate blueprints")
    assert summary.diversification_targets.startswith("Favor spec-compatible")
    assert summary.coverage_notes.startswith("Basic housing-support")


@pytest.mark.parametrize(
    "field_name",
    ["redundancy_patterns", "diversification_targets", "coverage_notes"],
)
def test_planning_summary_state_rejects_empty_summary_fields(
    base_payload: dict[str, str],
    field_name: str,
) -> None:
    """Schema rejects empty planning-summary fields."""
    payload = {**base_payload, field_name: ""}

    with pytest.raises(ValidationError) as exc_info:
        PlanningSummaryState.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field_name,)
    assert errors[0]["type"] == "string_too_short"


@pytest.mark.parametrize(
    "field_name",
    ["redundancy_patterns", "diversification_targets", "coverage_notes"],
)
def test_planning_summary_state_rejects_fields_longer_than_300_chars(
    base_payload: dict[str, str],
    field_name: str,
) -> None:
    """Schema rejects planning-summary fields longer than 300 characters."""
    payload = {**base_payload, field_name: "x" * 301}

    with pytest.raises(ValidationError) as exc_info:
        PlanningSummaryState.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field_name,)
    assert errors[0]["type"] == "string_too_long"


def test_planning_summary_state_rejects_extra_fields(base_payload: dict[str, str]) -> None:
    """Schema rejects unexpected planning-summary fields."""
    payload = {**base_payload, "unexpected": "value"}

    with pytest.raises(ValidationError) as exc_info:
        PlanningSummaryState.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("unexpected",)
    assert errors[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    "field_name",
    ["redundancy_patterns", "diversification_targets", "coverage_notes"],
)
def test_planning_summary_state_rejects_missing_required_field(
    base_payload: dict[str, str],
    field_name: str,
) -> None:
    """Schema rejects payloads missing a required planning-summary field."""
    payload = dict(base_payload)
    payload.pop(field_name)

    with pytest.raises(ValidationError) as exc_info:
        PlanningSummaryState.model_validate(payload)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field_name,)
    assert errors[0]["type"] == "missing"


def test_planning_summary_state_fields_define_descriptions() -> None:
    """Planning summary fields expose descriptions for structured LLM output."""
    for field in PlanningSummaryState.model_fields.values():
        assert field.description
        assert field.description.strip()
