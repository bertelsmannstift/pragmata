"""Unit tests for the synthetic query generation input contract."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_input import QueryGenSpec, WeightedValue


@pytest.fixture()
def base_payload() -> dict:
    """Minimal valid QueryGenSpec payload."""
    return {
        "domain_context": {"domains": "public_policy", "roles": "citizen", "languages": "en"},
        "knowledge_scope": {"topics": "housing"},
        "scenario": {"intents": "information", "tasks": "eligibility"},
    }


def test_scalar_choice_canonicalizes(base_payload: dict) -> None:
    """Scalar input canonicalizes to a single weighted choice."""
    spec = QueryGenSpec.model_validate(base_payload)
    assert spec.domain_context.languages == [WeightedValue(value="en", weight=1.0)]


def test_list_str_choice_uniform(base_payload: dict) -> None:
    """list[str] input canonicalizes to a uniform distribution."""
    base_payload["domain_context"]["languages"] = ["en", "de"]
    spec = QueryGenSpec.model_validate(base_payload)
    assert spec.domain_context.languages == [
        WeightedValue(value="en", weight=0.5),
        WeightedValue(value="de", weight=0.5),
    ]


def test_weighted_list_choice_preserves_weights(base_payload: dict) -> None:
    """Weighted list input preserves weights when they sum to 1."""
    base_payload["knowledge_scope"]["topics"] = [
        {"value": "housing", "weight": 0.7},
        {"value": "energy", "weight": 0.3},
    ]
    spec = QueryGenSpec.model_validate(base_payload)
    assert spec.knowledge_scope.topics == [
        WeightedValue(value="housing", weight=0.7),
        WeightedValue(value="energy", weight=0.3),
    ]


def test_weighted_list_rejects_sum_not_one(base_payload: dict) -> None:
    """Weighted list rejects weights that do not sum to 1."""
    base_payload["knowledge_scope"]["topics"] = [
        {"value": "housing", "weight": 0.2},
        {"value": "energy", "weight": 0.2},
    ]
    with pytest.raises(ValidationError):
        QueryGenSpec.model_validate(base_payload)


def test_weighted_list_rejects_negative_weight(base_payload: dict) -> None:
    """Weighted list rejects negative weights."""
    base_payload["knowledge_scope"]["topics"] = [
        {"value": "housing", "weight": 1.1},
        {"value": "energy", "weight": -0.1},
    ]
    with pytest.raises(ValidationError):
        QueryGenSpec.model_validate(base_payload)


def test_choice_str_rejects_empty_list() -> None:
    """ChoiceStr rejects empty lists."""
    with pytest.raises(ValidationError, match="ChoiceStr list must not be empty"):
        QueryGenSpec.model_validate(
            {
                "domain_context": {
                    "domains": [],
                    "roles": ["caseworker"],
                    "languages": ["en"],
                },
                "knowledge_scope": {"topics": ["eligibility"]},
                "scenario": {"intents": ["ask"], "tasks": ["check"]},
            }
        )


def test_choice_str_rejects_mixed_list() -> None:
    """ChoiceStr rejects lists mixing strings and weighted values."""
    with pytest.raises(
        ValidationError,
        match="ChoiceStr list must contain either only strings or only weighted values",
    ):
        QueryGenSpec.model_validate(
            {
                "domain_context": {
                    "domains": ["policy", {"value": "benefits", "weight": 0.5}],
                    "roles": ["caseworker"],
                    "languages": ["en"],
                },
                "knowledge_scope": {"topics": ["eligibility"]},
                "scenario": {"intents": ["ask"], "tasks": ["check"]},
            }
        )


def test_optional_choice_fields_accept_none(base_payload: dict) -> None:
    """Optional ChoiceStr fields accept None."""
    base_payload["scenario"]["difficulty"] = None
    base_payload["format_requests"] = {"formats": None}
    spec = QueryGenSpec.model_validate(base_payload)
    assert spec.scenario.difficulty is None
    assert spec.format_requests.formats is None


def test_choice_str_rejects_none_for_required_field(base_payload: dict) -> None:
    """Required ChoiceStr fields reject None."""
    base_payload["domain_context"]["languages"] = None
    with pytest.raises(ValidationError, match="ChoiceStr must not be None"):
        QueryGenSpec.model_validate(base_payload)


def test_querygen_spec_rejects_extra_keys(base_payload: dict) -> None:
    """Schema rejects unexpected extra keys."""
    base_payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        QueryGenSpec.model_validate(base_payload)
