"""Unit tests for the synthetic query generation input contract."""

import pytest
from pydantic import ValidationError

from chatboteval.core.schemas.querygen_input import QueryGenSpec, WeightedValue


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


@pytest.mark.parametrize(
    "patch",
    [
        lambda p: p["scenario"].update(
            intents=[
                {"value": "information", "weight": 1.1},
                {"value": "x", "weight": -0.1},
            ]
        ),
        lambda p: p["scenario"].update(
            tasks=[
                {"value": "eligibility", "weight": 0.2},
                {"value": "procedure", "weight": 0.2},
            ]
        ),
        lambda p: p["domain_context"].update(domains=[]),
        lambda p: p["domain_context"].update(
            languages=[
                "en",
                {"value": "de", "weight": 0.5},
            ]
        ),
        lambda p: p["domain_context"].update(unexpected="nope"),
    ],
)
def test_invalid_inputs_raise_validation_error(base_payload: dict, patch) -> None:
    """Invalid ChoiceStr shapes/weights/keys raise an error."""
    patch(base_payload)
    with pytest.raises((ValidationError, TypeError, ValueError)):
        QueryGenSpec.model_validate(base_payload)