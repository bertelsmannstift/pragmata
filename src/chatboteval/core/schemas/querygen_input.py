"""Input contract for synthetic query generation.

This module defines a stable Pydantic v2 schema ("contract") for user-controlled
knobs to generate synthetic evaluation queries.

All ChoiceStr inputs are canonicalized to `list[WeightedValue]` via
`validate_choice_str()`.
"""

from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

WEIGHT_SUM_TOL: float = 1e-6

class WeightedValue(BaseModel):
    """A weighted categorical choice."""

    model_config = ConfigDict(extra="forbid")

    value: str
    weight: float

    @field_validator("value")
    @classmethod
    def _value_non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("value must be a non-empty string")
        return v2

    @field_validator("weight")
    @classmethod
    def _weight_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight must be non-negative")
        return float(v)


ChoiceStr: TypeAlias = str | list[str] | list[WeightedValue]


def validate_choice_str(value: Any) -> list[WeightedValue]:
    """Parse, validate, and canonicalize ChoiceStr inputs.

    Canonical output is always list[WeightedValue].
    """
    if value is None:
        raise TypeError("ChoiceStr must not be None")

    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("ChoiceStr string must be non-empty")
        return [WeightedValue(value=s, weight=1.0)]

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("ChoiceStr list must not be empty")

        if all(isinstance(v, str) for v in value):
            values = [v.strip() for v in value]
            if any(not v for v in values):
                raise ValueError("ChoiceStr list[str] entries must be non-empty")
            w = 1.0 / float(len(values))
            return [WeightedValue(value=v, weight=w) for v in values]

        if any(isinstance(v, str) for v in value):
            raise TypeError(
                "ChoiceStr lists must be homogeneous: either list[str] or list[WeightedValue]"
            )

        items: list[WeightedValue] = [
            v if isinstance(v, WeightedValue) else WeightedValue.model_validate(v)
            for v in value
        ]

        total = sum(i.weight for i in items)
        if total <= 0:
            raise ValueError("ChoiceStr weighted list must have total weight > 0")
        if abs(total - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError("ChoiceStr weighted list must have weights summing to 1")

        return items

    raise TypeError("ChoiceStr must be a string, a list[str], or a list[WeightedValue]")


def validate_choice_str_optional(value: Any) -> list[WeightedValue] | None:
    """Optional wrapper: allow None for optional fields."""
    if value is None:
        return None
    return validate_choice_str(value)


ChoiceStrField: TypeAlias = Annotated[list[WeightedValue], BeforeValidator(validate_choice_str)]
ChoiceStrFieldOptional: TypeAlias = Annotated[
    list[WeightedValue] | None, BeforeValidator(validate_choice_str_optional)
]


class DomainContextSpec(BaseModel):
    """Domain and audience knobs for synthetic query generation.

    These control *who* is asking and in *what domain*.
    """

    model_config = ConfigDict(extra="forbid")

    domains: ChoiceStrField
    roles: ChoiceStrField
    languages: ChoiceStrField


class KnowledgeScopeSpec(BaseModel):
    """Knowledge scope knobs.

    These control *what subject matter* the synthetic query should be about.
    """

    model_config = ConfigDict(extra="forbid")

    topics: ChoiceStrField


class ScenarioSpec(BaseModel):
    """Scenario knobs for query generation.

    These control the *intent* and *task* framing for the question.
    """

    model_config = ConfigDict(extra="forbid")

    intents: ChoiceStrField
    tasks: ChoiceStrField
    difficulty: ChoiceStrFieldOptional = None


class FormatRequestsSpec(BaseModel):
    """Output formatting knobs.

    These control how the chatbot is asked to present the answer.
    """

    model_config = ConfigDict(extra="forbid")

    formats: ChoiceStrFieldOptional = None


class SafetySpec(BaseModel):
    """Safety-related constraints for query generation."""

    model_config = ConfigDict(extra="forbid")

    disallowed_topics: list[str] | None = None

    @field_validator("disallowed_topics")
    @classmethod
    def _disallowed_topics_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("disallowed_topics entries must be non-empty strings")
        return cleaned


class QueryGenSpec(BaseModel):
    """Top-level input contract for synthetic query generation.

    This bundles semantically grouped knobs into one stable schema so downstream
    query-plan generation can rely on one canonical structure.
    """

    model_config = ConfigDict(extra="forbid")

    domain_context: DomainContextSpec
    knowledge_scope: KnowledgeScopeSpec
    scenario: ScenarioSpec
    format_requests: FormatRequestsSpec = Field(default_factory=FormatRequestsSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)