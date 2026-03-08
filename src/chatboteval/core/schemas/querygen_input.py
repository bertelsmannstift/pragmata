"""Input contract for synthetic query generation."""

from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator


class WeightedValue(BaseModel):
    """A weighted categorical choice."""

    model_config = ConfigDict(extra="forbid")

    value: str
    weight: float

    @field_validator("value")
    @classmethod
    def value_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("value must be a non-empty string")
        return v

    @field_validator("weight")
    @classmethod
    def weight_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight must be non-negative")
        return float(v)


def validate_choice_str(value: Any) -> list[WeightedValue]:
    """Canonicalize ChoiceStr to list[WeightedValue]."""
    if value is None:
        raise ValueError("ChoiceStr must not be None")

    if isinstance(value, str):
        return [WeightedValue(value=value, weight=1.0)]

    if not isinstance(value, list):
        raise ValueError("ChoiceStr must be a string or a list")

    if not value:
        raise ValueError("ChoiceStr list must not be empty")

    if all(isinstance(v, str) for v in value):
        w = 1.0 / float(len(value))
        return [WeightedValue(value=v, weight=w) for v in value]

    if all(isinstance(v, (WeightedValue, dict)) for v in value):
        items: list[WeightedValue] = [
            v if isinstance(v, WeightedValue) else WeightedValue.model_validate(v) for v in value
        ]
        total = sum(i.weight for i in items)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("ChoiceStr weighted list must have weights summing to 1")
        return items

    raise ValueError("ChoiceStr list must contain either only strings or only weighted values")


def validate_choice_str_optional(value: Any) -> list[WeightedValue] | None:
    """Optional wrapper for ChoiceStr fields that allow None."""
    if value is None:
        return None
    return validate_choice_str(value)


ChoiceStrField: TypeAlias = Annotated[
    list[WeightedValue],
    BeforeValidator(validate_choice_str),
]
ChoiceStrFieldOptional: TypeAlias = Annotated[
    list[WeightedValue] | None,
    BeforeValidator(validate_choice_str_optional),
]


class DomainContextSpec(BaseModel):
    """Domain and audience knobs."""

    model_config = ConfigDict(extra="forbid")

    domains: ChoiceStrField
    roles: ChoiceStrField
    languages: ChoiceStrField


class KnowledgeScopeSpec(BaseModel):
    """Knowledge scope knobs."""

    model_config = ConfigDict(extra="forbid")

    topics: ChoiceStrField


class ScenarioSpec(BaseModel):
    """Scenario knobs."""

    model_config = ConfigDict(extra="forbid")

    intents: ChoiceStrField
    tasks: ChoiceStrField
    difficulty: ChoiceStrFieldOptional = None


class FormatRequestsSpec(BaseModel):
    """Output formatting knobs."""

    model_config = ConfigDict(extra="forbid")

    formats: ChoiceStrFieldOptional = None


class SafetySpec(BaseModel):
    """Safety constraints."""

    model_config = ConfigDict(extra="forbid")

    disallowed_topics: list[str] | None = None

    @field_validator("disallowed_topics")
    @classmethod
    def disallowed_topics_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("disallowed_topics entries must be non-empty strings")
        return cleaned


class QueryGenSpec(BaseModel):
    """Synthetic query generation input schema."""

    model_config = ConfigDict(extra="forbid")

    domain_context: DomainContextSpec
    knowledge_scope: KnowledgeScopeSpec
    scenario: ScenarioSpec
    format_requests: FormatRequestsSpec = Field(default_factory=FormatRequestsSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)
