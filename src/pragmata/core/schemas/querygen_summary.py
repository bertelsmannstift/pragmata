"""Structured output contract for LLM stage 1 query planning memory."""

from pydantic import BaseModel, ConfigDict, Field


class PlanningSummaryState(BaseModel):
    """Compact advisory planning-memory state used across planning batches and runs."""

    model_config = ConfigDict(extra="forbid")

    redundancy_patterns: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Concise description of recurring candidate blueprint patterns, including repeated scenarios, information "
            "needs, or semantic framings, that are already overrepresented and should be avoided in the next planning "
            "batch."
        ),
    )
    diversification_targets: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Concrete guidance on spec-compatible candidate blueprint patterns, including scenarios, information "
            "needs, or semantic framings that would improve diversity in the next planning batch."
        ),
    )
    coverage_notes: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Brief notes on candidate blueprint patterns, including scenarios and information needs, or semantic "
            "framings, that have already appeared and should not be revisited too closely in the next planning batch."
        ),
    )
