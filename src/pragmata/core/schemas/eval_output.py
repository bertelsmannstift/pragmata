"""Output schemas for eval artifacts."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from pragmata.core.schemas.annotation_task import Task

type Rate = Annotated[float, Field(ge=0.0, le=1.0)]
# Confidence level in the open interval (0, 1)
type CiLevel = Annotated[float, Field(gt=0.0, lt=1.0)]


class MetricScore(BaseModel):
    """A single reported metric: point estimate with its confidence interval.

    ``point``, ``ci_lower`` and ``ci_upper`` share the metric's [0, 1] scale.
    ``method`` records how the interval was derived (``wilson`` for proportion
    metrics, ``bootstrap`` for continuous ones), and ``n`` is the effective
    denominator the estimate is over: the number of queries, or the cited
    subset for a conditional metric.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: Rate
    ci_lower: Rate
    ci_upper: Rate
    method: Literal["wilson", "bootstrap"]
    n: PositiveInt

    @model_validator(mode="after")
    def _check_ci_ordering(self) -> "MetricScore":
        # A degenerate interval (lower == upper) is valid: a bootstrap CI on a
        # constant metric collapses to the point estimate. Only inversion is wrong.
        if self.ci_lower > self.ci_upper:
            raise ValueError(f"ci_lower ({self.ci_lower}) must not exceed ci_upper ({self.ci_upper})")
        return self


class EvalTrainMeta(BaseModel):
    """Pragmata-owned metadata for a completed evaluator training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task: Task
    annotation_export_id: str | None = None


class RetrievalScoreReport(BaseModel):
    """Schema for retrieval_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.RETRIEVAL] = Task.RETRIEVAL
    annotation_export_id: str | None = None
    notes: str = ""
    created_at: datetime
    n_examples: PositiveInt
    top_k: PositiveInt
    ci_level: CiLevel
    topical_precision_at_k: MetricScore
    sufficiency_hit_at_k: MetricScore
    sufficiency_rate_at_k: MetricScore
    misleading_context_rate_at_k: MetricScore
    mean_reciprocal_rank_at_k: MetricScore
    ndcg_at_k: MetricScore


class GroundingScoreReport(BaseModel):
    """Schema for grounding_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GROUNDING] = Task.GROUNDING
    annotation_export_id: str | None = None
    notes: str = ""
    created_at: datetime
    n_examples: PositiveInt
    ci_level: CiLevel
    grounding_presence_rate: MetricScore
    unsupported_claim_rate: MetricScore
    contradiction_rate: MetricScore
    citation_presence_rate: MetricScore
    conditional_fabrication_rate: MetricScore | None = None


class GenerationScoreReport(BaseModel):
    """Schema for generation_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GENERATION] = Task.GENERATION
    annotation_export_id: str | None = None
    notes: str = ""
    created_at: datetime
    n_examples: PositiveInt
    ci_level: CiLevel
    proper_action_rate: MetricScore
    on_topic_rate: MetricScore
    helpfulness_rate: MetricScore
    incompleteness_rate: MetricScore
    unsafe_content_rate: MetricScore
