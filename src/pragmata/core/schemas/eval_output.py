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


class ScoreInputSource(BaseModel):
    """Provenance of the labeled data a score report was computed from.

    ``kind`` records the source category: ``direct_path`` (a path supplied
    directly), ``annotation_export`` (resolved from a workspace annotation
    export), or ``model_prediction`` (labels produced by ``pragmata eval
    predict``). ``ref`` is the selector's value (the ``path``, ``export_id``, or
    ``prediction_id``); ``resolved_path`` is the concrete CSV that was read,
    recorded relative to the workspace (or absolute when the input lies outside
    it). ``kind`` also drives ingestion: only ``model_prediction`` inputs are
    tlmtc-shaped and need text-column restoration before validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["direct_path", "annotation_export", "model_prediction"]
    ref: str
    resolved_path: str


class EvalTrainMeta(BaseModel):
    """Pragmata-owned metadata for a completed evaluator training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task: Task
    annotation_export_id: str | None = None


class EvalPredictMeta(BaseModel):
    """Pragmata-owned metadata for a completed evaluator prediction run.

    ``run_id`` is the evaluator training run id: tlmtc keys prediction output by
    the same ``run_id`` it loads the evaluator from, so a run is identified by
    which evaluator produced it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task: Task
    unlabeled_data_path: str | None = None


class RetrievalScoreReport(BaseModel):
    """Schema for retrieval_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.RETRIEVAL] = Task.RETRIEVAL
    source: ScoreInputSource
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

    def metric_scores(self) -> list[tuple[str, MetricScore | None]]:
        """The report's metrics in display order, each paired with its field name."""
        return [
            ("topical_precision_at_k", self.topical_precision_at_k),
            ("sufficiency_hit_at_k", self.sufficiency_hit_at_k),
            ("sufficiency_rate_at_k", self.sufficiency_rate_at_k),
            ("misleading_context_rate_at_k", self.misleading_context_rate_at_k),
            ("mean_reciprocal_rank_at_k", self.mean_reciprocal_rank_at_k),
            ("ndcg_at_k", self.ndcg_at_k),
        ]


class GroundingScoreReport(BaseModel):
    """Schema for grounding_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GROUNDING] = Task.GROUNDING
    source: ScoreInputSource
    notes: str = ""
    created_at: datetime
    n_examples: PositiveInt
    ci_level: CiLevel
    grounding_presence_rate: MetricScore
    unsupported_claim_rate: MetricScore
    contradiction_rate: MetricScore
    citation_presence_rate: MetricScore
    conditional_fabrication_rate: MetricScore | None = None

    def metric_scores(self) -> list[tuple[str, MetricScore | None]]:
        """The report's metrics in display order, each paired with its field name."""
        return [
            ("grounding_presence_rate", self.grounding_presence_rate),
            ("unsupported_claim_rate", self.unsupported_claim_rate),
            ("contradiction_rate", self.contradiction_rate),
            ("citation_presence_rate", self.citation_presence_rate),
            ("conditional_fabrication_rate", self.conditional_fabrication_rate),
        ]


class GenerationScoreReport(BaseModel):
    """Schema for generation_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GENERATION] = Task.GENERATION
    source: ScoreInputSource
    notes: str = ""
    created_at: datetime
    n_examples: PositiveInt
    ci_level: CiLevel
    proper_action_rate: MetricScore
    on_topic_rate: MetricScore
    helpfulness_rate: MetricScore
    incompleteness_rate: MetricScore
    unsafe_content_rate: MetricScore

    def metric_scores(self) -> list[tuple[str, MetricScore | None]]:
        """The report's metrics in display order, each paired with its field name."""
        return [
            ("proper_action_rate", self.proper_action_rate),
            ("on_topic_rate", self.on_topic_rate),
            ("helpfulness_rate", self.helpfulness_rate),
            ("incompleteness_rate", self.incompleteness_rate),
            ("unsafe_content_rate", self.unsafe_content_rate),
        ]
