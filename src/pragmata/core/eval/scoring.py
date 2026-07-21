"""Assemble task-specific eval score reports from labeled per-query values.

This is the scoring layer referenced by ``core/eval/metrics.py``: point estimates
come from ``metrics.corpus_mean`` and the per-query formulas, while each metric's
confidence interval comes from the shared ``core.annotation.uncertainty`` helpers
(Wilson for proportion metrics, percentile bootstrap for the continuous retrieval
metrics). It is the report-assembly step the API orchestrates - pure and I/O-free:
the caller resolves and reads the frame and writes the report.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pragmata.core.annotation.uncertainty import percentile_bootstrap, wilson_interval
from pragmata.core.eval import grouping, metrics
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import (
    GenerationScoreReport,
    GroundingScoreReport,
    MetricScore,
    RetrievalScoreReport,
    ScoreInputSource,
)

ScoreReport = RetrievalScoreReport | GroundingScoreReport | GenerationScoreReport


def _wilson_metric(values: list[float] | NDArray, *, alpha: float) -> MetricScore:
    """MetricScore for a proportion metric: point = proportion, CI = Wilson."""
    array = np.asarray(values)
    n = len(array)
    lower, upper = wilson_interval(int(array.sum()), n, alpha=alpha)
    return MetricScore(point=metrics.corpus_mean(values), ci_lower=lower, ci_upper=upper, method="wilson", n=n)


def _bootstrap_metric(values: list[float], *, alpha: float, n_resamples: int, seed: int | None) -> MetricScore:
    """MetricScore for a continuous metric: point = mean, CI = percentile bootstrap."""
    array = np.asarray(values, dtype=float)
    lower, upper = percentile_bootstrap(
        len(array),
        lambda idx: float(array[idx].mean()),
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    return MetricScore(
        point=metrics.corpus_mean(values), ci_lower=lower, ci_upper=upper, method="bootstrap", n=len(array)
    )


def _conditional_metric(units: NDArray, *, alpha: float) -> MetricScore | None:
    """MetricScore for the conditional fabrication rate, or None when no cited queries."""
    if len(units) == 0:
        return None
    return _wilson_metric(units, alpha=alpha)


def build_score_report(
    frame: pd.DataFrame,
    *,
    task: Task,
    ci: float,
    n_resamples: int,
    seed: int | None,
    source: ScoreInputSource,
    created_at: datetime,
) -> ScoreReport:
    """Assemble the task-specific report from per-query values (pure, I/O-free)."""
    alpha = 1.0 - ci

    def bootstrap(values: list[float]) -> MetricScore:
        # The same seed is reused for every metric's bootstrap. Each marginal CI stays
        # valid (given large enough n_resamples) and we never use the joint distribution
        # across metrics, so the shared seed is harmless. Flagged here in case that changes.
        return _bootstrap_metric(values, alpha=alpha, n_resamples=n_resamples, seed=seed)

    def wilson(values: list[float]) -> MetricScore:
        return _wilson_metric(values, alpha=alpha)

    if task == Task.RETRIEVAL:
        values = grouping.retrieval_per_query_values(frame)
        return RetrievalScoreReport(
            source=source,
            created_at=created_at,
            n_examples=len(values["topical_precision_at_k"]),
            # chunk_rank is 1-based (enforced by Field(ge=1) on import), so its max is the
            # number of retrieved chunks.
            top_k=int(frame["chunk_rank"].max()),
            ci_level=ci,
            topical_precision_at_k=bootstrap(values["topical_precision_at_k"]),
            sufficiency_hit_at_k=wilson(values["sufficiency_hit_at_k"]),
            sufficiency_rate_at_k=bootstrap(values["sufficiency_rate_at_k"]),
            misleading_context_rate_at_k=bootstrap(values["misleading_context_rate_at_k"]),
            mean_reciprocal_rank_at_k=bootstrap(values["mean_reciprocal_rank_at_k"]),
            ndcg_at_k=bootstrap(values["ndcg_at_k"]),
        )

    if task == Task.GROUNDING:
        values = grouping.grounding_per_query_values(frame)
        return GroundingScoreReport(
            source=source,
            created_at=created_at,
            n_examples=len(frame),
            ci_level=ci,
            grounding_presence_rate=wilson(values["grounding_presence_rate"]),
            unsupported_claim_rate=wilson(values["unsupported_claim_rate"]),
            contradiction_rate=wilson(values["contradiction_rate"]),
            citation_presence_rate=wilson(values["citation_presence_rate"]),
            conditional_fabrication_rate=_conditional_metric(
                grouping.conditional_fabrication_units(frame), alpha=alpha
            ),
        )

    if task == Task.GENERATION:
        values = grouping.generation_per_query_values(frame)
        return GenerationScoreReport(
            source=source,
            created_at=created_at,
            n_examples=len(frame),
            ci_level=ci,
            proper_action_rate=wilson(values["proper_action_rate"]),
            on_topic_rate=wilson(values["on_topic_rate"]),
            helpfulness_rate=wilson(values["helpfulness_rate"]),
            incompleteness_rate=wilson(values["incompleteness_rate"]),
            unsafe_content_rate=wilson(values["unsafe_content_rate"]),
        )

    raise ValueError(f"Unsupported task type: {task!r}")
