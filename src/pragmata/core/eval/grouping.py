"""Group validated eval score frames into the per-query values metrics consume.

The scoring resampling unit is the query (``record_uuid``). Retrieval rows are
grouped into queries and each query's chunks ordered by ``chunk_rank`` before the
per-query retrieval formulas run; grounding and generation carry one row per
query, so their per-query values are the row-level labels directly.

Each function returns ``{report_field_name: [per-query value, ...]}`` in a stable
query order (first appearance), which the scoring layer turns into point
estimates and confidence intervals. Input frames are assumed already validated
(``validate_eval_score_frame``) and complete; dataset-level policy for incomplete
or degenerate data is the scoring layer's responsibility, not this module's.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pragmata.core.eval import metrics

# Report field name -> source binary label column. Grounding and generation have
# one row per query, so the per-query value is the label itself (the conditional
# fabrication rate is handled separately, as it is not a plain column mean). This
# column wiring lives in the preparation layer; metrics.py stays column-agnostic.
GROUNDING_METRIC_LABELS: dict[str, str] = {
    "grounding_presence_rate": "support_present",
    "unsupported_claim_rate": "unsupported_claim_present",
    "contradiction_rate": "contradicted_claim_present",
    "citation_presence_rate": "source_cited",
}

GENERATION_METRIC_LABELS: dict[str, str] = {
    "proper_action_rate": "proper_action",
    "on_topic_rate": "response_on_topic",
    "helpfulness_rate": "helpful",
    "incompleteness_rate": "incomplete",
    "unsafe_content_rate": "unsafe_content",
}


def retrieval_per_query_values(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Per-query retrieval metric values, one entry per query in the frame."""
    values: dict[str, list[float]] = {
        "topical_precision_at_k": [],
        "sufficiency_hit_at_k": [],
        "sufficiency_rate_at_k": [],
        "misleading_context_rate_at_k": [],
        "mean_reciprocal_rank_at_k": [],
        "ndcg_at_k": [],
    }
    for _, query in frame.groupby("record_uuid", sort=False):
        ordered = query.sort_values("chunk_rank")
        topically_relevant = ordered["topically_relevant"].to_numpy()
        evidence_sufficient = ordered["evidence_sufficient"].to_numpy()
        misleading = ordered["misleading"].to_numpy()
        values["topical_precision_at_k"].append(metrics.topical_precision(topically_relevant))
        values["sufficiency_hit_at_k"].append(metrics.sufficiency_hit(evidence_sufficient))
        values["sufficiency_rate_at_k"].append(metrics.sufficiency_rate(evidence_sufficient))
        values["misleading_context_rate_at_k"].append(metrics.misleading_context_rate(misleading))
        values["mean_reciprocal_rank_at_k"].append(metrics.reciprocal_rank(topically_relevant))
        values["ndcg_at_k"].append(metrics.ndcg(topically_relevant, evidence_sufficient))
    return values


def _row_level_values(frame: pd.DataFrame, label_map: dict[str, str]) -> dict[str, list[float]]:
    """Extract each report field's binary label column as a per-query value list."""
    return {field: frame[column].astype(float).tolist() for field, column in label_map.items()}


def grounding_per_query_values(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Per-query values for the four unconditional grounding metrics.

    The conditional fabrication rate is not a plain column mean; use
    :func:`conditional_fabrication_units`.
    """
    return _row_level_values(frame, GROUNDING_METRIC_LABELS)


def generation_per_query_values(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Per-query values for the five generation metrics."""
    return _row_level_values(frame, GENERATION_METRIC_LABELS)


def conditional_fabrication_units(frame: pd.DataFrame) -> NDArray[np.int_]:
    """Fabrication flags over the cited subset, for the conditional fabrication rate."""
    return metrics.fabricated_among_cited(
        frame["source_cited"].to_numpy(),
        frame["fabricated_source"].to_numpy(),
    )
