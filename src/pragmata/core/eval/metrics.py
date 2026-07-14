"""Pure, deterministic metric formulas from the taxonomy.

Implements the per-query formulas in ``docs/methodology/metrics-taxonomy.md``.
Every function is I/O-free and operates on NumPy label arrays: retrieval
functions take one query's chunk labels (pre-sorted by ``chunk_rank`` ascending,
so array position ``i`` is rank ``i + 1``); grounding/generation values are the
row-level binary labels themselves.

Corpus aggregation (the mean over queries) and confidence intervals live in the
scoring layer - this module only computes the deterministic per-query numbers.
Dataset-level policy for incomplete or degenerate data is likewise out of scope
here; these functions assume each query they receive is complete.
"""

import numpy as np
from numpy.typing import NDArray


def _require_nonempty(values: NDArray) -> None:
    if values.shape[0] == 0:
        raise ValueError("retrieval metrics require at least one chunk per query")


def _mean(values: NDArray) -> float:
    _require_nonempty(values)
    return float(np.mean(values))


def corpus_mean(per_query_values: NDArray | list[float]) -> float:
    """Corpus-level point estimate: the taxonomy's ``1/I`` mean over per-query values.

    Applies to every metric's point estimate - a 0/1 array for the proportion
    metrics (where the mean is the proportion), continuous otherwise. The
    confidence interval is computed separately by the scoring layer.
    """
    return _mean(np.asarray(per_query_values, dtype=float))


# ---- retrieval per-query metrics (chunks pre-sorted by chunk_rank ascending) ----


def topical_precision(topically_relevant: NDArray) -> float:
    """Fraction of the query's chunks that are topically relevant."""
    return _mean(topically_relevant)


def sufficiency_hit(evidence_sufficient: NDArray) -> float:
    """1.0 if any chunk is sufficient evidence, else 0.0 (binary per query)."""
    _require_nonempty(evidence_sufficient)
    return float(np.any(evidence_sufficient == 1))


def sufficiency_rate(evidence_sufficient: NDArray) -> float:
    """Fraction of the query's chunks that are individually sufficient."""
    return _mean(evidence_sufficient)


def misleading_context_rate(misleading: NDArray) -> float:
    """Fraction of the query's chunks that are misleading."""
    return _mean(misleading)


def reciprocal_rank(topically_relevant: NDArray) -> float:
    """Reciprocal of the rank of the first topically-relevant chunk, else 0.0."""
    _require_nonempty(topically_relevant)
    hits = np.flatnonzero(topically_relevant == 1)
    if hits.size == 0:
        return 0.0
    return 1.0 / (int(hits[0]) + 1)


def _relevance_grades(topically_relevant: NDArray, evidence_sufficient: NDArray) -> NDArray:
    """Graded relevance: sufficient -> 2, relevant-only -> 1, else 0."""
    return np.where(evidence_sufficient == 1, 2, np.where(topically_relevant == 1, 1, 0))


def _dcg(grades: NDArray) -> float:
    """Discounted cumulative gain with gain ``2^grade - 1`` at 1-based positions."""
    positions = np.arange(1, grades.shape[0] + 1)
    gains = np.power(2.0, grades) - 1.0
    return float(np.sum(gains / np.log2(positions + 1)))


def ndcg(topically_relevant: NDArray, evidence_sufficient: NDArray) -> float:
    """Normalised DCG@K over the query's ranked chunks; 0.0 when no chunk is graded."""
    _require_nonempty(topically_relevant)
    grades = _relevance_grades(topically_relevant, evidence_sufficient)
    ideal = _dcg(np.sort(grades)[::-1])
    if ideal == 0.0:
        return 0.0
    return _dcg(grades) / ideal


# ---- conditional grounding metric ----


def fabricated_among_cited(source_cited: NDArray, fabricated_source: NDArray) -> NDArray:
    """Binary ``fabricated_source`` values over the cited subset only.

    The conditional fabrication rate is defined only where ``source_cited == 1``.
    Returns the per-cited-query fabrication flags; the scoring layer forms the
    rate and its Wilson interval from these (and reports ``None`` when empty).
    """
    cited = source_cited == 1
    return fabricated_source[cited].astype(int)
