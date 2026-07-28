"""Unit tests for the pure per-query metric formulas."""

import math

import numpy as np
import pytest

from pragmata.core.eval import metrics


def arr(*values: int) -> np.ndarray:
    return np.array(values, dtype=int)


class TestTopicalPrecision:
    def test_fraction_relevant(self):
        assert metrics.topical_precision(arr(1, 1, 0)) == pytest.approx(2 / 3)

    def test_all_relevant(self):
        assert metrics.topical_precision(arr(1, 1)) == 1.0

    def test_none_relevant(self):
        assert metrics.topical_precision(arr(0, 0)) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            metrics.topical_precision(arr())


class TestSufficiency:
    def test_hit_true_when_any_sufficient(self):
        assert metrics.sufficiency_hit(arr(0, 1, 0)) == 1.0

    def test_hit_false_when_none_sufficient(self):
        assert metrics.sufficiency_hit(arr(0, 0, 0)) == 0.0

    def test_rate_is_fraction(self):
        assert metrics.sufficiency_rate(arr(1, 0, 0)) == pytest.approx(1 / 3)


class TestMisleadingContextRate:
    def test_fraction_misleading(self):
        assert metrics.misleading_context_rate(arr(1, 1, 0, 0)) == 0.5


class TestReciprocalRank:
    def test_first_relevant_at_rank_two(self):
        # arrays are pre-sorted by chunk_rank, so index 1 is rank 2.
        assert metrics.reciprocal_rank(arr(0, 1, 0)) == pytest.approx(0.5)

    def test_first_relevant_at_rank_one(self):
        assert metrics.reciprocal_rank(arr(1, 0)) == 1.0

    def test_no_relevant_is_zero(self):
        assert metrics.reciprocal_rank(arr(0, 0, 0)) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            metrics.reciprocal_rank(arr())


class TestNdcg:
    def test_ideal_ordering_is_one(self):
        # grades [2, 1, 0] already descending -> DCG == IDCG.
        topically_relevant = arr(1, 1, 0)
        evidence_sufficient = arr(1, 0, 0)
        assert metrics.ndcg(topically_relevant, evidence_sufficient) == pytest.approx(1.0)

    def test_worked_example(self):
        # rank1: t=1,s=0 -> grade 1 (gain 1); rank2: t=1,s=1 -> grade 2 (gain 3); rank3: 0.
        # DCG  = 1/log2(2) + 3/log2(3)          = 2.892789
        # IDCG = 3/log2(2) + 1/log2(3)          = 3.630930  (grades sorted [2,1,0])
        # NDCG = 0.796708
        topically_relevant = arr(1, 1, 0)
        evidence_sufficient = arr(0, 1, 0)
        result = metrics.ndcg(topically_relevant, evidence_sufficient)
        assert result == pytest.approx(0.796708, abs=1e-5)
        assert 0.0 <= result <= 1.0

    def test_no_relevant_is_zero(self):
        assert metrics.ndcg(arr(0, 0), arr(0, 0)) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            metrics.ndcg(arr(), arr())

    def test_never_exceeds_one(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(1, 6))
            t = rng.integers(0, 2, size=n)
            s = rng.integers(0, 2, size=n)
            assert metrics.ndcg(t, s) <= 1.0 + 1e-9


class TestFabricatedAmongCited:
    def test_filters_to_cited_subset(self):
        source_cited = arr(1, 0, 1, 1)
        fabricated_source = arr(1, 1, 0, 1)  # index 1 is fabricated but NOT cited -> excluded
        result = metrics.fabricated_among_cited(source_cited, fabricated_source)
        assert np.array_equal(result, np.array([1, 0, 1]))

    def test_no_cited_returns_empty(self):
        result = metrics.fabricated_among_cited(arr(0, 0), arr(1, 1))
        assert result.shape[0] == 0


class TestCorpusMean:
    def test_mean_of_continuous_values(self):
        assert metrics.corpus_mean([0.5, 1.0, 0.0]) == pytest.approx(0.5)

    def test_mean_of_binary_is_proportion(self):
        assert metrics.corpus_mean([1, 0, 1, 1]) == pytest.approx(0.75)

    def test_accepts_ndarray(self):
        assert metrics.corpus_mean(np.array([0.2, 0.4])) == pytest.approx(0.3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            metrics.corpus_mean([])


def test_all_retrieval_values_are_finite_rates():
    """Sanity: every per-query retrieval value lands in [0, 1]."""
    t, s, m = arr(1, 0, 1), arr(1, 0, 0), arr(0, 1, 0)
    for value in (
        metrics.topical_precision(t),
        metrics.sufficiency_hit(s),
        metrics.sufficiency_rate(s),
        metrics.misleading_context_rate(m),
        metrics.reciprocal_rank(t),
        metrics.ndcg(t, s),
    ):
        assert not math.isnan(value)
        assert 0.0 <= value <= 1.0
