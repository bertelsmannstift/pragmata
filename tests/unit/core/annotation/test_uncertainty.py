"""Unit tests for the shared uncertainty helpers.

Covers the Wilson score interval and the generic percentile bootstrap, plus a
golden-value regression lock proving the ``bootstrap_alpha`` refactor onto
``percentile_bootstrap`` is behaviourally identical to the pre-refactor code.
"""

import math
from itertools import count

import numpy as np
import pytest

from pragmata.core.annotation.iaa import bootstrap_alpha, krippendorff_alpha_nominal
from pragmata.core.annotation.uncertainty import percentile_bootstrap, wilson_interval

NaN = np.nan

# 4 annotators x 12 binary items with missing data (alpha ~= 0.4907); shared
# with test_iaa.py.
REFERENCE_DATA = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, NaN, NaN],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, NaN, NaN],
        [NaN, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, NaN],
    ]
)
SIMPLE_DATA = np.array([[1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0]], dtype=float)


class TestWilsonInterval:
    """Tests for the Wilson score interval."""

    def test_reference_value(self):
        """8 successes out of 10 at 95%.

        Validated against R ``Hmisc::binconf(8, 10, method="wilson")`` and the
        textbook Wilson formula: ~(0.4901, 0.9433).
        """
        lower, upper = wilson_interval(8, 10)
        assert lower == pytest.approx(0.4901, abs=0.001)
        assert upper == pytest.approx(0.9433, abs=0.001)

    def test_point_within_interval(self):
        lower, upper = wilson_interval(5, 10)
        assert lower <= 0.5 <= upper

    def test_all_successes_upper_clamped(self):
        """All-1 stays informative: positive-width, upper clamps to 1.0."""
        lower, upper = wilson_interval(10, 10)
        assert 0.0 < lower < 1.0
        assert upper == pytest.approx(1.0)
        assert lower == pytest.approx(0.7224, abs=0.001)

    def test_zero_successes_lower_clamped(self):
        """All-0 mirrors all-1: lower clamps to 0.0, upper positive."""
        lower, upper = wilson_interval(0, 10)
        assert lower == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < upper < 1.0

    def test_ordering_and_bounds(self):
        for successes in range(0, 11):
            lower, upper = wilson_interval(successes, 10)
            assert 0.0 <= lower <= upper <= 1.0

    def test_single_trial(self):
        lower, upper = wilson_interval(1, 1)
        assert 0.0 <= lower <= upper <= 1.0

    def test_zero_n_returns_nan(self):
        lower, upper = wilson_interval(0, 0)
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_ci_level_widens_interval(self):
        narrow = wilson_interval(8, 20, ci=0.90)
        wide = wilson_interval(8, 20, ci=0.99)
        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]


class TestPercentileBootstrap:
    """Tests for the generic percentile bootstrap."""

    def test_constant_statistic(self):
        lower, upper = percentile_bootstrap(5, lambda _idx: 0.7, n_resamples=50, seed=1)
        assert lower == pytest.approx(0.7)
        assert upper == pytest.approx(0.7)

    def test_all_nan_returns_nan(self):
        lower, upper = percentile_bootstrap(5, lambda _idx: float("nan"), n_resamples=50, seed=1)
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_nan_replicates_dropped(self):
        """Degenerate replicates are dropped, not propagated."""
        counter = count(1)

        def statistic(_idx):
            return float("nan") if next(counter) <= 3 else 1.0

        lower, upper = percentile_bootstrap(5, statistic, n_resamples=10, seed=1)
        assert (lower, upper) == (1.0, 1.0)

    def test_indices_within_range(self):
        maxes: list[int] = []

        def statistic(idx):
            maxes.append(int(idx.max()))
            assert idx.min() >= 0
            return float(idx.mean())

        percentile_bootstrap(7, statistic, n_resamples=20, seed=2)
        assert max(maxes) < 7

    def test_seed_reproducibility(self):
        data = np.array([0.1, 0.5, 0.9, 0.2, 0.7])

        def statistic(idx):
            return float(data[idx].mean())

        r1 = percentile_bootstrap(5, statistic, n_resamples=100, seed=99)
        r2 = percentile_bootstrap(5, statistic, n_resamples=100, seed=99)
        assert r1 == r2

    def test_ordering(self):
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        lower, upper = percentile_bootstrap(5, lambda idx: float(data[idx].mean()), n_resamples=200, seed=3)
        assert lower <= upper
        assert not math.isnan(lower)


class TestBootstrapAlphaRegression:
    """Locks the ``bootstrap_alpha`` -> ``percentile_bootstrap`` refactor.

    Golden values were captured from the pre-refactor implementation. PCG64 and
    ``np.percentile`` are stable across versions, so exact equality is a tight,
    safe lock that fails if the refactor drifts.
    """

    def test_golden_reference_dataset(self):
        assert bootstrap_alpha(REFERENCE_DATA, n_resamples=100, seed=42) == (
            -7.827072323607348e-16,
            0.8571428571428571,
        )

    def test_golden_simple_dataset(self):
        assert bootstrap_alpha(SIMPLE_DATA, n_resamples=200, seed=42) == (
            -0.09999999999999987,
            1.0,
        )

    def test_delegates_faithfully(self):
        """Delegation matches a direct percentile_bootstrap call, same seed."""
        direct = percentile_bootstrap(
            REFERENCE_DATA.shape[1],
            lambda idx: krippendorff_alpha_nominal(REFERENCE_DATA[:, idx]),
            n_resamples=100,
            seed=42,
        )
        assert bootstrap_alpha(REFERENCE_DATA, n_resamples=100, seed=42) == direct
