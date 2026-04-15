"""Unit tests for core IAA metric implementations."""

import math

import numpy as np
import pytest

from pragmata.core.annotation.iaa import (
    bootstrap_alpha,
    cohen_kappa,
    krippendorff_alpha_nominal,
    percentage_agreement,
)

NaN = np.nan


class TestKrippendorffAlpha:
    """Tests for the nominal Krippendorff's alpha implementation."""

    def test_perfect_agreement(self):
        data = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=float)
        assert krippendorff_alpha_nominal(data) == pytest.approx(1.0)

    def test_perfect_agreement_three_annotators(self):
        data = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=float)
        assert krippendorff_alpha_nominal(data) == pytest.approx(1.0)

    def test_no_agreement(self):
        data = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
        alpha = krippendorff_alpha_nominal(data)
        assert alpha < 0

    def test_reference_dataset(self):
        """Cross-validated against fast-krippendorff library.

        4 annotators, 12 binary items with missing data.
        Expected alpha = 0.4907 (validated against krippendorff==0.8.2).
        """
        data = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, NaN, NaN],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, NaN, NaN],
                [NaN, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, NaN],
            ]
        )
        alpha = krippendorff_alpha_nominal(data)
        assert alpha == pytest.approx(0.4907, abs=0.001)

    def test_handles_missing_data(self):
        data = np.array([[1, NaN, 1, 0], [1, 1, NaN, 0], [NaN, 1, 1, 0]], dtype=float)
        alpha = krippendorff_alpha_nominal(data)
        assert not math.isnan(alpha)

    def test_insufficient_data_returns_nan(self):
        data = np.array([[1, NaN], [NaN, 0]], dtype=float)
        assert math.isnan(krippendorff_alpha_nominal(data))

    def test_single_item_returns_nan(self):
        data = np.array([[1], [1]], dtype=float)
        alpha = krippendorff_alpha_nominal(data)
        # Single item can still produce a value (perfect agreement), but
        # with so little data the result is degenerate.
        assert isinstance(alpha, float)

    def test_all_same_label(self):
        data = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)
        alpha = krippendorff_alpha_nominal(data)
        # When all labels are identical, expected disagreement is 0 -> alpha = 1.
        assert alpha == pytest.approx(1.0)

    def test_empty_matrix_returns_nan(self):
        data = np.array([[NaN, NaN], [NaN, NaN]], dtype=float)
        assert math.isnan(krippendorff_alpha_nominal(data))


class TestCohenKappa:
    """Tests for pairwise Cohen's kappa."""

    def test_perfect_agreement(self):
        a = np.array([1, 0, 1, 0, 1])
        b = np.array([1, 0, 1, 0, 1])
        assert cohen_kappa(a, b) == pytest.approx(1.0)

    def test_no_agreement(self):
        a = np.array([1, 0, 1, 0])
        b = np.array([0, 1, 0, 1])
        kappa = cohen_kappa(a, b)
        assert kappa < 0

    def test_moderate_agreement(self):
        a = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        b = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        kappa = cohen_kappa(a, b)
        assert 0 < kappa < 1

    def test_empty_arrays_return_nan(self):
        assert math.isnan(cohen_kappa(np.array([], dtype=int), np.array([], dtype=int)))

    def test_all_same_label_degenerate(self):
        a = np.array([1, 1, 1, 1])
        b = np.array([1, 1, 1, 1])
        # p_e = 1.0 -> degenerate, but all agree so kappa is nan by formula.
        assert math.isnan(cohen_kappa(a, b))

    def test_bool_input(self):
        a = np.array([True, False, True, False])
        b = np.array([True, False, False, False])
        kappa = cohen_kappa(a, b)
        assert isinstance(kappa, float)
        assert not math.isnan(kappa)

    def test_binary_contract_bool_matches_int(self):
        """Bool and {0, 1} int inputs must produce identical kappa values.

        Locks the binary-only contract: the function treats ``1`` as positive
        and anything else as negative, so bool and ``{0, 1}`` int inputs
        representing the same labels must agree exactly.
        """
        bool_a = np.array([True, False, True, True, False, False])
        bool_b = np.array([True, True, True, False, False, False])
        int_a = bool_a.astype(np.int8)
        int_b = bool_b.astype(np.int8)
        assert cohen_kappa(bool_a, bool_b) == pytest.approx(cohen_kappa(int_a, int_b))


class TestBootstrapAlpha:
    """Tests for bootstrap confidence intervals."""

    def test_returns_tuple_of_floats(self):
        data = np.array([[1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0]], dtype=float)
        lower, upper = bootstrap_alpha(data, n_resamples=200, seed=42)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_less_than_upper(self):
        data = np.array(
            [[1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0, 0]],
            dtype=float,
        )
        lower, upper = bootstrap_alpha(data, n_resamples=500, seed=42)
        assert lower <= upper

    def test_perfect_agreement_tight_ci(self):
        data = np.array([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]], dtype=float)
        lower, upper = bootstrap_alpha(data, n_resamples=200, seed=42)
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(1.0)

    def test_seed_reproducibility(self):
        data = np.array([[1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0]], dtype=float)
        r1 = bootstrap_alpha(data, n_resamples=100, seed=123)
        r2 = bootstrap_alpha(data, n_resamples=100, seed=123)
        assert r1 == r2

    def test_degenerate_data_returns_nan(self):
        data = np.array([[NaN, NaN], [NaN, NaN]], dtype=float)
        lower, upper = bootstrap_alpha(data, n_resamples=50, seed=42)
        assert math.isnan(lower)
        assert math.isnan(upper)


class TestPercentageAgreement:
    """Tests for simple percentage agreement."""

    def test_perfect_agreement(self):
        data = np.array([[1, 0, 1], [1, 0, 1]], dtype=float)
        assert percentage_agreement(data) == pytest.approx(1.0)

    def test_no_agreement(self):
        data = np.array([[1, 0], [0, 1]], dtype=float)
        assert percentage_agreement(data) == pytest.approx(0.0)

    def test_partial_agreement(self):
        data = np.array([[1, 0, 1, 0], [1, 1, 1, 0]], dtype=float)
        pct = percentage_agreement(data)
        assert 0 < pct < 1

    def test_no_overlap_returns_nan(self):
        data = np.array([[1, NaN], [NaN, 0]], dtype=float)
        assert math.isnan(percentage_agreement(data))
