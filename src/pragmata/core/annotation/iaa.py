"""Pure-NumPy implementations of inter-annotator agreement metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def krippendorff_alpha_nominal(data: NDArray[np.floating]) -> float:
    """Compute Krippendorff's alpha for nominal (binary/categorical) data.

    Args:
        data: 2-D array of shape (annotators, items). Use ``np.nan`` for
            missing annotations. Values are treated as nominal categories.

    Returns:
        Alpha in the range [-1, 1]. Returns ``nan`` when fewer than two
        pairable values exist across all items.
    """
    n_units = data.shape[1]

    # Collect observed coincidences across all items.
    observed: dict[tuple[float, float], float] = {}
    expected_counts: dict[float, float] = {}
    total_pairs = 0.0

    for col in range(n_units):
        values = data[:, col]
        valid = values[~np.isnan(values)]
        m = len(valid)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i in range(m):
            v = valid[i]
            expected_counts[v] = expected_counts.get(v, 0.0) + 1.0
            for j in range(m):
                if i == j:
                    continue
                pair = (v, valid[j])
                observed[pair] = observed.get(pair, 0.0) + weight
        total_pairs += m

    if total_pairs < 2:
        return float("nan")

    # Observed disagreement.
    d_o = 0.0
    n_o = 0.0
    for (v, w), count in observed.items():
        if v != w:
            d_o += count
        n_o += count

    if n_o == 0:
        return float("nan")

    d_o /= n_o

    # Expected disagreement from marginal frequencies.
    n_total = sum(expected_counts.values())
    d_e = 0.0
    vals = list(expected_counts.keys())
    for i, v in enumerate(vals):
        for w in vals[i + 1 :]:
            d_e += expected_counts[v] * expected_counts[w]
    d_e = (2.0 * d_e) / (n_total * (n_total - 1))

    if d_e == 0:
        return 1.0

    return 1.0 - d_o / d_e


def cohen_kappa(labels_a: NDArray[np.integer | np.bool_], labels_b: NDArray[np.integer | np.bool_]) -> float:
    """Compute Cohen's kappa for two annotators on the same items.

    Args:
        labels_a: 1-D array of labels from annotator A (no NaN).
        labels_b: 1-D array of labels from annotator B, same length.

    Returns:
        Kappa in the range [-1, 1]. Returns ``nan`` when the expected
        agreement equals 1 (degenerate case).
    """
    n = len(labels_a)
    if n == 0:
        return float("nan")

    a = np.asarray(labels_a, dtype=np.int8)
    b = np.asarray(labels_b, dtype=np.int8)

    agree = int(np.sum(a == b))
    p_o = agree / n

    a1 = int(np.sum(a == 1))
    b1 = int(np.sum(b == 1))
    p_e = (a1 * b1 + (n - a1) * (n - b1)) / (n * n)

    if p_e == 1.0:
        return float("nan")

    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_alpha(
    data: NDArray[np.floating],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Krippendorff's alpha.

    Resamples items (columns) with replacement and computes alpha on each
    resample.

    Args:
        data: 2-D array of shape (annotators, items) with ``np.nan`` for
            missing annotations.
        n_resamples: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` tuple.
    """
    rng = np.random.default_rng(seed)
    n_items = data.shape[1]
    alphas = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n_items, size=n_items)
        alphas[i] = krippendorff_alpha_nominal(data[:, indices])

    # Drop NaN resamples (can occur with degenerate samples).
    alphas = alphas[~np.isnan(alphas)]
    if len(alphas) == 0:
        return (float("nan"), float("nan"))

    tail = (1.0 - ci) / 2.0
    lower = float(np.percentile(alphas, tail * 100))
    upper = float(np.percentile(alphas, (1.0 - tail) * 100))
    return lower, upper


def percentage_agreement(data: NDArray[np.floating]) -> float:
    """Compute simple percentage agreement across all annotator pairs.

    For each item with >= 2 annotations, counts the fraction of annotator
    pairs that agree, then averages across items.

    Args:
        data: 2-D array of shape (annotators, items), ``np.nan`` for missing.

    Returns:
        Proportion in [0, 1]. Returns ``nan`` if no items have >= 2
        annotations.
    """
    n_items = data.shape[1]
    agreements = []
    for col in range(n_items):
        values = data[:, col]
        valid = values[~np.isnan(values)]
        m = len(valid)
        if m < 2:
            continue
        n_pairs = m * (m - 1) / 2
        n_agree = sum(1 for i in range(m) for j in range(i + 1, m) if valid[i] == valid[j])
        agreements.append(n_agree / n_pairs)

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))
