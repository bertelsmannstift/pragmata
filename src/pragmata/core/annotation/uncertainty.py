"""Shared uncertainty helpers for annotation and eval reporting.

Confidence-interval primitives reused across reporting stacks: IAA bootstraps
Krippendorff's alpha (:func:`percentile_bootstrap`), and eval scoring attaches a
CI to every metric - Wilson intervals for proportions
(:func:`wilson_interval`), percentile bootstrap for continuous per-query means.

Kept dependency-light: NumPy plus the stdlib normal quantile, no SciPy. Lives
under ``core.annotation`` because the bootstrap logic originated in ``iaa.py``;
eval imports from here. Promote to a broader ``core.stats`` only if a wider
stats surface later emerges.
"""

from collections.abc import Callable
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray


def wilson_interval(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at small ``n`` and for extreme
    proportions (all-0 / all-1): it stays within ``[0, 1]`` and remains a
    positive-width interval instead of collapsing, which is why proportion
    metrics use it rather than bootstrapping.

    Args:
        successes: Number of positive outcomes (``0 <= successes <= n``).
        n: Number of trials (the effective denominator).
        alpha: Significance level; the interval is at confidence ``1 - alpha``
            (e.g. ``alpha=0.05`` for a 95% interval).

    Returns:
        ``(ci_lower, ci_upper)`` clamped to ``[0, 1]``. Returns
        ``(nan, nan)`` when ``n <= 0``.

    Raises:
        ValueError: If ``successes`` is not in ``[0, n]``.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return (float(lower), float(upper))


def percentile_bootstrap(
    n_units: int,
    statistic: Callable[[NDArray[np.intp]], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval over a resampling unit.

    Resamples unit indices ``[0, n_units)`` with replacement, applies
    ``statistic`` to each resample, and takes percentile cut-points. NaN
    replicates (from degenerate resamples) are dropped, matching the IAA
    bootstrap.

    Args:
        n_units: Number of resampling units (e.g. queries, items).
        statistic: Maps a resample's index array to a scalar estimate; may
            return ``nan`` for a degenerate resample.
        n_resamples: Number of bootstrap iterations.
        alpha: Significance level; the interval is at confidence ``1 - alpha``
            (e.g. ``alpha=0.05`` for a 95% interval).
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)``. Returns ``(nan, nan)`` when every replicate
        is NaN.
    """
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n_units, size=n_units)
        values[i] = statistic(indices)

    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))

    tail = alpha / 2.0
    lower, upper = np.percentile(values, [tail * 100.0, (1.0 - tail) * 100.0])
    return (float(lower), float(upper))
