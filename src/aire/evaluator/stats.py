"""Statistics for comparing evaluation runs (stdlib only, seeded, reproducible).

- bootstrap_mean_ci: percentile bootstrap interval for one run's mean.
- paired_bootstrap: the mean per-case difference between two runs over the
  same cases, its interval, and a two-sided p-value from the same resamples.
- mcnemar_exact: exact McNemar test for metrics that are 0 or 1 per case.
- holm_adjust: Holm's step-down correction for a family of p-values.
- cohens_kappa: chance-corrected agreement between two labellers.

Percentile convention. With B resamples and level 1 - alpha, the interval is
(sorted[k], sorted[B - 1 - k]) with k = ceil(B * alpha / 2) - 1, and the
p-value is 2 * min(#{m >= 0}, #{m <= 0}) / B. With that pairing, the interval
excludes zero exactly when p < alpha, so the interval and the test can never
disagree about significance.

Intervals describe uncertainty from which cases were sampled into the eval
set; they say nothing about run-to-run noise of a nondeterministic system.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


def _tail_index(n_resamples: int, alpha: float) -> int:
    return max(0, math.ceil(n_resamples * alpha / 2) - 1)


def _resampled_means(values: Sequence[float], n_resamples: int, rng: random.Random) -> list[float]:
    n = len(values)
    return sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for the mean; (None, None) for no values."""
    if not values:
        return None, None
    if min(values) == max(values):
        return float(values[0]), float(values[0])
    means = _resampled_means(list(values), n_resamples, random.Random(seed))
    k = _tail_index(n_resamples, 1.0 - confidence)
    return means[k], means[n_resamples - 1 - k]


@dataclass(frozen=True)
class PairedTest:
    delta: float  # mean of candidate - baseline over the pairs
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    test: str = "paired_bootstrap"


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedTest:
    """Bootstrap the mean per-case difference candidate - baseline."""
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have the same number of values")
    if not baseline:
        raise ValueError("paired_bootstrap needs at least one pair")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    n = len(diffs)
    delta = sum(diffs) / n
    if min(diffs) == max(diffs):
        return PairedTest(delta, delta, delta, 1.0 if delta == 0 else 0.0, n)
    means = _resampled_means(diffs, n_resamples, random.Random(seed))
    k = _tail_index(n_resamples, 1.0 - confidence)
    at_or_above = sum(1 for m in means if m >= 0)
    at_or_below = sum(1 for m in means if m <= 0)
    p_value = min(1.0, 2.0 * min(at_or_above, at_or_below) / n_resamples)
    return PairedTest(delta, means[k], means[n_resamples - 1 - k], p_value, n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    b = cases where only the baseline scored 1, c = cases where only the
    candidate did. Under no difference each discordant case is a fair coin.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two label sequences; NaN when there are no items."""
    if len(a) != len(b):
        raise ValueError("both labellers must label the same items")
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(k) / n) * (list(b).count(k) / n) for k in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
