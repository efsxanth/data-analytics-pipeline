"""Small statistical utilities shared by the condition-effect analyses.

Three pieces live here because more than one analysis needs each of them, and
none belongs to one caller more than another:

* a percentile bootstrap that resamples **runs**, never observations, so an interval
  never inherits the pseudo-replication the rest of the pipeline works to avoid,
  in a grouped form (one sample per condition) and a single-sample form (one
  pooled set of run-level values);
* Benjamini-Hochberg (1995), for the several comparisons that come out of splitting
  one question ("do the conditions differ?") into every pair of conditions.

Both are deliberately plain. With a handful of runs per condition there is no
sophistication worth having: the honest output is a wide interval and a
correction that says how many comparisons produced it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Enough resamples that the interval is stable to about a percentage point,
# which is far finer than a handful of runs can actually justify.
DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Fixed so a rerun of the same data reproduces the same interval. An analysis
# whose numbers move when nothing changed is not one anyone can check.
DEFAULT_BOOTSTRAP_SEED = 20260101

# Below three run-level values a single-sample bootstrap has nothing to
# resample: every replicate is one of a handful of arrangements of the same two
# numbers, and the resulting "interval" describes the sample, not the estimate.
MIN_VALUES_FOR_1SAMPLE_CI = 3


def bootstrap_ci(
    groups: Sequence[np.ndarray],
    statistic: Callable[[list[np.ndarray]], float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for a statistic of several groups.

    Each group is resampled with replacement to its own size, which keeps the
    design's shape -- a condition with four runs stays a four-run condition in
    every resample, so the interval reflects how little that condition can
    actually support.

    Args:
        groups: One array of run-level values per condition.
        statistic: Computes the quantity of interest from the groups. May return
            NaN for a degenerate resample (all values identical, say); those
            resamples are dropped rather than poisoning the percentiles.
        n_resamples: Bootstrap replicates.
        confidence_level: 0.95 gives the 2.5th and 97.5th percentiles.
        seed: Fixed for reproducibility.

    Returns:
        (low, high), or (None, None) when the interval would be meaningless --
        fewer than two groups, any group with fewer than two values (resampling
        it just returns the same value every time), or too few usable replicates.
    """

    if len(groups) < 2 or any(np.size(g) < 2 for g in groups):
        return None, None

    # Sorted, because otherwise the seed does not fix what it claims to. The
    # generator produces one fixed sequence of index positions, and applying it
    # to the same values in a different order draws different resamples -- so a
    # caller whose grouping happened to emit its runs in another order would get
    # a different interval from identical data. On this project's data that was
    # a CLES upper bound of 0.60 on one run and 0.54 on the next. Sorting costs
    # nothing at these group sizes, and every statistic bootstrapped here is
    # order-invariant, so it changes no result -- it only makes the result the
    # same one every time.
    groups = [np.sort(np.asarray(g, dtype=float)) for g in groups]

    rng = np.random.default_rng(seed)
    sizes = [np.size(g) for g in groups]

    replicates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        resampled = [
            g[rng.integers(0, size, size)] for g, size in zip(groups, sizes)
        ]
        replicates[index] = statistic(resampled)

    return _percentile_interval(replicates, confidence_level)


def bootstrap_ci_1sample(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Percentile interval for a statistic of one sample of run-level values.

    The grouped form above exists to preserve a design in which each condition
    contributes its own runs. This one is for a quantity that has no condition
    structure to preserve: the multivariable model fits one coefficient per run
    and asks how stable that coefficient is across *all* runs, so the sample is
    the runs themselves and resampling them is the whole procedure.

    Returns:
        (low, high), or (None, None) below three values -- resampling two
        numbers produces an interval running between them and nothing more.
    """

    # Sorted for the same reason as the grouped form above: the seed has to fix
    # the interval, not the interval-given-this-row-order.
    sample = np.asarray(values, dtype=float)
    sample = np.sort(sample[np.isfinite(sample)])

    if sample.size < MIN_VALUES_FOR_1SAMPLE_CI:
        return None, None

    rng = np.random.default_rng(seed)
    n = sample.size

    replicates = np.array(
        [statistic(sample[rng.integers(0, n, n)]) for _ in range(n_resamples)],
        dtype=float,
    )

    return _percentile_interval(replicates, confidence_level)


def _percentile_interval(
    replicates: np.ndarray,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    """The two percentiles of a set of bootstrap replicates.

    Non-finite replicates come from degenerate resamples -- every value
    identical, a singular design matrix -- and are dropped rather than allowed
    to drag a percentile to infinity. If too few survive, the interval would be
    computed from a sample that no longer represents the resampling
    distribution, so nothing is reported at all.
    """

    usable = replicates[np.isfinite(replicates)]
    if usable.size < replicates.size // 2:
        return None, None

    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(usable, [tail, 1.0 - tail])
    return float(low), float(high)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float | None]:
    """Adjusted p-values controlling the false-discovery rate.

    Splitting "do the conditions differ?" into every pair produces several
    tests at once, and at the usual threshold a share of them come out
    significant by luck alone. Benjamini-Hochberg rescales each p so that the
    familiar 0.05 cutoff still means what it did with one test -- controlling
    the expected *proportion* of significant findings that are false.

    Bonferroni would control the chance of any false finding at all, which is
    stricter than exploratory comparisons across conditions warrant.

    The step-up procedure implemented below is:

        Benjamini Y, Hochberg Y (1995). "Controlling the false discovery rate:
        a practical and powerful approach to multiple testing." Journal of the
        Royal Statistical Society: Series B (Methodological) 57(1): 289-300.
        https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

    ``None`` entries pass through untouched and take no part in the ranking, so
    a pair that could not be tested does not make the others harder to clear.
    """

    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]

    if not indexed:
        return [None] * len(p_values)

    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)

    adjusted: list[float | None] = [None] * len(p_values)
    running_min = 1.0

    # Walk from the largest p down, keeping the running minimum: this enforces
    # the monotonicity BH requires, so a smaller raw p can never end up with a
    # larger adjusted one.
    for rank in range(m, 0, -1):
        original_index, p = indexed[rank - 1]
        running_min = min(running_min, p * m / rank)
        adjusted[original_index] = float(min(running_min, 1.0))

    return adjusted
