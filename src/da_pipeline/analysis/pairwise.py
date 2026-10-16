"""Which conditions differ, by how much, and in which direction.

Eta-squared is an omnibus statistic: it says *something* among the conditions
differs, and stops there. It cannot say whether D beats B, whether A and C are
distinguishable at all, or how large any of those gaps are. This module answers
that, one condition pair at a time.

It covers **every metric**, not only the extracted response. "Does reference_levels
the outcome metric differ between condition A and condition B" is a legitimate
question -- often the most important one, since it is what the work exists to
improve -- and every condition is processed identically downstream, so the same
fixed transformation reaches all of them.

What the comparison supports is an **association**: these conditions differ on
this metric. It does not support "the condition raised the outcome by improving
signal-to-noise", which is a mechanistic claim the relationship analyses
address.

Like eta-squared, it works on **run medians**, never on observations. The
experimental unit is the run.

Why CLES rather than Cohen's d
------------------------------
The common-language effect size is the probability that a randomly chosen run of
one condition scores above a randomly chosen run of the other, with ties
counting a half. At a handful of runs per condition it is simply a count of
pairings -- "in 9 of 9 comparisons D exceeded B" is a fact about the data.
Cohen's d would instead estimate a population parameter from those few numbers
and divide by a standard deviation they barely determine, producing a
confident-looking figure resting on almost nothing.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import polars as pl
from scipy import stats

from ..constants import columns as col
from .inference import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_RESAMPLES,
    benjamini_hochberg,
    bootstrap_ci,
)

logger = logging.getLogger(__name__)

# Below this many runs in either condition, a comparison says nothing: with two
# runs against two the most extreme possible result is still unremarkable.
MIN_RUNS_PER_CONDITION = 2


def common_language_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 * P(a == b), computed exactly over every pairing.

    Equals the Mann-Whitney U statistic divided by the product of the group
    sizes, which is what makes it directly readable as "how often A wins".
    """
    differences = a[:, None] - b[None, :]
    wins = float((differences > 0).sum())
    ties = float((differences == 0).sum())
    return (wins + 0.5 * ties) / (a.size * b.size)


def smallest_attainable_two_sided_p(n_a: int, n_b: int) -> float:
    """The p-value complete separation returns for groups of these sizes.

    Every run of one condition above every run of the other is the most extreme
    arrangement the data can take, so the p-value it produces is a floor: no
    result at these sizes can go below it. Reporting it beside a comparison that
    did not separate is what stops a large p being read as evidence of sameness,
    and it has to be computed from the sizes in hand because it changes sharply
    with them -- three against three cannot go below 0.1, while three against
    two cannot go below 0.2.
    """

    lower, upper = list(range(n_b)), list(range(n_b, n_b + n_a))

    return float(stats.mannwhitneyu(upper, lower, alternative="two-sided").pvalue)


class PairwiseConditionComparator:
    """Compares each pair of conditions on the extracted response."""

    def __init__(
        self,
        response_name: str,
        *,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ):
        self.response_name = response_name
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # One pair
    # ------------------------------------------------------------------
    def _compare(
        self,
        name_a: str,
        values_a: np.ndarray,
        name_b: str,
        values_b: np.ndarray,
        metric_name: str,
        level: int | None,
    ) -> dict | None:
        """One row comparing two conditions' run medians."""

        if values_a.size < MIN_RUNS_PER_CONDITION or values_b.size < MIN_RUNS_PER_CONDITION:
            logger.warning(
                f"Skipping {name_a} vs {name_b} for {metric_name}: "
                f"{values_a.size} and {values_b.size} runs; at least "
                f"{MIN_RUNS_PER_CONDITION} each are needed for the comparison to "
                f"carry any information."
            )
            return None

        effect = common_language_effect_size(values_a, values_b)

        # Exact for these sample sizes, which matters: the smallest attainable
        # p-value at three runs against three is 0.1, and a normal
        # approximation would quietly report something else.
        p_value = float(stats.mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)

        ci_low, ci_high = bootstrap_ci(
            [values_a, values_b],
            lambda groups: common_language_effect_size(groups[0], groups[1]),
            n_resamples=self.n_resamples,
            confidence_level=self.confidence_level,
        )

        return {
            col.METRIC: metric_name,
            col.LEVEL: level,
            col.CONDITION_A: name_a,
            col.CONDITION_B: name_b,
            col.N_RUNS_A: int(values_a.size),
            col.N_RUNS_B: int(values_b.size),
            col.CLES: effect,
            col.CLES_CI_LOW: ci_low,
            col.CLES_CI_HIGH: ci_high,
            col.MEDIAN_DIFFERENCE: float(np.median(values_a) - np.median(values_b)),
            # Named rather than left to the sign of a difference, so the
            # direction survives being read out of order or out of context.
            col.HIGHER_CONDITION: (
                name_a if effect > 0.5 else name_b if effect < 0.5 else None
            ),
            col.P_VALUE: p_value,
            col.CI_LEVEL: self.confidence_level,
            col.CI_METHOD: col.CI_METHOD_BOOTSTRAP_RUNS,
            col.N_BOOTSTRAP: self.n_resamples,
        }

    # ------------------------------------------------------------------
    # Every pair, every level
    # ------------------------------------------------------------------
    def _compare_all_pairs(
        self,
        data_df: pl.DataFrame,
        value_col: str,
        *,
        metric_name: str,
        level: int | None = None,
    ) -> list[dict]:
        """Every condition pair for one metric, corrected as one family.

        The multiple-comparison family is **one metric** (and, for the response,
        one level): the pairs within it are the parts of a single
        question -- "which conditions differ on this?" -- while a different
        metric is a different question and should not make this one harder to
        answer. The family is recorded on each row so a reader can see what was
        corrected against what rather than having to infer it.
        """

        usable = (
            data_df
            .select([col.CONDITION, col.RUN_ID, value_col])
            .drop_nulls()
            .filter(pl.col(value_col).is_finite())
        )

        if usable.is_empty():
            return []

        # Sorted, and not merely for tidiness. CLES and Mann-Whitney do not care
        # what order the run medians arrive in, but the bootstrap does: it draws
        # index positions from a seeded generator and applies them to whatever
        # array it is handed, so the same runs in a different order produce
        # different resamples and a different interval. Polars' group_by does
        # not promise an order, and without this the fixed seed bought nothing
        # -- two runs of the same data returned CLES upper bounds of 0.60 and
        # 0.54 for the same comparison.
        run_medians = (
            usable
            .group_by([col.CONDITION, col.RUN_ID])
            .agg(pl.col(value_col).median().alias(value_col))
            .sort([col.CONDITION, col.RUN_ID])
        )

        by_condition = {
            condition: subset[value_col].to_numpy().astype(float)
            for (condition,), subset in run_medians.group_by(
                [col.CONDITION], maintain_order=True
            )
        }

        rows = [
            row
            for name_a, name_b in itertools.combinations(sorted(by_condition), 2)
            if (
                row := self._compare(
                    name_a, by_condition[name_a],
                    name_b, by_condition[name_b],
                    metric_name=metric_name,
                    level=level,
                )
            ) is not None
        ]

        family = (
            metric_name if level is None
            else f'{metric_name} [{col.LEVEL}={level}]'
        )
        for row, adjusted_p in zip(rows, benjamini_hochberg([r[col.P_VALUE] for r in rows])):
            row[col.P_VALUE_ADJUSTED] = adjusted_p
            row[col.P_ADJUST_METHOD] = col.P_ADJUST_BENJAMINI_HOCHBERG
            row[col.P_ADJUST_FAMILY] = family

        return rows

    def build_pairwise_df(
        self,
        *,
        final_responses_df: pl.DataFrame,
        final_signal_to_noise_df: pl.DataFrame,
        final_predictor_metrics_df: pl.DataFrame,
        final_outcome_metrics_df: pl.DataFrame,
        predictor_metric_names: list[str],
        outcome_metric_names: list[str],
    ) -> pl.DataFrame:
        """Compare every condition pair, on the response and every metric."""

        rows = []

        for (level,), group in final_responses_df.group_by(
            [col.LEVEL], maintain_order=True
        ):
            rows += self._compare_all_pairs(
                group,
                self.response_name,
                metric_name=f'{self.response_name} (level {level})',
                level=level,
            )

        for frame, metrics in (
            (final_signal_to_noise_df, [col.SIGNAL_TO_NOISE_RATIO]),
            (final_predictor_metrics_df, predictor_metric_names),
            (final_outcome_metrics_df, outcome_metric_names),
        ):
            for metric in metrics:
                rows += self._compare_all_pairs(
                    frame, metric, metric_name=metric,
                )

        pairwise_df = pl.DataFrame(rows)

        if pairwise_df.height:
            pairwise_df = pairwise_df.sort(
                [col.METRIC, col.CONDITION_A, col.CONDITION_B]
            )

        logger.info(f"- Pairwise condition comparisons: {pairwise_df.shape}")

        return pairwise_df
