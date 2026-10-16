"""How much the runs of a single condition disagree with each other.

The variance decomposition answers this globally -- what share of a metric's
spread is run-to-run across the whole dataset -- but it cannot say *which*
condition carries that spread. Knowing that a metric is 16% run-driven overall
does not tell you whether one condition's runs are tightly clustered and
another's are scattered.

This module describes each condition's runs on their own terms, so the question
a reader actually asks of a difference between conditions can be answered from a
number instead of by eye:

    is this difference consistent across the runs, or is one run carrying it?

The headline column is ``max_leave_one_out_shift``: how far the condition's
central value moves if the single most influential run is dropped, together with
which run that was. A shift comparable to the difference between two conditions
means the comparison rests on one run and should not be relied on. The per-run
markers on the boxplots show the same thing visually; this puts a figure on it.

Everything here works from per-run medians, never from pooled observations -- see
``analysis.variance`` for why the run is the unit.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ..constants import columns as col

logger = logging.getLogger(__name__)


class RunSpreadSummariser:
    """Summarises the run-to-run spread inside each condition, per metric."""

    def __init__(self, response_name: str):
        self.response_name = response_name

    # ------------------------------------------------------------------
    # One metric, one condition
    # ------------------------------------------------------------------
    @staticmethod
    def _leave_one_out_shift(run_medians: np.ndarray, run_ids: list[str]) -> tuple:
        """Largest move in the condition's **mean** run median from dropping one run.

        The shift is measured on the mean, not the median, for two reasons. The
        median is deliberately robust: with four runs at 9.9, 10.0, 10.1 and
        40.0, dropping *any* of them moves the median by the same 0.05, so a
        leave-one-out on the median reports an extreme outlier as no more
        influential than an ordinary run -- it cannot answer the question this
        column exists for. And the mean is what the downstream statistics
        actually consume: eta-squared and the one-way ANOVA both compare group
        means of the run medians, so influence on the mean is influence on the
        result being reported.

        Returns (shift, run_id), or (None, None) with fewer than three runs,
        where dropping one leaves a single value and the "shift" is just a
        restatement of the gap rather than a measure of influence.
        """
        if run_medians.size < 3:
            return None, None

        baseline = float(run_medians.mean())

        shifts = [
            abs(float(np.delete(run_medians, index).mean()) - baseline)
            for index in range(run_medians.size)
        ]

        worst = int(np.argmax(shifts))
        return float(shifts[worst]), run_ids[worst]

    def summarise_metric(
        self,
        data_df: pl.DataFrame,
        value_col: str,
        *,
        level: int | None = None,
        metric_name: str | None = None,
    ) -> list[dict]:
        """One row per condition describing how far that condition's runs spread."""

        usable = (
            data_df
            .select([col.CONDITION, col.RUN_ID, value_col])
            .drop_nulls()
            .filter(pl.col(value_col).is_finite())
        )

        if usable.is_empty():
            return []

        run_medians = (
            usable
            .group_by([col.CONDITION, col.RUN_ID])
            .agg(pl.col(value_col).median().alias(value_col))
            .sort([col.CONDITION, col.RUN_ID])
        )

        rows = []

        for (condition,), group in run_medians.group_by([col.CONDITION], maintain_order=True):
            medians = group[value_col].to_numpy().astype(float)
            run_ids = group[col.RUN_ID].to_list()

            centre = float(np.median(medians))
            spread = float(medians.max() - medians.min())
            shift, influential_run = self._leave_one_out_shift(medians, run_ids)

            rows.append({
                col.CONDITION: condition,
                col.METRIC: metric_name or value_col,
                col.LEVEL: level,
                col.N_RUNS_TOTAL: medians.size,
                # Two centres, because two things read them: the median matches
                # what the boxplot and its per-run markers suggest, while the
                # mean is what eta-squared and the ANOVA compare, and so is the
                # one the leave-one-out shift is measured against.
                col.CONDITION_MEDIAN: centre,
                col.CONDITION_MEAN: float(medians.mean()),
                col.RUN_MEDIAN_MIN: float(medians.min()),
                col.RUN_MEDIAN_MAX: float(medians.max()),
                col.RUN_MEDIAN_RANGE: spread,
                col.RUN_MEDIAN_IQR: float(
                    np.quantile(medians, 0.75) - np.quantile(medians, 0.25)
                ),
                # Relative spread makes metrics on different scales comparable.
                # Undefined when the centre is at zero, which some metrics can
                # approach, so it is left null rather than reported as infinite.
                col.RUN_MEDIAN_RANGE_PCT: (
                    100.0 * spread / abs(centre) if centre != 0 else None
                ),
                col.MAX_LEAVE_ONE_OUT_SHIFT: shift,
                col.MOST_INFLUENTIAL_RUN: influential_run,
            })

        return rows

    # ------------------------------------------------------------------
    # Every metric of a run
    # ------------------------------------------------------------------
    def build_run_spread_df(
        self,
        *,
        final_responses_df: pl.DataFrame,
        final_signal_to_noise_df: pl.DataFrame,
        final_predictor_metrics_df: pl.DataFrame,
        final_outcome_metrics_df: pl.DataFrame,
        predictor_metric_names: list[str],
        outcome_metric_names: list[str],
    ) -> pl.DataFrame:
        """One row per (condition, metric), covering the response and every metric.

        Unlike the effect size, this carries no attribution claim -- it describes
        how much a condition's own repeats disagree -- so every metric is
        included, downstream ones as well.
        """

        rows = []

        for (level,), group in final_responses_df.group_by(
            [col.LEVEL], maintain_order=True
        ):
            rows += self.summarise_metric(
                group,
                self.response_name,
                level=level,
                metric_name=f'{self.response_name} (level {level})',
            )

        for frame, metrics in (
            (final_signal_to_noise_df, [col.SIGNAL_TO_NOISE_RATIO]),
            (final_predictor_metrics_df, predictor_metric_names),
            (final_outcome_metrics_df, outcome_metric_names),
        ):
            for metric in metrics:
                rows += self.summarise_metric(frame, metric)

        run_spread_df = pl.DataFrame(rows)

        logger.info(f"- Run spread by condition: {run_spread_df.shape}")

        return run_spread_df
