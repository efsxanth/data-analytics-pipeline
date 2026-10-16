"""Where a metric's spread actually comes from.

Every quality metric in this pipeline varies for three reasons at once, nested
inside one another:

    condition  ->  run within that condition  ->  observation within that run

A single effect size cannot separate them. Eta-squared says how much of the
total spread tracks the condition, but it is silent on the comparison that
decides whether a condition difference is worth acting on: how large that
difference is *next to how much two runs of the same condition already differ*.
A condition effect no larger than the run-to-run spread around it has weak
support, however many observations were measured.

This module splits the total sum of squares into those three layers::

    SS_total = SS_condition + SS_run(within condition) + SS_observation(within run)

The identity is exact -- the cross terms vanish because each layer is measured
against the mean of the layer above it -- so the three shares always sum to the
whole, with nothing unassigned.

Every metric reports all three shares
------------------------------------
The condition share is an **association**: it says how much of a metric's spread
tracks which condition an observation belonged to. That is a legitimate thing to ask
of the outcome metric, not only of the raw response -- the outcome is what
the work exists to improve, and every condition is processed identically
downstream, so the same fixed transformation reaches all of them.

What the share does *not* license is a mechanistic reading: "the condition
raised the outcome by improving signal-to-noise" needs the relationship analyses,
not this table. An earlier version suppressed the condition share for some
metrics on those grounds, which was the wrong tool -- it removed information a
reader needed instead of qualifying it. Every metric now reports its share, and
the qualification lives here, where it can be explained.
"""

from __future__ import annotations

import logging

import polars as pl

from ..constants import columns as col

logger = logging.getLogger(__name__)

# Working column names, dropped before the frame is returned.
_RUN_MEAN = '_run_mean'
_CONDITION_MEAN = '_condition_mean'
_N = '_n'


class VarianceDecomposer:
    """Splits per-observation metrics into condition / run / observation variance."""

    def __init__(self, response_name: str):
        # Named so the response's rows can be labelled with the level
        # level they came from, which no other metric carries.
        self.response_name = response_name

    # ------------------------------------------------------------------
    # One metric
    # ------------------------------------------------------------------
    def decompose(
        self,
        data_df: pl.DataFrame,
        value_col: str,
        *,
        level: int | None = None,
        metric_name: str | None = None,
    ) -> dict | None:
        """Split one metric's variance across the three layers of the design.

        Args:
            data_df: Per-observation frame carrying the condition and run columns.
            value_col: The metric to decompose.
            level: Recorded on the row when the metric is per-level.
            metric_name: Overrides the reported name; defaults to ``value_col``.

        Returns:
            One row as a dict, or None when the metric cannot be decomposed --
            fewer than two runs, or no variance at all.
        """

        # Signal-to-noise is inf wherever a level 0 median was zero, by design.
        # Those rows would make every sum of squares inf, so they leave here
        # rather than poisoning the whole decomposition.
        usable = (
            data_df
            .select([col.CONDITION, col.RUN_ID, value_col])
            .drop_nulls()
            .filter(pl.col(value_col).is_finite())
        )

        n_observations = usable.height
        n_runs = usable.select(pl.struct([col.CONDITION, col.RUN_ID]).n_unique()).item()
        n_conditions = usable[col.CONDITION].n_unique()

        if n_observations == 0 or n_runs < 2:
            logger.warning(
                f"Skipping variance decomposition for {metric_name or value_col}: "
                f"{n_observations} usable observations across {n_runs} run(s); at least two "
                f"runs are needed to separate run-to-run spread from observation noise."
            )
            return None

        grand_mean = usable[value_col].mean()

        run_means = (
            usable
            .group_by([col.CONDITION, col.RUN_ID])
            .agg(pl.col(value_col).mean().alias(_RUN_MEAN), pl.len().alias(_N))
        )
        condition_means = (
            usable
            .group_by(col.CONDITION)
            .agg(pl.col(value_col).mean().alias(_CONDITION_MEAN), pl.len().alias(_N))
        )

        ss_condition = float(
            condition_means
            .select((pl.col(_N) * (pl.col(_CONDITION_MEAN) - grand_mean) ** 2).sum())
            .item()
        )

        ss_run = float(
            run_means
            .join(condition_means.select([col.CONDITION, _CONDITION_MEAN]), on=col.CONDITION)
            .select((pl.col(_N) * (pl.col(_RUN_MEAN) - pl.col(_CONDITION_MEAN)) ** 2).sum())
            .item()
        )

        ss_observation = float(
            usable
            .join(run_means.select([col.CONDITION, col.RUN_ID, _RUN_MEAN]),
                  on=[col.CONDITION, col.RUN_ID])
            .select(((pl.col(value_col) - pl.col(_RUN_MEAN)) ** 2).sum())
            .item()
        )

        ss_total = ss_condition + ss_run + ss_observation

        if ss_total == 0:
            logger.warning(
                f"Skipping variance decomposition for {metric_name or value_col}: "
                f"every observation holds the same value, so there is no variance to split."
            )
            return None

        return self._build_row(
            metric_name=metric_name or value_col,
            level=level,
            ss_condition=ss_condition,
            ss_run=ss_run,
            ss_observation=ss_observation,
            ss_total=ss_total,
            n_conditions=n_conditions,
            n_runs=n_runs,
            n_observations=n_observations,
        )

    @staticmethod
    def _build_row(
        *,
        metric_name: str,
        level: int | None,
        ss_condition: float,
        ss_run: float,
        ss_observation: float,
        ss_total: float,
        n_conditions: int,
        n_runs: int,
        n_observations: int,
    ) -> dict:
        """Turn the sums of squares into shares of the total variance."""

        # How the run-to-run spread compares with the condition effect it sits
        # beside. Above 1 means two runs of one condition already differ by more
        # than the conditions do, which is weak support for the difference.
        ratio = float(ss_run / ss_condition) if ss_condition > 0 else None

        return {
            col.METRIC: metric_name,
            col.LEVEL: level,
            col.PCT_CONDITION: 100.0 * ss_condition / ss_total,
            col.PCT_RUN_WITHIN_CONDITION: 100.0 * ss_run / ss_total,
            col.PCT_OBSERVATION_WITHIN_RUN: 100.0 * ss_observation / ss_total,
            col.RUN_TO_CONDITION_RATIO: ratio,
            col.SS_CONDITION: ss_condition,
            col.SS_RUN_WITHIN_CONDITION: ss_run,
            col.SS_OBSERVATION_WITHIN_RUN: ss_observation,
            col.SS_TOTAL: ss_total,
            col.N_CONDITIONS: n_conditions,
            col.N_RUNS_TOTAL: n_runs,
            col.N_OBSERVATIONS: n_observations,
        }

    # ------------------------------------------------------------------
    # Every metric of a run
    # ------------------------------------------------------------------
    def build_variance_decomposition_df(
        self,
        *,
        final_responses_df: pl.DataFrame,
        final_signal_to_noise_df: pl.DataFrame,
        final_predictor_metrics_df: pl.DataFrame,
        final_outcome_metrics_df: pl.DataFrame,
        predictor_metric_names: list[str],
        outcome_metric_names: list[str],
    ) -> pl.DataFrame:
        """Decompose the response (per level) and every per-observation metric."""

        rows = []

        for (level,), group in final_responses_df.group_by(
            [col.LEVEL], maintain_order=True
        ):
            row = self.decompose(
                group,
                self.response_name,
                level=level,
                metric_name=f'{self.response_name} (level {level})',
            )
            if row is not None:
                rows.append(row)

        for frame, metrics in (
            (final_signal_to_noise_df, [col.SIGNAL_TO_NOISE_RATIO]),
            (final_predictor_metrics_df, predictor_metric_names),
            (final_outcome_metrics_df, outcome_metric_names),
        ):
            for metric in metrics:
                row = self.decompose(frame, metric)
                if row is not None:
                    rows.append(row)

        variance_df = pl.DataFrame(rows)

        logger.info(f"- Variance decomposition: {variance_df.shape}")

        return variance_df
