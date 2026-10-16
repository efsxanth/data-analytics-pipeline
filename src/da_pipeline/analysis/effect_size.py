
import logging

import numpy as np
import polars as pl
from scipy import stats

from ..constants import columns as col
from .inference import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_N_RESAMPLES, bootstrap_ci


def _bounded_eta_squared(groups: list[np.ndarray]) -> float:
    """Eta-squared for a bootstrap resample, NaN where it is undefined.

    A resample can draw the same value into every group, leaving no total
    variance to apportion. Returning NaN drops that replicate from the interval
    rather than letting a 0/0 propagate into the percentiles.
    """
    if np.ptp(np.concatenate(groups)) == 0:
        return float('nan')
    return float(compute_eta_squared(groups))


def compute_eta_squared(datasets: list) -> np.float64:
    """
    Computes the eta-squared statistic for a given list of groups.

    Eta-squared is a measure of effect size for the variance explained
    in a one-way ANOVA. It is calculated as the ratio of the sum of
    squares between groups (SSB) to the total sum of squares (SST).

    Args:
        datasets: A list of numpy arrays corresponding to different groups.

    Returns:
        The eta-squared statistic.
    """

    # Mean of each group
    group_means = np.array([np.mean(data) for data in datasets])

    # Size of each group
    group_sizes = np.array([len(data) for data in datasets])

    # Concatenate all observations
    all_data = np.concatenate(datasets)

    # Grand mean (weighted automatically by group size)
    grand_mean = np.mean(all_data)

    # Total sum of squares
    sst = np.sum((all_data - grand_mean) ** 2)

    # Between-groups sum of squares
    ssb = np.sum(group_sizes * (group_means - grand_mean) ** 2)

    # Eta-squared
    eta_squared = ssb / sst

    return eta_squared


def anova_p_value(groups: list[np.ndarray]) -> float | None:
    """One-way ANOVA p-value for the same groups eta-squared is computed from.

    Returns None rather than a number whenever the F statistic is undefined:
    with fewer than two groups there is nothing to compare, and with no
    within-group scatter at all (every group a single observation, or every
    observation identical) F is infinite and scipy would report ``p = 0``. A
    fabricated zero is worse than an absent value, so the column stays null.
    """
    if len(groups) < 2:
        return None

    within_group_scatter = sum(
        float(((g - g.mean()) ** 2).sum()) for g in groups if len(g) > 1
    )
    if within_group_scatter == 0:
        return None

    return float(stats.f_oneway(*groups).pvalue)


class EffectSizeCalculator:
    """
    Computes effect-size tables from responses data.

    The unit of analysis is the **run**, not the observation. Observations inside one run
    were all measured together, under the same setup, on the same day, so they
    are repeated measurements of that run rather than independent observations
    of the condition; treating tens of thousands of them as replicates makes any
    accompanying test wildly anti-conservative. Each run is therefore reduced to
    a single median before the groups are compared, which leaves one observation
    per run -- typically a handful per condition, which is the honest sample size.

    Effect size is deliberately computed on the extracted response only. The
    response is the first quantity read off the instrument, so a difference
    between conditions is attributable to the condition. Downstream metrics such
    as a derived outcome arrive after several derivation steps, with too
    much intervening processing for an effect size to mean the same thing.
    """

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
    # Helpers
    # ------------------------------------------------------------------
    def run_medians(self, df: pl.DataFrame) -> pl.DataFrame:
        """Reduce each run to one median: the experimental unit, one row each."""
        return (
            df.group_by([col.CONDITION, col.RUN_ID])
            .agg(pl.col(self.response_name).median().alias(self.response_name))
            .sort([col.CONDITION, col.RUN_ID])
        )

    def _group_data_by_condition(self, df: pl.DataFrame) -> list[np.ndarray]:
        """
        Groups the per-run medians by condition in a stable condition order.
        Returns a list where each observation holds one condition's run medians.
        """
        grouped = (
            self.run_medians(df)
            .group_by(col.CONDITION)
            .agg(pl.col(self.response_name))
            .sort(col.CONDITION)
        )
        return [
            np.asarray(values, dtype=float)
            for values in grouped[self.response_name].to_list()
        ]

    @staticmethod
    def _describe_run_counts(groups: list[np.ndarray], conditions: list[str]) -> str:
        """'Condition_A:6; Condition_B:5' -- the replicate count behind each group."""
        return "; ".join(
            f"{name}:{len(values)}" for name, values in zip(conditions, groups)
        )

    @staticmethod
    def _validate_single_label(filtered_df: pl.DataFrame, level, group_id) -> str | None:
        """
        Ensures the filtered dataframe contains at most one unique label.
        Returns the label if present, otherwise None.
        """
        if filtered_df.is_empty():
            return None

        unique_labels = filtered_df[col.GROUP_LABEL].unique()

        if len(unique_labels) == 0:
            return None

        if len(unique_labels) != 1:
            raise ValueError(
                f"Expected exactly one label for group_id={group_id} at "
                f"level={level}, but found {len(unique_labels)} labels: {unique_labels}"
            )

        return unique_labels[0]

    def _compute_effect_size(
        self,
        *,
        data_df: pl.DataFrame,
        group_ids: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """
        Eta-squared effect-size engine.

        Parameters
        ----------
        data_df
            data dataframe.
        group_ids
            If provided, compute at (level, group_id) granularity.
            If None, compute at level-only granularity.
        """

        rows = []
        levels = data_df[col.LEVEL].unique()

        group_loop_values = group_ids if group_ids is not None else [None]

        for level in levels:
            for group_id in group_loop_values:

                if group_id is None:
                    filtered = data_df.filter(
                        pl.col(col.LEVEL) == level
                    )
                else:
                    filtered = data_df.filter(
                        (pl.col(col.LEVEL) == level) &
                        (pl.col(col.GROUP_ID) == group_id)
                    )

                if filtered.is_empty():
                    logging.debug(
                        f"No data for level={level}, group_id={group_id}"
                    )
                    continue

                grouped_data = self._group_data_by_condition(filtered)
                n_groups = len(grouped_data)

                if n_groups <= 1:
                    logging.warning(
                        f"Skipping eta-squared for level={level}, "
                        f"group_id={group_id}: need >=2 conditions, found {n_groups}."
                    )
                    continue

                # Every run median identical leaves no total variance to
                # apportion, so the ratio is 0/0. Skip rather than emit a NaN
                # that reads like a computed result.
                if np.ptp(np.concatenate(grouped_data)) == 0:
                    logging.warning(
                        f"Skipping eta-squared for level={level}, "
                        f"group_id={group_id}: every run median is identical, "
                        f"so there is no variance to explain."
                    )
                    continue

                conditions = sorted(filtered[col.CONDITION].unique().to_list())
                value = compute_eta_squared(grouped_data)

                # An interval over runs, because runs are what could have come
                # out differently. At a handful of runs per condition it comes back
                # wide -- that width is the finding, not a defect of the method.
                ci_low, ci_high = bootstrap_ci(
                    grouped_data,
                    _bounded_eta_squared,
                    n_resamples=self.n_resamples,
                    confidence_level=self.confidence_level,
                )

                row = {
                    col.LEVEL: level,
                    col.ETA_SQUARED: value,
                    col.ETA_SQUARED_CI_LOW: ci_low,
                    col.ETA_SQUARED_CI_HIGH: ci_high,
                    col.CI_LEVEL: self.confidence_level,
                    col.CI_METHOD: col.CI_METHOD_BOOTSTRAP_RUNS,
                    col.N_BOOTSTRAP: self.n_resamples,
                    col.ANOVA_P: anova_p_value(grouped_data),
                    col.UNIT_OF_ANALYSIS: col.UNIT_RUN_MEDIAN,
                    col.N_RUNS_TOTAL: sum(len(g) for g in grouped_data),
                    col.N_CONDITIONS: n_groups,
                    col.N_RUNS_PER_CONDITION: self._describe_run_counts(
                        grouped_data, conditions
                    ),
                }

                if group_id is not None:
                    label = self._validate_single_label(filtered, level=level, group_id=group_id)
                    if label is None:
                        logging.warning(
                            f"Skipping eta-squared row for level={level}, "
                            f"group_id={group_id}: no label found."
                        )
                        continue

                    row.update({
                        col.GROUP_ID: group_id,
                        col.GROUP_LABEL: label,
                    })

                rows.append(row)

        effect_size_df = pl.DataFrame(rows)
        logging.info(f"- {col.ETA_SQUARED}: {effect_size_df.shape}")

        return effect_size_df

    # ------------------------------------------------------------------
    # Effect Size
    # ------------------------------------------------------------------
    def eta_squared_by_level(self, data_df: pl.DataFrame) -> pl.DataFrame:
        return self._compute_effect_size(
            data_df=data_df,
            group_ids=None,
        )
