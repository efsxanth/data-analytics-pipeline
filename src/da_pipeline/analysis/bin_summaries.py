
import logging

import numpy as np
import polars as pl

from ..constants import columns as col
from ..frames import sort_dataframe


class DataBinSummariser:
    """
    Builds summary DataFrames from bin-level data.

    Bins are grouped by ``OUTCOME_BIN_INDEX`` (the ordered, unique bin identity)
    rather than by the range label, because equal-count tie-splitting can give
    two adjacent bins the same label. The range label is carried through into the
    summary as a display column.

    Every summary describes the run's outcome metric alongside the metrics being
    summarised against it, so the summariser is told which metric that is and
    names none itself.
    """

    def __init__(self, outcome: str):
        """
        Args:
            outcome: The run's outcome metric -- the column the bins were cut
                from, summarised into every bin row as the grouping context.
        """
        self.outcome = outcome

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_distribution_stats(row: dict, values: np.ndarray, prefix: str):

        if values.size == 0:
            return

        q10 = np.quantile(values, 0.10)
        q25 = np.quantile(values, 0.25)
        q50 = np.quantile(values, 0.50)
        q75 = np.quantile(values, 0.75)
        q90 = np.quantile(values, 0.90)

        central_spread = q75 - q25
        outer_spread = q90 - q10

        row[f"{prefix}{col.SUFFIX_MEDIAN}"] = q50
        row[f"{prefix}{col.SUFFIX_Q10}"] = q10
        row[f"{prefix}{col.SUFFIX_Q25}"] = q25
        row[f"{prefix}{col.SUFFIX_Q75}"] = q75
        row[f"{prefix}{col.SUFFIX_Q90}"] = q90

        row[f"{prefix}{col.SUFFIX_CENTRAL_SPREAD_IQR}"] = central_spread
        row[f"{prefix}{col.SUFFIX_OUTER_SPREAD}"] = outer_spread

        row[f"{prefix}{col.SUFFIX_QUANTILE_SKEW}"] = (
            (q75 + q25 - 2.0 * q50) / central_spread
            if central_spread != 0 else np.nan
        )

        row[f"{prefix}{col.SUFFIX_TAIL_SKEW}"] = (
            (q90 + q10 - 2.0 * q50) / outer_spread
            if outer_spread != 0 else np.nan
        )

        row[f"{prefix}{col.SUFFIX_QUANTILE_KURTOSIS}"] = (
            (outer_spread / central_spread)
            if central_spread != 0 else np.nan
        )

    def _build_summary_df(
        self,
        *,
        bins_df: pl.DataFrame,
        value_columns: list[str],
        pair: str,
    ) -> pl.DataFrame:

        # Group by condition and the ordered bin *index* (unique identity).
        # The outcome range label is functionally dependent on the group and
        # is carried through as a display column. Groups are pooled within the
        # condition, so no group column takes part in the grouping.
        group_cols = [
            col.CONDITION,
            col.OUTCOME_BIN_INDEX,
        ]

        rows = []

        for key, group in bins_df.group_by(group_cols):

            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key))

            # Representative range label for this bin index (display only).
            row[col.OUTCOME_BIN] = group[col.OUTCOME_BIN][0]

            # Outcome stats
            outcome_values = group[self.outcome].to_numpy()
            self._add_distribution_stats(row, outcome_values, self.outcome)

            # Value columns stats
            for column in value_columns:
                values = group[column].to_numpy()
                self._add_distribution_stats(row, values, column)

            row[col.N_POINTS_BIN] = outcome_values.size

            rows.append(row)

        summary_df = pl.DataFrame(rows)

        summary_df = sort_dataframe(
            df=summary_df,
            columns_to_sort=group_cols,
        )

        logging.info(f"- {pair} bin summary: {summary_df.shape}")

        return summary_df

    # ------------------------------------------------------------------
    # SNR bin summaries
    # ------------------------------------------------------------------
    def build_snr_and_outcome_bin_summaries_df(self, snr_and_outcome_bins_df: pl.DataFrame):

        return self._build_summary_df(
            bins_df=snr_and_outcome_bins_df,
            value_columns=[col.SIGNAL_TO_NOISE_RATIO],
            pair=f"SNR and {self.outcome}",
        )

    # ------------------------------------------------------------------
    # Predictor-metric bin summaries
    # ------------------------------------------------------------------
    def build_predictor_and_outcome_bin_summaries_df(
        self,
        predictor_and_outcome_bins_df: pl.DataFrame,
        predictor_metric_names: list[str],
    ):

        return self._build_summary_df(
            bins_df=predictor_and_outcome_bins_df,
            value_columns=list(predictor_metric_names),
            pair=f"Predictor metrics and {self.outcome}",
        )

    # ------------------------------------------------------------------
    # Component-metric bin summaries
    # ------------------------------------------------------------------
    def build_component_and_outcome_bin_summaries_df(
        self,
        outcome_metrics_and_bins_df: pl.DataFrame,
        component_metric_names: list[str],
    ):
        """
        Summarise the outcome's components against its own bins.

        The outcome itself is not in ``component_metric_names``: it defines the
        bins, so its summary columns are the grouping context every row already
        carries.
        """

        return self._build_summary_df(
            bins_df=outcome_metrics_and_bins_df,
            value_columns=list(component_metric_names),
            pair=f"Component metrics and {self.outcome}",
        )
