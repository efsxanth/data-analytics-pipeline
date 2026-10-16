import logging

import polars as pl

from ..config import PipelineConfig
from ..constants import columns as col
from ..metrics import MetricRoles

logger = logging.getLogger(__name__)


class DataBinner:
    """
    Builds bin-level DataFrames.

    The run's outcome metric is divided into ``n_bins`` equal-count (quantile)
    bins computed independently *within each condition*, so every bin in a
    condition holds the same number of observations (+/- 1). Which metric that
    is comes from the run's resolved roles; nothing here names one.

    Note what this means analytically: a bin spans all groups of a condition, so
    if groups differ systematically in the outcome the low bins fill with the
    weaker groups rather than with each group's own low tail. That is the
    intended reading here -- bins describe the condition's population.

    An outcome metric is often highly discrete (only a few dozen distinct values,
    heavily tied near the top), so a single value can span
    a decile boundary. Bins are therefore assigned by sorted *rank* rather than
    by value threshold: this guarantees equal counts, at the cost of occasionally
    splitting identical outcome values across two adjacent bins.

    Two bin columns are produced:

      - ``OUTCOME_BIN_INDEX`` (int, 1..n_bins): the canonical, ordered bin
        identity. Because tie-splitting can give two adjacent bins the *same*
        value range, this index -- not the label -- must be used for grouping,
        joining and sorting downstream.

      - ``OUTCOME_BIN`` (str, e.g. "0.8036-0.8393"): a human-readable range
        label showing the span of outcome values that fell in the bin,
        formatted to 4 decimal places. Display only; not guaranteed unique.
    """

    def __init__(self, config: PipelineConfig, roles: MetricRoles):
        # Number of equal-count bins (deciles by default). Validated as >= 2 by
        # the config, so no fallback is needed here.
        self.n_bins: int = config.outcome_num_bins

        # The column the bins are cut from. Configuration, not a constant: see
        # constants.columns for why no symbol names it.
        self.outcome: str = roles.outcome

    # ------------------------------------------------------------------
    # Outcome equal-count bins
    # ------------------------------------------------------------------
    def build_outcome_bins_df(
        self,
        final_outcome_metrics_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Creates a reusable DataFrame containing outcome values and their
        equal-count bin (index + label) for each observation.

        Bins are computed once, per condition, from the outcome metric and can
        subsequently be joined onto other per-observation datasets (e.g. SNR,
        predictor metrics) using the metadata keys.

        Args:
            final_outcome_metrics_df:
                Outcome-metrics DataFrame containing the outcome column.

        Returns:
            Polars DataFrame containing:
            - metadata columns per observation,
            - outcome values,
            - outcome bin index (ordered identity),
            - outcome bin label (display range).
        """

        outcome = self.outcome
        bin_col = col.OUTCOME_BIN
        idx_col = col.OUTCOME_BIN_INDEX
        part = [col.CONDITION]
        n_bins = self.n_bins

        # Deterministic order within each group: by the outcome, tie-broken on
        # stable metadata so tie-splitting is reproducible run-to-run. The full
        # key is unique per row, so the ordering is deterministic without
        # needing maintain_order.
        sort_cols = part + [
            outcome,
            col.RUN_ID,
            col.OBSERVATION_ID,
        ]

        # Assign a within-group 0-based rank, then map rank -> bin index 1..n.
        # A global sort makes each group a contiguous block, so the group's
        # rank is simply (global row index - group's first global row index).
        ranked = (
            final_outcome_metrics_df
            .select(col.METADATA_COLS_PER_OBSERVATION + [outcome])
            .sort(sort_cols)
            .with_row_index("_gidx")
            .with_columns(
                (pl.col("_gidx") - pl.col("_gidx").min().over(part)).alias("_rank"),
                pl.len().over(part).alias("_n"),
            )
            .with_columns(
                ((pl.col("_rank") * n_bins) // pl.col("_n") + 1)
                .cast(pl.Int64)
                .alias(idx_col)
            )
            .drop("_gidx", "_rank", "_n")
        )

        # Build a small (condition, bin) -> range label lookup and join it back.
        # Formatting to fixed 4dp is done on this tiny frame only.
        label_lut = (
            ranked
            .group_by(part + [idx_col])
            .agg(
                pl.col(outcome).min().round(4).alias("_lo"),
                pl.col(outcome).max().round(4).alias("_hi"),
            )
            .with_columns(
                pl.struct("_lo", "_hi")
                .map_elements(
                    lambda s: f'{s["_lo"]:.4f}-{s["_hi"]:.4f}',
                    return_dtype=pl.Utf8,
                )
                .alias(bin_col)
            )
            .drop("_lo", "_hi")
        )

        outcome_bins_df = (
            ranked
            .join(label_lut, on=part + [idx_col], how="left")
            .select(
                col.METADATA_COLS_PER_OBSERVATION
                + [outcome, idx_col, bin_col]
            )
        )

        logger.info(f"- {outcome} bins: {outcome_bins_df.shape}")

        return outcome_bins_df

    # ------------------------------------------------------------------
    # Attaching bins to other per-observation frames
    # ------------------------------------------------------------------
    def _attach_outcome_bins(
        self,
        data_df: pl.DataFrame,
        outcome_bins_df: pl.DataFrame,
        include_outcome: bool = True,
    ) -> pl.DataFrame:
        """
        Join the outcome bin identity and label onto a per-observation frame.

        Args:
            data_df:
                Any per-observation frame carrying the metadata columns.
            outcome_bins_df:
                Lookup produced by ``build_outcome_bins_df``.
            include_outcome:
                Whether to bring the outcome value across as well. False for
                the outcome-metrics frame, which already holds that column --
                joining it again would produce a suffixed duplicate.

        Returns:
            ``data_df`` augmented with the bin index and label (and, optionally,
            the outcome). The join is inner, so observations missing from either
            side are dropped; that is what ``_warn_on_uneven_bins`` later detects.
        """

        bin_cols = [col.OUTCOME_BIN_INDEX, col.OUTCOME_BIN]

        if include_outcome:
            bin_cols.insert(0, self.outcome)

        return data_df.join(
            outcome_bins_df.select(col.METADATA_COLS_PER_OBSERVATION + bin_cols),
            on=col.METADATA_COLS_PER_OBSERVATION,
            how="inner",
        )

    # ------------------------------------------------------------------
    # SNR x outcome bins
    # ------------------------------------------------------------------
    def build_snr_and_outcome_bins_df(
        self,
        final_signal_to_noise_df: pl.DataFrame,
        outcome_bins_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Attaches outcome values and the outcome bin (index + label) to the
        signal-to-noise DataFrame via the per-observation metadata columns.
        """

        return self._attach_outcome_bins(final_signal_to_noise_df, outcome_bins_df)

    # ------------------------------------------------------------------
    # Predictor metrics x outcome bins
    # ------------------------------------------------------------------
    def build_predictor_and_outcome_bins_df(
        self,
        final_predictor_metrics_df: pl.DataFrame,
        outcome_bins_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Attaches outcome values and the outcome bin (index + label) to the
        predictor-metrics DataFrame via the per-observation metadata columns.
        """

        return self._attach_outcome_bins(final_predictor_metrics_df, outcome_bins_df)

    # ------------------------------------------------------------------
    # Outcome metrics x outcome bins
    # ------------------------------------------------------------------
    def build_outcome_metrics_and_bins_df(
        self,
        final_outcome_metrics_df: pl.DataFrame,
        outcome_bins_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Attaches the outcome bin (index + label) to the outcome-metrics DataFrame.

        The outcome value is not joined across here: this frame is where the bins
        came from, so it already carries that column.
        """

        return self._attach_outcome_bins(
            final_outcome_metrics_df,
            outcome_bins_df,
            include_outcome=False,
        )
