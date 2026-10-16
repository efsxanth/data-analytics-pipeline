
import logging
import os

import polars as pl

from ..config import PipelineConfig
from ..constants import outputs as out
from ..metrics import MetricRoles

logger = logging.getLogger(__name__)


## ----------------------------------------------------------------------------------------------------------------
## Low-level write primitives (domain-agnostic, reusable)
## ----------------------------------------------------------------------------------------------------------------

def _ensure_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def write_csv(df: pl.DataFrame, output_dir: str, filename: str) -> str:
    """
    Write a single DataFrame to output_dir/filename, creating the directory if needed.
    """
    _ensure_dir(output_dir)
    fpath = os.path.join(output_dir, filename)
    df.write_csv(fpath)
    return fpath


class DataSaver:
    """
    Uses a consistent “single CSV per dataset” strategy.

    All CSVs are written under ``<output_dir>/csv_exports``. Bin-summary CSVs are
    further nested under ``<output_dir>/csv_exports/bin_summaries``.
    """

    def __init__(self, config: PipelineConfig, roles: MetricRoles):
        self.config = config
        self.output_dir = str(config.resolved_output_dir)

        # Bin-summary file names carry the outcome metric's own name, so a run
        # organised around a metric called 'yield' writes
        # 'degradation_1_and_yield_bin_summary.csv'.
        self.outcome_metric = roles.outcome

        # Root for every CSV this saver writes.
        self.csv_export_dir = os.path.join(self.output_dir, out.CSV_EXPORTS_SUBDIR)

    # ------------------------------------------------------------------------------------------------------------
    # Generic helper
    # ------------------------------------------------------------------------------------------------------------

    def _save_single_csv(
        self,
        df: pl.DataFrame,
        filename: str,
        label: str,
        subdir: str | None = None,
    ) -> None:
        if df is None or df.height == 0:
            logger.info(f"{label}: nothing to save.")
            return

        output_dir = (
            os.path.join(self.csv_export_dir, subdir)
            if subdir
            else self.csv_export_dir
        )

        write_csv(
            df=df,
            output_dir=output_dir,
            filename=filename,
        )

    # ------------------------------------------------------------------------------------------------------------
    # Dataset-specific wrappers
    # ------------------------------------------------------------------------------------------------------------

    def save_observation_count_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            filename=out.OBSERVATION_COUNT_CSV_FNAME,
            label="Observation count",
        )

    def save_responses_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            filename=out.RESPONSES_CSV_FNAME,
            label="Responses",
        )

    def save_signal_to_noise_ratios_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            filename=out.SNR_CSV_FNAME,
            label="Signal-to-noise",
        )

    def save_predictor_metrics_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            filename=out.PREDICTOR_METRICS_CSV_FNAME,
            label="Predictor metrics",
        )

    def save_outcome_metrics_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            filename=out.OUTCOME_METRICS_CSV_FNAME,
            label="Outcome metrics",
        )

    def save_effect_size_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the effect-size table.

        Always written, unlike the large per-observation exports: it is a handful of
        rows, and it carries the unit of analysis and replicate counts that make
        the eta-squared values on the figure interpretable.
        """
        self._save_single_csv(
            df=df,
            filename=out.EFFECT_SIZE_CSV_FNAME,
            label="Effect size",
        )

    def save_variance_decomposition_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the variance decomposition.

        Always written, for the same reason as the effect size: the figure shows
        shares, but the sums of squares and the replicate counts behind them are
        what make the shares checkable.
        """
        self._save_single_csv(
            df=df,
            filename=out.VARIANCE_CSV_FNAME,
            label="Variance decomposition",
        )

    def save_pairwise_conditions_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the pairwise condition comparisons.

        Written whenever the effect size is calculated: it answers the question
        the omnibus eta-squared raises but cannot settle -- which conditions
        actually differ, and by how much.
        """
        self._save_single_csv(
            df=df,
            filename=out.PAIRWISE_CSV_FNAME,
            label="Pairwise condition comparisons",
        )

    def save_run_spread_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the per-condition run spread.

        Always written. It is the numeric form of what the per-run markers on
        the boxplots show, and the only output that says whether one run is
        carrying a condition's result.
        """
        self._save_single_csv(
            df=df,
            filename=out.RUN_SPREAD_CSV_FNAME,
            label="Run spread by condition",
        )

    def save_within_run_correlations_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            subdir=out.CORRELATIONS_SUBDIR,
            filename=out.WITHIN_RUN_CORRELATIONS_CSV_FNAME,
            label="Within-run correlations",
        )

    def save_between_run_correlations_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            subdir=out.CORRELATIONS_SUBDIR,
            filename=out.BETWEEN_RUN_CORRELATIONS_CSV_FNAME,
            label="Between-run correlations",
        )

    def save_correlations_csv(self, df: pl.DataFrame) -> None:
        self._save_single_csv(
            df=df,
            subdir=out.CORRELATIONS_SUBDIR,
            filename=out.CORRELATIONS_CSV_FNAME,
            label="Correlations",
        )

    def save_outcome_model_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the multivariable model summary.

        Always written when the model runs. One row per predictor, and they
        carry the cross-validated fit that says whether the coefficients beside
        them are worth reading -- which the figure cannot show.
        """
        self._save_single_csv(
            df=df,
            filename=out.MODEL_CSV_FNAME,
            label="outcome model",
        )

    def save_outcome_model_by_run_csv(self, df: pl.DataFrame) -> None:
        """
        Persist the per-run fits behind the model summary.

        The summary reports a median across runs and how many agreed on its
        sign; this file is what lets a reader see the runs themselves, and
        whether one of them is carrying the result.
        """
        self._save_single_csv(
            df=df,
            filename=out.MODEL_BY_RUN_CSV_FNAME,
            label="outcome model by run",
        )

    # ------------------------------------------------------------------------------------------------------------
    # Bin-summary wrappers
    # ------------------------------------------------------------------------------------------------------------

    def save_snr_bin_summary(self, df: pl.DataFrame) -> None:
        """
        Persist the SNR against outcome-bins summary as a single CSV.
        """
        self._save_single_csv(
            df=df,
            subdir=out.BIN_SUMMARY_SUBDIR,
            filename=out.snr_bin_summary_fname(self.outcome_metric),
            label="SNR bin summary",
        )

    def _save_bin_summary_per_metric(self, df: pl.DataFrame, metrics: list, label: str) -> None:
        """
        Persist one bin-summary CSV per metric.

        The incoming frame holds, per bin: the grouping columns, the outcome
        metric's summary columns, the summary columns for every metric in
        ``metrics``, and the per-bin point count. Each output file keeps
        everything *except* the other metrics' columns, so the outcome data (and
        grouping / point-count columns) is retained identically in every file —
        the only difference between them is which metric they carry.

        Dropping the "other" metrics, rather than hand-listing the columns to
        keep, means new stat columns are carried through without touching this
        method.
        """
        if df is None or df.height == 0:
            logger.info(f"{label}: nothing to save.")
            return

        for metric in metrics:
            other_metrics = [m for m in metrics if m != metric]

            # Columns belonging to the metrics we are NOT keeping in this file
            # (all their distribution-stat columns share the "<metric>_" prefix).
            cols_to_drop = [
                col
                for col in df.columns
                if any(col.startswith(f"{other}_") for other in other_metrics)
            ]

            metric_df = df.select(pl.exclude(cols_to_drop)) if cols_to_drop else df

            self._save_single_csv(
                df=metric_df,
                subdir=out.BIN_SUMMARY_SUBDIR,
                filename=out.bin_summary_fname(metric, self.outcome_metric),
                label=f"{metric} bin summary",
            )

    def save_predictor_bin_summaries(self, df: pl.DataFrame, metrics: list) -> None:
        """One CSV per predictor metric, however many the run has."""

        self._save_bin_summary_per_metric(
            df=df,
            metrics=list(metrics),
            label="Predictor bin summaries",
        )

    def save_component_bin_summaries(self, df: pl.DataFrame, metrics: list) -> None:
        """
        One CSV per component of the outcome.

        The outcome itself is absent from this list: it defines the bins, so its
        summary columns are the grouping context every file already carries.
        """

        self._save_bin_summary_per_metric(
            df=df,
            metrics=list(metrics),
            label="Component bin summaries",
        )
