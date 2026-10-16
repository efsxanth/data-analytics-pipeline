"""Per-bin distribution summaries."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.analysis.bin_summaries import DataBinSummariser
from da_pipeline.analysis.binning import DataBinner
from da_pipeline.config import PipelineConfig
from da_pipeline.constants import columns as col
from da_pipeline.metrics import MetricRoles

from .conftest import OUTCOME_METRIC

# Stand-ins for whatever a producer happens to declare. The summariser is told
# which columns to summarise, so the names below are arbitrary on purpose.
PREDICTORS = ("degradation_1", "degradation_2", "degradation_3")
COMPONENTS = ("Accuracy_error_1", "Accuracy_error_2")


@pytest.fixture
def summariser():
    return DataBinSummariser(outcome=OUTCOME_METRIC)


@pytest.fixture
def snr_bins_df(tmp_path, per_observation_frame):
    config = PipelineConfig(
        config_path=tmp_path / "runs.json", response_name="auc", outcome_num_bins=5
    )
    roles = MetricRoles(outcome=OUTCOME_METRIC, components=(), predictors=())
    binner = DataBinner(config=config, roles=roles)
    bins = binner.build_outcome_bins_df(per_observation_frame)

    rng = np.random.default_rng(0)
    snr = per_observation_frame.drop(OUTCOME_METRIC).with_columns(
        pl.Series(col.SIGNAL_TO_NOISE_RATIO, rng.normal(5.0, 1.0, per_observation_frame.height))
    )

    return binner.build_snr_and_outcome_bins_df(snr, bins)


class TestShape:
    def test_one_row_per_condition_and_bin(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)

        expected = snr_bins_df.select(
            [col.CONDITION, col.OUTCOME_BIN_INDEX]
        ).unique().height

        assert summary.height == expected

    def test_point_counts_account_for_every_input_row(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)

        assert summary[col.N_POINTS_BIN].sum() == snr_bins_df.height

    def test_rows_are_sorted_by_condition_then_bin_index(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)

        assert summary.equals(summary.sort([col.CONDITION, col.OUTCOME_BIN_INDEX]))


class TestStatistics:
    def test_every_metric_gets_the_full_stat_block(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)

        for suffix in (
            "median", "q10", "q25", "q75", "q90",
            "central_spread_iqr", "outer_spread",
            "quantile_skew", "tail_skew", "quantile_kurtosis",
        ):
            assert f"{col.SIGNAL_TO_NOISE_RATIO}_{suffix}" in summary.columns
            # The outcome is summarised alongside, in every file.
            assert f"{OUTCOME_METRIC}_{suffix}" in summary.columns

    def test_quantiles_are_ordered(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)
        prefix = col.SIGNAL_TO_NOISE_RATIO

        assert (summary[f"{prefix}_q10"] <= summary[f"{prefix}_q25"]).all()
        assert (summary[f"{prefix}_q25"] <= summary[f"{prefix}_median"]).all()
        assert (summary[f"{prefix}_median"] <= summary[f"{prefix}_q75"]).all()
        assert (summary[f"{prefix}_q75"] <= summary[f"{prefix}_q90"]).all()

    def test_spreads_are_non_negative(self, summariser, snr_bins_df):
        summary = summariser.build_snr_and_outcome_bin_summaries_df(snr_bins_df)
        prefix = col.SIGNAL_TO_NOISE_RATIO

        assert (summary[f"{prefix}_central_spread_iqr"] >= 0).all()
        assert (summary[f"{prefix}_outer_spread"] >= 0).all()

    def test_zero_spread_yields_null_shape_statistics(self, summariser, snr_bins_df):
        # A constant metric makes the IQR zero; the ratios are undefined rather
        # than infinite, and must not raise.
        constant = snr_bins_df.with_columns(pl.lit(1.0).alias(col.SIGNAL_TO_NOISE_RATIO))

        summary = summariser.build_snr_and_outcome_bin_summaries_df(constant)

        assert summary[f"{col.SIGNAL_TO_NOISE_RATIO}_quantile_skew"].is_nan().all()


class TestMultiMetricSummaries:
    def _with_metrics(self, snr_bins_df, names, seed):
        rng = np.random.default_rng(seed)
        return snr_bins_df.drop(col.SIGNAL_TO_NOISE_RATIO).with_columns(
            [
                pl.Series(name, rng.uniform(0, 0.2, snr_bins_df.height))
                for name in names
            ]
        )

    def test_predictor_summary_covers_every_predictor_it_was_given(
        self, summariser, snr_bins_df
    ):
        frame = self._with_metrics(snr_bins_df, PREDICTORS, seed=1)

        summary = summariser.build_predictor_and_outcome_bin_summaries_df(
            frame, list(PREDICTORS)
        )

        for name in PREDICTORS:
            assert f"{name}_median" in summary.columns

    def test_any_number_of_predictors_is_summarised(self, summariser, snr_bins_df):
        # The metric set is open-ended, so seven is as valid as three.
        many = tuple(f"degradation_{i}" for i in range(1, 8))
        frame = self._with_metrics(snr_bins_df, many, seed=3)

        summary = summariser.build_predictor_and_outcome_bin_summaries_df(frame, list(many))

        for name in many:
            assert f"{name}_median" in summary.columns

    def test_component_summary_excludes_the_binning_variable(self, summariser, snr_bins_df):
        frame = self._with_metrics(snr_bins_df, COMPONENTS, seed=2)

        summary = summariser.build_component_and_outcome_bin_summaries_df(
            frame, list(COMPONENTS)
        )

        # The outcome defines the bins, so it appears as context, never as a
        # metric summarised against them.
        assert OUTCOME_METRIC not in COMPONENTS
        for name in COMPONENTS:
            assert f"{name}_median" in summary.columns
        assert f"{OUTCOME_METRIC}_median" in summary.columns
