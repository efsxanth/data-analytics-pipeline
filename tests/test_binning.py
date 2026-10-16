"""Equal-count binning of the run's outcome metric.

The binner's contract is stronger than "cut the range into ten": every bin
within a condition holds the same number of observations, give or take one. That
is what makes a bin comparable to the bin beside it, and it is only achievable
by ranking rather than by value thresholds -- an outcome is often discrete enough
that a single repeated value can straddle a decile boundary.

Which metric is binned is configuration, so these tests pass the roles in rather
than assuming a name.
"""

from __future__ import annotations

import polars as pl
import pytest

from da_pipeline.analysis.binning import DataBinner
from da_pipeline.config import PipelineConfig
from da_pipeline.constants import columns as col
from da_pipeline.metrics import MetricRoles

from .conftest import OUTCOME_METRIC


@pytest.fixture
def frame_roles():
    """Roles matching the hand-built per-observation frame: outcome only."""
    return MetricRoles(outcome=OUTCOME_METRIC, components=(), predictors=())


def make_binner(tmp_path, roles, n_bins=10):
    config = PipelineConfig(
        config_path=tmp_path / "runs.json", response_name="auc", outcome_num_bins=n_bins
    )
    return DataBinner(config=config, roles=roles)


@pytest.fixture
def binner(tmp_path, frame_roles):
    return make_binner(tmp_path, frame_roles)


@pytest.fixture
def bins_df(binner, per_observation_frame):
    return binner.build_outcome_bins_df(per_observation_frame)


class TestEqualCount:
    def test_every_bin_holds_the_same_number_of_observations(self, bins_df):
        # The fixture has 100 observations per condition and 10 bins, so the counts
        # divide exactly: no +/- 1 tolerance is needed here.
        counts = (
            bins_df
            .group_by([col.CONDITION, col.OUTCOME_BIN_INDEX])
            .len()["len"]
            .to_list()
        )

        assert set(counts) == {10}

    def test_counts_stay_within_one_when_they_do_not_divide(self, binner, per_observation_frame):
        # 97 rows over 10 bins cannot be equal; the guarantee is +/- 1.
        uneven = per_observation_frame.filter(pl.col(col.CONDITION) == "Condition_A").head(97)

        counts = (
            binner.build_outcome_bins_df(uneven)
            .group_by(col.OUTCOME_BIN_INDEX)
            .len()["len"]
        )

        assert counts.max() - counts.min() <= 1

    def test_bin_count_follows_the_config(self, tmp_path, frame_roles, per_observation_frame):
        bins = make_binner(tmp_path, frame_roles, n_bins=4).build_outcome_bins_df(
            per_observation_frame
        )

        assert bins[col.OUTCOME_BIN_INDEX].unique().len() == 4

    def test_no_row_is_dropped(self, bins_df, per_observation_frame):
        assert bins_df.height == per_observation_frame.height


class TestTheBinnedColumnIsTheConfiguredOutcome:
    """Nothing here assumes which metric the bins were cut from."""

    def test_bins_are_cut_from_whatever_the_roles_name(self, tmp_path, per_observation_frame):
        renamed = per_observation_frame.rename({OUTCOME_METRIC: "yield"})
        roles = MetricRoles(outcome="yield", components=(), predictors=())

        bins = make_binner(tmp_path, roles).build_outcome_bins_df(renamed)

        assert "yield" in bins.columns
        assert OUTCOME_METRIC not in bins.columns

    def test_a_frame_without_the_named_outcome_is_an_error(self, tmp_path, per_observation_frame):
        roles = MetricRoles(outcome="not_in_the_frame", components=(), predictors=())

        with pytest.raises(Exception):  # noqa: B017 - Polars ColumnNotFoundError
            make_binner(tmp_path, roles).build_outcome_bins_df(per_observation_frame)


class TestBinIdentity:
    def test_index_is_one_based_and_contiguous(self, bins_df):
        indices = sorted(bins_df[col.OUTCOME_BIN_INDEX].unique().to_list())
        assert indices == list(range(1, 11))

    def test_higher_bin_index_means_a_higher_outcome(self, bins_df):
        # The index is an ordered identity, so this ordering is the whole point.
        medians = (
            bins_df
            .filter(pl.col(col.CONDITION) == "Condition_A")
            .group_by(col.OUTCOME_BIN_INDEX)
            .agg(pl.col(OUTCOME_METRIC).median().alias("m"))
            .sort(col.OUTCOME_BIN_INDEX)["m"]
            .to_list()
        )

        assert medians == sorted(medians)

    def test_bins_are_computed_within_each_condition(self, bins_df):
        # A high-quality and a low-quality condition should not share edges:
        # bins describe a condition's own population.
        top_bin = bins_df.filter(pl.col(col.OUTCOME_BIN_INDEX) == 1)
        by_condition = (
            top_bin.group_by(col.CONDITION)
            .agg(pl.col(OUTCOME_METRIC).max().alias("edge"))
            .sort(col.CONDITION)["edge"]
            .to_list()
        )

        assert by_condition[0] != by_condition[1]

    def test_label_is_a_formatted_range(self, bins_df):
        label = bins_df[col.OUTCOME_BIN][0]
        low, _, high = label.partition("-")

        assert float(low) <= float(high)
        assert len(low.split(".")[1]) == 4      # fixed 4dp, as documented

    def test_binning_is_reproducible(self, binner, per_observation_frame):
        # Ties are split by a stable metadata key, so two runs must agree.
        first = binner.build_outcome_bins_df(per_observation_frame)
        second = binner.build_outcome_bins_df(per_observation_frame)

        assert first.equals(second)


class TestAttachingBins:
    def test_snr_frame_keeps_every_row_and_gains_the_bin_columns(
        self, binner, bins_df, per_observation_frame
    ):
        snr = per_observation_frame.drop(OUTCOME_METRIC).with_columns(
            pl.lit(4.0).alias(col.SIGNAL_TO_NOISE_RATIO)
        )

        joined = binner.build_snr_and_outcome_bins_df(snr, bins_df)

        assert joined.height == snr.height
        assert col.OUTCOME_BIN_INDEX in joined.columns
        assert OUTCOME_METRIC in joined.columns

    def test_predictor_frame_gains_the_outcome_and_the_bins(
        self, binner, bins_df, per_observation_frame
    ):
        predictors = per_observation_frame.drop(OUTCOME_METRIC).with_columns(
            pl.lit(0.02).alias("degradation_1")
        )

        joined = binner.build_predictor_and_outcome_bins_df(predictors, bins_df)

        assert joined.height == predictors.height
        assert OUTCOME_METRIC in joined.columns
        assert col.OUTCOME_BIN in joined.columns

    def test_outcome_frame_does_not_get_a_duplicated_outcome_column(
        self, binner, bins_df, per_observation_frame
    ):
        # The outcome frame is where the bins came from, so joining the outcome
        # back would produce a suffixed duplicate.
        joined = binner.build_outcome_metrics_and_bins_df(per_observation_frame, bins_df)

        assert f"{OUTCOME_METRIC}_right" not in joined.columns
        assert joined.columns.count(OUTCOME_METRIC) == 1
