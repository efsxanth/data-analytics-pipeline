"""Pairwise correlations between per-observation metrics, within each condition."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.analysis.correlations import DataCorrelator
from da_pipeline.constants import columns as col

from .conftest import OUTCOME_METRIC


def matrix(run_id: str = "A.0", **metrics) -> pl.DataFrame:
    """A per-observation matrix carrying the metadata keys plus the given metrics."""
    n = len(next(iter(metrics.values())))
    return pl.DataFrame({
        col.RUN_ID: [run_id] * n,
        col.CONDITION: ["Condition_A"] * n,
        col.GROUP_ID: [1] * n,
        col.GROUP_LABEL: ["group_1"] * n,
        col.OBSERVATION_ID: list(range(n)),
        **metrics,
    })


@pytest.fixture
def correlator():
    return DataCorrelator()


def row(df: pl.DataFrame, var_x: str, var_y: str) -> dict:
    found = df.filter((pl.col(col.CORR_VAR_X) == var_x) & (pl.col(col.CORR_VAR_Y) == var_y))
    assert found.height == 1, f"expected one row for {var_x} vs {var_y}"
    return found.to_dicts()[0]


class TestCoefficients:
    def test_perfect_positive_relationship(self, correlator):
        x = np.linspace(0, 1, 50)
        df = correlator.build_correlations_df(matrix(a=x, b=2 * x + 1))

        result = row(df, "a", "b")
        assert result[col.CORR_PEARSON_R] == pytest.approx(1.0)
        assert result[col.CORR_SPEARMAN_R] == pytest.approx(1.0)
        assert result[col.CORR_N_OBS] == 50

    def test_perfect_negative_relationship(self, correlator):
        x = np.linspace(0, 1, 50)
        df = correlator.build_correlations_df(matrix(a=x, b=-3 * x))

        assert row(df, "a", "b")[col.CORR_PEARSON_R] == pytest.approx(-1.0)

    def test_monotone_but_non_linear_separates_the_two_coefficients(self, correlator):
        # Spearman sees the perfect ranking; Pearson sees the curvature.
        x = np.linspace(1, 10, 60)
        df = correlator.build_correlations_df(matrix(a=x, b=np.exp(x / 2)))

        result = row(df, "a", "b")
        assert result[col.CORR_SPEARMAN_R] == pytest.approx(1.0)
        assert result[col.CORR_PEARSON_R] < 0.95

    def test_no_p_values_are_reported(self, correlator):
        """A deliberate omission, so it needs a guard against being helpfully restored.

        A p-value over observations pooled from several runs treats repeated
        measurements of one run as independent replicates; it reads as
        overwhelmingly significant regardless of the truth. The coefficients
        stay because they describe the data honestly.
        """
        rng = np.random.default_rng(0)
        df = correlator.build_correlations_df(
            matrix(a=rng.normal(size=30), b=rng.normal(size=30))
        )

        assert not [c for c in df.columns if c.endswith("_p") or c == "p_value"]
        assert col.CORR_PEARSON_R in df.columns
        assert col.CORR_SPEARMAN_R in df.columns

    def test_every_unordered_pair_appears_once(self, correlator):
        rng = np.random.default_rng(0)
        df = correlator.build_correlations_df(
            matrix(a=rng.normal(size=30), b=rng.normal(size=30), c=rng.normal(size=30))
        )

        pairs = set(zip(df[col.CORR_VAR_X].to_list(), df[col.CORR_VAR_Y].to_list()))
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


class TestPairwiseDeletion:
    def test_non_finite_rows_are_dropped_per_pair(self, correlator):
        # SNR can be inf by design (a zero level 0 median), so this is a real case.
        a = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = row(correlator.build_correlations_df(matrix(a=a, b=b)), "a", "b")

        assert result[col.CORR_N_OBS] == 4
        assert result[col.CORR_PEARSON_R] == pytest.approx(1.0)

    def test_nan_is_dropped_too(self, correlator):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])

        assert row(correlator.build_correlations_df(matrix(a=a, b=b)), "a", "b")[
            col.CORR_N_OBS
        ] == 3

    def test_too_few_points_yields_null_rather_than_raising(self, correlator):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 4.0])

        result = row(correlator.build_correlations_df(matrix(a=a, b=b)), "a", "b")

        assert result[col.CORR_N_OBS] == 2
        assert result[col.CORR_PEARSON_R] is None or np.isnan(result[col.CORR_PEARSON_R])

    def test_constant_variable_yields_null_rather_than_raising(self, correlator):
        # An undefined coefficient, not an exception that aborts the run.
        a = np.ones(20)
        b = np.linspace(0, 1, 20)

        result = row(correlator.build_correlations_df(matrix(a=a, b=b)), "a", "b")

        assert result[col.CORR_PEARSON_R] is None or np.isnan(result[col.CORR_PEARSON_R])


class TestWithinRunCorrelations:
    """Computing inside each run, then summarising across runs."""

    @staticmethod
    def two_runs_with_no_internal_relationship() -> pl.DataFrame:
        """Two runs, X and Y independent inside each, but run 2 higher on both.

        This is the shape that makes pooling misleading: nothing is related to
        anything, yet the two clouds sit on a diagonal.
        """
        rng = np.random.default_rng(7)
        frames = []
        for run_index, offset in enumerate((0.0, 5.0)):
            frames.append(
                matrix(
                    run_id=f"A.{run_index}",
                    a=rng.normal(offset, 0.5, 300),
                    b=rng.normal(offset, 0.5, 300),
                )
            )
        return pl.concat(frames)

    def test_pooling_invents_a_relationship_that_within_run_does_not_see(self, correlator):
        frame = self.two_runs_with_no_internal_relationship()

        pooled = row(correlator.build_correlations_df(frame), "a", "b")
        within = correlator.build_within_run_correlations_df(frame).to_dicts()[0]

        assert pooled[col.CORR_PEARSON_R] > 0.9          # an artefact of pooling
        assert abs(within[col.CORR_PEARSON_R_MEDIAN]) < 0.2   # the truth

    def test_reports_one_coefficient_per_run(self, correlator):
        frame = self.two_runs_with_no_internal_relationship()

        within = correlator.build_within_run_correlations_df(frame).to_dicts()[0]

        assert within[col.CORR_N_RUNS] == 2
        assert within[col.CORR_PEARSON_R_MIN] <= within[col.CORR_PEARSON_R_MEDIAN]
        assert within[col.CORR_PEARSON_R_MEDIAN] <= within[col.CORR_PEARSON_R_MAX]

    def test_sign_agreement_counts_the_majority_side(self, correlator):
        x = np.linspace(0, 1, 40)
        frames = [
            matrix(run_id="A.0", a=x, b=x),        # +1
            matrix(run_id="A.1", a=x, b=x),        # +1
            matrix(run_id="A.2", a=x, b=-x),       # -1
        ]

        within = correlator.build_within_run_correlations_df(pl.concat(frames)).to_dicts()[0]

        assert within[col.CORR_N_RUNS] == 3
        assert within[col.CORR_PEARSON_SAME_SIGN] == 2

    def test_the_unit_of_analysis_is_not_described_as_a_run_median(self, correlator):
        """Each coefficient comes from the observations of one run.

        Calling this "run_median" would suggest the correlation was taken
        *between* run medians -- as few points as a condition has runs -- which is a
        different and far weaker quantity than a coefficient over thousands of
        observations, summarised across runs.
        """
        within = correlator.build_within_run_correlations_df(
            self.two_runs_with_no_internal_relationship()
        )

        assert within[col.UNIT_OF_ANALYSIS][0] == col.UNIT_OBSERVATION_WITHIN_RUN
        assert within[col.UNIT_OF_ANALYSIS][0] != col.UNIT_RUN_MEDIAN

    def test_a_run_too_small_to_correlate_is_dropped_not_fatal(self, correlator):
        x = np.linspace(0, 1, 40)
        frames = [
            matrix(run_id="A.0", a=x, b=x),
            matrix(run_id="A.1", a=np.array([1.0, 2.0]), b=np.array([1.0, 2.0])),  # n < 3
        ]

        within = correlator.build_within_run_correlations_df(pl.concat(frames)).to_dicts()[0]

        assert within[col.CORR_N_RUNS] == 1


class TestBetweenRunCorrelations:
    """Correlating the run medians themselves -- the third view."""

    @staticmethod
    def runs_that_differ_only_in_level() -> pl.DataFrame:
        """Four runs, flat inside, sitting at four levels of both metrics.

        Nothing varies between the observations of a run, so the within-run view has
        nothing to see. The runs themselves are perfectly ordered on both
        metrics, which is precisely what the between-run view exists to detect.
        """
        return pl.concat([
            matrix(run_id=f"A.{index}", a=np.full(20, level), b=np.full(20, 2 * level))
            for index, level in enumerate((1.0, 2.0, 3.0, 4.0))
        ])

    def test_sees_a_run_level_relationship_the_within_run_view_cannot(self, correlator):
        frame = self.runs_that_differ_only_in_level()

        within = correlator.build_within_run_correlations_df(frame).to_dicts()[0]
        between = correlator.build_between_run_correlations_df(frame).to_dicts()[0]

        # No run could produce a coefficient: both metrics are constant inside.
        assert within[col.CORR_N_RUNS] == 0
        assert within[col.CORR_PEARSON_R_MEDIAN] is None

        assert between[col.CORR_N_RUNS] == 4
        assert between[col.CORR_PEARSON_R] == pytest.approx(1.0)

    def test_the_unit_of_analysis_is_the_run_median(self, correlator):
        between = correlator.build_between_run_correlations_df(
            self.runs_that_differ_only_in_level()
        )

        assert between[col.UNIT_OF_ANALYSIS][0] == col.UNIT_RUN_MEDIAN

    def test_all_runs_form_one_coefficient_regardless_of_condition(self, correlator):
        """No condition column: pooling the conditions is the point, not a bug.

        Splitting by condition would leave a handful of runs per coefficient,
        which is why the confound is documented instead of removed.
        """
        frames = [
            matrix(run_id=f"A.{i}", a=np.full(20, float(i)), b=np.full(20, float(i)))
            for i in range(3)
        ] + [
            matrix(run_id=f"B.{i}", a=np.full(20, 10.0 + i), b=np.full(20, 10.0 + i))
            .with_columns(pl.lit("Condition_B").alias(col.CONDITION))
            for i in range(3)
        ]

        between = correlator.build_between_run_correlations_df(pl.concat(frames))

        assert col.CONDITION not in between.columns
        assert between.height == 1
        assert between[col.CORR_N_RUNS][0] == 6

    def test_no_p_values_are_reported_here_either(self, correlator):
        """Runs are independent, but the coefficient is confounded with condition.

        A p-value would test "these runs differ", not "the relationship is
        real", so it would be read as evidence for something it cannot support.
        """
        between = correlator.build_between_run_correlations_df(
            self.runs_that_differ_only_in_level()
        )

        assert not [c for c in between.columns if c.endswith("_p") or c == "p_value"]


class TestGroupingAndColumns:
    def test_conditions_are_correlated_separately(self, correlator):
        x = np.linspace(0, 1, 40)
        frames = [
            matrix(a=x, b=x),                     # perfectly positive
            matrix(a=x, b=-x).with_columns(pl.lit("Condition_B").alias(col.CONDITION)),
        ]

        df = correlator.build_correlations_df(pl.concat(frames))

        by_condition = dict(zip(df[col.CONDITION].to_list(), df[col.CORR_PEARSON_R].to_list()))
        assert by_condition["Condition_A"] == pytest.approx(1.0)
        assert by_condition["Condition_B"] == pytest.approx(-1.0)

    def test_metadata_and_label_columns_are_not_correlated(self, correlator):
        # group_id and observation_index are numeric but are identity, not measurement.
        rng = np.random.default_rng(0)
        df = correlator.build_correlations_df(matrix(a=rng.normal(size=20), b=rng.normal(size=20)))

        correlated = set(df[col.CORR_VAR_X].to_list()) | set(df[col.CORR_VAR_Y].to_list())
        assert correlated == {"a", "b"}

    def test_the_three_views_cover_the_same_pairs(self, correlator):
        base = matrix(x=np.zeros(5))
        snr = base.with_columns(pl.lit(1.0).alias(col.SIGNAL_TO_NOISE_RATIO)).drop("x")
        predictors = base.with_columns(pl.lit(0.1).alias("degradation_1")).drop("x")
        outcomes = base.with_columns(pl.lit(0.9).alias(OUTCOME_METRIC)).drop("x")

        merged = DataCorrelator.build_per_observation_matrix(snr, predictors, outcomes)
        pooled, within_run, between_run = correlator.build_correlation_tables(merged)

        # The first two are per condition, the third across all runs at once,
        # so only the pair sets are comparable -- not the row counts.
        assert pooled.height == within_run.height

        def pairs(df):
            return set(zip(df[col.CORR_VAR_X].to_list(), df[col.CORR_VAR_Y].to_list()))

        assert pairs(pooled) == pairs(within_run) == pairs(between_run)

    def test_merging_the_three_frames_keeps_one_label_column(self):
        base = matrix(x=np.zeros(5))
        snr = base.with_columns(pl.lit(1.0).alias(col.SIGNAL_TO_NOISE_RATIO)).drop("x")
        predictors = base.with_columns(pl.lit(0.1).alias("degradation_1")).drop("x")
        outcomes = base.with_columns(pl.lit(0.9).alias(OUTCOME_METRIC)).drop("x")

        merged = DataCorrelator.build_per_observation_matrix(snr, predictors, outcomes)

        assert merged.columns.count(col.GROUP_LABEL) == 1
        assert f"{col.GROUP_LABEL}_right" not in merged.columns
        assert merged.height == 5
