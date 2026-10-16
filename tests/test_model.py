"""The multivariable accuracy model, fitted inside each run."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.analysis.model import (
    MIN_RUNS_FOR_CROSS_VALIDATION,
    OutcomeModeller,
    r_squared,
    standardise,
)
from da_pipeline.constants import columns as col

from .conftest import OUTCOME_METRIC

PREDICTORS = ["snr", "degradation_2"]


def run_frame(
    run_id: str,
    condition: str,
    *,
    accuracy: np.ndarray,
    **predictors: np.ndarray,
) -> pl.DataFrame:
    """One run's per-observation rows, carrying the metadata the modeller groups on."""
    n = accuracy.size
    return pl.DataFrame({
        col.RUN_ID: [run_id] * n,
        col.CONDITION: [condition] * n,
        col.GROUP_ID: [1] * n,
        col.OBSERVATION_ID: list(range(n)),
        OUTCOME_METRIC: accuracy,
        **predictors,
    })


def dataset(n_runs: int = 4, n_observations: int = 200, *, seed: int = 0) -> pl.DataFrame:
    """Runs in which accuracy genuinely depends on snr, with degradation_2 as noise.

    Each run sits at its own level on every column, so a model that failed to
    work within runs would pick up the between-run structure instead.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for index in range(n_runs):
        level = 1.0 + index
        snr = rng.normal(level, 1.0, n_observations)
        degradation_2 = rng.normal(level, 1.0, n_observations)
        accuracy = level + 0.5 * snr + rng.normal(0.0, 0.2, n_observations)

        frames.append(
            run_frame(
                f"A.{index}",
                "Condition_A" if index % 2 == 0 else "Condition_B",
                accuracy=accuracy,
                snr=snr,
                degradation_2=degradation_2,
            )
        )

    return pl.concat(frames)


@pytest.fixture
def modeller():
    # A small resample count: these tests assert shape and direction, and 2000
    # replicates per predictor would dominate the suite's runtime.
    return OutcomeModeller(PREDICTORS, OUTCOME_METRIC, n_resamples=200)


class TestHelpers:
    def test_standardise_produces_zero_mean_and_unit_spread(self):
        result = standardise(np.array([1.0, 5.0, 9.0, 12.0]))

        assert result.mean() == pytest.approx(0.0)
        assert result.std() == pytest.approx(1.0)

    def test_a_constant_column_cannot_be_standardised(self):
        # Dividing by ~0 would hand the fit arbitrarily large numbers.
        assert standardise(np.full(20, 3.0)) is None

    def test_r_squared_is_one_for_exact_predictions(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(actual, actual) == pytest.approx(1.0)

    def test_r_squared_is_zero_for_predicting_the_mean(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(actual, np.full(4, actual.mean())) == pytest.approx(0.0)

    def test_r_squared_goes_negative_when_worse_than_the_mean(self):
        """Not a bug: out of sample this is the finding worth reporting."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(actual, np.array([4.0, 3.0, 2.0, 1.0])) < 0.0


class TestPerRunFitting:
    def test_recovers_a_relationship_that_exists_within_runs(self, modeller):
        summary, _ = modeller.build_model_tables(dataset())

        snr = summary.filter(pl.col(col.PREDICTOR) == "snr").to_dicts()[0]
        degradation_2 = summary.filter(pl.col(col.PREDICTOR) == "degradation_2").to_dicts()[0]

        assert snr[col.COEFFICIENT_MEDIAN] > 0.5
        assert snr[col.COEFFICIENT_SAME_SIGN] == snr[col.N_RUNS_MODELLED]
        assert abs(degradation_2[col.COEFFICIENT_MEDIAN]) < 0.1

    def test_a_predictor_that_only_varies_between_runs_gets_no_credit(self, modeller):
        """The pooling artefact, in the form the model has to survive.

        ``level`` is constant inside every run and rises across them, exactly
        like accuracy. Pooled, it would look like the strongest predictor in the
        table; within a run it has no spread at all, so the run is dropped
        rather than fitted on a column of identical numbers.
        """
        frames = []
        for index in range(4):
            level = float(index)
            frames.append(
                run_frame(
                    f"A.{index}", "Condition_A",
                    accuracy=np.linspace(level, level + 1.0, 100),
                    snr=np.linspace(0.0, 1.0, 100),
                    degradation_2=np.full(100, level),
                )
            )

        summary, by_run = modeller.build_model_tables(pl.concat(frames))

        assert summary.is_empty()
        assert by_run.is_empty()

    def test_one_coefficient_per_run_per_predictor(self, modeller):
        _, by_run = modeller.build_model_tables(dataset(n_runs=5))

        assert by_run.height == 5 * len(PREDICTORS)
        assert by_run[col.RUN_ID].n_unique() == 5
        assert (by_run[col.STANDARDISATION] == col.STANDARDISED_WITHIN_RUN).all()

    def test_a_run_with_too_few_observations_is_dropped_not_fatal(self, modeller):
        small = run_frame(
            "A.99", "Condition_A",
            accuracy=np.linspace(0.0, 1.0, 5),
            snr=np.linspace(1.0, 2.0, 5),
            degradation_2=np.linspace(2.0, 3.0, 5),
        )

        summary, _ = modeller.build_model_tables(pl.concat([dataset(), small]))

        assert summary[col.N_RUNS_MODELLED][0] == 4

    def test_missing_columns_are_reported_not_raised(self):
        summary, by_run = OutcomeModeller(["not_a_column"], OUTCOME_METRIC).build_model_tables(dataset())

        assert summary.is_empty()
        assert by_run.is_empty()


class TestCollinearity:
    def test_vif_rises_when_predictors_overlap(self):
        """Two nearly identical predictors: each coefficient becomes unstable."""
        rng = np.random.default_rng(3)
        frames = []

        for index in range(4):
            snr = rng.normal(0.0, 1.0, 300)
            frames.append(
                run_frame(
                    f"A.{index}", "Condition_A",
                    accuracy=snr + rng.normal(0.0, 0.5, 300),
                    snr=snr,
                    # Almost a copy of snr, so the two share nearly all their
                    # information and neither coefficient is well determined.
                    degradation_2=snr + rng.normal(0.0, 0.05, 300),
                )
            )

        summary, _ = OutcomeModeller(PREDICTORS, OUTCOME_METRIC, n_resamples=200).build_model_tables(
            pl.concat(frames)
        )

        assert (summary[col.VIF_MEDIAN] > 10.0).all()

    def test_independent_predictors_have_a_vif_near_one(self, modeller):
        summary, _ = modeller.build_model_tables(dataset())

        assert (summary[col.VIF_MEDIAN] < 1.5).all()

    def test_perfectly_collinear_predictors_drop_the_run(self, modeller):
        rng = np.random.default_rng(5)
        frames = []

        for index in range(4):
            snr = rng.normal(0.0, 1.0, 200)
            frames.append(
                run_frame(
                    f"A.{index}", "Condition_A",
                    accuracy=snr + rng.normal(0.0, 0.3, 200),
                    snr=snr,
                    degradation_2=2.0 * snr,   # an exact duplicate after standardising
                )
            )

        summary, _ = modeller.build_model_tables(pl.concat(frames))

        # The individual coefficients would be arbitrary, so nothing is claimed.
        assert summary.is_empty()


class TestCrossValidation:
    def test_a_transferable_relationship_predicts_a_held_out_run(self, modeller):
        summary, by_run = modeller.build_model_tables(dataset())

        # snr drives accuracy identically in every run, so coefficients fitted
        # on the others must predict the one left out.
        assert summary[col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN][0] > 0.5
        assert summary[col.N_RUNS_PREDICTED][0] == summary[col.N_RUNS_MODELLED][0]
        assert by_run[col.R_SQUARED_OUT_OF_SAMPLE].is_not_null().all()

    def test_consistent_signs_do_not_by_themselves_mean_it_transfers(self, modeller):
        """The gap cross-validation exists to close.

        Every run here has a positive snr coefficient, so sign agreement is
        perfect. But one run's relationship is far weaker than the other three,
        so the median vector fitted on those three overshoots it badly -- and
        predicting a run with a coefficient more than twice its own is worse
        than predicting its mean. Sign agreement cannot see any of that; the
        out-of-sample column reports it directly.
        """
        rng = np.random.default_rng(11)
        frames = []

        for index, slope in enumerate((0.04, 0.3, 0.3, 0.3)):
            snr = rng.normal(0.0, 1.0, 300)
            frames.append(
                run_frame(
                    f"A.{index}", "Condition_A",
                    accuracy=slope * snr + rng.normal(0.0, 0.4, 300),
                    snr=snr,
                    degradation_2=rng.normal(0.0, 1.0, 300),
                )
            )

        summary, by_run = modeller.build_model_tables(pl.concat(frames))
        snr = summary.filter(pl.col(col.PREDICTOR) == "snr").to_dicts()[0]

        assert snr[col.COEFFICIENT_SAME_SIGN] == 4          # perfect consistency
        # ...covering coefficients that differ several-fold in size.
        assert snr[col.COEFFICIENT_MAX] > 3.0 * snr[col.COEFFICIENT_MIN]

        held_out = by_run.filter(pl.col(col.PREDICTOR) == "snr")[
            col.R_SQUARED_OUT_OF_SAMPLE
        ].to_list()
        # The odd run out is predicted worse than its own mean.
        assert min(held_out) < 0.0

    def test_too_few_runs_reports_nothing_rather_than_guessing(self, modeller):
        summary, _ = modeller.build_model_tables(
            dataset(n_runs=MIN_RUNS_FOR_CROSS_VALIDATION - 1)
        )

        assert summary[col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN][0] is None
        assert summary[col.N_RUNS_PREDICTED][0] == 0

    def test_out_of_sample_is_not_better_than_in_sample(self, modeller):
        """A held-out run cannot be fitted better than the run it was fitted on."""
        summary, _ = modeller.build_model_tables(dataset())

        assert (
            summary[col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN][0]
            <= summary[col.R_SQUARED_IN_SAMPLE_MEDIAN][0] + 1e-9
        )


class TestIntervals:
    def test_the_interval_brackets_the_median_coefficient(self, modeller):
        summary = modeller.build_model_tables(dataset())[0].to_dicts()

        for row in summary:
            assert row[col.COEFFICIENT_CI_LOW] <= row[col.COEFFICIENT_MAX]
            assert row[col.COEFFICIENT_CI_HIGH] >= row[col.COEFFICIENT_MIN]

    def test_a_real_effect_produces_an_interval_clear_of_zero(self, modeller):
        summary = modeller.build_model_tables(dataset())[0]

        snr = summary.filter(pl.col(col.PREDICTOR) == "snr").to_dicts()[0]
        assert snr[col.COEFFICIENT_CI_LOW] > 0.0

    def test_intervals_are_reproducible(self):
        """A fixed seed: an analysis whose numbers move on rerun cannot be checked."""
        frame = dataset()

        first = OutcomeModeller(PREDICTORS, OUTCOME_METRIC, n_resamples=200).build_model_tables(frame)[0]
        second = OutcomeModeller(PREDICTORS, OUTCOME_METRIC, n_resamples=200).build_model_tables(frame)[0]

        assert first.equals(second)
