"""Eta-squared: how much of the variance in a response is explained by condition."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from da_pipeline.analysis.effect_size import EffectSizeCalculator, compute_eta_squared
from da_pipeline.constants import columns as col


def responses(values_by_condition: dict, level: int = 1) -> pl.DataFrame:
    """One single-observation run per value, so a run median *is* the value.

    Keeps the arithmetic in these tests readable: the aggregation step is
    exercised separately by ``responses_with_runs`` below.
    """
    runs_by_condition = {
        condition: [[value] for value in values]
        for condition, values in values_by_condition.items()
    }
    return responses_with_runs(runs_by_condition, level=level)


def responses_with_runs(runs_by_condition: dict, level: int = 1) -> pl.DataFrame:
    """Frame from {condition: [[run 1 observation values], [run 2 ...], ...]}."""
    rows = []
    for condition, runs in runs_by_condition.items():
        for run_index, observation_values in enumerate(runs):
            for value in observation_values:
                rows.append({
                    col.RUN_ID: f"{condition}.{run_index}",
                    col.CONDITION: condition,
                    col.LEVEL: level,
                    col.GROUP_ID: 1,
                    col.GROUP_LABEL: "group_1",
                    "auc": float(value),
                })
    return pl.DataFrame(rows)


@pytest.fixture
def calculator():
    return EffectSizeCalculator(response_name="auc")


class TestComputeEtaSquared:
    def test_identical_groups_explain_nothing(self):
        groups = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]

        assert compute_eta_squared(groups) == pytest.approx(0.0)

    def test_separated_groups_with_no_internal_spread_explain_everything(self):
        groups = [np.zeros(10), np.ones(10)]

        assert compute_eta_squared(groups) == pytest.approx(1.0)

    def test_result_is_bounded(self):
        rng = np.random.default_rng(0)
        groups = [rng.normal(0, 1, 50), rng.normal(0.5, 1, 50), rng.normal(1.0, 1, 50)]

        value = compute_eta_squared(groups)

        assert 0.0 <= value <= 1.0

    def test_more_separation_means_a_larger_effect(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, 200)

        near = compute_eta_squared([base, base + 0.2])
        far = compute_eta_squared([base, base + 3.0])

        assert far > near

    def test_grand_mean_is_weighted_by_group_size(self):
        """Pins the convention: the textbook one-way ANOVA grand mean.

        The grand mean is the mean of every observation, so a larger group pulls
        it further. This matters because the pipeline's design is unbalanced --
        conditions contribute different numbers of runs and observations -- and an
        unweighted grand mean (the mean of the group means, counting every
        condition equally) gives a materially different answer on such data.
        """
        small = np.array([9.0, 11.0])            # n=2,  mean 10
        large = np.tile([-1.0, 1.0], 49)         # n=98, mean 0

        # Weighted grand mean = 0.2, so ssb ~196.0 against a within-group
        # scatter of 100 -> ~0.662.
        weighted_grand = np.concatenate([small, large]).mean()
        assert weighted_grand == pytest.approx(0.2)

        ssb = 2 * (10 - weighted_grand) ** 2 + 98 * (0 - weighted_grand) ** 2
        assert compute_eta_squared([small, large]) == pytest.approx(ssb / (ssb + 100))
        assert compute_eta_squared([small, large]) == pytest.approx(0.662, abs=1e-3)

        # The unweighted convention would have reported 2500/2600 ~= 0.96 here,
        # attributing far more of the variance to group membership.
        assert compute_eta_squared([small, large]) < 2500 / 2600

    def test_result_never_exceeds_one(self):
        # sst decomposes into within-group scatter plus ssb for any grand mean,
        # so the ratio is bounded however the grand mean is defined.
        rng = np.random.default_rng(3)
        groups = [rng.normal(0, 1, 3), rng.normal(50, 1, 200), rng.normal(-20, 5, 7)]

        assert compute_eta_squared(groups) <= 1.0


class TestByLevel:
    def test_one_row_per_level_level(self, calculator):
        rng = np.random.default_rng(0)
        frames = [
            responses({"A": rng.normal(1, 0.1, 20), "B": rng.normal(2, 0.1, 20)}, level=level)
            for level in (0, 1, 2)
        ]

        result = calculator.eta_squared_by_level(pl.concat(frames))

        assert result.height == 3
        assert sorted(result[col.LEVEL].to_list()) == [0, 1, 2]

    def test_columns_are_level_and_eta_squared(self, calculator):
        result = calculator.eta_squared_by_level(
            responses({"A": [1.0, 1.1, 0.9], "B": [3.0, 3.1, 2.9]})
        )

        # The effect size never travels alone: the interval, the unit of
        # analysis and the replicate counts are what stop the number being
        # misread as more settled than a handful of runs can support.
        assert set(result.columns) == {
            col.LEVEL,
            col.ETA_SQUARED,
            col.ETA_SQUARED_CI_LOW,
            col.ETA_SQUARED_CI_HIGH,
            col.CI_LEVEL,
            col.CI_METHOD,
            col.N_BOOTSTRAP,
            col.ANOVA_P,
            col.UNIT_OF_ANALYSIS,
            col.N_RUNS_TOTAL,
            col.N_CONDITIONS,
            col.N_RUNS_PER_CONDITION,
        }

    def test_well_separated_conditions_give_a_large_effect(self, calculator):
        result = calculator.eta_squared_by_level(
            responses({"A": [1.0, 1.01, 0.99], "B": [9.0, 9.01, 8.99]})
        )

        assert result[col.ETA_SQUARED][0] > 0.95

    def test_single_condition_is_skipped(self, calculator):
        # Eta-squared across conditions needs at least two of them; the run logs
        # and continues rather than failing.
        result = calculator.eta_squared_by_level(responses({"A": [1.0, 2.0, 3.0]}))

        assert result.height == 0


class TestUnitOfAnalysisIsTheRun:
    """Observations inside a run are not independent replicates -- the run is."""

    def test_observations_are_reduced_to_one_median_per_run(self, calculator):
        medians = calculator.run_medians(
            responses_with_runs({"A": [[1.0, 3.0, 5.0], [10.0, 12.0]]})
        )

        assert medians.height == 2                      # two runs, not five observations
        assert sorted(medians["auc"].to_list()) == [3.0, 11.0]

    def test_result_ignores_how_many_observations_a_run_contributed(self, calculator):
        """A run weighs the same whether it held 2 observations or 2,000.

        This is the whole point: with pooled observations, a run that happened to
        yield more observations would pull the answer towards itself.
        """
        small = responses_with_runs({"A": [[1.0], [2.0]], "B": [[5.0], [6.0]]})
        large = responses_with_runs({
            "A": [[1.0] * 500, [2.0] * 900],
            "B": [[5.0] * 50, [6.0] * 3000],
        })

        assert calculator.eta_squared_by_level(small)[col.ETA_SQUARED][0] == pytest.approx(
            calculator.eta_squared_by_level(large)[col.ETA_SQUARED][0]
        )

    def test_pooling_observations_would_have_given_a_different_answer(self, calculator):
        """Guards against a silent revert to observation-level aggregation.

        Every run here is internally noisy but the run medians separate cleanly,
        so the run-level answer is far higher than the observation-level one.
        """
        rng = np.random.default_rng(0)
        noise = lambda centre: (centre + rng.normal(0, 3.0, 200)).tolist()  # noqa: E731

        frame = responses_with_runs({
            "A": [noise(0.0), noise(0.2), noise(-0.1)],
            "B": [noise(5.0), noise(5.2), noise(4.9)],
        })

        run_level = calculator.eta_squared_by_level(frame)[col.ETA_SQUARED][0]

        pooled_groups = [
            frame.filter(pl.col(col.CONDITION) == c)["auc"].to_numpy() for c in ("A", "B")
        ]
        pooled = compute_eta_squared(pooled_groups)

        assert run_level > 0.98
        assert pooled < 0.6

    def test_output_states_the_unit_and_the_replicate_counts(self, calculator):
        result = calculator.eta_squared_by_level(
            responses_with_runs({"A": [[1.0], [1.2], [0.9]], "B": [[5.0], [5.1]]})
        )

        row = result.to_dicts()[0]
        assert row[col.UNIT_OF_ANALYSIS] == col.UNIT_RUN_MEDIAN
        assert row[col.N_RUNS_TOTAL] == 5
        assert row[col.N_CONDITIONS] == 2
        assert row[col.N_RUNS_PER_CONDITION] == "A:3; B:2"


class TestAnovaPValue:
    def test_matches_scipy(self, calculator):
        frame = responses_with_runs({
            "A": [[1.0], [1.4], [0.8], [1.1]],
            "B": [[3.0], [2.6], [3.3], [3.1]],
        })

        result = calculator.eta_squared_by_level(frame)

        expected = stats.f_oneway([1.0, 1.4, 0.8, 1.1], [3.0, 2.6, 3.3, 3.1]).pvalue
        assert result[col.ANOVA_P][0] == pytest.approx(expected)

    def test_is_null_when_every_condition_has_one_run(self, calculator):
        # No within-group scatter, so F is infinite and scipy would report p=0.
        # A fabricated zero is worse than an absent value.
        result = calculator.eta_squared_by_level(
            responses_with_runs({"A": [[1.0]], "B": [[5.0]]})
        )

        assert result[col.ETA_SQUARED][0] == pytest.approx(1.0)
        assert result[col.ANOVA_P][0] is None

    def test_identical_run_medians_are_skipped_rather_than_dividing_by_zero(self, calculator):
        # No total variance to apportion: eta-squared would be 0/0.
        result = calculator.eta_squared_by_level(
            responses_with_runs({"A": [[2.0], [2.0]], "B": [[2.0], [2.0]]})
        )

        assert result.height == 0
