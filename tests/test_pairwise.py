"""Pairwise condition comparisons on the extracted response."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from da_pipeline.analysis.pairwise import (
    PairwiseConditionComparator,
    common_language_effect_size,
)
from da_pipeline.constants import columns as col


def responses(runs_by_condition: dict, level: int = 1) -> pl.DataFrame:
    """Frame from {condition: [[run 1 observation values], [run 2 ...], ...]}."""
    rows = []
    for condition, runs in runs_by_condition.items():
        for run_index, values in enumerate(runs):
            for value in values:
                rows.append({
                    col.RUN_ID: f"{condition}.{run_index}",
                    col.CONDITION: condition,
                    col.LEVEL: level,
                    "auc": float(value),
                })
    return pl.DataFrame(rows)


def single_run_responses(values_by_condition: dict, level: int = 1) -> pl.DataFrame:
    """One run per value, so a run median is the value itself."""
    return responses(
        {c: [[v] for v in values] for c, values in values_by_condition.items()},
        level=level,
    )


@pytest.fixture
def comparator():
    # Fewer resamples keeps the suite quick; the interval is still stable.
    return PairwiseConditionComparator(response_name="auc", n_resamples=300)


EMPTY = pl.DataFrame(
    schema={col.CONDITION: pl.Utf8, col.RUN_ID: pl.Utf8, col.SIGNAL_TO_NOISE_RATIO: pl.Float64}
)


def response_only(comparator, frame: pl.DataFrame) -> pl.DataFrame:
    """Compare on the response alone, with no downstream metrics supplied."""
    return comparator.build_pairwise_df(
        final_responses_df=frame,
        final_signal_to_noise_df=EMPTY,
        final_predictor_metrics_df=EMPTY,
        final_outcome_metrics_df=EMPTY,
        predictor_metric_names=[],
        outcome_metric_names=[],
    )


class TestCommonLanguageEffectSize:
    def test_complete_separation_is_one(self):
        assert common_language_effect_size(
            np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0])
        ) == pytest.approx(1.0)

    def test_complete_separation_the_other_way_is_zero(self):
        assert common_language_effect_size(
            np.array([1.0, 2.0]), np.array([5.0, 6.0, 7.0])
        ) == pytest.approx(0.0)

    def test_identical_groups_are_a_coin_flip(self):
        values = np.array([1.0, 2.0, 3.0])

        assert common_language_effect_size(values, values) == pytest.approx(0.5)

    def test_ties_count_a_half(self):
        # One pairing, and it is a tie.
        assert common_language_effect_size(
            np.array([1.0]), np.array([1.0])
        ) == pytest.approx(0.5)

    def test_it_equals_the_mann_whitney_statistic_over_the_pairings(self):
        """A cross-check against scipy: CLES is U / (n1 * n2)."""
        rng = np.random.default_rng(0)
        a, b = rng.normal(1, 1, 7), rng.normal(0, 1, 5)

        u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic

        assert common_language_effect_size(a, b) == pytest.approx(u / (a.size * b.size))


class TestComparisons:
    def test_every_unordered_pair_appears_once_per_level(self, comparator):
        df = response_only(comparator,
            single_run_responses({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
        )

        pairs = set(zip(df[col.CONDITION_A].to_list(), df[col.CONDITION_B].to_list()))
        assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_it_names_the_higher_condition(self, comparator):
        df = response_only(comparator,
            single_run_responses({"A": [1.0, 1.5, 2.0], "B": [8.0, 9.0, 10.0]})
        )

        row = df.to_dicts()[0]
        assert row[col.HIGHER_CONDITION] == "B"
        assert row[col.CLES] == pytest.approx(0.0)      # A never beats B

    def test_it_works_on_run_medians_not_observations(self, comparator):
        """A run of many scattered observations is still one observation."""
        scattered = responses({
            "A": [[0.0, 2.0], [0.5, 1.5], [0.8, 1.2]],   # medians 1.0, 1.0, 1.0
            "B": [[9.0, 11.0], [9.5, 10.5], [9.8, 10.2]],  # medians 10.0 each
        })

        row = response_only(comparator, scattered).to_dicts()[0]

        assert row[col.N_RUNS_A] == 3
        assert row[col.MEDIAN_DIFFERENCE] == pytest.approx(-9.0)

    def test_the_interval_is_reported_with_the_estimate(self, comparator):
        df = response_only(comparator,
            single_run_responses({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        )

        row = df.to_dicts()[0]
        assert row[col.CLES_CI_LOW] <= row[col.CLES] <= row[col.CLES_CI_HIGH]
        assert row[col.CI_METHOD] == col.CI_METHOD_BOOTSTRAP_RUNS

    def test_a_condition_with_one_run_is_skipped(self, comparator):
        df = response_only(comparator,
            single_run_responses({"A": [1.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0]})
        )

        pairs = set(zip(df[col.CONDITION_A].to_list(), df[col.CONDITION_B].to_list()))
        assert pairs == {("B", "C")}


class TestSmallSamplesAreHonest:
    def test_perfect_separation_can_still_be_non_significant(self, comparator):
        """The point of reporting the effect size beside the p-value.

        Three runs against three separate completely -- CLES is 1.0 -- yet the
        smallest attainable two-sided p at that size is 0.1, so nothing can be
        called significant. Both facts belong in the output.
        """
        df = response_only(comparator,
            single_run_responses({"A": [1.0, 2.0, 3.0], "B": [10.0, 11.0, 12.0]})
        )

        row = df.to_dicts()[0]
        assert row[col.CLES] == pytest.approx(0.0)     # complete separation
        assert row[col.P_VALUE] >= 0.1                 # yet not significant


class TestReproducibility:
    """The fixed bootstrap seed has to actually fix something."""

    def test_the_interval_does_not_depend_on_the_row_order_of_the_input(self, comparator):
        """A regression test for a real bug, not a hypothetical one.

        CLES and Mann-Whitney are order-invariant, so this looked safe. The
        bootstrap is not: it draws index positions from the seeded generator and
        applies them to whatever array it was handed, so the same runs in a
        different order give different resamples. Polars' ``group_by`` does not
        promise an order, and two runs of the same real dataset returned CLES
        upper bounds of 0.60 and 0.54 for the same comparison.
        """
        frame = single_run_responses({
            "Condition_A": [0.71, 0.83, 0.79, 0.88, 0.75],
            "Condition_B": [0.62, 0.55, 0.69, 0.58, 0.64],
        })

        # Reversed *rows*, so every run keeps its own value and only the order
        # the grouping meets them in changes. Reversing the values instead would
        # move them between runs, which is different data.
        original = response_only(comparator, frame)
        reordered = response_only(comparator, frame.reverse())

        assert original[col.CLES_CI_LOW].to_list() == reordered[col.CLES_CI_LOW].to_list()
        assert original[col.CLES_CI_HIGH].to_list() == reordered[col.CLES_CI_HIGH].to_list()

    def test_two_identical_calls_agree_completely(self, comparator):
        frame = single_run_responses({
            "Condition_A": [0.71, 0.83, 0.79, 0.88],
            "Condition_B": [0.62, 0.55, 0.69, 0.58],
        })

        assert response_only(comparator, frame).equals(response_only(comparator, frame))


class TestMultipleComparisons:
    def test_the_family_is_one_metric_at_one_level_level(self, comparator):
        """Each level is its own question, so it gets its own family.

        Correcting across levels would make "which conditions differ at level 1?"
        harder to answer because level 0 was also asked.
        """
        frame = pl.concat([
            single_run_responses({"A": [1.0, 2.0, 3.0], "B": [7.0, 8.0, 9.0]}, level=0),
            single_run_responses({"A": [1.0, 2.0, 3.0], "B": [7.0, 8.0, 9.0]}, level=1),
        ])

        df = response_only(comparator, frame)

        assert set(df[col.P_ADJUST_FAMILY].to_list()) == {
            f"auc (level 0) [{col.LEVEL}=0]",
            f"auc (level 1) [{col.LEVEL}=1]",
        }

    def test_adjustment_only_raises_p_values(self, comparator):
        df = response_only(comparator,
            single_run_responses({
                "A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0],
                "C": [7.0, 8.0, 9.0], "D": [10.0, 11.0, 12.0],
            })
        )

        assert (df[col.P_VALUE_ADJUSTED] >= df[col.P_VALUE]).all()
        assert (df[col.P_ADJUST_METHOD] == col.P_ADJUST_BENJAMINI_HOCHBERG).all()

    def test_more_pairs_mean_a_harsher_correction_for_the_same_raw_p(self, comparator):
        """Six comparisons cost more than one, which is the whole point."""
        two = response_only(comparator,
            single_run_responses({"A": [1.0, 2.0, 3.0], "B": [7.0, 8.0, 9.0]})
        ).to_dicts()[0]

        four = response_only(comparator,
            single_run_responses({
                "A": [1.0, 2.0, 3.0], "B": [7.0, 8.0, 9.0],
                "C": [13.0, 14.0, 15.0], "D": [19.0, 20.0, 21.0],
            })
        )
        ab = four.filter(
            (pl.col(col.CONDITION_A) == "A") & (pl.col(col.CONDITION_B) == "B")
        ).to_dicts()[0]

        assert ab[col.P_VALUE] == pytest.approx(two[col.P_VALUE])
        assert ab[col.P_VALUE_ADJUSTED] >= two[col.P_VALUE_ADJUSTED]
