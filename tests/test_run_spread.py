"""Per-condition run spread: is a condition's result carried by one run?"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.analysis.run_spread import RunSpreadSummariser
from da_pipeline.constants import columns as col


def observations(runs_by_condition: dict, value_col: str = "auc") -> pl.DataFrame:
    """Frame from {condition: [[run 1 observation values], [run 2 ...], ...]}."""
    rows = []
    for condition, runs in runs_by_condition.items():
        for run_index, values in enumerate(runs):
            for value in values:
                rows.append({
                    col.CONDITION: condition,
                    col.RUN_ID: f"{condition}.{run_index}",
                    value_col: float(value),
                })
    return pl.DataFrame(rows)


@pytest.fixture
def summariser():
    return RunSpreadSummariser(response_name="auc")


def only_row(rows: list[dict], condition: str) -> dict:
    matches = [r for r in rows if r[col.CONDITION] == condition]
    assert len(matches) == 1
    return matches[0]


class TestSpreadDescribesTheRuns:
    def test_one_row_per_condition(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [2.0]], "B": [[5.0], [6.0]], "C": [[9.0], [8.0]]}), "auc"
        )

        assert {r[col.CONDITION] for r in rows} == {"A", "B", "C"}

    def test_range_is_taken_over_run_medians_not_observations(self, summariser):
        """A run full of scattered observations is still one number here."""
        rows = summariser.summarise_metric(
            observations({"A": [[0.0, 10.0], [4.0, 6.0]]}), "auc"
        )

        row = only_row(rows, "A")
        # Both runs have a median of 5.0, so the runs agree exactly, even though
        # the observations inside them range from 0 to 10.
        assert row[col.RUN_MEDIAN_RANGE] == pytest.approx(0.0)
        assert row[col.CONDITION_MEDIAN] == pytest.approx(5.0)

    def test_min_max_and_range_agree(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [4.0], [9.0]]}), "auc"
        )

        row = only_row(rows, "A")
        assert row[col.RUN_MEDIAN_MIN] == pytest.approx(1.0)
        assert row[col.RUN_MEDIAN_MAX] == pytest.approx(9.0)
        assert row[col.RUN_MEDIAN_RANGE] == pytest.approx(8.0)

    def test_relative_spread_makes_scales_comparable(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[8.0], [10.0], [12.0]]}), "auc"
        )

        # Range 4 around a centre of 10.
        assert only_row(rows, "A")[col.RUN_MEDIAN_RANGE_PCT] == pytest.approx(40.0)

    def test_relative_spread_is_null_rather_than_infinite_at_zero(self, summariser):
        """Drift metrics sit close to zero; a divide there would report inf."""
        rows = summariser.summarise_metric(
            observations({"A": [[-1.0], [0.0], [1.0]]}), "auc"
        )

        assert only_row(rows, "A")[col.RUN_MEDIAN_RANGE_PCT] is None


class TestLeaveOneOut:
    """The headline: how far the answer moves if one run is dropped."""

    def test_the_shift_is_measured_on_the_mean_not_the_median(self, summariser):
        """A median is too robust to answer this question.

        With runs at 9.9, 10.0, 10.1 and 40.0, dropping *any* of them moves the
        median by the same 0.05 -- so a leave-one-out on the median would rank an
        extreme outlier as no more influential than an ordinary run. The mean is
        also what eta-squared and the ANOVA compare, so it is the quantity whose
        sensitivity matters.
        """
        medians = np.array([10.0, 10.1, 9.9, 40.0])

        median_shifts = np.array([
            abs(float(np.median(np.delete(medians, i))) - float(np.median(medians)))
            for i in range(medians.size)
        ])
        # Every run looks equally influential to the median, to within floating
        # point -- so it cannot tell the 40.0 apart from the 9.9.
        assert np.ptp(median_shifts) < 1e-9

        rows = summariser.summarise_metric(
            observations({"A": [[10.0], [10.1], [9.9], [40.0]]}), "auc"
        )
        row = only_row(rows, "A")

        # Dropping the 40.0 moves the mean of the four by ~7.5.
        assert row[col.MAX_LEAVE_ONE_OUT_SHIFT] == pytest.approx(7.5, abs=0.01)
        assert row[col.MOST_INFLUENTIAL_RUN] == "A.3"

    def test_an_outlying_run_produces_a_large_shift(self, summariser):
        tight = observations({"A": [[10.0], [10.1], [9.9], [10.05]]})
        carried = observations({"A": [[10.0], [10.1], [9.9], [40.0]]})

        tight_shift = only_row(summariser.summarise_metric(tight, "auc"), "A")[
            col.MAX_LEAVE_ONE_OUT_SHIFT
        ]
        carried_shift = only_row(summariser.summarise_metric(carried, "auc"), "A")[
            col.MAX_LEAVE_ONE_OUT_SHIFT
        ]

        assert carried_shift > tight_shift

    def test_it_names_the_run_responsible(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [1.1], [0.9], [25.0]]}), "auc"
        )

        assert only_row(rows, "A")[col.MOST_INFLUENTIAL_RUN] == "A.3"

    def test_both_centres_are_reported(self, summariser):
        # They diverge exactly when one run is pulling the result.
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [1.1], [0.9], [25.0]]}), "auc"
        )

        row = only_row(rows, "A")
        assert row[col.CONDITION_MEDIAN] == pytest.approx(1.05)
        assert row[col.CONDITION_MEAN] == pytest.approx(7.0)

    def test_too_few_runs_reports_nothing_rather_than_a_meaningless_number(self, summariser):
        # With two runs, dropping either leaves a single value; the "shift" would
        # just be half the gap and would say nothing about influence.
        rows = summariser.summarise_metric(observations({"A": [[1.0], [9.0]]}), "auc")

        row = only_row(rows, "A")
        assert row[col.MAX_LEAVE_ONE_OUT_SHIFT] is None
        assert row[col.MOST_INFLUENTIAL_RUN] is None
        # The descriptive columns still work.
        assert row[col.RUN_MEDIAN_RANGE] == pytest.approx(8.0)


class TestGuards:
    def test_non_finite_values_are_excluded(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [2.0], [np.inf]]}), "auc"
        )

        # The inf run contributes no usable observation, so it is not a run here.
        assert only_row(rows, "A")[col.N_RUNS_TOTAL] == 2

    def test_an_empty_frame_yields_no_rows(self, summariser):
        empty = observations({"A": [[1.0]]}).clear()

        assert summariser.summarise_metric(empty, "auc") == []

    def test_conditions_are_summarised_independently(self, summariser):
        rows = summariser.summarise_metric(
            observations({"A": [[1.0], [1.0], [1.0]], "B": [[0.0], [5.0], [10.0]]}), "auc"
        )

        assert only_row(rows, "A")[col.RUN_MEDIAN_RANGE] == pytest.approx(0.0)
        assert only_row(rows, "B")[col.RUN_MEDIAN_RANGE] == pytest.approx(10.0)
