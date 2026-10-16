"""Splitting a metric's variance into condition / run / observation layers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.analysis.effect_size import compute_eta_squared
from da_pipeline.analysis.variance import VarianceDecomposer
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
def decomposer():
    return VarianceDecomposer(response_name="auc")


def decompose(decomposer, frame, **kwargs):
    return decomposer.decompose(frame, "auc", **kwargs)


class TestTheDecompositionIsExact:
    """The three layers must account for the whole spread, with nothing left over."""

    def test_sums_of_squares_add_up_to_the_total(self, decomposer):
        rng = np.random.default_rng(0)
        frame = observations({
            "A": [rng.normal(0, 1, 50), rng.normal(0.5, 2, 80), rng.normal(-1, 1, 30)],
            "B": [rng.normal(4, 1, 60), rng.normal(5, 3, 25)],
        })

        row = decompose(decomposer, frame)

        assert row[col.SS_CONDITION] + row[col.SS_RUN_WITHIN_CONDITION] + row[
            col.SS_OBSERVATION_WITHIN_RUN
        ] == pytest.approx(row[col.SS_TOTAL])

    def test_shares_sum_to_one_hundred_percent(self, decomposer):
        rng = np.random.default_rng(1)
        frame = observations({
            "A": [rng.normal(0, 1, 40), rng.normal(1, 1, 40)],
            "B": [rng.normal(3, 1, 40), rng.normal(4, 1, 40)],
        })

        row = decompose(decomposer, frame)

        total = (
            row[col.PCT_CONDITION]
            + row[col.PCT_RUN_WITHIN_CONDITION]
            + row[col.PCT_OBSERVATION_WITHIN_RUN]
        )
        assert total == pytest.approx(100.0)

    def test_condition_share_equals_the_pooled_eta_squared(self, decomposer):
        """They are the same quantity, so the two must never disagree."""
        rng = np.random.default_rng(2)
        frame = observations({
            "A": [rng.normal(0, 1, 70), rng.normal(0.4, 1, 55)],
            "B": [rng.normal(2, 1, 40), rng.normal(2.5, 1, 90)],
        })

        row = decompose(decomposer, frame)

        groups = [
            frame.filter(pl.col(col.CONDITION) == c)["auc"].to_numpy() for c in ("A", "B")
        ]
        assert row[col.PCT_CONDITION] / 100.0 == pytest.approx(compute_eta_squared(groups))


class TestEachLayerIsAttributedCorrectly:
    def test_a_pure_condition_effect_lands_on_condition(self, decomposer):
        # Runs identical within a condition, observations identical within a run.
        frame = observations({"A": [[1.0] * 20, [1.0] * 20], "B": [[9.0] * 20, [9.0] * 20]})

        row = decompose(decomposer, frame)

        assert row[col.PCT_CONDITION] == pytest.approx(100.0)
        assert row[col.PCT_RUN_WITHIN_CONDITION] == pytest.approx(0.0)

    def test_a_pure_run_effect_lands_on_run(self, decomposer):
        # Condition means identical; the runs inside them differ wildly.
        frame = observations({"A": [[0.0] * 20, [10.0] * 20], "B": [[0.0] * 20, [10.0] * 20]})

        row = decompose(decomposer, frame)

        assert row[col.PCT_CONDITION] == pytest.approx(0.0)
        assert row[col.PCT_RUN_WITHIN_CONDITION] == pytest.approx(100.0)

    def test_pure_observation_noise_lands_on_observation(self, decomposer):
        rng = np.random.default_rng(3)
        frame = observations({
            "A": [rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)],
            "B": [rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)],
        })

        row = decompose(decomposer, frame)

        assert row[col.PCT_OBSERVATION_WITHIN_RUN] > 99.0

    def test_ratio_flags_a_condition_effect_smaller_than_its_run_spread(self, decomposer):
        """Above 1 means two runs of one condition differ more than the conditions do."""
        fragile = observations({"A": [[0.0] * 20, [10.0] * 20], "B": [[1.0] * 20, [11.0] * 20]})
        solid = observations({"A": [[0.0] * 20, [0.5] * 20], "B": [[9.0] * 20, [9.5] * 20]})

        assert decompose(decomposer, fragile)[col.RUN_TO_CONDITION_RATIO] > 1
        assert decompose(decomposer, solid)[col.RUN_TO_CONDITION_RATIO] < 1


class TestEveryMetricReportsItsConditionShare:
    """No metric has its result withheld on the grounds of how it was derived.

    An earlier version suppressed the condition share for metrics computed
    further downstream. That removed information rather than qualifying it, and
    the deciding fact makes it unnecessary: every condition is processed
    identically, so the same fixed transformation reaches all of them and a
    difference is still attributable to the condition. The qualification -- that
    this is an association, not a mechanism -- lives in the documentation, where
    it can be explained, rather than in a column that only names a stage.
    """

    def test_a_derived_metric_still_reports_its_condition_share(self, decomposer):
        frame = observations({"A": [[1.0, 2.0], [3.0, 4.0]], "B": [[5.0, 6.0], [7.0, 8.0]]})

        row = decompose(decomposer, frame)

        assert row[col.PCT_CONDITION] > 0
        assert row[col.SS_CONDITION] > 0
        assert row[col.SS_TOTAL] > 0

    def test_no_row_records_a_processing_stage(self, decomposer):
        """A guard against the label being helpfully reintroduced.

        The output is meant to read the same to someone who knows nothing about
        how the metrics were produced, so it names what was measured and not
        where in the pipeline it came from.
        """
        frame = observations({"A": [[1.0, 2.0], [3.0, 4.0]], "B": [[5.0, 6.0], [7.0, 8.0]]})

        row = decompose(decomposer, frame)

        assert not [k for k in row if "stage" in k.lower()]
        assert not [
            v for v in row.values()
            if isinstance(v, str) and ("basecall" in v.lower() or "_feature" in v.lower())
        ]


class TestGuards:
    def test_non_finite_values_are_excluded(self, decomposer):
        """Signal-to-noise is inf where a level 0 median was zero, by design."""
        clean = observations({"A": [[1.0, 2.0], [3.0, 4.0]], "B": [[5.0, 6.0], [7.0, 8.0]]})
        with_inf = pl.concat([
            clean,
            observations({"A": [[np.inf]]}),
        ])

        assert decompose(decomposer, with_inf)[col.N_OBSERVATIONS] == clean.height

    def test_a_single_run_cannot_be_decomposed(self, decomposer):
        # With one run there is no way to tell run-to-run spread from observation noise.
        frame = observations({"A": [[1.0, 2.0, 3.0]]})

        assert decompose(decomposer, frame) is None

    def test_zero_variance_is_skipped_rather_than_dividing_by_zero(self, decomposer):
        frame = observations({"A": [[2.0, 2.0], [2.0, 2.0]], "B": [[2.0, 2.0], [2.0, 2.0]]})

        assert decompose(decomposer, frame) is None

    def test_counts_describe_the_usable_rows(self, decomposer):
        frame = observations({"A": [[1.0, 2.0], [3.0]], "B": [[5.0, 6.0, 7.0]]})

        row = decompose(decomposer, frame)

        assert row[col.N_CONDITIONS] == 2
        assert row[col.N_RUNS_TOTAL] == 3
        assert row[col.N_OBSERVATIONS] == 6
