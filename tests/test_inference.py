"""The shared bootstrap and multiple-comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from da_pipeline.analysis.inference import (
    benjamini_hochberg,
    bootstrap_ci,
    bootstrap_ci_1sample,
)


def mean_difference(groups: list[np.ndarray]) -> float:
    return float(groups[0].mean() - groups[1].mean())


class TestBootstrapCI:
    def test_the_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        groups = [rng.normal(5, 1, 40), rng.normal(0, 1, 40)]

        low, high = bootstrap_ci(groups, mean_difference)

        assert low < mean_difference(groups) < high

    def test_fewer_runs_give_a_wider_interval(self):
        """The width is the honest signal about how little the design supports."""
        rng = np.random.default_rng(1)
        many = [rng.normal(5, 2, 60), rng.normal(0, 2, 60)]
        few = [many[0][:5], many[1][:5]]

        wide_low, wide_high = bootstrap_ci(few, mean_difference)
        tight_low, tight_high = bootstrap_ci(many, mean_difference)

        assert (wide_high - wide_low) > (tight_high - tight_low)

    def test_it_is_reproducible(self):
        rng = np.random.default_rng(2)
        groups = [rng.normal(1, 1, 20), rng.normal(0, 1, 20)]

        assert bootstrap_ci(groups, mean_difference) == bootstrap_ci(groups, mean_difference)

    def test_group_sizes_are_preserved_in_every_resample(self):
        """A four-run condition must stay a four-run condition."""
        seen = []

        def record(groups):
            seen.append(tuple(g.size for g in groups))
            return mean_difference(groups)

        bootstrap_ci(
            [np.arange(4.0), np.arange(9.0)], record, n_resamples=25
        )

        assert set(seen) == {(4, 9)}

    def test_a_single_valued_group_yields_no_interval(self):
        # Resampling one value returns that value every time, so the "interval"
        # would be an artefact of the other group alone.
        assert bootstrap_ci([np.array([1.0]), np.arange(5.0)], mean_difference) == (None, None)

    def test_degenerate_replicates_are_dropped_not_propagated(self):
        """A statistic may return NaN for an impossible resample."""
        def sometimes_nan(groups):
            if np.ptp(np.concatenate(groups)) == 0:
                return float("nan")
            return mean_difference(groups)

        # Half the values identical, so some resamples are degenerate.
        low, high = bootstrap_ci(
            [np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])], sometimes_nan
        )

        assert low is not None and np.isfinite(low)


class TestTheSeedFixesTheInterval:
    """A fixed seed has to fix the answer, not the answer given this row order.

    This is a regression test for a real defect. The generator emits one fixed
    sequence of index positions; applying it to the same values in a different
    order draws different resamples. Every statistic bootstrapped here is
    order-invariant, so nothing warned that the *interval* was not -- and
    ``group_by`` makes no promise about the order it emits runs in. Two runs of
    the same real dataset returned CLES upper bounds of 0.60 and 0.54 for the
    same comparison before the inputs were sorted.
    """

    def test_grouped_interval_ignores_the_order_within_a_group(self):
        a = np.array([0.71, 0.83, 0.79, 0.88, 0.75])
        b = np.array([0.62, 0.55, 0.69, 0.58, 0.64])

        assert bootstrap_ci([a, b], mean_difference) == bootstrap_ci(
            [a[::-1], b[::-1]], mean_difference
        )

    def test_grouped_interval_ignores_an_arbitrary_permutation(self):
        rng = np.random.default_rng(4)
        a = rng.normal(5, 1, 6)
        b = rng.normal(0, 1, 5)

        for _ in range(5):
            shuffled = [rng.permutation(a), rng.permutation(b)]
            assert bootstrap_ci(shuffled, mean_difference) == bootstrap_ci(
                [a, b], mean_difference
            )

    def test_single_sample_interval_ignores_order_too(self):
        values = np.array([0.42, -0.11, 0.38, 0.19, 0.05, 0.27])

        assert bootstrap_ci_1sample(values, np.median) == bootstrap_ci_1sample(
            values[::-1], np.median
        )


class TestBootstrapCI1Sample:
    def test_the_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(2)
        values = rng.normal(0.5, 0.2, 20)

        low, high = bootstrap_ci_1sample(values, np.median)

        assert low < float(np.median(values)) < high

    def test_two_values_are_too_few_to_resample(self):
        """Resampling two numbers describes the sample, not the estimate."""
        assert bootstrap_ci_1sample(np.array([1.0, 2.0]), np.median) == (None, None)

    def test_non_finite_values_are_dropped(self):
        clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dirty = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0, 5.0])

        assert bootstrap_ci_1sample(dirty, np.median) == bootstrap_ci_1sample(
            clean, np.median
        )


class TestBenjaminiHochberg:
    def test_adjustment_never_shrinks_a_p_value(self):
        raw = [0.001, 0.01, 0.04, 0.5]

        adjusted = benjamini_hochberg(raw)

        assert all(a >= p for a, p in zip(adjusted, raw))

    def test_the_largest_p_value_is_unchanged(self):
        # Its threshold is (m/m) * alpha, so BH leaves it where it is.
        raw = [0.001, 0.01, 0.04]

        assert benjamini_hochberg(raw)[2] == pytest.approx(0.04)

    def test_order_is_preserved(self):
        """A smaller raw p can never end up with a larger adjusted one."""
        raw = [0.04, 0.001, 0.01]

        adjusted = benjamini_hochberg(raw)

        assert np.argsort(raw).tolist() == np.argsort(adjusted).tolist()

    def test_results_are_returned_in_the_input_order(self):
        adjusted = benjamini_hochberg([0.5, 0.001])

        assert adjusted[0] > adjusted[1]

    def test_it_is_less_strict_than_bonferroni(self):
        raw = [0.001, 0.01, 0.04]
        m = len(raw)

        bh = benjamini_hochberg(raw)
        bonferroni = [min(p * m, 1.0) for p in raw]

        assert all(a <= b for a, b in zip(bh, bonferroni))

    def test_untestable_comparisons_do_not_penalise_the_others(self):
        """A pair that could not be tested should not raise everyone else's bar."""
        with_gap = benjamini_hochberg([0.01, None, 0.02])
        without_gap = benjamini_hochberg([0.01, 0.02])

        assert with_gap[1] is None
        assert with_gap[0] == pytest.approx(without_gap[0])
        assert with_gap[2] == pytest.approx(without_gap[1])

    def test_an_empty_family_is_handled(self):
        assert benjamini_hochberg([None, None]) == [None, None]
