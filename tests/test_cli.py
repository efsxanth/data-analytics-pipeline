"""The CLI is an adapter: these tests check that argv lands in the right fields
and that unusable combinations exit as usage errors rather than tracebacks.
"""

from __future__ import annotations

import pytest

from da_pipeline.cli import build_config, str2bool

BASE_ARGV = ["--data_analytics_config_fpath", "runs.json", "--response_name", "auc"]


def build(*extra):
    return build_config(BASE_ARGV + list(extra))


class TestDefaults:
    def test_minimal_invocation_produces_a_usable_config(self):
        config = build()

        assert config.response_name == "auc"
        assert config.config_path.name == "runs.json"
        assert config.step_range.as_slice == slice(0, 17)
        assert config.group_ids is None

    def test_export_defaults_match_the_documented_behaviour(self):
        exports = build().exports

        # Only these two are written unless asked; the rest are large frames.
        assert exports.signal_to_noise is True
        assert exports.correlations is True
        assert exports.responses is False
        assert exports.observation_counts is False


class TestParsing:
    def test_step_range_is_one_based_and_inclusive(self):
        config = build("--step_range", "5", "20")
        assert (config.step_range.start, config.step_range.stop) == (5, 20)
        assert config.step_range.as_slice == slice(4, 20)

    def test_group_ids_and_labels_are_paired_by_position(self):
        config = build("--group_ids", "3", "7", "--group_labels", "P1", "P2")
        assert config.group_ids == (3, 7)
        assert config.group_labels == ("P1", "P2")

    def test_metric_arguments_land_in_the_selection(self):
        config = build(
            "--metric_names", "Accuracy", "degradation_1",
            "--outcome_metric", "Accuracy",
            "--predictor_metric_names", "degradation_1",
        )

        assert config.metrics.names == ("Accuracy", "degradation_1")
        assert config.metrics.outcome == "Accuracy"
        assert config.metrics.predictors == ("degradation_1",)

    def test_metric_labels_parse_into_pairs(self):
        config = build("--metric_labels", "degradation_1=Read-ahead", "Accuracy=Score")

        assert dict(config.metrics.labels) == {
            "degradation_1": "Read-ahead", "Accuracy": "Score",
        }

    def test_snr_levels_are_configurable(self):
        config = build("--snr_baseline_level", "0", "--snr_comparison_level", "2")
        assert (config.snr_baseline_level, config.snr_comparison_level) == (0, 2)

    def test_level_ranges_fold_into_a_mapping(self):
        config = build("--response_level_yaxis_ranges", "0:0,2", "1:0,8")
        ranges = config.plots.response_level_yaxis_ranges

        assert ranges == {0: (0.0, 2.0), 1: (0.0, 8.0)}

    def test_boolean_flags_accept_several_spellings(self):
        assert build("--save_responses_csv", "yes").exports.responses is True
        assert build("--save_responses_csv", "0").exports.responses is False
        assert build("--calculate_effect_size", "False").calculate_effect_size is False

    def test_colours_stay_unset_unless_given(self):
        assert build().plots.colours is None
        assert build("--colours", "#111111", "crimson").plots.colours == ("#111111", "crimson")


class TestUsageErrors:
    """Every one of these exits through argparse, so the user sees usage text."""

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param(["--step_range", "9", "2"], id="inverted_step_range"),
            pytest.param(["--step_range", "0", "9"], id="zero_based_step_range"),
            pytest.param(["--group_labels", "P1"], id="labels_without_ids"),
            pytest.param(["--group_ids", "1", "--group_labels", "P1", "P2"], id="label_count"),
            pytest.param(["--snr_comparison_level", "0"], id="snr_levels_equal"),
            pytest.param(["--snr_comparison_level", "-1"], id="negative_snr_level"),
            pytest.param(
                ["--metric_names", "a", "a"], id="metric_named_twice",
            ),
            pytest.param(
                ["--predictor_metric_names", "x", "--component_metric_names", "x"],
                id="predictor_and_component",
            ),
            pytest.param(["--metric_labels", "no_equals_sign"], id="malformed_metric_label"),
            pytest.param(["--outcome_num_bins", "1"], id="single_bin"),
            pytest.param(["--max_workers", "0"], id="no_workers"),
            pytest.param(["--dpi", "0"], id="zero_dpi"),
            pytest.param(["--figure_width", "-1"], id="negative_width"),
            pytest.param(["--signal_to_noise_yaxis_range", "20", "0"], id="inverted_yaxis"),
            pytest.param(["--output_folder_name_suffix", "x" * 40], id="long_suffix"),
        ],
    )
    def test_invalid_combinations_exit(self, extra):
        with pytest.raises(SystemExit):
            build(*extra)

    def test_duplicate_level_level_is_rejected(self):
        # The second entry would silently win, so it is caught rather than folded.
        with pytest.raises(SystemExit):
            build("--response_level_yaxis_ranges", "0:0,2", "0:0,5")

    def test_an_unknown_metric_is_not_rejected_here(self):
        """The parser cannot know what a valid metric name is.

        Metric names are declared by the extract files, so a name that means
        nothing is caught at the start of the run, against the files, where the
        error can list what they actually declared. Rejecting it here would
        require a fixed list -- which is exactly what the open-ended metric set
        removed.
        """
        config = build("--metric_names", "Accuracy", "nothing_declares_this")

        assert config.metrics.names == ("Accuracy", "nothing_declares_this")

    def test_required_arguments_are_required(self):
        with pytest.raises(SystemExit):
            build_config(["--response_name", "auc"])


class TestStr2Bool:
    @pytest.mark.parametrize("value", ["yes", "true", "TRUE", "t", "y", "1", "enabled"])
    def test_truthy_spellings(self, value):
        assert str2bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "F", "n", "0", "disabled"])
    def test_falsy_spellings(self, value):
        assert str2bool(value) is False

    def test_passes_through_actual_booleans(self):
        assert str2bool(True) is True
