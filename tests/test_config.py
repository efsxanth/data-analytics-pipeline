"""The config is the pipeline's contract: everything it rejects is a run that
would otherwise have failed slowly, or worse, produced a plausible wrong answer.
"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from da_pipeline.config import (
    ConfigError,
    ExportSelection,
    PipelineConfig,
    PlotSettings,
    StepRange,
)
from da_pipeline.metrics import MetricSelection


def make_config(**overrides) -> PipelineConfig:
    defaults = {"config_path": Path("runs.json"), "response_name": "auc"}
    return PipelineConfig(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# StepRange
# ---------------------------------------------------------------------------

class TestStepRange:
    def test_converts_to_zero_based_half_open_slice(self):
        # Steps 1-17 inclusive is 17 steps, which is [0:17) as a slice.
        assert StepRange(1, 17).as_slice == slice(0, 17)
        assert len(StepRange(1, 17)) == 17

    def test_mid_range_window_keeps_both_endpoints(self):
        assert StepRange(5, 8).as_slice == slice(4, 8)
        assert len(StepRange(5, 8)) == 4

    def test_single_step_is_allowed(self):
        assert len(StepRange(3, 3)) == 1

    def test_rejects_zero_based_start(self):
        # A user passing 0 means "the first step"; silently accepting it would
        # shift the whole window by one.
        with pytest.raises(ConfigError, match="must be >= 1"):
            StepRange(0, 10)

    def test_rejects_inverted_range(self):
        with pytest.raises(ConfigError, match="must not exceed"):
            StepRange(10, 4)


# ---------------------------------------------------------------------------
# Group IDs and labels
# ---------------------------------------------------------------------------

class TestGroupLabels:
    def test_labels_require_ids(self):
        with pytest.raises(ConfigError, match="requires group_ids"):
            make_config(group_labels=["P1", "P2"])

    def test_label_count_must_match_id_count(self):
        with pytest.raises(ConfigError, match="3 label"):
            make_config(group_ids=[1, 2], group_labels=["P1", "P2", "P3"])

    def test_matching_counts_are_accepted(self):
        config = make_config(group_ids=[1, 2], group_labels=["P1", "P2"])
        assert config.group_ids == (1, 2)
        assert config.group_labels == ("P1", "P2")

    def test_has_group_filter_reflects_whether_a_subset_was_requested(self):
        assert make_config().has_group_filter is False
        assert make_config(group_ids=[3]).has_group_filter is True

    def test_sequences_are_frozen_into_tuples(self):
        # Lists would let a caller mutate a config that worker processes share.
        config = make_config(group_ids=[3, 1])
        assert isinstance(config.group_ids, tuple)


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

class TestAnalysisSettings:
    def test_the_metric_selection_is_not_checked_against_any_fixed_list(self):
        # Which metrics exist is a question about the extract files, and at
        # construction none has been opened. A name that means nothing is caught
        # at the start of the run instead -- see metrics.resolve_metric_roles.
        config = make_config(metrics=MetricSelection(names=["whatever_the_files_say"]))
        assert config.metrics.names == ("whatever_the_files_say",)

    def test_rejects_a_metric_selection_of_the_wrong_type(self):
        with pytest.raises(ConfigError, match="expected a MetricSelection"):
            make_config(metrics=["Accuracy"])

    def test_rejects_snr_levels_that_are_the_same(self):
        with pytest.raises(ConfigError, match="a ratio of a level against itself"):
            make_config(snr_baseline_level=1, snr_comparison_level=1)

    def test_rejects_a_negative_snr_level(self):
        with pytest.raises(ConfigError, match="must be >= 0"):
            make_config(snr_comparison_level=-1)

    def test_snr_levels_default_to_the_first_two(self):
        config = make_config()
        assert (config.snr_baseline_level, config.snr_comparison_level) == (0, 1)

    def test_snr_levels_are_free_to_be_any_other_pair(self):
        config = make_config(snr_baseline_level=0, snr_comparison_level=2)
        assert (config.snr_baseline_level, config.snr_comparison_level) == (0, 2)

    def test_rejects_single_bin(self):
        with pytest.raises(ConfigError, match="must be >= 2"):
            make_config(outcome_num_bins=1)

    def test_rejects_zero_workers(self):
        with pytest.raises(ConfigError, match="must be >= 1"):
            make_config(max_workers=0)

    def test_rejects_empty_response_name(self):
        with pytest.raises(ConfigError, match="non-empty"):
            make_config(response_name="")


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

class TestOutputNaming:
    def test_default_folder_name(self):
        assert make_config().output_folder_name == "Data_Analytics"

    def test_suffix_is_appended(self):
        config = make_config(output_folder_name_suffix="MyAnalysis")
        assert config.output_folder_name == "Data_Analytics_MyAnalysis"

    def test_suffix_is_stripped(self):
        assert make_config(output_folder_name_suffix="  ok  ").output_folder_name_suffix == "ok"

    def test_rejects_suffix_with_spaces(self):
        with pytest.raises(ConfigError, match="cannot contain spaces"):
            make_config(output_folder_name_suffix="my analysis")

    def test_rejects_over_long_suffix(self):
        with pytest.raises(ConfigError, match="cannot exceed"):
            make_config(output_folder_name_suffix="x" * 36)

    def test_output_dir_is_absent_until_the_run_creates_it(self):
        config = make_config()
        assert config.output_dir is None
        with pytest.raises(ConfigError, match="has not been set"):
            _ = config.resolved_output_dir

    def test_with_output_dir_returns_a_new_config(self):
        config = make_config()
        located = config.with_output_dir(Path("/tmp/out"))

        assert located.resolved_output_dir == Path("/tmp/out")
        # The original is untouched: replace(), not mutation.
        assert config.output_dir is None


# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------

class TestPlotSettings:
    def test_rejects_inverted_axis_range(self):
        with pytest.raises(ConfigError, match="must be less than"):
            PlotSettings(signal_to_noise_yaxis_range=(20, 0))

    def test_rejects_inverted_level_range(self):
        with pytest.raises(ConfigError, match="must be less than"):
            PlotSettings(response_level_yaxis_ranges={0: (5.0, 1.0)})

    def test_rejects_negative_level_level(self):
        with pytest.raises(ConfigError, match="must be >= 0"):
            PlotSettings(response_level_yaxis_ranges={-1: (0.0, 1.0)})

    def test_rejects_non_positive_figure_size(self):
        with pytest.raises(ConfigError, match="greater than 0"):
            PlotSettings(figure_width=0)

    def test_rejects_out_of_range_rotation(self):
        with pytest.raises(ConfigError, match="between -360 and 360"):
            PlotSettings(xtick_label_rotation=400)

    def test_colours_default_to_none_so_the_theme_decides(self):
        # Resolving the palette here would drag matplotlib into the config layer.
        assert PlotSettings().colours is None

    def test_figsize_scales_height_only(self):
        settings = PlotSettings(figure_width=9.0, figure_height=4.0)
        assert settings.figsize() == (9.0, 4.0)
        assert settings.figsize(2.0) == (9.0, 8.0)


# ---------------------------------------------------------------------------
# Immutability and transport
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_config_attributes_cannot_be_rebound(self):
        config = make_config()
        with pytest.raises(FrozenInstanceError):
            config.response_name = "other"

    def test_level_ranges_cannot_be_mutated(self):
        # frozen=True alone would leave the dict inside writable.
        config = make_config(plots=PlotSettings(response_level_yaxis_ranges={0: (0, 2)}))
        with pytest.raises(TypeError, match="immutable"):
            config.plots.response_level_yaxis_ranges[1] = (0, 3)

    def test_config_survives_a_pickle_round_trip(self):
        # The plotting layer sends the config to ProcessPoolExecutor workers, so
        # a config that cannot be pickled fails only under multiprocessing.
        config = make_config(
            group_ids=[1, 2],
            metrics=MetricSelection(names=["Accuracy"], labels={"Accuracy": "Score"}),
            plots=PlotSettings(response_level_yaxis_ranges={0: (0.0, 2.0)}),
            exports=ExportSelection(responses=True),
        )

        restored = pickle.loads(pickle.dumps(config))

        assert restored == config
        assert restored.plots.response_level_yaxis_ranges[0] == (0.0, 2.0)
        # Still read-only on the other side.
        with pytest.raises(TypeError):
            restored.plots.response_level_yaxis_ranges[0] = (1.0, 2.0)


def test_describe_lists_every_setting():
    lines = make_config(group_ids=[3]).describe()
    joined = "\n".join(lines)

    assert "- response_name: auc" in joined
    assert "- group_ids: [3]" in joined
    assert "- metrics.outcome: Accuracy" in joined
    assert "- snr_baseline_level: 0" in joined
    # Nested settings are flattened so the log shows the whole run.
    assert "- exports.signal_to_noise: True" in joined
    assert "- plots.dpi: 300" in joined
