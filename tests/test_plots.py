"""What the plotting layer puts on a figure, without rendering one.

Two things the open-ended metric set introduced are easy to plumb wrongly and
invisible in a PNG: the display label a metric is drawn under, and the word used
for the outcome's bins. Both are asserted here by capturing what the plotter
hands its drawing helpers, so the tests stay fast and say what they check.
"""

from __future__ import annotations

import polars as pl
import pytest

from da_pipeline.config import PipelineConfig
from da_pipeline.constants import columns as col
from da_pipeline.constants import display as disp
from da_pipeline.metrics import MetricRoles
from da_pipeline.plotting.plots import DataPlotter

from .conftest import OUTCOME_METRIC

CONDITIONS = ["Condition_A", "Condition_B"]
COLOURS = {"Condition_A": "#111111", "Condition_B": "#222222"}


def make_plotter(tmp_path, roles):
    config = PipelineConfig(
        config_path=tmp_path / "runs.json", response_name="auc"
    ).with_output_dir(tmp_path / "out")

    return DataPlotter(
        config=config,
        roles=roles,
        condition_order=CONDITIONS,
        condition_colours=COLOURS,
        condition_run_counts=dict.fromkeys(CONDITIONS, 2),
        condition_plot_labels={c: c for c in CONDITIONS},
    )


@pytest.fixture
def frame():
    return pl.DataFrame({
        col.GROUP_LABEL: ["group_1"] * 4,
        col.RUN_ID: ["Condition_A.0", "Condition_A.1", "Condition_B.0", "Condition_B.1"],
        col.CONDITION: ["Condition_A", "Condition_A", "Condition_B", "Condition_B"],
        "degradation_1": [0.01, 0.02, 0.03, 0.04],
    })


class TestMetricLabelsReachTheFigure:
    """``--metric_labels degradation_1=Read-ahead`` has to change what is drawn."""

    def _captured_title(self, monkeypatch, plotter, frame, metric):
        captured = {}

        def fake_boxplot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotter, "_boxplot_by_condition", fake_boxplot)
        plotter.plot_single_predictor_metric(data_df=frame, metric_name=metric)

        return captured["title"]

    def test_an_unlabelled_metric_is_drawn_under_its_own_name(
        self, tmp_path, monkeypatch, frame
    ):
        roles = MetricRoles(
            outcome=OUTCOME_METRIC, components=(), predictors=("degradation_1",)
        )
        plotter = make_plotter(tmp_path, roles)

        title = self._captured_title(monkeypatch, plotter, frame, "degradation_1")

        assert "degradation_1" in title

    def test_a_labelled_metric_is_drawn_under_its_label(
        self, tmp_path, monkeypatch, frame
    ):
        roles = MetricRoles(
            outcome=OUTCOME_METRIC,
            components=(),
            predictors=("degradation_1",),
            labels=(("degradation_1", "Read-ahead"),),
        )
        plotter = make_plotter(tmp_path, roles)

        title = self._captured_title(monkeypatch, plotter, frame, "degradation_1")

        assert "Read-ahead" in title
        assert "degradation_1" not in title

    def test_the_file_name_still_uses_the_metric_not_the_label(
        self, tmp_path, monkeypatch, frame
    ):
        """A label is for people; the file name has to stay joinable to the data."""
        roles = MetricRoles(
            outcome=OUTCOME_METRIC,
            components=(),
            predictors=("degradation_1",),
            labels=(("degradation_1", "Read-ahead"),),
        )
        plotter = make_plotter(tmp_path, roles)

        captured = {}
        monkeypatch.setattr(plotter, "_boxplot_by_condition", lambda **kw: captured.update(kw))
        plotter.plot_single_predictor_metric(data_df=frame, metric_name="degradation_1")

        assert captured["img_fname"] == "degradation_1_boxplot.png"


class TestTheBinAxisFollowsTheOutcome:
    def test_the_default_outcome_names_the_axis(self, tmp_path):
        roles = MetricRoles(outcome=OUTCOME_METRIC, components=(), predictors=())

        assert make_plotter(tmp_path, roles).bin_axis_label == f"{OUTCOME_METRIC} bin"

    def test_another_outcome_names_the_axis_after_itself(self, tmp_path):
        roles = MetricRoles(outcome="yield", components=(), predictors=())

        plotter = make_plotter(tmp_path, roles)

        assert plotter.bin_axis_label == "yield bin"
        assert OUTCOME_METRIC not in plotter.bin_axis_label

    def test_the_label_helper_is_what_the_plotter_uses(self, tmp_path):
        # One place builds the wording; the plotter must not compose its own.
        roles = MetricRoles(outcome="yield", components=(), predictors=())

        assert make_plotter(tmp_path, roles).bin_axis_label == disp.bin_axis_label("yield")


class TestEmptyRolesAreSkippedNotCrashed:
    """A run may legitimately have no predictors, or no components."""

    def test_no_predictor_metrics_logs_and_returns(self, tmp_path, frame, caplog):
        roles = MetricRoles(outcome=OUTCOME_METRIC, components=(), predictors=())
        plotter = make_plotter(tmp_path, roles)

        with caplog.at_level("INFO"):
            plotter.plot_predictor_metrics(data_df=frame)

        assert "no predictor metrics" in caplog.text

    def test_no_component_metrics_logs_and_returns(self, tmp_path, frame, caplog):
        roles = MetricRoles(outcome=OUTCOME_METRIC, components=(), predictors=())
        plotter = make_plotter(tmp_path, roles)

        with caplog.at_level("INFO"):
            plotter.plot_component_metrics_by_outcome_bins(data_df=frame)

        assert "no component metrics" in caplog.text
