"""CSV output: where files land, and which ones are written at all."""

from __future__ import annotations

import polars as pl
import pytest

from da_pipeline.config import PipelineConfig
from da_pipeline.constants import columns as col
from da_pipeline.constants import outputs as out
from da_pipeline.io.writers import DataSaver, write_csv
from da_pipeline.metrics import MetricRoles

from .conftest import OUTCOME_METRIC

# Arbitrary on purpose: the saver is told which metrics it is writing.
PREDICTORS = ("degradation_1", "degradation_2", "degradation_3")


def make_saver(tmp_path, roles):
    config = PipelineConfig(
        config_path=tmp_path / "runs.json", response_name="auc"
    ).with_output_dir(tmp_path / "Data_Analytics")
    return DataSaver(config=config, roles=roles)


@pytest.fixture
def saver(tmp_path):
    roles = MetricRoles(
        outcome=OUTCOME_METRIC, components=("Accuracy_error_1",), predictors=PREDICTORS
    )
    return make_saver(tmp_path, roles)


@pytest.fixture
def frame():
    return pl.DataFrame({col.CONDITION: ["A", "B"], "value": [1.0, 2.0]})


class TestWriteCsv:
    def test_creates_missing_directories(self, tmp_path, frame):
        group = tmp_path / "deep" / "nested"

        path = write_csv(frame, str(group), "out.csv")

        assert (group / "out.csv").exists()
        assert path.endswith("out.csv")

    def test_round_trips_the_frame(self, tmp_path, frame):
        write_csv(frame, str(tmp_path), "out.csv")

        assert pl.read_csv(tmp_path / "out.csv").equals(frame)


class TestLayout:
    def test_datasets_go_under_csv_exports(self, saver, frame):
        saver.save_responses_csv(frame)

        expected = (
            saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR / out.RESPONSES_CSV_FNAME
        )
        assert expected.exists()

    def test_correlations_get_their_own_subdirectory(self, saver, frame):
        saver.save_correlations_csv(frame)

        expected = (
            saver.config.resolved_output_dir
            / out.CSV_EXPORTS_SUBDIR
            / out.CORRELATIONS_SUBDIR
            / out.CORRELATIONS_CSV_FNAME
        )
        assert expected.exists()

    def test_effect_size_sits_beside_the_other_datasets(self, saver, frame):
        saver.save_effect_size_csv(frame)

        expected = (
            saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR / out.EFFECT_SIZE_CSV_FNAME
        )
        assert expected.exists()

    def test_within_run_correlations_sit_with_the_pooled_ones(self, saver, frame):
        saver.save_within_run_correlations_csv(frame)

        expected = (
            saver.config.resolved_output_dir
            / out.CSV_EXPORTS_SUBDIR
            / out.CORRELATIONS_SUBDIR
            / out.WITHIN_RUN_CORRELATIONS_CSV_FNAME
        )
        assert expected.exists()

    def test_bin_summaries_get_their_own_subdirectory(self, saver, frame):
        saver.save_snr_bin_summary(frame)

        expected = (
            saver.config.resolved_output_dir
            / out.CSV_EXPORTS_SUBDIR
            / out.BIN_SUMMARY_SUBDIR
            / out.snr_bin_summary_fname(OUTCOME_METRIC)
        )
        assert expected.exists()


class TestBinSummaryFileNamesFollowTheOutcome:
    """A fixed word in the file name would name the wrong quantity."""

    def test_the_snr_summary_is_named_after_the_configured_outcome(self, tmp_path, frame):
        roles = MetricRoles(outcome="yield", components=(), predictors=PREDICTORS)
        saver = make_saver(tmp_path, roles)

        saver.save_snr_bin_summary(frame)

        bin_dir = (
            saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR / out.BIN_SUMMARY_SUBDIR
        )
        assert (bin_dir / "snr_and_yield_bin_summary.csv").exists()
        assert not (bin_dir / "snr_and_Accuracy_bin_summary.csv").exists()

    def test_per_metric_summaries_are_named_after_the_configured_outcome(self, tmp_path):
        roles = MetricRoles(outcome="yield", components=(), predictors=("pressure",))
        saver = make_saver(tmp_path, roles)

        summary = pl.DataFrame({
            col.CONDITION: ["A"],
            col.OUTCOME_BIN_INDEX: [1],
            "yield_median": [0.9],
            "pressure_median": [0.05],
            col.N_POINTS_BIN: [10],
        })

        saver.save_predictor_bin_summaries(summary, ["pressure"])

        bin_dir = (
            saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR / out.BIN_SUMMARY_SUBDIR
        )
        assert (bin_dir / "pressure_and_yield_bin_summary.csv").exists()


class TestEmptyFrames:
    def test_empty_frame_writes_nothing(self, saver):
        saver.save_responses_csv(pl.DataFrame({col.CONDITION: []}))

        assert not (saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR).exists()

    def test_none_writes_nothing(self, saver):
        saver.save_responses_csv(None)

        assert not (saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR).exists()


class TestPerMetricBinSummaries:
    @pytest.fixture
    def predictor_summary(self):
        """A summary frame carrying stat columns for every predictor metric."""
        data = {
            col.CONDITION: ["A"],
            col.OUTCOME_BIN_INDEX: [1],
            f"{OUTCOME_METRIC}_median": [0.9],
            col.N_POINTS_BIN: [10],
        }
        for metric in PREDICTORS:
            data[f"{metric}_median"] = [0.05]
            data[f"{metric}_q25"] = [0.04]
        return pl.DataFrame(data)

    def _bin_dir(self, saver):
        return (
            saver.config.resolved_output_dir / out.CSV_EXPORTS_SUBDIR / out.BIN_SUMMARY_SUBDIR
        )

    def test_one_file_per_metric(self, saver, predictor_summary):
        saver.save_predictor_bin_summaries(predictor_summary, list(PREDICTORS))

        bin_dir = self._bin_dir(saver)
        for metric in PREDICTORS:
            assert (bin_dir / out.bin_summary_fname(metric, OUTCOME_METRIC)).exists()

    def test_however_many_metrics_the_run_has(self, saver):
        # The metric set is open-ended, so the saver must not assume a count.
        many = [f"degradation_{i}" for i in range(1, 8)]
        data = {
            col.CONDITION: ["A"],
            col.OUTCOME_BIN_INDEX: [1],
            f"{OUTCOME_METRIC}_median": [0.9],
            col.N_POINTS_BIN: [10],
        }
        for metric in many:
            data[f"{metric}_median"] = [0.05]

        saver.save_predictor_bin_summaries(pl.DataFrame(data), many)

        bin_dir = self._bin_dir(saver)
        written = {path.name for path in bin_dir.glob("*.csv")}
        assert len(written) == len(many)

    def test_each_file_keeps_only_its_own_metric(self, saver, predictor_summary):
        saver.save_predictor_bin_summaries(predictor_summary, list(PREDICTORS))

        written = pl.read_csv(
            self._bin_dir(saver) / out.bin_summary_fname("degradation_1", OUTCOME_METRIC)
        )

        assert "degradation_1_median" in written.columns
        assert "degradation_2_median" not in written.columns
        assert "degradation_3_median" not in written.columns

    def test_shared_context_columns_survive_in_every_file(self, saver, predictor_summary):
        saver.save_predictor_bin_summaries(predictor_summary, list(PREDICTORS))

        written = pl.read_csv(
            self._bin_dir(saver) / out.bin_summary_fname("degradation_2", OUTCOME_METRIC)
        )

        # The outcome, the grouping keys and the point count are the context
        # every file needs to be readable on its own.
        assert f"{OUTCOME_METRIC}_median" in written.columns
        assert col.CONDITION in written.columns
        assert col.N_POINTS_BIN in written.columns
