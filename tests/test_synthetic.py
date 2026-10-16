"""The reference producer and the demo command.

The generator is test infrastructure, but it is also what makes the repository
runnable by a stranger and the only writer of this project's format, so its
invariants are worth asserting: the files it writes must satisfy the same
contract the validator enforces, and the quality gradient it plants must be real.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from da_pipeline.demo import main as demo_main
from da_pipeline.io.extract_reader import open_extract
from da_pipeline.synthetic import (
    ConditionSpec,
    build_group_panel,
    generate_dataset,
    metric_names,
    write_extract,
)
from da_pipeline.validation import DataValidator

SPECS = (
    ConditionSpec("Condition_A", n_runs=2, quality=0.9),
    ConditionSpec("Condition_B", n_runs=1, quality=0.4),
)


class TestGroupPanel:
    def test_panel_pieces_are_co_indexed(self):
        panel = build_group_panel(n_groups=5, n_steps=20, seed=0)

        assert panel.group_ids.size == 5
        assert panel.reference_levels.shape == (5, 20)

    def test_every_group_carries_both_snr_levels_early(self):
        # Signal-to-noise is one level's median over another's, so a group missing
        # either is silently dropped from the SNR frame.
        panel = build_group_panel(n_groups=4, n_steps=20, seed=3)

        assert (panel.reference_levels[:, 0] == 0).all()
        assert (panel.reference_levels[:, 1] == 1).all()

    def test_the_guaranteed_levels_follow_the_configured_ones(self):
        # A run comparing level 2 against level 0 needs a panel that has them.
        panel = build_group_panel(
            n_groups=4, n_steps=20, seed=3, baseline_level=0, comparison_level=2
        )

        assert (panel.reference_levels[:, 0] == 0).all()
        assert (panel.reference_levels[:, 1] == 2).all()

    def test_generation_is_deterministic(self):
        first = build_group_panel(n_groups=4, n_steps=20, seed=5)
        second = build_group_panel(n_groups=4, n_steps=20, seed=5)

        assert np.array_equal(first.reference_levels, second.reference_levels)
        assert np.array_equal(first.group_ids, second.group_ids)

    def test_different_seeds_give_different_panels(self):
        first = build_group_panel(n_groups=4, n_steps=20, seed=5)
        second = build_group_panel(n_groups=4, n_steps=20, seed=6)

        assert not np.array_equal(first.reference_levels, second.reference_levels)


class TestWrittenExtract:
    @pytest.fixture
    def extract(self, tmp_path):
        panel = build_group_panel(n_groups=3, n_steps=20, seed=0)
        return write_extract(
            tmp_path / "run.h5", panel=panel, run_id="A.0",
            observations_per_group=10, seed=0,
        )

    def test_observation_arrays_are_co_indexed(self, extract):
        with open_extract(extract) as reader:
            assert reader.observation_ids().size == reader.observation_group_ids().size
            assert reader.signal().shape[0] == reader.n_observations

    def test_the_file_declares_its_own_metric_set(self, extract):
        with open_extract(extract) as reader:
            assert reader.metric_names == metric_names()

    def test_metrics_are_in_their_natural_ranges(self, extract):
        with open_extract(extract) as reader:
            metrics = reader.metrics()

        accuracy = metrics["Accuracy"]
        assert ((accuracy >= 0) & (accuracy <= 1)).all()
        assert (metrics["degradation_1"] >= 0).all()

    def test_signal_rises_with_level(self, extract):
        # The whole point of the synthetic signal: a level 1 reads higher than a
        # level 0, which is what makes signal-to-noise meaningful.
        with open_extract(extract) as reader:
            signal = reader.signal()
            reference_levels = reader.reference_levels()
            observation_group_ids = reader.observation_group_ids()
            group_ids = reader.reference_group_ids()

        per_observation = reference_levels[
            np.searchsorted(group_ids, observation_group_ids)
        ]

        assert signal[per_observation == 1].mean() > signal[per_observation == 0].mean()

    def test_the_file_is_reproducible_apart_from_when_it_was_written(self, tmp_path):
        panel = build_group_panel(n_groups=3, n_steps=20, seed=0)
        kwargs = {
            "panel": panel, "run_id": "A.0", "observations_per_group": 10, "seed": 0,
            "created_utc": "2026-01-01T00:00:00+00:00",
        }

        first = write_extract(tmp_path / "a.h5", **kwargs)
        second = write_extract(tmp_path / "b.h5", **kwargs)

        assert first.read_bytes() == second.read_bytes()


class TestArbitraryMetricCounts:
    """The metric set is a parameter of the producer, not a fact of the format."""

    @pytest.mark.parametrize("n_degradation_metrics", [0, 1, 3, 7])
    def test_the_requested_number_of_metrics_is_planted(self, tmp_path, n_degradation_metrics):
        panel = build_group_panel(n_groups=2, n_steps=20, seed=0)
        path = write_extract(
            tmp_path / f"run_{n_degradation_metrics}.h5",
            panel=panel, run_id="A.0", observations_per_group=10,
            n_degradation_metrics=n_degradation_metrics, seed=0,
        )

        with open_extract(path) as reader:
            names = reader.metric_names
            metrics = reader.metrics()
            n_observations = reader.n_observations

        planted = [name for name in names if name.startswith("degradation_")]

        assert len(planted) == n_degradation_metrics
        assert names == metric_names(n_degradation_metrics=n_degradation_metrics)
        assert all(metrics[name].shape == (n_observations,) for name in names)

    def test_the_metrics_are_not_copies_of_one_another(self, tmp_path):
        # A figure comparing them has to show distinguishable distributions,
        # however many were asked for.
        panel = build_group_panel(n_groups=2, n_steps=20, seed=0)
        path = write_extract(
            tmp_path / "many.h5", panel=panel, run_id="A.0",
            observations_per_group=200, n_degradation_metrics=4, seed=0,
        )

        with open_extract(path) as reader:
            metrics = reader.metrics()

        medians = [
            float(np.median(metrics[f"degradation_{i}"])) for i in range(1, 5)
        ]

        assert len(set(medians)) == len(medians)
        # Tapered, so each is smaller than the one before it.
        assert medians == sorted(medians, reverse=True)

    def test_a_negative_count_is_refused(self, tmp_path):
        panel = build_group_panel(n_groups=2, n_steps=20, seed=0)

        with pytest.raises(ValueError, match="cannot be negative"):
            write_extract(
                tmp_path / "bad.h5", panel=panel, run_id="A.0",
                observations_per_group=10, n_degradation_metrics=-1, seed=0,
            )


class TestGenerateDataset:
    @pytest.fixture
    def config_path(self, tmp_path):
        return generate_dataset(
            tmp_path, conditions=SPECS, n_groups=3, n_steps=20,
            observations_per_group=10, seed=0,
        )

    def test_config_lists_every_condition_and_run(self, config_path):
        config = json.loads(config_path.read_text(encoding="utf-8"))

        assert list(config) == ["Condition_A", "Condition_B"]
        assert len(config["Condition_A"]) == 2
        assert len(config["Condition_B"]) == 1

    def test_every_listed_file_is_readable(self, config_path):
        config = json.loads(config_path.read_text(encoding="utf-8"))

        for condition, paths in config.items():
            for path in paths:
                with open_extract(path) as reader:
                    assert reader.run_id.startswith(condition)
                    assert reader.response_name == "auc"

    def test_every_file_declares_the_same_metric_set(self, config_path):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        declared = set()

        for paths in config.values():
            for path in paths:
                with open_extract(path) as reader:
                    declared.add(reader.metric_names)

        assert len(declared) == 1

    def test_the_dataset_passes_the_real_validator(self, config_path):
        # The generator's most important property: files that disagree about the
        # group panel would be rejected, so this proves it writes a coherent set.
        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)
        validator.validate_group_ids()
        validator.validate_reference_levels()
        validator.validate_metric_names()

        assert validator.group_ids.size == 3

    def test_quality_gradient_is_real(self, config_path):
        config = json.loads(config_path.read_text(encoding="utf-8"))

        def median_accuracy(path):
            with open_extract(path) as reader:
                return float(np.median(reader.metrics(["Accuracy"])["Accuracy"]))

        good = median_accuracy(config["Condition_A"][0])
        poor = median_accuracy(config["Condition_B"][0])

        assert good > poor


class TestDemoCommand:
    def test_generate_only_writes_data_without_running(self, tmp_path, capsys):
        demo_main([
            "--output_dir", str(tmp_path / "demo"),
            "--observations_per_group", "5",
            "--generate_only",
        ])

        out = capsys.readouterr().out

        assert "Wrote 10 extract files" in out
        assert (tmp_path / "demo" / "demo_config.json").exists()
        # Nothing was analysed, so no results directory exists yet.
        assert not (tmp_path / "demo" / "Data_Analytics").exists()

    def test_the_degradation_metric_count_reaches_the_files(self, tmp_path, capsys):
        demo_main([
            "--output_dir", str(tmp_path / "demo"),
            "--observations_per_group", "5",
            "--n_degradation_metrics", "5",
            "--generate_only",
        ])

        config = json.loads(
            (tmp_path / "demo" / "demo_config.json").read_text(encoding="utf-8")
        )

        with open_extract(config["Condition_A"][0]) as reader:
            planted = [n for n in reader.metric_names if n.startswith("degradation_")]

        assert len(planted) == 5
