"""Cross-file validation.

The pipeline pools observations from many files into one comparison. If two
files disagree about which group an ID refers to, or about which metrics they
carry, the comparison is meaningless but still produces numbers -- so these
checks are the pipeline's most important safety property, and the reason the run
stops rather than warns.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from da_pipeline.synthetic import build_group_panel, metric_names, write_extract
from da_pipeline.validation import DataValidator, default_group_label

from .conftest import N_GROUPS, N_STEPS


@pytest.fixture
def validator(dataset_root):
    validator = DataValidator()
    validator.validate_config_dict(config_fpath=dataset_root)
    return validator


class TestHappyPath:
    def test_consistent_dataset_validates(self, validator):
        validator.validate_group_ids()
        validator.validate_reference_levels()
        validator.validate_metric_names()

        assert validator.group_ids.size == N_GROUPS
        assert len(validator.reference_levels) == N_GROUPS
        assert validator.metric_names == metric_names()

    def test_label_dict_is_sorted_by_group_id(self, validator):
        validator.validate_group_ids()
        validator.validate_reference_levels()

        labels = validator.get_group_label_dict()

        assert list(labels) == sorted(labels)

    def test_labels_are_derived_from_the_group_ids(self, validator):
        # The format carries no labels, so every run gets them the same way and
        # two files can never disagree about what a group is called.
        validator.validate_group_ids()
        validator.validate_reference_levels()

        labels = validator.get_group_label_dict()

        assert labels == {
            gid: default_group_label(gid) for gid in validator.group_ids
        }

    def test_user_labels_override_the_derived_ones(self, validator):
        validator.validate_group_ids()
        validator.validate_reference_levels()

        labels = validator.get_group_label_dict(
            user_provided_group_labels=[f"P{i}" for i in range(N_GROUPS)]
        )

        assert set(labels.values()) == {f"P{i}" for i in range(N_GROUPS)}

    def test_wrong_number_of_user_labels_is_rejected(self, validator):
        validator.validate_group_ids()
        validator.validate_reference_levels()

        with pytest.raises(ValueError, match="does not match"):
            validator.get_group_label_dict(user_provided_group_labels=["only_one"])

    def test_reference_levels_cover_every_validated_group(self, validator):
        validator.validate_group_ids()
        validator.validate_reference_levels()

        reference_levels = validator.get_reference_levels_dict()

        assert set(reference_levels) == set(validator.group_ids.tolist())
        assert next(iter(reference_levels.values())).shape == (N_STEPS,)


class TestGroupSubset:
    def test_requested_subset_is_kept_in_the_requested_order(self, validator):
        validator.validate_group_ids(user_provided_group_ids=[3, 1])

        assert validator.group_ids.tolist() == [3, 1]

    def test_missing_group_names_the_offending_file(self, validator):
        with pytest.raises(ValueError, match="not present"):
            validator.validate_group_ids(user_provided_group_ids=[9999])

    def test_subset_restricts_the_reference_levels(self, validator):
        validator.validate_group_ids(user_provided_group_ids=[2])
        validator.validate_reference_levels(user_provided_group_ids=[2])

        assert set(validator.get_reference_levels_dict()) == {2}


class TestMetricDiscovery:
    def test_the_declared_metric_set_is_recorded(self, validator):
        validator.validate_metric_names()

        assert validator.get_declared_metric_names() == metric_names()

    def test_the_metric_set_is_whatever_the_files_declare(self, tmp_path):
        # Nothing about the count is assumed: a dataset with a different number
        # of metrics validates just as happily.
        panel = build_group_panel(n_groups=2, n_steps=N_STEPS, seed=1)
        paths = [
            str(write_extract(tmp_path / f"run{i}.h5", panel=panel, run_id=f"A.{i}",
                              observations_per_group=5, n_degradation_metrics=6, seed=i))
            for i in range(2)
        ]
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"Condition_A": paths}), encoding="utf-8")

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)
        validator.validate_metric_names()

        assert validator.metric_names == metric_names(n_degradation_metrics=6)

    def test_metrics_must_be_read_before_they_can_be_asked_for(self, validator):
        with pytest.raises(Exception, match="validate_metric_names"):
            validator.get_declared_metric_names()


class TestOrderingGuards:
    def test_group_ids_must_be_validated_before_reference_levels(self, validator):
        with pytest.raises(Exception, match="validate_group_ids"):
            validator.validate_reference_levels()

    def test_reference_levels_must_be_validated_before_labels(self, validator):
        validator.validate_group_ids()

        with pytest.raises(Exception, match="validate_reference_levels"):
            validator.get_group_label_dict()

    def test_config_must_be_loaded_first(self):
        with pytest.raises(Exception, match="validate_config_dict"):
            DataValidator().validate_group_ids()


class TestInconsistentFiles:
    """A second file whose panel disagrees must stop the run."""

    def _dataset_with_odd_file(self, tmp_path, odd_panel, **odd_kwargs):
        good = build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)

        first = write_extract(tmp_path / "good.h5", panel=good, run_id="A.0",
                              observations_per_group=5, seed=1)
        second = write_extract(tmp_path / "odd.h5", panel=odd_panel, run_id="A.1",
                               observations_per_group=5, seed=2, **odd_kwargs)

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"Condition_A": [str(first), str(second)]}), encoding="utf-8"
        )
        return config_path

    def test_different_group_ids_are_rejected(self, tmp_path):
        odd = build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)
        odd = type(odd)(
            group_ids=odd.group_ids + 100,
            reference_levels=odd.reference_levels,
        )

        config_path = self._dataset_with_odd_file(tmp_path, odd)

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)

        with pytest.raises(Exception, match="inconsistency"):
            validator.validate_group_ids()

    def test_different_reference_levels_are_rejected(self, tmp_path):
        # Same group IDs, but one file disagrees about what the correct result
        # is. The outcome metric is scored against that reference, so comparing
        # the two files would be comparing answers to two different questions.
        odd = build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)
        mutated = np.array(odd.reference_levels, copy=True)
        mutated[0] = (mutated[0] + 1) % 3
        odd = type(odd)(
            group_ids=odd.group_ids,
            reference_levels=mutated,
        )

        config_path = self._dataset_with_odd_file(tmp_path, odd)

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)
        validator.validate_group_ids()

        with pytest.raises(Exception, match="Reference-level inconsistency"):
            validator.validate_reference_levels()

    def test_different_metric_sets_are_rejected(self, tmp_path):
        # Pooling observations from files carrying different metrics would build
        # a frame whose columns mean different things for different rows.
        good = build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)

        config_path = self._dataset_with_odd_file(
            tmp_path, good, n_degradation_metrics=5
        )

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)

        with pytest.raises(Exception, match="Metric inconsistency"):
            validator.validate_metric_names()

    def test_the_metric_mismatch_names_both_files_and_the_difference(self, tmp_path):
        good = build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)

        config_path = self._dataset_with_odd_file(
            tmp_path, good, n_degradation_metrics=5
        )

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=config_path)

        with pytest.raises(Exception) as excinfo:
            validator.validate_metric_names()

        message = str(excinfo.value)
        assert "good.h5" in message and "odd.h5" in message
        assert "degradation_4" in message
