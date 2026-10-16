"""Reading one extract file.

These tests run against real HDF5 files written by the reference producer, so
they fail if the reader and the schema module ever disagree.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from da_pipeline.constants import schema as sch
from da_pipeline.io.extract_reader import (
    ExtractSchemaError,
    ExtractSchemaVersionError,
    open_extract,
)

from .conftest import N_GROUPS, N_STEPS, OBSERVATIONS_PER_GROUP


def write_minimal(path, *, version=sch.SCHEMA_VERSION, metrics=None, metric_names=None):
    """A file carrying only what a given test needs, for the failure modes."""
    with h5py.File(path, "w") as handle:
        if version is not None:
            handle.attrs[sch.ATTR_SCHEMA_VERSION] = version
        handle.attrs[sch.ATTR_RUN_ID] = "run"

        if metrics is not None:
            observations = handle.create_group(sch.GROUP_OBSERVATIONS)
            dataset = observations.create_dataset(sch.DS_METRICS, data=metrics)
            if metric_names is not None:
                dataset.attrs[sch.ATTR_METRIC_NAMES] = np.array(
                    metric_names, dtype=h5py.string_dtype()
                )

    return path


class TestIdentity:
    def test_reads_root_attributes(self, extract_path):
        with open_extract(extract_path) as extract:
            assert extract.schema_version == sch.SCHEMA_VERSION
            assert extract.run_id == "Condition_A.0"
            assert extract.response_name == "auc"
            assert extract.created_utc

    def test_reports_shape(self, extract_path):
        with open_extract(extract_path) as extract:
            assert extract.n_groups == N_GROUPS
            assert extract.n_observations == N_GROUPS * OBSERVATIONS_PER_GROUP
            assert extract.n_steps == N_STEPS


class TestReferenceTable:
    def test_reference_rows_are_co_indexed_with_group_ids(self, extract_path):
        with open_extract(extract_path) as extract:
            n = extract.reference_group_ids().size
            assert extract.reference_levels().shape[0] == n
            assert n == extract.n_groups

    def test_levels_dict_maps_group_id_to_its_row(self, extract_path):
        with open_extract(extract_path) as extract:
            levels = extract.reference_levels_dict()
            first = extract.reference_group_ids()[0]

            assert np.array_equal(levels[first], extract.reference_levels()[0])


class TestPerObservationData:
    def test_observation_arrays_are_co_indexed(self, extract_path):
        # Everything downstream joins on this identity pair.
        with open_extract(extract_path) as extract:
            assert extract.observation_ids().size == extract.observation_group_ids().size

    def test_signal_is_observations_by_steps(self, extract_path):
        with open_extract(extract_path) as extract:
            assert extract.signal().shape == (extract.n_observations, extract.n_steps)


class TestMetricDiscovery:
    """The metric set is data, not code: it is read out of the file."""

    def test_metric_names_come_from_the_matrix_attribute(self, extract_path):
        with open_extract(extract_path) as extract:
            names = extract.metric_names
            width = extract.dataset(sch.GROUP_OBSERVATIONS, sch.DS_METRICS).shape[1]

        assert len(names) == width
        assert all(isinstance(name, str) for name in names)

    def test_reading_everything_returns_one_array_per_declared_metric(self, extract_path):
        with open_extract(extract_path) as extract:
            metrics = extract.metrics()

            assert set(metrics) == set(extract.metric_names)
            assert all(
                values.shape == (extract.n_observations,) for values in metrics.values()
            )

    def test_a_subset_reads_the_same_values_as_the_whole(self, extract_path):
        with open_extract(extract_path) as extract:
            wanted = extract.metric_names[1]
            everything = extract.metrics()
            subset = extract.metrics([wanted])

        assert list(subset) == [wanted]
        assert np.array_equal(subset[wanted], everything[wanted])

    def test_an_undeclared_metric_is_an_error_not_an_omission(self, extract_path):
        # Callers resolve their selection against metric_names first, where an
        # absent name is a warning. Reaching this far means a real mistake.
        with open_extract(extract_path) as extract, pytest.raises(
            ExtractSchemaError, match="not in this extract"
        ):
            extract.metrics(["no_such_metric"])


class TestSchemaVersion:
    def test_a_file_without_a_version_is_rejected_on_open(self, tmp_path):
        path = write_minimal(tmp_path / "unversioned.h5", version=None)

        with pytest.raises(ExtractSchemaVersionError, match="declares no"), open_extract(path):
            pass

    def test_an_unknown_version_names_both_it_and_what_is_supported(self, tmp_path):
        path = write_minimal(tmp_path / "future.h5", version=99)

        with pytest.raises(ExtractSchemaVersionError, match="version 99 is not supported"), open_extract(path):
            pass

    def test_the_supported_version_is_the_one_the_producer_writes(self, extract_path):
        # Two constants that must not drift apart: the version written and the
        # set accepted. A producer that writes something the reader rejects
        # would only show up as an unreadable dataset.
        assert sch.SCHEMA_VERSION in sch.SUPPORTED_SCHEMA_VERSIONS

        with open_extract(extract_path) as extract:
            assert extract.schema_version in sch.SUPPORTED_SCHEMA_VERSIONS


class TestFailureModes:
    def test_metric_names_must_match_the_matrix_width(self, tmp_path):
        # Two separate pieces of the file; a mismatch would bind a metric to the
        # wrong column rather than fail, which is why it is checked.
        path = write_minimal(
            tmp_path / "skew.h5",
            metrics=np.zeros((4, 3), dtype=np.float32),
            metric_names=["a", "b"],
        )

        with open_extract(path) as extract, pytest.raises(ExtractSchemaError, match="must agree"):
            _ = extract.metric_names

    def test_a_repeated_metric_name_is_rejected(self, tmp_path):
        path = write_minimal(
            tmp_path / "repeat.h5",
            metrics=np.zeros((4, 2), dtype=np.float32),
            metric_names=["a", "a"],
        )

        with open_extract(path) as extract, pytest.raises(ExtractSchemaError, match="repeats"):
            _ = extract.metric_names

    def test_a_matrix_without_names_says_so(self, tmp_path):
        path = write_minimal(
            tmp_path / "nameless.h5", metrics=np.zeros((4, 2), dtype=np.float32)
        )

        with open_extract(path) as extract, pytest.raises(
            ExtractSchemaError, match="cannot be named"
        ):
            _ = extract.metric_names

    def test_missing_dataset_lists_what_is_available(self, tmp_path):
        # The reader names the group it looked in and what it found there, so a
        # malformed file points at itself rather than at a bare KeyError.
        path = write_minimal(tmp_path / "no_metrics.h5")
        with h5py.File(path, "a") as handle:
            handle.create_group(sch.GROUP_OBSERVATIONS)

        with open_extract(path) as extract, pytest.raises(ExtractSchemaError, match="Available"):
            extract.observation_ids()

    def test_missing_group_lists_what_is_present(self, tmp_path):
        path = write_minimal(tmp_path / "empty.h5")

        with open_extract(path) as extract, pytest.raises(ExtractSchemaError, match="is missing"):
            extract.observation_ids()

    def test_missing_root_attribute_lists_what_is_present(self, tmp_path):
        path = write_minimal(tmp_path / "bare.h5")

        with open_extract(path) as extract, pytest.raises(ExtractSchemaError, match="Present"):
            _ = extract.response_name

    def test_arrays_outlive_the_file_handle(self, extract_path):
        # The reader materialises with [:] so callers can close and keep the data.
        with open_extract(extract_path) as extract:
            ids = extract.observation_ids()

        assert ids.sum() >= 0
