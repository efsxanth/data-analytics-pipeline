from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from ..constants import schema as sch


class ExtractSchemaError(KeyError):
    """The extract does not contain something the caller asked for."""


class ExtractSchemaVersionError(ValueError):
    """The extract is not written in a version of the format this build reads."""


def _decode(values) -> np.ndarray:
    """h5py may hand back bytes or str depending on version; normalise to str."""
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in np.atleast_1d(values)]
    )


class ExtractReader:
    """Read-only view of one extract file.

    The format version is checked on construction rather than lazily, so an
    unreadable file is rejected at the point it is opened -- before a caller has
    pulled arrays out of it and started building frames on them.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._check_schema_version()

    # -- identity ----------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return int(self._attr(sch.ATTR_SCHEMA_VERSION))

    @property
    def run_id(self) -> str:
        return str(self._attr(sch.ATTR_RUN_ID))

    @property
    def response_name(self) -> str:
        return str(self._attr(sch.ATTR_RESPONSE_NAME))

    @property
    def created_utc(self) -> str:
        return str(self._attr(sch.ATTR_CREATED_UTC))

    @property
    def n_observations(self) -> int:
        return int(self._dataset(sch.GROUP_OBSERVATIONS, sch.DS_OBSERVATION_ID).shape[0])

    @property
    def n_groups(self) -> int:
        return int(self._dataset(sch.GROUP_REFERENCE, sch.DS_REFERENCE_GROUP_ID).shape[0])

    @property
    def n_steps(self) -> int:
        return int(self._dataset(sch.GROUP_SIGNAL, self.response_name).shape[1])

    # -- per-observation ---------------------------------------------------

    def observation_ids(self) -> np.ndarray:
        return self._array(sch.GROUP_OBSERVATIONS, sch.DS_OBSERVATION_ID)

    def observation_group_ids(self) -> np.ndarray:
        """Group id per observation. Co-indexed with observation_ids()."""
        return self._array(sch.GROUP_OBSERVATIONS, sch.DS_OBSERVATION_GROUP_ID)

    # -- metrics -----------------------------------------------------------

    @cached_property
    def metric_names(self) -> tuple[str, ...]:
        """The metrics this file declares, in the column order of the matrix.

        A run's whole metric set is discovered from here. Nothing in the
        pipeline knows a metric name before this attribute has been read.
        """
        dataset = self._dataset(sch.GROUP_OBSERVATIONS, sch.DS_METRICS)

        if sch.ATTR_METRIC_NAMES not in dataset.attrs:
            raise ExtractSchemaError(
                f"{sch.GROUP_OBSERVATIONS}/{sch.DS_METRICS} carries no "
                f"{sch.ATTR_METRIC_NAMES!r} attribute, so its columns cannot be named. "
                f"Present: {sorted(dataset.attrs)}"
            )

        names = tuple(_decode(dataset.attrs[sch.ATTR_METRIC_NAMES]).tolist())

        # The matrix and its column names are two separate pieces of the file, and
        # nothing but this check keeps them in step. A mismatch would hand every
        # caller a metric silently bound to the wrong column.
        if len(names) != dataset.shape[1]:
            raise ExtractSchemaError(
                f"{sch.ATTR_METRIC_NAMES} names {len(names)} metric(s) but "
                f"{sch.GROUP_OBSERVATIONS}/{sch.DS_METRICS} has {dataset.shape[1]} "
                f"column(s); the two must agree."
            )

        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise ExtractSchemaError(
                f"{sch.ATTR_METRIC_NAMES} repeats {repeated}; a metric name must "
                f"identify one column."
            )

        return names

    def metrics(self, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """One array per requested metric, keyed by name.

        Args:
            names: Metrics to read. ``None`` reads every metric the file
                declares. A name the file does not declare is an error rather
                than a silent omission: callers resolve their selection against
                :attr:`metric_names` first, where an absent name is a warning.

        Returns:
            ``{metric name: (n_observations,) array}``, co-indexed with
            :meth:`observation_ids`.
        """
        declared = self.metric_names
        wanted = tuple(declared) if names is None else tuple(names)

        unknown = [name for name in wanted if name not in declared]
        if unknown:
            raise ExtractSchemaError(
                f"Metric(s) {unknown} are not in this extract. Declared: {list(declared)}"
            )

        dataset = self._dataset(sch.GROUP_OBSERVATIONS, sch.DS_METRICS)
        position = {name: index for index, name in enumerate(declared)}

        # One column read per metric rather than one read of the whole matrix, so
        # a run that analyses three of a dozen metrics pays for three columns.
        return {name: np.asarray(dataset[:, position[name]]) for name in wanted}

    # -- signal ------------------------------------------------------------

    def signal(self) -> np.ndarray:
        """Raw values for this file's response, (n_observations, n_steps)."""
        return self._array(sch.GROUP_SIGNAL, self.response_name)

    # -- reference ---------------------------------------------------------

    def reference_group_ids(self) -> np.ndarray:
        """Group ids. Defines the row order of everything in reference/."""
        return self._array(sch.GROUP_REFERENCE, sch.DS_REFERENCE_GROUP_ID)

    def reference_levels(self) -> np.ndarray:
        """(n_groups, n_steps), row order = reference_group_ids()."""
        return self._array(sch.GROUP_REFERENCE, sch.DS_REFERENCE_LEVELS)

    def reference_levels_dict(self) -> dict:
        """{group_id: reference levels}, ready to use as reference_levels_dict."""
        return dict(zip(self.reference_group_ids().tolist(), self.reference_levels()))

    # -- escape hatch ------------------------------------------------------

    def dataset(self, group: str, name: str) -> h5py.Dataset:
        """The live h5py dataset, for partial reads. Valid only while open."""
        return self._dataset(group, name)

    def names(self, group: str) -> list[str]:
        return list(self._group(group))

    # -- internals ---------------------------------------------------------

    def _check_schema_version(self) -> None:
        """Reject anything not written in a version of the format we read.

        A file predating the attribute is rejected by the same path as one from
        a future build: in both cases the layout behind the handle is unknown,
        and guessing at it is how a rename becomes a silently wrong column.
        """
        supported = ', '.join(str(version) for version in sch.SUPPORTED_SCHEMA_VERSIONS)

        if sch.ATTR_SCHEMA_VERSION not in self._file.attrs:
            raise ExtractSchemaVersionError(
                f"This file declares no {sch.ATTR_SCHEMA_VERSION!r} attribute, so it is "
                f"not an extract in this format. Supported version(s): {supported}."
            )

        version = int(self._file.attrs[sch.ATTR_SCHEMA_VERSION])

        if version not in sch.SUPPORTED_SCHEMA_VERSIONS:
            raise ExtractSchemaVersionError(
                f"Extract schema version {version} is not supported by this build. "
                f"Supported version(s): {supported}."
            )

    def _attr(self, name: str):
        if name not in self._file.attrs:
            raise ExtractSchemaError(
                f"Root attribute {name!r} is missing. "
                f"Present: {sorted(self._file.attrs)}"
            )
        return self._file.attrs[name]

    def _group(self, name: str) -> h5py.Group:
        if name not in self._file:
            raise ExtractSchemaError(
                f"Group {name!r} is missing. Present: {sorted(self._file)}"
            )
        return self._file[name]

    def _dataset(self, group_name: str, name: str) -> h5py.Dataset:
        group = self._group(group_name)
        if name not in group:
            raise ExtractSchemaError(
                f"{name!r} is not in {group_name!r}. Available: {sorted(group)}"
            )
        return group[name]

    def _array(self, group_name: str, name: str) -> np.ndarray:
        # [:] materialises now, so the array outlives the file handle.
        return np.asarray(self._dataset(group_name, name)[:])


@contextlib.contextmanager
def open_extract(path: str | Path) -> Iterator[ExtractReader]:
    """Open an extract file read-only and yield an ExtractReader."""
    with h5py.File(str(path), "r") as handle:
        yield ExtractReader(handle)
