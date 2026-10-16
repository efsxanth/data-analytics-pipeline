import logging
import os

import numpy as np

from .io.extract_reader import open_extract
from .io.run_config import load_json_containing_paths

logger = logging.getLogger(__name__)


def default_group_label(group_id) -> str:
    """The display label a group gets when configuration supplies none.

    The extract format carries no labels: a group is identified by its id, and
    what to call it on a figure is a presentation question the data has no
    opinion about. Deriving the default here means every run has labels without
    two files ever being able to disagree about them.
    """
    return f"group_{group_id}"


class DataValidator:
    """
    Loads a configuration mapping conditions to extract-file paths, then checks
    that group IDs, reference levels and declared metrics agree across every file
    listed, with optional filtering to a subset of group IDs.
    """

    def __init__(self):

        self.config_dict = None
        self.reference_levels = None
        self.group_ids = None
        self.group_ids_original = {}
        self.metric_names = None

        self.first_fpath = None

        # Auxiliary variables indicating the status of an operation.
        self._config_validated = False
        self._group_ids_verified = False
        self._reference_levels_verified = False
        self._metric_names_verified = False

    def __check_if_config_validated(self):
        if not self._config_validated:
            raise Exception("[ERROR] You must first run 'validate_config_dict()'.")

    def __check_if_group_ids_verified(self):
        if not self._group_ids_verified:
            raise Exception("[ERROR] You must first run 'validate_group_ids()'.")

    def __check_if_reference_levels_verified(self):
        if not self._reference_levels_verified:
            raise Exception("[ERROR] You must first run 'validate_reference_levels()'.")

    def __check_if_metric_names_verified(self):
        if not self._metric_names_verified:
            raise Exception("[ERROR] You must first run 'validate_metric_names()'.")

    def validate_config_dict(self, config_fpath: str) -> None:
        """
        Load and validate the JSON configuration file mapping conditions to file paths.

        Args:
            config_fpath (str): Path to the configuration JSON file.

        Side Effects:
            - Populates `self.config_dict` with the loaded mapping.
            - Sets `self.first_fpath` to the first path in the mapping.
            - Marks configuration as validated.
        """

        self.config_dict = load_json_containing_paths(config_fpath=config_fpath)
        self.first_fpath = next(iter(self.config_dict.values()))[0]
        self._config_validated = True

    def _iter_paths(self):
        """Every extract path in the config, condition by condition."""
        for condition_fpaths in self.config_dict.values():
            yield from condition_fpaths

    def get_consistent_group_ids(self, user_provided_group_ids: list) -> np.ndarray:
        """
        Retrieve and verify consistent group IDs across all files.

        Args:
            user_provided_group_ids (list): If provided, filter to these IDs
                and ensure they exist in each file.

        Returns:
            np.ndarray: Array of consistent group IDs.
        """

        reference_group_ids = None

        for fpath in self._iter_paths():

            with open_extract(fpath) as extract:
                group_ids = extract.reference_group_ids()
                # the key (below) is not a run id, as there are cases where the
                # same file is used with different parameters.
                self.group_ids_original[fpath] = group_ids

            if user_provided_group_ids:
                mask = np.isin(user_provided_group_ids, group_ids)
                if np.all(mask):
                    group_ids = np.array(user_provided_group_ids)
                else:
                    group_ids_not_present = np.setdiff1d(user_provided_group_ids, group_ids)
                    raise ValueError(f"The selected group IDs, {group_ids_not_present}, are not present in this "
                                     f" file: {fpath}.")

            if reference_group_ids is None:
                reference_group_ids = group_ids

            elif not np.all(group_ids == reference_group_ids):
                raise Exception(f'Group ID inconsistency detected in this file: {os.path.basename(fpath)}.')

        return reference_group_ids

    def validate_group_ids(self, user_provided_group_ids: list = None) -> None:
        """
        Verify group IDs for consistency across multiple files.

        Reason:
            Group IDs originate from a human-written config file upstream, which
            makes them prone to misordering and duplication. Nothing guarantees
            ascending order or uniqueness at the point they are assigned, so both
            are checked here before any data is extracted against them.

        Args:
            user_provided_group_ids (list): If provided, filter to these IDs
                and ensure they exist in each file.
        """

        self.__check_if_config_validated()

        self.group_ids = self.get_consistent_group_ids(user_provided_group_ids=user_provided_group_ids)

        if self.group_ids is None:
            raise Exception("Failed to retrieve or verify group IDs.")

        self._group_ids_verified = True

    def get_consistent_reference_levels(self, user_provided_group_ids: list = None) -> np.ndarray:
        """
        Retrieve each group's reference levels across files. If any mismatch is
        detected, print the group IDs that do agree everywhere and raise an
        exception. Otherwise, return the reference levels in group-ID order.

        Args:
            user_provided_group_ids (list): If provided, filter to these IDs.
        """

        levels_per_id = {gid: [] for gid in self.group_ids}

        for fpath in self._iter_paths():

            with open_extract(fpath) as extract:
                reference_levels = extract.reference_levels()

            if user_provided_group_ids:
                mask = np.isin(self.group_ids_original[fpath], user_provided_group_ids)
                reference_levels = reference_levels[mask]
            for gid, levels in zip(self.group_ids, reference_levels):
                levels_per_id[gid].append(levels)

        # Check for inconsistencies
        for gid, levels in levels_per_id.items():
            if not all(np.array_equal(levels[0], other) for other in levels[1:]):

                # Report which groups *are* consistent, so a reader can tell whether
                # one group changed or the files describe different experiments.
                common_ids = [g for g, seen in levels_per_id.items()
                              if all(np.array_equal(seen[0], other) for other in seen[1:])]
                logger.info(f"Group IDs with identical reference levels across all files: {common_ids}")

                raise Exception(f"Reference-level inconsistency detected for group ID {gid}.")

        # If no inconsistency, return the reference levels in group-ID order
        return np.array([levels_per_id[gid][0] for gid in self.group_ids])

    def validate_reference_levels(self, user_provided_group_ids: list = None) -> None:
        """
        Verify that every file states the same reference levels for each group.

        Reason:
            Before extracting data across multiple files, it is crucial that they
            agree on what the correct result is at each step. The outcome metric
            is scored against exactly this reference, so if two files disagree
            about it and their outcomes are compared anyway, the comparison is
            between two different questions and the result is misleading.

        Args:
            user_provided_group_ids (list): If provided, filter to these IDs.
        """

        self.__check_if_config_validated()
        self.__check_if_group_ids_verified()

        reference_levels = self.get_consistent_reference_levels(
            user_provided_group_ids=user_provided_group_ids
        )

        if len(self.group_ids) != len(reference_levels):
            raise ValueError(f"Shape mismatch: {len(self.group_ids)} group ID(s) but "
                             f"{len(reference_levels)} row(s) of reference levels.")

        if reference_levels is not None:
            self.reference_levels = reference_levels
        else:
            raise Exception("Failed to retrieve or verify reference levels.")

        self._reference_levels_verified = True

    def get_consistent_metric_names(self) -> tuple[str, ...]:
        """
        Retrieve and verify that every file declares the same metrics.

        The metric set is discovered rather than assumed, so what the run
        analyses is whatever these files agree on. They must agree: pooling
        observations from files carrying different metrics would produce a frame
        whose columns mean different things for different rows.

        Returns:
            The declared metric names, in the column order of the first file.
        """

        reference_names = None
        reference_fpath = None

        for fpath in self._iter_paths():

            with open_extract(fpath) as extract:
                names = extract.metric_names

            if reference_names is None:
                reference_names, reference_fpath = names, fpath

            elif set(names) != set(reference_names):
                only_here = sorted(set(names) - set(reference_names))
                only_there = sorted(set(reference_names) - set(names))
                raise Exception(
                    f"Metric inconsistency between {os.path.basename(reference_fpath)} and "
                    f"{os.path.basename(fpath)}. "
                    f"Only in the second: {only_here or 'nothing'}. "
                    f"Only in the first: {only_there or 'nothing'}."
                )

        return tuple(reference_names)

    def validate_metric_names(self) -> None:
        """
        Verify that every file declares the same metric set, and record it.

        What is recorded is the *declared* set, not the analysed one. Which of
        these metrics a run actually uses, and what part each plays, is settled
        afterwards by :func:`da_pipeline.metrics.resolve_metric_roles` against
        the run's configuration.
        """

        self.__check_if_config_validated()

        self.metric_names = self.get_consistent_metric_names()

        if not self.metric_names:
            raise Exception("Failed to retrieve or verify the declared metric names.")

        self._metric_names_verified = True

    def get_group_label_dict(self, user_provided_group_labels: list = None) -> dict:
        """
        Build the group ID to display label mapping for the run.

        Labels are not carried by the extract format, so by default each group is
        labelled from its own ID. A caller may replace them wholesale.

        Args:
            user_provided_group_labels: Labels to use instead of the derived
                ones, matched to the validated group IDs **by position**.

        Returns:
            Mapping of group IDs to their corresponding labels, sorted by ID.
        """

        self.__check_if_config_validated()
        self.__check_if_group_ids_verified()
        self.__check_if_reference_levels_verified()

        # The user must ensure the correct order of group labels before specifying them!
        if user_provided_group_labels:

            if len(self.group_ids) != len(user_provided_group_labels):
                raise ValueError('The number of group IDs does not match the number of labels.')

            labels = list(user_provided_group_labels)

        else:
            labels = [default_group_label(gid) for gid in self.group_ids]

        labels_dict = dict(zip(self.group_ids, labels))

        # Sort the dictionary by group ids (keys)
        return dict(sorted(labels_dict.items()))

    def get_reference_levels_dict(self) -> dict:
        """
        Returns the reference-levels dictionary for the analysis.
        """

        self.__check_if_config_validated()
        self.__check_if_reference_levels_verified()

        with open_extract(self.first_fpath) as extract:
            reference_levels = extract.reference_levels_dict()

        # Restricted to the validated group ids, which may be a user-provided
        # subset of what the file holds.
        return {
            group_id: reference_levels[group_id]
            for group_id in self.group_ids
        }

    def get_declared_metric_names(self) -> tuple[str, ...]:
        """The metric names every file agreed on."""

        self.__check_if_metric_names_verified()

        return self.metric_names
