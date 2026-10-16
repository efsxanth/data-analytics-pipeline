import concurrent.futures
import logging
import os
from collections.abc import Callable
from functools import cached_property

import numpy as np
import polars as pl

from .config import PipelineConfig
from .constants import columns as col
from .frames import reorder_dataframe, sort_dataframe
from .io.extract_reader import open_extract
from .metrics import MetricRoles

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Extracts and processes observation counts, responses, signal-to-noise ratios
    and the run's metrics from extract files.

    Which metrics those are is not known to this class. It is handed a
    :class:`~da_pipeline.metrics.MetricRoles`, reads whatever it names, and
    produces two per-observation frames: the predictor metrics, and the outcome
    metric with its components. No metric name appears anywhere below.

    It uses thread-based parallelism to scale across multiple experimental
    conditions, and ensures consistent labelling, ordering, and formatting.
    """

    def __init__(self,
                 config: PipelineConfig,
                 roles: MetricRoles,
                 config_dict: dict | None = None,
                 group_ids: np.ndarray | None = None,
                 group_label_dict: dict | None = None,
                 reference_levels_dict: dict | None = None):
        """
        Args:
            config: The run's settings.
            roles: The run's resolved metric set: which metric is the outcome,
                which restate it, and which are predictors.
            config_dict: {condition: [extract file paths]}.
            group_ids: The validated group IDs to extract, in output order.
            group_label_dict: {group_id: display label}.
            reference_levels_dict: {group_id: reference_levels}, read once from the
                first file and reused, since validation has established that
                every file agrees.
        """

        self.config = config
        self.roles = roles

        # Read once here: this is the array-indexing form of the user's 1-based
        # step window, and it is used on every file.
        self.step_slice = config.step_range.as_slice

        self.config_dict = config_dict
        self.group_ids = group_ids
        self.group_label_dict = group_label_dict
        self.reference_levels_dict = reference_levels_dict

    @property
    def response_name(self) -> str:
        return self.config.response_name

    # ----------------------------------------------------------------------------------------------------------------
    # Shared helpers
    # ----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _check_co_indexed(observation_ids: np.ndarray, observation_group_ids: np.ndarray,
                          fpath: str) -> None:
        """
        The id and group arrays must be co-indexed for the per-row identity
        to be valid. Everything downstream joins on that identity, so a length
        mismatch would misalign silently rather than fail.
        """
        if len(observation_ids) != len(observation_group_ids):
            raise ValueError(
                f"{os.path.basename(fpath)}: observation id/group arrays are not "
                f"co-indexed ({len(observation_ids)} vs {len(observation_group_ids)})."
            )

    def get_masked_group_reference_levels(self, group_id) -> np.ndarray:
        """Retrieves and masks the reference_levels data for a given group ID."""

        group_reference_levels = self.reference_levels_dict[group_id]
        return group_reference_levels[self.step_slice]

    @cached_property
    # cached_property is a decorator from Python's functools.  It turns a method
    # into a property whose value is computed once on first access and then stored,
    # so every later access returns the stored value instead of recomputing.
    def _labels_df(self) -> pl.DataFrame:
        """Group-ID -> label lookup frame, built once and reused across files."""

        return pl.DataFrame(
            {
                col.GROUP_ID: list(self.group_label_dict.keys()),
                col.GROUP_LABEL: list(self.group_label_dict.values()),
            }
        ).with_columns(
            pl.col(col.GROUP_ID).cast(pl.Int64)  # consistent dtype for joins
        )

    def _filter_to_user_groups(self, observation_group_ids: np.ndarray, observation_ids: np.ndarray,
                               value_data: dict) -> tuple:
        """
        If the run is not restricted to a subset of groups, returns the inputs
        untouched. Otherwise, returns the ids, indices, and value arrays filtered
        to self.group_ids.

        The observation ids are masked with the SAME boolean mask as the group ids and
        the value arrays, so all three stay co-indexed after filtering.
        """

        if not self.config.has_group_filter:
            return observation_group_ids, observation_ids, value_data

        mask = np.isin(observation_group_ids, self.group_ids)
        filtered = {name: arr[mask] for name, arr in value_data.items()}
        return observation_group_ids[mask], observation_ids[mask], filtered

    def _assemble_wide_frame(self, observation_group_ids: np.ndarray, observation_ids: np.ndarray,
                             value_data: dict, value_columns: list, run_id, condition: str) -> pl.DataFrame:
        """
        Assemble a per-file 'wide' frame (one row per observation, one column per metric):
        joins group labels, attaches run/condition metadata, enforces a stable column order,
        and rechunks for a contiguous memory layout.

        The observation id is carried in the same base frame as the group id, so the two stay
        mutually consistent regardless of join ordering; it becomes the per-row identity used
        downstream to align this frame against others by key rather than by row position.
        """

        base_df = pl.DataFrame(
            {
                col.GROUP_ID: observation_group_ids,
                col.OBSERVATION_ID: observation_ids,
            }
        ).with_columns(
            pl.col(col.GROUP_ID).cast(pl.Int64),  # consistent dtype for joins
            pl.col(col.OBSERVATION_ID).cast(pl.Int64),              # canonical dtype for the identity key
        )
        values_df = pl.from_dict(value_data)

        # hstack rather than a horizontal concat: the two frames are one row per
        # observation and must stay that way, and hstack raises on a height mismatch
        # instead of padding the shorter one with nulls. The label join is a left
        # join on a unique group id, so it cannot change the height -- and if it
        # ever did, silently padded metrics would be far worse than an exception.
        return base_df.join(
            self._labels_df, on=col.GROUP_ID, how="left",
        ).hstack(values_df).with_columns(
            pl.lit(run_id).alias(col.RUN_ID),
            pl.lit(condition).alias(col.CONDITION),
        ).select(
            [
                col.RUN_ID,
                col.CONDITION,
                col.GROUP_ID,
                col.GROUP_LABEL,
                col.OBSERVATION_ID,
                *value_columns,
            ]
        ).rechunk()

    def _run_parallel(
            self,
            task_fn: Callable[[str, str], pl.DataFrame],
            label: str,
            columns_to_sort: list[str],
            frame_transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    ) -> pl.DataFrame:

        frames = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
        ) as executor:

            tasks = [
                executor.submit(
                    task_fn,
                    fpath,
                    condition,
                )
                for condition, fpaths in self.config_dict.items()
                for fpath in fpaths
            ]

            for future in concurrent.futures.as_completed(tasks):

                try:
                    df = future.result()

                except Exception:
                    logger.exception(
                        f"{label}: a file failed to process "
                        f"and was skipped."
                    )
                    continue

                if df is None or df.height == 0:
                    continue

                if frame_transform is not None:
                    df = frame_transform(df)

                frames.append(df)

        if not frames:
            raise RuntimeError(
                f"No data extracted for {label}: "
                f"all files failed or returned empty."
            )

        # Ensure all worker outputs share the same schema before concatenation.
        expected_schema = frames[0].schema
        for idx, frame in enumerate(frames[1:], start=1):
            if frame.schema != expected_schema:
                raise ValueError(
                    f"{label}: schema mismatch in frame {idx}\n\n"
                    f"Expected:\n{expected_schema}\n\n"
                    f"Observed:\n{frame.schema}"
                )

        combined = pl.concat(
            frames,
            rechunk=True,
        )

        combined = sort_dataframe(
            df=combined,
            columns_to_sort=columns_to_sort,
        )

        if self.config.has_group_filter:
            combined = combined.with_columns(
                pl.col(
                    col.GROUP_ID
                ).cast(pl.Int64)
            )

            combined = reorder_dataframe(
                df=combined,
                column=col.GROUP_ID,
                group_id_order=self.group_ids,
            )

        logger.info(
            f"- {label}: {combined.shape}"
        )

        return combined

    # ----------------------------------------------------------------------------------------------------------------
    # Observation count
    # ----------------------------------------------------------------------------------------------------------------

    def extract_observation_count(self, fpath: str, condition: str) -> pl.DataFrame:
        """
        Extracts observation count (number of observations).

        Args:
            fpath: Path to the file.
            condition: The experimental condition.

        Returns:
            Polars DataFrame containing observation count per group ID,
            including run ID, condition, and labels.
        """

        with open_extract(fpath) as extract:
            run_id = extract.run_id
            observation_group_ids = extract.observation_group_ids()

        # Compute observation count per group
        unique_group_ids, observation_count_values = np.unique(
            observation_group_ids, return_counts=True
        )

        observation_count_map = dict(zip(unique_group_ids, observation_count_values))

        observation_count = np.array([
            observation_count_map.get(i, 0) for i in self.group_ids
        ])

        return (
            pl.DataFrame({
                col.OBSERVATION_COUNT: observation_count
            })
            .with_columns([
                pl.lit(run_id).alias(col.RUN_ID),
                pl.lit(condition).alias(col.CONDITION),
                pl.Series(col.GROUP_ID, self.group_ids),
                pl.Series(
                    col.GROUP_LABEL,
                    [self.group_label_dict[i] for i in self.group_ids]
                ),
            ])
            .select(col.METADATA_COLS + [col.GROUP_LABEL] + [col.OBSERVATION_COUNT])
        )

    def process_observation_count(self) -> pl.DataFrame:
        """Processes and aggregates observation count data."""

        combined = self._run_parallel(
            self.extract_observation_count,
            label="Observation counts",
            columns_to_sort=col.SORT_COLS_OBSERVATION_COUNTS
        )

        final_observation_counts_df = (
            combined
            .group_by([
                col.CONDITION,
                col.GROUP_ID,
                col.GROUP_LABEL
            ])
            .agg([
                pl.col(col.OBSERVATION_COUNT)
                .sum()
                .alias(col.OBSERVATION_COUNT_TOTAL),

                pl.col(col.OBSERVATION_COUNT)
                .median()
                .alias(col.OBSERVATION_COUNT_MEDIAN),
            ])
            .sort([
                col.CONDITION,
                col.GROUP_ID
            ])
        )

        return final_observation_counts_df

    # ----------------------------------------------------------------------------------------------------------------
    # Responses
    # ----------------------------------------------------------------------------------------------------------------

    def extract_responses(self, fpath: str) -> tuple:
        """
        Extracts the responses from a single file.

        Args:
            fpath: The path to the file to extract responses from.

        Returns:
            A tuple containing the file path and its extracted responses.
        """

        with open_extract(fpath) as extract:
            responses = extract.signal()

        return fpath, responses

    def preload_responses(self) -> dict:
        """
        Preloads responses for each file in parallel.

        Returns:
            A dictionary mapping each file path to its responses.
        """

        responses_dict = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.extract_responses, fpath): fpath
                for condition, fpaths in self.config_dict.items()
                for fpath in fpaths
            }

            for future in concurrent.futures.as_completed(futures):
                fpath = futures[future]
                try:
                    fpath, responses = future.result()
                except Exception:
                    logger.exception(
                        f'response preload: {os.path.basename(fpath)} failed to load and was skipped.'
                    )
                    continue
                responses_dict[fpath] = responses

        if not responses_dict:
            raise RuntimeError(
                'No responses preloaded: all files failed or returned empty.'
            )

        return responses_dict

    def create_responses_dataframe(self, responses: np.ndarray, fpath: str, condition: str) -> pl.DataFrame:
        """
        Extracts and aggregates responses by group and level for a single file.

        Args:
            responses: The response data for this file.
            fpath: The path to the file.
            condition: The experimental condition.

        Returns:
            Aggregated (and exploded) responses DataFrame for the file.
        """

        with open_extract(fpath) as extract:
            run_id = extract.run_id
            observation_group_ids = extract.observation_group_ids()  # read once per file
            observation_ids = extract.observation_ids()

        self._check_co_indexed(observation_ids, observation_group_ids, fpath)

        responses_data_list = []

        for group_id in self.group_ids:
            group_id_mask = observation_group_ids == group_id

            if not group_id_mask.any():
                logger.info(f'  --> {os.path.basename(fpath)}: Group ID {group_id} '
                            f'does not have any responses available.')
                continue

            group_responses = responses[group_id_mask, self.step_slice]
            group_reference_levels_masked = self.get_masked_group_reference_levels(group_id=group_id)
            group_observation_ids = observation_ids[group_id_mask]

            for level in np.unique(group_reference_levels_masked):
                level_positions = np.flatnonzero(group_reference_levels_masked == level)

                row = {
                    col.RUN_ID: run_id,
                    col.CONDITION: condition,
                    col.GROUP_ID: group_id,
                    col.LEVEL: level,
                    col.GROUP_LABEL: self.group_label_dict[group_id],
                }

                # One median per observation across the step positions of this
                # level, so each value maps to exactly one observation and
                # the row identity stays intact all the way downstream.
                row[col.OBSERVATION_ID] = group_observation_ids.tolist()
                row[self.response_name] = np.median(group_responses[:, level_positions], axis=1).tolist()

                responses_data_list.append(row)

        if not responses_data_list:
            return None

        # Build the frame once, then explode the per-observation list columns into
        # one value per row. The id is exploded alongside the response, so
        # every row keeps the observation it came from.
        #
        # empty_as_null is stated rather than left to the default, which Polars
        # 2.0 flips. A file with no observations for a group is a real possibility,
        # and it should keep producing one null row rather than silently
        # vanishing -- a dropped row would take the group out of the run's
        # counts without anything reporting that it had gone.
        responses_df = (
            pl.DataFrame(responses_data_list)
            .explode([self.response_name, col.OBSERVATION_ID], empty_as_null=True)
            .with_columns(pl.col(col.OBSERVATION_ID).cast(pl.Int64))
        )

        return responses_df

    def process_response_datasets(self, preloaded_responses_dict: dict) -> pl.DataFrame:
        """
        Processes responses for each file using preloaded responses.

        Args:
            preloaded_responses_dict: A dictionary mapping each file path to its responses.

        Returns:
            Concatenated DataFrame containing responses for all files.
        """

        def task_fn(fpath: str, condition: str) -> pl.DataFrame:
            responses = preloaded_responses_dict.get(fpath)
            if responses is None:
                return None  # file was skipped during preload; _run_parallel drops None frames
            return self.create_responses_dataframe(responses, fpath, condition)

        return self._run_parallel(
            task_fn,
            label="Responses",
            columns_to_sort=col.SORT_COLS_RESPONSES
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Signal-to-noise (SNR)
    # ----------------------------------------------------------------------------------------------------------------

    def calculate_signal_to_noise_ratios(self, responses: np.ndarray, fpath: str, condition: str) -> pl.DataFrame:
        """
        Calculates signal-to-noise ratios for each group ID in a file.

        The ratio is the median response at the comparison level over the median
        at the baseline level, both taken across the steps of the analysed
        window. Which two levels those are comes from the config: "a reading"
        and "nothing" are properties of the reference, not of this code.

        Args:
            responses: The response data for this file.
            fpath: Path to the file.
            condition: Experimental condition associated with the file.

        Returns:
            pl.DataFrame: Signal-to-noise ratios for each group ID, or None if no data.
        """

        baseline_level = self.config.snr_baseline_level
        comparison_level = self.config.snr_comparison_level

        with open_extract(fpath) as extract:
            run_id = extract.run_id
            observation_group_ids = extract.observation_group_ids()
            observation_ids = extract.observation_ids()

        # Co-indexed so observation_ids[mask] lines up with the per-observation SNR
        # values computed from the same mask.
        self._check_co_indexed(observation_ids, observation_group_ids, fpath)

        signal_to_noise_ratios_list = []

        for group_id in self.group_ids:
            group_id_mask = observation_group_ids == group_id

            if not group_id_mask.any():
                # Missing-group logging is handled in response extraction to avoid duplicate messages.
                continue

            group_responses = responses[group_id_mask, self.step_slice]
            group_observation_ids = observation_ids[group_id_mask]  # same observations, same order as the ratios
            group_reference_levels_masked = self.get_masked_group_reference_levels(group_id=group_id)

            baseline_positions = np.flatnonzero(group_reference_levels_masked == baseline_level)
            comparison_positions = np.flatnonzero(group_reference_levels_masked == comparison_level)

            if baseline_positions.size > 0 and comparison_positions.size > 0:
                baseline_median = np.median(group_responses[:, baseline_positions], axis=1)
                comparison_median = np.median(group_responses[:, comparison_positions], axis=1)

                # comparison / baseline. NB: a zero baseline median yields inf/nan; left as-is by design.
                signal_to_noise_ratios = comparison_median / baseline_median

                signal_to_noise_ratios_list.append({
                    col.RUN_ID: run_id,
                    col.CONDITION: condition,
                    col.GROUP_ID: group_id,
                    col.GROUP_LABEL: self.group_label_dict[group_id],
                    col.OBSERVATION_ID: group_observation_ids.tolist(),
                    col.SIGNAL_TO_NOISE_RATIO: signal_to_noise_ratios.tolist(),
                })
            else:
                logger.info(
                    f'Group ID {group_id} does not have valid level {baseline_level} or '
                    f'level {comparison_level} positions.'
                )

        if not signal_to_noise_ratios_list:
            return None

        signal_to_noise_ratios_df = pl.DataFrame(signal_to_noise_ratios_list)
        # Explode the id alongside the ratios so each row keeps its own observation identity,
        # then cast to the canonical dtype used as the downstream join key.
        # empty_as_null is stated for the reason given in create_responses_dataframe.
        return (
            signal_to_noise_ratios_df
            .explode([col.SIGNAL_TO_NOISE_RATIO, col.OBSERVATION_ID], empty_as_null=True)
            .with_columns(pl.col(col.OBSERVATION_ID).cast(pl.Int64))
        )

    def process_signal_to_noise(self, preloaded_responses_dict: dict) -> pl.DataFrame:
        """
        Processes signal-to-noise ratios across multiple files in parallel.

        Args:
            preloaded_responses_dict: A dictionary mapping each file path to its responses.

        Returns:
            Polars DataFrame containing signal-to-noise ratios for all processed files.
        """

        def task_fn(fpath: str, condition: str) -> pl.DataFrame:
            return self.calculate_signal_to_noise_ratios(
                preloaded_responses_dict[fpath], fpath, condition
            )

        return self._run_parallel(
            task_fn,
            label="signal-to-noise",
            columns_to_sort=col.SORT_COLS_PER_OBSERVATION
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------------------------------------------------

    def _extract_metrics(self, fpath: str, condition: str, metric_names: list[str],
                         clip_to_unit_interval: bool = False) -> pl.DataFrame:
        """
        Read a named set of metrics out of one file as a per-observation frame.

        Args:
            fpath: Path to the file.
            condition: The experimental condition.
            metric_names: The metrics to read. Any subset of what the file
                declares; validation has already established that the name is
                one the files carry.
            clip_to_unit_interval: Whether to clip values into [0, 1].

        Returns:
            Run/condition metadata, group ID and label, observation ID, and one
            column per metric.
        """

        with open_extract(fpath) as extract:
            run_id = extract.run_id
            observation_group_ids = extract.observation_group_ids()
            observation_ids = extract.observation_ids()

            metrics_data = extract.metrics(metric_names)

        if clip_to_unit_interval:
            metrics_data = {
                name: np.clip(values, 0.0, 1.0) for name, values in metrics_data.items()
            }

        self._check_co_indexed(observation_ids, observation_group_ids, fpath)

        observation_group_ids, observation_ids, metrics_data = self._filter_to_user_groups(
            observation_group_ids, observation_ids, metrics_data
        )

        return self._assemble_wide_frame(
            observation_group_ids=observation_group_ids,
            observation_ids=observation_ids,
            value_data=metrics_data,
            value_columns=list(metric_names),
            run_id=run_id,
            condition=condition,
        )

    def _process_metrics(self, metric_names: list[str], label: str,
                         clip_to_unit_interval: bool = False) -> pl.DataFrame:
        """Extract one metric family across every file, in parallel."""

        def task_fn(fpath: str, condition: str) -> pl.DataFrame:
            return self._extract_metrics(
                fpath, condition, metric_names,
                clip_to_unit_interval=clip_to_unit_interval,
            )

        def to_float32(df: pl.DataFrame) -> pl.DataFrame:
            # Cast metric columns to a common dtype so per-file frames share an
            # identical schema for concat.
            return df.with_columns([
                pl.col(metric).cast(pl.Float32) for metric in metric_names
            ])

        return self._run_parallel(
            task_fn,
            label=label,
            columns_to_sort=col.SORT_COLS_PER_OBSERVATION,
            frame_transform=to_float32,
        )

    def process_predictor_metrics(self) -> pl.DataFrame:
        """
        Extract the run's predictor metrics across every file.

        Values are clipped into [0, 1]: these describe a share of the measurement
        that went one way or another, so a value outside the interval is a
        producer's rounding artefact rather than a real reading.
        """

        return self._process_metrics(
            metric_names=list(self.roles.predictors),
            label="Predictor metrics",
            clip_to_unit_interval=True,
        )

    def process_outcome_metrics(self) -> pl.DataFrame:
        """Extract the outcome metric and its components across every file."""

        return self._process_metrics(
            metric_names=list(self.roles.outcome_metrics),
            label="Outcome metrics",
        )
