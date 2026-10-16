"""The reference producer for this project's extract format.

The format is defined by :mod:`da_pipeline.constants.schema`, and this module is
what writes it. The measurements the pipeline was developed against cannot be
published, so the generator is not a stand-in for a real writer -- it *is* the
writer this project ships, and the files it produces are the worked example of
the layout. That makes it two things at once:

* the reason anyone who clones the repository can run the pipeline end to end;
* the reason the test suite reads real files rather than mocks of the reader.

The data is *plausible*, not real. Each condition is given a ``quality`` between
0 and 1, and the generator makes higher-quality conditions produce higher
accuracy, cleaner separation between the baseline and comparison levels of the
signal (so signal-to-noise rises), and less degradation. That is enough
structure for every figure in the pipeline to show something a reader can
interpret, and for a test to assert that a difference between conditions
actually survives to the output.

Everything is driven by a seed, so a given call always writes the same values.
The one exception is the ``created_utc`` attribute, which records when the file
was written; pass it explicitly for byte-identical output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .constants import schema as sch

# A group's reference is drawn from these levels: mostly 0 and 1, with a few 2.
LEVEL_VALUES = (0, 1, 2)
LEVEL_WEIGHTS = (0.45, 0.40, 0.15)

DEFAULT_N_STEPS = 24
DEFAULT_N_GROUPS = 4
DEFAULT_OBSERVATIONS_PER_GROUP = 120

# Levels the signal-to-noise ratio is built from by default. The panel
# guarantees both appear early in every group's reference, because a group
# missing either is dropped from the SNR frame entirely.
DEFAULT_BASELINE_LEVEL = 0
DEFAULT_COMPARISON_LEVEL = 1

# Run-to-run spread within one condition. Without it every run of a condition is
# an almost exact copy, which makes any run-level statistic look implausibly
# clean -- eta-squared near 1.0, no visible spread in the per-run overlay. Real
# repeats differ because the equipment and materials behind them do.
DEFAULT_RUN_QUALITY_SD = 0.06

# -------------------------------------------------------------------------------------------------
# The metric set
# -------------------------------------------------------------------------------------------------
# Metric names are the producer's to choose -- the format carries them as data,
# in the metric_names attribute, and neither the reader nor the analysis layer
# knows any of them in advance. These are simply what *this* producer writes.

# The metric the demo run is organised around, and the components of it: each
# one is an alternative description of the same result, named after the outcome
# so that the format's naming convention marks them as components.
OUTCOME_METRIC = sch.DEFAULT_OUTCOME_METRIC
N_COMPONENT_METRICS = 5

# Degradation metrics: independent quantities that describe how an observation
# falls out of step as the steps progress. How many there are is a parameter,
# not a property of the format.
DEGRADATION_METRIC_PREFIX = 'degradation'
DEFAULT_N_DEGRADATION_METRICS = 3


def component_metric_names(n: int = N_COMPONENT_METRICS) -> tuple[str, ...]:
    """Names of the outcome's components, e.g. ``Accuracy_error_1``."""
    return tuple(f"{OUTCOME_METRIC}{sch.COMPONENT_SEPARATOR}error_{i}" for i in range(1, n + 1))


def degradation_metric_names(n: int = DEFAULT_N_DEGRADATION_METRICS) -> tuple[str, ...]:
    """Names of the degradation metrics, e.g. ``degradation_1``."""
    return tuple(f"{DEGRADATION_METRIC_PREFIX}_{i}" for i in range(1, n + 1))


def metric_names(
    n_degradation_metrics: int = DEFAULT_N_DEGRADATION_METRICS,
    n_component_metrics: int = N_COMPONENT_METRICS,
) -> tuple[str, ...]:
    """Every metric this producer writes, in matrix column order.

    Outcome first, then its components, then the degradation metrics. The order
    is the producer's choice and the file records it; nothing reads the matrix
    by position.
    """
    return (
        OUTCOME_METRIC,
        *component_metric_names(n_component_metrics),
        *degradation_metric_names(n_degradation_metrics),
    )


@dataclass(frozen=True)
class ConditionSpec:
    """One experimental condition in a synthetic dataset.

    Args:
        name: Condition name, used as the config key and the plot label.
        n_runs: How many extract files make up the condition.
        quality: 0-1. Drives accuracy, signal-to-noise and degradation together,
            so conditions are ordered and the difference is visible in the figures.
    """

    name: str
    n_runs: int = 2
    quality: float = 0.7


DEFAULT_CONDITIONS = (
    ConditionSpec("Condition_A", n_runs=3, quality=0.90),
    ConditionSpec("Condition_B", n_runs=3, quality=0.75),
    ConditionSpec("Condition_C", n_runs=2, quality=0.60),
    ConditionSpec("Condition_D", n_runs=2, quality=0.45),
)


# ---------------------------------------------------------------------------
# The group panel: shared by every file, because validation requires it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupPanel:
    """The groups every file in a dataset must agree on.

    The pipeline refuses to run if two extracts disagree about group IDs or
    reference levels, so the panel is generated once and written into every file
    unchanged. Getting this wrong is exactly the failure the validator exists to
    catch, which makes it a useful thing for tests to be able to reproduce
    deliberately.

    The panel carries no display labels: the format does not, either. Labels are
    derived from the group ids when a run needs them, and configuration can
    replace them.
    """

    group_ids: np.ndarray
    reference_levels: np.ndarray          # (n_groups, n_steps), integer levels

    @property
    def n_groups(self) -> int:
        return len(self.group_ids)

    @property
    def n_steps(self) -> int:
        return self.reference_levels.shape[1]


def build_group_panel(
    n_groups: int = DEFAULT_N_GROUPS,
    n_steps: int = DEFAULT_N_STEPS,
    seed: int = 0,
    baseline_level: int = DEFAULT_BASELINE_LEVEL,
    comparison_level: int = DEFAULT_COMPARISON_LEVEL,
) -> GroupPanel:
    """Build the group panel shared by every file in a dataset.

    Args:
        n_groups: How many groups the panel describes.
        n_steps: Steps per run.
        seed: Makes the panel reproducible.
        baseline_level: Level the signal-to-noise ratio divides by.
        comparison_level: Level the signal-to-noise ratio measures.
    """

    rng = np.random.default_rng(seed)

    group_ids = np.arange(1, n_groups + 1, dtype=np.int64)

    reference_levels = rng.choice(
        LEVEL_VALUES, size=(n_groups, n_steps), p=LEVEL_WEIGHTS
    ).astype(np.int64)

    # Signal-to-noise is a ratio of two levels' medians, so a group whose
    # reference has neither level in the analysed window is silently dropped
    # from the SNR frame. Guarantee both are present in the first two steps,
    # which every sensible step range includes.
    reference_levels[:, 0] = baseline_level
    reference_levels[:, 1] = comparison_level

    return GroupPanel(
        group_ids=group_ids,
        reference_levels=reference_levels,
    )


# ---------------------------------------------------------------------------
# One extract file
# ---------------------------------------------------------------------------

def write_extract(
    path: Path,
    *,
    panel: GroupPanel,
    run_id: str,
    response_name: str = "auc",
    quality: float = 0.7,
    observations_per_group: int = DEFAULT_OBSERVATIONS_PER_GROUP,
    n_degradation_metrics: int = DEFAULT_N_DEGRADATION_METRICS,
    seed: int = 0,
    created_utc: str | None = None,
) -> Path:
    """Write one synthetic extract file.

    The per-observation arrays are all built from the same ``group_id``
    ordering, so the co-indexing the extractor checks for holds by construction.

    Args:
        path: File to write.
        panel: The group panel every file in the dataset shares.
        run_id: Identifier for this run.
        response_name: Names the dataset under ``/signal``.
        quality: 0-1, as on :class:`ConditionSpec`.
        observations_per_group: Observations per group. Drives file size.
        n_degradation_metrics: How many degradation metrics to plant. Any
            number is valid; the format fixes neither the count nor the names.
        seed: Makes the file reproducible.
        created_utc: Written to the ``created_utc`` attribute. Defaults to now,
            which is the only part of the file a seed does not determine.
    """

    if n_degradation_metrics < 0:
        raise ValueError(
            f"n_degradation_metrics: {n_degradation_metrics} cannot be negative."
        )

    rng = np.random.default_rng(seed)

    n_observations = panel.n_groups * observations_per_group
    observation_id = np.arange(n_observations, dtype=np.int64)
    observation_group_id = np.repeat(panel.group_ids, observations_per_group)

    # -- per-observation accuracy ------------------------------------------
    # Beta gives a bounded, left-skewed accuracy distribution with a long lower
    # tail, and enough distinct values for equal-count binning to have something
    # to split.
    #
    # Accuracy is drawn first because everything else about an observation
    # follows from it: one that scores badly is usually one with a poor signal,
    # not a random one.
    accuracy = rng.beta(2.0 + 18.0 * quality, 2.0, size=n_observations).astype(np.float32)

    # -- signal ------------------------------------------------------------
    # One row per observation, one column per step. A reading at a step is
    # proportional to that step's reference level, plus background and noise.
    #
    # Background, step size and noise are all per observation and all follow
    # that observation's accuracy: an inaccurate one gets a higher baseline, a
    # smaller step per level and more scatter. Signal-to-noise is a ratio of
    # medians across steps, so noise alone would average out across the window
    # and leave SNR flat -- it is the background and the step size that have to
    # move for SNR to track accuracy *within* a condition. That within-condition
    # structure is what the "relationships" figures and the correlation stage
    # exist to reveal, so drawing the two independently would misrepresent them.
    per_observation_reference_levels = panel.reference_levels[
        np.searchsorted(panel.group_ids, observation_group_id)
    ]

    background = (0.6 + 0.5 * (1.0 - accuracy))[:, None]
    level_step = ((1.0 + 2.0 * quality) * (0.55 + 0.5 * accuracy))[:, None]
    per_observation_noise = (0.4 * (1.15 - accuracy))[:, None]

    signal = (
        background
        + level_step * per_observation_reference_levels                   # signal
        + rng.normal(0.0, 1.0, size=per_observation_reference_levels.shape) * per_observation_noise
    ).astype(np.float32)

    # -- components of the outcome -----------------------------------------
    # Rarer when accuracy is high. They are drawn from the accuracy itself so
    # that the correlation stage has a real relationship to find. They share one
    # budget so that they move together and each takes a different slice of it,
    # which is what makes them correlated without being copies of one another.
    error_budget = (1.0 - accuracy) * rng.uniform(0.6, 1.4, size=n_observations)
    error_shares = (
        (0.15, 0.35), (0.10, 0.30), (0.10, 0.30), (0.05, 0.20), (0.05, 0.20),
    )
    component_values = [
        (error_budget * rng.uniform(low, high, size=n_observations)).astype(np.float32)
        for low, high in error_shares[:N_COMPONENT_METRICS]
    ]

    # -- degradation metrics -----------------------------------------------
    # Small positive fractions, worse in low-quality conditions. Each one is
    # tapered a little tighter and smaller than the last, so the set is never n
    # copies of a single distribution and a figure comparing them shows n
    # distinguishable shapes however many were asked for.
    degradation_scale = 0.02 + 0.06 * (1.0 - quality)
    degradation_values = []
    for index in range(n_degradation_metrics):
        location = degradation_scale / (1.0 + 0.5 * index)
        degradation_values.append(
            np.abs(rng.normal(location, location / 3, n_observations)).astype(np.float32)
        )

    # -- write -------------------------------------------------------------
    names = metric_names(n_degradation_metrics=n_degradation_metrics)
    metrics = np.column_stack(
        [accuracy, *component_values, *degradation_values]
    ).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype()

    with h5py.File(path, "w") as handle:
        handle.attrs[sch.ATTR_SCHEMA_VERSION] = sch.SCHEMA_VERSION
        handle.attrs[sch.ATTR_RUN_ID] = run_id
        handle.attrs[sch.ATTR_RESPONSE_NAME] = response_name
        handle.attrs[sch.ATTR_CREATED_UTC] = (
            created_utc
            if created_utc is not None
            else datetime.now(UTC).isoformat()
        )

        observations = handle.create_group(sch.GROUP_OBSERVATIONS)
        observations.create_dataset(sch.DS_OBSERVATION_ID, data=observation_id)
        observations.create_dataset(sch.DS_OBSERVATION_GROUP_ID, data=observation_group_id)

        metrics_dataset = observations.create_dataset(sch.DS_METRICS, data=metrics)
        metrics_dataset.attrs[sch.ATTR_METRIC_NAMES] = np.array(names, dtype=str_dtype)

        signal_group = handle.create_group(sch.GROUP_SIGNAL)
        signal_group.create_dataset(response_name, data=signal)

        reference = handle.create_group(sch.GROUP_REFERENCE)
        reference.create_dataset(sch.DS_REFERENCE_LEVELS, data=panel.reference_levels)
        reference.create_dataset(sch.DS_REFERENCE_GROUP_ID, data=panel.group_ids)

    return path


# ---------------------------------------------------------------------------
# A whole dataset, plus the config file that describes it
# ---------------------------------------------------------------------------

def generate_dataset(
    root: Path,
    conditions: tuple[ConditionSpec, ...] = DEFAULT_CONDITIONS,
    *,
    response_name: str = "auc",
    n_groups: int = DEFAULT_N_GROUPS,
    n_steps: int = DEFAULT_N_STEPS,
    observations_per_group: int = DEFAULT_OBSERVATIONS_PER_GROUP,
    n_degradation_metrics: int = DEFAULT_N_DEGRADATION_METRICS,
    baseline_level: int = DEFAULT_BASELINE_LEVEL,
    comparison_level: int = DEFAULT_COMPARISON_LEVEL,
    seed: int = 0,
    config_name: str = "demo_config.json",
    run_quality_sd: float = DEFAULT_RUN_QUALITY_SD,
) -> Path:
    """Write a full synthetic dataset and the JSON config that describes it.

    Args:
        root: Directory to write into. Extracts go in ``root/extracts/<condition>/``
            and the config in ``root/<config_name>``. Results of a later run will
            land beside the config, which is what the pipeline does with real data.
        conditions: The conditions and how many files each has.
        response_name: Response name written into every file.
        n_groups: Groups in the shared panel.
        n_steps: Steps per run. Must cover the step range the pipeline is asked for.
        observations_per_group: Observations per group, per file. Drives file size.
        n_degradation_metrics: How many degradation metrics each file carries.
        baseline_level: Level the run's signal-to-noise ratio divides by.
        comparison_level: Level the run's signal-to-noise ratio measures.
        seed: Makes the whole dataset reproducible.
        config_name: Name of the config file to write.
        run_quality_sd: Spread of run quality around the condition's nominal
            value. Runs of one condition are repeats of the same experiment, but
            they are not identical -- each was set up and measured separately --
            so each run is offset slightly. Set to 0 for perfectly uniform runs.

    Returns:
        Path to the written config file, ready to pass as
        ``--data_analytics_config_fpath``.
    """

    root = Path(root)
    panel = build_group_panel(
        n_groups=n_groups,
        n_steps=n_steps,
        seed=seed,
        baseline_level=baseline_level,
        comparison_level=comparison_level,
    )

    # Drawn from its own generator so that changing observations_per_group or the
    # group panel does not shuffle which run got which offset.
    jitter_rng = np.random.default_rng(seed + 10_000)

    config: dict[str, list[str]] = {}
    file_seed = seed

    for condition in conditions:
        paths = []

        for run_index in range(condition.n_runs):
            file_seed += 1
            path = (
                root / "extracts" / condition.name / f"{condition.name}_run{run_index}.h5"
            )
            # Each run is the same experiment set up afresh: centred on the
            # condition's quality, but never exactly at it.
            run_quality = float(
                np.clip(condition.quality + jitter_rng.normal(0.0, run_quality_sd), 0.05, 0.99)
            )
            write_extract(
                path,
                panel=panel,
                run_id=f"{condition.name}.{run_index}",
                response_name=response_name,
                quality=run_quality,
                observations_per_group=observations_per_group,
                n_degradation_metrics=n_degradation_metrics,
                seed=file_seed,
            )
            paths.append(str(path))

        config[condition.name] = paths

    root.mkdir(parents=True, exist_ok=True)
    config_path = root / config_name
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")

    return config_path
