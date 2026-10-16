"""The run description: what to analyse, how, and where to put the results.

Everything the pipeline needs to know about a run lives in one frozen
``PipelineConfig``. Components take that object and read typed attributes from
it, which is what lets a reader of ``extraction.py`` see that
``config.response_name`` is a ``str`` without going to look at how the process
was started.

Validation lives here rather than in the argument parser on purpose. A config
built in a notebook or a test gets exactly the same checks as one built from
``sys.argv``, so there is no set of values that only the CLI can reject:

    >>> PipelineConfig(config_path=Path("cfg.json"), response_name="auc",
    ...                step_range=StepRange(4, 2))
    Traceback (most recent call last):
    ConfigError: step_range: start (4) must not exceed stop (2).

One kind of check cannot live here. Which metrics a run analyses is answered
against the metric names the extract files declare, and at construction no file
has been opened -- so the metric selection is validated by
:func:`da_pipeline.metrics.resolve_metric_roles` once the files are read. What
this module still checks is that a selection is internally coherent.

The dataclasses are frozen because a config is read from worker processes.
Nothing should be able to change the step range halfway through a run and leave
two figures describing different windows of the same data. Derived values are
produced by returning a new config -- see :meth:`PipelineConfig.with_output_dir`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .constants import outputs as out
from .metrics import MetricSelection

# -------------------------------------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------------------------------------
# Named here, once. The CLI advertises these values in --help and the
# dataclasses fall back to them, so there is no second copy to drift.

DEFAULT_STEP_START = 1
DEFAULT_STEP_STOP = 17
DEFAULT_DPI = 300
DEFAULT_MAX_WORKERS = 6
DEFAULT_OUTCOME_NUM_BINS = 10

# The two reference levels a signal-to-noise ratio is built from: the level
# where nothing is expected, and the first level where something is.
DEFAULT_SNR_BASELINE_LEVEL = 0
DEFAULT_SNR_COMPARISON_LEVEL = 1

DEFAULT_FIGURE_WIDTH = 9.0
DEFAULT_FIGURE_HEIGHT = 4.5
DEFAULT_XTICK_LABEL_ROTATION = 45


class ConfigError(ValueError):
    """A configuration value is unusable, and the run cannot start.

    Raised at construction, so an invalid combination is caught before any file
    is opened rather than several minutes into extraction.
    """


def _as_tuple(values, cast=None):
    """Freeze a sequence into a tuple, or pass None through untouched."""
    if values is None:
        return None
    return tuple(cast(v) for v in values) if cast else tuple(values)


def _check_range(name: str, bounds) -> tuple[float, float]:
    """Validate a (min, max) axis range."""
    low, high = bounds
    if low >= high:
        raise ConfigError(f"{name}: min ({low}) must be less than max ({high}).")
    return float(low), float(high)


class MappingProxy(dict):
    """A dict that refuses writes, so a frozen config stays frozen all the way down.

    ``dataclasses.frozen`` only stops attribute rebinding; it does nothing about
    mutating a dict the config holds. Subclassing dict (rather than using
    ``types.MappingProxyType``) keeps the object picklable, which matters
    because the plotting layer sends the config to worker processes.
    """

    def __reduce__(self):
        """Rebuild through the constructor when pickled.

        The default dict-subclass pickle path replays the contents through
        ``__setitem__``, which this class refuses -- so an unpickle would raise
        instead of returning the mapping. Going through ``dict.__init__``
        sidesteps the blocked methods. This matters because the plotting layer
        sends the config to ProcessPoolExecutor workers.
        """
        return self.__class__, (dict(self),)

    def _readonly(self, *args, **kwargs):
        raise TypeError(
            "PipelineConfig is immutable; build a new one with dataclasses.replace()."
        )

    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly


# -------------------------------------------------------------------------------------------------
# Step range
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StepRange:
    """A 1-based, inclusive window of steps: ``StepRange(1, 17)`` is steps 1-17.

    Users count steps from 1 and expect both ends included, which is not what a
    Python slice means. Rather than scatter ``- 1`` adjustments through the
    extraction code, the conversion happens once, here, and the extractor asks
    for :attr:`as_slice`.
    """

    start: int = DEFAULT_STEP_START
    stop: int = DEFAULT_STEP_STOP

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ConfigError(
                f"step_range: start ({self.start}) must be >= 1; steps are numbered from 1."
            )
        if self.start > self.stop:
            raise ConfigError(
                f"step_range: start ({self.start}) must not exceed stop ({self.stop})."
            )

    @property
    def as_slice(self) -> slice:
        """The equivalent 0-based, half-open slice for indexing arrays."""
        return slice(self.start - 1, self.stop)

    def __len__(self) -> int:
        return self.stop - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.stop}"


# -------------------------------------------------------------------------------------------------
# CSV exports
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportSelection:
    """Which datasets are written out as CSV.

    The defaults are the two a run is normally kept for. The rest are
    per-observation or per-step frames that reach hundreds of megabytes on a
    full dataset, so they are opt-in rather than opt-out.
    """

    observation_counts: bool = False
    responses: bool = False
    signal_to_noise: bool = True
    predictor_metrics: bool = False
    outcome_metrics: bool = False
    correlations: bool = True


# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotSettings:
    """Figure geometry, encoding and axis limits for the run.

    ``colours`` is ``None`` by default, meaning "use the colour cycle in
    analytics.mplstyle". Resolving it here would make this module import
    matplotlib, which a config object has no business doing; the plotting layer
    resolves it instead, at the one point that already owns the theme.
    """

    colours: tuple[str, ...] | None = None
    dpi: int = DEFAULT_DPI
    figure_width: float = DEFAULT_FIGURE_WIDTH
    figure_height: float = DEFAULT_FIGURE_HEIGHT
    show_outliers: bool = False
    boxplot_show_means: bool = True
    xtick_label_rotation: int = DEFAULT_XTICK_LABEL_ROTATION

    # Append '(6 runs)' to each condition's plot label. Affects labels only:
    # the raw condition names stay in the dataframes.
    show_condition_counts: bool = True

    # Per-level y-axis limits, {level: (min, max)}.
    response_level_yaxis_ranges: Mapping[int, tuple[float, float]] | None = None
    signal_to_noise_yaxis_range: tuple[float, float] | None = None
    predictor_metric_yaxis_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'colours', _as_tuple(self.colours, str))

        if self.dpi < 1:
            raise ConfigError(f"dpi: {self.dpi} must be >= 1.")

        for name in ('figure_width', 'figure_height'):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"{name}: {value} must be greater than 0.")

        if not -360 <= self.xtick_label_rotation <= 360:
            raise ConfigError(
                f"xtick_label_rotation: {self.xtick_label_rotation} must be between -360 and 360."
            )

        for name in ('signal_to_noise_yaxis_range', 'predictor_metric_yaxis_range'):
            bounds = getattr(self, name)
            if bounds is not None:
                object.__setattr__(self, name, _check_range(name, bounds))

        if self.response_level_yaxis_ranges is not None:
            validated = {}
            for level, bounds in self.response_level_yaxis_ranges.items():
                if level < 0:
                    raise ConfigError(
                        f"response_level_yaxis_ranges: level must be >= 0; got {level}."
                    )
                validated[int(level)] = _check_range(
                    f"response_level_yaxis_ranges[{level}]", bounds
                )
            object.__setattr__(
                self, 'response_level_yaxis_ranges', MappingProxy(validated)
            )

    def figsize(self, height_scale: float = 1.0) -> tuple[float, float]:
        return self.figure_width, self.figure_height * height_scale


# -------------------------------------------------------------------------------------------------
# The run
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Everything one run of the pipeline needs to know.

    Args:
        config_path: JSON file mapping each condition to its list of extract
            files. Also fixes where results are written: the output directory is
            created beside it.
        response_name: The response carried by the extracts, e.g. ``auc``. Must
            match the ``response_name`` root attribute in the files.
        step_range: 1-based inclusive window of steps to analyse.
        group_ids: Restrict the analysis to these groups. ``None`` means every
            group present in the files.
        group_labels: Replacement display labels, positionally matched to
            ``group_ids``. Requires ``group_ids``, because the mapping is by
            position and there is nothing else to align against.
        metrics: Which of the metrics the files declare to analyse, and what
            part each plays. Resolved against the files once they are opened.
        snr_baseline_level: The reference level the signal-to-noise ratio
            divides by -- the reading where nothing is expected.
        snr_comparison_level: The reference level the signal-to-noise ratio
            measures against that baseline.
        outcome_num_bins: Number of equal-count bins of the outcome metric,
            computed per condition.
        calculate_effect_size: Compute eta-squared across conditions.
        max_workers: Upper bound on worker threads and figure processes.
        output_folder_name_suffix: Appended to the output folder name, to keep
            several analyses of the same config side by side.
        output_dir: Filled in once the directory has been created. ``None``
            until then -- see :meth:`with_output_dir`.
    """

    config_path: Path
    response_name: str

    step_range: StepRange = field(default_factory=StepRange)
    group_ids: tuple[int, ...] | None = None
    group_labels: tuple[str, ...] | None = None

    metrics: MetricSelection = field(default_factory=MetricSelection)
    snr_baseline_level: int = DEFAULT_SNR_BASELINE_LEVEL
    snr_comparison_level: int = DEFAULT_SNR_COMPARISON_LEVEL
    outcome_num_bins: int = DEFAULT_OUTCOME_NUM_BINS
    calculate_effect_size: bool = True
    max_workers: int = DEFAULT_MAX_WORKERS

    output_folder_name_suffix: str | None = None
    output_dir: Path | None = None

    exports: ExportSelection = field(default_factory=ExportSelection)
    plots: PlotSettings = field(default_factory=PlotSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'config_path', Path(self.config_path))
        object.__setattr__(self, 'group_ids', _as_tuple(self.group_ids, int))
        object.__setattr__(self, 'group_labels', _as_tuple(self.group_labels, str))

        if self.output_dir is not None:
            object.__setattr__(self, 'output_dir', Path(self.output_dir))

        if not self.response_name:
            raise ConfigError("response_name: must be a non-empty string.")

        # Which metrics exist is a question about the files, not about this
        # object, so the selection is checked for coherence here and resolved
        # against the declared names in metrics.resolve_metric_roles.
        if not isinstance(self.metrics, MetricSelection):
            raise ConfigError(
                f"metrics: expected a MetricSelection, got {type(self.metrics).__name__}."
            )

        self._validate_snr_levels()

        if self.outcome_num_bins < 2:
            raise ConfigError(
                f"outcome_num_bins: {self.outcome_num_bins} must be >= 2; "
                f"one bin is not a comparison."
            )

        if self.max_workers < 1:
            raise ConfigError(f"max_workers: {self.max_workers} must be >= 1.")

        self._validate_group_labels()
        self._validate_suffix()

    def _validate_snr_levels(self) -> None:
        """The two reference levels the signal-to-noise ratio is built from.

        Generalised from a fixed level-1-over-level-0 median ratio: which levels
        separate "a reading" from "nothing" is a property of the data, not of
        the pipeline, and a reference that uses other levels is perfectly valid.
        """
        for name in ('snr_baseline_level', 'snr_comparison_level'):
            value = getattr(self, name)
            if value < 0:
                raise ConfigError(f"{name}: {value} must be >= 0; levels are counted from 0.")

        if self.snr_baseline_level == self.snr_comparison_level:
            raise ConfigError(
                f"snr_baseline_level and snr_comparison_level are both "
                f"{self.snr_baseline_level}; a ratio of a level against itself is 1 "
                f"by construction."
            )

    def _validate_group_labels(self) -> None:
        if self.group_labels is None:
            return

        if not self.group_ids:
            raise ConfigError(
                "group_labels: requires group_ids, since labels are mapped to group IDs by position."
            )

        if len(self.group_labels) != len(self.group_ids):
            raise ConfigError(
                f"group_labels: got {len(self.group_labels)} label(s) for "
                f"{len(self.group_ids)} group ID(s). The counts must match so that "
                f"labels map to IDs by position."
            )

    def _validate_suffix(self) -> None:
        suffix = self.output_folder_name_suffix

        if suffix is None:
            return

        suffix = suffix.strip()

        if ' ' in suffix:
            raise ConfigError(f"output_folder_name_suffix: {suffix!r} cannot contain spaces.")

        if len(suffix) > out.MAX_OUTPUT_FOLDER_SUFFIX_LEN:
            raise ConfigError(
                f"output_folder_name_suffix: {suffix!r} cannot exceed "
                f"{out.MAX_OUTPUT_FOLDER_SUFFIX_LEN} characters (got {len(suffix)})."
            )

        object.__setattr__(self, 'output_folder_name_suffix', suffix)

    # ---------------------------------------------------------------------
    # Derived values
    # ---------------------------------------------------------------------

    @property
    def output_folder_name(self) -> str:
        """Name of this run's output directory, suffix included."""
        if self.output_folder_name_suffix:
            return f"{out.DA_OUTPUT_FOLDER_NAME}_{self.output_folder_name_suffix}"
        return out.DA_OUTPUT_FOLDER_NAME

    @property
    def has_group_filter(self) -> bool:
        """Whether the run is restricted to a user-chosen subset of groups."""
        return bool(self.group_ids)

    @property
    def resolved_output_dir(self) -> Path:
        """The output directory, or a clear error if the run has not created it yet."""
        if self.output_dir is None:
            raise ConfigError(
                "output_dir has not been set. Call with_output_dir() (the pipeline "
                "does this once the directory exists)."
            )
        return self.output_dir

    def with_output_dir(self, output_dir: Path | str) -> PipelineConfig:
        """Return a copy carrying the created output directory."""
        return replace(self, output_dir=Path(output_dir))

    def describe(self) -> list[str]:
        """The run's settings as log lines, one per setting."""
        lines = [
            f"- config_path: {self.config_path}",
            f"- response_name: {self.response_name}",
            f"- step_range: {self.step_range}",
            f"- group_ids: {list(self.group_ids) if self.group_ids else 'all'}",
            f"- group_labels: {list(self.group_labels) if self.group_labels else 'derived from group IDs'}",
            *self.metrics.describe(),
            f"- snr_baseline_level: {self.snr_baseline_level}",
            f"- snr_comparison_level: {self.snr_comparison_level}",
            f"- outcome_num_bins: {self.outcome_num_bins}",
            f"- calculate_effect_size: {self.calculate_effect_size}",
            f"- max_workers: {self.max_workers}",
            f"- output_dir: {self.output_dir}",
        ]
        lines += [f"- exports.{name}: {value}" for name, value in vars(self.exports).items()]
        lines += [f"- plots.{name}: {value}" for name, value in vars(self.plots).items()]
        return lines
