"""Command-line front end.

This module does two things and deliberately not a third: it turns strings from
``argv`` into Python values, and it assembles a :class:`~da_pipeline.config.PipelineConfig`.
It does *not* decide whether those values make sense together -- the config does
that, so a run started from a notebook is checked exactly as strictly as one
started from a shell.

The pay-off is visible in :func:`build_config`: cross-argument rules that used
to live here as a sequence of ``parser.error`` calls are now a single
``try/except ConfigError``. The CLI's job is to render the complaint as a usage
error rather than a traceback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_DPI,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_MAX_WORKERS,
    DEFAULT_OUTCOME_NUM_BINS,
    DEFAULT_SNR_BASELINE_LEVEL,
    DEFAULT_SNR_COMPARISON_LEVEL,
    DEFAULT_STEP_START,
    DEFAULT_STEP_STOP,
    DEFAULT_XTICK_LABEL_ROTATION,
    ConfigError,
    ExportSelection,
    PipelineConfig,
    PlotSettings,
    StepRange,
)
from .constants import schema as sch
from .metrics import MetricSelection, MetricSelectionError

# =====================================================================================================================
# Argument types
# =====================================================================================================================
# Converters here answer "is this string well-formed?". Whether the resulting
# value is *usable* is the config's question, not theirs.


def str2bool(value):
    """Accept the spellings of true/false that people actually type.

    KNOWN LIMITATION: rejection is signalled with ``assert``, so a typo raises
    AssertionError (a traceback) instead of argparse's usage message, and under
    ``python -O`` the assertions are stripped and the function returns None.
    Kept as-is here to hold this refactor behaviour-identical; replacing both
    with ``argparse.ArgumentTypeError`` is a one-line change that belongs in its
    own commit.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ('yes', 'true', 'enable', 'enabled', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'disable', 'disabled', 'f', 'n', '0'):
            return False
        else:
            assert False, '[ERROR] Input string is not a valid boolean'  # noqa: B011
    else:
        assert False, '[ERROR] Input value is not a string'  # noqa: B011


def _level_yaxis_range(value: str) -> tuple:
    """Parse one ``level:min,max`` entry into an ``(level, (min, max))`` pair."""
    if value.count(":") != 1:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be formatted as level:min,max (e.g. 0:0,2)."
        )

    level_part, range_part = value.split(":")
    bounds = range_part.split(",")

    if len(bounds) != 2:
        raise argparse.ArgumentTypeError(
            f"'{value}' must specify exactly two bounds as min,max; got {len(bounds)}."
        )

    try:
        level = int(level_part)
        min_val, max_val = (float(bound) for bound in bounds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' contains non-numeric values.") from exc

    return level, (min_val, max_val)


def _metric_label(value: str) -> tuple[str, str]:
    """Parse one ``metric=Display label`` entry into a ``(metric, label)`` pair."""
    metric, separator, label = value.partition("=")

    if not separator or not metric.strip() or not label.strip():
        raise argparse.ArgumentTypeError(
            f"'{value}' must be formatted as metric=label (e.g. degradation_1='Read-ahead')."
        )

    return metric.strip(), label.strip()


# =====================================================================================================================
# Parser
# =====================================================================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Flag names are snake_case to match the original tool, so existing scripts
    and shell history keep working.
    """

    parser = argparse.ArgumentParser(
        prog="da-pipeline",
        description="Data Analytics pipeline command-line interface (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # ---------------------------------------------------------------- inputs
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--data_analytics_config_fpath",
        required=True,
        type=str,
        help="Path of a json file mapping each condition to its extract (.h5) files.",
    )
    inputs.add_argument(
        "--response_name",
        required=True,
        type=str,
        help="Response name carried by the extract files, e.g. auc. Used to name "
             "the response column and to cross-check the inputs.",
    )
    inputs.add_argument(
        "--step_range",
        type=int,
        nargs=2,
        metavar=("STEP_FROM", "STEP_TO"),
        default=[DEFAULT_STEP_START, DEFAULT_STEP_STOP],
        help="Step window to extract responses from, 1-based and inclusive. "
             "Example: --step_range 1 100",
    )
    inputs.add_argument(
        "--group_ids",
        type=int,
        nargs="+",
        help="Restrict the analysis to these group IDs. Default: every group in the files.",
    )
    inputs.add_argument(
        "--group_labels",
        type=str,
        nargs="+",
        help="Replacement display labels, e.g. 'group_1' -> 'A'. Requires "
             "--group_ids: labels are matched to IDs by position, so the order "
             "must correspond.",
    )

    # -------------------------------------------------------------- analysis
    #
    # None of the metric arguments carries a `choices` list, and that is the
    # point: the valid names are whatever the extract files declare, which is
    # not known until a file is opened. They are resolved against the files at
    # the start of the run, where a name that matches nothing is an error naming
    # everything that was declared.
    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--metric_names",
        type=str,
        nargs="+",
        metavar="METRIC",
        help="Metrics to analyse, from those the extract files declare. "
             "Default: every metric declared.",
    )
    analysis.add_argument(
        "--outcome_metric",
        type=str,
        default=sch.DEFAULT_OUTCOME_METRIC,
        metavar="METRIC",
        help="The metric the run is organised around: it defines the equal-count "
             "bins every other metric is summarised against, and it is what the "
             "multivariable model predicts.",
    )
    analysis.add_argument(
        "--predictor_metric_names",
        type=str,
        nargs="+",
        metavar="METRIC",
        help="Metrics to treat as predictors of the outcome. Default: every "
             "analysed metric that is neither the outcome nor a component of it.",
    )
    analysis.add_argument(
        "--component_metric_names",
        type=str,
        nargs="+",
        metavar="METRIC",
        help="Metrics that restate the outcome rather than predict it; reported "
             "beside it, never modelled against it. Default: every analysed "
             "metric named after the outcome, e.g. Accuracy_error_1 beside Accuracy.",
    )
    analysis.add_argument(
        "--metric_labels",
        type=_metric_label,
        nargs="+",
        metavar="METRIC=LABEL",
        help="Display labels for metrics in figures and the report. E.g. "
             "--metric_labels degradation_1='Read-ahead' degradation_2='Lag'",
    )
    analysis.add_argument(
        "--snr_baseline_level",
        type=int,
        default=DEFAULT_SNR_BASELINE_LEVEL,
        help="Reference level the signal-to-noise ratio divides by -- the level "
             "at which no response is expected.",
    )
    analysis.add_argument(
        "--snr_comparison_level",
        type=int,
        default=DEFAULT_SNR_COMPARISON_LEVEL,
        help="Reference level the signal-to-noise ratio measures against the baseline.",
    )
    analysis.add_argument(
        "--outcome_num_bins",
        type=int,
        default=DEFAULT_OUTCOME_NUM_BINS,
        help="Number of equal-count (quantile) bins of the outcome metric.",
    )
    analysis.add_argument(
        "--calculate_effect_size",
        type=str2bool,
        default=True,
        help="Calculate effect size (eta-squared) across conditions.",
    )
    analysis.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of workers for multiprocessing.",
    )

    # ---------------------------------------------------------------- output
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output_folder_name_suffix",
        type=str,
        help="Suffix for the output folder name, e.g. 'MyAnalysis' produces "
             "Data_Analytics_MyAnalysis. Results are written beside the config file.",
    )
    output.add_argument(
        "--save_observation_counts_csv", type=str2bool, default=False,
        help="Save observation count data as a CSV file.",
    )
    output.add_argument(
        "--save_responses_csv", type=str2bool, default=False,
        help="Save response data as a CSV file.",
    )
    output.add_argument(
        "--save_signal_to_noise_ratios_csv", type=str2bool, default=True,
        help="Save signal-to-noise ratio data as a CSV file.",
    )
    output.add_argument(
        "--save_predictor_metrics_csv", type=str2bool, default=False,
        help="Save the predictor metrics as a CSV file.",
    )
    output.add_argument(
        "--save_outcome_metrics_csv", type=str2bool, default=False,
        help="Save the outcome metric and its components as a CSV file.",
    )
    output.add_argument(
        "--save_correlations_csv", type=str2bool, default=True,
        help="Save cross-variable Pearson/Spearman correlations as a CSV file.",
    )

    # -------------------------------------------------------------- plotting
    plotting = parser.add_argument_group("plotting")
    plotting.add_argument(
        "--colours",
        type=str,
        nargs="+",
        default=None,
        help="One colour per condition, in condition order. Accepts any matplotlib "
             "colour: hex (#1f77b4) or a named colour (tab:blue, crimson). "
             "Default: the colour cycle in analytics.mplstyle.",
    )
    plotting.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help="DPI for output images.",
    )
    plotting.add_argument(
        "--figure_width", type=float, default=DEFAULT_FIGURE_WIDTH, help="Width of the figures.",
    )
    plotting.add_argument(
        "--figure_height", type=float, default=DEFAULT_FIGURE_HEIGHT, help="Height of the figures.",
    )
    plotting.add_argument(
        "--show_outliers", type=str2bool, default=False, help="Show outliers in the boxplots.",
    )
    plotting.add_argument(
        "--boxplot_show_means", type=str2bool, default=True,
        help="Include mean markers in the boxplots.",
    )
    plotting.add_argument(
        "--xtick_label_rotation", type=int, default=DEFAULT_XTICK_LABEL_ROTATION,
        help="Rotation angle (in degrees) for x-axis tick labels.",
    )
    plotting.add_argument(
        "--show_condition_N_in_labels", type=str2bool, default=True,
        help="Append the run count '(2 runs)' to each condition's plot label, "
             "based on the number of file paths per condition. This counts runs, "
             "not rows; row counts appear alongside it as 'obs'. Affects plot "
             "labels only -- the raw condition names stay in the dataframes.",
    )
    plotting.add_argument(
        "--response_level_yaxis_ranges",
        type=_level_yaxis_range,
        nargs="+",
        metavar="LEVEL:MIN,MAX",
        help="Y-axis range per level. E.g. "
             "--response_level_yaxis_ranges 0:0,2 1:0,8 4:0,11",
    )
    plotting.add_argument(
        "--signal_to_noise_yaxis_range",
        type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Y-axis range for signal-to-noise plots. Example: "
             "--signal_to_noise_yaxis_range 0 20",
    )
    plotting.add_argument(
        "--predictor_metric_yaxis_range",
        type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Y-axis range for the boxplots of the predictor metrics.",
    )

    return parser


# =====================================================================================================================
# argv -> PipelineConfig
# =====================================================================================================================


def _level_ranges(parser: argparse.ArgumentParser, entries) -> dict | None:
    """Fold repeated ``level:min,max`` entries into a mapping, rejecting repeats.

    A repeated level is a CLI-shaped mistake -- the second entry
    would silently win -- so it is caught here rather than in the config, which
    only ever sees the assembled mapping.
    """
    if not entries:
        return None

    ranges: dict[int, tuple[float, float]] = {}
    for level, bounds in entries:
        if level in ranges:
            parser.error(
                f"argument --response_level_yaxis_ranges: "
                f"level {level} was specified more than once."
            )
        ranges[level] = bounds

    return ranges


def build_config(argv=None) -> PipelineConfig:
    """Parse ``argv`` and assemble a validated :class:`PipelineConfig`.

    Args:
        argv: Argument list. ``None`` reads ``sys.argv``.

    Returns:
        A config that has passed every semantic check.

    Raises:
        SystemExit: on malformed or unusable arguments, after argparse has
            printed a usage message. Configuration problems surface the same
            way as syntax problems, so the user sees one style of error.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        return PipelineConfig(
            config_path=Path(args.data_analytics_config_fpath),
            response_name=args.response_name,
            step_range=StepRange(*args.step_range),
            group_ids=args.group_ids,
            group_labels=args.group_labels,
            metrics=MetricSelection(
                names=args.metric_names,
                outcome=args.outcome_metric,
                predictors=args.predictor_metric_names,
                components=args.component_metric_names,
                labels=args.metric_labels or (),
            ),
            snr_baseline_level=args.snr_baseline_level,
            snr_comparison_level=args.snr_comparison_level,
            outcome_num_bins=args.outcome_num_bins,
            calculate_effect_size=args.calculate_effect_size,
            max_workers=args.max_workers,
            output_folder_name_suffix=args.output_folder_name_suffix,
            exports=ExportSelection(
                observation_counts=args.save_observation_counts_csv,
                responses=args.save_responses_csv,
                signal_to_noise=args.save_signal_to_noise_ratios_csv,
                predictor_metrics=args.save_predictor_metrics_csv,
                outcome_metrics=args.save_outcome_metrics_csv,
                correlations=args.save_correlations_csv,
            ),
            plots=PlotSettings(
                colours=args.colours,
                dpi=args.dpi,
                figure_width=args.figure_width,
                figure_height=args.figure_height,
                show_outliers=args.show_outliers,
                boxplot_show_means=args.boxplot_show_means,
                xtick_label_rotation=args.xtick_label_rotation,
                show_condition_counts=args.show_condition_N_in_labels,
                response_level_yaxis_ranges=_level_ranges(
                    parser, args.response_level_yaxis_ranges
                ),
                signal_to_noise_yaxis_range=args.signal_to_noise_yaxis_range,
                predictor_metric_yaxis_range=args.predictor_metric_yaxis_range,
            ),
        )
    except (ConfigError, MetricSelectionError) as exc:
        # The config found the problem; argparse renders it in its own idiom.
        # A metric selection can be incoherent on its own terms -- the same
        # metric named as both a predictor and a component, say -- and that is
        # answerable here. Whether the metrics *exist* is not: it is settled
        # against the files at the start of the run.
        parser.error(str(exc))
