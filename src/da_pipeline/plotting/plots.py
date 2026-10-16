
import concurrent.futures
import logging
import multiprocessing
from collections.abc import Callable, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ..config import PipelineConfig
from ..constants import columns as col
from ..constants import display as disp
from ..constants import outputs as out
from ..frames import format_count
from ..metrics import MetricRoles
from .figures import FigureConfig, figure

logger = logging.getLogger(__name__)


class DataPlotter:

    def __init__(self, config: PipelineConfig,
                 roles: MetricRoles,
                 condition_order: list,
                 condition_colours: dict,
                 condition_run_counts: dict | None = None,
                 condition_plot_labels: dict | None = None):
        """
        Args:
            config: The run's settings.
            roles: The run's resolved metric set. Every metric this class plots,
                and every label naming one, comes from here rather than from a
                constant -- including the word used for the bins, so a run
                organised around a metric called 'yield' says 'yield bin'.
            condition_order: Conditions in the order the config file listed
                them, which is the order every figure uses.
            condition_colours: {condition: colour}, built and validated once by
                the caller so a condition cannot change colour between figures.
            condition_run_counts: {condition: number of files}.
            condition_plot_labels: {condition: display label}.
        """

        self.config = config

        self.condition_order = condition_order
        self.condition_run_counts = condition_run_counts or {}
        self.condition_plot_labels = condition_plot_labels
        self.condition_colours = condition_colours

        self.response_name = config.response_name
        self.roles = roles
        self.outcome_metric = roles.outcome
        self.max_workers = config.max_workers

        # Axis text for the equal-count bins, built once from the run's outcome.
        self.bin_axis_label = disp.bin_axis_label(roles.outcome)

        # Size, DPI and destination travel together as one frozen object.
        self.fig_config = FigureConfig.from_config(config)

        plots = config.plots
        self.show_outliers = plots.show_outliers
        self.xtick_label_rotation = plots.xtick_label_rotation
        self.boxplot_show_means = plots.boxplot_show_means
        self.response_level_yaxis_ranges = plots.response_level_yaxis_ranges
        self.snr_yaxis_range = plots.signal_to_noise_yaxis_range
        self.predictor_metric_yaxis_range = plots.predictor_metric_yaxis_range

    # -----------------------------------------------------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def _boxplot_style(self) -> dict:
        """Marker/outlier styling shared by every boxplot, so the figures stay visually consistent."""

        return {
            'showmeans': self.boxplot_show_means,
            'meanprops': {'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'none'},
            'showfliers': self.show_outliers,
            'flierprops': {'marker': 'o', 'markersize': 3},
        }

    @property
    def _condition_label_map(self) -> dict:
        """
        Raw condition name -> display label (e.g. with the per-condition n appended).

        The 'condition' column keeps the raw names so the data stays clean; the counts live only in
        condition_plot_labels and are applied at display time. Accepts either a mapping or a sequence
        aligned with condition_order.
        """

        if not self.condition_plot_labels:
            return {}

        if isinstance(self.condition_plot_labels, dict):
            return self.condition_plot_labels

        return dict(zip(self.condition_order, self.condition_plot_labels))

    def _condition_display_name(self, condition: str) -> str:
        """Display label for a condition, falling back to the raw name."""

        return self._condition_label_map.get(condition, condition)

    def _condition_colour(self, condition: str) -> str:
        """
        The colour this condition carries in every figure of the run.

        A KeyError here means the colour map was built from a different set of
        conditions than the data contains, which would otherwise show up as two
        figures disagreeing about what a colour means.
        """

        try:
            return self.condition_colours[condition]
        except KeyError:
            raise KeyError(
                f"No colour assigned to condition {condition!r}. "
                f"Known: {sorted(self.condition_colours)}"
            ) from None

    def _figure(self, filename: str, *subdirs: str, **kwargs):
        """
        Create, style, save and close one figure.

            with self._figure('snr.png', out.SNR_SUBDIR) as (fig, ax):
                ...

        Delegates to plot_config.figure, which holds the theme for the whole
        block. Passing the run's FigureConfig here is the only place geometry
        and the output location are needed.
        """

        return figure(self.fig_config, filename, *subdirs, **kwargs)

    def _condition_tick_labels(self,
                               order: Sequence[str],
                               obs_counts: dict | None = None) -> list:
        """
        Two-line x tick label per condition:

            Condition_A
            (3 runs, 1,440 obs)

        Both counts sit under the box they describe, so neither has to be
        matched back to a condition by position. 'runs' are the files that
        make up the condition; 'obs' are the rows that went into the box.

        Args:
            order: conditions, in the order they appear on the axis.
            obs_counts: {condition: row count}. Omitted where a row count would
                not mean anything to the reader.
        """

        labels = []

        for condition in order:
            parts = []

            n_runs_in_condition = self.condition_run_counts.get(condition)
            if n_runs_in_condition is not None:
                parts.append(format_count(n_runs_in_condition, disp.UNIT_RUN_SINGULAR, disp.UNIT_RUN_PLURAL))

            if obs_counts is not None:
                parts.append(f'{obs_counts.get(condition, 0):,} {disp.UNIT_OBS}')

            labels.append(f'{condition}\n({", ".join(parts)})' if parts else condition)

        return labels

    @staticmethod
    def _rows_per_condition(data_df: pl.DataFrame) -> dict:
        """{condition: number of rows} for the frame being plotted."""

        return dict(data_df.group_by(col.CONDITION).len().iter_rows())

    def _style_bin_axis(self, ax, xlabel: str = '') -> None:
        """Apply the configured x-tick rotation and label to a bin/label axis."""

        ax.set_xlabel(xlabel)
        ax.tick_params(axis='x', labelrotation=self.xtick_label_rotation)

    @staticmethod
    def _annotate_bar_heights(ax, fmt: Callable[[float], str] = lambda v: f'{int(v)}',
                              fontsize: int = 7, rotation: int = 80) -> None:
        """Write each bar's height above the bar."""

        for patch in ax.patches:
            height = patch.get_height()

            if height is None or not np.isfinite(height):
                continue

            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                height,
                fmt(height),
                ha='center',
                va='bottom',
                fontsize=fontsize,
                rotation=rotation,
            )

    def _run_plot_tasks(self,
                        task_fn: Callable,
                        task_kwargs: Sequence[dict],
                        label: str,
                        parallel: bool = True) -> None:
        """
        Run a group of figure-generating calls, isolating failures so one bad figure cannot abort the run.

        Mirrors DataExtractor._run_parallel: a single place that owns the executor, the
        exception handling and the logging. Falls back to a plain loop when parallelism is disabled
        or pointless, which is also the escape hatch for the plots that hang under a process pool.

        Args:
            task_fn: Bound method producing one figure per call.
            task_kwargs: One kwargs dict per figure.
            label: Human-readable group name used in log lines.
            parallel: Set False to force sequential execution.
        """

        if not task_kwargs:
            logger.warning(f'{label}: no data to plot, skipped.')
            return

        if not parallel or not self.max_workers or self.max_workers < 2 or len(task_kwargs) == 1:
            for kwargs in task_kwargs:
                try:
                    task_fn(**kwargs)
                except Exception:
                    logger.exception(f'{label}: a figure failed and was skipped.')
            return

        # fork is the Linux default and deadlocks here: by this point the parent
        # holds thread pools and native library state from extraction, and a
        # forked child inherits the locks without the threads that would release
        # them. spawn is what Windows has always used, where this has never hung.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("spawn"),
        ) as executor:

            futures = [executor.submit(task_fn, **kwargs) for kwargs in task_kwargs]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    logger.exception(f'{label}: a figure failed and was skipped.')

    # -----------------------------------------------------------------------------------------------------------------
    # Outcome-bin helpers
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _bin_index_labels(data_condition_df: pl.DataFrame, bin_indices: list) -> list:
        """
        Tick text for each bin index within one condition: the outcome range.

        The binner emits the range as a display-only string that is not
        guaranteed unique, so it can never be the categorical key. Here it is
        only ever the text drawn under a position established by the index.

        Per-bin counts are deliberately not shown. Equal-count binning makes
        them identical give or take one, so a row of near-identical numbers
        under the ticks costs space and says nothing; the panel title carries
        the per-bin figure once instead.
        """

        ranges = dict(
            data_condition_df
            .select([col.OUTCOME_BIN_INDEX, col.OUTCOME_BIN])
            .unique()
            .iter_rows()
        )

        return [str(ranges.get(bin_index, bin_index)) for bin_index in bin_indices]

    @staticmethod
    def _warn_on_uneven_bins(data_condition_df: pl.DataFrame, condition: str, context: str) -> int:
        """
        Equal-count binning guarantees every bin in a condition holds the same number of
        observations give or take one. A wider spread than that means rows were lost somewhere between
        binning and plotting — typically the inner join in DataBinner._attach_outcome_bins dropping
        unmatched observations.

        This replaces the old per-bin counts barplot, which under equal-count binning would draw a row of
        identical bars. Returns the total number of points.

        Args:
            data_condition_df: Rows for one condition.
            condition: Condition name, for the log line.
            context: Metric/figure name, for the log line.

        Returns:
            Total number of points in the frame.
        """

        counts = (
            data_condition_df
            .group_by(col.OUTCOME_BIN_INDEX)
            .agg(pl.len().alias(col.N_POINTS_BIN))[col.N_POINTS_BIN]
        )

        if counts.len() and (counts.max() - counts.min()) > 1:
            logger.warning(
                f'{context} / {condition}: bin sizes range {counts.min()}-{counts.max()}, but equal-count '
                f'binning should vary by at most 1. Rows were likely dropped by an upstream join.'
            )

        return int(counts.sum()) if counts.len() else 0

    # -----------------------------------------------------------------------------------------------------------------
    # Per-run overlay
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _run_medians(data_df: pl.DataFrame, y_col: str) -> pl.DataFrame | None:
        """One median per (condition, run), or None if the frame has no run column.

        Returning None rather than raising keeps the overlay optional: a caller
        that trims the run column out of its plotting frame still gets a figure,
        just without the replicate dots.
        """
        if col.RUN_ID not in data_df.columns:
            return None

        return (
            data_df
            .group_by([col.CONDITION, col.RUN_ID])
            .agg(pl.col(y_col).median().alias(y_col))
        )

    def _overlay_run_medians(self, ax, run_medians, y_col: str, order: list) -> None:
        """Draw one marker per run over the boxes, and say what it is."""

        if run_medians is None or run_medians.height == 0:
            return

        sns.stripplot(
            ax=ax,
            data=run_medians,
            x=col.CONDITION,
            y=y_col,
            order=order,
            jitter=disp.RUN_MARKER_JITTER,
            size=disp.RUN_MARKER_SIZE,
            color=disp.RUN_MARKER_COLOUR,
            edgecolor=disp.RUN_MARKER_EDGE_COLOUR,
            linewidth=disp.RUN_MARKER_EDGE_WIDTH,
            legend=False,
        )

        # A legend, because a reader who does not know what the dots are will
        # otherwise read them as outliers. The mean marker is listed alongside
        # them when it is drawn: both are small black shapes sitting inside the
        # box, and leaving one unlabelled invites the reader to conflate a
        # summary of every observation with a single run.
        handles = [
            mlines.Line2D(
                [], [],
                marker='o', linestyle='none',
                markersize=disp.RUN_MARKER_SIZE,
                markerfacecolor=disp.RUN_MARKER_COLOUR,
                markeredgecolor=disp.RUN_MARKER_EDGE_COLOUR,
                label=disp.RUN_MARKER_LABEL,
            )
        ]

        if self.boxplot_show_means:
            handles.append(
                mlines.Line2D(
                    [], [],
                    marker='^', linestyle='none',
                    markersize=disp.RUN_MARKER_SIZE,
                    markerfacecolor='black', markeredgecolor='none',
                    label=disp.MEAN_MARKER_LABEL,
                )
            )

        ax.legend(handles=handles, loc='best', fontsize='small', framealpha=0.8)

    # -----------------------------------------------------------------------------------------------------------------
    # Shared figure builders
    # -----------------------------------------------------------------------------------------------------------------

    def _boxplot_by_condition(self,
                              data_df: pl.DataFrame,
                              y_col: str,
                              y_lim: tuple | None,
                              title: str,
                              output_subdir: str,
                              img_fname: str) -> None:
        """
        Single-panel boxplot: one box per condition, groups pooled within each,
        with one dot per run laid over it.

        Conditions are the x categories, so colour carries no extra information
        and there is no legend -- each box is labelled directly beneath itself.
        The colour map is still applied so a condition keeps the same colour it
        has in every other figure of the run.

        The box describes the observations; the dots are the run medians. That
        distinction is the point of the figure. A box drawn from the 1,440
        observations of a demo condition looks like overwhelming evidence, but
        those observations came from three runs and only the runs are
        independent -- so the dots, not the box, are what
        a comparison between conditions actually rests on. Overlaying them makes
        an apparent difference driven by a single outlying run visible instead
        of hidden inside a narrow box.

        The convention is a "SuperPlot":

            Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020).
            "SuperPlots: Communicating reproducibility and variability in cell
            biology." Journal of Cell Biology 219(6): e202001064.
            https://doi.org/10.1083/jcb.202001064

        Args:
            data_df: Data to plot. Must contain the condition column and y_col;
                if it also carries the run column, the per-run overlay is drawn.
            y_col: Column for the y-axis.
            y_lim: y-axis limits, or None to autoscale.
            title: Figure title.
            output_subdir: Sub-directory of the Data Analytics output dir.
            img_fname: File name of the saved image.
        """

        # Only the conditions actually present, in the configured order: seaborn
        # would otherwise leave an empty slot for a condition with no rows.
        order = [c for c in self.condition_order
                 if c in set(data_df[col.CONDITION].unique().to_list())]

        run_medians = self._run_medians(data_df, y_col)

        with self._figure(img_fname, output_subdir) as (fig, ax):
            sns.boxplot(
                ax=ax,
                data=data_df,
                x=col.CONDITION,
                y=y_col,
                hue=col.CONDITION,     # same column as x: colours the boxes...
                hue_order=order,
                legend=False,           # ...without producing a legend
                order=order,
                palette=self.condition_colours,
                **self._boxplot_style,
            )

            self._overlay_run_medians(ax, run_medians, y_col, order)

            if y_lim is not None:
                ax.set_ylim(y_lim)

            ax.set_title(title)

            # Counts live under each box. Two-line labels are left upright:
            # rotating them makes the second line hard to associate with a box.
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(
                self._condition_tick_labels(order, self._rows_per_condition(data_df)),
                rotation=0,
            )
            ax.set_xlabel('')

    def _faceted_boxplot_by_condition(self,
                                      data_df: pl.DataFrame,
                                      y_col: str,
                                      y_lim: tuple | None,
                                      title: str,
                                      img_fname: str,
                                      subdirs: tuple,
                                      context: str) -> None:
        """
        One stacked panel per condition, y-axis shared, x-axis = outcome bin index.

        This is the answer to equal-count binning: because the binner derives its edges per
        condition, bin 4 spans a different outcome range in each condition, so overlaying
        conditions on one categorical axis would put unlike things at the same x position. Giving each
        condition its own panel keeps every panel internally honest, and sharing the y-axis keeps the
        comparison across panels valid — which is the comparison that matters.

        The x-axis is the bin index (ordered, and the only unique bin identity); the range labels are
        applied per panel as tick text, so each panel shows its own edges.

        Args:
            data_df: Rows for one metric, all groups pooled.
            y_col: Metric on the y-axis.
            y_lim: y-axis limits, or None to autoscale.
            title: Figure suptitle.
            img_fname: File name of the saved image.
            subdirs: Output sub-directories under the Data Analytics output dir.
            context: Metric/figure name used in log lines.
        """

        present_conditions = set(data_df[col.CONDITION].unique().to_list())
        conditions = [c for c in self.condition_order if c in present_conditions]

        if not conditions:
            logger.warning(f'{context}: no conditions present, figure skipped.')
            return

        # Same bin positions in every panel, so the panels line up vertically.
        bin_indices = sorted(data_df[col.OUTCOME_BIN_INDEX].unique().to_list())

        # The bin index is an ordered category, not a quantity, but it is stored as an integer. Handing
        # numeric values (or their plain string forms, "1", "2", ...) to a categorical axis makes seaborn
        # warn that they are parsable as floats and guess at the intent. Prefixing the index makes the
        # category key unambiguously non-numeric; the tick text is overwritten with the range labels
        # afterwards, so 'bin1' never reaches the reader.
        data_df = data_df.with_columns(
            (pl.lit('bin') + pl.col(col.OUTCOME_BIN_INDEX).cast(pl.Utf8))
            .alias(col.OUTCOME_BIN_CATEGORY)
        )

        # Explicit order: 'bin10' sorts before 'bin2' lexicographically.
        bin_category_order = [f'bin{bin_index}' for bin_index in bin_indices]

        n_panels = len(conditions)

        # sharey: the whole point, values are comparable across panels.
        # sharex stays off: each condition has its own bin edges, so each panel
        # needs its own tick labels.
        with self._figure(
            img_fname,
            *subdirs,
            nrows=n_panels,
            sharey=True,
            height_scale=max(1.0, disp.FACET_PANEL_HEIGHT_FRACTION * n_panels),
            tight_layout_rect=[0, 0, 1, 0.99],   # headroom for the suptitle
        ) as (fig, axes):

            for panel_index, (ax, condition) in enumerate(zip(axes, conditions)):

                data_condition = data_df.filter(pl.col(col.CONDITION) == condition)

                n_points = self._warn_on_uneven_bins(data_condition, condition, context)

                colour = self._condition_colour(condition)

                sns.boxplot(
                    ax=ax,
                    data=data_condition,
                    x=col.OUTCOME_BIN_CATEGORY,
                    y=y_col,
                    hue=col.CONDITION,
                    hue_order=[condition],
                    palette=[colour] if colour is not None else None,
                    order=bin_category_order,
                    **self._boxplot_style,
                )

                if y_lim is not None:
                    ax.set_ylim(y_lim)

                # One condition per panel makes the legend redundant; the panel title carries it instead.
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                # Same vocabulary as the pooled boxplots: the display name
                # carries the run count, this adds the observation count and
                # the per-bin depth. Equal-count binning makes every bin the
                # same size give or take one, so one figure describes them all.
                n_bins = len(bin_indices)
                per_bin = f'{round(n_points / n_bins):,}' if n_bins else '0'

                ax.set_title(
                    f'{self._condition_display_name(condition)}'
                    f'  |  {n_points:,} {disp.UNIT_OBS}'
                    f'  |  {per_bin} {disp.UNIT_OBS}/bin',
                    fontsize=plt.rcParams['axes.labelsize'],
                )
                ax.set_ylabel(y_col)

                ax.set_xticks(range(len(bin_indices)))
                ax.set_xticklabels(
                    self._bin_index_labels(data_condition, bin_indices),
                    rotation=self.xtick_label_rotation,
                )

                # Only the bottom panel needs the axis title.
                ax.set_xlabel(self.bin_axis_label if panel_index == n_panels - 1 else '')

            fig.suptitle(title)

    def _plot_single_metric_by_outcome_bins(self,
                                           data_df: pl.DataFrame,
                                           metric_name: str,
                                           y_lim: tuple | None = None,
                                           title_prefix: str | None = None,
                                           fname_prefix: str | None = None) -> None:
        """Create one relationship figure for one metric, all groups pooled.

        ``data_df`` is already restricted to the three columns the figure needs.
        Keeping this worker at one-figure granularity lets the process pool
        distribute the SNR and predictor-metric relationship figures independently without
        sending the full joined dataframe to every worker.
        """

        title_prefix = title_prefix or f'{self.roles.label(metric_name)} vs {self.outcome_metric}'
        fname_prefix = fname_prefix or metric_name

        self._faceted_boxplot_by_condition(
            data_df=data_df,
            y_col=metric_name,
            y_lim=y_lim,
            title=f'{title_prefix}',
            img_fname=f'{fname_prefix}_by_{self.outcome_metric}_bins.png',
            subdirs=(out.RELATIONSHIPS_SUBDIR,),
            context=metric_name,
        )

    def _build_outcome_relationship_tasks(self,
                                         data_df: pl.DataFrame,
                                         metric_names: Sequence[str],
                                         y_limits: dict | None = None,
                                         title_prefixes: dict | None = None,
                                         fname_prefixes: dict | None = None) -> list:
        """
        Build one task per metric.

        Each worker receives only the condition, the bin identity/label and the
        metric being plotted, rather than the full SNR or predictor-metric dataframe.
        """

        y_limits = y_limits or {}
        title_prefixes = title_prefixes or {}
        fname_prefixes = fname_prefixes or {}
        tasks = []

        for metric_name in metric_names:
            plot_df = (
                data_df
                .select([
                    col.CONDITION,
                    col.OUTCOME_BIN_INDEX,
                    col.OUTCOME_BIN,
                    metric_name,
                ])
                .rechunk()
            )

            if plot_df.height == 0:
                logger.warning(f'{self.outcome_metric} relationships: no rows for {metric_name}, skipped.')
                continue

            tasks.append({
                'data_df': plot_df,
                'metric_name': metric_name,
                'y_lim': y_limits.get(metric_name),
                'title_prefix': title_prefixes.get(metric_name),
                'fname_prefix': fname_prefixes.get(metric_name),
            })

        return tasks

    def _plot_metrics_by_outcome_bins(self,
                                     data_df: pl.DataFrame,
                                     metric_names: Sequence[str],
                                     label: str,
                                     y_limits: dict | None = None,
                                     title_prefixes: dict | None = None,
                                     fname_prefixes: dict | None = None) -> None:
        """Plot all requested metric relationship figures in parallel."""

        tasks = self._build_outcome_relationship_tasks(
            data_df=data_df,
            metric_names=metric_names,
            y_limits=y_limits,
            title_prefixes=title_prefixes,
            fname_prefixes=fname_prefixes,
        )

        self._run_plot_tasks(
            self._plot_single_metric_by_outcome_bins,
            task_kwargs=tasks,
            label=label,
        )

    def _plot_metric_by_outcome_bins(self,
                                    data_df: pl.DataFrame,
                                    metric_name: str,
                                    y_lim: tuple | None = None,
                                    title_prefix: str | None = None,
                                    fname_prefix: str | None = None) -> None:
        """Convenience wrapper for plotting a single metric."""

        self._plot_metrics_by_outcome_bins(
            data_df=data_df,
            metric_names=[metric_name],
            label=f'{metric_name} by {self.outcome_metric} bins',
            y_limits={metric_name: y_lim},
            title_prefixes={metric_name: title_prefix},
            fname_prefixes={metric_name: fname_prefix},
        )


# =====================================================================================================================
# Plotting Methods
# =====================================================================================================================

    # -----------------------------------------------------------------------------------------------------------------
    # Observation counts
    # -----------------------------------------------------------------------------------------------------------------

    def _barplot_by_condition(self, data_df: pl.DataFrame, y_col: str, title: str, ylabel: str,
                              img_fname: str, output_subdir: str, estimator: str) -> None:
        """
        One bar per condition.

        process_observation_count() reduces to one row per (condition, group), so
        pooling groups means those rows must be combined and an estimator is
        required. It differs by statistic and cannot be inferred: totals sum
        across groups, medians take the median of the per-group medians. Note
        that the latter is a median of medians, not the median of the
        underlying observations.

        Args:
            estimator: seaborn aggregation over the per-group rows, 'sum' or 'median'.
        """

        order = [c for c in self.condition_order
                 if c in set(data_df[col.CONDITION].unique().to_list())]

        with self._figure(img_fname, output_subdir) as (fig, ax):
            sns.barplot(
                data=data_df,
                x=col.CONDITION,
                y=y_col,
                hue=col.CONDITION,   # same column as x: colours the bars...
                hue_order=order,
                legend=False,         # ...without producing a legend
                order=order,
                palette=self.condition_colours,
                estimator=estimator,
                ax=ax,
                errorbar=None,
            )

            ax.set_title(title)
            ax.set_ylabel(ylabel)

            # No obs count here: an observation-count row is one group, so the number
            # would describe the group panel rather than the bar's sample size.
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(
                self._condition_tick_labels(order),
                rotation=0,
            )
            ax.set_xlabel('')

    def plot_single_observation_count_stat(self, data_df: pl.DataFrame, stat_col: str) -> None:
        """
        Plots one observation-count statistic per condition, groups pooled.

        Args:
            data_df: Output of process_observation_count(), containing the aggregated count columns.
            stat_col: Column to plot; a key of OBSERVATION_COUNT_STATS_REGISTRY.
        """

        stat_label = disp.OBSERVATION_COUNT_STATS_REGISTRY[stat_col]

        # Totals add up across groups; medians do not.
        estimator = 'sum' if stat_col == col.OBSERVATION_COUNT_TOTAL else 'median'

        self._barplot_by_condition(
            data_df=data_df,
            y_col=stat_col,
            title=f'{stat_label} Number of Observations by Condition',
            ylabel=f'{stat_label} Number of Observations',
            img_fname=f'{stat_label.lower()}_number_of_observations_barplot.png',
            output_subdir=out.OBSERVATION_COUNT_SUBDIR,
            estimator=estimator,
        )

    def plot_observation_counts(self, data_df: pl.DataFrame) -> None:

        logger.info('- Observation counts')

        tasks = [
            {
                'data_df': data_df.select([
                    col.CONDITION,
                    stat_col,
                ]),
                'stat_col': stat_col,
            }
            for stat_col in disp.OBSERVATION_COUNT_STATS_REGISTRY
        ]

        self._run_plot_tasks(
            self.plot_single_observation_count_stat,
            task_kwargs=tasks,
            label='Observation counts',
            # Observation counts are already aggregated to one row per condition and group,
            # so their DataFrame is probably small. Multiprocessing may actually cost
            # more than it saves when producing only two figures.
            parallel=False,
        )

    # -----------------------------------------------------------------------------------------------------------------
    # Responses
    # -----------------------------------------------------------------------------------------------------------------

    def plot_responses_single_level(self, level: int, data_masked_df: pl.DataFrame) -> None:
        """
        Plots a boxplot for a single reference level.

        Args:
            level: Reference level.
            data_masked_df: Rows for this level only.
        """

        y_lim = None
        if self.response_level_yaxis_ranges:
            y_lim = self.response_level_yaxis_ranges.get(level)

        self._boxplot_by_condition(
            data_df=data_masked_df,
            y_col=self.response_name,
            y_lim=y_lim,
            title=f'{self.response_name} | level {level}',
            output_subdir=out.RESPONSES_SUBDIR,
            img_fname=f'level_{level}_{self.response_name}.png',
        )

    def plot_responses_by_level(self, data_df: pl.DataFrame) -> None:

        logger.info('- Responses')

        tasks = []

        for level_key, data_masked_df in data_df.group_by(
                col.LEVEL,
                maintain_order=True,
        ):
            level = level_key[0]

            plot_df = (
                data_masked_df
                .select([
                    col.RUN_ID,
                    col.CONDITION,
                    self.response_name,
                ])
                .rechunk()
            )

            tasks.append({
                'level': level,
                'data_masked_df': plot_df,
            })

        self._run_plot_tasks(
            self.plot_responses_single_level,
            task_kwargs=tasks,
            label='Responses by level',
        )

    # -----------------------------------------------------------------------------------------------------------------
    # Signal-to-noise
    # -----------------------------------------------------------------------------------------------------------------

    def plot_signal_to_noise_by_condition(self, data_df: pl.DataFrame) -> None:
        """
        Plots signal-to-noise (level 1/level 0) per condition, groups pooled.

        Args:
            data_df: Data containing signal-to-noise ratios to plot.
        """

        logger.info('- Signal-to-noise')

        self._boxplot_by_condition(
            data_df=data_df,
            y_col=col.SIGNAL_TO_NOISE_RATIO,
            y_lim=self.snr_yaxis_range,
            title=f'{self.response_name} | signal-to-noise',
            output_subdir=out.SNR_SUBDIR,
            img_fname=f'snr_by_condition_{self.response_name}_boxplot.png'
        )

    def plot_signal_to_noise_by_outcome_bins(self, data_df: pl.DataFrame) -> None:
        """
        Plots signal-to-noise against the equal-count outcome bins, one panel per condition.

        Args:
            data_df: Data containing signal-to-noise ratios joined to outcome bins.
        """

        logger.info(f'- Relationships: snr by {self.outcome_metric} bins')

        self._plot_metric_by_outcome_bins(
            data_df=data_df,
            metric_name=col.SIGNAL_TO_NOISE_RATIO,
            y_lim=self.snr_yaxis_range,
            title_prefix=f'{self.response_name} signal-to-noise (SNR) vs {self.outcome_metric}',
            fname_prefix=f'snr_{self.response_name}',
        )

    # -----------------------------------------------------------------------------------------------------------------
    # Predictor metrics
    # -----------------------------------------------------------------------------------------------------------------

    def plot_single_predictor_metric(self, data_df: pl.DataFrame, metric_name: str) -> None:
        """
        Plots a single predictor metric per condition, groups pooled.

        Args:
            data_df: Data containing the metric to plot.
            metric_name: Name of the predictor metric to plot.
        """

        self._boxplot_by_condition(
            data_df=data_df,
            y_col=metric_name,
            y_lim=self.predictor_metric_yaxis_range,
            title=f'{self.roles.label(metric_name)} | {self.response_name}',
            output_subdir=out.PREDICTOR_METRICS_SUBDIR,
            img_fname=f'{metric_name}_boxplot.png',
        )

    def plot_predictor_metrics(self, data_df: pl.DataFrame) -> None:

        logger.info('- Predictor metrics')

        tasks = []

        for metric_name in self.roles.predictors:
            plot_df = (
                data_df
                .select([
                    col.GROUP_LABEL,
                    col.RUN_ID,
                    col.CONDITION,
                    metric_name,
                ])
                .rechunk()
            )

            tasks.append({
                'data_df': plot_df,
                'metric_name': metric_name,
            })

        if not tasks:
            logger.info('  no predictor metrics in this run, skipped.')
            return

        self._run_plot_tasks(
            self.plot_single_predictor_metric,
            task_kwargs=tasks,
            label='Predictor metrics',
        )

    def plot_component_metrics_by_outcome_bins(self, data_df: pl.DataFrame) -> None:
        """
        Plots the outcome's component metrics against its equal-count bins, one
        figure per metric and one panel per condition.

        The outcome itself is not plotted: it defines the bins, so it would only
        redraw the bin edges.

        Args:
            data_df: Data containing the outcome metrics joined to the bins.
        """

        logger.info(f'- Relationships: component metrics by {self.outcome_metric} bins')

        metric_names = list(self.roles.components)

        if not metric_names:
            logger.info('  no component metrics in this run, skipped.')
            return

        self._plot_metrics_by_outcome_bins(
            data_df=data_df,
            metric_names=metric_names,
            label=f'Component metrics by {self.outcome_metric} bins',
            y_limits=dict.fromkeys(metric_names, (0, 1.02)),
        )

    def plot_predictor_metrics_by_outcome_bins(self, data_df: pl.DataFrame) -> None:
        """
        Plots the predictor metrics against the outcome's equal-count bins, one
        figure per metric and one panel per condition.

        Args:
            data_df: Data containing the predictor metrics joined to the bins.
        """

        logger.info(f'- Relationships: predictor metrics by {self.outcome_metric} bins')

        metric_names = list(self.roles.predictors)

        if not metric_names:
            logger.info('  no predictor metrics in this run, skipped.')
            return

        self._plot_metrics_by_outcome_bins(
            data_df=data_df,
            metric_names=metric_names,
            label=f'Predictor metrics by {self.outcome_metric} bins',
            y_limits=dict.fromkeys(metric_names, self.predictor_metric_yaxis_range),
        )

    # -----------------------------------------------------------------------------------------------------------------
    # Outcome metric and its components
    # -----------------------------------------------------------------------------------------------------------------

    def plot_single_outcome_metric(self, data_df: pl.DataFrame, metric_name: str) -> None:
        """
        Plots a single outcome-family metric as a boxplot.

        Args:
            data_df: Data containing the metric to plot.
            metric_name: Name of the metric to plot.
        """

        self._boxplot_by_condition(
            data_df=data_df,
            y_col=metric_name,
            y_lim=(0, 1.02),
            title=f'{self.roles.label(metric_name)} | {self.response_name}',
            output_subdir=out.OUTCOME_METRICS_SUBDIR,
            img_fname=f'{metric_name}_boxplot.png',
        )

    def plot_outcome_metrics(self, data_df: pl.DataFrame) -> None:

        logger.info('- Outcome metrics')

        tasks = []

        for metric_name in self.roles.outcome_metrics:
            plot_df = (
                data_df
                .select([
                    col.GROUP_LABEL,
                    col.RUN_ID,
                    col.CONDITION,
                    metric_name,
                ])
                .rechunk()
            )

            tasks.append({
                'data_df': plot_df,
                'metric_name': metric_name,
            })

        self._run_plot_tasks(
            self.plot_single_outcome_metric,
            task_kwargs=tasks,
            label='Outcome metrics',
        )

    # -----------------------------------------------------------------------------------------------------------------
    # Effect size
    # -----------------------------------------------------------------------------------------------------------------

    def _plot_effect_size_bar_chart(self, labels: list, effect_size_data: list, title: str,
                                    xlabel: str, ylabel: str, rotation: int, output_fname: str,
                                    bar_sublabels: Sequence[str] | None = None,
                                    error_bars: np.ndarray | None = None) -> None:
        """
        Annotated bar chart of effect-size values.

        Args:
            labels: x-axis labels (levels).
            effect_size_data: Effect-size values, one per label.
            title: Figure title.
            xlabel: x-axis label.
            ylabel: y-axis label.
            rotation: Rotation (degrees) for the x-tick labels.
            output_fname: File name of the saved image.
        """

        if len(labels) != len(effect_size_data):
            raise ValueError(
                f'{output_fname}: {len(labels)} labels but {len(effect_size_data)} effect-size values.'
            )

        with self._figure(output_fname, out.EFFECT_SIZE_SUBDIR) as (fig, ax):
            # Error bars, where an interval was computable. At a handful of runs
            # per condition they are wide, which is the point: a bar drawn
            # without them reads as far more settled than the design supports.
            ax.bar(labels, effect_size_data, yerr=error_bars, capsize=4,
                   error_kw={'ecolor': 'black', 'elinewidth': 1.2})
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=rotation)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)

            self._annotate_bar_heights(ax, fmt=lambda v: f'{v:.4f}', fontsize=8, rotation=0)

            if bar_sublabels:
                self._annotate_bar_sublabels(ax, bar_sublabels)

    @staticmethod
    def _annotate_bar_sublabels(ax, sublabels: Sequence[str], fontsize: int = 7) -> None:
        """Write a second, smaller line just inside the top of each bar."""

        for patch, text in zip(ax.patches, sublabels):
            height = patch.get_height()

            if not text or height is None or not np.isfinite(height):
                continue

            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                height,
                text,
                ha='center',
                va='top',
                fontsize=fontsize,
                color='white' if height > 0.08 else 'black',
            )

    # -----------------------------------------------------------------------------------------------------------------
    # Variance decomposition
    # -----------------------------------------------------------------------------------------------------------------

    def plot_variance_decomposition(self, data_df: pl.DataFrame) -> None:
        """Horizontal stacked bars: where each metric's spread comes from.

        One bar per metric, split into condition / run-within-condition /
        observation-within-run. The comparison the figure exists for is the first two
        segments against each other: run-to-run variation as large as the
        condition band means two runs of one condition already differ by about
        as much as the conditions do.

        Stated as evidence rather than prediction, deliberately. This is a
        decomposition of the sums of squares actually observed, not an estimate
        of random-effect variance components, so it can say the present evidence
        for a condition difference is weak -- it cannot forecast what a future
        run will show.
        """

        logger.info('- Variance decomposition')

        if data_df.is_empty():
            logger.warning('Variance decomposition: nothing to plot, skipped.')
            return

        # Largest condition share at the top: the metrics a condition comparison
        # is best supported on come first.
        ordered = data_df.sort(
            [col.PCT_CONDITION, col.PCT_RUN_WITHIN_CONDITION],
            descending=True,
            nulls_last=True,
        )

        # The metric name alone. Every bar is a share of the same denominator,
        # so nothing else has to be qualified on the axis.
        labels = ordered[col.METRIC].to_list()

        segments = [
            (col.PCT_CONDITION, disp.VARIANCE_CONDITION_LABEL, disp.VARIANCE_CONDITION_COLOUR),
            (col.PCT_RUN_WITHIN_CONDITION, disp.VARIANCE_RUN_LABEL, disp.VARIANCE_RUN_COLOUR),
            (col.PCT_OBSERVATION_WITHIN_RUN, disp.VARIANCE_OBSERVATION_LABEL, disp.VARIANCE_OBSERVATION_COLOUR),
        ]

        positions = np.arange(len(labels))
        height_scale = max(1.0, disp.VARIANCE_ROW_HEIGHT_FRACTION * len(labels))

        with self._figure(
            'variance_decomposition.png',
            out.VARIANCE_SUBDIR,
            height_scale=height_scale,
        ) as (fig, ax):

            left = np.zeros(len(labels))

            for column, label, colour in segments:
                # A missing condition share is 0 width, not a gap in the bar.
                widths = np.array(
                    [0.0 if v is None else float(v) for v in ordered[column].to_list()]
                )
                ax.barh(positions, widths, left=left, label=label, color=colour)

                for y, (width, start) in enumerate(zip(widths, left)):
                    if width >= disp.VARIANCE_ANNOTATION_MIN_PCT:
                        ax.text(
                            start + width / 2.0, y, f'{width:.0f}%',
                            ha='center', va='center', fontsize=7,
                            color='white' if colour != disp.VARIANCE_OBSERVATION_COLOUR else 'black',
                        )

                left += widths

            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel('share of variance (%)')
            ax.set_title(
                'Where each metric\'s variance comes from\n'
                'run-to-run variation as large as the condition band beside it '
                'means weaker evidence of reproducibility across runs'
            )
            ax.legend(loc='lower right', fontsize='small', framealpha=0.9)

    # -----------------------------------------------------------------------------------------------------------------
    # Correlations: the three views, and where they disagree
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _join_correlation_views(
        pooled_df: pl.DataFrame, within_run_df: pl.DataFrame
    ) -> pl.DataFrame:
        """One row per (condition, pair) carrying both coefficients.

        Inner join, then drop rows where either view is missing: a pair no run
        could correlate has nothing to compare, and plotting it as a point at
        zero would invent a disagreement that was never measured.
        """

        keys = [col.CONDITION, col.CORR_VAR_X, col.CORR_VAR_Y]

        return (
            pooled_df
            .select(keys + [col.CORR_PEARSON_R])
            .join(
                within_run_df.select(keys + [
                    col.CORR_PEARSON_R_MEDIAN,
                    col.CORR_PEARSON_R_MIN,
                    col.CORR_PEARSON_R_MAX,
                    col.CORR_PEARSON_SAME_SIGN,
                    col.CORR_N_RUNS,
                ]),
                on=keys,
                how="inner",
            )
            .drop_nulls([col.CORR_PEARSON_R, col.CORR_PEARSON_R_MEDIAN])
            .with_columns(
                (pl.col(col.CORR_PEARSON_R) - pl.col(col.CORR_PEARSON_R_MEDIAN))
                .abs().alias("_gap"),
                (
                    pl.col(col.CORR_PEARSON_R).sign()
                    != pl.col(col.CORR_PEARSON_R_MEDIAN).sign()
                ).alias("_reversed"),
            )
        )

    @staticmethod
    def _pair_labels(df: pl.DataFrame) -> list[str]:
        """'degradation_1  /  Accuracy' -- both metrics, unabbreviated."""
        return [
            f'{x}{disp.PAIR_SEPARATOR}{y}'
            for x, y in zip(df[col.CORR_VAR_X].to_list(), df[col.CORR_VAR_Y].to_list())
        ]

    def plot_correlation_views(
        self, pooled_df: pl.DataFrame, within_run_df: pl.DataFrame
    ) -> None:
        """The headline scatter, then one detail figure per condition."""

        if pooled_df is None or within_run_df is None:
            return
        if pooled_df.is_empty() or within_run_df.is_empty():
            logger.warning('Correlation views: nothing to plot, skipped.')
            return

        joined = self._join_correlation_views(pooled_df, within_run_df)

        if joined.is_empty():
            logger.warning('Correlation views: no pair has both coefficients, skipped.')
            return

        self._plot_pooled_against_within_run(joined)

        for condition in self.condition_order:
            subset = joined.filter(pl.col(col.CONDITION) == condition)
            if subset.height:
                self._plot_condition_correlation_detail(condition, subset)

    def _plot_pooled_against_within_run(self, joined: pl.DataFrame) -> None:
        """Every pair as one point: pooled on x, within-run on y.

        The diagonal is the claim "pooling did not distort this relationship".
        A point far below it is a coefficient the pooled table inflates; a point
        in either shaded quadrant is one the two views disagree about in
        *direction*, which is the failure mode that matters most -- a reader of
        the pooled table alone would draw the opposite conclusion.

        Points hugging the x-axis are the starkest case of all: a strong pooled
        coefficient whose within-run counterpart is nothing. That is the
        two-cloud picture -- the better run was better on both measures, read as
        though it were a statement about observations.
        """

        logger.info('- Correlations: pooled against within-run')

        reversed_count = int(joined["_reversed"].sum())

        with self._figure(
            'pooled_vs_within_run.png',
            out.CORRELATIONS_SUBDIR,
            # Equal aspect makes the axes box square, so extra height is what
            # lets the square grow into the figure's width instead of leaving
            # it as margin.
            height_scale=1.5,
        ) as (fig, ax):

            # The two quadrants where the coefficients differ in sign. Drawn
            # first so every marker sits on top of them.
            for x0, y0 in ((-1.0, 0.0), (0.0, -1.0)):
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0), 1.0, 1.0,
                        facecolor=disp.SIGN_REVERSAL_SHADE, edgecolor='none', zorder=0,
                    )
                )

            ax.axhline(0.0, color='0.75', linewidth=0.8, zorder=1)
            ax.axvline(0.0, color='0.75', linewidth=0.8, zorder=1)
            ax.plot(
                [-1, 1], [-1, 1],
                linestyle='--', color='0.45', linewidth=1.0, zorder=2,
                label='pooling did not distort',
            )

            for condition in self.condition_order:
                subset = joined.filter(pl.col(col.CONDITION) == condition)
                if not subset.height:
                    continue
                ax.scatter(
                    subset[col.CORR_PEARSON_R].to_numpy(),
                    subset[col.CORR_PEARSON_R_MEDIAN].to_numpy(),
                    s=34,
                    color=self.condition_colours[condition],
                    edgecolor='white',
                    linewidth=0.5,
                    zorder=3,
                    label=condition,
                )

            # Ringed individually, not left to the shading alone. A reversal is
            # usually a strong pooled coefficient against a within-run one near
            # zero, so the reversed points hug the axes rather than filling the
            # quadrants -- shading a large empty area understates how many
            # there are.
            flipped = joined.filter(pl.col("_reversed"))
            if flipped.height:
                ax.scatter(
                    flipped[col.CORR_PEARSON_R].to_numpy(),
                    flipped[col.CORR_PEARSON_R_MEDIAN].to_numpy(),
                    s=110,
                    facecolor='none',
                    edgecolor=disp.SIGN_REVERSAL_COLOUR,
                    linewidth=1.1,
                    zorder=4,
                    label=f'opposite signs ({reversed_count})',
                )

            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('pooled r  (every observation of the condition at once)')
            ax.set_ylabel('within-run r  (median across the runs)')
            ax.set_title(
                'Does pooling change the relationship?\n'
                f'{joined.height} metric pairs; {reversed_count} come out with '
                f'opposite signs, and the further a point sits below the diagonal '
                f'the more pooling inflated it'
            )
            ax.legend(loc='lower right', fontsize='small', framealpha=0.95)

    def _plot_condition_correlation_detail(
        self, condition: str, subset: pl.DataFrame
    ) -> None:
        """One condition's pairs, pooled and within-run joined by a line.

        Ordered by how far apart the two views land, so the pairs a reader
        should not trust the pooled figure for come first. The bar behind each
        row is the range of the per-run coefficients: a wide one means the
        relationship itself varies from run to run, which is a different
        caution from the two views disagreeing.
        """

        ordered = subset.sort("_gap", descending=True)
        positions = np.arange(ordered.height)

        pooled = ordered[col.CORR_PEARSON_R].to_numpy()
        within = ordered[col.CORR_PEARSON_R_MEDIAN].to_numpy()
        lows = ordered[col.CORR_PEARSON_R_MIN].to_numpy()
        highs = ordered[col.CORR_PEARSON_R_MAX].to_numpy()
        flipped = ordered["_reversed"].to_list()

        with self._figure(
            f'{condition}_pooled_vs_within_run.png',
            out.CORRELATIONS_SUBDIR,
            height_scale=max(1.0, disp.CORRELATION_ROW_HEIGHT_FRACTION * ordered.height),
        ) as (fig, ax):

            ax.axvline(0.0, color='0.55', linewidth=1.0, zorder=1)

            for y, (low, high) in enumerate(zip(lows, highs)):
                if low is not None and high is not None and np.isfinite(low):
                    ax.hlines(
                        y, low, high,
                        color='0.85', linewidth=5.0, zorder=2,
                        label=disp.WITHIN_RUN_RANGE_LABEL if y == 0 else None,
                    )

            for y, (a, b, is_flipped) in enumerate(zip(pooled, within, flipped)):
                ax.plot(
                    [a, b], [y, y],
                    color=disp.SIGN_REVERSAL_COLOUR if is_flipped else '0.55',
                    linewidth=1.6 if is_flipped else 1.0,
                    zorder=3,
                )

            ax.scatter(
                pooled, positions,
                s=30, facecolor='white', edgecolor=disp.POOLED_MARKER_COLOUR,
                linewidth=1.2, zorder=4, label=disp.POOLED_LABEL,
            )
            ax.scatter(
                within, positions,
                s=30, color=disp.WITHIN_RUN_MARKER_COLOUR,
                zorder=5, label=disp.WITHIN_RUN_LABEL,
            )

            ax.set_yticks(positions)
            ax.set_yticklabels(self._pair_labels(ordered), fontsize=6.5)
            ax.invert_yaxis()
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel('Pearson r')

            n_flipped = sum(flipped)
            n_runs = ordered[col.CORR_N_RUNS].max()
            ax.set_title(
                f'{condition}: what pooling does to each relationship\n'
                f'{ordered.height} pairs over {n_runs} runs, widest disagreement first'
                + (f'; {n_flipped} reverse sign (red)' if n_flipped else '')
            )
            ax.legend(loc='lower right', fontsize='x-small', framealpha=0.9)

    def plot_between_run_correlations(self, data_df: pl.DataFrame) -> None:
        """Do the runs that scored better on one metric score better on the other?

        The only view whose unit of analysis is the run, and the only one that
        can say whether whatever separates one run from another travels through
        a given metric. Kept to its own figure because it carries a caveat the
        other two do not: the runs span every condition, so a coefficient here
        is confounded with the condition effect. Computing it inside a
        condition would remove the confound and leave only as many points as that
        condition has runs, which is worse, so it is reported across all runs with the confound stated.
        """

        logger.info('- Correlations: between runs')

        if data_df is None or data_df.is_empty():
            logger.warning('Between-run correlations: nothing to plot, skipped.')
            return

        ordered = (
            data_df
            .drop_nulls(col.CORR_PEARSON_R)
            .sort(col.CORR_PEARSON_R, descending=True)
        )

        if ordered.is_empty():
            logger.warning('Between-run correlations: no usable pair, skipped.')
            return

        values = ordered[col.CORR_PEARSON_R].to_numpy()
        positions = np.arange(ordered.height)
        n_runs = ordered[col.CORR_N_RUNS].max()

        with self._figure(
            'between_run.png',
            out.CORRELATIONS_SUBDIR,
            height_scale=max(1.0, disp.CORRELATION_ROW_HEIGHT_FRACTION * ordered.height),
        ) as (fig, ax):

            ax.axvline(0.0, color='0.55', linewidth=1.0, zorder=1)
            ax.hlines(
                positions, 0.0, values,
                color=disp.WITHIN_RUN_MARKER_COLOUR, linewidth=1.2, zorder=2,
            )
            ax.scatter(
                values, positions,
                s=28, color=disp.WITHIN_RUN_MARKER_COLOUR, zorder=3,
            )

            ax.set_yticks(positions)
            ax.set_yticklabels(self._pair_labels(ordered), fontsize=6.5)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel('Pearson r between run medians')
            ax.set_title(
                'Do the runs that scored better on one metric score better on the other?\n'
                f'one median per run, correlated across all {n_runs} runs '
                f'-- confounded with condition, since the runs span every condition'
            )

    # -----------------------------------------------------------------------------------------------------------------
    # Multivariable outcome model
    # -----------------------------------------------------------------------------------------------------------------

    def plot_model_coefficients(self, data_df: pl.DataFrame) -> None:
        """Forest plot: each predictor's standardised coefficient across runs.

        One row per predictor -- the median coefficient over the per-run fits,
        with its bootstrap interval. The line at zero is the whole figure: a
        predictor whose interval crosses it did not hold a consistent direction
        once the other predictors were held constant, however strong its
        one-at-a-time correlation looked.

        Because both sides were standardised within each run, the four
        coefficients are on one scale and can be read against each other
        directly: "standard deviations of the outcome per standard deviation of
        predictor".
        """

        logger.info(f'- {self.outcome_metric} model coefficients')

        if data_df is None or data_df.is_empty():
            logger.warning(f'{self.outcome_metric} model: nothing to plot, skipped.')
            return

        # Strongest effect at the top, sign ignored -- a large negative
        # coefficient is as much a finding as a large positive one.
        ordered = data_df.sort(
            pl.col(col.COEFFICIENT_MEDIAN).abs(), descending=True, nulls_last=True
        )

        medians = np.array(ordered[col.COEFFICIENT_MEDIAN].to_list(), dtype=float)
        positions = np.arange(len(medians))
        errors = self._coefficient_error_bars(ordered, medians)

        labels = [
            f'{predictor}\n({same_sign}/{n_runs} runs agree, VIF {vif:.1f})'
            if vif is not None else f'{predictor}\n({same_sign}/{n_runs} runs agree)'
            for predictor, same_sign, n_runs, vif in zip(
                ordered[col.PREDICTOR].to_list(),
                ordered[col.COEFFICIENT_SAME_SIGN].to_list(),
                ordered[col.N_RUNS_MODELLED].to_list(),
                ordered[col.VIF_MEDIAN].to_list(),
            )
        ]

        with self._figure(
            f'{self.outcome_metric}_model_coefficients.png',
            out.MODEL_SUBDIR,
            height_scale=max(0.5, disp.MODEL_ROW_HEIGHT_FRACTION * len(medians)),
        ) as (fig, ax):

            ax.axvline(0.0, color='0.4', linewidth=1.0, zorder=1)
            ax.errorbar(
                medians, positions,
                xerr=errors,
                fmt='o',
                color=disp.VARIANCE_CONDITION_COLOUR,
                ecolor='0.35',
                elinewidth=1.2,
                capsize=4,
                markersize=7,
                zorder=3,
            )

            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(
                'standardised coefficient\n'
                f'(SD of {self.outcome_metric} per SD of predictor, other predictors held constant)'
            )
            ax.set_title(self._model_title(ordered))

    def _model_title(self, df: pl.DataFrame) -> str:
        """Title plus the two numbers that decide whether to believe the figure.

        The out-of-sample fit goes in the title rather than a footnote because
        it governs everything below it: coefficients that do not transfer to a
        held-out run describe the runs that were fitted, not the process.
        """

        n_runs = df[col.N_RUNS_MODELLED][0]
        in_sample = df[col.R_SQUARED_IN_SAMPLE_MEDIAN][0]
        held_out = df[col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN][0]
        predicted = df[col.N_RUNS_PREDICTED][0]

        lines = [
            f'What predicts {self.outcome_metric} within a run',
            f'ordinary least squares, fitted separately within each of {n_runs} runs',
        ]

        # Mathtext, so the exponent reads as one: 'R^2' on a figure is the kind
        # of detail a reader notices before they read the number beside it.
        fit = []
        if in_sample is not None:
            fit.append(f'median $R^2$ {in_sample:.3f} in sample')
        if held_out is not None:
            fit.append(
                f'{held_out:.3f} on a held-out run '
                f'({predicted}/{n_runs} beat that run\'s own mean)'
            )
        if fit:
            lines.append(', '.join(fit))

        return '\n'.join(lines)

    @staticmethod
    def _coefficient_error_bars(
        df: pl.DataFrame, medians: np.ndarray
    ) -> np.ndarray | None:
        """Asymmetric (lower, upper) offsets for matplotlib's ``xerr``.

        Same shape as the effect-size bars and for the same reason: a percentile
        interval is not symmetric about the estimate, and a predictor with no
        computable interval gets zero-length whiskers rather than removing the
        error bars from every other predictor too.
        """

        needed = {col.COEFFICIENT_CI_LOW, col.COEFFICIENT_CI_HIGH}
        if not needed <= set(df.columns):
            return None

        lows = np.array(
            [np.nan if v is None else v for v in df[col.COEFFICIENT_CI_LOW].to_list()],
            dtype=float,
        )
        highs = np.array(
            [np.nan if v is None else v for v in df[col.COEFFICIENT_CI_HIGH].to_list()],
            dtype=float,
        )

        if np.all(np.isnan(lows)):
            return None

        lower = np.nan_to_num(medians - lows, nan=0.0)
        upper = np.nan_to_num(highs - medians, nan=0.0)
        return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])

    def _effect_size_subtitle(self, df: pl.DataFrame) -> str:
        """Second line of the effect-size title: what was measured, on what unit.

        The unit is spelled out because the number is easy to misread. An
        eta-squared computed over a hundred thousand observations and one computed
        over twenty run medians look identical on a bar chart and mean very
        different things, so the figure states which it is.
        """

        parts = [f'calculated on raw {self.response_name} responses between conditions']

        if df.height and col.UNIT_OF_ANALYSIS in df.columns:
            n_runs = df[col.N_RUNS_TOTAL][0]
            n_conditions = df[col.N_CONDITIONS][0]
            parts.append(
                f'unit of analysis: per-run medians '
                f'({n_runs} runs across {n_conditions} conditions)'
            )

        return '\n'.join(parts)

    def plot_effect_size_by_level(self, df: pl.DataFrame, effect_size_metric: str) -> None:
        """
        Plots a bar chart of effect size per level.

        Args:
            df: Data containing effect-size values.
            effect_size_metric: Eta-squared.
        """

        logger.info(f'- {effect_size_metric}')

        sublabels = []
        if col.ANOVA_P in df.columns:
            sublabels = [
                '' if p is None else f'p={p:.3g}'
                for p in df[col.ANOVA_P].to_list()
            ]

        error_bars = self._effect_size_error_bars(df, effect_size_metric)

        self._plot_effect_size_bar_chart(
            labels=df[col.LEVEL].to_list(),
            effect_size_data=df[effect_size_metric].to_list(),
            title=(
                f'{effect_size_metric} values by level\n'
                f'{self._effect_size_subtitle(df)}'
            ),
            xlabel='Level',
            ylabel=effect_size_metric,
            rotation=0,
            output_fname=f'{effect_size_metric}_by_level.png',
            bar_sublabels=sublabels,
            error_bars=error_bars,
        )

    @staticmethod
    def _effect_size_error_bars(df: pl.DataFrame, effect_size_metric: str) -> np.ndarray | None:
        """Asymmetric (lower, upper) offsets for matplotlib's ``yerr``.

        The bootstrap interval is not symmetric about the estimate, so the two
        halves are passed separately rather than as a single +/- value. Bars with
        no computable interval get zero-length whiskers instead of dropping the
        error bars from the whole figure.
        """
        needed = {col.ETA_SQUARED_CI_LOW, col.ETA_SQUARED_CI_HIGH}
        if not needed <= set(df.columns):
            return None

        values = np.array(df[effect_size_metric].to_list(), dtype=float)
        lows = np.array(
            [v if v is not None else np.nan for v in df[col.ETA_SQUARED_CI_LOW].to_list()],
            dtype=float,
        )
        highs = np.array(
            [v if v is not None else np.nan for v in df[col.ETA_SQUARED_CI_HIGH].to_list()],
            dtype=float,
        )

        if np.all(np.isnan(lows)):
            return None

        lower = np.nan_to_num(values - lows, nan=0.0)
        upper = np.nan_to_num(highs - values, nan=0.0)
        # Clip: a percentile interval can sit slightly the wrong side of the
        # point estimate, and a negative offset makes matplotlib raise.
        return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])
