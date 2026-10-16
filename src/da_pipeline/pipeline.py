"""Orchestration: the order the stages run in, and what they hand each other.

This module is deliberately flat and readable end to end. Every stage is a call
into a component that knows nothing about the others, so the sequence below is
the whole story of a run:

    validate -> extract -> analyse -> save -> plot

The large frames are deleted as soon as their last consumer is done. On a full
dataset the per-observation frames are the biggest objects in the process, and
plotting forks worker processes -- so holding a frame one stage longer than
necessary is paid for once per worker, not once.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import matplotlib

# Every figure is written to disk and none are shown interactively, so use the
# file-writing backend. Agg needs no display, opens no windows, and is the
# fastest path to a saved PNG. This must run before pyplot is imported
# anywhere, hence its position above the other imports.
#
# Figure styling is deliberately NOT set here: it travels with each figure
# instead, which is what makes it survive the plotting process pool.
matplotlib.use("Agg")

from .analysis.bin_summaries import DataBinSummariser  # noqa: E402
from .analysis.binning import DataBinner  # noqa: E402
from .analysis.correlations import DataCorrelator  # noqa: E402
from .analysis.effect_size import EffectSizeCalculator  # noqa: E402
from .analysis.model import OutcomeModeller  # noqa: E402
from .analysis.pairwise import PairwiseConditionComparator  # noqa: E402
from .analysis.run_spread import RunSpreadSummariser  # noqa: E402
from .analysis.variance import VarianceDecomposer  # noqa: E402
from .cli import build_config  # noqa: E402
from .config import PipelineConfig  # noqa: E402
from .constants import columns as col  # noqa: E402
from .extraction import DataExtractor  # noqa: E402
from .frames import (  # noqa: E402
    get_condition_order,
    get_condition_plot_labels,
    get_condition_run_counts,
)
from .io.paths import create_output_dir, setup_logging  # noqa: E402
from .io.writers import DataSaver  # noqa: E402
from .metrics import resolve_metric_roles  # noqa: E402
from .plotting.figures import build_condition_colours, resolve_palette  # noqa: E402
from .plotting.plots import DataPlotter  # noqa: E402
from .reporting.evidence import AI_SUBDIR, write_evidence_pack  # noqa: E402
from .reporting.findings import build_run_findings  # noqa: E402
from .reporting.html_report import write_report  # noqa: E402
from .reporting.prompt import write_prompt  # noqa: E402
from .validation import DataValidator  # noqa: E402

logger = logging.getLogger(__name__)


def run(config: PipelineConfig) -> Path:
    """Run the full pipeline for one config.

    Args:
        config: The run's settings. If it does not yet carry an output
            directory, one is created beside the config file.

    Returns:
        The output directory the results were written to.
    """

    if config.output_dir is None:
        config = config.with_output_dir(create_output_dir(config))

    # ------------------------------------------ [ Data Validations ] -------------------------------------------------

    validator = DataValidator()

    validator.validate_config_dict(config_fpath=config.config_path)
    validator.validate_group_ids(user_provided_group_ids=config.group_ids)
    validator.validate_reference_levels(user_provided_group_ids=config.group_ids)
    validator.validate_metric_names()

    logger.info('\nAll data validations completed successfully!')

    # Collect validated data
    config_dict = validator.config_dict
    group_ids = validator.group_ids
    group_label_dict = validator.get_group_label_dict(
        user_provided_group_labels=config.group_labels
    )
    reference_levels_dict = validator.get_reference_levels_dict()

    # The run's metric set, decided here and nowhere else: the files say which
    # metrics exist, the config says which to analyse and what part each plays,
    # and every stage below is handed the answer rather than naming a metric.
    # A configured metric the files do not declare stops the run here.
    roles = resolve_metric_roles(
        declared=validator.get_declared_metric_names(),
        selection=config.metrics,
    )

    logger.info('\nMetrics:')
    for line in roles.describe():
        logger.info(line)

    condition_order = get_condition_order(config_dict=config_dict)

    # 'config_dict' keeps the raw condition names, so the 'Condition' column stays clean.
    # The counts live only in 'condition_plot_labels', which is handed to the plot legends.
    condition_plot_labels = get_condition_plot_labels(
        config_dict, config.plots.show_condition_counts
    )

    # Kept as numbers so the plotting layer can compose its own tick labels.
    condition_run_counts = get_condition_run_counts(config_dict)

    # condition -> colour, fixed for the whole run. Built and validated here so
    # a missing or duplicated condition fails now, not part-way through plotting.
    condition_colours = build_condition_colours(
        conditions=condition_order,
        palette=resolve_palette(config),
    )
    logger.info(f'\nCondition colours: {condition_colours}')

    logger.info(f'\nDetected {len(config_dict)} unique conditions.')
    logger.info('\nData Analytics configuration file:')

    for condition, paths in config_dict.items():
        logger.info(f'\n- Condition: {condition}')
        logger.info('- File paths:')
        for path in paths:
            logger.info(f'    {path}')

    logger.info('\nIterating through conditions and corresponding files...')
    logger.info('\nData Processing...')

    # ------------------------------------- [ Data Extraction ] ------------------------------------------------------

    data_extractor = DataExtractor(
        config=config,
        roles=roles,
        config_dict=config_dict,
        group_ids=group_ids,
        group_label_dict=group_label_dict,
        reference_levels_dict=reference_levels_dict,
    )

    # Observation count
    final_observation_counts_df = data_extractor.process_observation_count()

    # Responses
    preloaded_responses_dict = data_extractor.preload_responses()
    final_responses_df = data_extractor.process_response_datasets(
        preloaded_responses_dict=preloaded_responses_dict
    )

    # eta-squared, on the response only: "variance explained by condition" is the
    # most attribution-flavoured phrasing in the output, and the pairwise
    # comparisons below answer the practical question better.
    eta_squared_by_level_df = None
    if config.calculate_effect_size:
        effect_size_calculator = EffectSizeCalculator(response_name=config.response_name)
        eta_squared_by_level_df = effect_size_calculator.eta_squared_by_level(
            data_df=final_responses_df
        )

    # SNR
    final_signal_to_noise_df = data_extractor.process_signal_to_noise(
        preloaded_responses_dict=preloaded_responses_dict
    )

    # Both consumers of the preloaded responses are done; release the
    # largest in-memory structure before continuing.
    del preloaded_responses_dict

    # Predictor metrics
    final_predictor_metrics_df = data_extractor.process_predictor_metrics()

    # Outcome metric and its components
    final_outcome_metrics_df = data_extractor.process_outcome_metrics()

    # ------------------------------------- [ Cross-variable Correlations ] ------------------------------------------

    data_correlator = DataCorrelator(config=config)

    # One merge, read four ways below. Built here rather than inside each
    # analysis so the join is paid for once: on a full dataset this is the
    # widest frame the process holds.
    per_observation_matrix_df = DataCorrelator.build_per_observation_matrix(
        final_signal_to_noise_df=final_signal_to_noise_df,
        final_predictor_metrics_df=final_predictor_metrics_df,
        final_outcome_metrics_df=final_outcome_metrics_df,
    )

    # Three views of one relationship: pooled across every observation of a
    # condition, computed within each run then summarised across runs, and
    # computed between the run medians themselves. A single pooled coefficient
    # answers all three questions at once and reports one number for them.
    (
        correlations_df,
        within_run_correlations_df,
        between_run_correlations_df,
    ) = data_correlator.build_correlation_tables(per_observation_matrix_df)

    # ------------------------------------- [ Multivariable Outcome Model ] ---------------------------------------

    # The correlations take one predictor at a time, so they cannot separate
    # predictors that move together. This fits all of them at once, inside each
    # run, and cross-validates on a held-out run.
    #
    # The predictors are the run's own: signal-to-noise, plus every metric the
    # roles put in the predictor role. The outcome's components are deliberately
    # absent -- see analysis.model for why a restatement of the outcome cannot
    # be a predictor of it.
    outcome_model_df, outcome_model_by_run_df = OutcomeModeller(
        predictors=[col.SIGNAL_TO_NOISE_RATIO, *roles.predictors],
        outcome=roles.outcome,
    ).build_model_tables(per_observation_matrix_df)

    del per_observation_matrix_df

    # ------------------------------------- [ Variance Decomposition ] -----------------------------------------------

    # Runs before the frames are released, and needs all four of them: the
    # comparison it exists for is between metrics, not within one.
    variance_df = VarianceDecomposer(
        response_name=config.response_name
    ).build_variance_decomposition_df(
        final_responses_df=final_responses_df,
        final_signal_to_noise_df=final_signal_to_noise_df,
        final_predictor_metrics_df=final_predictor_metrics_df,
        final_outcome_metrics_df=final_outcome_metrics_df,
        predictor_metric_names=list(roles.predictors),
        outcome_metric_names=list(roles.outcome_metrics),
    )

    # Which conditions actually differ, on every metric. Runs here rather than
    # beside eta-squared because it needs the downstream frames too.
    pairwise_conditions_df = None
    if config.calculate_effect_size:
        pairwise_conditions_df = PairwiseConditionComparator(
            response_name=config.response_name
        ).build_pairwise_df(
            final_responses_df=final_responses_df,
            final_signal_to_noise_df=final_signal_to_noise_df,
            final_predictor_metrics_df=final_predictor_metrics_df,
            final_outcome_metrics_df=final_outcome_metrics_df,
            predictor_metric_names=list(roles.predictors),
            outcome_metric_names=list(roles.outcome_metrics),
        )

    # Same run medians, asked a different question: the decomposition says how
    # much spread is run-to-run overall, this says which condition carries it.
    run_spread_df = RunSpreadSummariser(
        response_name=config.response_name
    ).build_run_spread_df(
        final_responses_df=final_responses_df,
        final_signal_to_noise_df=final_signal_to_noise_df,
        final_predictor_metrics_df=final_predictor_metrics_df,
        final_outcome_metrics_df=final_outcome_metrics_df,
        predictor_metric_names=list(roles.predictors),
        outcome_metric_names=list(roles.outcome_metrics),
    )

    # ----------------------------------------------- [ Data Bins ] --------------------------------------------------

    binner = DataBinner(config=config, roles=roles)

    # Equal-count bins of the outcome metric
    outcome_bins_df = binner.build_outcome_bins_df(final_outcome_metrics_df)

    # SNR + outcome bins
    final_snr_outcome_bins_df = binner.build_snr_and_outcome_bins_df(
        final_signal_to_noise_df, outcome_bins_df
    )

    # Predictor metrics + outcome bins
    final_predictor_outcome_bins_df = binner.build_predictor_and_outcome_bins_df(
        final_predictor_metrics_df, outcome_bins_df
    )

    # Outcome metrics + outcome bins
    final_outcome_metrics_bins_df = binner.build_outcome_metrics_and_bins_df(
        final_outcome_metrics_df, outcome_bins_df
    )

    # ----------------------------------------- [ Data Bin Summaries ] -----------------------------------------------

    data_bin_summariser = DataBinSummariser(outcome=roles.outcome)

    snr_bin_summaries_df = (
        data_bin_summariser.build_snr_and_outcome_bin_summaries_df(
            final_snr_outcome_bins_df
        )
    )

    predictor_bin_summaries_df = (
        data_bin_summariser.build_predictor_and_outcome_bin_summaries_df(
            final_predictor_outcome_bins_df, list(roles.predictors)
        )
    )

    component_bin_summaries_df = (
        data_bin_summariser.build_component_and_outcome_bin_summaries_df(
            final_outcome_metrics_bins_df, list(roles.components)
        )
    )

    # ------------------------------------------------ [ Save Data ] -------------------------------------------------

    data_saver = DataSaver(config=config, roles=roles)
    exports = config.exports

    if exports.observation_counts:
        data_saver.save_observation_count_csv(final_observation_counts_df)

    if exports.responses:
        data_saver.save_responses_csv(final_responses_df)

    if exports.signal_to_noise:
        data_saver.save_signal_to_noise_ratios_csv(final_signal_to_noise_df)

    if exports.predictor_metrics:
        data_saver.save_predictor_metrics_csv(final_predictor_metrics_df)

    if exports.outcome_metrics:
        data_saver.save_outcome_metrics_csv(final_outcome_metrics_df)

    if exports.correlations:
        data_saver.save_correlations_csv(correlations_df)
        data_saver.save_within_run_correlations_csv(within_run_correlations_df)
        data_saver.save_between_run_correlations_csv(between_run_correlations_df)

    # Effect size is always written. It is a handful of rows, and the figure
    # alone does not carry the replicate counts or the unit of analysis.
    if config.calculate_effect_size:
        data_saver.save_effect_size_csv(eta_squared_by_level_df)
        data_saver.save_pairwise_conditions_csv(pairwise_conditions_df)

    data_saver.save_variance_decomposition_csv(variance_df)
    data_saver.save_run_spread_csv(run_spread_df)
    data_saver.save_outcome_model_csv(outcome_model_df)
    data_saver.save_outcome_model_by_run_csv(outcome_model_by_run_df)

    # Bin summaries are always written: they are the compact form of the run.
    data_saver.save_snr_bin_summary(snr_bin_summaries_df)
    data_saver.save_predictor_bin_summaries(predictor_bin_summaries_df, list(roles.predictors))
    data_saver.save_component_bin_summaries(component_bin_summaries_df, list(roles.components))

    # ------------------------------------------------ [ Data Plots ] ------------------------------------------------

    plotter = DataPlotter(
        config=config,
        roles=roles,
        condition_order=condition_order,
        condition_colours=condition_colours,
        condition_run_counts=condition_run_counts,
        condition_plot_labels=condition_plot_labels,
    )

    logger.info('\nGenerating plots...')

    # Observation counts
    plotter.plot_observation_counts(data_df=final_observation_counts_df)
    del final_observation_counts_df

    # Responses
    plotter.plot_responses_by_level(data_df=final_responses_df)
    del final_responses_df

    # Effect size
    if config.calculate_effect_size:
        plotter.plot_effect_size_by_level(
            df=eta_squared_by_level_df,
            effect_size_metric=col.ETA_SQUARED,
        )

    # SNR
    plotter.plot_signal_to_noise_by_condition(data_df=final_signal_to_noise_df)
    del final_signal_to_noise_df
    plotter.plot_signal_to_noise_by_outcome_bins(data_df=final_snr_outcome_bins_df)
    del final_snr_outcome_bins_df

    # Predictor metrics
    plotter.plot_predictor_metrics(data_df=final_predictor_metrics_df)
    del final_predictor_metrics_df
    plotter.plot_predictor_metrics_by_outcome_bins(data_df=final_predictor_outcome_bins_df)
    del final_predictor_outcome_bins_df

    # Correlations: the two per-condition views against each other, then the
    # between-run view on its own. These are the only figures that show where
    # the three views disagree, which no single table does.
    plotter.plot_correlation_views(
        pooled_df=correlations_df,
        within_run_df=within_run_correlations_df,
    )
    plotter.plot_between_run_correlations(data_df=between_run_correlations_df)

    # Variance decomposition
    plotter.plot_variance_decomposition(data_df=variance_df)

    # What predicts the outcome within a run
    plotter.plot_model_coefficients(data_df=outcome_model_df)

    # Outcome metric and its components
    plotter.plot_outcome_metrics(data_df=final_outcome_metrics_df)
    del final_outcome_metrics_df
    plotter.plot_component_metrics_by_outcome_bins(data_df=final_outcome_metrics_bins_df)
    del final_outcome_metrics_bins_df

    # ------------------------------------------- [ Report and AI inputs ] -------------------------------------------

    # Last, because both read the figures and CSVs the stages above wrote. The
    # analysis frames are still held, but the files on disk are what gets read:
    # if the two ever disagreed, the files are what somebody would check.
    logger.info('\nBuilding the report and AI inputs...')

    output_dir = config.resolved_output_dir

    findings = build_run_findings(
        response_name=config.response_name,
        outcome_metric=roles.outcome,
        # Recorded in the pack and the briefing, because the metric set is
        # discovered from the files: a reader has no other way to know which
        # metrics this run carried, or what part each of them played.
        metrics={
            "outcomes": [roles.outcome],
            "components": list(roles.components),
            "predictors": list(roles.predictors),
        },
        condition_run_counts=condition_run_counts,
        effect_size_df=eta_squared_by_level_df,
        pairwise_df=pairwise_conditions_df,
        variance_df=variance_df,
        run_spread_df=run_spread_df,
        correlations_df=correlations_df,
        within_run_correlations_df=within_run_correlations_df,
        model_df=outcome_model_df,
    )

    write_report(findings, output_dir)

    # The pack is scanned for run identifiers across every frame that can carry
    # one, so a pseudonym is consistent between, say, the run spread table and
    # the per-run model fits.
    write_evidence_pack(
        findings,
        output_dir,
        analysis_frames=[
            run_spread_df,
            outcome_model_by_run_df,
        ],
    )
    write_prompt(findings, output_dir / AI_SUBDIR)

    return output_dir


def main(argv=None) -> None:
    """Entry point for the ``da-pipeline`` command."""

    config = build_config(argv)
    config = config.with_output_dir(create_output_dir(config))

    setup_logging(output_dir=config.resolved_output_dir)

    logger.info(f'\nNumber of available CPUs: {os.cpu_count()}')

    logger.info('\nInput Arguments:')
    for line in config.describe():
        logger.info(line)

    start_time = time.time()
    run(config)
    execution_time = time.time() - start_time

    logger.info(
        f'\nExecution Time: {(execution_time / 60): .2f} min ({execution_time:.2f} sec)'
    )
    logger.info('\nDone!')


if __name__ == '__main__':
    main()
