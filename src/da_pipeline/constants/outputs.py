"""Where a run puts its results.

One run writes a single tree, rooted next to the config file that described it::

    Data_Analytics[_<suffix>]/
        data_analytics_process.log
        observation_counts/     responses/            signal_to_noise/
        predictor_metrics/      outcome_metrics/
        relationships/          effect_size/
        variance_decomposition/ outcome_model/
        csv_exports/
            <dataset>.csv
            correlations/correlations[_within_run|_between_run].csv
            bin_summaries/<metric>_and_<outcome>_bin_summary.csv

Figures go in the topic directories; every CSV goes under ``csv_exports`` so a
reader can tell at a glance which files are pictures and which are data.

Directory names are fixed and name a *role*, so a reader always knows where to
look. File names that describe a particular metric are built from the run's own
metric names instead -- a bin summary of ``degradation_1`` against an outcome
called ``yield`` is written as ``degradation_1_and_yield_bin_summary.csv``,
because a fixed word there would name the wrong quantity.
"""

# -------------------------------------------------------------------------------------------------
# Output root
# -------------------------------------------------------------------------------------------------

DA_OUTPUT_FOLDER_NAME = 'Data_Analytics'
LOG_FNAME = 'data_analytics_process.log'

# A suffix ends up in a directory name, so it is length-capped rather than
# left to produce a path the filesystem will reject.
MAX_OUTPUT_FOLDER_SUFFIX_LEN = 35

# -------------------------------------------------------------------------------------------------
# Figure sub-directories
# -------------------------------------------------------------------------------------------------

OBSERVATION_COUNT_SUBDIR = 'observation_counts'
RESPONSES_SUBDIR = 'responses'
SNR_SUBDIR = 'signal_to_noise'
PREDICTOR_METRICS_SUBDIR = 'predictor_metrics'
OUTCOME_METRICS_SUBDIR = 'outcome_metrics'
RELATIONSHIPS_SUBDIR = 'relationships'
EFFECT_SIZE_SUBDIR = 'effect_size'
VARIANCE_SUBDIR = 'variance_decomposition'
MODEL_SUBDIR = 'outcome_model'

# -------------------------------------------------------------------------------------------------
# CSV sub-directories
# -------------------------------------------------------------------------------------------------

CSV_EXPORTS_SUBDIR = 'csv_exports'
BIN_SUMMARY_SUBDIR = 'bin_summaries'
CORRELATIONS_SUBDIR = 'correlations'

# -------------------------------------------------------------------------------------------------
# CSV file names
# -------------------------------------------------------------------------------------------------

OBSERVATION_COUNT_CSV_FNAME = f'{OBSERVATION_COUNT_SUBDIR}.csv'
RESPONSES_CSV_FNAME = f'{RESPONSES_SUBDIR}.csv'
SNR_CSV_FNAME = f'{SNR_SUBDIR}.csv'
PREDICTOR_METRICS_CSV_FNAME = f'{PREDICTOR_METRICS_SUBDIR}.csv'
OUTCOME_METRICS_CSV_FNAME = f'{OUTCOME_METRICS_SUBDIR}.csv'
CORRELATIONS_CSV_FNAME = f'{CORRELATIONS_SUBDIR}.csv'
WITHIN_RUN_CORRELATIONS_CSV_FNAME = 'correlations_within_run.csv'
BETWEEN_RUN_CORRELATIONS_CSV_FNAME = 'correlations_between_run.csv'
MODEL_CSV_FNAME = f'{MODEL_SUBDIR}.csv'
MODEL_BY_RUN_CSV_FNAME = f'{MODEL_SUBDIR}_by_run.csv'
EFFECT_SIZE_CSV_FNAME = f'{EFFECT_SIZE_SUBDIR}.csv'
VARIANCE_CSV_FNAME = f'{VARIANCE_SUBDIR}.csv'
RUN_SPREAD_CSV_FNAME = 'run_spread_by_condition.csv'
PAIRWISE_CSV_FNAME = 'pairwise_conditions.csv'

# -------------------------------------------------------------------------------------------------
# File-name vocabulary
# -------------------------------------------------------------------------------------------------

# The signal-to-noise ratio's short form, as it appears in file names.
SNR_LABEL = 'snr'

# The remaining words file names are assembled from, beyond the directory and
# dataset names above, the column names in ``columns``, the metrics a run's
# extracts declared, and the caller's own response and condition names.
#
# Declared here so the output-path guard in the test suite can be written the
# positive way round, as an allow-list: any word appearing in an output path
# that is not one of these and not declared somewhere else is a term that leaked
# into the output. A ban-list would only catch the words somebody thought of.
FILENAME_VOCABULARY = (
    'and',
    'barplot',
    'bins',
    'boxplot',
    'coefficients',
    'number',
    'summary',
    'vs',
)

# Extensions a run writes.
FILE_EXTENSIONS = ('png', 'csv', 'json', 'md', 'html', 'log')

# -------------------------------------------------------------------------------------------------
# Metric-dependent file names
# -------------------------------------------------------------------------------------------------
# Built from the run's own metric names rather than fixed, for the reason given
# in the module docstring.


def bin_summary_fname(metric: str, outcome_metric: str) -> str:
    """CSV file name for one metric's summary against the outcome's bins."""
    return f'{metric}_and_{outcome_metric}_bin_summary.csv'


def snr_bin_summary_fname(outcome_metric: str) -> str:
    """CSV file name for the signal-to-noise summary against the outcome's bins."""
    return bin_summary_fname(SNR_LABEL, outcome_metric)
