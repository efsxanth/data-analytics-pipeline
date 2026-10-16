"""Dataframe column names and the key orders built from them.

Every column the pipeline creates is named here exactly once. Extraction,
binning, summarising, saving and plotting all reach for the same symbol, so a
rename is a one-line change that cannot leave one stage writing ``group_id``
while another reads ``groupId``.

The sort-key lists live beside the columns they are built from, because a sort
order is only correct with respect to a particular frame's shape: per-observation
frames carry an observation id, response frames additionally carry a level, and
observation-count frames carry neither.
"""

# -------------------------------------------------------------------------------------------------
# Identity
# -------------------------------------------------------------------------------------------------

RUN_ID = 'run_id'
CONDITION = 'condition'
GROUP_ID = 'group_id'
GROUP_LABEL = 'group_label'
OBSERVATION_ID = 'observation_id'

# -------------------------------------------------------------------------------------------------
# Measured quantities
# -------------------------------------------------------------------------------------------------

LEVEL = 'level'
SIGNAL_TO_NOISE_RATIO = 'signal_to_noise_ratio'
OBSERVATION_COUNT = 'observation_count'
OBSERVATION_COUNT_TOTAL = 'observation_count_total'
OBSERVATION_COUNT_MEDIAN = 'observation_count_median'
ETA_SQUARED = 'eta_squared'

# There is deliberately no constant for the outcome metric's own column. Which
# metric is the outcome is configuration, and the frames carry every metric
# under the name its file declared -- so the column is named by
# ``MetricRoles.outcome``, not from here. A constant would be a second place for
# that name to live, and the one place it could go stale.

# -------------------------------------------------------------------------------------------------
# Unit of analysis
# -------------------------------------------------------------------------------------------------
# Observations within one run are not independent replicates -- they were all
# measured together, on the same equipment, under the same setup, on the same
# day. The run is the experimental unit, so any statistic meant to
# support a claim about a *condition* is computed on one value per run. These
# columns state that on the output so a reader cannot mistake a run-level
# figure for an observation-level one.

UNIT_OF_ANALYSIS = 'unit_of_analysis'

# Values UNIT_OF_ANALYSIS takes. They are not interchangeable, and the
# distinction is easy to lose: a statistic computed *from* run medians is not
# the same as a statistic computed inside each run and then summarised across
# runs. Eta-squared is the first -- every run contributes one number. The
# within-run correlations are the second -- each coefficient is computed over
# the observations of one run, and only the summary is at run level.
UNIT_RUN_MEDIAN = 'run_median'
UNIT_OBSERVATION_WITHIN_RUN = 'observation_within_run_summarised_over_runs'

ANOVA_P = 'anova_p'
ETA_SQUARED_CI_LOW = 'eta_squared_ci_low'
ETA_SQUARED_CI_HIGH = 'eta_squared_ci_high'
CI_LEVEL = 'ci_level'
CI_METHOD = 'ci_method'
N_BOOTSTRAP = 'n_bootstrap'
CI_METHOD_BOOTSTRAP_RUNS = 'percentile_bootstrap_over_runs'

# -------------------------------------------------------------------------------------------------
# Pairwise condition comparisons
# -------------------------------------------------------------------------------------------------
# Computed on every metric, on run medians. Unlike the omnibus eta-squared,
# which is kept to the raw response, these answer the practical question -- which
# conditions differ, by how much, in which direction -- and that is worth asking
# of the outcome metrics too, not only of the response they are derived from.

CONDITION_A = 'condition_a'
CONDITION_B = 'condition_b'
N_RUNS_A = 'n_runs_a'
N_RUNS_B = 'n_runs_b'
HIGHER_CONDITION = 'higher_condition'
MEDIAN_DIFFERENCE = 'median_difference'

# Common-language effect size: the probability that a randomly chosen run of A
# scores above a randomly chosen run of B, ties counting a half. Far easier to
# read at a handful of runs than a standardised difference -- "5 of 6 pairings"
# is a fact, where a Cohen's d would be an estimate of a population parameter.
CLES = 'cles'
CLES_CI_LOW = 'cles_ci_low'
CLES_CI_HIGH = 'cles_ci_high'

P_VALUE = 'p_value'
P_VALUE_ADJUSTED = 'p_value_adjusted'
P_ADJUST_METHOD = 'p_adjust_method'
P_ADJUST_FAMILY = 'p_adjust_family'
P_ADJUST_BENJAMINI_HOCHBERG = 'benjamini_hochberg'
N_RUNS_TOTAL = 'n_runs_total'
N_CONDITIONS = 'n_conditions'
N_RUNS_PER_CONDITION = 'n_runs_per_condition'
N_OBSERVATIONS = 'n_observations'

# -------------------------------------------------------------------------------------------------
# Variance decomposition
# -------------------------------------------------------------------------------------------------
# A metric's spread split into the three layers of the design: which condition an
# observation belongs to, which run inside that condition, and the observation
# itself. The shares answer a question no single statistic does -- whether a difference
# between conditions is large next to how much two runs of the *same* condition
# already differ.

METRIC = 'metric'
PCT_CONDITION = 'pct_condition'
PCT_RUN_WITHIN_CONDITION = 'pct_run_within_condition'
PCT_OBSERVATION_WITHIN_RUN = 'pct_observation_within_run'
SS_CONDITION = 'ss_condition'
SS_RUN_WITHIN_CONDITION = 'ss_run_within_condition'
SS_OBSERVATION_WITHIN_RUN = 'ss_observation_within_run'
SS_TOTAL = 'ss_total'
RUN_TO_CONDITION_RATIO = 'run_to_condition_ratio'

# No column records where a metric sits in the processing chain. Every condition
# statistic here is an *association* -- the design supports "this metric differs
# between these conditions", not "the condition improved it through this
# mechanism" -- and that distinction belongs in the documentation, which can
# explain it, rather than in a per-row label that only names a pipeline stage.
# Every metric reports its comparison either way; nothing is suppressed.

# -------------------------------------------------------------------------------------------------
# Per-condition run spread
# -------------------------------------------------------------------------------------------------
# The variance decomposition is global: it reports how much of a metric's spread
# is run-to-run across the whole dataset, not which condition carries it. These
# columns close that gap by describing one condition's runs on their own, and
# answer the question a reader actually asks of a difference between conditions
# -- is it consistent across the runs, or is one run carrying it?

CONDITION_MEDIAN = 'condition_median'
CONDITION_MEAN = 'condition_mean'
RUN_MEDIAN_MIN = 'run_median_min'
RUN_MEDIAN_MAX = 'run_median_max'
RUN_MEDIAN_RANGE = 'run_median_range'
RUN_MEDIAN_IQR = 'run_median_iqr'
RUN_MEDIAN_RANGE_PCT = 'run_median_range_pct'
MAX_LEAVE_ONE_OUT_SHIFT = 'max_leave_one_out_shift'
MOST_INFLUENTIAL_RUN = 'most_influential_run'

# -------------------------------------------------------------------------------------------------
# Binning
# -------------------------------------------------------------------------------------------------

# The ordered, unique bin identity. Group, join and sort on this.
OUTCOME_BIN_INDEX = 'outcome_bin_index'

# Human-readable range label ("0.8036-0.8393"). Display only, not unique:
# equal-count tie-splitting can give two adjacent bins the same range.
OUTCOME_BIN = 'outcome_bin'

N_POINTS_BIN = 'n_points_bin'

# A bin summary describes each metric's distribution inside the bin, and names
# the results by suffixing the metric: ``degradation_1`` becomes
# ``degradation_1_median``, ``degradation_1_q10`` and so on. The suffixes are
# declared here rather than written inline where the rows are built, so this
# module still holds every column name the pipeline can produce -- the summary
# columns are the only ones whose full name is not known until the run's metric
# set has been discovered.
SUFFIX_MEDIAN = '_median'
SUFFIX_Q10 = '_q10'
SUFFIX_Q25 = '_q25'
SUFFIX_Q75 = '_q75'
SUFFIX_Q90 = '_q90'
SUFFIX_CENTRAL_SPREAD_IQR = '_central_spread_iqr'
SUFFIX_OUTER_SPREAD = '_outer_spread'
SUFFIX_QUANTILE_SKEW = '_quantile_skew'
SUFFIX_TAIL_SKEW = '_tail_skew'
SUFFIX_QUANTILE_KURTOSIS = '_quantile_kurtosis'

DISTRIBUTION_SUFFIXES = (
    SUFFIX_MEDIAN,
    SUFFIX_Q10,
    SUFFIX_Q25,
    SUFFIX_Q75,
    SUFFIX_Q90,
    SUFFIX_CENTRAL_SPREAD_IQR,
    SUFFIX_OUTER_SPREAD,
    SUFFIX_QUANTILE_SKEW,
    SUFFIX_TAIL_SKEW,
    SUFFIX_QUANTILE_KURTOSIS,
)

# Plotting-only column: the bin index rendered as a non-numeric category key
# ('bin1', 'bin2', ...). See plotting.plots._faceted_boxplot_by_condition for
# why the numeric index cannot go straight onto a categorical axis.
OUTCOME_BIN_CATEGORY = 'outcome_bin_category'

# -------------------------------------------------------------------------------------------------
# Correlations
# -------------------------------------------------------------------------------------------------

CORR_VAR_X = 'var_x'
CORR_VAR_Y = 'var_y'
CORR_N_OBS = 'n_obs'
CORR_PEARSON_R = 'pearson_r'
CORR_SPEARMAN_R = 'spearman_r'

# There are deliberately no p-value columns here. A p-value computed over
# observations pooled from several runs treats repeated measurements of one run
# as independent replicates, so it comes out overwhelmingly "significant" whether
# or not the relationship is real. The coefficients are kept -- they describe
# the observation population honestly -- but the inference is not supported by the
# design. See analysis.correlations.DataCorrelator for the full reasoning, and
# the within-run summary below for the reproducibility claim that replaces it.

# Within-run correlations: one coefficient per run, then summarised across the
# runs of a condition. Pooling observations from several runs mixes how
# observations covary *inside* a run with how run averages covary *across* runs,
# which can inflate a coefficient or reverse its sign.
CORR_N_RUNS = 'n_runs'
CORR_PEARSON_R_MEDIAN = 'pearson_r_median'
CORR_PEARSON_R_MIN = 'pearson_r_min'
CORR_PEARSON_R_MAX = 'pearson_r_max'
CORR_PEARSON_SAME_SIGN = 'pearson_n_runs_same_sign'
CORR_SPEARMAN_R_MEDIAN = 'spearman_r_median'
CORR_SPEARMAN_R_MIN = 'spearman_r_min'
CORR_SPEARMAN_R_MAX = 'spearman_r_max'
CORR_SPEARMAN_SAME_SIGN = 'spearman_n_runs_same_sign'

# Between-run correlations reuse CORR_VAR_X/Y, the two coefficients and
# CORR_N_RUNS above, marked with UNIT_RUN_MEDIAN. They take one median per run
# and correlate across the runs themselves: the third of the three questions
# pooling silently merges -- "do the runs that came out better on one metric
# also come out better on the other?" -- and the only one whose unit of
# analysis really is the run.

# -------------------------------------------------------------------------------------------------
# Multivariable outcome model
# -------------------------------------------------------------------------------------------------
# The correlations answer one predictor at a time, which cannot separate
# predictors that move together: signal-to-noise and the degradation metrics each
# correlate with the outcome partly because they correlate with each other. A
# regression carrying all of them at once reports what each still contributes
# with the others held constant.
#
# Fitted inside each run, for the same reason correlations are, then summarised
# across runs. Condition is deliberately absent as a predictor -- it is constant
# within a run, so a per-run model cannot estimate it even in principle.

OUTCOME = 'outcome'
PREDICTOR = 'predictor'
COEFFICIENT = 'coefficient'
COEFFICIENT_MEDIAN = 'coefficient_median'
COEFFICIENT_MIN = 'coefficient_min'
COEFFICIENT_MAX = 'coefficient_max'
COEFFICIENT_CI_LOW = 'coefficient_ci_low'
COEFFICIENT_CI_HIGH = 'coefficient_ci_high'
COEFFICIENT_SAME_SIGN = 'n_runs_same_sign'

# Predictors and outcome are standardised within each run, so a coefficient is
# "standard deviations of the outcome per standard deviation of predictor", which
# makes them directly comparable to one another and across runs. Standardising the
# outcome as well is what removes the intercept: both sides are centred, so it
# is zero by construction and is not reported.
STANDARDISATION = 'standardisation'
STANDARDISED_WITHIN_RUN = 'z_scored_within_run'

# Variance inflation factor: how much a predictor's coefficient is destabilised
# by its overlap with the others. Reported rather than used to drop predictors,
# because a high VIF makes an individual coefficient unreliable without making
# the model's fit any worse -- the reader needs to know which it is.
VIF = 'vif'
VIF_MEDIAN = 'vif_median'
VIF_MAX = 'vif_max'

R_SQUARED_IN_SAMPLE = 'r_squared_in_sample'
R_SQUARED_IN_SAMPLE_MEDIAN = 'r_squared_in_sample_median'

# Leave-one-run-out: fit on every other run, then predict the held-out one.
# Sign agreement across runs measures *consistency*; only a held-out run
# measures whether the relationship transfers to a run the fit never saw. A
# negative value means the fitted coefficients did worse than predicting that
# run's own mean, which is the honest baseline for a within-run model.
R_SQUARED_OUT_OF_SAMPLE = 'r_squared_out_of_sample'
R_SQUARED_OUT_OF_SAMPLE_MEDIAN = 'r_squared_out_of_sample_median'
N_RUNS_PREDICTED = 'n_runs_predicted_better_than_run_mean'
N_RUNS_MODELLED = 'n_runs_modelled'

# -------------------------------------------------------------------------------------------------
# Metadata / identity column groups
# -------------------------------------------------------------------------------------------------

METADATA_COLS = [
    RUN_ID,
    CONDITION,
    GROUP_ID,
]

METADATA_COLS_PER_OBSERVATION = METADATA_COLS + [OBSERVATION_ID]

# -------------------------------------------------------------------------------------------------
# Sort orders
# -------------------------------------------------------------------------------------------------

# Observation-count extraction output, prior to aggregation.
SORT_COLS_OBSERVATION_COUNTS = [
    RUN_ID,
    CONDITION,
    GROUP_ID,
]

# Response datasets carry an additional level dimension.
SORT_COLS_RESPONSES = [
    RUN_ID,
    CONDITION,
    GROUP_ID,
    LEVEL,
    OBSERVATION_ID,
]

# Per-observation datasets: signal-to-noise, predictor metrics, outcome metrics.
SORT_COLS_PER_OBSERVATION = [
    RUN_ID,
    CONDITION,
    GROUP_ID,
    OBSERVATION_ID,
]
