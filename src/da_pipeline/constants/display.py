"""Words and proportions that appear in figures.

Kept apart from the column names because these are read by people, not by
joins: changing ``UNIT_OBS`` alters what a reader sees, changing a column name
alters what the code does, and conflating the two in one module made it easy to
edit the wrong kind of string.
"""

from . import columns as col

# -------------------------------------------------------------------------------------------------
# Count vocabulary
# -------------------------------------------------------------------------------------------------
# Two different counts appear in labels and titles, and a bare 'N' for both is
# ambiguous. Each is therefore named by its unit:
#
#   RUNS -- how many files (runs) make up a condition.
#   OBS  -- how many observations (rows) went into the box, panel or bin.
#
# Both appear together as 'Condition_A\n(3 runs, 1,440 obs)'. 'obs' is an
# abbreviation, so it does not inflect; 'run' does, so a condition built from
# one file reads '1 run' rather than '1 runs'.

# -------------------------------------------------------------------------------------------------
# Per-run overlay ("SuperPlot") markers -- see plotting.plots._boxplot_by_condition
# for the rationale and the Lord et al. (2020) reference.
# -------------------------------------------------------------------------------------------------
# One marker per run, drawn over the boxes so the replicate count is visible
# rather than implied by a caption. Deliberately monochrome: the runs of a
# condition are interchangeable repeats, and colouring them would suggest a
# pairing between conditions that does not exist.

RUN_MARKER_LABEL = 'per-run median'
MEAN_MARKER_LABEL = 'mean of all observations'


# -------------------------------------------------------------------------------------------------
# Variance decomposition
# -------------------------------------------------------------------------------------------------
# Ordered outermost-inwards, which is also the stacking order on the bar: the
# layer you manipulated, then the layer you repeated, then the measurement.

VARIANCE_CONDITION_LABEL = 'condition'
VARIANCE_RUN_LABEL = 'run within condition'
VARIANCE_OBSERVATION_LABEL = 'observation within run'

VARIANCE_CONDITION_COLOUR = '#2c6e8e'
VARIANCE_RUN_COLOUR = '#d98032'
VARIANCE_OBSERVATION_COLOUR = '#bfbfbf'

# Height of one metric row, as a fraction of the configured figure height.
VARIANCE_ROW_HEIGHT_FRACTION = 0.16

# -------------------------------------------------------------------------------------------------
# Multivariable outcome model
# -------------------------------------------------------------------------------------------------
# The forest plot carries one row per predictor -- a handful, where the variance
# figure carries every analysed metric -- so its rows are given more height each
# and the figure still comes out compact.

MODEL_ROW_HEIGHT_FRACTION = 0.22

# -------------------------------------------------------------------------------------------------
# Correlation views
# -------------------------------------------------------------------------------------------------
# Three views of one relationship, and the figures exist mainly to show where
# they disagree. A pooled coefficient blends how observations covary inside a run
# with how run averages covary across runs, so it can come out larger, smaller
# or opposite in sign to the within-run one -- and only a picture makes that
# visible without joining two CSVs by hand.

POOLED_LABEL = 'pooled across the condition'
WITHIN_RUN_LABEL = 'within-run median'
WITHIN_RUN_RANGE_LABEL = 'within-run range across runs'

# Open marker for pooled, filled for within-run: the filled one is the estimate
# the design supports, so it is the one that reads as solid.
POOLED_MARKER_COLOUR = '#6d6d6d'
WITHIN_RUN_MARKER_COLOUR = '#2c6e8e'

# Reserved for the pairs whose two views land on opposite sides of zero. Nothing
# else on these figures is red, so it needs no legend entry to be understood.
SIGN_REVERSAL_COLOUR = '#c1362f'

# Behind the two quadrants where the pooled and within-run coefficients disagree
# in sign -- the region a reader should be able to find without reading a label.
SIGN_REVERSAL_SHADE = '#f2dedc'

# 45 pairs is a tall figure whatever is done, so the row height is set low
# enough that the whole set still fits on one page.
CORRELATION_ROW_HEIGHT_FRACTION = 0.10

# A pair is named by both its metrics, joined so the row reads as one label.
PAIR_SEPARATOR = '  /  '

# Below this share a segment is too narrow to hold its own label legibly.
VARIANCE_ANNOTATION_MIN_PCT = 6.0
RUN_MARKER_COLOUR = 'black'
RUN_MARKER_EDGE_COLOUR = 'white'
RUN_MARKER_EDGE_WIDTH = 0.6
RUN_MARKER_SIZE = 6
RUN_MARKER_JITTER = 0.12


UNIT_RUN_SINGULAR = 'run'
UNIT_RUN_PLURAL = 'runs'
UNIT_OBS = 'obs'

# -------------------------------------------------------------------------------------------------
# Axis labels
# -------------------------------------------------------------------------------------------------

def bin_axis_label(outcome_metric: str) -> str:
    """Axis label for the equal-count bins of the run's outcome metric.

    Built from the configured outcome rather than fixed, so a run organised
    around a metric called ``yield`` says "yield bin" everywhere a run organised
    around ``Accuracy`` says "Accuracy bin". A constant string here would have
    been the one place the output still assumed which metric the outcome was.
    """
    return f'{outcome_metric} bin'

# -------------------------------------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------------------------------------

# Height of one condition panel in a faceted figure, as a fraction of the
# configured figure height.
FACET_PANEL_HEIGHT_FRACTION = 0.7

# -------------------------------------------------------------------------------------------------
# Registries
# -------------------------------------------------------------------------------------------------

# Observation-count statistic column -> the word used in titles, axis labels and
# file names ('Total' -> total_number_of_observations_barplot.png).
OBSERVATION_COUNT_STATS_REGISTRY = {
    col.OBSERVATION_COUNT_TOTAL: 'Total',
    col.OBSERVATION_COUNT_MEDIAN: 'Median',
}
