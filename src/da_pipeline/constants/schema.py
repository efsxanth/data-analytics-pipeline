"""The extract-file format: this project's own specification.

This project defines the format. The real measurements behind its development
cannot be published, so there is no external file to conform to and no upstream
writer whose choices have to be accommodated: the layout below is the contract,
:mod:`da_pipeline.synthetic` is its reference producer, and
:mod:`da_pipeline.io.extract_reader` is the only module that reads it.

Layout of one extract file::

    /                          attrs: schema_version, run_id, response_name, created_utc
    /observations/id           (n,)                    int
    /observations/group_id     (n,)                    int    co-indexed with id
    /observations/metrics      (n, m)                  float
                               attr: metric_names (m,) str
    /signal/<response_name>    (n, n_steps)            float
    /reference/levels          (n_groups, n_steps)     int
    /reference/group_id        (n_groups,)             int

A *group* is something many observations measure independently; a *step* is one
stage of the repeated measurement every observation goes through.
``/reference/levels`` says what the correct level is at each step of each group,
which is what an accuracy metric is scored against.

The metric set is deliberately open. ``/observations/metrics`` is one matrix
with a ``metric_names`` attribute naming its columns, so a producer may write
any number of metrics under any names and the pipeline discovers them on read.
Nothing in this module names an individual metric, and nothing in the analysis
layer does either: which metrics to analyse, and what part each plays, is
declared in the run's configuration.

One naming convention carries meaning. A metric called ``<outcome>_<something>``
-- ``Accuracy_error_1`` beside ``Accuracy``, say -- is read as a *component* of
that outcome: an alternative description of the same result rather than an
independent quantity, so it is reported alongside the outcome but never used to
predict it. Every other discovered metric is a predictor. Both lists can be
overridden in configuration; the convention is only the default.

Ordering is an invariant the file does not record. ``/reference/levels`` is in
``/reference/group_id`` order, and the per-observation tables are in
``/observations/id`` order. The reader treats both as given, and the extractor
re-checks the co-indexing it depends on.

Anything else present in a file is simply not read: the reader opens it
read-only and touches only the paths above.
"""

# -------------------------------------------------------------------------------------------------
# Version
# -------------------------------------------------------------------------------------------------
# Written into every file and checked on every read. A file that predates the
# attribute, or carries a version this build does not know, is rejected with a
# message naming both -- which is the whole reason the attribute exists.

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1,)

# -------------------------------------------------------------------------------------------------
# Groups
# -------------------------------------------------------------------------------------------------

GROUP_OBSERVATIONS = 'observations'
GROUP_SIGNAL = 'signal'
GROUP_REFERENCE = 'reference'

# -------------------------------------------------------------------------------------------------
# Root attributes
# -------------------------------------------------------------------------------------------------

ATTR_SCHEMA_VERSION = 'schema_version'
ATTR_RUN_ID = 'run_id'
ATTR_RESPONSE_NAME = 'response_name'
ATTR_CREATED_UTC = 'created_utc'

# -------------------------------------------------------------------------------------------------
# Dataset attributes
# -------------------------------------------------------------------------------------------------

# Column names of the metrics matrix, one per column and in column order. This
# is the only place a metric name appears anywhere in the format.
ATTR_METRIC_NAMES = 'metric_names'

# -------------------------------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------------------------------

DS_OBSERVATION_ID = 'id'
DS_OBSERVATION_GROUP_ID = 'group_id'
DS_METRICS = 'metrics'
DS_REFERENCE_LEVELS = 'levels'
DS_REFERENCE_GROUP_ID = 'group_id'

# -------------------------------------------------------------------------------------------------
# Metric roles
# -------------------------------------------------------------------------------------------------

# The metric a run is organised around: it defines the bins every other metric
# is summarised against, and it is the outcome the multivariable model predicts.
# A default rather than a fixed name -- configuration may point at any metric a
# file declares.
DEFAULT_OUTCOME_METRIC = 'Accuracy'

# Separates an outcome from the component of it named after it, as described in
# the module docstring.
COMPONENT_SEPARATOR = '_'
