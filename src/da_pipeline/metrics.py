"""Which metrics a run analyses, and what part each of them plays.

The extract format carries its metric set as data: a file declares its own
column names, and neither the reader nor the analysis layer knows any of them in
advance. That leaves one question this module answers -- given the metrics a set
of files turned out to declare, and what the run asked for, which metric is
which?

Three roles, because the analysis treats them differently:

``outcome``
    The metric the run is organised around. It defines the equal-count bins
    every other metric is summarised against, and it is what the multivariable
    model predicts. Exactly one.
``components``
    Alternative descriptions of the same result as the outcome -- error modes
    beside an accuracy, say. They are reported next to the outcome, and are
    deliberately never used to predict it: a component is a restatement of the
    outcome, so a model carrying one would be predicting a number from itself.
``predictors``
    Everything else. Independent quantities that may or may not relate to the
    outcome, and the ones the model and the correlations are actually about.

By default the roles are derived from the discovered names using the format's
naming convention (see :mod:`da_pipeline.constants.schema`): a metric called
``<outcome>_<something>`` is a component of that outcome, and every other
discovered metric is a predictor. Configuration can state any of the three
explicitly instead.

Discovery is permissive and selection is strict, which are two different things
and the distinction matters:

* The pipeline accepts whatever metric set the files declare. A metric the run
  does not mention is simply not analysed -- no error, no special case.
* A metric the run *names* and the files do not declare is an error, and the
  message lists what they did declare. A typo would otherwise mean the analysis
  somebody asked for silently did not run, which is worse than a run that stops.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .constants import schema as sch

logger = logging.getLogger(__name__)


class MetricSelectionError(ValueError):
    """The run names metrics the extract files do not declare, or names them incoherently.

    Raised once the files have been opened and their metric sets read, which is
    the earliest point at which the question can be answered at all: a config
    built in a notebook has no files to check itself against.
    """


def _as_tuple(values, cast=str):
    """Freeze a sequence into a tuple, or pass None through untouched."""
    if values is None:
        return None
    if isinstance(values, str):
        return (cast(values),)
    return tuple(cast(value) for value in values)


def _as_pairs(labels) -> tuple[tuple[str, str], ...]:
    """Normalise a mapping or an iterable of pairs into a tuple of pairs.

    Kept as pairs rather than a dict so the selection stays hashable, immutable
    and picklable -- the config it belongs to is sent to worker processes.
    """
    if labels is None:
        return ()
    items = labels.items() if isinstance(labels, Mapping) else labels
    return tuple((str(name), str(label)) for name, label in items)


def _format(names: Iterable[str]) -> str:
    return ', '.join(names)


# -------------------------------------------------------------------------------------------------
# What the run asks for
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSelection:
    """What a run asks of whatever metrics its extracts turn out to declare.

    Every field is optional, and the default selection -- analyse everything,
    with the roles derived by convention -- is what makes the pipeline runnable
    against a file it has never seen.

    Args:
        names: Metrics to analyse. ``None`` means every metric the files
            declare. The outcome is always analysed, so naming a subset that
            omits it is rejected rather than quietly corrected.
        outcome: The metric the run is organised around.
        predictors: Override for the derived predictor list. Must be a subset of
            the analysed metrics.
        components: Override for the derived component list. Must be a subset of
            the analysed metrics.
        labels: ``{metric: display label}``, used in figures and the report.
            A metric with no label is displayed under its own name.
    """

    names: tuple[str, ...] | None = None
    outcome: str = sch.DEFAULT_OUTCOME_METRIC
    predictors: tuple[str, ...] | None = None
    components: tuple[str, ...] | None = None
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'names', _as_tuple(self.names))
        object.__setattr__(self, 'predictors', _as_tuple(self.predictors))
        object.__setattr__(self, 'components', _as_tuple(self.components))
        object.__setattr__(self, 'labels', _as_pairs(self.labels))

        if not self.outcome:
            raise MetricSelectionError("outcome: must name a metric.")

        for field_name in ('names', 'predictors', 'components'):
            values = getattr(self, field_name)
            if values is None:
                continue
            repeated = sorted({v for v in values if values.count(v) > 1})
            if repeated:
                raise MetricSelectionError(
                    f"{field_name}: {_format(repeated)} listed more than once."
                )

        if self.predictors and self.components:
            both = sorted(set(self.predictors) & set(self.components))
            if both:
                raise MetricSelectionError(
                    f"{_format(both)} cannot be both a predictor and a component of "
                    f"{self.outcome}; a component restates the outcome, so it is "
                    f"never used to predict it."
                )

        labelled = [name for name, _ in self.labels]
        repeated_labels = sorted({n for n in labelled if labelled.count(n) > 1})
        if repeated_labels:
            raise MetricSelectionError(
                f"labels: {_format(repeated_labels)} labelled more than once."
            )

    def describe(self) -> list[str]:
        """The selection as log lines, one per setting."""
        return [
            f"- metrics.names: {list(self.names) if self.names else 'every metric declared'}",
            f"- metrics.outcome: {self.outcome}",
            f"- metrics.predictors: {list(self.predictors) if self.predictors else 'derived'}",
            f"- metrics.components: {list(self.components) if self.components else 'derived'}",
            f"- metrics.labels: {dict(self.labels) or 'metric names as written'}",
        ]


# -------------------------------------------------------------------------------------------------
# What the files turned out to hold
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricRoles:
    """The resolved metric set of one run: which metric plays which part.

    Built by :func:`resolve_metric_roles` once the files have been read, and
    passed down through the pipeline from there. Every stage that needs to know
    what to extract, summarise, plot or model asks this object, which is what
    keeps metric names out of the analysis layer entirely.
    """

    outcome: str
    components: tuple[str, ...]
    predictors: tuple[str, ...]
    labels: tuple[tuple[str, str], ...] = ()

    @property
    def outcome_metrics(self) -> tuple[str, ...]:
        """The outcome and its components: one frame, one set of figures."""
        return (self.outcome, *self.components)

    @property
    def analysed(self) -> tuple[str, ...]:
        """Every metric this run reads out of the files."""
        return (self.outcome, *self.components, *self.predictors)

    def label(self, metric: str) -> str:
        """The display name for a metric, falling back to the metric itself."""
        return dict(self.labels).get(metric, metric)

    def describe(self) -> list[str]:
        """The resolved roles as log lines, one per role."""
        return [
            f"- outcome metric: {self.outcome}",
            f"- component metrics: {_format(self.components) or 'none'}",
            f"- predictor metrics: {_format(self.predictors) or 'none'}",
        ]


# -------------------------------------------------------------------------------------------------
# Resolution
# -------------------------------------------------------------------------------------------------

def _check_declared(field_name: str, wanted, declared: Sequence[str]) -> None:
    """Reject any name the files do not declare, listing what they do.

    The message carries the whole declared set on purpose. A metric name is
    discovered from the data, so a caller who mistyped one has no other way to
    find out what was available, and the alternative -- dropping the name and
    running a smaller analysis than the one that was asked for -- fails quietly
    at exactly the moment somebody is relying on the result.
    """
    if not wanted:
        return

    unknown = [name for name in wanted if name not in declared]
    if unknown:
        raise MetricSelectionError(
            f"{field_name}: {_format(unknown)} "
            f"{'is' if len(unknown) == 1 else 'are'} not declared by the extract "
            f"files. Declared: {_format(declared)}."
        )


def resolve_metric_roles(
    declared: Sequence[str],
    selection: MetricSelection | None = None,
) -> MetricRoles:
    """Work out which discovered metric plays which part in this run.

    Args:
        declared: The metric names the extract files carry, as read from the
            files themselves.
        selection: What the run asked for. ``None`` takes every default: analyse
            everything, with the roles derived by convention.

    Returns:
        The resolved roles, ready to be handed to the extraction and analysis
        stages.

    Raises:
        MetricSelectionError: if the run names a metric the files do not
            declare, or names a set of roles that cannot hold at once.
    """

    selection = selection or MetricSelection()
    declared = tuple(declared)

    if not declared:
        raise MetricSelectionError(
            "The extract files declare no metrics, so there is nothing to analyse."
        )

    _check_declared("metric_names", selection.names, declared)
    _check_declared("outcome_metric", (selection.outcome,), declared)
    _check_declared("predictor_metric_names", selection.predictors, declared)
    _check_declared("component_metric_names", selection.components, declared)
    _check_declared("metric_labels", [name for name, _ in selection.labels], declared)

    analysed = selection.names if selection.names is not None else declared

    if selection.outcome not in analysed:
        raise MetricSelectionError(
            f"metric_names: the outcome metric {selection.outcome} is not among "
            f"{_format(analysed)}. The outcome defines the bins every other metric "
            f"is summarised against, so it is always analysed."
        )

    for field_name, values in (
        ("predictor_metric_names", selection.predictors),
        ("component_metric_names", selection.components),
    ):
        if values:
            outside = [name for name in values if name not in analysed]
            if outside:
                raise MetricSelectionError(
                    f"{field_name}: {_format(outside)} not among the analysed metrics "
                    f"({_format(analysed)}). Add them to metric_names, or drop them here."
                )

    # A metric named after the outcome restates it. Derived rather than listed,
    # so a producer can add a component without the pipeline being told about it.
    prefix = f"{selection.outcome}{sch.COMPONENT_SEPARATOR}"
    components = (
        selection.components
        if selection.components is not None
        else tuple(name for name in analysed if name.startswith(prefix))
    )

    predictors = (
        selection.predictors
        if selection.predictors is not None
        else tuple(
            name for name in analysed
            if name != selection.outcome and name not in components
        )
    )

    overlap = sorted(set(predictors) & set(components))
    if overlap:
        raise MetricSelectionError(
            f"{_format(overlap)} cannot be both a predictor and a component of "
            f"{selection.outcome}."
        )

    roles = MetricRoles(
        outcome=selection.outcome,
        components=components,
        predictors=predictors,
        labels=selection.labels,
    )

    # Discovery is permissive: a metric the run did not ask for is not an error,
    # but it is worth saying out loud that it was read and set aside.
    ignored = [name for name in declared if name not in roles.analysed]
    if ignored:
        logger.info(
            f"- metrics declared but not analysed: {_format(ignored)}"
        )

    return roles
