"""What a run actually found, extracted once from the analysis frames.

The tables hold everything, but a table cannot say which of its rows mattered.
This module reads them and writes down the findings in sentences: which
conditions separated, which metric is least reproducible, which relationship
pooling manufactured, whether the model transfers to a run it never saw.

Every sentence here is computed from the frames rather than written by hand, so
a report cannot drift out of step with the data it describes. Where a finding
comes with a caution -- and most of them do -- the caution is attached to the
finding rather than left in a footnote, because the two get separated the moment
anyone quotes the number.

The phrasing is deliberately careful in two places, both of which are easy to
get wrong and were got wrong during this project's development:

* a comparison that does not reach significance is reported as *not separated*,
  never as "the same" or "indistinguishable". At these replicate counts the
  smallest attainable p-value can be large even when every pairing went one way,
  so absence of evidence here is close to worthless as evidence of absence. What
  "large" comes to is computed from the run's own replicate counts rather than
  asserted here, because it changes with them;
* every condition finding is phrased as a difference *associated with* the
  condition, never as the condition having caused it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from ..constants import columns as col
from .replicates import p_floor_span_clause, replicate_phrase

logger = logging.getLogger(__name__)

# The conventional threshold, applied to the corrected p-value. Named rather
# than inlined so the report can state what it used.
SIGNIFICANCE_THRESHOLD = 0.05

# Above this, a metric's run-to-run spread is at least as large as the spread
# associated with the condition, which is the point at which a difference stops
# being well supported however many observations were measured.
RUN_DOMINATES_RATIO = 1.0

# A within-run coefficient this close to zero counts as "nothing there" when
# describing a pooled coefficient that disagrees with it.
NEGLIGIBLE_R = 0.1

# Enough to show the shape of the result without turning one sentence into a
# table; the full set is in the pairwise export beside it.
MAX_LISTED_COMPARISONS = 6

# Stable slugs identifying each finding. The report routes its tabs by these, so
# renaming one moves a finding between tabs -- they are an interface, not labels.
KIND_EFFECT_SIZE = 'effect_size'
KIND_PAIRWISE = 'pairwise'
KIND_REPRODUCIBILITY = 'reproducibility'
KIND_RUN_SPREAD = 'run_spread'
KIND_CORRELATIONS = 'correlations'
KIND_MODEL = 'model'


@dataclass(frozen=True)
class Finding:
    """One question, what the run answered, and what limits the answer.

    ``kind`` is a stable slug naming which question this is. The report groups
    its tabs by it, so the routing lives with the finding rather than in a
    lookup the two have to keep in step.
    """

    kind: str
    question: str
    answer: str
    caution: str | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "question": self.question,
            "answer": self.answer,
            "caution": self.caution,
        }


@dataclass(frozen=True)
class RunFindings:
    """Everything a report or an evidence pack needs about one run."""

    response_name: str
    design: dict
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "response_name": self.response_name,
            "design": self.design,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _first(df: pl.DataFrame | None, column: str):
    """The first value of a column, or None if it is not there to take."""
    if df is None or df.is_empty() or column not in df.columns:
        return None
    return df[column][0]


def _fmt(value, places: int = 3, default: str = "not available") -> str:
    """A number for prose, or a phrase saying it could not be computed."""
    if value is None:
        return default
    return f"{value:.{places}f}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------

def describe_design(
    variance_df: pl.DataFrame,
    condition_run_counts: dict,
    response_name: str,
    outcome_metric: str,
    metrics: dict | None = None,
) -> dict:
    """How much data the run rests on, and what the replicate count really is.

    The observation count is by far the largest number the pipeline produces and
    the least meaningful as a sample size, so it is reported beside the run count
    rather than on its own.

    The metric set is recorded too. It is discovered from the extract files
    rather than fixed by the code, so a reader of the pack has no other way to
    tell which metrics this run carried or what part each of them played.
    """

    n_observations = _first(variance_df, col.N_OBSERVATIONS)
    n_runs = _first(variance_df, col.N_RUNS_TOTAL)

    return {
        "response_name": response_name,
        "outcome_metric": outcome_metric,
        "metrics": metrics or {},
        "n_conditions": len(condition_run_counts),
        "n_runs": int(n_runs) if n_runs is not None else sum(condition_run_counts.values()),
        "runs_per_condition": dict(condition_run_counts),
        "n_observations": int(n_observations) if n_observations is not None else None,
        "unit_of_analysis": (
            "The run. All the observations in one run were measured together, on "
            "the same equipment, under the same setup, on the same day, so they "
            "are repeated measurements of that run rather than independent "
            "observations of the condition. Every statistic supporting a claim "
            "about a condition is computed on one value per run."
        ),
    }


# ---------------------------------------------------------------------------
# Individual findings
# ---------------------------------------------------------------------------

def effect_size_finding(
    effect_size_df: pl.DataFrame | None,
    response_name: str,
    condition_run_counts: dict | None = None,
) -> Finding | None:
    """How much of the response's run-level variation goes with the condition.

    ``condition_run_counts`` only reaches the caution, which has to say how wide
    the interval is wide *from*. Without it the caution still stands, in the
    vaguer wording ``replicates.py`` falls back to.
    """

    if effect_size_df is None or effect_size_df.is_empty():
        return None

    parts = []
    for row in effect_size_df.iter_rows(named=True):
        eta = row.get(col.ETA_SQUARED)
        if eta is None:
            continue
        interval = ""
        low, high = row.get(col.ETA_SQUARED_CI_LOW), row.get(col.ETA_SQUARED_CI_HIGH)
        if low is not None and high is not None:
            interval = f" [{low:.2f}-{high:.2f}]"
        parts.append(f"level {row[col.LEVEL]} {eta:.2f}{interval}")

    if not parts:
        return None

    n_runs = _first(effect_size_df, col.N_RUNS_TOTAL)
    basis = (
        f" Computed on {_plural(int(n_runs), 'run')}, not on observations."
        if n_runs is not None
        else ""
    )

    return Finding(
        kind=KIND_EFFECT_SIZE,
        question=f"How much of the variation in {response_name} goes with the condition?",
        answer=(
            "Eta-squared on the per-run medians, by level: "
            + "; ".join(parts) + "." + basis
        ),
        caution=(
            "This is an omnibus figure: it says the conditions differ somewhere, "
            "not which of them differ. The interval is a percentile bootstrap over "
            f"runs, and at {replicate_phrase(condition_run_counts)} it is wide -- "
            "that width is the honest signal about how much the design supports."
        ),
    )


def _describe_comparison(row: dict) -> str | None:
    """One separated comparison, with CLES turned to face the winner.

    CLES is directional: it is the probability that a run of ``condition_a``
    scores above a run of ``condition_b``, so a decisive win for B comes back as
    0.00. Printed beside "Condition_B higher" that reads as a contradiction, and
    two different pairs that B both won look like the same row repeated. Naming
    both conditions and reporting the figure from the winner's side fixes each
    of those without changing the number: 0.00 for A over B is 1.00 for B over A.
    """

    higher = row.get(col.HIGHER_CONDITION)
    cles = row.get(col.CLES)

    if higher is None or cles is None:
        return None

    a, b = row[col.CONDITION_A], row[col.CONDITION_B]
    other = b if higher == a else a
    oriented = cles if higher == a else 1.0 - cles

    return (
        f"{row[col.METRIC]}: {higher} above {other} "
        f"(CLES {oriented:.2f}, adjusted p {row[col.P_VALUE_ADJUSTED]:.3f})"
    )


def _run_counts_from_pairwise(pairwise_df: pl.DataFrame) -> dict:
    """Each condition's run count, read back off the comparisons themselves.

    The pairwise table records the group sizes it used in every row, so the
    caution can state the floor under its own p-values without the design being
    threaded in beside it. Reading them here also keeps the caution true of the
    comparisons that were actually made: a condition dropped for having too few
    runs never appears in this table, and should not appear in a sentence about
    what the table can support either.
    """

    counts = {}
    for side, size in ((col.CONDITION_A, col.N_RUNS_A), (col.CONDITION_B, col.N_RUNS_B)):
        if side not in pairwise_df.columns or size not in pairwise_df.columns:
            continue
        for name, n_runs in pairwise_df.select(side, size).iter_rows():
            if name is not None and n_runs is not None:
                counts[name] = int(n_runs)

    return counts


def pairwise_finding(pairwise_df: pl.DataFrame | None) -> Finding | None:
    """Which condition pairs separated, on which metrics."""

    if pairwise_df is None or pairwise_df.is_empty():
        return None
    if col.P_VALUE_ADJUSTED not in pairwise_df.columns:
        return None

    run_counts = _run_counts_from_pairwise(pairwise_df)

    separated = pairwise_df.filter(
        pl.col(col.P_VALUE_ADJUSTED).is_not_null()
        & (pl.col(col.P_VALUE_ADJUSTED) < SIGNIFICANCE_THRESHOLD)
    )

    if separated.is_empty():
        answer = (
            f"No condition pair separated on any metric at an adjusted "
            f"p below {SIGNIFICANCE_THRESHOLD} across {pairwise_df.height} comparisons."
        )
    else:
        clearest = separated.sort(col.P_VALUE_ADJUSTED).head(MAX_LISTED_COMPARISONS)
        lines = [_describe_comparison(row) for row in clearest.iter_rows(named=True)]
        lines = [line for line in lines if line]

        remaining = separated.height - len(lines)
        more = f", and {remaining} more" if remaining > 0 else ""

        answer = (
            f"{separated.height} of {pairwise_df.height} condition-pair comparisons "
            f"separated at an adjusted p below {SIGNIFICANCE_THRESHOLD}"
            + (f". The clearest: {'; '.join(lines)}{more}." if lines else ".")
        )

    return Finding(
        kind=KIND_PAIRWISE,
        question="Which conditions actually differ, and by how much?",
        answer=answer,
        caution=(
            "A pair that did not separate has not been shown to be equivalent. "
            f"With {replicate_phrase(run_counts)} "
            f"{p_floor_span_clause(run_counts)}, even when every pairing went the "
            "same way, so a large p here is weak evidence of anything. Read the "
            "effect size beside it. Every difference is an association with the "
            "condition, not a demonstration that the condition caused it."
        ),
    )


def reproducibility_finding(variance_df: pl.DataFrame) -> Finding | None:
    """Which metrics vary more between runs than between conditions."""

    if variance_df is None or variance_df.is_empty():
        return None

    usable = variance_df.filter(pl.col(col.RUN_TO_CONDITION_RATIO).is_not_null())
    if usable.is_empty():
        return None

    fragile = usable.filter(pl.col(col.RUN_TO_CONDITION_RATIO) >= RUN_DOMINATES_RATIO)
    worst = usable.sort(col.RUN_TO_CONDITION_RATIO, descending=True).row(0, named=True)

    if fragile.is_empty():
        answer = (
            "Every metric's condition-associated variation exceeds its run-to-run "
            f"variation. The narrowest margin is {worst[col.METRIC]}, where run-to-run "
            f"spread is {worst[col.RUN_TO_CONDITION_RATIO]:.2f} times the "
            f"condition-associated spread."
        )
    else:
        names = ", ".join(
            f"{r[col.METRIC]} ({r[col.RUN_TO_CONDITION_RATIO]:.2f}x)"
            for r in fragile.sort(col.RUN_TO_CONDITION_RATIO, descending=True).iter_rows(named=True)
        )
        answer = (
            f"{_plural(fragile.height, 'metric')} vary at least as much between runs "
            f"of one condition as between conditions: {names}. Two runs of the same "
            f"condition already differ by about as much as the conditions do."
        )

    return Finding(
        kind=KIND_REPRODUCIBILITY,
        question="How reproducible is each difference across runs?",
        answer=answer,
        caution=(
            "This is a decomposition of the variation actually observed, not an "
            "estimate of variance components, so it reports that present evidence "
            "for a difference is weak. It does not forecast what a future run "
            "will show."
        ),
    )


def run_spread_finding(run_spread_df: pl.DataFrame) -> Finding | None:
    """Whether any condition's result rests on a single run."""

    if run_spread_df is None or run_spread_df.is_empty():
        return None
    if col.MAX_LEAVE_ONE_OUT_SHIFT not in run_spread_df.columns:
        return None

    usable = run_spread_df.filter(
        pl.col(col.MAX_LEAVE_ONE_OUT_SHIFT).is_not_null()
        & pl.col(col.CONDITION_MEAN).is_not_null()
        & (pl.col(col.CONDITION_MEAN).abs() > 0)
    ).with_columns(
        (pl.col(col.MAX_LEAVE_ONE_OUT_SHIFT) / pl.col(col.CONDITION_MEAN).abs() * 100.0)
        .alias("_shift_pct")
    )

    if usable.is_empty():
        return None

    worst = usable.sort("_shift_pct", descending=True).row(0, named=True)

    return Finding(
        kind=KIND_RUN_SPREAD,
        question="Is any condition's result carried by a single run?",
        answer=(
            f"Ranked by how far a condition's mean moves when its most influential "
            f"run is dropped, relative to that condition's level, the most sensitive "
            f"combination is {worst[col.METRIC]} in {worst[col.CONDITION]}: dropping "
            f"{worst[col.MOST_INFLUENTIAL_RUN]} moves the mean by "
            f"{worst[col.MAX_LEAVE_ONE_OUT_SHIFT]:.4g} against a level of "
            f"{worst[col.CONDITION_MEAN]:.4g}, a shift of {worst['_shift_pct']:.0f}%."
        ),
        caution=(
            "Relative shift is what makes metrics on different scales comparable, "
            "but it also means a metric whose level sits near zero will top this "
            "ranking on a small absolute movement -- read the raw shift beside the "
            "percentage before concluding a result is fragile. The shift is "
            "measured on the mean of the run medians rather than the median, "
            "because the median is robust by design: with runs at 9.9, 10.0, 10.1 "
            "and 40.0, dropping any of them moves the median identically, so a "
            "leave-one-out on the median cannot tell an extreme run from an "
            "ordinary one. Full per-condition detail is in "
            "run_spread_by_condition."
        ),
    )


def correlation_finding(
    pooled_df: pl.DataFrame,
    within_run_df: pl.DataFrame,
) -> Finding | None:
    """How far the pooled view and the within-run view disagree."""

    if pooled_df is None or within_run_df is None:
        return None
    if pooled_df.is_empty() or within_run_df.is_empty():
        return None

    keys = [col.CONDITION, col.CORR_VAR_X, col.CORR_VAR_Y]
    joined = (
        pooled_df.select(keys + [col.CORR_PEARSON_R])
        .join(within_run_df.select(keys + [col.CORR_PEARSON_R_MEDIAN]), on=keys, how="inner")
        .drop_nulls([col.CORR_PEARSON_R, col.CORR_PEARSON_R_MEDIAN])
    )

    if joined.is_empty():
        return None

    joined = joined.with_columns(
        (pl.col(col.CORR_PEARSON_R) - pl.col(col.CORR_PEARSON_R_MEDIAN)).abs().alias("_gap"),
        (
            pl.col(col.CORR_PEARSON_R).sign() != pl.col(col.CORR_PEARSON_R_MEDIAN).sign()
        ).alias("_reversed"),
    )

    reversals = int(joined["_reversed"].sum())
    worst = joined.sort("_gap", descending=True).row(0, named=True)

    manufactured = (
        abs(worst[col.CORR_PEARSON_R_MEDIAN]) < NEGLIGIBLE_R
        and abs(worst[col.CORR_PEARSON_R]) >= NEGLIGIBLE_R
    )

    return Finding(
        kind=KIND_CORRELATIONS,
        question="Which relationships survive being computed inside a run?",
        answer=(
            f"Of {joined.height} condition-and-pair combinations, {reversals} come out "
            f"with opposite signs pooled and within-run. The widest disagreement is "
            f"{worst[col.CORR_VAR_X]} against {worst[col.CORR_VAR_Y]} in "
            f"{worst[col.CONDITION]}: {worst[col.CORR_PEARSON_R]:+.3f} pooled, "
            f"{worst[col.CORR_PEARSON_R_MEDIAN]:+.3f} within-run"
            + (
                " -- a relationship that exists only between runs, not between the "
                "observations inside one."
                if manufactured
                else "."
            )
        ),
        caution=(
            "Pooled coefficients are honest descriptions of the observation population "
            "measured, and are not evidence: observations from several runs are not "
            "independent observations. No p-values accompany them for that reason. "
            "The reproducibility claim the design supports is the within-run table's "
            "count of how many runs agreed on a sign. The between-run view answers a "
            "third question and is confounded with the condition effect, since the "
            "runs span every condition."
        ),
    )


def model_finding(model_df: pl.DataFrame) -> Finding | None:
    """What predicts the outcome within a run, and whether it generalises."""

    if model_df is None or model_df.is_empty():
        return None

    ordered = model_df.sort(pl.col(col.COEFFICIENT_MEDIAN).abs(), descending=True)
    top = ordered.row(0, named=True)

    parts = [
        f"{r[col.PREDICTOR]} {r[col.COEFFICIENT_MEDIAN]:+.2f}"
        + (
            f" [{r[col.COEFFICIENT_CI_LOW]:+.2f},{r[col.COEFFICIENT_CI_HIGH]:+.2f}]"
            if r[col.COEFFICIENT_CI_LOW] is not None
            else ""
        )
        + f" ({r[col.COEFFICIENT_SAME_SIGN]}/{r[col.N_RUNS_MODELLED]} runs agree)"
        for r in ordered.iter_rows(named=True)
    ]

    held_out = top.get(col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN)
    in_sample = top.get(col.R_SQUARED_IN_SAMPLE_MEDIAN)
    predicted = top.get(col.N_RUNS_PREDICTED)
    n_runs = top.get(col.N_RUNS_MODELLED)

    generalisation = (
        f" Fitted on the other runs and applied to a held-out one, the median "
        f"R-squared is {_fmt(held_out)} against {_fmt(in_sample)} in sample, and "
        f"{predicted} of {n_runs} runs were predicted better than their own mean."
        if held_out is not None
        else " Cross-validation could not be run on this many runs."
    )

    return Finding(
        kind=KIND_MODEL,
        question=f"What predicts {top[col.OUTCOME]} within a run?",
        answer=(
            "Standardised coefficients from a regression fitted separately inside "
            "each run, largest first: " + "; ".join(parts) + "." + generalisation
        ),
        caution=(
            "These are associations between observations inside a run. They do not show "
            "that changing a predictor on a given observation would change its outcome, "
            "and nothing here is a mediation analysis. The condition cannot appear "
            "as a predictor: it is constant within a run, so a per-run model cannot "
            "estimate it. A negative out-of-sample R-squared means the fitted "
            "coefficients predicted that run worse than its own mean would have."
        ),
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_run_findings(
    *,
    response_name: str,
    outcome_metric: str,
    condition_run_counts: dict,
    metrics: dict | None = None,
    effect_size_df: pl.DataFrame | None,
    pairwise_df: pl.DataFrame | None,
    variance_df: pl.DataFrame,
    run_spread_df: pl.DataFrame,
    correlations_df: pl.DataFrame,
    within_run_correlations_df: pl.DataFrame,
    model_df: pl.DataFrame,
) -> RunFindings:
    """Read every analysis frame and write down what the run found.

    Ordered as a reader would ask the questions: does anything differ, which
    things differ, would it happen again, is one run carrying it, what relates
    to what, and what predicts the outcome. A frame that could not be computed
    contributes no finding rather than an empty one.
    """

    candidates = [
        effect_size_finding(effect_size_df, response_name, condition_run_counts),
        pairwise_finding(pairwise_df),
        reproducibility_finding(variance_df),
        run_spread_finding(run_spread_df),
        correlation_finding(correlations_df, within_run_correlations_df),
        model_finding(model_df),
    ]

    findings = [f for f in candidates if f is not None]

    logger.info(f"- Findings: {len(findings)} of {len(candidates)} could be computed")

    return RunFindings(
        response_name=response_name,
        design=describe_design(
            variance_df, condition_run_counts, response_name,
            outcome_metric, metrics,
        ),
        findings=findings,
    )
