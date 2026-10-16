"""The instructions that travel with the evidence pack.

A language model handed a table of coefficients and p-values will do the
obvious, confident, wrong thing with it: report that two conditions are
equivalent because a p-value was large, read a pooled correlation as evidence,
or explain that raising signal-to-noise would raise the outcome metric. Every one of those
mistakes was made by a capable reader during this project's development, more
than once, including by the model that helped build it.

None of them are careless. They are what the numbers look like they mean. The
only reliable fix is to state the rules in advance, in as many words, next to
the data -- which is what this file is. It is written as a briefing rather than
a list of prohibitions, because a reader who understands *why* the run is the
replicate will avoid a dozen errors nobody thought to enumerate.

The template carries no results itself. Run specifics are filled in at the end
so the same briefing serves every deployment of the pipeline.

That includes the replicate counts the rules argue from. Rules 1, 2 and 9 each
turn on how many runs a condition has -- the floor under the p-value, what a
CLES of 1.00 amounts to, why the intervals are wide -- and a rule that quotes
the wrong count is not merely stale, it is a rule the reader can check against
section 2 and find false. They are built from ``runs_per_condition`` in
``replicates.py`` rather than written out here.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from .findings import RunFindings
from .replicates import p_floor_clause, replicate_phrase

logger = logging.getLogger(__name__)

PROMPT_FNAME = 'prompt.md'

PROMPT_TEMPLATE = """\
# Interpreting a Data Analytics pipeline run

You are interpreting the output of a data analytics pipeline that compares
experimental conditions. The accompanying `evidence_pack.json` holds every
derived statistic the analysis produced. Read this briefing first: the numbers
are straightforward, but several of them mean something narrower than they
appear to, and the difference decides whether an interpretation is useful or
misleading.

---

## 1. What was measured

One **run** of the experiment measures many thousands of **observations** in
parallel. Each observation is measured over a series of **steps**, and a known
reference states what the correct response is at each step, so every observation
can be scored against ground truth. The **level** is how much response the
reference expects at a given step: `level 0` none, `level 1` one unit, `level 2` two.

The metrics below are the ones *this* run carried. They are not fixed by the
pipeline: each extract file declares its own metric set, and the run's
configuration decides which of them to analyse and what part each one plays. Do
not assume any metric exists that is not listed here.

From the raw per-observation signal the pipeline derives:

- the **response** (here `{response_name}`) — the raw measured signal per
  observation per step, split by level. The most direct measurement there is;
  everything below is derived from it.
- **signal_to_noise_ratio** — per observation, the separation between steps where
  a response was expected and steps where none was. Treated as a predictor. Can
  be infinite by design when the baseline is zero; those rows are excluded
  wherever they appear.

Every other metric plays one of three roles, and the role decides how it may be
read:

**outcome** — the metric this run is organised around. It defines the
equal-count bins every other metric is summarised against, and it is what the
multivariable model predicts. There is exactly one.

**component of the outcome** — another description of the same result as the
outcome, not an independent quantity. Deliberately excluded from the model:
regressing the outcome on one would mostly recover the arithmetic relating the
two. Never read a component as a cause or a predictor of the outcome.

**predictor** — an independent per-observation quantity that may or may not
relate to the outcome. These are what the correlations and the model are about.

This run carried:

| Metric | Role |
|---|---|
{metric_rows}

A **condition** is an experimental group — a setting, a protocol, a formulation.
Each condition comprises several **runs**, and each run
contributes thousands of observations. Comparing conditions is the point of the
exercise.

## 2. The design, and the one fact that governs everything

{design_summary}

**The run is the experimental unit. The observation is not.**

All the observations in one run were measured together, on the same equipment,
under the same setup, on the same day. They are repeated measurements of that run,
not independent observations of the condition. This single fact is why the analysis
looks the way it does, and it is the thing most easily lost when reading the
output.

To see what is at stake, here is the comparison on this project's own demo
dataset, which anyone can regenerate with `da-pipeline-demo`. Condition_A and
Condition_B differ in median `Accuracy` by 0.013. Pooling their 2,880
observations returns `p = 1.9e-08`. The same comparison on the six run medians
that produced those observations -- three runs each -- returns `p = 0.2`.

The first number is not a stronger version of the second. It is an artefact of
treating 2,880 correlated measurements as 2,880 independent ones. The second is
what the design supports, and at three runs against three, `p = 0.2` is close to
the smallest value the test can return at all.

So every statistic here that supports a claim about a condition is computed on
**one value per run**. When you cite a sample size, cite the run count.

## 3. Rules you must follow

These are not stylistic preferences. Each corresponds to a specific way this
data has been misread.

{rule_equivalence}

{rule_effect_size}

3. **Pooled correlations are descriptions, not evidence.** They honestly
   describe the observation population measured. They carry no p-values, deliberately.
   Do not use one to argue that a relationship is real.

4. **Prefer the within-run correlations for any reproducibility claim.** Each
   is computed inside one run and then summarised across runs. The count of runs
   agreeing on the sign is the claim the replicate count supports. Where the
   pooled and within-run coefficients disagree — and in this project they often
   do, sometimes reversing sign — the within-run one is the one about observations.

5. **Between-run correlations are confounded with the condition.** The runs span
   every condition, so a strong coefficient there may be re-measuring the
   condition effect rather than a run-level relationship. Say so when you cite one.

6. **Every condition result is an association.** The design supports "this
   metric differs between these conditions". It does not support "the condition
   improved the metric by way of this mechanism". Do not write causal or
   mediation language.

7. **Model coefficients are within-run associations between observations.** They
   do not license "raising signal-to-noise would raise the outcome metric". The model cannot
   contain the condition as a predictor — the condition is constant within a run,
   so a per-run model cannot estimate it even in principle. Do not suggest adding it.

8. **A negative out-of-sample R-squared is a finding, not an error.** It means
   the coefficients fitted on the other runs predicted that run worse than that
   run's own mean would have. Runs that fail this way are worth naming.

{rule_intervals}

10. **Distinguish "the difference is small" from "the evidence is weak".** The
    variance decomposition tells you which. A metric whose run-to-run variation
    is as large as its condition-associated variation has weak support for a
    condition difference however many observations were measured.

## 4. What is in the evidence pack

| Key | Contents |
|---|---|
| `design` | condition and run counts, and the unit of analysis |
| `findings` | the pipeline's own summary of what the run found, each with its caution |
| `tables.effect_size` | eta-squared per level, on run medians, with bootstrap intervals and a one-way ANOVA p |
| `tables.pairwise_conditions` | every condition pair on every metric: CLES with interval, Mann-Whitney p, Benjamini-Hochberg (1995) corrected within each metric — the family is the metric, so `p_adjust_family` says what a given adjusted p was corrected against |
| `tables.variance_decomposition` | each metric's variation split into condition / run-within-condition / observation-within-run. The three shares sum to 100 exactly |
| `tables.run_spread_by_condition` | each condition's own run medians: range, IQR, and how far the result moves if the most influential run is dropped |
| `tables.correlations` | pooled, per condition |
| `tables.correlations_within_run` | computed inside each run, summarised across runs, with sign agreement |
| `tables.correlations_between_run` | one median per run, correlated across all runs |
| `tables.outcome_model` | per-predictor standardised coefficients with bootstrap intervals, VIF, and leave-one-run-out cross-validation |
| `tables.outcome_model_by_run` | the per-run fits behind that summary |
| `tables.bin_summaries` | every metric summarised against equal-count bins of the outcome metric, computed per condition. The bins hold equal numbers of observations by rank, so bin 4 spans a different range in each condition |

Run identifiers have been replaced with `run_01`, `run_02`, … The mapping back
to the original identifiers was not included and stays on the machine that
produced this pack.

## 5. What to produce

Answer these, in this order:

1. **Which conditions differ, on what, and how confidently?** Name the metrics
   and give the effect size with its interval. State plainly which comparisons
   did not separate, without claiming they are equivalent.

2. **Would these differences reproduce?** Use the variance decomposition and the
   per-condition run spread. Call out any result carried by a single run, and
   name that run.

3. **What relates to the outcome metric, and does it hold inside a run?** Compare the
   pooled and within-run views explicitly. Flag any relationship that exists only
   between runs.

4. **What does the multivariable model add?** Which predictors survive being
   considered together, how consistent are they across runs, and does the model
   transfer to a held-out run?

5. **What should be investigated next?** Ground each suggestion in a specific
   number from the pack.

Write for someone who knows the experiment and did not run the analysis. Be
concrete: cite the metric, the conditions, the number and its interval. Where the
evidence is weak, say so and say why. A short honest answer is worth more than a
long confident one.

If something in the pack is missing, null, or internally inconsistent, say so
rather than working around it.
"""


# The three rules that argue from replicate counts. They live here rather than in
# the template because their length depends on the run -- a design of unequal
# conditions quotes a floor for every pair of sizes -- and a paragraph wrapped
# for one design goes ragged for the next. Written as one long line each and
# wrapped at the point of use, which is the only way the width can follow the
# text.
RULE_EQUIVALENCE = (
    "**Never report a non-significant comparison as showing equivalence.** At "
    "these replicate counts the test has a floor: {p_floor_clause} — *even when "
    "every single pairing went the same way*. A comparison can therefore be as "
    "one-sided as the data allows and still not reach a conventional threshold. "
    '"Did not separate" is what you may say. "Are the same", "are '
    'indistinguishable", "show no difference" are not — they claim evidence of '
    "absence that this design cannot produce."
)

RULE_EFFECT_SIZE = (
    "**Always read the effect size beside the p-value.** At these sample sizes "
    "they come apart in both directions. CLES is the probability that a randomly "
    "chosen run of one condition scores above a randomly chosen run of the "
    "other; at {replicate_phrase} it is simply a count of pairings."
)

RULE_INTERVALS = (
    "**Wide confidence intervals are the honest answer.** Every interval is a "
    "percentile bootstrap that resamples runs, not observations. At "
    "{replicate_phrase} they are wide. Do not present a wide interval as a "
    "precise estimate, and do not apologise for it either."
)

# The width the rest of the briefing is wrapped to, and the indent a numbered
# rule's continuation lines carry in it.
RULE_WIDTH = 79
RULE_INDENT = "   "


def _rule(number: int, body: str, **values) -> str:
    """One numbered rule, filled in and wrapped to match the rules beside it."""

    return textwrap.fill(
        f"{number}. {body.format(**values)}",
        width=RULE_WIDTH,
        subsequent_indent=RULE_INDENT,
        # Hyphenated terms are single words here: "two-sided" split across a
        # line break reads as a typo in a document about being precise.
        break_on_hyphens=False,
        break_long_words=False,
    )


# The name each role goes by in the briefing. What the roles *mean* is stated
# once in the template above, as prose: a table that repeated the same paragraph
# beside every metric would be longer and harder to read than one that names the
# role and trusts the reader to have read the definition.
ROLE_LABELS = {
    "outcome": "outcome",
    "component": "component of the outcome",
    "predictor": "predictor",
}


def _metric_rows(metrics: dict) -> str:
    """One table row per metric this run actually carried.

    Generated rather than written out, because the metric set is discovered from
    the extract files: a fixed list here would describe some other run, and a
    briefing that names metrics the data does not contain is worse than one that
    names none.
    """

    rows = [
        f"| `{name}` | {ROLE_LABELS[role]} |"
        for role in ("outcome", "component", "predictor")
        for name in metrics.get(f"{role}s", [])
    ]

    if not rows:
        return (
            "| *(none)* | This run carried no metrics beyond the response and its "
            "signal-to-noise ratio. |"
        )

    return "\n".join(rows)


def build_prompt(findings: RunFindings) -> str:
    """The briefing, with this run's design, metric set and replicate counts filled in."""

    design = findings.design
    runs_per_condition = design.get("runs_per_condition", {})
    per_condition = ", ".join(
        f"{name} ({count} runs)" for name, count in runs_per_condition.items()
    )

    observations = design.get("n_observations")
    summary = (
        f"This run covers **{design.get('n_conditions')} conditions** over "
        f"**{design.get('n_runs')} runs** — {per_condition} — comprising "
        + (
            f"{observations:,} observations in total."
            if observations
            else "the observations of those runs."
        )
    )

    phrase = replicate_phrase(runs_per_condition)

    return PROMPT_TEMPLATE.format(
        response_name=findings.response_name,
        design_summary=summary,
        metric_rows=_metric_rows(design.get("metrics", {})),
        rule_equivalence=_rule(
            1, RULE_EQUIVALENCE, p_floor_clause=p_floor_clause(runs_per_condition)
        ),
        rule_effect_size=_rule(2, RULE_EFFECT_SIZE, replicate_phrase=phrase),
        rule_intervals=_rule(9, RULE_INTERVALS, replicate_phrase=phrase),
    )


def write_prompt(findings: RunFindings, ai_dir: Path) -> Path:
    """Drop the briefing beside the pack it describes."""

    path = ai_dir / PROMPT_FNAME
    path.write_text(build_prompt(findings), encoding="utf-8")

    logger.info(f"- AI prompt: {PROMPT_FNAME}")

    return path
