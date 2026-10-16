"""One self-contained HTML file describing a completed run.

Everything is inlined -- styles, and every figure as a base64 data URI -- so the
file can be attached to an email or dropped in a shared folder and still render
exactly as it did on the machine that made it. A report that only works beside
its own directory is a report that arrives broken.

The structure follows the questions rather than the file system, split across
tabs so a reader lands on a summary rather than on thirty-odd figures. Each tab
leads with the findings it is about and then shows the figures that support
them, because a figure in a gallery at the end of a document is a figure nobody
connects to anything.

The tabs are **CSS only** -- hidden radio inputs and labels, no script. That
keeps the page inert, which is what lets it render in locked-down viewers and
anywhere a script would be stripped. The radios are clipped rather than
``display: none`` so they stay keyboard-reachable.

Tabs cost two things, and one of them is bought back. Find-in-page only searches
the open tab, which is inherent to hiding content. Printing would have the same
problem, so a print stylesheet reveals every panel: ``Ctrl+P`` or save-as-PDF
gives the whole report as one continuous document whatever tab is showing.

No filesystem path is written into the page. The output directory's *name*
appears, because it identifies the run to whoever produced it, but the machine
it sat on does not travel with the file.
"""

from __future__ import annotations

import base64
import html
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..constants import outputs as out
from ..llm.questions import RESEARCH_QUESTIONS
from . import findings as fnd
from .ai_report import render_ai_interpretation, require_complete_ai_results

if TYPE_CHECKING:
    from ..llm.orchestrator import QuestionRunResult
from .findings import RunFindings
from .replicates import p_floor_span_clause, replicate_phrase

logger = logging.getLogger(__name__)

REPORT_FNAME = 'report.html'

# Per-observation exports run to six figures of rows and are inputs to the analysis
# rather than results of it. They stay in the CSVs; rendering them would add
# tens of megabytes and tell a reader nothing.
RAW_EXPORTS = frozenset({
    out.RESPONSES_CSV_FNAME,
    out.SNR_CSV_FNAME,
    out.PREDICTOR_METRICS_CSV_FNAME,
    out.OUTCOME_METRICS_CSV_FNAME,
})

# Long enough that no analysis table is truncated in practice, short enough
# that a surprise cannot bloat the page.
MAX_TABLE_ROWS = 200

# The tabs, in reading order. Each carries the findings it is about (by their
# ``kind`` slug) and the figure directories that support them, so a figure sits
# beside the sentence it is evidence for rather than in a gallery.
#
# Directories absent from a given run are skipped rather than left as empty
# headings, and a tab that ends up with nothing at all is not rendered.

@dataclass(frozen=True)
class Tab:
    slug: str
    label: str
    blurb: str
    finding_kinds: tuple[str, ...] = ()
    figure_dirs: tuple[tuple[str, str], ...] = ()
    body: str = ''          # 'design', 'tables' or 'reading' for the special panels


TABS = (
    Tab(
        slug='overview',
        label='Overview',
        blurb='What the run measured, and everything it found.',
        finding_kinds=(
            fnd.KIND_EFFECT_SIZE, fnd.KIND_PAIRWISE, fnd.KIND_REPRODUCIBILITY,
            fnd.KIND_RUN_SPREAD, fnd.KIND_CORRELATIONS, fnd.KIND_MODEL,
        ),
        figure_dirs=((out.OBSERVATION_COUNT_SUBDIR, 'How much data each condition contributed'),),
        body='design',
    ),
    Tab(
        slug='conditions',
        label='Conditions',
        blurb='Whether the conditions differ, on what, and by how much.',
        finding_kinds=(fnd.KIND_EFFECT_SIZE, fnd.KIND_PAIRWISE),
        figure_dirs=(
            (out.RESPONSES_SUBDIR, 'The raw measured signal, by level'),
            (out.EFFECT_SIZE_SUBDIR, 'How much of the variation goes with the condition'),
        ),
    ),
    Tab(
        slug='reproducibility',
        label='Reproducibility',
        blurb='Whether those differences would happen again in another run.',
        finding_kinds=(fnd.KIND_REPRODUCIBILITY, fnd.KIND_RUN_SPREAD),
        figure_dirs=(
            (out.VARIANCE_SUBDIR, "Where each metric's variation comes from"),
            (out.SNR_SUBDIR, 'Signal-to-noise, one dot per run over the observations'),
            (out.PREDICTOR_METRICS_SUBDIR,
             'The predictor metrics, one dot per run over the observations'),
            (out.OUTCOME_METRICS_SUBDIR,
             'The outcome metric and its components, one dot per run over the observations'),
        ),
    ),
    Tab(
        slug='relationships',
        label='Relationships',
        blurb='What goes with what, and whether it holds inside a single run.',
        finding_kinds=(fnd.KIND_CORRELATIONS,),
        figure_dirs=(
            (out.CORRELATIONS_SUBDIR, 'How the three correlation views compare'),
            (out.RELATIONSHIPS_SUBDIR, "Each metric against the outcome's bins"),
        ),
    ),
    Tab(
        slug='model',
        label='Multivariable model',
        blurb='What the model adds when predictors are considered together.',
        finding_kinds=(fnd.KIND_MODEL,),
        figure_dirs=((out.MODEL_SUBDIR, 'Model coefficients and held-out performance'),),
    ),
    Tab(
        slug='tables',
        label='Tables',
        blurb='Every analysis table the run wrote, as it was written.',
        body='tables',
    ),
    Tab(
        slug='reading',
        label='How to read this',
        blurb='The rules that decide whether an interpretation is honest.',
        body='reading',
    ),
)

STYLES = """
:root {
  --bg: #ffffff; --fg: #1c1c1e; --muted: #5b5b60; --rule: #e2e2e6;
  --panel: #f7f7f9; --accent: #2c6e8e; --caution-bg: #fdf6ec;
  --caution-edge: #d98032;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --fg: #e8e8ea; --muted: #a0a0a8; --rule: #2e2f34;
    --panel: #1e1f23; --accent: #6fb3d2; --caution-bg: #2a2118;
    --caution-edge: #d98032;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem 6rem; background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .4rem; }
h2 {
  font-size: 1.3rem; margin: 3.5rem 0 1rem; padding-bottom: .4rem;
  border-bottom: 2px solid var(--rule);
}
h3 { font-size: 1.02rem; margin: 2rem 0 .5rem; color: var(--muted); font-weight: 600; }
p { margin: 0 0 1rem; }
.subtitle { color: var(--muted); margin-bottom: 2rem; }
.design {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
  gap: .75rem; margin: 1.5rem 0;
}
.stat { background: var(--panel); border-radius: 8px; padding: .85rem 1rem; }
.stat .value { font-size: 1.5rem; font-weight: 650; color: var(--accent); }
.stat .label { font-size: .78rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; }
.finding { margin: 0 0 2.5rem; }
.finding .answer { margin-bottom: .85rem; }
.caution {
  background: var(--caution-bg); border-left: 3px solid var(--caution-edge);
  padding: .7rem .95rem; border-radius: 0 6px 6px 0; font-size: .92rem;
  color: var(--fg);
}
.caution strong { color: var(--caution-edge); }
figure { margin: 1.5rem 0; }
figure img {
  max-width: 100%; height: auto; display: block; border: 1px solid var(--rule);
  border-radius: 6px; background: #fff;
}
figcaption { font-size: .82rem; color: var(--muted); margin-top: .4rem; }
.table-wrap { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; font-size: .82rem; min-width: 100%; }
th, td { padding: .35rem .6rem; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--rule); }
th { text-align: right; font-weight: 650; color: var(--muted);
  border-bottom: 2px solid var(--rule); position: sticky; top: 0;
  background: var(--bg); }
td:first-child, th:first-child { text-align: left; }
.rows { font-size: .8rem; color: var(--muted); }
ul { margin: 0 0 1rem; padding-left: 1.2rem; }
li { margin-bottom: .5rem; }
/* A citation sits under the rule it supports, quieter than the rule itself.
   Plain text, not a link: the page must survive being read with no network. */
.cite { display: block; margin-top: .35rem; font-size: .82rem; color: var(--muted); }
/* Separate AI interpretation from deterministic findings, without implying
   that numerical validation proves the narrative scientifically correct. */
.ai-interpretation {
  margin: 2rem 0; padding: 1rem 1.25rem; border: 1px solid var(--rule);
  border-radius: 8px; background: var(--panel);
}
.ai-interpretation h3 { margin-top: 0; color: var(--fg); }
.ai-interpretation h4 { margin: 1rem 0 .4rem; }
.ai-interpretation summary { cursor: pointer; color: var(--accent); }
.ai-interpretation code { overflow-wrap: anywhere; }
footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--rule);
  font-size: .82rem; color: var(--muted); }

/* --- Tabs, without a line of script ------------------------------------
   The radios are clipped rather than display:none, which keeps them
   keyboard-reachable: a screen-reader user can still arrow between tabs. */
.tab-state {
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap;
}
.tab-bar {
  position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap;
  gap: .25rem; margin: 1.5rem 0 0; padding: .5rem 0;
  background: var(--bg); border-bottom: 2px solid var(--rule);
}
.tab-bar label {
  padding: .45rem .85rem; border-radius: 6px 6px 0 0; cursor: pointer;
  font-size: .9rem; font-weight: 600; color: var(--muted);
  border: 1px solid transparent; border-bottom: none; user-select: none;
}
.tab-bar label:hover { background: var(--panel); color: var(--fg); }
.panels > section { display: none; }
.panels > section > h2:first-child { margin-top: 2rem; }
.tab-blurb { color: var(--muted); margin: -.6rem 0 2rem; }
/* Focus ring on the label when its hidden radio has keyboard focus. */
.tab-state:focus-visible + .tab-bar label { outline: none; }
"""


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _cell(value) -> str:
    """A table value, floats trimmed to something a person can scan."""
    if value is None:
        return "&mdash;"
    if isinstance(value, float):
        if value != value:                        # NaN
            return "&mdash;"
        if value == int(value) and abs(value) < 1e15:
            return _esc(int(value))
        return _esc(f"{value:.4g}")
    return _esc(value)


def _embed(path: Path) -> str:
    """One figure as a data URI, so the page carries its own pictures."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_block(path: Path) -> str:
    caption = path.stem.replace("_", " ")
    return (
        f'<figure><img alt="{_esc(caption)}" src="{_embed(path)}">'
        f'<figcaption>{_esc(caption)}</figcaption></figure>'
    )


def _table_block(df: pl.DataFrame, title: str) -> str:
    """One analysis table, capped and labelled if the cap bit."""

    shown = df.head(MAX_TABLE_ROWS)

    header = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in row) + "</tr>"
        for row in shown.iter_rows()
    )

    note = (
        f'<p class="rows">first {MAX_TABLE_ROWS} of {df.height} rows; '
        f'the CSV holds all of them</p>'
        if df.height > MAX_TABLE_ROWS
        else f'<p class="rows">{df.height} rows</p>'
    )

    return (
        f"<h3>{_esc(title)}</h3>{note}"
        f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _design_block(design: dict) -> str:
    stats = [
        (design.get("n_conditions"), "conditions"),
        (design.get("n_runs"), "runs"),
        (f'{design.get("n_observations"):,}' if design.get("n_observations") else None,
         "observations"),
        (design.get("response_name"), "response"),
        (design.get("outcome_metric"), "outcome metric"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="value">{_esc(value)}</div>'
        f'<div class="label">{_esc(label)}</div></div>'
        for value, label in stats if value is not None
    )

    per_condition = ", ".join(
        f"{_esc(name)} {count}" for name, count in design.get("runs_per_condition", {}).items()
    )

    return (
        f'<div class="design">{cards}</div>'
        f"<p>Runs per condition: {per_condition}.</p>"
        f'<div class="caution"><strong>Unit of analysis.</strong> '
        f'{_esc(design.get("unit_of_analysis", ""))}</div>'
    )


def _findings_block(findings: RunFindings, kinds: tuple[str, ...]) -> str:
    """The findings this tab is about, in the order the pipeline produced them.

    A finding appears on the Overview tab *and* on its own topic tab, which is
    deliberate duplication of a short paragraph: the overview is the summary a
    reader arrives at, and the topic tab needs it again because a figure without
    the sentence it supports is the gallery problem this layout exists to avoid.
    """

    blocks = []
    for finding in findings.findings:
        if finding.kind not in kinds:
            continue
        caution = (
            f'<div class="caution"><strong>How to read it.</strong> '
            f"{_esc(finding.caution)}</div>"
            if finding.caution
            else ""
        )
        blocks.append(
            f'<div class="finding"><h3>{_esc(finding.question)}</h3>'
            f'<p class="answer">{_esc(finding.answer)}</p>{caution}</div>'
        )
    return "".join(blocks)


def _figures_block(output_dir: Path, figure_dirs: tuple[tuple[str, str], ...]) -> str:
    sections = []
    for subdir, heading in figure_dirs:
        paths = sorted((output_dir / subdir).glob("*.png"))
        if not paths:
            continue
        sections.append(
            f"<h3>{_esc(heading)}</h3>"
            + "".join(_figure_block(p) for p in paths)
        )
    return "".join(sections)


def _tables_block(output_dir: Path) -> str:
    csv_root = output_dir / out.CSV_EXPORTS_SUBDIR
    if not csv_root.is_dir():
        return ""

    blocks = []
    for path in sorted(csv_root.rglob("*.csv")):
        if path.name in RAW_EXPORTS:
            continue
        try:
            frame = pl.read_csv(path)
        except Exception:                          # noqa: BLE001 - a bad CSV must not lose the report
            logger.warning(f"Report: could not read {path.name}, table skipped.")
            continue
        title = str(path.relative_to(csv_root)).replace("\\", "/")
        blocks.append(_table_block(frame, title))

    return "".join(blocks)


HOW_TO_READ_TEMPLATE = """
<ul>
<li><strong>The run is the replicate, not the observation.</strong> All the observations in
one run were measured together, on the same equipment, under the same setup, on
the same day. A statistic computed over tens of thousands of pooled observations is
answering a question about a handful of runs while claiming the precision of
tens of thousands of observations.</li>

<li><strong>A large p-value is not evidence of equivalence.</strong> With
{replicate_phrase} {p_floor_span_clause}, even when every pairing went the same
way. A comparison that did not separate has not been shown to be the same.</li>

<li><strong>Effect size and p-value go together.</strong> At these sample sizes
they come apart in both directions: complete separation with a large p, and a
small p on a difference too small to act on.</li>

<li><strong>The p-values in the pairwise table are adjusted.</strong> Splitting one
question into every pair of conditions runs several tests at once, so they are
corrected by Benjamini&ndash;Hochberg, which controls the expected proportion of
significant findings that are false. The family is one metric, so an answer about
one column does not depend on how many other columns were extracted; both the
method and the family are named in every row of the table.
<span class="cite">Benjamini Y, Hochberg Y (1995). Controlling the false discovery
rate: a practical and powerful approach to multiple testing. <em>Journal of the
Royal Statistical Society: Series B (Methodological)</em> 57(1): 289&ndash;300.
doi:10.1111/j.2517-6161.1995.tb02031.x</span></li>

<li><strong>Pooled correlations describe, they do not test.</strong> They are
true statements about the observation population measured. No p-values accompany
them, deliberately, because the observations are not independent replicates. The
within-run table's count of how many runs agreed on a sign is the
reproducibility claim the design supports.</li>

<li><strong>Between-run correlations are confounded with the condition.</strong>
The runs span every condition, so a coefficient there partly re-measures the
condition effect.</li>

<li><strong>Every condition result is an association.</strong> The design
supports "this metric differs between these conditions". It does not support
"the condition improved the metric by way of this mechanism".</li>

<li><strong>Model coefficients are within-run associations.</strong> They do not
show that changing a predictor on an observation would change its outcome. A negative
out-of-sample R-squared means the coefficients predicted that run worse than its
own mean would have.</li>

<li><strong>Wide intervals are the honest answer.</strong> Every interval is a
percentile bootstrap that resamples runs. At {replicate_phrase} they are wide,
and that width is information rather than a defect to be narrowed.</li>
</ul>
"""


def how_to_read(design: dict) -> str:
    """The reading rules, with this run's replicate counts filled in.

    Two of the rules are only worth reading next to a number -- what the p-value
    could not have gone below, what the intervals are wide compared to -- and
    both of those numbers belong to the run the report is about. Left as fixed
    prose they described whichever design was in front of the author, which is
    how the same claim came to sit beside a table that contradicted it.
    """

    runs_per_condition = (design or {}).get("runs_per_condition", {})

    return HOW_TO_READ_TEMPLATE.format(
        replicate_phrase=replicate_phrase(runs_per_condition),
        p_floor_span_clause=p_floor_span_clause(runs_per_condition),
    )


AI_TAB_QUESTION_IDS = {
    'conditions': 'condition_comparisons',
    'reproducibility': 'reproducibility',
    'relationships': 'relationships',
    'model': 'multivariable_model',
}


def _panel_body(
    tab: Tab,
    findings: RunFindings,
    output_dir: Path,
    ai_by_id: Mapping[str, QuestionRunResult] | None = None,
) -> str:
    """Everything inside one tab, in the order a reader wants it.

    Findings first, then the figures that support them. The three special
    panels -- the design summary, the tables and the reading rules -- have no
    figures and say so by carrying a ``body`` instead.
    """

    parts = []

    if tab.body == 'design':
        parts.append(_design_block(findings.design))

    parts.append(_findings_block(findings, tab.finding_kinds))
    parts.append(_figures_block(output_dir, tab.figure_dirs))

    question_id = AI_TAB_QUESTION_IDS.get(tab.slug)
    if ai_by_id is not None and question_id is not None:
        parts.append(render_ai_interpretation(ai_by_id[question_id]))

    if tab.body == 'tables':
        parts.append(_tables_block(output_dir))
    elif tab.body == 'reading':
        parts.append(how_to_read(findings.design))

    return "".join(parts)


def _tab_css(slugs: list[str]) -> str:
    """The :checked rules, one pair per tab, plus the print override.

    Generated rather than written out because the selectors have to name every
    slug: a tab added to ``TABS`` with no matching rule here would render as a
    label that does nothing, which is worse than a tab that is simply absent.
    """

    rules = [
        f'#tab-{slug}:checked ~ .panels > #panel-{slug} {{ display: block; }}\n'
        f'#tab-{slug}:checked ~ .tab-bar label[for="tab-{slug}"] {{ '
        f'background: var(--panel); color: var(--accent); '
        f'border-color: var(--rule); }}'
        for slug in slugs
    ]

    # Printing has the same problem tabs create -- only the open panel would
    # come out -- so the printed copy is the whole report, in order, whatever
    # is showing on screen.
    rules.append(
        '@media print {\n'
        '  .tab-bar { display: none; }\n'
        '  .panels > section { display: block !important; page-break-before: always; }\n'
        '  .panels > section:first-child { page-break-before: auto; }\n'
        '  figure img { border: none; }\n'
        '}'
    )

    return "\n".join(rules)


def build_report_html(
    findings: RunFindings,
    output_dir: Path,
    *,
    ai_results: Iterable[QuestionRunResult] | None = None,
) -> str:
    """Assemble the whole page as one string.

    Tabs with nothing in them are dropped entirely: a run that produced no
    model, say, should not offer an empty Relationships tab. The first tab that
    does have content is the one checked on load.
    """

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Existing offline/local report generation stays backward-compatible.
    # The GitHub Pages publishing command must supply all four AI results;
    # providing an incomplete set is an error, never a partial AI report.
    ai_by_id = (
        require_complete_ai_results(
            ai_results, (question.identifier for question in RESEARCH_QUESTIONS)
        )
        if ai_results is not None
        else None
    )

    panels = [
        (tab, _panel_body(tab, findings, output_dir, ai_by_id)) for tab in TABS
    ]
    panels = [(tab, body) for tab, body in panels if body.strip()]

    if not panels:
        logger.warning("Report: no tab had any content.")
        panels = [(TABS[0], _design_block(findings.design))]

    slugs = [tab.slug for tab, _ in panels]

    radios = "".join(
        f'<input class="tab-state" type="radio" name="report-tab" '
        f'id="tab-{tab.slug}"{" checked" if index == 0 else ""}>'
        for index, (tab, _) in enumerate(panels)
    )
    bar = "".join(
        f'<label for="tab-{tab.slug}">{_esc(tab.label)}</label>' for tab, _ in panels
    )
    sections = "".join(
        f'<section id="panel-{tab.slug}"><h2>{_esc(tab.label)}</h2>'
        f'<p class="tab-blurb">{_esc(tab.blurb)}</p>{body}</section>'
        for tab, body in panels
    )

    return (
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Data Analytics report &mdash; {_esc(output_dir.name)}</title>"
        f"<style>{STYLES}\n{_tab_css(slugs)}</style></head><body><main>"
        f"<h1>Data Analytics report</h1>"
        f'<p class="subtitle">{_esc(output_dir.name)} &middot; response '
        f"<code>{_esc(findings.response_name)}</code> &middot; generated {generated}</p>"
        f'<div class="tabs">{radios}'
        f'<nav class="tab-bar">{bar}</nav>'
        f'<div class="panels">{sections}</div></div>'
        f"<footer>Generated by the Data Analytics pipeline. Every figure is "
        f"embedded in this file; it needs nothing beside it to render. "
        f"Printing shows every tab as one continuous document.</footer>"
        f"</main></body></html>"
    )


def write_report(
    findings: RunFindings,
    output_dir: Path,
    *,
    ai_results: Iterable[QuestionRunResult] | None = None,
) -> Path:
    """Render the report and drop it at the root of the run's output."""

    path = output_dir / REPORT_FNAME
    path.write_text(
        build_report_html(findings, output_dir, ai_results=ai_results),
        encoding="utf-8",
    )

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"- Report: {REPORT_FNAME} ({size_mb:.1f} MB, self-contained)")

    return path
