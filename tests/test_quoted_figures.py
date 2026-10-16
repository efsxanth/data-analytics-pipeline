"""Every number quoted in prose must be reproducible from this repository.

The briefing in ``reporting/prompt.py`` and the reasoning in
``analysis/correlations.py`` both argue from worked examples, and a worked
example is only worth anything if a reader can check it. These tests recompute
each quoted figure from the demo dataset the repository ships and assert that the
prose still says what the data says.

That makes a stale number a test failure rather than something a reader has to
notice. It also keeps the repository honest about provenance: a figure that
cannot be regenerated here does not belong in the text.

**Replicate counts are quoted figures too**, and for a long time these tests did
not treat them as such. A statistic was pinned to the demo data while, a few
lines away, two of the briefing's rules went on naming a larger and evenly sized
design than the run they shipped with -- one of them counting pairings for a
comparison the run never made, the other calling the intervals wide from a
replicate count no condition had. Nothing checked them, because a spelled-out
count does not look like a number.

So the last two classes here check counts rather than statistics. One scans the
text the pipeline generates and requires every replicate count it states to be
one the run it describes actually has; the other scans the repository's own
prose -- source, tests and README -- and requires the same of the demo design,
which is the only dataset a reader can run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from da_pipeline.analysis import correlations as correlations_module
from da_pipeline.analysis.correlations import DataCorrelator
from da_pipeline.analysis.pairwise import smallest_attainable_two_sided_p
from da_pipeline.config import PipelineConfig, StepRange
from da_pipeline.constants import columns as col
from da_pipeline.extraction import DataExtractor
from da_pipeline.metrics import resolve_metric_roles
from da_pipeline.reporting.findings import (
    RunFindings,
    effect_size_finding,
    pairwise_finding,
)
from da_pipeline.reporting.html_report import how_to_read
from da_pipeline.reporting.prompt import PROMPT_TEMPLATE, build_prompt
from da_pipeline.synthetic import DEFAULT_CONDITIONS, generate_dataset
from da_pipeline.validation import DataValidator

OUTCOME = "Accuracy"

# The pair the briefing argues from, and the metrics the correlation docstring
# does. Named here so a change to either piece of prose fails loudly.
EXAMPLE_A = "Condition_A"
EXAMPLE_B = "Condition_B"
EXAMPLE_VAR_X = "signal_to_noise_ratio"
EXAMPLE_VAR_Y = "degradation_2"


@pytest.fixture(scope="module")
def demo_config_path(tmp_path_factory):
    """Exactly the dataset ``da-pipeline-demo`` writes with no arguments."""
    root = tmp_path_factory.mktemp("quoted")
    return generate_dataset(root, conditions=DEFAULT_CONDITIONS, seed=0)


@pytest.fixture(scope="module")
def outcome_by_condition(demo_config_path):
    """Per-observation outcome values and per-run medians, by condition."""
    from da_pipeline.io.extract_reader import open_extract

    config = json.loads(demo_config_path.read_text(encoding="utf-8"))
    pooled, run_medians = {}, {}

    for condition, paths in config.items():
        values = []
        for path in paths:
            with open_extract(path) as reader:
                values.append(reader.metrics([OUTCOME])[OUTCOME])
        pooled[condition] = np.concatenate(values)
        run_medians[condition] = [float(np.median(v)) for v in values]

    return pooled, run_medians


@pytest.fixture(scope="module")
def demo_run_counts(demo_config_path) -> dict:
    """How many runs each demo condition has, read off the config it ships with."""
    config = json.loads(demo_config_path.read_text(encoding="utf-8"))
    return {condition: len(paths) for condition, paths in config.items()}


@pytest.fixture(scope="module")
def demo_findings(demo_run_counts) -> RunFindings:
    """A findings object carrying the demo design, which is all the prose needs.

    The generated text depends on the design and nothing else, so this skips the
    analysis rather than running the whole pipeline to reach it.
    """
    return RunFindings(
        response_name="auc",
        design={
            "response_name": "auc",
            "outcome_metric": OUTCOME,
            "metrics": {"outcome": [OUTCOME], "predictors": [EXAMPLE_VAR_X]},
            "n_conditions": len(demo_run_counts),
            "n_runs": sum(demo_run_counts.values()),
            "runs_per_condition": demo_run_counts,
            "n_observations": None,
        },
    )


@pytest.fixture(scope="module")
def demo_briefing(demo_findings) -> str:
    """``prompt.md`` as the pipeline would write it for the demo dataset."""
    return build_prompt(demo_findings)


@pytest.fixture(scope="module")
def demo_pairwise(demo_run_counts) -> pl.DataFrame:
    """One comparison per condition pair, carrying the group sizes it used."""
    names = sorted(demo_run_counts)
    pairs = [(a, b) for index, a in enumerate(names) for b in names[index + 1:]]

    return pl.DataFrame({
        col.METRIC: [OUTCOME] * len(pairs),
        col.CONDITION_A: [a for a, _ in pairs],
        col.CONDITION_B: [b for _, b in pairs],
        col.N_RUNS_A: [demo_run_counts[a] for a, _ in pairs],
        col.N_RUNS_B: [demo_run_counts[b] for _, b in pairs],
        col.CLES: [0.6] * len(pairs),
        col.HIGHER_CONDITION: [a for a, _ in pairs],
        col.P_VALUE_ADJUSTED: [0.42] * len(pairs),
    })


@pytest.fixture(scope="module")
def demo_effect_size() -> pl.DataFrame:
    """Enough of an effect-size frame for the finding that cites replicate counts."""
    return pl.DataFrame({
        col.LEVEL: [0, 1, 2],
        col.ETA_SQUARED: [0.2, 0.3, 0.4],
        col.ETA_SQUARED_CI_LOW: [0.0, 0.1, 0.1],
        col.ETA_SQUARED_CI_HIGH: [0.7, 0.8, 0.9],
    })


# ---------------------------------------------------------------------------
# Replicate counts stated in prose
# ---------------------------------------------------------------------------

# Spelled-out counts as far as the prose spells them, which is as far as
# ``reporting/replicates.py`` will generate them.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_NUMBER = r"(?:\d+|" + "|".join(NUMBER_WORDS) + r")"

# The four shapes a replicate count is stated in anywhere in this repository.
# Each requires a number on both sides of the construction, which is what keeps
# them off the many sentences that use "against" or "each" for something else:
# "scored against ground truth" carries no count and matches nothing.
#
# A pattern rather than a list of forbidden phrases, deliberately. A list only
# catches the wordings somebody already noticed were wrong, and the two that
# prompted this test had both survived several passes precisely because nobody
# thought to look for them.
CLAIM_PATTERNS = (
    ("count", re.compile(rf"\b({_NUMBER}) runs? per condition\b", re.I)),
    ("count", re.compile(rf"\b({_NUMBER}) runs? each\b", re.I)),
    ("range", re.compile(rf"\b({_NUMBER}) to ({_NUMBER})(?= runs?\b| per condition\b)", re.I)),
    ("pair", re.compile(rf"\b({_NUMBER}) (?:runs? )?against ({_NUMBER})\b", re.I)),
)


def _as_int(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]


def replicate_claims(text: str) -> list[tuple[str, tuple[int, ...], str]]:
    """Every replicate count the text states, as (kind, values, wording)."""

    return [
        (kind, tuple(_as_int(group) for group in match.groups()), match.group(0))
        for kind, pattern in CLAIM_PATTERNS
        for match in pattern.finditer(text)
    ]


def assert_claims_match(text: str, counts: set[int], where: str) -> int:
    """Fail on any replicate count the design does not have. Returns the number checked.

    A single count has to be one some condition really has. A range has to span
    the design exactly, from its smallest condition to its largest: a range that
    merely contained the true counts would pass while describing a study several
    times the size. A pairing has to name two counts that both exist, since that
    is what a comparison in the pairwise table is made of.
    """

    claims = replicate_claims(text)

    for kind, values, wording in claims:
        if kind == "range":
            assert values == (min(counts), max(counts)), (
                f"{where}: {wording!r} states a replicate range this design does "
                f"not have; its conditions run from {min(counts)} to {max(counts)}."
            )
        else:
            unknown = [value for value in values if value not in counts]
            assert not unknown, (
                f"{where}: {wording!r} states a replicate count no condition has; "
                f"this design has {sorted(counts)}."
            )

    return len(claims)


class TestTheBriefingsWorkedExample:
    """Section 2 of prompt.md argues from a real comparison; check every figure."""

    @pytest.fixture(scope="module")
    def example(self, outcome_by_condition):
        pooled, run_medians = outcome_by_condition
        a, b = pooled[EXAMPLE_A], pooled[EXAMPLE_B]
        ma, mb = run_medians[EXAMPLE_A], run_medians[EXAMPLE_B]

        return {
            "median_difference": abs(float(np.median(a)) - float(np.median(b))),
            "n_observations": a.size + b.size,
            "p_pooled": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
            "p_runs": stats.mannwhitneyu(ma, mb, alternative="two-sided").pvalue,
            "n_runs_a": len(ma),
            "n_runs_b": len(mb),
        }

    def test_the_quoted_median_difference_is_the_real_one(self, example):
        assert f"{example['median_difference']:.3f}" == "0.013"
        assert "differ in median `Accuracy` by 0.013" in PROMPT_TEMPLATE

    def test_the_quoted_observation_count_is_the_real_one(self, example):
        assert f"{example['n_observations']:,}" == "2,880"
        assert "2,880\nobservations" in PROMPT_TEMPLATE
        assert "treating 2,880 correlated measurements as 2,880 independent ones" in (
            PROMPT_TEMPLATE
        )

    def test_the_quoted_pooled_p_value_is_the_real_one(self, example):
        assert f"{example['p_pooled']:.1e}" == "1.9e-08"
        assert "`p = 1.9e-08`" in PROMPT_TEMPLATE

    def test_the_quoted_run_level_p_value_is_the_real_one(self, example):
        assert f"{example['p_runs']:.1f}" == "0.2"
        assert "`p = 0.2`" in PROMPT_TEMPLATE

    def test_the_quoted_run_counts_are_the_real_ones(self, example):
        assert (example["n_runs_a"], example["n_runs_b"]) == (3, 3)
        assert "three runs each" in PROMPT_TEMPLATE

    def test_the_example_actually_makes_its_point(self, example):
        """The pooled comparison must look overwhelming and the honest one must not.

        If the generator ever changed so that both agreed, the passage would be
        quoting real numbers that no longer illustrate anything.
        """
        assert example["p_pooled"] < 1e-6
        assert example["p_runs"] > 0.05


class TestThePValueFloorClaim:
    """Rule 1 states the smallest two-sided p each comparison in the run could return.

    The floors themselves are a property of the test rather than of the data, so
    they are checked against scipy. What has to come from the run is *which*
    floors get quoted: a design whose conditions are not all the same size
    produces a different floor for each pair of sizes, and a briefing that
    mentions only the best of them has told the reader that every comparison in
    the table is sharper than it is.
    """

    @staticmethod
    def smallest_two_sided_p(n_a: int, n_b: int) -> float:
        """Complete separation between groups of these sizes: the best case there is."""
        low, high = list(range(n_b)), list(range(n_b, n_b + n_a))
        return stats.mannwhitneyu(high, low, alternative="two-sided").pvalue

    def test_the_helper_computes_complete_separation(self):
        """The pipeline's own floor agrees with one worked out here from scratch."""
        for n_a, n_b in ((3, 3), (3, 2), (2, 2), (4, 4)):
            assert smallest_attainable_two_sided_p(n_a, n_b) == pytest.approx(
                self.smallest_two_sided_p(n_a, n_b)
            )

    def test_every_pairing_in_the_demo_design_is_quoted_with_its_own_floor(
        self, demo_briefing, demo_run_counts
    ):
        """Each pair of group sizes the run will compare appears with its real floor."""
        sizes = sorted(set(demo_run_counts.values()), reverse=True)
        pairings = [(a, b) for index, a in enumerate(sizes) for b in sizes[index:]]

        assert pairings, "the demo design must produce at least one comparison"

        for n_a, n_b in pairings:
            floor = f"{self.smallest_two_sided_p(n_a, n_b):.3g}"
            assert floor in demo_briefing, (
                f"rule 1 does not quote the floor {floor} for {n_a} runs against {n_b}"
            )

    def test_the_briefing_quotes_no_floor_from_some_other_design(self, demo_briefing):
        """A floor for group sizes this run does not have is the original bug exactly."""
        rule_one = demo_briefing.split("1. **Never report")[1].split("2. **")[0]

        for absent in (4, 5, 6):
            floor = f"{self.smallest_two_sided_p(absent, absent):.3g}"
            assert floor not in rule_one

    def test_the_floor_falls_as_runs_are_added(self):
        floors = [self.smallest_two_sided_p(n, n) for n in (2, 3, 4, 5, 6)]
        assert floors == sorted(floors, reverse=True)


class TestTheCorrelationDocstringExample:
    """``DataCorrelator`` cites a pooled/within-run sign reversal. Check it."""

    @pytest.fixture(scope="module")
    def views(self, demo_config_path):
        config = PipelineConfig(
            config_path=demo_config_path,
            response_name="auc",
            step_range=StepRange(1, 17),
            max_workers=2,
        )

        validator = DataValidator()
        validator.validate_config_dict(config_fpath=demo_config_path)
        validator.validate_group_ids()
        validator.validate_reference_levels()
        validator.validate_metric_names()

        roles = resolve_metric_roles(
            validator.get_declared_metric_names(), config.metrics
        )

        extractor = DataExtractor(
            config=config,
            roles=roles,
            config_dict=validator.config_dict,
            group_ids=validator.group_ids,
            group_label_dict=validator.get_group_label_dict(),
            reference_levels_dict=validator.get_reference_levels_dict(),
        )

        responses = extractor.preload_responses()
        matrix = DataCorrelator.build_per_observation_matrix(
            final_signal_to_noise_df=extractor.process_signal_to_noise(responses),
            final_predictor_metrics_df=extractor.process_predictor_metrics(),
            final_outcome_metrics_df=extractor.process_outcome_metrics(),
        )

        pooled, within_run, _ = DataCorrelator(config=config).build_correlation_tables(matrix)
        return pooled, within_run

    def _row(self, frame: pl.DataFrame) -> dict:
        matched = frame.filter(
            (pl.col(col.CONDITION) == EXAMPLE_A)
            & (pl.col(col.CORR_VAR_X) == EXAMPLE_VAR_X)
            & (pl.col(col.CORR_VAR_Y) == EXAMPLE_VAR_Y)
        )
        assert matched.height == 1, f"expected one row, got {matched.height}"
        return matched.row(0, named=True)

    def test_the_quoted_coefficients_are_the_real_ones(self, views):
        pooled, within_run = views

        assert f"{self._row(pooled)[col.CORR_PEARSON_R]:+.3f}" == "-0.107"
        assert f"{self._row(within_run)[col.CORR_PEARSON_R_MEDIAN]:+.3f}" == "+0.018"

    def test_the_docstring_quotes_those_coefficients(self):
        docstring = correlations_module.DataCorrelator.build_within_run_correlations_df.__doc__

        assert "-0.107 pooled" in docstring
        assert "+0.018 within-run" in docstring
        assert EXAMPLE_VAR_Y in docstring

    def test_the_example_is_a_real_sign_reversal(self, views):
        """The passage exists to show the two views disagreeing in sign."""
        pooled, within_run = views

        assert self._row(pooled)[col.CORR_PEARSON_R] < 0
        assert self._row(within_run)[col.CORR_PEARSON_R_MEDIAN] > 0

    def test_the_quoted_honest_description_is_the_real_one(self, views):
        """The module docstring cites one coefficient as a fair description."""
        pooled, _ = views

        row = pooled.filter(
            (pl.col(col.CONDITION) == EXAMPLE_A)
            & (pl.col(col.CORR_VAR_X) == EXAMPLE_VAR_X)
            & (pl.col(col.CORR_VAR_Y) == OUTCOME)
        ).row(0, named=True)

        assert f"{row[col.CORR_PEARSON_R]:.3f}" == "0.620"
        assert f"{row[col.CORR_N_OBS]:,}" == "1,440"

        docstring = correlations_module.DataCorrelator.__doc__
        assert "r of 0.620" in docstring
        assert "1,440 observations" in docstring


class TestTheBoxplotDocstringExample:
    """The SuperPlot rationale quotes the size of one condition's box."""

    def test_a_demo_condition_really_holds_that_many_observations(
        self, outcome_by_condition
    ):
        pooled, run_medians = outcome_by_condition

        assert f"{pooled[EXAMPLE_A].size:,}" == "1,440"
        assert len(run_medians[EXAMPLE_A]) == 3

    def test_the_docstring_quotes_those_figures(self):
        from da_pipeline.plotting.plots import DataPlotter

        docstring = DataPlotter._boxplot_by_condition.__doc__

        assert "1,440" in docstring
        assert "three runs" in docstring

    def test_the_tick_label_example_matches_a_demo_condition(self):
        from da_pipeline.plotting.plots import DataPlotter

        docstring = DataPlotter._condition_tick_labels.__doc__

        assert "(3 runs, 1,440 obs)" in docstring


class TestReplicateCountsInGeneratedOutput:
    """Every run count the pipeline writes down must belong to the run it describes.

    The briefing, the findings' cautions and the HTML report's reading rules all
    argue from replicate counts, and all three used to state them as fixed prose.
    That is how ``prompt.md`` came to tell a reader about a comparison of five
    runs against five in the same file as a design summary listing conditions of
    three and two.

    These check the generated text against the design it was generated from, so
    the failure arrives at the source rather than at whoever reads the output.
    """

    def test_the_briefing_states_only_this_designs_counts(
        self, demo_briefing, demo_run_counts
    ):
        checked = assert_claims_match(
            demo_briefing, set(demo_run_counts.values()), "prompt.md"
        )
        assert checked, "the briefing states no replicate count at all"

    def test_the_briefing_names_the_range_the_conditions_span(
        self, demo_briefing, demo_run_counts
    ):
        """Rules 2 and 9 both turn on it, so it has to be there and it has to be right."""
        counts = sorted(set(demo_run_counts.values()))
        spelled = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        expected = (
            f"{spelled[counts[0]]} to {spelled[counts[-1]]} runs per condition"
            if len(counts) > 1
            else f"{spelled[counts[0]]} runs per condition"
        )

        # Normalised, because both sentences are hard-wrapped and the
        # generated phrase can land either side of a line break.
        unwrapped = " ".join(demo_briefing.split())

        assert f"at {expected} it is simply a count of pairings" in unwrapped
        assert f"At {expected} they are wide" in unwrapped

    def test_the_reading_rules_state_only_this_designs_counts(
        self, demo_findings, demo_run_counts
    ):
        checked = assert_claims_match(
            how_to_read(demo_findings.design),
            set(demo_run_counts.values()),
            "the HTML report's reading rules",
        )
        assert checked, "the reading rules state no replicate count at all"

    def test_the_findings_cautions_state_only_this_designs_counts(
        self, demo_pairwise, demo_effect_size, demo_run_counts
    ):
        cautions = [
            pairwise_finding(demo_pairwise).caution,
            effect_size_finding(demo_effect_size, "auc", demo_run_counts).caution,
        ]

        checked = sum(
            assert_claims_match(caution, set(demo_run_counts.values()), "a finding's caution")
            for caution in cautions
        )
        assert checked, "no caution states a replicate count at all"

    def test_a_caution_follows_the_comparisons_it_is_attached_to(self, demo_pairwise):
        """The pairwise caution reads its counts off its own table, not off the design.

        A condition dropped for having too few runs never reaches this table, so
        a caution built from the design would describe comparisons the reader
        cannot see.
        """
        reduced = demo_pairwise.head(1).with_columns(
            pl.lit(3).alias(col.N_RUNS_A), pl.lit(3).alias(col.N_RUNS_B)
        )

        assert_claims_match(pairwise_finding(reduced).caution, {3}, "a reduced caution")


class TestReplicateCountsInTheRepositorysOwnProse:
    """Source, tests and README may only state counts the demo dataset really has.

    A reader of a public repository cannot check a number against data they do
    not have. Every replicate count written down here therefore has to be one
    ``da-pipeline-demo`` produces -- which is also the rule that would have
    caught the two stale claims in the briefing, since they were written into
    the template as ordinary prose.

    Prose that does not want to commit to a count can simply not state one:
    "a handful of runs per condition" matches nothing here and is checked by
    nothing.
    """

    @pytest.fixture(scope="module")
    def repository_prose(self) -> list[tuple[str, str]]:
        root = Path(__file__).resolve().parents[1]

        if not (root / "src").is_dir():
            pytest.skip("running against an installed package, not the source tree")

        paths = [
            path
            for directory in ("src", "tests")
            for path in sorted((root / directory).rglob("*.py"))
            if "__pycache__" not in path.parts
        ]
        paths.append(root / "README.md")

        return [
            (str(path.relative_to(root)), path.read_text(encoding="utf-8"))
            for path in paths
            if path.exists()
        ]

    def test_no_file_states_a_count_the_demo_design_does_not_have(
        self, repository_prose, demo_run_counts
    ):
        counts = set(demo_run_counts.values())
        checked = sum(
            assert_claims_match(text, counts, name) for name, text in repository_prose
        )

        assert checked, "the scan found nothing, which means it stopped working"

    def test_the_scan_catches_the_wording_that_prompted_it(self, demo_run_counts):
        """The two shapes of claim this test exists for, against a design lacking them.

        Built from the counts rather than written out, so that the examples stay
        wrong for whatever design the demo generator produces -- and so that this
        file does not itself contain the stale sentences the scan above forbids.
        """
        counts = set(demo_run_counts.values())
        absent = max(counts) + 2

        for wording in (
            f"at {absent} runs against {absent} it is simply a count of pairings",
            f"At {absent} to {absent + 1} runs they are wide",
        ):
            with pytest.raises(AssertionError):
                assert_claims_match(wording, counts, "a deliberately stale sentence")
