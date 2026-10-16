"""The run report, the evidence pack, and the briefing that goes with it."""

from __future__ import annotations

import json

import polars as pl
import pytest

from da_pipeline.constants import columns as col
from da_pipeline.reporting.evidence import (
    AI_SUBDIR,
    PACK_FNAME,
    RUN_MAPPING_FNAME,
    build_run_mapping,
    collect_run_ids,
    merge_bin_summaries,
    pseudonymise,
    pseudonymise_text,
    renames_anything,
    write_evidence_pack,
)
from da_pipeline.reporting.findings import (
    KIND_EFFECT_SIZE,
    Finding,
    RunFindings,
    build_run_findings,
    correlation_finding,
    pairwise_finding,
    run_spread_finding,
)
from da_pipeline.reporting.html_report import build_report_html
from da_pipeline.reporting.prompt import build_prompt

from .conftest import OUTCOME_METRIC  # noqa: I001


class TestPseudonymisation:
    """Renaming a foreign dataset's runs to ``run_01``, ``run_02``, ...

    The examples below are ``run-1``, ``run-13`` and so on -- the same word, a
    different spelling. That is deliberate: pseudonymisation only does anything
    to identifiers that are *not* already in the canonical ``run_NN`` form, so a
    test written with canonical ids would exercise the skip path instead (see
    TestCanonicalRunIdsNeedNoMapping for that one).
    """

    def test_runs_are_numbered_in_natural_order(self):
        """run-2 must precede run-10, or the mapping is unreadable beside the original."""
        mapping = build_run_mapping(["run-10", "run-2", "run-1"])

        assert mapping == {"run-1": "run_01", "run-2": "run_02", "run-10": "run_03"}

    def test_identifiers_are_collected_from_every_column_that_carries_one(self):
        spread = pl.DataFrame({
            col.CONDITION: ["A", "B"],
            col.MOST_INFLUENTIAL_RUN: ["run-04", "run-09"],
        })
        by_run = pl.DataFrame({col.RUN_ID: ["run-01", "run-04"]})

        assert collect_run_ids([spread, by_run]) == ["run-01", "run-04", "run-09"]

    def test_a_frame_with_no_run_column_passes_through(self):
        frame = pl.DataFrame({col.METRIC: ["snr"], col.PCT_CONDITION: [12.0]})

        assert pseudonymise(frame, {"run-01": "run_01"}).equals(frame)

    def test_the_same_run_gets_the_same_pseudonym_in_every_table(self):
        mapping = build_run_mapping(["run-01", "run-02"])
        spread = pl.DataFrame({col.MOST_INFLUENTIAL_RUN: ["run-02"]})
        by_run = pl.DataFrame({col.RUN_ID: ["run-02"]})

        assert (
            pseudonymise(spread, mapping)[col.MOST_INFLUENTIAL_RUN][0]
            == pseudonymise(by_run, mapping)[col.RUN_ID][0]
        )


class TestPseudonymisingProse:
    """The findings name runs inside sentences, so columns alone are not enough."""

    def test_an_identifier_inside_a_sentence_is_replaced(self):
        mapping = {"run-13": "run_08"}

        assert pseudonymise_text(
            "dropping run-13 moves the mean", mapping
        ) == "dropping run_08 moves the mean"

    def test_a_longer_identifier_is_not_eaten_by_a_shorter_one(self):
        """A regression test: run-1 must not rewrite the front of run-13.

        Replacing shortest-first turns 'run-13' into 'run_013', which is a
        pseudonym that maps to nothing and looks like a real one.
        """
        mapping = build_run_mapping(["run-1", "run-13"])

        result = pseudonymise_text("run-13 and run-1 differ", mapping)

        assert result == f"{mapping['run-13']} and {mapping['run-1']} differ"
        assert "run-" not in result

    def test_empty_input_is_returned_untouched(self):
        assert pseudonymise_text(None, {"a": "b"}) is None
        assert pseudonymise_text("text", {}) == "text"


class TestFindings:
    def test_a_non_significant_comparison_is_never_called_equivalent(self):
        """The single most important phrasing rule in the whole report.

        At a handful of runs the smallest attainable two-sided p is large, so
        "did not separate" is a statement about the evidence and "the same" is a
        claim the design cannot support.
        """
        pairwise = pl.DataFrame({
            col.METRIC: ["snr", "snr"],
            col.CONDITION_A: ["A", "A"],
            col.CONDITION_B: ["B", "C"],
            col.CLES: [0.55, 0.60],
            col.HIGHER_CONDITION: ["A", "A"],
            col.P_VALUE_ADJUSTED: [0.42, 0.61],
        })

        finding = pairwise_finding(pairwise)
        text = f"{finding.answer} {finding.caution}".lower()

        assert "no condition pair separated" in finding.answer.lower()
        for forbidden in ("indistinguishable", "equivalent", "no difference", "the same"):
            assert f"are {forbidden}" not in text

    def test_cles_is_reported_from_the_winning_condition_s_side(self):
        """CLES is directional, so a decisive win for B arrives as 0.00.

        Printed beside "Condition_B higher" that reads as a contradiction, and
        two different pairs B won both render as the same row. Both conditions
        are named, and the figure is turned to face the winner.
        """
        pairwise = pl.DataFrame({
            col.METRIC: ["snr", "snr"],
            col.CONDITION_A: ["A", "B"],
            col.CONDITION_B: ["B", "C"],
            col.CLES: [0.0, 1.0],                 # B wins both, from either side
            col.HIGHER_CONDITION: ["B", "B"],
            col.P_VALUE_ADJUSTED: [0.02, 0.03],
        })

        answer = pairwise_finding(pairwise).answer

        assert "B above A (CLES 1.00" in answer
        assert "B above C (CLES 1.00" in answer
        assert "CLES 0.00" not in answer

    def test_the_caution_says_a_large_p_is_not_evidence_of_absence(self):
        pairwise = pl.DataFrame({
            col.METRIC: ["snr"], col.CONDITION_A: ["A"], col.CONDITION_B: ["B"],
            col.CLES: [0.5], col.HIGHER_CONDITION: [None], col.P_VALUE_ADJUSTED: [0.9],
        })

        assert "has not been shown to be equivalent" in pairwise_finding(pairwise).caution

    def test_a_manufactured_correlation_is_named_as_such(self):
        """Strong pooled, nothing within-run: the pooling artefact, in one row."""
        keys = {col.CONDITION: ["A"], col.CORR_VAR_X: ["snr"], col.CORR_VAR_Y: ["acc"]}
        pooled = pl.DataFrame({**keys, col.CORR_PEARSON_R: [0.94]})
        within = pl.DataFrame({**keys, col.CORR_PEARSON_R_MEDIAN: [0.01]})

        finding = correlation_finding(pooled, within)

        assert "only between runs" in finding.answer
        assert "not evidence" in finding.caution

    def test_the_relative_shift_ranking_warns_about_near_zero_metrics(self):
        """A metric near zero tops the ranking on a tiny absolute movement."""
        spread = pl.DataFrame({
            col.CONDITION: ["A", "B"],
            col.METRIC: ["big_metric", "tiny_metric"],
            col.CONDITION_MEAN: [100.0, 0.01],
            col.MAX_LEAVE_ONE_OUT_SHIFT: [5.0, 0.01],
            col.MOST_INFLUENTIAL_RUN: ["run-01", "run-02"],
        })

        finding = run_spread_finding(spread)

        assert "tiny_metric" in finding.answer
        assert "near zero" in finding.caution

    def test_a_frame_that_could_not_be_computed_contributes_no_finding(self):
        empty = pl.DataFrame()
        variance = pl.DataFrame({
            col.METRIC: ["snr"], col.RUN_TO_CONDITION_RATIO: [0.4],
            col.N_OBSERVATIONS: [900], col.N_RUNS_TOTAL: [6],
        })

        findings = build_run_findings(
            response_name="auc",
            outcome_metric=OUTCOME_METRIC,
            metrics={
                "outcomes": [OUTCOME_METRIC],
                "components": ["Accuracy_error_1"],
                "predictors": ["degradation_1"],
            },
            condition_run_counts={"A": 3, "B": 3},
            effect_size_df=None,
            pairwise_df=None,
            variance_df=variance,
            run_spread_df=empty,
            correlations_df=empty,
            within_run_correlations_df=empty,
            model_df=empty,
        )

        # Only the variance frame was usable, so only its finding appears.
        assert len(findings.findings) == 1
        assert findings.design["n_conditions"] == 2
        assert findings.design["n_runs"] == 6


class TestBinSummaryMerge:
    def test_shared_columns_appear_once_and_metrics_survive(self, tmp_path):
        """Nine files repeating fourteen identical columns was most of the pack."""
        shared = {
            col.CONDITION: ["A", "A"],
            col.OUTCOME_BIN_INDEX: [1, 2],
            "Accuracy_median": [0.8, 0.9],
        }
        for metric in ("degradation_3", "degradation_2"):
            pl.DataFrame({**shared, f"{metric}_median": [0.1, 0.2]}).write_csv(
                tmp_path / f"{metric}.csv"
            )

        merged = merge_bin_summaries(sorted(tmp_path.glob("*.csv")))

        assert merged.height == 2
        assert merged.columns.count("Accuracy_median") == 1
        assert {"degradation_3_median", "degradation_2_median"} <= set(merged.columns)

    def test_no_files_is_not_an_error(self):
        assert merge_bin_summaries([]) is None

    def test_files_without_the_shared_key_are_left_alone(self, tmp_path):
        pl.DataFrame({"a": [1]}).write_csv(tmp_path / "odd.csv")

        assert merge_bin_summaries([tmp_path / "odd.csv"]) is None


@pytest.fixture
def minimal_findings():
    return RunFindings(
        response_name="auc",
        design={
            "response_name": "auc",
            "n_conditions": 2,
            "n_runs": 6,
            "runs_per_condition": {"Condition_A": 3, "Condition_B": 3},
            "n_observations": 12345,
            "unit_of_analysis": "The run.",
        },
        findings=[
            Finding(KIND_EFFECT_SIZE, "Do they differ?", "Yes, a bit.",
                    "Not by much, though."),
        ],
    )


class TestReportHtml:
    def test_the_page_is_self_contained(self, minimal_findings, tmp_path):
        """No external request, or the report breaks the moment it is emailed."""
        html = build_report_html(minimal_findings, tmp_path)

        assert "<!doctype html>" in html.lower()
        assert "<style>" in html
        for external in ("http://", "https://", "<script"):
            assert external not in html

    def test_the_findings_and_their_cautions_both_appear(self, minimal_findings, tmp_path):
        html = build_report_html(minimal_findings, tmp_path)

        assert "Do they differ?" in html
        assert "Yes, a bit." in html
        assert "Not by much, though." in html

    def test_values_are_escaped(self, tmp_path):
        findings = RunFindings(
            response_name="auc",
            design={"n_conditions": 1, "runs_per_condition": {}},
            findings=[Finding(KIND_EFFECT_SIZE, "<script>x</script>", "a & b", None)],
        )

        html = build_report_html(findings, tmp_path)

        assert "<script>x</script>" not in html
        assert "&lt;script&gt;" in html


class TestPrompt:
    def test_the_briefing_carries_the_rules_that_are_easy_to_get_wrong(self, minimal_findings):
        prompt = build_prompt(minimal_findings)

        # Each of these corresponds to a mistake made while building this
        # project. Kept short so a line wrap in the template cannot break them.
        assert "Never report a non-significant comparison as showing equivalence" in prompt
        assert "Pooled correlations are descriptions, not evidence" in prompt
        assert "condition as a predictor" in prompt
        assert "negative out-of-sample R-squared is a finding" in prompt
        assert "Wide confidence intervals are the honest answer" in prompt

    def test_the_run_design_is_filled_in(self, minimal_findings):
        prompt = build_prompt(minimal_findings)

        assert "2 conditions" in prompt
        assert "6 runs" in prompt
        assert "Condition_A (3 runs)" in prompt
        assert "{" not in prompt.split("## 1.")[0]      # no unfilled placeholder

    def test_the_briefing_describes_the_design_it_was_given(self, minimal_findings):
        """Anyone should be able to clone this and run it on their own data.

        The briefing explains the *structure* -- conditions, runs, observations,
        repeated measurements -- and nothing about the equipment or material a
        particular dataset came from, because it has no way of knowing either.
        Everything it can say about the setting comes from the design dict and
        the metric names, so asserting that it is built from those is what keeps
        it portable.
        """
        prompt = build_prompt(minimal_findings)

        for term in ("condition", "run", "observation"):
            assert term in prompt.lower()

    def test_tabs_need_no_script(self, minimal_findings, tmp_path):
        """Radio inputs and labels, so the page stays inert and renders anywhere."""
        html = build_report_html(minimal_findings, tmp_path)

        assert "<script" not in html
        assert 'type="radio" name="report-tab"' in html
        assert ":checked ~ .panels" in html

    def test_exactly_one_tab_is_open_on_load(self, minimal_findings, tmp_path):
        html = build_report_html(minimal_findings, tmp_path)

        assert html.count(" checked>") == 1

    def test_the_radios_stay_keyboard_reachable(self, minimal_findings, tmp_path):
        """Clipped, not display:none -- a hidden input cannot be focused."""
        html = build_report_html(minimal_findings, tmp_path)

        assert "clip-path: inset(50%)" in html
        assert ".tab-state {\n  position: absolute" in html

    def test_printing_reveals_every_panel(self, minimal_findings, tmp_path):
        """Tabs would otherwise print as whichever one happened to be open."""
        html = build_report_html(minimal_findings, tmp_path)

        print_block = html[html.index("@media print"):]
        assert ".panels > section { display: block !important" in print_block
        assert ".tab-bar { display: none" in print_block

    def test_a_tab_with_nothing_in_it_is_dropped(self, tmp_path):
        """An empty run should not offer tabs that open onto nothing."""
        findings = RunFindings(
            response_name="auc",
            design={"n_conditions": 1, "runs_per_condition": {"A": 2}},
            findings=[Finding(KIND_EFFECT_SIZE, "q", "a", None)],
        )

        html = build_report_html(findings, tmp_path)

        # No figures and no CSVs on disk, so only the tabs with their own body
        # survive: the overview, and the reading rules.
        assert 'id="panel-overview"' in html
        assert 'id="panel-reading"' in html
        assert 'id="panel-relationships"' not in html
        assert 'id="panel-tables"' not in html

    def test_every_tab_label_points_at_a_panel_that_exists(self, minimal_findings, tmp_path):
        """A label with no matching panel renders as a tab that does nothing."""
        import re

        html = build_report_html(minimal_findings, tmp_path)

        labelled = set(re.findall(r'<label for="tab-([a-z]+)"', html))
        panels = set(re.findall(r'<section id="panel-([a-z]+)"', html))
        styled = set(re.findall(r'#tab-([a-z]+):checked ~ \.panels', html))

        assert labelled == panels == styled


class TestEvidencePackEndToEnd:
    """Exercised against a real run in test_pipeline; these cover the shape."""

    def test_the_pack_is_valid_json_and_declares_its_schema(self, tmp_path):
        from da_pipeline.reporting.evidence import build_evidence_pack

        findings = RunFindings(
            response_name="auc",
            design={"n_conditions": 1, "runs_per_condition": {"A": 2}},
            findings=[Finding(KIND_EFFECT_SIZE, "q", "dropping run-01 helped", None)],
        )

        pack = build_evidence_pack(findings, tmp_path, {"run-01": "run_01"})

        assert pack["schema"] == "da-pipeline/evidence-pack/2"
        assert "run-01" not in json.dumps(pack)
        assert "run_01" in pack["findings"][0]["answer"]


class TestCanonicalRunIdsNeedNoMapping:
    """Runs already named run_NN map onto themselves, so nothing is written.

    A mapping file that pairs every name with itself conceals nothing and
    decodes nothing, but it still reads as a secret to whoever finds it beside
    a pack they were about to send.
    """

    @staticmethod
    def _findings():
        return RunFindings(
            response_name="auc",
            design={"n_conditions": 1, "runs_per_condition": {"A": 2}},
            findings=[Finding(KIND_EFFECT_SIZE, "q", "dropping run_01 helped", None)],
        )

    def test_an_identity_mapping_is_recognised(self):
        assert not renames_anything(build_run_mapping(["run_01", "run_02"]))
        assert renames_anything(build_run_mapping(["run-7", "run-9"]))

    def test_no_mapping_file_is_written_for_canonical_ids(self, tmp_path):
        frames = [pl.DataFrame({col.RUN_ID: ["run_01", "run_02"]})]

        write_evidence_pack(self._findings(), tmp_path, frames)

        ai_dir = tmp_path / AI_SUBDIR
        assert (ai_dir / PACK_FNAME).exists()
        assert not (ai_dir / RUN_MAPPING_FNAME).exists()

    def test_the_pack_says_the_identifiers_were_left_alone(self, tmp_path):
        frames = [pl.DataFrame({col.RUN_ID: ["run_01", "run_02"]})]

        write_evidence_pack(self._findings(), tmp_path, frames)

        pack = json.loads((tmp_path / AI_SUBDIR / PACK_FNAME).read_text(encoding="utf-8"))

        assert "unchanged" in pack["notes"]["run_identifiers"]
        assert "run_01" in pack["findings"][0]["answer"]

    def test_a_foreign_identifier_still_gets_its_mapping_file(self, tmp_path):
        # The skip must be conditional: someone else's run names still need
        # pseudonymising, and a pack with no key would be undecodable.
        frames = [pl.DataFrame({col.RUN_ID: ["run-7", "run-9"]})]

        write_evidence_pack(self._findings(), tmp_path, frames)

        mapping_path = tmp_path / AI_SUBDIR / RUN_MAPPING_FNAME
        assert mapping_path.exists()
        assert json.loads(mapping_path.read_text(encoding="utf-8"))["mapping"] == {
            "run_01": "run-7", "run_02": "run-9",
        }
