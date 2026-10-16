"""End-to-end runs over synthetic data.

These are the slow tests -- they render every figure -- but they are the only
ones that exercise the stages against each other, including the joins that
silently drop rows when a key drifts. Deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import json
import logging
import re

import polars as pl
import pytest

from da_pipeline.config import ExportSelection, PipelineConfig, StepRange
from da_pipeline.constants import columns as col
from da_pipeline.constants import display as disp
from da_pipeline.constants import outputs as out
from da_pipeline.pipeline import main, run
from da_pipeline.reporting import evidence, html_report, prompt
from da_pipeline.synthetic import degradation_metric_names, metric_names

from .conftest import N_STEPS, OUTCOME_METRIC, TEST_CONDITIONS

# What the reference producer writes, and therefore what a default run
# resolves to. Derived from the generator rather than restated, so a change
# to the metric set cannot leave this file asserting against a stale one.
DECLARED_METRICS = metric_names()
PREDICTORS = degradation_metric_names()
COMPONENTS = tuple(
    name for name in DECLARED_METRICS
    if name.startswith(f"{OUTCOME_METRIC}_")
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, dataset_root):
    """One full run, shared by the assertions below."""
    config = PipelineConfig(
        config_path=dataset_root,
        response_name="auc",
        step_range=StepRange(1, N_STEPS),
        outcome_num_bins=5,
        max_workers=2,
        output_folder_name_suffix="test",
        exports=ExportSelection(
            observation_counts=True,
            responses=True,
            signal_to_noise=True,
            predictor_metrics=True,
            outcome_metrics=True,
            correlations=True,
        ),
    )
    return run(config)


def read(output_dir, *parts) -> pl.DataFrame:
    return pl.read_csv(output_dir.joinpath(out.CSV_EXPORTS_SUBDIR, *parts))


class TestOutputTree:
    def test_run_returns_the_output_directory_it_created(self, completed_run):
        assert completed_run.is_dir()
        assert completed_run.name == "Data_Analytics_test"

    def test_output_sits_beside_the_config_file(self, completed_run, dataset_root):
        assert completed_run.parent == dataset_root.parent

    @pytest.mark.parametrize(
        "subdir",
        [
            out.OBSERVATION_COUNT_SUBDIR,
            out.RESPONSES_SUBDIR,
            out.SNR_SUBDIR,
            out.PREDICTOR_METRICS_SUBDIR,
            out.OUTCOME_METRICS_SUBDIR,
            out.RELATIONSHIPS_SUBDIR,
            out.EFFECT_SIZE_SUBDIR,
            out.VARIANCE_SUBDIR,
            out.MODEL_SUBDIR,
            out.CORRELATIONS_SUBDIR,
        ],
    )
    def test_every_figure_directory_has_content(self, completed_run, subdir):
        figures = list((completed_run / subdir).glob("*.png"))
        assert figures, f"{subdir} produced no figures"

    def test_one_relationship_figure_per_metric(self, completed_run):
        produced = {p.stem for p in (completed_run / out.RELATIONSHIPS_SUBDIR).glob("*.png")}

        # Every predictor metric and every component metric, plus the one that
        # defines the bins, plus signal-to-noise.
        expected = len(PREDICTORS) + len(COMPONENTS) + 1
        assert len(produced) == expected

    def test_correlation_figures_cover_all_three_views(self, completed_run):
        """The figures exist to show the three views disagreeing.

        A comparison figure needs both per-condition views on one axis, so one
        detail figure per condition; the between-run view has no condition and
        gets its own.
        """
        produced = {p.stem for p in (completed_run / out.CORRELATIONS_SUBDIR).glob("*.png")}

        assert "pooled_vs_within_run" in produced
        assert "between_run" in produced
        assert {f"{c.name}_pooled_vs_within_run" for c in TEST_CONDITIONS} <= produced

    def test_run_does_not_write_a_log_file(self, completed_run):
        # Logging is configured by main(), not by run(): a library call has no
        # business reconfiguring the root logger of its caller's process.
        assert not (completed_run / out.LOG_FNAME).exists()


class TestExports:
    def test_requested_csvs_are_all_written(self, completed_run):
        csv_dir = completed_run / out.CSV_EXPORTS_SUBDIR
        written = {p.name for p in csv_dir.glob("*.csv")}

        assert written == {
            out.OBSERVATION_COUNT_CSV_FNAME,
            out.RESPONSES_CSV_FNAME,
            out.SNR_CSV_FNAME,
            out.PREDICTOR_METRICS_CSV_FNAME,
            out.OUTCOME_METRICS_CSV_FNAME,
            # Always written, not gated on an export flag.
            out.EFFECT_SIZE_CSV_FNAME,
            out.VARIANCE_CSV_FNAME,
            out.RUN_SPREAD_CSV_FNAME,
            out.PAIRWISE_CSV_FNAME,
            out.MODEL_CSV_FNAME,
            out.MODEL_BY_RUN_CSV_FNAME,
        }

    def test_pairwise_comparisons_cover_every_metric(self, completed_run):
        """Every metric is comparable: the processing behind them is held constant."""
        pairwise = read(completed_run, out.PAIRWISE_CSV_FNAME)
        metrics = pairwise[col.METRIC].to_list()

        assert any("auc" in m for m in metrics)
        assert col.SIGNAL_TO_NOISE_RATIO in metrics
        assert OUTCOME_METRIC in metrics

        # Every condition pair, for every metric.
        n_conditions = len(TEST_CONDITIONS)
        expected_pairs = n_conditions * (n_conditions - 1) // 2
        assert pairwise.height == expected_pairs * pairwise[col.METRIC].n_unique()

    def test_a_derived_metric_carries_a_real_comparison(self, completed_run):
        """Nothing is suppressed on the grounds of how a metric was derived."""
        pairwise = read(completed_run, out.PAIRWISE_CSV_FNAME)

        outcome = pairwise.filter(pl.col(col.METRIC) == OUTCOME_METRIC)
        assert outcome.height
        assert outcome[col.CLES].is_not_null().all()

    def test_pairwise_p_values_are_corrected_within_each_metric(self, completed_run):
        pairwise = read(completed_run, out.PAIRWISE_CSV_FNAME)

        assert (pairwise[col.P_ADJUST_METHOD] == col.P_ADJUST_BENJAMINI_HOCHBERG).all()
        # One family per metric, not one for the whole table.
        assert pairwise[col.P_ADJUST_FAMILY].n_unique() == pairwise[col.METRIC].n_unique()
        # Correction can only make a p-value larger.
        assert (pairwise[col.P_VALUE_ADJUSTED] >= pairwise[col.P_VALUE] - 1e-12).all()

    def test_run_spread_covers_every_condition_and_metric(self, completed_run):
        run_spread = read(completed_run, out.RUN_SPREAD_CSV_FNAME)

        assert set(run_spread[col.CONDITION].unique()) == {c.name for c in TEST_CONDITIONS}
        assert col.SIGNAL_TO_NOISE_RATIO in run_spread[col.METRIC].to_list()
        # Unlike the effect size, this carries no attribution claim, so
        # downstream metrics are included too.
        assert OUTCOME_METRIC in run_spread[col.METRIC].to_list()

    def test_variance_decomposition_covers_every_metric(self, completed_run):
        variance = read(completed_run, out.VARIANCE_CSV_FNAME)
        metrics = variance[col.METRIC].to_list()

        # One row per level of the response, plus every per-observation
        # metric: signal-to-noise, every predictor metric, and the outcome family.
        assert any("auc" in m for m in metrics)
        assert col.SIGNAL_TO_NOISE_RATIO in metrics
        assert set(PREDICTORS) <= set(metrics)
        assert {OUTCOME_METRIC, *COMPONENTS} <= set(metrics)

    def test_every_metric_reports_its_condition_share(self, completed_run):
        variance = read(completed_run, out.VARIANCE_CSV_FNAME)

        assert variance[col.PCT_CONDITION].is_not_null().all()

        outcome = variance.filter(pl.col(col.METRIC) == OUTCOME_METRIC)
        assert outcome.height
        assert outcome[col.PCT_CONDITION].is_not_null().all()

    def test_the_three_shares_account_for_the_whole_spread(self, completed_run):
        variance = read(completed_run, out.VARIANCE_CSV_FNAME)

        total = (
            variance[col.PCT_CONDITION]
            + variance[col.PCT_RUN_WITHIN_CONDITION]
            + variance[col.PCT_OBSERVATION_WITHIN_RUN]
        )
        assert ((total - 100.0).abs() < 1e-6).all()

    def test_all_three_correlation_views_are_written(self, completed_run):
        """Pooled, within-run and between-run: the three questions pooling merges."""
        corr_dir = completed_run / out.CSV_EXPORTS_SUBDIR / out.CORRELATIONS_SUBDIR

        assert {p.name for p in corr_dir.glob("*.csv")} == {
            out.CORRELATIONS_CSV_FNAME,
            out.WITHIN_RUN_CORRELATIONS_CSV_FNAME,
            out.BETWEEN_RUN_CORRELATIONS_CSV_FNAME,
        }

    def test_between_run_correlations_are_over_the_runs_themselves(self, completed_run):
        between = read(
            completed_run, out.CORRELATIONS_SUBDIR, out.BETWEEN_RUN_CORRELATIONS_CSV_FNAME
        )

        assert (between[col.UNIT_OF_ANALYSIS] == col.UNIT_RUN_MEDIAN).all()
        # Every run contributes one point to one coefficient per pair, so the
        # conditions are pooled and the count is the total number of runs.
        assert (between[col.CORR_N_RUNS] == sum(c.n_runs for c in TEST_CONDITIONS)).all()


def declared_names(module) -> set[str]:
    """Every string a constants module publishes, tuples flattened."""
    names: set[str] = set()

    for key, value in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, tuple | list):
            names.update(v for v in value if isinstance(v, str))

    return names


class TestEveryOutputColumnWasDeclared:
    """No output column may appear that ``constants.columns`` did not name.

    Written the positive way round, as an allow-list rather than a list of
    banned words, and that is the point of it. A ban catches only what somebody
    thought to ban; this catches *any* column the pipeline grows that was not
    declared first -- a stray name from a new analysis, an internal working
    column that escaped into a CSV, a rename applied in one stage and not the
    other.

    It also enforces the contract ``constants.columns`` states in its own
    docstring: every column the pipeline creates is named there exactly once.
    Three kinds of name cannot be known in advance and are derived rather than
    listed -- the response, which the caller chooses; the metrics, which the
    extract files declare and the run discovers; and the bin-summary columns,
    which are one of those names plus a declared suffix.

    Deriving the metric names from the reference producer, rather than from a
    constant, is what keeps this strict now that the metric set is open-ended: a
    metric column that no file declared is still a failure.
    """

    def allowed(self, response_name: str) -> set[str]:
        base = declared_names(col) | set(DECLARED_METRICS) | {response_name}
        return base | {
            f"{metric}{suffix}"
            for metric in base
            for suffix in col.DISTRIBUTION_SUFFIXES
        }

    def test_a_metric_no_file_declared_is_not_waved_through(self):
        """The allow-list must not have become 'anything that looks like a metric'."""
        allowed = self.allowed("auc")

        assert "degradation_1" in allowed          # the producer declares it
        assert "degradation_99" not in allowed     # nothing does
        assert "degradation_1_median" in allowed   # declared metric + declared suffix
        assert "degradation_99_median" not in allowed

    def test_no_csv_carries_an_undeclared_column(self, completed_run):
        allowed = self.allowed("auc")

        strays = [
            (path.name, column)
            for path in (completed_run / out.CSV_EXPORTS_SUBDIR).rglob("*.csv")
            for column in pl.read_csv(path, n_rows=0).columns
            if column not in allowed
        ]

        assert not strays

    def test_the_response_column_is_the_one_the_caller_asked_for(self, completed_run):
        """The response is named by configuration, so it is not in the allow-list.

        Guarding it separately keeps the test above strict: without this, any
        stray column could be waved through by claiming to be the response.
        """
        responses = read(completed_run, out.RESPONSES_CSV_FNAME)

        assert "auc" in responses.columns
        assert set(responses.columns) - {"auc"} <= declared_names(col)


def words(text: str) -> set[str]:
    """Every word in a path segment, punctuation and digits set aside."""
    return {part for part in re.split(r"[^A-Za-z]+", text) if part}


class TestEveryOutputPathWasDeclared:
    """No directory or file may be named with a word nothing declared.

    The same shape as the column guard above, applied to the output tree, and
    for the same reason: a ban-list catches only the words somebody thought to
    ban, while an allow-list catches *any* word that reaches a path without
    having been declared somewhere first -- a stray label from a new figure, a
    rename applied in one stage and not another, or vocabulary from a metric set
    the code was never supposed to know about.

    Five sources are legitimate, and between them they should account for every
    word in the tree:

    * ``constants.outputs`` -- the directory and dataset names, plus the
      file-name vocabulary it declares for exactly this purpose;
    * ``constants.columns`` and ``constants.display`` -- names that reach file
      names through the figures built from them;
    * the reporting modules' own file names (``report.html``, ``prompt.md``, …);
    * the metrics the extract files declared, discovered per run;
    * the caller's own choices -- the response name and the condition names.
    """

    def allowed(self, response_name: str, conditions) -> set[str]:
        declared: set[str] = set()

        for module in (out, col, disp):
            for name in declared_names(module):
                declared |= words(name)

        for name in (
            evidence.AI_SUBDIR,
            evidence.PACK_FNAME,
            evidence.RUN_MAPPING_FNAME,
            prompt.PROMPT_FNAME,
            html_report.REPORT_FNAME,
        ):
            declared |= words(name)

        for metric in DECLARED_METRICS:
            declared |= words(metric)

        declared |= words(response_name)
        for condition in conditions:
            declared |= words(condition)

        return declared

    def test_no_output_path_uses_an_undeclared_word(self, completed_run):
        allowed = self.allowed("auc", [c.name for c in TEST_CONDITIONS])

        strays = sorted({
            (str(path.relative_to(completed_run)), word)
            for path in completed_run.rglob("*")
            for segment in path.relative_to(completed_run).parts
            for word in words(segment) - allowed
        })

        assert not strays

    def test_the_guard_would_catch_a_word_that_was_never_declared(self, completed_run):
        """A guard that cannot fail is not a guard.

        Without this, the test above would keep passing if ``allowed`` ever grew
        broad enough to admit anything at all.
        """
        allowed = self.allowed("auc", [c.name for c in TEST_CONDITIONS])

        assert "boxplot" in allowed                    # declared vocabulary
        assert "degradation" in allowed                # a metric the files declared
        assert "auc" in allowed                        # the caller's response
        assert not words("chromatograph") <= allowed   # nothing declares this

    def test_the_output_root_and_its_directories_are_declared(self, completed_run):
        """Directory names are fixed and name a role, so they are exact matches."""
        expected = {
            out.OBSERVATION_COUNT_SUBDIR, out.RESPONSES_SUBDIR, out.SNR_SUBDIR,
            out.PREDICTOR_METRICS_SUBDIR, out.OUTCOME_METRICS_SUBDIR,
            out.RELATIONSHIPS_SUBDIR, out.EFFECT_SIZE_SUBDIR, out.VARIANCE_SUBDIR,
            out.MODEL_SUBDIR, out.CSV_EXPORTS_SUBDIR, out.CORRELATIONS_SUBDIR,
            out.BIN_SUMMARY_SUBDIR, evidence.AI_SUBDIR,
        }
        produced = {p.name for p in completed_run.rglob("*") if p.is_dir()}

        assert produced <= expected

    def test_every_figure_is_named_for_something_the_run_declared(self, completed_run):
        """Beyond the shared vocabulary, a figure names what it is about.

        A metric, a condition, a response level, or the observation count -- all
        of them things the run discovered or the caller chose, never a word the
        code carries about a subject matter.
        """
        subjects = (
            set(DECLARED_METRICS)
            | {c.name for c in TEST_CONDITIONS}
            | {"auc", out.SNR_LABEL, col.LEVEL, col.ETA_SQUARED}
            # The observation-count barplots are about the run's size.
            | {col.OBSERVATION_COUNT.split("_")[0]}
        )

        unattributed = [
            path.name
            for path in completed_run.rglob("*.png")
            if not any(subject in path.stem for subject in subjects)
        ]

        # The variance and correlation-overview figures are about the run as a
        # whole rather than about one metric, and are named for their directory.
        assert set(unattributed) <= {
            f"{out.VARIANCE_SUBDIR}.png",
            "pooled_vs_within_run.png",
            "between_run.png",
        }


class TestReportAndAIInputs:
    def test_the_report_is_one_self_contained_file(self, completed_run):
        """It has to survive being emailed away from its own directory."""
        report = completed_run / "report.html"
        assert report.is_file()

        html = report.read_text(encoding="utf-8")
        assert "data:image/png;base64," in html      # figures travel with it
        for external in ("http://", "https://", "<script"):
            assert external not in html

    def test_the_report_carries_no_filesystem_path(self, completed_run):
        html = (completed_run / "report.html").read_text(encoding="utf-8")

        # The directory's name identifies the run; the machine it sat on does not.
        assert completed_run.name in html
        assert str(completed_run.parent) not in html

    def test_the_ai_directory_holds_a_pack_a_prompt_and_a_local_mapping(self, completed_run):
        ai_dir = completed_run / "ai"

        assert {p.name for p in ai_dir.glob("*")} == {
            "evidence_pack.json", "prompt.md", "run_id_mapping.json"
        }

    def test_the_pack_leaks_no_path_and_no_original_run_identifier(self, completed_run):
        """The pack is the file most likely to be pasted somewhere else."""
        blob = (completed_run / "ai" / "evidence_pack.json").read_text(encoding="utf-8")
        pack = json.loads(blob)

        assert pack["schema"] == "da-pipeline/evidence-pack/2"
        assert str(completed_run) not in blob
        assert ".h5" not in blob

        # The synthetic dataset names its runs after the condition, so none of
        # them is in the canonical form and every one has to be replaced. Built
        # from the fixture rather than read back from the mapping file, so this
        # still fails if the mapping is written empty.
        for condition in TEST_CONDITIONS:
            for index in range(condition.n_runs):
                assert f"{condition.name}.{index}" not in blob

    def test_the_mapping_stays_out_of_the_pack_but_is_kept_locally(self, completed_run):
        mapping = json.loads(
            (completed_run / "ai" / "run_id_mapping.json").read_text(encoding="utf-8")
        )

        assert "do not send this file with it" in mapping["purpose"]
        assert len(mapping["mapping"]) == sum(c.n_runs for c in TEST_CONDITIONS)

    def test_the_pack_carries_the_analysis_tables_but_not_the_raw_exports(self, completed_run):
        pack = json.loads(
            (completed_run / "ai" / "evidence_pack.json").read_text(encoding="utf-8")
        )
        tables = pack["tables"]

        assert {"variance_decomposition", "run_spread_by_condition", "outcome_model"} <= set(tables)
        # Per-observation exports are inputs to the analysis, not results of it.
        assert not {
            "responses", "signal_to_noise", "predictor_metrics", "outcome_metrics",
        } & set(tables)

    def test_the_prompt_is_written_and_names_this_run(self, completed_run):
        prompt = (completed_run / "ai" / "prompt.md").read_text(encoding="utf-8")

        assert f"{len(TEST_CONDITIONS)} conditions" in prompt
        assert "The run is the experimental unit" in prompt


class TestOutcomeModel:
    """The multivariable model: what predicts the outcome once they compete."""

    def test_every_predictor_gets_a_row(self, completed_run):
        model = read(completed_run, out.MODEL_CSV_FNAME)

        assert set(model[col.PREDICTOR]) == {
            col.SIGNAL_TO_NOISE_RATIO, *PREDICTORS
        }
        assert (model[col.OUTCOME] == OUTCOME_METRIC).all()

    def test_condition_is_not_a_predictor(self, completed_run):
        """Not a modelling choice: condition is constant within a run.

        A per-run design matrix carrying it would be singular, so the question
        "does the condition change the outcome?" is answered by the pairwise
        comparisons on run medians instead.
        """
        model = read(completed_run, out.MODEL_CSV_FNAME)

        assert col.CONDITION not in model[col.PREDICTOR].to_list()

    def test_the_fit_is_within_run_and_says_so(self, completed_run):
        model = read(completed_run, out.MODEL_CSV_FNAME)

        assert (model[col.UNIT_OF_ANALYSIS] == col.UNIT_OBSERVATION_WITHIN_RUN).all()
        assert (model[col.STANDARDISATION] == col.STANDARDISED_WITHIN_RUN).all()
        assert (model[col.N_RUNS_MODELLED] == sum(c.n_runs for c in TEST_CONDITIONS)).all()

    def test_sign_agreement_cannot_exceed_the_runs_fitted(self, completed_run):
        model = read(completed_run, out.MODEL_CSV_FNAME)

        assert (model[col.COEFFICIENT_SAME_SIGN] <= model[col.N_RUNS_MODELLED]).all()
        # The majority side is at least half the runs, by definition.
        assert (model[col.COEFFICIENT_SAME_SIGN] * 2 >= model[col.N_RUNS_MODELLED]).all()

    def test_cross_validation_ran_and_is_reported(self, completed_run):
        """Sign agreement measures consistency; only a held-out run measures transfer."""
        model = read(completed_run, out.MODEL_CSV_FNAME)

        assert model[col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN].is_not_null().all()
        assert (model[col.N_RUNS_PREDICTED] <= model[col.N_RUNS_MODELLED]).all()

    def test_in_sample_fit_is_a_real_share_of_variance(self, completed_run):
        model = read(completed_run, out.MODEL_CSV_FNAME)

        in_sample = model[col.R_SQUARED_IN_SAMPLE_MEDIAN]
        assert ((in_sample >= 0.0) & (in_sample <= 1.0)).all()

    def test_the_per_run_file_backs_the_summary(self, completed_run):
        by_run = read(completed_run, out.MODEL_BY_RUN_CSV_FNAME)
        model = read(completed_run, out.MODEL_CSV_FNAME)

        n_runs = sum(c.n_runs for c in TEST_CONDITIONS)
        assert by_run.height == n_runs * model.height
        assert set(by_run[col.CONDITION].unique()) == {c.name for c in TEST_CONDITIONS}

        # The summary's median is the median of the rows in this file.
        for predictor, expected in zip(
            model[col.PREDICTOR].to_list(), model[col.COEFFICIENT_MEDIAN].to_list()
        ):
            coefficients = by_run.filter(pl.col(col.PREDICTOR) == predictor)[col.COEFFICIENT]
            assert coefficients.median() == pytest.approx(expected)

    def test_effect_size_export_states_its_unit_of_analysis(self, completed_run):
        effect_size = read(completed_run, out.EFFECT_SIZE_CSV_FNAME)

        assert (effect_size[col.UNIT_OF_ANALYSIS] == col.UNIT_RUN_MEDIAN).all()
        # One observation per run, not per observation.
        assert effect_size[col.N_RUNS_TOTAL][0] == sum(c.n_runs for c in TEST_CONDITIONS)

    def test_bin_summaries_are_written_per_metric(self, completed_run):
        summaries = {
            p.name
            for p in (completed_run / out.CSV_EXPORTS_SUBDIR / out.BIN_SUMMARY_SUBDIR).glob("*.csv")
        }

        expected = {out.snr_bin_summary_fname(OUTCOME_METRIC)} | {
            out.bin_summary_fname(metric, OUTCOME_METRIC)
            for metric in (*PREDICTORS, *COMPONENTS)
        }
        assert summaries == expected

    def test_bin_summary_names_carry_the_outcome_metric(self, completed_run):
        """A fixed word in the file name would name the wrong quantity."""
        summaries = {
            p.name
            for p in (completed_run / out.CSV_EXPORTS_SUBDIR / out.BIN_SUMMARY_SUBDIR).glob("*.csv")
        }

        assert all(f"_and_{OUTCOME_METRIC}_bin_summary.csv" in name for name in summaries)


class TestResultsAreCoherent:
    def test_every_condition_reaches_the_output(self, completed_run):
        snr = read(completed_run, out.SNR_CSV_FNAME)

        assert set(snr[col.CONDITION].unique()) == {c.name for c in TEST_CONDITIONS}

    def test_per_observation_frames_agree_on_row_count(self, completed_run):
        # These three are joined on the observation identity downstream. If they
        # disagree, an inner join has quietly dropped observations.
        heights = {
            name: read(completed_run, name).height
            for name in (
                out.SNR_CSV_FNAME, out.PREDICTOR_METRICS_CSV_FNAME, out.OUTCOME_METRICS_CSV_FNAME,
            )
        }

        assert len(set(heights.values())) == 1, heights

    def test_the_planted_quality_gradient_survives_the_pipeline(self, completed_run):
        """The generator makes A the best condition and D the worst.

        This is the assertion that ties the synthetic data to the output: if any
        stage mixed up conditions, pooled the wrong rows, or lost the condition
        label, this ordering would not come out the other end.
        """
        medians = (
            read(completed_run, out.OUTCOME_METRICS_CSV_FNAME)
            .group_by(col.CONDITION)
            .agg(pl.col(OUTCOME_METRIC).median().alias("m"))
            .sort(col.CONDITION)["m"]
            .to_list()
        )

        # Conditions are named in descending quality order in the fixture.
        assert medians == sorted(medians, reverse=True)

    def test_signal_to_noise_follows_the_same_gradient(self, completed_run):
        medians = (
            read(completed_run, out.SNR_CSV_FNAME)
            .group_by(col.CONDITION)
            .agg(pl.col(col.SIGNAL_TO_NOISE_RATIO).median().alias("m"))
            .sort(col.CONDITION)["m"]
            .to_list()
        )

        assert medians == sorted(medians, reverse=True)

    def test_observation_counts_are_aggregated_per_condition_and_group(self, completed_run):
        observation_counts = read(completed_run, out.OBSERVATION_COUNT_CSV_FNAME)

        assert observation_counts.select([col.CONDITION, col.GROUP_ID]).is_duplicated().sum() == 0
        assert (observation_counts[col.OBSERVATION_COUNT_TOTAL] > 0).all()


class TestCommandLineEntryPoint:
    """The ``da-pipeline`` command, from argv to files on disk."""

    @pytest.fixture
    def restore_logging(self):
        # main() reconfigures the root logger; put it back for the other tests.
        # The handlers it installed own an open log file, so they are closed
        # rather than just dropped.
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        yield
        for handler in root.handlers[:]:
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:], root.level = saved_handlers, saved_level

    def test_main_runs_and_writes_its_log(self, dataset_root, restore_logging):
        main([
            "--data_analytics_config_fpath", str(dataset_root),
            "--response_name", "auc",
            "--step_range", "1", str(N_STEPS),
            "--outcome_num_bins", "3",
            "--max_workers", "2",
            "--calculate_effect_size", "false",
            "--output_folder_name_suffix", "viacli",
        ])

        output_dir = dataset_root.parent / "Data_Analytics_viacli"
        log = (output_dir / out.LOG_FNAME).read_text(encoding="utf-8")

        assert "All data validations completed successfully" in log
        # The log opens with the settings the run actually used.
        assert "- response_name: auc" in log
        assert "- outcome_num_bins: 3" in log
        assert (output_dir / out.SNR_SUBDIR).is_dir()


class TestGroupSubset:
    def test_restricting_to_one_group_restricts_the_output(self, dataset_root, tmp_path):
        config = PipelineConfig(
            config_path=dataset_root,
            response_name="auc",
            step_range=StepRange(1, N_STEPS),
            group_ids=[2],
            outcome_num_bins=3,
            max_workers=2,
            calculate_effect_size=False,
        ).with_output_dir(tmp_path / "subset")

        output_dir = run(config)

        snr = read(output_dir, out.SNR_CSV_FNAME)
        assert set(snr[col.GROUP_ID].unique()) == {2}
