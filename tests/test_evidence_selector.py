
"""Tests for question-driven statistical evidence selection."""

from copy import deepcopy

import pytest

from da_pipeline.llm.evidence_selector import (
    EvidenceSelectionError,
    select_evidence,
)
from da_pipeline.llm.questions import RESEARCH_QUESTIONS


@pytest.fixture
def evidence_pack():
    """Minimal evidence pack with the supported table structure."""
    return {
        "schema": "da-pipeline/evidence-pack/2",
        "design": {
            "n_conditions": 4,
            "n_runs": 10,
            "unit_of_analysis": "The run",
            "outcome_metric": "Accuracy",
            "metrics": {
                "outcomes": ["Accuracy"],
                "components": ["Accuracy_error_1"],
                "predictors": [
                    "degradation_1",
                    "degradation_2",
                    "degradation_3",
                ],
            },
        },
        "findings": [
            {"kind": "pairwise", "answer": "Precomputed answer"}
        ],
        "tables": {
            "effect_size": [{"eta_squared": 0.9}],
            "pairwise_conditions": [{"p_value_adjusted": 0.2}],
            "variance_decomposition": [{"metric": "Accuracy"}],
            "run_spread_by_condition": [{"condition": "Condition_A"}],
            "correlations": {
                "correlations": [
                    {
                        "var_x": "signal_to_noise_ratio",
                        "var_y": "Accuracy",
                        "pearson_r": 0.5,
                    },
                    {
                        "var_x": "degradation_1",
                        "var_y": "degradation_2",
                        "pearson_r": 0.3,
                    },
                    {
                        "var_x": "Accuracy_error_1",
                        "var_y": "Accuracy",
                        "pearson_r": -0.8,
                    },
                ],
                "correlations_within_run": [
                    {
                        "var_x": "signal_to_noise_ratio",
                        "var_y": "Accuracy",
                        "pearson_r_median": 0.4,
                    },
                ],
                "correlations_between_run": [
                    {
                        "var_x": "signal_to_noise_ratio",
                        "var_y": "Accuracy",
                        "pearson_r": 0.3,
                    },
                ],
            },
            "outcome_model": [{"predictor": "signal_to_noise_ratio"}],
            "outcome_model_by_run": [{"run_id": "run_01"}],
            "bin_summaries": [{"outcome_bin_index": 1}],
        },
        "notes": {
            "rounding": "Six significant figures",
        },
    }


def test_each_question_receives_expected_tables(evidence_pack):
    expected = {
        "condition_comparisons": {
            "effect_size",
            "pairwise_conditions",
        },
        "reproducibility": {
            "variance_decomposition",
            "run_spread_by_condition",
        },
        "relationships": {
            "correlations",
        },
        "multivariable_model": {
            "outcome_model",
            "outcome_model_by_run",
        },
    }

    for question in RESEARCH_QUESTIONS:
        selected = select_evidence(evidence_pack, question)

        assert set(selected["tables"]) == expected[question.identifier]


def test_relationships_include_all_three_correlation_views(evidence_pack):
    question = RESEARCH_QUESTIONS[2]

    selected = select_evidence(evidence_pack, question)

    assert set(selected["tables"]["correlations"]) == {
        "correlations",
        "correlations_within_run",
        "correlations_between_run",
    }

def test_precomputed_findings_are_excluded(evidence_pack):
    for question in RESEARCH_QUESTIONS:
        selected = select_evidence(evidence_pack, question)

        assert "findings" not in selected


def test_design_and_notes_are_preserved(evidence_pack):
    selected = select_evidence(evidence_pack, RESEARCH_QUESTIONS[0])

    assert selected["design"] == evidence_pack["design"]
    assert selected["notes"] == evidence_pack["notes"]
    assert selected["schema"] == evidence_pack["schema"]


def test_original_pack_is_not_modified(evidence_pack):
    original = deepcopy(evidence_pack)

    selected = select_evidence(evidence_pack, RESEARCH_QUESTIONS[0])
    selected["tables"]["effect_size"][0]["eta_squared"] = 0.0

    assert evidence_pack == original


def test_missing_required_table_raises_error(evidence_pack):
    del evidence_pack["tables"]["effect_size"]

    with pytest.raises(
        EvidenceSelectionError,
        match="Required evidence table 'effect_size' is missing",
    ):
        select_evidence(evidence_pack, RESEARCH_QUESTIONS[0])


def test_missing_tables_section_raises_error(evidence_pack):
    del evidence_pack["tables"]

    with pytest.raises(
        EvidenceSelectionError,
        match="Evidence pack has no valid 'tables' section",
    ):
        select_evidence(evidence_pack, RESEARCH_QUESTIONS[0])


def test_relationships_exclude_unrelated_metric_pairs(evidence_pack):
    selected = select_evidence(evidence_pack, RESEARCH_QUESTIONS[2])

    pooled = selected["tables"]["correlations"]["correlations"]

    assert len(pooled) == 1
    assert pooled[0]["var_x"] == "signal_to_noise_ratio"
    assert pooled[0]["var_y"] == "Accuracy"


def test_relationships_support_dynamic_metric_names(evidence_pack):
    evidence_pack["design"]["outcome_metric"] = "yield"
    evidence_pack["design"]["metrics"]["outcomes"] = ["yield"]
    evidence_pack["design"]["metrics"]["predictors"] = ["temperature"]

    for rows in evidence_pack["tables"]["correlations"].values():
        rows.append({
            "var_x": "temperature",
            "var_y": "yield",
            "pearson_r": 0.6,
        })

    selected = select_evidence(evidence_pack, RESEARCH_QUESTIONS[2])

    for rows in selected["tables"]["correlations"].values():
        assert len(rows) == 1
        assert rows[0]["var_x"] == "temperature"
        assert rows[0]["var_y"] == "yield"

def test_relationships_accept_reversed_metric_order(evidence_pack):
    evidence_pack["tables"]["correlations"]["correlations"].append(
        {
            "var_x": "Accuracy",
            "var_y": "degradation_1",
            "pearson_r": -0.4,
        }
    )

    selected = select_evidence(evidence_pack, RESEARCH_QUESTIONS[2])
    pooled = selected["tables"]["correlations"]["correlations"]

    assert len(pooled) == 2
    assert any(
        row["var_x"] == "Accuracy"
        and row["var_y"] == "degradation_1"
        for row in pooled
    )

def test_relationships_reject_missing_relevant_correlations(evidence_pack):
    evidence_pack["tables"]["correlations"]["correlations"] = [
        {
            "var_x": "degradation_1",
            "var_y": "degradation_2",
            "pearson_r": 0.3,
        }
    ]

    with pytest.raises(
        EvidenceSelectionError,
        match="No outcome-predictor correlations found",
    ):
        select_evidence(evidence_pack, RESEARCH_QUESTIONS[2])
