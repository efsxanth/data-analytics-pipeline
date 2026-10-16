
"""Tests for question-specific LLM prompt assembly."""

import json

import pytest

from da_pipeline.llm.guardrails import STATISTICAL_GUARDRAILS
from da_pipeline.llm.prompt_builder import build_interpretation_messages
from da_pipeline.llm.questions import RESEARCH_QUESTIONS


@pytest.fixture
def evidence_pack():
    return {
        "schema": "da-pipeline/evidence-pack/2",
        "design": {
            "outcome_metric": "yield",
            "metrics": {
                "outcomes": ["yield"],
                "components": ["yield_error"],
                "predictors": ["temperature"],
            },
            "n_runs": 6,
            "unit_of_analysis": "run",
        },
        "findings": [
            {
                "kind": "correlations",
                "answer": "SECRET_PRECOMPUTED_ANSWER",
            }
        ],
        "tables": {
            "effect_size": [{"metric": "yield", "eta_squared": 0.4}],
            "pairwise_conditions": [{"metric": "yield", "p_adj": 0.03}],
            "variance_decomposition": [{"metric": "yield"}],
            "run_spread_by_condition": [{"metric": "yield"}],
            "correlations": {
                "correlations": [
                    {
                        "var_x": "temperature",
                        "var_y": "yield",
                        "pearson_r": 0.6,
                    },
                    {
                        "var_x": "temperature",
                        "var_y": "yield_error",
                        "pearson_r": 0.2,
                    },
                ],
                "correlations_within_run": [
                    {
                        "var_x": "temperature",
                        "var_y": "yield",
                        "pearson_r_median": 0.5,
                    }
                ],
                "correlations_between_run": [
                    {
                        "var_x": "temperature",
                        "var_y": "yield",
                        "pearson_r": 0.4,
                    }
                ],
            },
            "outcome_model": [{"predictor": "temperature"}],
            "outcome_model_by_run": [{"run_id": "run_01"}],
        },
        "notes": {"rounding": "Six significant figures"},
    }


@pytest.mark.parametrize("question", RESEARCH_QUESTIONS)
def test_prompt_contains_question_and_guardrails(evidence_pack, question):
    system_message, user_message = build_interpretation_messages(
        evidence_pack, question
    )

    assert question.question in user_message
    assert question.identifier in user_message
    assert STATISTICAL_GUARDRAILS.strip() in system_message


@pytest.mark.parametrize("question", RESEARCH_QUESTIONS)
def test_prompt_excludes_precomputed_answers(evidence_pack, question):
    system_message, user_message = build_interpretation_messages(
        evidence_pack, question
    )

    assert "SECRET_PRECOMPUTED_ANSWER" not in system_message
    assert "SECRET_PRECOMPUTED_ANSWER" not in user_message


def test_relationships_prompt_filters_unrelated_pairs(evidence_pack):
    _, user_message = build_interpretation_messages(
        evidence_pack, RESEARCH_QUESTIONS[2]
    )

    evidence_json = user_message.split(
        "Interpret the following statistical evidence:\n", 1
    )[1]
    selected = json.loads(evidence_json)

    pooled = selected["tables"]["correlations"]["correlations"]
    columns = pooled["columns"]
    rows = pooled["rows"]

    var_x_index = columns.index("var_x")
    var_y_index = columns.index("var_y")

    assert len(rows) == 1
    assert rows[0][var_x_index] == "temperature"
    assert rows[0][var_y_index] == "yield"



def test_prompt_contains_only_question_specific_tables(evidence_pack):
    _, user_message = build_interpretation_messages(
        evidence_pack, RESEARCH_QUESTIONS[0]
    )

    evidence_json = user_message.split(
        "Interpret the following statistical evidence:\n", 1
    )[1]
    selected = json.loads(evidence_json)

    assert set(selected["tables"]) == {
        "effect_size",
        "pairwise_conditions",
    }

    assert "findings" not in selected
    assert "correlations" not in selected["tables"]

    effect_size = selected["tables"]["effect_size"]

    assert effect_size["columns"] == ["row_index", "metric", "eta_squared"]
    assert effect_size["rows"] == [[0, "yield", 0.4]]


@pytest.mark.parametrize("question", RESEARCH_QUESTIONS)
def test_prompt_requires_complete_evidence_references(
    evidence_pack, question
):
    system_message, _ = build_interpretation_messages(
        evidence_pack, question
    )

    assert "REQUIRED OUTPUT FIELDS" in system_message

    for field in ("table", "row_index", "values", "explanation"):
        assert f"'{field}'" in system_message

    assert "Never omit 'explanation'" in system_message


@pytest.mark.parametrize("question", RESEARCH_QUESTIONS)
def test_prompt_describes_numerical_value_list(evidence_pack, question):
    system_message, _ = build_interpretation_messages(
        evidence_pack,
        question,
    )

    assert "The 'values' field must be a list of objects" in system_message
    assert "'column'" in system_message
    assert "'value'" in system_message
    assert "numerical value copied from the referenced row" in system_message


@pytest.mark.parametrize("question", RESEARCH_QUESTIONS)
def test_prompt_lists_allowed_evidence_table_paths(
    evidence_pack, question
):
    _, user_message = build_interpretation_messages(
        evidence_pack,
        question,
    )

    assert "ALLOWED EVIDENCE TABLE PATHS" in user_message

    for path in question.table_paths:
        assert f"- {path}" in user_message
