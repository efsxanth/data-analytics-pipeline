
"""Tests for deterministic LLM evidence-reference validation."""

import pytest

from da_pipeline.llm.questions import RESEARCH_QUESTIONS
from da_pipeline.llm.schemas import (
    EvidenceReference,
    NumericalValue,
    QuestionInterpretation,
)
from da_pipeline.llm.validator import (
    EvidenceValidationError,
    validate_evidence_references,
)


@pytest.fixture
def question():
    return RESEARCH_QUESTIONS[2]


@pytest.fixture
def selected_evidence():
    return {
        "tables": {
            "correlations": {
                "correlations": [
                    {
                        "var_x": "degradation_1",
                        "var_y": "Accuracy",
                        "pearson_r": 0.5,
                    }
                ],
                "correlations_within_run": [
                    {
                        "var_x": "degradation_1",
                        "var_y": "Accuracy",
                        "pearson_r": 0.5,
                    }
                ],
                "correlations_between_run": [
                    {
                        "var_x": "degradation_1",
                        "var_y": "Accuracy",
                        "pearson_r": 0.5,
                    }
                ],
            }
        }
    }


def make_interpretation(
    table="correlations.correlations_within_run",
    row_index=0,
    question_id="relationships",
):
    """Build a minimal valid Pydantic interpretation."""
    return QuestionInterpretation(
        question_id=question_id,
        interpretation="The relationship varies across analytical levels.",
        evidence=[
            EvidenceReference(
                table=table,
                row_index=row_index,
                values=[NumericalValue(column="pearson_r", value=0.5)],
                explanation="This statistical row supports the interpretation.",
            )
        ],
        caveats=["The number of independent runs is limited."],
    )


def test_valid_evidence_reference_is_accepted(question, selected_evidence):
    result = make_interpretation()

    assert validate_evidence_references(
        result, question, selected_evidence
    ) is None


def test_nested_correlation_views_are_accepted(question, selected_evidence):
    for table in question.table_paths:
        result = make_interpretation(table=table)

        assert validate_evidence_references(
            result, question, selected_evidence
        ) is None


def test_wrong_question_id_is_rejected(question, selected_evidence):
    result = make_interpretation(question_id="reproducibility")

    with pytest.raises(EvidenceValidationError, match="question ID"):
        validate_evidence_references(result, question, selected_evidence)


def test_unrelated_table_is_rejected(question, selected_evidence):
    result = make_interpretation(table="effect_size")

    with pytest.raises(EvidenceValidationError, match="not allowed"):
        validate_evidence_references(result, question, selected_evidence)


def test_missing_table_is_rejected(question, selected_evidence):
    del selected_evidence["tables"]["correlations"]["correlations_within_run"]

    result = make_interpretation()

    with pytest.raises(EvidenceValidationError, match="does not exist"):
        validate_evidence_references(result, question, selected_evidence)


def test_out_of_range_row_index_is_rejected(question, selected_evidence):
    result = make_interpretation(row_index=10)

    with pytest.raises(EvidenceValidationError, match="Row 10 does not exist"):
        validate_evidence_references(result, question, selected_evidence)


def test_non_list_table_is_rejected(question, selected_evidence):
    selected_evidence["tables"]["correlations"][
        "correlations_within_run"
    ] = {"invalid": "structure"}

    result = make_interpretation()

    with pytest.raises(EvidenceValidationError, match="not a row list"):
        validate_evidence_references(result, question, selected_evidence)


def test_incorrect_numerical_value_is_rejected(question, selected_evidence):
    result = make_interpretation()
    result.evidence[0].values = [
        NumericalValue(column="pearson_r", value=-0.9)
    ]

    with pytest.raises(EvidenceValidationError, match="Numerical mismatch"):
        validate_evidence_references(result, question, selected_evidence)


def test_nonexistent_numerical_column_is_rejected(question, selected_evidence):
    result = make_interpretation()
    result.evidence[0].values = [
    NumericalValue(column="invented_statistic", value=0.5)
]

    with pytest.raises(EvidenceValidationError, match="not found"):
        validate_evidence_references(result, question, selected_evidence)


def test_numerical_value_within_tolerance_is_accepted(
    question, selected_evidence
):
    result = make_interpretation()
    result.evidence[0].values = [
    NumericalValue(column="pearson_r", value=0.5000001)
]

    assert validate_evidence_references(
        result, question, selected_evidence
    ) is None


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nonfinite_reported_values_are_rejected(
    question, selected_evidence, invalid_value
):
    result = make_interpretation()
    result.evidence[0].values = [
    NumericalValue(column="pearson_r", value=invalid_value)
]

    with pytest.raises(EvidenceValidationError, match="non-finite value"):
        validate_evidence_references(result, question, selected_evidence)


def test_nonfinite_actual_value_is_rejected(question, selected_evidence):
    selected_evidence["tables"]["correlations"][
        "correlations_within_run"
    ][0]["pearson_r"] = float("nan")

    result = make_interpretation()

    with pytest.raises(EvidenceValidationError, match="not a numerical value"):
        validate_evidence_references(result, question, selected_evidence)


def test_missing_actual_value_is_rejected(question, selected_evidence):
    selected_evidence["tables"]["correlations"][
        "correlations_within_run"
    ][0]["pearson_r"] = None

    result = make_interpretation()

    with pytest.raises(EvidenceValidationError, match="cell is null"):
        validate_evidence_references(result, question, selected_evidence)


def test_duplicate_numerical_columns_are_rejected(
    question, selected_evidence
):
    result = make_interpretation()

    result.evidence[0].values = [
        NumericalValue(column="pearson_r", value=0.5),
        NumericalValue(column="pearson_r", value=0.5),
    ]

    with pytest.raises(
        EvidenceValidationError,
        match="Duplicate numerical column",
    ):
        validate_evidence_references(
            result,
            question,
            selected_evidence,
        )


def test_null_cell_message_names_the_cell_for_retry_feedback(
    question, selected_evidence
):
    selected_evidence["tables"]["correlations"][
        "correlations_within_run"
    ][0]["pearson_r"] = None

    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence_references(make_interpretation(), question, selected_evidence)

    message = str(excinfo.value)
    assert "'pearson_r'" in message
    assert "row 0" in message
    assert "Do not cite null cells" in message
