"""Offline regression tests for conservative scientific-review warnings."""

from da_pipeline.llm.schemas import (
    EvidenceReference,
    NumericalValue,
    QuestionInterpretation,
)
from da_pipeline.llm.scientific_review import review_interpretation


def make_interpretation(question_id, narrative, table, column, value):
    return QuestionInterpretation(
        question_id=question_id,
        interpretation=narrative,
        evidence=[
            EvidenceReference(
                table=table,
                row_index=0,
                values=[NumericalValue(column=column, value=value)],
                explanation="A supporting statistical result.",
            )
        ],
        caveats=[],
    )


def test_unsupported_reduced_model_comparison_is_flagged():
    result = make_interpretation(
        "multivariable_model",
        "The multivariable model adds no predictive value beyond signal-to-noise ratio.",
        "outcome_model",
        "r_squared_out_of_sample_median",
        0.426099,
    )

    warnings = review_interpretation(result, {"tables": {"outcome_model": []}})

    assert [warning.code for warning in warnings] == [
        "unsupported_model_comparison"
    ]


def test_model_comparison_warning_requires_model_comparison_claim():
    result = make_interpretation(
        "multivariable_model",
        "The full model has held-out R-squared of about 0.43.",
        "outcome_model",
        "r_squared_out_of_sample_median",
        0.426099,
    )

    assert review_interpretation(result, {"tables": {"outcome_model": []}}) == []


def test_comparison_table_prevents_unsupported_comparison_warning():
    result = make_interpretation(
        "multivariable_model",
        "The multivariable model adds no predictive value beyond a simpler model.",
        "outcome_model",
        "r_squared_out_of_sample_median",
        0.426099,
    )

    assert review_interpretation(
        result,
        {"tables": {"outcome_model": [], "model_comparison": []}},
    ) == []


def test_blanket_weak_within_run_claim_is_flagged():
    result = make_interpretation(
        "relationships",
        "The within-run associations are weak across all predictors.",
        "correlations.correlations_within_run",
        "pearson_r_median",
        0.65822,
    )

    warnings = review_interpretation(result, {"tables": {}})

    assert [warning.code for warning in warnings] == [
        "overgeneralized_within_run_weakness"
    ]


def test_moderate_within_run_correlation_without_blanket_claim_is_not_flagged():
    result = make_interpretation(
        "relationships",
        "Signal-to-noise ratio shows a moderate positive within-run association.",
        "correlations.correlations_within_run",
        "pearson_r_median",
        0.65822,
    )

    assert review_interpretation(result, {"tables": {}}) == []
