
"""Tests for structured LLM interpretation schemas."""

import pytest
from pydantic import ValidationError

from da_pipeline.llm.schemas import (
    EvidenceReference,
    NumericalValue,
    QuestionInterpretation,
)


def test_valid_evidence_reference():
    reference = EvidenceReference(
        table="effect_size",
        row_index=0,
        values=[NumericalValue(column="eta_squared", value=0.9)],
        explanation="The effect size quantifies the condition difference.",
    )

    assert reference.table == "effect_size"
    assert reference.row_index == 0


def test_negative_row_index_is_rejected():
    with pytest.raises(ValidationError):
        EvidenceReference(
            table="effect_size",
            row_index=-1,
            values=[NumericalValue(column="eta_squared", value=0.9)],
            explanation="Supporting evidence.",
        )


def test_empty_explanation_is_rejected():
    with pytest.raises(ValidationError):
        EvidenceReference(
            table="effect_size",
            row_index=0,
            values=[NumericalValue(column="eta_squared", value=0.9)],
            explanation="",
        )


def test_valid_question_interpretation():
    result = QuestionInterpretation(
        question_id="condition_comparisons",
        interpretation="The results indicate differences across conditions.",
        evidence=[
            EvidenceReference(
                table="effect_size",
                row_index=0,
                values=[NumericalValue(column="eta_squared", value=0.9)],
                explanation="This row reports the effect size.",
            )
        ],
        caveats=["The number of independent runs is limited."],
    )

    assert result.question_id == "condition_comparisons"
    assert len(result.evidence) == 1
    assert len(result.caveats) == 1


def test_interpretation_requires_evidence():
    with pytest.raises(ValidationError):
        QuestionInterpretation(
            question_id="condition_comparisons",
            interpretation="An interpretation without supporting evidence.",
            evidence=[],
            caveats=[],
        )


def test_empty_interpretation_is_rejected():
    with pytest.raises(ValidationError):
        QuestionInterpretation(
            question_id="condition_comparisons",
            interpretation="",
            evidence=[
                EvidenceReference(
                    table="effect_size",
                    row_index=0,
                    values=[NumericalValue(column="eta_squared", value=0.9)],
                    explanation="Supporting evidence.",
                )
            ],
            caveats=[],
        )


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        EvidenceReference(
            table="effect_size",
            row_index=0,
            values=[NumericalValue(column="eta_squared", value=0.9)],
            explanation="Supporting evidence.",
            invented_field="unexpected",
        )


def test_caveats_are_required():
    with pytest.raises(ValidationError):
        QuestionInterpretation(
            question_id="reproducibility",
            interpretation="Run-to-run variability should be considered.",
            evidence=[
                EvidenceReference(
                    table="variance_decomposition",
                    row_index=0,
                    values=[
                        NumericalValue(column="n_runs", value=10)
                    ],
                    explanation="This row describes variance across runs.",
                )
            ],
        )


def test_empty_evidence_values_are_rejected():
    with pytest.raises(ValidationError):
        EvidenceReference(
            table="effect_size",
            row_index=0,
            values=[],
            explanation="Supporting evidence.",
        )


def test_nonnumeric_evidence_values_are_rejected():
    with pytest.raises(ValidationError):
        NumericalValue(
            column="eta_squared",
            value="not a number",
        )
