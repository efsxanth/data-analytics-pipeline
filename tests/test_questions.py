
"""Tests for the predefined LLM research questions."""

from da_pipeline.llm.questions import RESEARCH_QUESTIONS


def test_four_research_questions_are_defined():
    assert len(RESEARCH_QUESTIONS) == 4


def test_question_identifiers_are_unique():
    identifiers = [question.identifier for question in RESEARCH_QUESTIONS]

    assert len(identifiers) == len(set(identifiers))


def test_every_question_has_text_and_evidence():
    for question in RESEARCH_QUESTIONS:
        assert question.identifier
        assert question.question
        assert question.table_paths


def test_evidence_paths_are_unique_per_question():
    for question in RESEARCH_QUESTIONS:
        assert len(question.table_paths) == len(set(question.table_paths))


def test_research_questions_have_expected_identifiers():
    assert tuple(question.identifier for question in RESEARCH_QUESTIONS) == (
        "condition_comparisons",
        "reproducibility",
        "relationships",
        "multivariable_model",
    )


def test_reproducibility_uses_both_statistical_tables():
    question = next(
        q for q in RESEARCH_QUESTIONS
        if q.identifier == "reproducibility"
    )

    assert question.table_paths == (
        "variance_decomposition",
        "run_spread_by_condition",
    )
