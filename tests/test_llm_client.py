
"""Offline tests for the LangChain interpretation client."""

import pytest
from langchain_core.runnables import RunnableLambda

from da_pipeline.llm.client import generate_interpretation
from da_pipeline.llm.questions import RESEARCH_QUESTIONS
from da_pipeline.llm.schemas import (
    EvidenceReference,
    NumericalValue,
    QuestionInterpretation,
)
from da_pipeline.llm.validator import EvidenceValidationError


class FakeStructuredModel:
    """Fake model that returns a predetermined structured response."""

    def __init__(self, response):
        self.response = response
        self.requested_schema = None
        self.received_messages = None

    def with_structured_output(self, schema, **kwargs):
        self.requested_schema = schema
        self.structured_output_options = kwargs

        def respond(messages):
            self.received_messages = messages
            return self.response

        return RunnableLambda(respond)


@pytest.fixture
def evidence_pack():
    return {
        "schema": "da-pipeline/evidence-pack/2",
        "design": {
            "outcome_metric": "yield",
            "metrics": {
                "outcomes": ["yield"],
                "components": [],
                "predictors": ["temperature"],
            },
            "n_runs": 6,
            "unit_of_analysis": "run",
        },
        "findings": [
            {"kind": "correlations", "answer": "SECRET_PYTHON_ANSWER"}
        ],
        "tables": {
            "effect_size": [{"metric": "yield", "eta_squared": 0.4}],
            "pairwise_conditions": [{"metric": "yield", "p_adj": 0.03}],
        },
        "notes": {},
    }


def test_client_requests_pydantic_schema(evidence_pack):
    response = QuestionInterpretation(
        question_id="condition_comparisons",
        interpretation="The condition effect size is 0.4.",
        evidence=[
            EvidenceReference(
                table="effect_size",
                row_index=0,
                values=[NumericalValue(column="eta_squared", value=0.4)],
                explanation="This row reports the condition effect size.",
            )
        ],
        caveats=["The number of runs is limited."],
    )

    fake_model = FakeStructuredModel(response)

    result = generate_interpretation(
        evidence_pack,
        RESEARCH_QUESTIONS[0],
        fake_model,
    )

    schema = fake_model.requested_schema
    assert schema["title"] == "QuestionInterpretation"
    assert schema["properties"]["question_id"]["enum"] == ["condition_comparisons"]
    reference = schema["properties"]["evidence"]["items"]["properties"]
    assert reference["table"]["enum"] == ["effect_size", "pairwise_conditions"]
    assert result == response


def test_client_sends_question_and_excludes_python_answer(evidence_pack):
    response = QuestionInterpretation(
        question_id="condition_comparisons",
        interpretation="Condition differences should be interpreted cautiously.",
        evidence=[
            EvidenceReference(
                table="effect_size",
                row_index=0,
                values=[NumericalValue(column="eta_squared", value=0.4)],
                explanation="The effect-size row supports this assessment.",
            )
        ],
        caveats=[],
    )

    fake_model = FakeStructuredModel(response)

    generate_interpretation(
        evidence_pack,
        RESEARCH_QUESTIONS[0],
        fake_model,
    )

    messages = fake_model.received_messages.to_messages()

    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"

    combined_text = "\n".join(message.content for message in messages)

    assert "Which conditions differ" in combined_text
    assert "EXPERIMENTAL UNIT" in combined_text
    assert "SECRET_PYTHON_ANSWER" not in combined_text


def test_client_rejects_incorrect_numerical_evidence(evidence_pack):
    response = QuestionInterpretation(
        question_id="condition_comparisons",
        interpretation="The condition effect size is 0.9.",
        evidence=[
            EvidenceReference(
                table="effect_size",
                row_index=0,
                values=[NumericalValue(column="eta_squared", value=0.9)],
                explanation="This row reports an effect size of 0.9.",
            )
        ],
        caveats=[],
    )

    fake_model = FakeStructuredModel(response)

    with pytest.raises(
        EvidenceValidationError,
        match="Numerical mismatch",
    ):
        generate_interpretation(
            evidence_pack,
            RESEARCH_QUESTIONS[0],
            fake_model,
        )


def test_client_uses_strict_structured_output(evidence_pack):
    response = QuestionInterpretation(
        question_id="condition_comparisons",
        interpretation="The condition effect size is 0.4.",
        evidence=[
            EvidenceReference(
                table="effect_size",
                row_index=0,
                values=[
                    NumericalValue(column="eta_squared", value=0.4)
                ],
                explanation="The effect-size row supports this interpretation.",
            )
        ],
        caveats=[],
    )

    fake_model = FakeStructuredModel(response)

    generate_interpretation(
        evidence_pack,
        RESEARCH_QUESTIONS[0],
        fake_model,
    )

    assert fake_model.structured_output_options == {
        "method": "json_schema",
        "strict": True,
    }


def test_dict_response_is_validated_into_pydantic_model(evidence_pack):
    fake_model = FakeStructuredModel(
        {
            "question_id": "condition_comparisons",
            "interpretation": "The condition effect size is 0.4.",
            "evidence": [
                {
                    "table": "effect_size",
                    "row_index": 0,
                    "values": [{"column": "eta_squared", "value": 0.4}],
                    "explanation": "This row reports the effect size.",
                }
            ],
            "caveats": [],
        }
    )

    result = generate_interpretation(evidence_pack, RESEARCH_QUESTIONS[0], fake_model)

    assert isinstance(result, QuestionInterpretation)
    assert result.evidence[0].values[0].value == 0.4


def test_local_constraints_still_apply_to_dict_responses(evidence_pack):
    from pydantic import ValidationError

    fake_model = FakeStructuredModel(
        {
            "question_id": "condition_comparisons",
            "interpretation": "No evidence cited.",
            "evidence": [],
            "caveats": [],
        }
    )

    with pytest.raises(ValidationError):
        generate_interpretation(evidence_pack, RESEARCH_QUESTIONS[0], fake_model)


def test_wire_schema_has_no_unsupported_constraint_keywords():
    import json

    from da_pipeline.llm.schemas import response_json_schema

    text = json.dumps(response_json_schema("q", ("a", "b.c")))

    for keyword in ("minLength", "minItems", "minimum", "$ref", "$defs"):
        assert keyword not in text


def test_feedback_is_added_to_retry_prompt(evidence_pack):
    from da_pipeline.llm.prompt_builder import build_interpretation_messages

    _, first = build_interpretation_messages(evidence_pack, RESEARCH_QUESTIONS[0])
    _, retry = build_interpretation_messages(
        evidence_pack,
        RESEARCH_QUESTIONS[0],
        feedback="Numerical mismatch for 'p': reported 0.4, actual 0.3.",
    )

    assert "REJECTED" not in first
    assert "REJECTED" in retry
    assert "reported 0.4, actual 0.3" in retry
