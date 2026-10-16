"""Offline tests for four-question LLM orchestration."""

from da_pipeline.llm import orchestrator
from da_pipeline.llm.questions import RESEARCH_QUESTIONS
from da_pipeline.llm.schemas import (
    EvidenceReference,
    NumericalValue,
    QuestionInterpretation,
)


def make_response(question, interpretation="A cautious interpretation."):
    """Return a well-formed fake model response without an API call."""
    return QuestionInterpretation(
        question_id=question.identifier,
        interpretation=interpretation,
        evidence=[
            EvidenceReference(
                table=question.table_paths[0],
                row_index=0,
                values=[NumericalValue(column="statistic", value=0.5)],
                explanation="A cited statistical result.",
            )
        ],
        caveats=[],
    )


def test_orchestrator_processes_all_questions_in_order(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    calls = []
    model = object()
    pack = {}

    def fake_generate(evidence_pack, question, passed_model):
        assert evidence_pack is pack
        assert passed_model is model
        calls.append(question.identifier)
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        pack, model, generate=fake_generate
    )

    expected_ids = [question.identifier for question in RESEARCH_QUESTIONS]
    assert calls == expected_ids
    assert [result.question_id for result in results] == expected_ids
    assert all(result.succeeded for result in results)
    assert all(result.error_type is None for result in results)
    assert all(result.warnings == () for result in results)


def test_one_generation_failure_does_not_stop_other_questions(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    failing_id = RESEARCH_QUESTIONS[1].identifier

    def fake_generate(pack, question, model):
        if question.identifier == failing_id:
            raise TimeoutError("Do not expose potentially sensitive API details")
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=fake_generate
    )

    assert len(results) == 4
    assert [result.succeeded for result in results] == [True, False, True, True]
    assert results[1].error_type == "TimeoutError"
    assert results[1].interpretation is None
    assert results[1].warnings == ()
    assert "sensitive" not in repr(results)


def test_scientific_review_warning_is_attached_to_success(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )

    def fake_generate(pack, question, model):
        if question.identifier == "multivariable_model":
            return make_response(
                question,
                "The model has no additional predictive value beyond "
                "signal-to-noise ratio.",
            )
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=fake_generate
    )
    multivariable = results[3]

    assert multivariable.succeeded
    assert multivariable.error_type is None
    assert [warning.code for warning in multivariable.warnings] == [
        "unsupported_model_comparison"
    ]
    assert all(not result.warnings for result in results[:3])


def test_review_failure_is_isolated(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )

    def fake_review(interpretation, selected_evidence):
        if interpretation.question_id == RESEARCH_QUESTIONS[2].identifier:
            raise ValueError("review failed")
        return []

    monkeypatch.setattr(orchestrator, "review_interpretation", fake_review)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=lambda pack, question, model: make_response(question)
    )

    assert [result.succeeded for result in results] == [True, True, False, True]
    assert results[2].error_type == "ValueError"


def test_delay_occurs_only_between_requests_even_after_failure(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    events = []

    def fake_generate(pack, question, model):
        events.append(("request", question.identifier))
        if question == RESEARCH_QUESTIONS[1]:
            raise TimeoutError("temporary failure")
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {},
        object(),
        generate=fake_generate,
        delay_seconds=75,
        sleep=lambda seconds: events.append(("sleep", seconds)),
    )
    assert events == [
        ("request", RESEARCH_QUESTIONS[0].identifier),
        ("sleep", 75),
        ("request", RESEARCH_QUESTIONS[1].identifier),
        ("sleep", 75),
        ("request", RESEARCH_QUESTIONS[2].identifier),
        ("sleep", 75),
        ("request", RESEARCH_QUESTIONS[3].identifier),
    ]
    assert [result.succeeded for result in results] == [True, False, True, True]


def test_invalid_delay_is_rejected():
    import pytest

    for bad_value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="delay_seconds"):
            orchestrator.interpret_all_questions(
                {}, object(), delay_seconds=bad_value
            )


class FakeGroqError(Exception):
    """Shaped like groq.APIStatusError: a status code and a parsed body."""

    def __init__(self, status_code, body):
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code = status_code
        self.body = body


def test_api_error_detail_keeps_groq_reason_but_not_generation(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    error = FakeGroqError(
        400,
        {
            "error": {
                "message": "Failed to generate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": "SECRET_MODEL_OUTPUT",
            }
        },
    )

    def fake_generate(pack, question, model):
        if question == RESEARCH_QUESTIONS[0]:
            raise error
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=fake_generate
    )

    detail = results[0].error_detail
    assert detail.startswith("HTTP 400")
    assert "code=json_validate_failed" in detail
    assert "Failed to generate JSON" in detail
    assert "SECRET_MODEL_OUTPUT" not in repr(results)


def test_request_too_large_detail_is_reported():
    error = FakeGroqError(
        413,
        {
            "error": {
                "message": "Request too large for model `openai/gpt-oss-20b` "
                "on tokens per minute (TPM): Limit 8000, Requested 9120",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )

    detail = orchestrator.describe_error(error)

    assert detail.startswith("HTTP 413; type=tokens; code=rate_limit_exceeded")
    assert "Requested 9120" in detail


def test_error_detail_is_truncated():
    error = FakeGroqError(400, {"error": {"message": "x" * 5000}})

    assert len(orchestrator.describe_error(error)) < 400


def test_local_validation_message_is_kept_but_other_messages_are_not():
    from da_pipeline.llm.validator import EvidenceValidationError

    assert orchestrator.describe_error(
        EvidenceValidationError("Row 9 does not exist in table 'effect_size'.")
    ) == "Row 9 does not exist in table 'effect_size'."
    assert orchestrator.describe_error(ValueError("model said SECRET")) is None
    assert orchestrator.describe_error(FakeGroqError(503, "not json")) == "HTTP 503"


def test_wrong_answer_is_retried_once_with_feedback(monkeypatch):
    from da_pipeline.llm.validator import EvidenceValidationError

    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    calls = []

    def fake_generate(pack, question, model, *, feedback=None):
        calls.append((question.identifier, feedback))
        if question == RESEARCH_QUESTIONS[0] and feedback is None:
            raise EvidenceValidationError(
                "Numerical mismatch for 'p': reported 0.4, actual 0.3."
            )
        return make_response(question)

    sleeps = []
    results = orchestrator.interpret_all_questions(
        {},
        object(),
        generate=fake_generate,
        delay_seconds=75,
        sleep=sleeps.append,
        max_attempts=2,
    )

    assert all(result.succeeded for result in results)
    assert results[0].attempts == 2
    assert calls[0] == (RESEARCH_QUESTIONS[0].identifier, None)
    assert calls[1] == (
        RESEARCH_QUESTIONS[0].identifier,
        "Numerical mismatch for 'p': reported 0.4, actual 0.3.",
    )
    # One extra pause before the retry, then one before each later question.
    assert sleeps == [75, 75, 75, 75]


def test_retry_gives_up_after_max_attempts(monkeypatch):
    from da_pipeline.llm.validator import EvidenceValidationError

    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    attempts = []

    def fake_generate(pack, question, model, *, feedback=None):
        if question == RESEARCH_QUESTIONS[1]:
            attempts.append(feedback)
            raise EvidenceValidationError("Table 'x' is not allowed for this question.")
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=fake_generate, max_attempts=2
    )

    assert len(attempts) == 2
    assert not results[1].succeeded
    assert results[1].attempts == 2
    assert results[1].error_detail == "Table 'x' is not allowed for this question."


def test_budget_and_rate_limit_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "select_evidence", lambda pack, question: {"tables": {}}
    )
    calls = []

    def fake_generate(pack, question, model, *, feedback=None):
        calls.append(question.identifier)
        if question == RESEARCH_QUESTIONS[0]:
            raise FakeGroqError(413, {"error": {"code": "rate_limit_exceeded"}})
        return make_response(question)

    results = orchestrator.interpret_all_questions(
        {}, object(), generate=fake_generate, max_attempts=2
    )

    assert calls.count(RESEARCH_QUESTIONS[0].identifier) == 1
    assert results[0].attempts == 1


def test_json_validate_failed_is_correctable_but_other_400s_are_not():
    assert orchestrator.is_correctable(
        FakeGroqError(400, {"error": {"code": "json_validate_failed"}})
    )
    assert not orchestrator.is_correctable(
        FakeGroqError(400, {"error": {"code": "invalid_request"}})
    )
    assert not orchestrator.is_correctable(TimeoutError())
