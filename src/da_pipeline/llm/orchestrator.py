"""Run the four research-question interpretations independently.

This module does not create an API client or send requests by itself. The caller
supplies a model; each question is evaluated using the existing LLM client.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .client import generate_interpretation
from .evidence_selector import EvidenceSelectionError, select_evidence
from .questions import RESEARCH_QUESTIONS, ResearchQuestion
from .schemas import QuestionInterpretation
from .scientific_review import ScientificReviewWarning, review_interpretation
from .validator import EvidenceValidationError

# Long enough for Groq's explanation of a rejected request ("Request too large
# for model ... Limit 8000, Requested 9120"), short enough to stay a log line.
_DETAIL_LIMIT = 300

# Local errors whose messages this package writes itself and that name only
# tables, rows and columns -- never model prose or API payloads.
_SAFE_LOCAL_ERRORS = (EvidenceValidationError, EvidenceSelectionError)


@dataclass(frozen=True)
class QuestionRunResult:
    """One question's validated result, review warnings, or failure."""

    question_id: str
    question: str
    interpretation: QuestionInterpretation | None
    warnings: tuple[ScientificReviewWarning, ...] = ()
    error_type: str | None = None
    error_detail: str | None = None
    attempts: int = 1

    @property
    def succeeded(self) -> bool:
        """Whether the question returned a numerically validated answer."""
        return self.interpretation is not None


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _DETAIL_LIMIT else text[: _DETAIL_LIMIT - 3] + "..."


def is_correctable(exc: BaseException) -> bool:
    """Whether the model answered but wrongly, so asking again may succeed."""
    if isinstance(exc, EvidenceValidationError | ValidationError):
        return True
    if getattr(exc, "status_code", None) == 400:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        return isinstance(error, dict) and error.get("code") == "json_validate_failed"
    return False


def describe_error(exc: BaseException) -> str | None:
    """A short, log-safe account of why a request failed, or None.

    Without this, a failed publish shows only 'BadRequestError', which cannot
    distinguish a rejected schema, an over-budget request, or a truncated
    generation. For API errors only the structured fields Groq returns are
    used -- status, error type/code and its own message. ``failed_generation``,
    which echoes model output, is deliberately never read, and ``str(exc)`` is
    avoided because it embeds the whole response body.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        parts = [f"HTTP {status}"]
        if isinstance(error, dict):
            for key in ("type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value:
                    parts.append(f"{key}={value}")
            message = error.get("message")
            if isinstance(message, str) and message:
                parts.append(_truncate(message))
        return "; ".join(parts)

    if isinstance(exc, _SAFE_LOCAL_ERRORS):
        return _truncate(str(exc))

    return None


def interpret_all_questions(
    evidence_pack: dict,
    model: Any,
    *,
    generate: Callable[
        [dict, ResearchQuestion, Any], QuestionInterpretation
    ] = generate_interpretation,
    delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 1,
) -> list[QuestionRunResult]:
    """Interpret each predefined question; preserve independent failures.

    A successful answer has passed the numerical validator in the supplied
    generator. Scientific warnings are advisory, not proof of correctness.
    The optional generator dependency enables tests with no network calls.
    A pause between requests can reduce token-per-minute rate-limit failures.

    With ``max_attempts`` > 1, an answer the model got wrong -- a number that
    does not match its cited row, a disallowed table, malformed JSON -- is
    asked again with the reason it was rejected. Budget, rate-limit and
    server errors are not retried here: repeating them cannot help.
    A retried generator is called with a ``feedback`` keyword argument.
    """
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be a finite non-negative number")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    results: list[QuestionRunResult] = []

    for index, question in enumerate(RESEARCH_QUESTIONS):
        if index > 0 and delay_seconds > 0:
            sleep(delay_seconds)

        attempt = 1
        feedback = None
        while True:
            try:
                if feedback is None:
                    interpretation = generate(evidence_pack, question, model)
                else:
                    interpretation = generate(
                        evidence_pack, question, model, feedback=feedback
                    )
                selected = select_evidence(evidence_pack, question)
                warnings = review_interpretation(interpretation, selected)
            except Exception as exc:
                if attempt < max_attempts and is_correctable(exc):
                    # The model answered, but wrongly; tell it what was wrong.
                    feedback = describe_error(exc) or (
                        "The answer did not match the required structure."
                    )
                    attempt += 1
                    if delay_seconds > 0:
                        sleep(delay_seconds)
                    continue
                # Never save raw exception messages: API errors can contain
                # generated text or portions of the submitted evidence. Only
                # the vetted summary from describe_error is kept.
                results.append(
                    QuestionRunResult(
                        question_id=question.identifier,
                        question=question.question,
                        interpretation=None,
                        error_type=type(exc).__name__,
                        error_detail=describe_error(exc),
                        attempts=attempt,
                    )
                )
                break

            results.append(
                QuestionRunResult(
                    question_id=question.identifier,
                    question=question.question,
                    interpretation=interpretation,
                    warnings=tuple(warnings),
                    attempts=attempt,
                )
            )
            break

    return results
