
"""Tests for LLM statistical interpretation guardrails."""

import pytest

from da_pipeline.llm.guardrails import STATISTICAL_GUARDRAILS


def test_guardrails_are_not_empty():
    assert STATISTICAL_GUARDRAILS.strip()


@pytest.mark.parametrize(
    "required_section",
    [
        "EXPERIMENTAL UNIT",
        "STATISTICAL SIGNIFICANCE",
        "EFFECT SIZES AND UNCERTAINTY",
        "CORRELATION LEVELS",
        "REPRODUCIBILITY",
        "MULTIVARIABLE MODELS",
        "NUMERICAL ACCURACY",
        "UNCERTAINTY AND LIMITATIONS",
        "INTERPRETATION SCOPE",
    ],
)
def test_required_guardrail_sections_exist(required_section):
    assert required_section in STATISTICAL_GUARDRAILS


def test_guardrails_preserve_independent_run_principle():
    assert "independent run" in STATISTICAL_GUARDRAILS.lower()
    assert "experimental unit" in STATISTICAL_GUARDRAILS.lower()


def test_guardrails_distinguish_correlation_levels():
    text = STATISTICAL_GUARDRAILS.lower()

    assert "pooled" in text
    assert "within-run" in text
    assert "between-run" in text


def test_guardrails_warn_against_causal_claims():
    assert "causation" in STATISTICAL_GUARDRAILS.lower()


def test_guardrails_require_numerical_accuracy():
    text = STATISTICAL_GUARDRAILS.lower()

    assert "never invent" in text
    assert "references" in text


def test_guardrails_evaluate_within_run_predictors_separately():
    text = STATISTICAL_GUARDRAILS.lower()

    assert "evaluate each predictor separately" in text
    assert "do not describe all within-run relationships as weak" in text
    assert "moderate or strong within-run associations" in text


def test_guardrails_avoid_generalizing_across_conditions():
    text = STATISTICAL_GUARDRAILS.lower()

    assert "distinguish evidence from one condition" in text
    assert "do not claim a pattern holds across multiple conditions" in text
    assert "unless the supplied results support that claim" in text


def test_guardrails_do_not_conflate_between_and_within_run():
    text = STATISTICAL_GUARDRAILS.lower()

    assert "a strong between-run correlation does not establish" in text
    assert "relationship holds within individual runs" in text
