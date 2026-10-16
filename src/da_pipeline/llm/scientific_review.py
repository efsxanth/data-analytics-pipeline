"""Conservative, offline review flags for LLM statistical interpretations.

Warnings prompt human review; they do not prove a narrative claim is false.
Numerical and table-reference validity are checked separately by validator.py.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .schemas import QuestionInterpretation


@dataclass(frozen=True)
class ScientificReviewWarning:
    """A possible overclaim requiring review of the cited evidence."""

    code: str
    message: str


def review_interpretation(
    interpretation: QuestionInterpretation,
    selected_evidence: Mapping,
) -> list[ScientificReviewWarning]:
    """Flag a few known overclaims without changing the generated answer.

    Call this only after ``validate_evidence_references`` succeeds.
    This is a deliberately limited set of heuristics, not semantic verification.
    """

    warnings: list[ScientificReviewWarning] = []
    narrative = interpretation.interpretation.lower()
    tables = selected_evidence.get("tables", {})

    # Rows for each predictor in a *full* model repeat the full model's R².
    # They cannot establish a comparison with a separately fitted reduced model.
    model_comparison_claim = bool(
        re.search(
            r"(?:no|little|zero) (?:additional )?predictive value beyond"
            r"|identical to (?:the )?(?:model|results) with only"
            r"|same (?:r²|r2|r-squared) as (?:the )?.*?only",
            narrative,
        )
    )
    has_reduced_model = isinstance(tables, Mapping) and any(
        name in tables
        for name in ("reduced_model", "baseline_model", "model_comparison")
    )
    if (
        interpretation.question_id == "multivariable_model"
        and model_comparison_claim
        and not has_reduced_model
    ):
        warnings.append(
            ScientificReviewWarning(
                code="unsupported_model_comparison",
                message=(
                    "The narrative compares full-model performance with a simpler "
                    "model, but no reduced-model comparison table was supplied. "
                    "Review this claim against separately fitted model results."
                ),
            )
        )

    # A blanket description of within-run correlations as weak deserves review
    # when the answer itself cites a moderately large within-run correlation.
    blanket_weak_claim = bool(
        re.search(r"within.run.{0,100}\b(?:are|were|remain) weak\b", narrative)
        or re.search(r"\bweak within.run (?:correlations|relationships|associations)", narrative)
    )
    substantial_within_run_reference = any(
        "within_run" in reference.table
        and any(
            value.column in ("pearson_r_median", "spearman_r_median", "pearson_r", "spearman_r")
            and abs(value.value) >= 0.5
            for value in reference.values
        )
        for reference in interpretation.evidence
    )
    if blanket_weak_claim and substantial_within_run_reference:
        warnings.append(
            ScientificReviewWarning(
                code="overgeneralized_within_run_weakness",
                message=(
                    "The narrative broadly calls within-run associations weak, "
                    "but a cited within-run correlation has magnitude at least 0.5. "
                    "Assess each predictor and condition separately."
                ),
            )
        )

    return warnings
