
"""Validate references in LLM-generated statistical interpretations."""

import math
from collections.abc import Mapping

from .questions import ResearchQuestion
from .schemas import QuestionInterpretation


class EvidenceValidationError(ValueError):
    """Raised when an interpretation references invalid evidence."""


def validate_evidence_references(
    interpretation: QuestionInterpretation,
    question: ResearchQuestion,
    selected_evidence: Mapping,
) -> None:
    """Ensure evidence references point to existing, permitted table rows."""

    if interpretation.question_id != question.identifier:
        raise EvidenceValidationError(
            "Interpretation question ID does not match the research question."
        )

    tables = selected_evidence.get("tables", {})

    for reference in interpretation.evidence:
        if reference.table not in question.table_paths:
            raise EvidenceValidationError(
                f"Table '{reference.table}' is not allowed for this question."
            )

        current = tables

        for key in reference.table.split("."):
            if not isinstance(current, Mapping) or key not in current:
                raise EvidenceValidationError(
                    f"Referenced table '{reference.table}' does not exist."
                )
            current = current[key]

        if not isinstance(current, list):
            raise EvidenceValidationError(
                f"Referenced table '{reference.table}' is not a row list."
            )

        if reference.row_index >= len(current):
            raise EvidenceValidationError(
                f"Row {reference.row_index} does not exist "
                f"in table '{reference.table}'."
            )

        row = current[reference.row_index]

        if not isinstance(row, Mapping):
            raise EvidenceValidationError(
                f"Referenced row is not a valid mapping "
                f"in table '{reference.table}'."
            )

        seen_columns = set()

        for numerical_value in reference.values:
            column = numerical_value.column
            reported_value = numerical_value.value

            if column in seen_columns:
                raise EvidenceValidationError(
                    f"Duplicate numerical column '{column}'."
                )

            seen_columns.add(column)

            if column not in row:
                raise EvidenceValidationError(
                    f"Numerical column '{column}' not found "
                    f"in table '{reference.table}'."
                )

            actual_value = row[column]
            location = (
                f"column '{column}' of row {reference.row_index} "
                f"in table '{reference.table}'"
            )

            # Specific messages, because they are fed back to the model on a
            # retry: "invalid value" alone led it to repeat the same citation.
            if actual_value is None:
                raise EvidenceValidationError(
                    f"Cited {location}, but that cell is null (no value is "
                    "available). Do not cite null cells."
                )
            if (
                isinstance(actual_value, bool)
                or not isinstance(actual_value, (int, float))
                or not math.isfinite(actual_value)
            ):
                raise EvidenceValidationError(
                    f"Cited {location}, which is not a numerical value. "
                    "Cite only numerical columns."
                )
            if not math.isfinite(reported_value):
                raise EvidenceValidationError(
                    f"Reported a non-finite value for {location}."
                )

            if not math.isclose(
                actual_value,
                reported_value,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise EvidenceValidationError(
                    f"Numerical mismatch for '{column}': "
                    f"reported {reported_value}, "
                    f"actual {actual_value}."
                )
