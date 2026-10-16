
"""Select statistical evidence for a predefined research question."""

from collections.abc import Mapping
from copy import deepcopy

from .questions import ResearchQuestion


class EvidenceSelectionError(ValueError):
    """Raised when a required statistical table is unavailable."""


def _get_table(tables: Mapping, path: str):
    """Resolve a table path within the evidence pack."""

    current = tables

    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise EvidenceSelectionError(
                f"Required evidence table '{path}' is missing."
            )
        current = current[key]

    return deepcopy(current)


def _filter_outcome_correlations(
    rows: list[dict],
    design: Mapping,
) -> list[dict]:
    """Keep correlations between the outcome and its predictors."""

    outcome = design.get("outcome_metric")
    metrics = design.get("metrics", {})
    predictors = set(metrics.get("predictors", []))
    predictors.add("signal_to_noise_ratio")

    if not outcome or not isinstance(rows, list):
        raise EvidenceSelectionError(
            "Cannot filter correlations without a valid outcome or rows."
        )

    filtered = [
        row
        for row in rows
        if (
            row.get("var_x") == outcome
            and row.get("var_y") in predictors
        ) or (
            row.get("var_y") == outcome
            and row.get("var_x") in predictors
        )
    ]

    if not filtered:
        raise EvidenceSelectionError(
            "No outcome-predictor correlations found."
        )

    return filtered


def select_evidence(
    evidence_pack: Mapping,
    question: ResearchQuestion,
) -> dict:
    """Return only the evidence required for a research question."""

    tables = evidence_pack.get("tables")

    if not isinstance(tables, Mapping):
        raise EvidenceSelectionError(
            "Evidence pack has no valid 'tables' section."
        )

    selected_tables = {}

    for path in question.table_paths:
        current = selected_tables
        keys = path.split(".")

        for key in keys[:-1]:
            current = current.setdefault(key, {})

        selected = _get_table(tables, path)

        if path.startswith("correlations."):
            selected = _filter_outcome_correlations(
                selected,
                evidence_pack.get("design", {}),
            )

        current[keys[-1]] = selected


    return {
        "schema": evidence_pack.get("schema"),
        "question_id": question.identifier,
        "question": question.question,
        "design": deepcopy(evidence_pack.get("design")),
        "tables": selected_tables,
        "notes": deepcopy(evidence_pack.get("notes", {})),
    }
