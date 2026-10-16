
"""Predefined research questions and their supporting evidence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchQuestion:
    """A research question and the evidence required to answer it."""

    identifier: str
    question: str
    table_paths: tuple[str, ...]


RESEARCH_QUESTIONS = (
    ResearchQuestion(
        identifier="condition_comparisons",
        question="Which conditions differ, on what, and how confidently?",
        table_paths=(
            "effect_size",
            "pairwise_conditions",
        ),
    ),
    ResearchQuestion(
        identifier="reproducibility",
        question="Would these differences reproduce?",
        table_paths=(
            "variance_decomposition",
            "run_spread_by_condition",
        ),
    ),
    ResearchQuestion(
        identifier="relationships",
        question="What relates to the outcome metric, and does it hold inside a run?",
        table_paths=(
            "correlations.correlations",
            "correlations.correlations_within_run",
            "correlations.correlations_between_run",
        ),
    ),
    ResearchQuestion(
        identifier="multivariable_model",
        question="What does the multivariable model add?",
        table_paths=(
            "outcome_model",
            "outcome_model_by_run",
        ),
    ),
)
