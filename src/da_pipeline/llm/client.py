"""LangChain adapter for structured statistical interpretations."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .evidence_selector import select_evidence
from .prompt_builder import build_interpretation_messages
from .questions import ResearchQuestion
from .schemas import QuestionInterpretation, response_json_schema
from .validator import validate_evidence_references


def generate_interpretation(
    evidence_pack: dict,
    question: ResearchQuestion,
    model: Any,
    *,
    feedback: str | None = None,
) -> QuestionInterpretation:
    """Generate one structured interpretation using an injected model.

    ``feedback`` explains why a previous attempt at this question was rejected,
    so a retry can correct it rather than repeat it.
    """

    system_message, user_message = build_interpretation_messages(
        evidence_pack,
        question,
        feedback=feedback,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]
    )

    structured_model = model.with_structured_output(
        response_json_schema(question.identifier, question.table_paths),
        method="json_schema",
        strict=True,
    )

    chain = prompt | structured_model

    raw = chain.invoke({})

    # A dict schema yields a dict; the Pydantic model enforces every local
    # constraint (non-empty text, at least one citation, row_index >= 0).
    result = (
        raw
        if isinstance(raw, QuestionInterpretation)
        else QuestionInterpretation.model_validate(raw)
    )

    selected_evidence = select_evidence(evidence_pack, question)

    validate_evidence_references(
        result,
        question,
        selected_evidence,
    )

    return result
