
"""Build evidence-grounded prompts for predefined research questions."""

import json
from collections.abc import Mapping

from .evidence_encoder import encode_evidence
from .evidence_selector import select_evidence
from .guardrails import STATISTICAL_GUARDRAILS
from .questions import ResearchQuestion


def build_interpretation_messages(
    evidence_pack: Mapping,
    question: ResearchQuestion,
    *,
    feedback: str | None = None,
) -> tuple[str, str]:
    """Return system and user messages for one research question.

    ``feedback`` is the reason a previous answer was rejected; it is appended
    so that a retry fixes that specific mistake.
    """

    selected = select_evidence(evidence_pack, question)
    encoded = encode_evidence(selected)

    system_message = (
        "You are a statistical research interpretation assistant.\n\n"
        f"{STATISTICAL_GUARDRAILS.strip()}\n\n"
        "Use only the supplied statistical evidence. "
        "Do not invent numerical results or claim causal effects. "
        "Return a structured interpretation with evidence references. "
        "Each reference must identify a permitted table path, a zero-based "
        "row index, and numerical values copied from that row."
        " Evidence tables use a compact column-oriented format: "
        "each table has 'columns' and 'rows'. "
        "Each row's values correspond positionally to the columns. "
        "The first column, 'row_index', holds each row's zero-based index: "
        "copy it into the reference's 'row_index' field, never count rows, "
        "and never cite 'row_index' inside 'values'. "
        "Copy every number from the same row whose row_index you cite. "
        "A null cell means no value is available (for example, a "
        "leave-one-out statistic for a condition with only two runs); "
        "never cite a null cell and never report it as 0. "
        "A table may also have 'constants': columns whose value is the same "
        "in every row of that table, written once; they belong to every row "
        "and may be cited with any of that table's row indices. "
        "Row indices are zero-based and must refer to the original "
        "row positions in the supplied tables."
        " IMPORTANT EVIDENCE REFERENCE RULES: "
        "Use the complete dot-separated table path exactly as it appears "
        "in the supplied evidence hierarchy. "
        "For example, use 'correlations.correlations_between_run', "
        "not 'correlations_between_run'. "
        "The 'values' field must be a list of objects. "
        "Each object must contain 'column' (the numerical statistical column name) "
        "and 'value' (the numerical value copied from the referenced row). "
        "For example: 'values': [{'column': 'pearson_r', 'value': 0.5}]. "
        "Never include text fields such as 'condition', 'var_x', 'var_y', "
        "'metric', or 'unit_of_analysis' inside 'values'. "
        "Use the 'explanation' field to identify the relevant metrics, "
        "conditions, and analytical level. "
        "Every numerical value must exactly match the cited row."
        " REQUIRED OUTPUT FIELDS: "
        "Every item in the 'evidence' list MUST contain all four fields: "
        "'table', 'row_index', 'values', and 'explanation'. "
        "Never omit 'explanation', even for repeated or closely related evidence. "
        "Write a short, non-empty explanation for every referenced row. "
        "Prefer a small number of highly relevant evidence references "
        "over a long list of repetitive references. "
        "Include the top-level fields 'question_id', 'interpretation', "
        "'evidence', and 'caveats'. "
        "Check that every evidence reference is complete before responding."
    )

    user_message = (
        f"Research question: {question.question}\n"
        f"Question ID: {question.identifier}\n"
        "ALLOWED EVIDENCE TABLE PATHS (use exactly these names):\n"
        + "\n".join(f"- {path}" for path in question.table_paths)
        + "\n\n"
        "Interpret the following statistical evidence:\n"
        f"{json.dumps(encoded, ensure_ascii=False, separators=(',', ':'))}"
    )

    if feedback:
        user_message += (
            "\n\nYOUR PREVIOUS ANSWER TO THIS QUESTION WAS REJECTED: "
            f"{feedback}\n"
            "Answer again. Re-read the cited row by its row_index, copy numbers "
            "exactly as written, cite only numerical columns, and use only the "
            "allowed table paths."
        )

    return system_message, user_message
