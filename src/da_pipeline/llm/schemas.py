
"""Structured schemas for LLM-generated statistical interpretations."""

from pydantic import BaseModel, ConfigDict, Field


class NumericalValue(BaseModel):
    """A named numerical value from an evidence table."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    value: float


class EvidenceReference(BaseModel):
    """A reference to a specific statistical evidence row."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(
        description="Evidence table path, such as 'effect_size'."
    )
    row_index: int = Field(
        ge=0,
        description="Zero-based index of the supporting table row.",
    )
    values: list[NumericalValue] = Field(
        min_length=1,
        description=(
            "Numerical values copied from the referenced table row. "
            "Each item identifies its statistical column and value."
        ),
    )
    explanation: str = Field(
        min_length=1,
        description="How this statistical result supports the interpretation.",
    )


class QuestionInterpretation(BaseModel):
    """The LLM-generated answer to one predefined research question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        description="Identifier of the predefined research question."
    )
    interpretation: str = Field(
        min_length=1,
        description="Evidence-grounded interpretation of the results.",
    )
    evidence: list[EvidenceReference] = Field(
        min_length=1,
        description="Statistical evidence supporting this interpretation.",
    )
    caveats: list[str] = Field(
        description=(
            "Statistical cautions and limitations. "
            "Use an empty list when no caveats are available."
        ),
    )


def response_json_schema(question_id: str, table_paths: tuple[str, ...]) -> dict:
    """The JSON Schema sent to Groq for one question.

    Written out rather than generated from the Pydantic models, for two
    reasons. ``minLength``/``minItems``/``minimum`` are dropped, because with
    them Groq has been seen answering with a string where the schema requires a
    number -- i.e. not constraining generation at all. The Pydantic models
    still enforce those limits locally on every answer. And ``table`` and
    ``question_id`` become enums, so the model cannot cite a shortened table
    name ('correlations_within_run') or another question's identifier.
    """
    numerical_value = {
        "type": "object",
        "description": "A named numerical value from an evidence table.",
        "properties": {
            "column": {
                "type": "string",
                "description": "Name of a numerical column of the cited row.",
            },
            "value": {
                "type": "number",
                "description": "The number in that column of the cited row.",
            },
        },
        "required": ["column", "value"],
        "additionalProperties": False,
    }
    evidence_reference = {
        "type": "object",
        "description": "A reference to a specific statistical evidence row.",
        "properties": {
            "table": {
                "type": "string",
                "enum": list(table_paths),
                "description": "Evidence table path, exactly as listed.",
            },
            "row_index": {
                "type": "integer",
                "description": "The row's row_index value (first value of the row).",
            },
            "values": {
                "type": "array",
                "items": numerical_value,
                "description": "Numbers copied from the cited row, one per column.",
            },
            "explanation": {
                "type": "string",
                "description": "How this statistical result supports the interpretation.",
            },
        },
        "required": ["table", "row_index", "values", "explanation"],
        "additionalProperties": False,
    }
    return {
        "title": "QuestionInterpretation",
        "description": "The answer to one predefined research question.",
        "type": "object",
        "properties": {
            "question_id": {"type": "string", "enum": [question_id]},
            "interpretation": {
                "type": "string",
                "description": "Evidence-grounded interpretation of the results.",
            },
            "evidence": {
                "type": "array",
                "items": evidence_reference,
                "description": "Statistical evidence supporting this interpretation.",
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Statistical cautions and limitations; may be empty.",
            },
        },
        "required": ["question_id", "interpretation", "evidence", "caveats"],
        "additionalProperties": False,
    }
