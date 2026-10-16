"""Render validated LLM interpretations as safe, self-contained HTML fragments.

The publishing workflow must require all four successful interpretations before
passing the results to the report builder. This module does no model calls.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from da_pipeline.llm.orchestrator import QuestionRunResult


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_ai_interpretation(result: QuestionRunResult) -> str:
    """Render one successful, validated research-question interpretation.

    Never silently render a missing answer as successful. All model-controlled
    strings are HTML-escaped, including column names and explanatory text.
    """
    answer = result.interpretation
    if answer is None or not result.succeeded:
        raise ValueError(f"Missing Groq interpretation for {result.question_id}")
    if answer.question_id != result.question_id:
        raise ValueError(f"Question ID mismatch for {result.question_id}")

    evidence_items: list[str] = []
    for reference in answer.evidence:
        numbers = ", ".join(
            f"{_escape(item.column)} = {_escape(item.value)}"
            for item in reference.values
        )
        evidence_items.append(
            "<li>"
            f"<code>{_escape(reference.table)}[{reference.row_index}]</code>"
            f": {numbers}. {_escape(reference.explanation)}"
            "</li>"
        )

    caveat_items = "".join(f"<li>{_escape(text)}</li>" for text in answer.caveats)
    warning_items = "".join(
        f"<li>{_escape(warning.message)}</li>" for warning in result.warnings
    )
    caveats = (
        f"<h4>Limitations</h4><ul>{caveat_items}</ul>" if caveat_items else ""
    )
    warnings = (
        '<div class="caution"><strong>Scientific-review flags:</strong>'
        f"<ul>{warning_items}</ul></div>"
        if warning_items else ""
    )

    return (
        f'<aside class="ai-interpretation" data-question-id="{_escape(result.question_id)}" '
        'aria-label="Groq AI interpretation">'
        "<h3>Groq AI interpretation</h3>"
        f"<p>{_escape(answer.interpretation)}</p>"
        '<p class="rows">AI-generated. Cited numerical values passed automated '
        'verification; scientific conclusions may still require review.</p>'
        "<details><summary>Supporting statistical evidence</summary>"
        f"<ul>{''.join(evidence_items)}</ul></details>"
        f"{caveats}{warnings}</aside>"
    )


def require_complete_ai_results(
    results: Iterable[QuestionRunResult],
    expected_ids: Iterable[str],
) -> dict[str, QuestionRunResult]:
    """Reject missing, failed, duplicated, or unexpected answers before publication."""
    expected = tuple(expected_ids)
    by_id: dict[str, QuestionRunResult] = {}
    for result in results:
        if result.question_id not in expected:
            raise ValueError(f"Unexpected question: {result.question_id}")
        if result.question_id in by_id:
            raise ValueError(f"Duplicate question: {result.question_id}")
        if not result.succeeded or result.interpretation is None:
            raise ValueError(f"Groq interpretation failed: {result.question_id}")
        if result.interpretation.question_id != result.question_id:
            raise ValueError(f"Question ID mismatch: {result.question_id}")
        by_id[result.question_id] = result

    missing = set(expected) - by_id.keys()
    if missing:
        raise ValueError(f"Missing Groq interpretations: {', '.join(sorted(missing))}")
    return by_id
