"""Complete the demo report with four required GroqCloud interpretations.

From the repository root, after running da-pipeline-demo:
    python scripts/build_ai_report.py --check-input
    python scripts/build_ai_report.py

The check-input mode makes no network requests and does not modify files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from da_pipeline.llm.client import generate_interpretation
from da_pipeline.llm.groq_factory import create_groq_model
from da_pipeline.llm.orchestrator import interpret_all_questions
from da_pipeline.llm.prompt_builder import build_interpretation_messages
from da_pipeline.llm.questions import RESEARCH_QUESTIONS
from da_pipeline.reporting.ai_report import require_complete_ai_results
from da_pipeline.reporting.findings import Finding, RunFindings
from da_pipeline.reporting.html_report import write_report


def _read_inputs(output_dir: Path) -> tuple[dict, RunFindings]:
    """Recover the original report's findings from the generated evidence pack."""
    pack_path = output_dir / "ai" / "evidence_pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    design = pack["design"]
    findings = RunFindings(
        response_name=design["response_name"],
        design=design,
        findings=[Finding(**item) for item in pack["findings"]],
    )
    return pack, findings


def _generate(pack: dict, question, model, *, feedback=None):
    """Allow the larger Q1 evidence pack to fit the Groq TPM request budget."""
    if question.identifier == RESEARCH_QUESTIONS[0].identifier:
        question_model = model.model_copy(update={"max_tokens": 1500})
    else:
        question_model = model
    return generate_interpretation(pack, question, question_model, feedback=feedback)


def _env_number(name: str, default: str, kind=float):
    text = os.getenv(name, default)
    try:
        value = kind(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _print_prompt_sizes(pack: dict) -> None:
    """Show each question's prompt size so token budgets can be checked offline.

    Roughly three characters per token for this numeric JSON; the estimate is
    deliberately pessimistic. Groq's free tier allows 8,000 tokens per minute
    for openai/gpt-oss-20b, and that budget must hold the prompt *and* the
    requested max_tokens.
    """
    for question in RESEARCH_QUESTIONS:
        system_message, user_message = build_interpretation_messages(pack, question)
        characters = len(system_message) + len(user_message)
        print(f"  {question.identifier}: {characters:,} prompt characters (~{characters // 3:,} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_data/Data_Analytics"),
        help="Directory containing report.html and ai/evidence_pack.json",
    )
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="Check generated inputs without Groq requests or file modifications",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    pack, findings = _read_inputs(output_dir)
    if not (output_dir / "report.html").is_file():
        parser.error(f"Missing original report.html in {output_dir}")

    print(f"Evidence and deterministic findings loaded: {len(findings.findings)} findings")
    _print_prompt_sizes(pack)
    if args.check_input:
        print("Input check PASSED. No Groq request sent; no files modified.")
        return

    gap_seconds = _env_number("GROQ_REQUEST_GAP_SECONDS", "75")
    # One corrective retry by default: a wrongly copied number or table name
    # is asked again with the validator's reason, not published.
    max_attempts = max(1, _env_number("GROQ_MAX_ATTEMPTS", "2", int))

    model = create_groq_model()
    print("Requesting all four required GroqCloud interpretations...")
    print(f"Model: {model.model_name}; up to {max_attempts} attempt(s) per question.")
    print(f"Waiting {gap_seconds:g} seconds between GroqCloud requests.", flush=True)
    results = interpret_all_questions(
        pack,
        model,
        generate=_generate,
        delay_seconds=gap_seconds,
        max_attempts=max_attempts,
    )
    for result in results:
        if result.succeeded:
            status = "validated"
            if result.attempts > 1:
                status += f" (attempt {result.attempts})"
        else:
            status = f"FAILED ({result.error_type})"
            if result.attempts > 1:
                status += f" after {result.attempts} attempts"
            if result.error_detail:
                status += f": {result.error_detail}"
        print(f"{result.question_id}: {status}", flush=True)

    # Raises before report.html is touched if any result is missing or failed.
    require_complete_ai_results(
        results, (question.identifier for question in RESEARCH_QUESTIONS)
    )
    path = write_report(findings, output_dir, ai_results=results)
    print(f"Complete Groq report written: {path}")


if __name__ == "__main__":
    main()
