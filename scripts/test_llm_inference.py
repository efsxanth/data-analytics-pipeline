"""Run one research-question interpretation against GroqCloud.

From the repository root, with the virtual environment active:
    python scripts/test_llm_inference.py 2

This sends one real request and uses the GROQ_API_KEY in your local .env file.
"""

import argparse
import json
from pathlib import Path

from da_pipeline.llm.client import generate_interpretation
from da_pipeline.llm.groq_factory import create_groq_model
from da_pipeline.llm.questions import RESEARCH_QUESTIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test one research question using GroqCloud."
    )
    parser.add_argument(
        "question_number",
        type=int,
        choices=range(1, len(RESEARCH_QUESTIONS) + 1),
        metavar="{1,2,3,4}",
        help="Research question number (1 through 4).",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    pack_path = (
        repository_root
        / "demo_data"
        / "Data_Analytics"
        / "ai"
        / "evidence_pack.json"
    )
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    question = RESEARCH_QUESTIONS[args.question_number - 1]

    print(f"Question {args.question_number}: {question.question}")
    print("Sending one request to GroqCloud...")

    try:
        result = generate_interpretation(pack, question, create_groq_model())
    except Exception as exc:
        print("\nRequest or evidence validation failed.")
        print("Error type:", type(exc).__name__)
        print("Error:", str(exc)[:2000])
        raise SystemExit(1) from None

    print("\nInterpretation:")
    print(result.interpretation)

    print("\nEvidence references:")
    for reference in result.evidence:
        print(f"\n{reference.table}[{reference.row_index}]")
        for item in reference.values:
            print(f"  {item.column}: {item.value}")
        print("  Explanation:", reference.explanation)

    print("\nCaveats:")
    if result.caveats:
        for caveat in result.caveats:
            print("-", caveat)
    else:
        print("(none supplied)")

    print("\nNumerical evidence validation: PASSED")
    print("Narrative scientific validity requires separate review.")


if __name__ == "__main__":
    main()
