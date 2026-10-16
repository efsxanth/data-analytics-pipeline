
"""Construct the Groq model used for statistical interpretations."""

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Values Groq accepts for its reasoning models (gpt-oss: low/medium/high;
# Qwen: none/default).
REASONING_EFFORTS = ("low", "medium", "high", "none", "default")


def create_groq_model() -> ChatGroq:
    """Create a Groq chat model using local environment configuration."""

    repository_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=repository_root / ".env")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Set it in your local .env file."
        )

    model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")


    temperature_text = os.getenv("GROQ_TEMPERATURE", "0.0")

    try:
        temperature = float(temperature_text)
    except ValueError as exc:
        raise ValueError(
            "GROQ_TEMPERATURE must be a valid number."
        ) from exc

    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(
            "GROQ_TEMPERATURE must be a finite, non-negative number."
        )


    max_tokens_text = os.getenv("GROQ_MAX_TOKENS", "2048")
    max_retries_text = os.getenv("GROQ_MAX_RETRIES", "3")
    try:
        max_tokens = int(max_tokens_text)
        max_retries = int(max_retries_text)
    except ValueError as exc:
        raise ValueError(
            "GROQ_MAX_TOKENS and GROQ_MAX_RETRIES must be integers."
        ) from exc
    if max_tokens < 1 or max_retries < 0:
        raise ValueError(
            "GROQ_MAX_TOKENS must be positive and GROQ_MAX_RETRIES non-negative."
        )

    # gpt-oss is a reasoning model, and its hidden reasoning is drawn from the
    # same max_tokens budget as the JSON answer. At Groq's default (medium) it
    # can use most of a 1,500-2,048 token budget before the answer is complete,
    # and a truncated structured output comes back as a 400. "low" leaves room
    # for the answer and keeps each request inside the free tier's 8K TPM.
    # Set GROQ_REASONING_EFFORT to an empty string to send no value at all.
    default_effort = "low" if model_name.startswith("openai/gpt-oss") else ""
    reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", default_effort).strip().lower()
    if reasoning_effort and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            "GROQ_REASONING_EFFORT must be one of: "
            + ", ".join(REASONING_EFFORTS)
            + " (or empty to omit it)."
        )

    options = {}
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        **options,
    )
