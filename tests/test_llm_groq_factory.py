
"""Offline tests for Groq model configuration."""

import pytest

from da_pipeline.llm import groq_factory


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        groq_factory,
        "load_dotenv",
        lambda **kwargs: None,
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        groq_factory.create_groq_model()


def test_empty_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        groq_factory,
        "load_dotenv",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("GROQ_API_KEY", "")

    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        groq_factory.create_groq_model()


def test_model_settings_are_read_from_environment(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        groq_factory,
        "load_dotenv",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GROQ_MODEL_NAME", "test-model")
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.2")

    def fake_chat_groq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(groq_factory, "ChatGroq", fake_chat_groq)

    groq_factory.create_groq_model()

    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2
    assert captured["max_retries"] == 3


def test_default_model_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        groq_factory,
        "load_dotenv",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)
    monkeypatch.delenv("GROQ_TEMPERATURE", raising=False)

    def fake_chat_groq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(groq_factory, "ChatGroq", fake_chat_groq)

    groq_factory.create_groq_model()

    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["temperature"] == 0.0


@pytest.mark.parametrize(
    "temperature",
    ["abc", "NaN", "inf", "-inf", "-0.1"],
)
def test_invalid_temperature_is_rejected(monkeypatch, temperature):
    monkeypatch.setattr(
        groq_factory,
        "load_dotenv",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GROQ_TEMPERATURE", temperature)

    with pytest.raises(ValueError, match="GROQ_TEMPERATURE"):
        groq_factory.create_groq_model()


def _capture_chat_groq(monkeypatch):
    captured = {}
    monkeypatch.setattr(groq_factory, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def fake_chat_groq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(groq_factory, "ChatGroq", fake_chat_groq)
    return captured


def test_gpt_oss_defaults_to_low_reasoning_effort(monkeypatch):
    captured = _capture_chat_groq(monkeypatch)
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)
    monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)

    groq_factory.create_groq_model()

    assert captured["reasoning_effort"] == "low"


def test_non_reasoning_model_sends_no_reasoning_effort(monkeypatch):
    captured = _capture_chat_groq(monkeypatch)
    monkeypatch.setenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)

    groq_factory.create_groq_model()

    assert "reasoning_effort" not in captured


def test_reasoning_effort_can_be_overridden_or_omitted(monkeypatch):
    captured = _capture_chat_groq(monkeypatch)
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)

    monkeypatch.setenv("GROQ_REASONING_EFFORT", "Medium")
    groq_factory.create_groq_model()
    assert captured["reasoning_effort"] == "medium"

    captured.clear()
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "")
    groq_factory.create_groq_model()
    assert "reasoning_effort" not in captured


def test_invalid_reasoning_effort_is_rejected(monkeypatch):
    _capture_chat_groq(monkeypatch)
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "maximum")

    with pytest.raises(ValueError, match="GROQ_REASONING_EFFORT"):
        groq_factory.create_groq_model()
