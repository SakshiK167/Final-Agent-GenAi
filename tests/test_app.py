"""
Unit tests for AI Research Brief Agent (Experiment 10 MLOps)
Tests cover input validation, bullet extraction, search mocking,
LLM mocking, and comprehensive error handling without external API calls.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError

from app import (
    sanitize_topic,
    validate_topic,
    get_openrouter_config,
    search_web,
    parse_and_validate_bullets,
    generate_research_brief,
    DEFAULT_MODEL,
)


# ---------------------------------------------------------------------------
# 1. Topic Validation Tests
# ---------------------------------------------------------------------------
def test_validate_topic_valid():
    """Verify that valid topic strings are stripped and accepted."""
    topic = "  Quantum Computing Applications  "
    assert validate_topic(topic) == "Quantum Computing Applications"


def test_validate_topic_empty():
    """Verify that empty topic string raises ValueError."""
    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        validate_topic("")


def test_validate_topic_whitespace():
    """Verify that whitespace-only topic string raises ValueError."""
    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        validate_topic("   \t\n  ")


def test_validate_topic_none():
    """Verify that None input raises ValueError."""
    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        validate_topic(None)


def test_validate_topic_truncates_long():
    """Verify that overly long topic strings are safely capped."""
    long_topic = "A" * 400
    validated = validate_topic(long_topic)
    assert len(validated) == 300


def test_sanitize_topic_removes_newlines_and_truncates():
    """Verify that topic sanitization strips control characters and truncates for logging."""
    unsanitized = "Quantum Computing\n\r\tin 2025" + "x" * 100
    sanitized = sanitize_topic(unsanitized, max_len=30)
    assert "\n" not in sanitized
    assert "\r" not in sanitized
    assert "\t" not in sanitized
    assert len(sanitized) == 30


# ---------------------------------------------------------------------------
# 2. Configuration & Environment Tests
# ---------------------------------------------------------------------------
def test_get_openrouter_config_defaults(monkeypatch):
    """Verify fallback defaults when environment variables are unset."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    api_key, model = get_openrouter_config()
    assert api_key == ""
    assert model == DEFAULT_MODEL


def test_get_openrouter_config_custom(monkeypatch):
    """Verify reading custom values from environment variables."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-custom-secret-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    api_key, model = get_openrouter_config()
    assert api_key == "sk-or-v1-custom-secret-key"
    assert model == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# 3. Web Search Function Tests (Mocked)
# ---------------------------------------------------------------------------
@patch("app.DDGS")
def test_search_web_success(mock_ddgs_cls):
    """Verify successful parsing of search results into structured dicts."""
    mock_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_instance
    mock_instance.text.return_value = [
        {
            "title": "Quantum Supremacy Milestone",
            "href": "https://example.com/quantum",
            "body": "Researchers demonstrated quantum speedup in complex calculations.",
        },
        {
            "title": "Quantum Error Correction",
            "link": "https://example.com/qec",
            "snippet": "Logical qubits showed lower error rates than physical qubits.",
        },
    ]

    results = search_web("Quantum Computing", max_results=2)
    assert len(results) == 2
    assert results[0]["title"] == "Quantum Supremacy Milestone"
    assert results[0]["url"] == "https://example.com/quantum"
    assert "speedup" in results[0]["snippet"]
    assert results[1]["url"] == "https://example.com/qec"


@patch("app.DDGS")
def test_search_web_handles_empty_or_malformed(mock_ddgs_cls):
    """Verify that results without valid URL or snippet are discarded."""
    mock_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_instance
    mock_instance.text.return_value = [
        {"title": "Empty Result", "href": "", "body": ""},
        {"title": "Valid Result", "href": "https://example.com", "body": "Valid snippet content."},
    ]

    results = search_web("Test Topic", max_results=2)
    assert len(results) == 1
    assert results[0]["title"] == "Valid Result"


@patch("app.DDGS")
def test_search_web_handles_exception(mock_ddgs_cls):
    """Verify search exceptions are wrapped into RuntimeError."""
    mock_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_instance
    mock_instance.text.side_effect = Exception("Connection timed out")

    with pytest.raises(RuntimeError, match="Web search encountered an error"):
        search_web("Test Topic")


# ---------------------------------------------------------------------------
# 4. Bullet Point Parsing & Validation Tests
# ---------------------------------------------------------------------------
def test_parse_and_validate_bullets_dash_format():
    """Verify dash bullet points are parsed into exactly three items."""
    raw = (
        "- First factual finding on quantum states.\n"
        "- Second finding regarding fault-tolerant architecture.\n"
        "- Third finding covering commercialization roadmaps."
    )
    bullets = parse_and_validate_bullets(raw)
    assert len(bullets) == 3
    assert bullets[0] == "First factual finding on quantum states."
    assert bullets[1] == "Second finding regarding fault-tolerant architecture."
    assert bullets[2] == "Third finding covering commercialization roadmaps."


def test_parse_and_validate_bullets_numbered_format():
    """Verify numbered list format is cleanly stripped to 3 items."""
    raw = (
        "1. First finding regarding superconducting qubits.\n"
        "2. Second finding regarding optical quantum systems.\n"
        "3. Third finding regarding trapped-ion performance."
    )
    bullets = parse_and_validate_bullets(raw)
    assert len(bullets) == 3
    assert bullets[0] == "First finding regarding superconducting qubits."
    assert bullets[1] == "Second finding regarding optical quantum systems."
    assert bullets[2] == "Third finding regarding trapped-ion performance."


def test_parse_and_validate_bullets_excess_items():
    """Verify that more than 3 bullet points are capped at exactly 3."""
    raw = "\n".join([f"- Factual discovery item number {i} with details." for i in range(1, 7)])
    bullets = parse_and_validate_bullets(raw)
    assert len(bullets) == 3
    assert "number 1" in bullets[0]
    assert "number 3" in bullets[2]


def test_parse_and_validate_bullets_padding():
    """Verify that fewer than 3 bullet points are safely padded to guarantee 3 items."""
    raw = "- Only one primary factual finding was returned by the model."
    bullets = parse_and_validate_bullets(raw)
    assert len(bullets) == 3
    assert bullets[0] == "Only one primary factual finding was returned by the model."
    assert "No additional distinct factual finding" in bullets[1]
    assert "No additional distinct factual finding" in bullets[2]


def test_parse_and_validate_bullets_empty_raises():
    """Verify empty string raises ValueError."""
    with pytest.raises(ValueError, match="empty response"):
        parse_and_validate_bullets("")


# ---------------------------------------------------------------------------
# 5. LLM Synthesis & Error Handling Tests (Mocked)
# ---------------------------------------------------------------------------
def test_generate_brief_success():
    """Verify successful brief generation with mocked OpenAI client."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "- Quantum entanglement enables high-precision sensing applications.\n"
        "- Error mitigation techniques improved algorithmic fidelity by 30%.\n"
        "- Commercial cloud deployments have opened quantum access to universities."
    )
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    search_data = [
        {"title": "Test Title 1", "url": "https://test1.com", "snippet": "Test snippet 1."},
        {"title": "Test Title 2", "url": "https://test2.com", "snippet": "Test snippet 2."},
    ]

    bullets = generate_research_brief(
        topic="Quantum Breakthroughs",
        search_results=search_data,
        api_key="mock-key",
        model="meta-llama/llama-3.1-8b-instruct:free",
        client=mock_client,
    )

    assert len(bullets) == 3
    assert "entanglement" in bullets[0]
    assert mock_client.chat.completions.create.called


def test_generate_brief_missing_api_key():
    """Verify ValueError is raised if api_key is empty."""
    with pytest.raises(ValueError, match="OpenRouter API key is missing"):
        generate_research_brief("Topic", [{"title": "T", "url": "U", "snippet": "S"}], api_key="", model="test")


def test_generate_brief_missing_search_results():
    """Verify ValueError is raised if search_results list is empty."""
    with pytest.raises(ValueError, match="Cannot generate a brief without search results"):
        generate_research_brief("Topic", [], api_key="sk-test", model="test")


def test_generate_brief_auth_error():
    """Verify PermissionError is raised on AuthenticationError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        "Invalid API key", response=MagicMock(status_code=401), body={}
    )

    with pytest.raises(PermissionError, match="Authentication Failed"):
        generate_research_brief(
            topic="Topic",
            search_results=[{"title": "T", "url": "U", "snippet": "S"}],
            api_key="bad-key",
            model="test-model",
            client=mock_client,
        )


def test_generate_brief_rate_limit_error():
    """Verify RuntimeError is raised on RateLimitError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded", response=MagicMock(status_code=429), body={}
    )

    with pytest.raises(RuntimeError, match="Rate Limit Exceeded"):
        generate_research_brief(
            topic="Topic",
            search_results=[{"title": "T", "url": "U", "snippet": "S"}],
            api_key="key",
            model="test-model",
            client=mock_client,
        )


def test_generate_brief_timeout_error():
    """Verify TimeoutError is raised on APITimeoutError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    with pytest.raises(TimeoutError, match="Request Timeout"):
        generate_research_brief(
            topic="Topic",
            search_results=[{"title": "T", "url": "U", "snippet": "S"}],
            api_key="key",
            model="test-model",
            client=mock_client,
        )


def test_generate_brief_connection_error():
    """Verify ConnectionError is raised on APIConnectionError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    with pytest.raises(ConnectionError, match="Network Connection Error"):
        generate_research_brief(
            topic="Topic",
            search_results=[{"title": "T", "url": "U", "snippet": "S"}],
            api_key="key",
            model="test-model",
            client=mock_client,
        )


# ---------------------------------------------------------------------------
# 6. Application Startup & Integrity Check
# ---------------------------------------------------------------------------
def test_application_integrity_and_exports():
    """Verify all core functions and configurations exist for app health."""
    import app
    assert hasattr(app, "validate_topic")
    assert hasattr(app, "search_web")
    assert hasattr(app, "generate_research_brief")
    assert hasattr(app, "parse_and_validate_bullets")
    assert hasattr(app, "main")
    assert app.DEFAULT_MODEL.startswith("meta-llama/") or "free" in app.DEFAULT_MODEL
