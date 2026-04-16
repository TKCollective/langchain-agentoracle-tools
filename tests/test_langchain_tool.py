"""
Unit tests for AgentOracle LangChain SDK tools.
18 tests covering all tools, error handling, retry logic, and convenience helpers.
No real HTTP calls — all network activity is mocked with unittest.mock.
"""
import pytest
from unittest.mock import patch, MagicMock, call
import requests

from langchain_agentoracle.tools import (
    AgentOracleEvaluateTool,
    AgentOraclePreviewTool,
    AgentOracleResearchTool,
    AgentOracleDeepResearchTool,
    AgentOracleBatchResearchTool,
    AgentOracleVerifyGateTool,
    AgentOracleTool,
    get_agentoracle_tools,
)


# ─────────────────────────────────────────────
# PREVIEW TOOL
# ─────────────────────────────────────────────

def test_preview_returns_summary(mock_preview_response):
    """Mock /preview 200 — result must contain 'PREVIEW'."""
    tool = AgentOraclePreviewTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_preview_response

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = tool._run(query="What are AI agent frameworks?")

    assert "PREVIEW" in result
    mock_post.assert_called_once()


def test_preview_rate_limit_handling():
    """Mock /preview 429 — result must contain a rate limit message."""
    tool = AgentOraclePreviewTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"X-RateLimit-Reset": "60"}

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(query="test query")

    assert "rate" in result.lower() or "Rate" in result or "limit" in result.lower()


# ─────────────────────────────────────────────
# EVALUATE TOOL
# ─────────────────────────────────────────────

def test_evaluate_402_returns_helpful_message():
    """Mock /evaluate 402 — result must contain payment/x402 guidance."""
    tool = AgentOracleEvaluateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.text = '{"requires":"payment"}'
    mock_resp.json.return_value = {"requires": "payment"}

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Some content to evaluate.")

    assert "payment" in result.lower() or "x402" in result.lower() or "402" in result


def test_evaluate_response_parsing(mock_evaluate_response):
    """Mock /evaluate 200 — assert EVALUATION RESULT, confidence 0.66, VERIFY, sonar."""
    tool = AgentOracleEvaluateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_evaluate_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="LangGraph leads AI agent frameworks in 2026.")

    assert "EVALUATION RESULT" in result
    assert "0.66" in result
    assert "VERIFY" in result
    assert "sonar" in result


def test_evaluate_parses_all_claims(mock_evaluate_response):
    """Mock /evaluate 200 — all 3 claim texts must appear in output."""
    tool = AgentOracleEvaluateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_evaluate_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Test content with multiple claims.")

    assert "LangGraph leads AI agent frameworks in 2026." in result
    assert "CrewAI was acquired by Google." in result
    assert "Bitcoin was created by Satoshi Nakamoto." in result


# ─────────────────────────────────────────────
# VERIFY GATE TOOL
# ─────────────────────────────────────────────

def test_verify_gate_pass(mock_verify_gate_pass):
    """Mock /verify-gate 200 passed=True — assert PASS and confidence 0.92."""
    tool = AgentOracleVerifyGateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_verify_gate_pass

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Bitcoin was invented by Satoshi Nakamoto.")

    assert "PASS" in result
    assert "0.92" in result


def test_verify_gate_fail(mock_verify_gate_fail):
    """Mock /verify-gate 200 passed=False — assert FAIL in output."""
    tool = AgentOracleVerifyGateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_verify_gate_fail

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="The moon is made of cheese.")

    assert "FAIL" in result


# ─────────────────────────────────────────────
# RESEARCH TOOL
# ─────────────────────────────────────────────

def test_research_formats_output(mock_research_response):
    """Mock /research 200 — assert RESEARCH RESULT, summary text, and sources present."""
    tool = AgentOracleResearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_research_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(query="AI agent frameworks 2026")

    assert "RESEARCH RESULT" in result
    assert "AI agent frameworks are rapidly evolving" in result
    assert "example.com" in result


# ─────────────────────────────────────────────
# DEEP RESEARCH TOOL
# ─────────────────────────────────────────────

def test_deep_research_uses_longer_timeout(mock_research_response):
    """Mock /deep-research 200 — assert requests.post was called with timeout >= 90."""
    tool = AgentOracleDeepResearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_research_response

    with patch("requests.post", return_value=mock_resp) as mock_post:
        tool._run(query="Deep analysis of AI infrastructure")

    _, kwargs = mock_post.call_args
    assert kwargs.get("timeout", 0) >= 90


# ─────────────────────────────────────────────
# BATCH RESEARCH TOOL
# ─────────────────────────────────────────────

def test_batch_research_rejects_over_10():
    """Passing 11 queries must return 'Maximum 10' message without making any HTTP call."""
    tool = AgentOracleBatchResearchTool()
    queries = [f"query {i}" for i in range(11)]

    with patch("requests.post") as mock_post:
        result = tool._run(queries=queries)

    assert "Maximum 10" in result
    mock_post.assert_not_called()


def test_batch_research_formats_all_results(mock_batch_response):
    """Mock /research/batch 200 with 2 results — assert BATCH header and $0.04 cost."""
    tool = AgentOracleBatchResearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_batch_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(queries=["query 1", "query 2"])

    assert "BATCH" in result
    assert "$0.04" in result


# ─────────────────────────────────────────────
# RETRY + ERROR HANDLING
# ─────────────────────────────────────────────

def test_retry_on_500():
    """Mock 500 twice then 200 — requests.post must be called exactly 3 times."""
    tool = AgentOraclePreviewTool()

    resp_500 = MagicMock()
    resp_500.status_code = 500

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "summary": "Recovered result",
        "confidence_score": 0.8,
    }

    with patch("requests.post", side_effect=[resp_500, resp_500, resp_200]) as mock_post:
        with patch("time.sleep"):  # suppress real sleep
            result = tool._run(query="retry test")

    assert mock_post.call_count == 3
    assert "PREVIEW" in result


def test_connection_error_handled():
    """Mock ConnectionError — tool must not raise and must return a string."""
    tool = AgentOraclePreviewTool()

    with patch("requests.post", side_effect=requests.ConnectionError("network down")):
        with patch("time.sleep"):
            result = tool._run(query="connection error test")

    assert isinstance(result, str)
    assert len(result) > 0


def test_timeout_handled():
    """Mock Timeout — tool must not raise and must return a string."""
    tool = AgentOraclePreviewTool()

    with patch("requests.post", side_effect=requests.Timeout("timed out")):
        with patch("time.sleep"):
            result = tool._run(query="timeout test")

    assert isinstance(result, str)
    assert len(result) > 0


# ─────────────────────────────────────────────
# CONVENIENCE HELPER
# ─────────────────────────────────────────────

def test_get_tools_all():
    """get_agentoracle_tools() with defaults must return exactly 6 tools."""
    tools = get_agentoracle_tools()
    assert len(tools) == 6


def test_get_tools_free_only():
    """get_agentoracle_tools(include_paid=False) must return exactly 2 free tools."""
    tools = get_agentoracle_tools(include_paid=False, include_free=True)
    assert len(tools) == 2


def test_get_tools_paid_only():
    """get_agentoracle_tools(include_free=False) must return exactly 4 paid tools."""
    tools = get_agentoracle_tools(include_paid=True, include_free=False)
    assert len(tools) == 4


# ─────────────────────────────────────────────
# LEGACY ALIAS
# ─────────────────────────────────────────────

def test_legacy_alias():
    """AgentOracleTool must be the same class as AgentOracleEvaluateTool."""
    assert AgentOracleTool is AgentOracleEvaluateTool
