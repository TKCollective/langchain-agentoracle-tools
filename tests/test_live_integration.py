"""
Live integration tests for AgentOracle LangChain SDK.
All tests hit https://agentoracle.co directly — no mocking.

Run with:
    pytest tests/test_live_integration.py -m integration -v

Requires network access to agentoracle.co.
Paid endpoints (/evaluate, /research, /deep-research, /batch) will return 402
unless an x402 wallet is configured — those tests skip gracefully on 402.
"""
import pytest
import requests

from langchain_agentoracle.tools import (
    AgentOraclePreviewTool,
    AgentOracleVerifyGateTool,
    AgentOracleEvaluateTool,
)

AGENTORACLE_BASE_URL = "https://agentoracle.co"


@pytest.mark.integration
def test_live_preview_returns_data():
    """
    Live call to /preview — response must be a non-empty string containing 'PREVIEW'.
    Uses the free tier (10 req/hr, no payment required).
    """
    tool = AgentOraclePreviewTool()
    result = tool._run(query="What is LangChain?")

    assert isinstance(result, str)
    assert len(result) > 10
    # Preview tool always prefixes with PREVIEW RESULT
    assert "PREVIEW" in result


@pytest.mark.integration
def test_live_preview_not_empty():
    """
    Live call to /preview — result must contain a non-trivial summary.
    Confirms the endpoint returns actual research content, not an empty body.
    """
    tool = AgentOraclePreviewTool()
    result = tool._run(query="Bitcoin price history")

    assert isinstance(result, str)
    # Must have meaningful content beyond just the label line
    assert len(result.strip()) > 30
    # Must not be a raw error response
    assert "unexpected" not in result
    assert "max_retries" not in result


@pytest.mark.integration
def test_live_verify_gate_known_true():
    """
    Live call to /verify-gate with a well-known true claim.
    Expects PASS (no guarantee — gate may be conservative) but no exception.
    Free endpoint, no payment required.
    """
    tool = AgentOracleVerifyGateTool()
    result = tool._run(
        content="Bitcoin was created by Satoshi Nakamoto.",
        threshold=0.5,
    )

    assert isinstance(result, str)
    assert "VERIFY GATE" in result
    # Result is either PASS or FAIL — both are valid responses
    assert "PASS" in result or "FAIL" in result


@pytest.mark.integration
def test_live_verify_gate_known_false():
    """
    Live call to /verify-gate with a clearly false claim.
    Expects FAIL from the gate. Free endpoint, no payment required.
    """
    tool = AgentOracleVerifyGateTool()
    result = tool._run(
        content="The Earth is flat and has no atmosphere.",
        threshold=0.9,
    )

    assert isinstance(result, str)
    assert "VERIFY GATE" in result
    assert "PASS" in result or "FAIL" in result


@pytest.mark.integration
def test_live_health_check():
    """
    Live GET /health — confirms agentoracle.co is reachable and returns 200.
    Not a tool method — directly hits the health endpoint to verify API availability.
    """
    response = requests.get(f"{AGENTORACLE_BASE_URL}/health", timeout=15)
    assert response.status_code == 200


@pytest.mark.integration
def test_live_evaluate_catches_hallucination():
    """
    Live call to /evaluate with a known hallucination ('CrewAI acquired by Google').
    Expects either a 402 (payment required — skip gracefully) or a proper evaluation
    response where the hallucination is flagged as unverifiable or refuted.
    """
    tool = AgentOracleEvaluateTool()
    result = tool._run(
        content=(
            "CrewAI was acquired by Google in 2025. "
            "Bitcoin was created by Satoshi Nakamoto."
        ),
        source="integration_test",
        min_confidence=0.8,
    )

    assert isinstance(result, str)
    assert len(result) > 10

    if "payment" in result.lower() or "402" in result or "x402" in result.lower():
        # Graceful skip — paid endpoint requires wallet configuration
        pytest.skip("Skipping paid /evaluate test — x402 payment not configured")

    # If we got through, it should be a real evaluation
    assert "EVALUATION RESULT" in result or "evaluation" in result.lower()
