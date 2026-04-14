"""AgentOracle tools for LangChain agents."""

from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import requests


class VerifyInput(BaseModel):
    """Input for the AgentOracle verify tool."""
    claim: str = Field(description="The claim or statement to verify for accuracy")


class EvaluateInput(BaseModel):
    """Input for the AgentOracle evaluate tool."""
    content: str = Field(description="Content containing one or more claims to evaluate")
    source: str = Field(default="langchain-agent", description="Identifier for the calling agent")


class PreviewInput(BaseModel):
    """Input for the AgentOracle preview tool."""
    query: str = Field(description="Research query to preview")


class AgentOracleVerifyTool(BaseTool):
    """Tool that verifies whether a claim is true using AgentOracle's
    multi-source verification engine.

    AgentOracle decomposes claims, cross-references them against multiple
    independent sources (including adversarial checking), and returns
    per-claim confidence scores with a recommendation to act, verify, or reject.

    This tool calls the free /preview endpoint. For full paid verification,
    use AgentOracleEvaluateTool.
    """

    name: str = "agentoracle_verify"
    description: str = (
        "Verify whether a claim or statement is true. Returns a confidence score "
        "(0.00-1.00) and recommendation (act/verify/reject). Use this before acting "
        "on any unverified data, especially data from paid APIs or external sources."
    )
    args_schema: Type[BaseModel] = VerifyInput
    base_url: str = "https://agentoracle.co"

    def _run(
        self,
        claim: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Verify a claim using AgentOracle."""
        try:
            response = requests.post(
                f"{self.base_url}/preview",
                json={"query": claim, "source": "langchain-agent"},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return (
                    f"Confidence: {data.get('confidence', 'N/A')}\n"
                    f"Summary: {data.get('summary', 'N/A')}\n"
                    f"Sources: {data.get('source_count', 'N/A')}"
                )
            return f"Verification returned status {response.status_code}"
        except Exception as e:
            return f"Verification failed: {str(e)}"


class AgentOracleEvaluateTool(BaseTool):
    """Tool that performs full multi-source claim evaluation via AgentOracle.

    Uses 4-source verification (Sonar + Sonar Pro + Adversarial + Gemma 4)
    with claim decomposition and confidence calibration.

    Note: This endpoint requires x402 payment ($0.02 USDC on Base).
    For free preview, use AgentOracleVerifyTool.
    """

    name: str = "agentoracle_evaluate"
    description: str = (
        "Perform full multi-source verification on content containing claims. "
        "Decomposes content into individual claims, verifies each against 4 independent "
        "sources, and returns per-claim verdicts (supported/refuted/unverifiable) with "
        "confidence scores. Costs $0.02 USDC via x402. Use for high-stakes decisions."
    )
    args_schema: Type[BaseModel] = EvaluateInput
    base_url: str = "https://agentoracle.co"

    def _run(
        self,
        content: str,
        source: str = "langchain-agent",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Evaluate claims using AgentOracle's full verification engine."""
        try:
            response = requests.post(
                f"{self.base_url}/evaluate",
                json={"content": content, "source": source},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get("evaluation", {})
                claims = evaluation.get("claims", [])
                result_lines = [
                    f"Overall confidence: {evaluation.get('overall_confidence', 'N/A')}",
                    f"Recommendation: {evaluation.get('recommendation', 'N/A')}",
                    f"Total claims: {evaluation.get('total_claims', 0)}",
                    f"Verified: {evaluation.get('verified_claims', 0)}",
                    f"Refuted: {evaluation.get('refuted_claims', 0)}",
                    "",
                ]
                for claim in claims:
                    result_lines.append(
                        f"- [{claim.get('verdict', '?')}] "
                        f"(conf: {claim.get('confidence', '?')}) "
                        f"{claim.get('claim', '?')[:100]}"
                    )
                return "\n".join(result_lines)
            elif response.status_code == 402:
                return "Payment required: This endpoint costs $0.02 USDC via x402 on Base."
            return f"Evaluation returned status {response.status_code}"
        except Exception as e:
            return f"Evaluation failed: {str(e)}"


class AgentOraclePreviewTool(BaseTool):
    """Tool that performs free research preview via AgentOracle.

    Returns a truncated summary with confidence score and source count.
    Rate limited to 10/hour per IP. No payment required.
    """

    name: str = "agentoracle_preview"
    description: str = (
        "Get a free research preview on any topic. Returns a truncated summary "
        "with confidence score and source count. Use this for quick fact-checks "
        "before committing to a paid evaluation. Rate limited to 10/hour."
    )
    args_schema: Type[BaseModel] = PreviewInput
    base_url: str = "https://agentoracle.co"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a research preview from AgentOracle."""
        try:
            response = requests.post(
                f"{self.base_url}/preview",
                json={"query": query, "source": "langchain-agent"},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return (
                    f"Summary: {data.get('summary', 'N/A')}\n"
                    f"Confidence: {data.get('confidence', 'N/A')}\n"
                    f"Sources: {data.get('source_count', 'N/A')}\n"
                    f"Note: This is a truncated preview. Use agentoracle_evaluate for full verification."
                )
            return f"Preview returned status {response.status_code}"
        except Exception as e:
            return f"Preview failed: {str(e)}"
