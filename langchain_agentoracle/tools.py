"""
AgentOracle LangChain Integration
Production-grade tools for per-claim trust verification.
All endpoints, full error handling, 402 payment support, retry logic.
"""
import time
import requests
from typing import Optional, Any, Dict, List, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

AGENTORACLE_BASE_URL = "https://agentoracle.co"
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2


def _make_request(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Core request handler with retry logic and 402 payment awareness.
    Returns structured dict — never raises on expected errors.
    """
    url = f"{AGENTORACLE_BASE_URL}{endpoint}"
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            if response.status_code == 402:
                return {
                    "success": False,
                    "error": "payment_required",
                    "message": (
                        "This endpoint requires x402 payment (USDC on Base). "
                        "Use /preview for free access (10 req/hr), or configure "
                        "an x402 wallet to access paid endpoints. "
                        "See: agentoracle.co/.well-known/x402.json"
                    ),
                    "payment_info": response.json() if response.text else {},
                }
            if response.status_code == 429:
                retry_after = int(response.headers.get("X-RateLimit-Reset", 60))
                return {
                    "success": False,
                    "error": "rate_limited",
                    "message": f"Rate limit exceeded. Resets in {retry_after} seconds.",
                    "retry_after": retry_after,
                }
            if response.status_code == 500:
                if attempt < retries - 1:
                    time.sleep(RETRY_BACKOFF ** attempt)
                    continue
                return {
                    "success": False,
                    "error": "server_error",
                    "message": "AgentOracle server error. Retry with exponential backoff.",
                }
            return {
                "success": False,
                "error": f"http_{response.status_code}",
                "message": response.text[:500],
            }
        except requests.Timeout:
            last_error = "Request timed out"
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
        except requests.ConnectionError:
            last_error = "Connection failed — check network or agentoracle.co status"
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
        except Exception as e:
            return {"success": False, "error": "unexpected", "message": str(e)}
    return {"success": False, "error": "max_retries", "message": last_error}


def _format_evaluation(data: Dict[str, Any]) -> str:
    """Format /evaluate response into readable agent output."""
    ev = data.get("evaluation", {})
    if not ev:
        return f"Evaluation error: {data}"
    lines = [
        f"EVALUATION RESULT",
        f"Overall confidence: {ev.get('overall_confidence', 0):.2f}",
        f"Recommendation: {ev.get('recommendation', 'unknown').upper()}",
        f"Claims found: {ev.get('total_claims', 0)} | "
        f"Supported: {ev.get('verified_claims', 0)} | "
        f"Refuted: {ev.get('refuted_claims', 0)} | "
        f"Unverifiable: {ev.get('unverifiable_claims', 0)}",
        f"Sources used: {', '.join(ev.get('sources_used', []))}",
        f"Evaluation time: {data.get('meta', {}).get('evaluation_time_ms', 0)}ms | "
        f"Cost: {data.get('meta', {}).get('price', '$0.01 USDC')}",
        "",
        "CLAIMS:",
    ]
    for claim in ev.get("claims", []):
        verdict = claim.get("verdict", "unknown").upper()
        confidence = claim.get("confidence", 0)
        text = claim.get("claim", "")
        evidence = claim.get("evidence", "")
        correction = claim.get("correction", "")
        adversarial = claim.get("adversarial_result", "")
        symbol = {"SUPPORTED": "✓", "REFUTED": "✗", "UNVERIFIABLE": "?"}.get(
            verdict, "?"
        )
        lines.append(f"  {symbol} [{verdict}] ({confidence:.2f}) {text}")
        if evidence:
            lines.append(f"    Evidence: {evidence[:200]}")
        if correction:
            lines.append(f"    Correction: {correction}")
        if adversarial:
            lines.append(f"    Adversarial: {adversarial}")
    gemma = data.get("gemma_calibration", {})
    if gemma:
        lines.append(
            f"\nGemma calibration: {gemma.get('calibrated_confidence', 0):.2f} "
            f"({gemma.get('agreement', 'unknown')} agreement)"
        )
    lines.append(f"\nEvaluation ID: {data.get('evaluation_id', 'unknown')}")
    return "\n".join(lines)


def _format_research(data: Dict[str, Any]) -> str:
    """Format /research or /deep-research response."""
    lines = [
        f"RESEARCH RESULT",
        f"Summary: {data.get('summary', 'No summary')}",
        "",
    ]
    facts = data.get("key_facts", [])
    if facts:
        lines.append("Key facts:")
        for fact in facts:
            lines.append(f"  • {fact}")
        lines.append("")
    sources = data.get("sources", [])
    if sources:
        lines.append("Sources:")
        for s in sources[:5]:
            url = s.get("url", s) if isinstance(s, dict) else s
            lines.append(f"  • {url}")
        lines.append("")
    confidence = data.get("confidence_score", data.get("confidence", {}).get("score") if isinstance(data.get("confidence"), dict) else None)
    if confidence is not None:
        lines.append(f"Confidence: {confidence:.2f}")
    meta = data.get("query_metadata", {})
    if meta:
        lines.append(
            f"Model: {meta.get('model', 'unknown')} | "
            f"Latency: {meta.get('latency_ms', 0)}ms | "
            f"Cost: ${meta.get('cost_usd', 0.02):.2f}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# INPUT SCHEMAS
# ─────────────────────────────────────────────
class EvaluateInput(BaseModel):
    content: str = Field(
        description=(
            "Text containing claims to verify. Can be raw text, a research result, "
            "news article excerpt, or any data your agent retrieved. "
            "AgentOracle will decompose it into individual claims and verify each one."
        )
    )
    source: Optional[str] = Field(
        default="langchain",
        description="Source identifier (e.g. 'exa', 'perplexity', 'web', 'langchain').",
    )
    min_confidence: Optional[float] = Field(
        default=0.8,
        description="Minimum confidence threshold (0.0-1.0). Claims below this flag as unverifiable.",
    )


class PreviewInput(BaseModel):
    query: str = Field(
        description="Research query to run against the free preview endpoint (10 req/hr, truncated results).",
    )


class ResearchInput(BaseModel):
    query: str = Field(description="Research query for full web research with sources and confidence score.")
    tier: Optional[str] = Field(
        default="standard",
        description="Research tier: 'standard' ($0.02) or 'deep' for Sonar Pro ($0.10).",
    )


class DeepResearchInput(BaseModel):
    query: str = Field(
        description="Complex query requiring multi-step deep analysis via Sonar Pro.",
    )


class BatchResearchInput(BaseModel):
    queries: List[str] = Field(
        description="List of research queries to run in batch. Each query costs $0.02 USDC.",
    )


class VerifyGateInput(BaseModel):
    content: str = Field(
        description=(
            "Text to run through the free pass/fail verification gate. "
            "Returns a simple boolean trust decision — no per-claim breakdown. "
            "Use /evaluate for full per-claim analysis."
        )
    )
    threshold: Optional[float] = Field(
        default=0.8,
        description="Confidence threshold for pass/fail decision.",
    )


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
class AgentOracleEvaluateTool(BaseTool):
    """
    Full 4-source claim verification via AgentOracle /evaluate.
    $0.01 USDC per evaluation via x402 on Base.
    Returns per-claim verdicts with confidence scores and ACT/VERIFY/REJECT recommendation.
    """

    name: str = "agentoracle_evaluate"
    description: str = (
        "Verify claims in any text before your agent acts on them. "
        "Submits content to AgentOracle's 4-source verification pipeline "
        "(Sonar + Sonar Pro + Adversarial + Gemma 4). "
        "Returns per-claim verdicts (supported/refuted/unverifiable), "
        "confidence scores (0.00-1.00), and a top-level recommendation: "
        "ACT (>0.8), VERIFY (0.5-0.8), or REJECT (<0.5). "
        "Cost: $0.01 USDC via x402. Use this before acting on retrieved data."
    )
    args_schema: Type[BaseModel] = EvaluateInput

    def _run(
        self,
        content: str,
        source: str = "langchain",
        min_confidence: float = 0.8,
    ) -> str:
        result = _make_request(
            "/evaluate",
            {
                "content": content,
                "source": source,
                "min_confidence": min_confidence,
            },
        )
        if not result["success"]:
            return (
                f"AgentOracle evaluation failed: {result.get('message', 'Unknown error')}\n"
                f"Error type: {result.get('error', 'unknown')}\n"
                f"Tip: Use agentoracle_preview for free verification (10 req/hr)."
            )
        return _format_evaluation(result["data"])

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOraclePreviewTool(BaseTool):
    """
    Free preview research via AgentOracle /preview.
    10 requests/hour. No payment required. Returns truncated results.
    """

    name: str = "agentoracle_preview"
    description: str = (
        "Free research preview via AgentOracle. "
        "No payment required — 10 requests per hour. "
        "Returns truncated research results with confidence score. "
        "Use this to test queries or when x402 payment is not configured. "
        "For full results use agentoracle_research ($0.02/query)."
    )
    args_schema: Type[BaseModel] = PreviewInput

    def _run(self, query: str) -> str:
        result = _make_request("/preview", {"query": query})
        if not result["success"]:
            return f"Preview failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        return (
            f"PREVIEW RESULT (truncated)\n"
            f"Summary: {data.get('summary', data.get('result', 'No summary'))}\n"
            f"Confidence: {data.get('confidence_score', 'N/A')}\n"
            f"Note: Use agentoracle_research for full results."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleResearchTool(BaseTool):
    """
    Full real-time research via AgentOracle /research.
    $0.02 USDC per query via x402 on Base.
    Returns structured JSON with sources, confidence, key facts.
    """

    name: str = "agentoracle_research"
    description: str = (
        "Real-time web research via AgentOracle. "
        "Returns structured results with sources, key facts, and confidence score. "
        "Powered by Perplexity Sonar. Cost: $0.02 USDC via x402. "
        "For deeper analysis use agentoracle_deep_research ($0.10/query). "
        "For claim verification use agentoracle_evaluate ($0.01/claim)."
    )
    args_schema: Type[BaseModel] = ResearchInput

    def _run(self, query: str, tier: str = "standard") -> str:
        endpoint = "/research"
        payload: Dict[str, Any] = {"query": query}
        if tier == "deep":
            payload["tier"] = "deep"
        result = _make_request(endpoint, payload)
        if not result["success"]:
            return f"Research failed: {result.get('message', 'Unknown error')}"
        return _format_research(result["data"])

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleDeepResearchTool(BaseTool):
    """
    Deep multi-step research via AgentOracle /deep-research.
    $0.10 USDC per query via x402. Uses Sonar Pro for comprehensive analysis.
    """

    name: str = "agentoracle_deep_research"
    description: str = (
        "Deep multi-step research via AgentOracle using Sonar Pro. "
        "Best for complex questions requiring comprehensive source verification. "
        "Returns extended analysis with higher confidence scoring. "
        "Cost: $0.10 USDC via x402. "
        "Use for due diligence, market research, or any query needing depth."
    )
    args_schema: Type[BaseModel] = DeepResearchInput

    def _run(self, query: str) -> str:
        result = _make_request("/deep-research", {"query": query}, timeout=90)
        if not result["success"]:
            return f"Deep research failed: {result.get('message', 'Unknown error')}"
        return _format_research(result["data"])

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleBatchResearchTool(BaseTool):
    """
    Batch research via AgentOracle /research/batch.
    $0.02 USDC per query. Run multiple queries in one call.
    """

    name: str = "agentoracle_batch_research"
    description: str = (
        "Run multiple research queries in a single batch call. "
        "Cost: $0.02 USDC per query via x402. "
        "More efficient than individual calls for 3+ queries. "
        "Returns structured results for each query."
    )
    args_schema: Type[BaseModel] = BatchResearchInput

    def _run(self, queries: List[str]) -> str:
        if not queries:
            return "No queries provided."
        if len(queries) > 10:
            return "Maximum 10 queries per batch. Split into smaller batches."
        result = _make_request(
            "/research/batch",
            {"queries": queries},
            timeout=120,
        )
        if not result["success"]:
            return f"Batch research failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        results = data.get("results", [])
        if not results:
            return f"Batch returned no results: {data}"
        lines = [f"BATCH RESEARCH RESULTS ({len(results)} queries)\n"]
        for i, r in enumerate(results):
            lines.append(f"Query {i+1}: {queries[i] if i < len(queries) else 'unknown'}")
            lines.append(_format_research(r))
            lines.append("")
        total_cost = len(queries) * 0.02
        lines.append(f"Total cost: ${total_cost:.2f} USDC")
        return "\n".join(lines)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleVerifyGateTool(BaseTool):
    """
    Free pass/fail verification gate via AgentOracle /verify-gate.
    No payment required. Returns simple boolean trust decision.
    """

    name: str = "agentoracle_verify_gate"
    description: str = (
        "Free pass/fail verification gate. "
        "Quickly determine if content meets a confidence threshold. "
        "Returns PASS or FAIL — no per-claim breakdown. "
        "No payment required. "
        "Use agentoracle_evaluate for full per-claim analysis."
    )
    args_schema: Type[BaseModel] = VerifyGateInput

    def _run(self, content: str, threshold: float = 0.8) -> str:
        result = _make_request(
            "/verify-gate",
            {"content": content, "threshold": threshold},
        )
        if not result["success"]:
            return f"Verify gate failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        passed = data.get("passed", data.get("verified", False))
        confidence = data.get("confidence", data.get("score", 0))
        recommendation = data.get("recommendation", "verify")
        return (
            f"VERIFY GATE: {'PASS' if passed else 'FAIL'}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Recommendation: {recommendation.upper()}\n"
            f"Threshold applied: {threshold}\n"
            f"Use agentoracle_evaluate for per-claim breakdown."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


# ─────────────────────────────────────────────
# CONVENIENCE — get all tools at once
# ─────────────────────────────────────────────
def get_agentoracle_tools(
    include_paid: bool = True,
    include_free: bool = True,
) -> List[BaseTool]:
    """
    Return AgentOracle tools for use with any LangChain agent.

    Args:
        include_paid: Include tools that require x402 payment (evaluate, research, deep_research, batch)
        include_free: Include free tools (preview, verify_gate)

    Returns:
        List of LangChain BaseTool instances

    Example:
        from langchain_agentoracle import get_agentoracle_tools
        from langchain.agents import initialize_agent, AgentType
        from langchain_openai import ChatOpenAI

        tools = get_agentoracle_tools()
        agent = initialize_agent(tools, ChatOpenAI(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        agent.run("Research AI agent frameworks and verify the claims before reporting.")
    """
    tools = []
    if include_free:
        tools.extend([
            AgentOraclePreviewTool(),
            AgentOracleVerifyGateTool(),
        ])
    if include_paid:
        tools.extend([
            AgentOracleEvaluateTool(),
            AgentOracleResearchTool(),
            AgentOracleDeepResearchTool(),
            AgentOracleBatchResearchTool(),
        ])
    return tools


# Legacy alias — keeps backward compat with 0.1.0
AgentOracleTool = AgentOracleEvaluateTool
