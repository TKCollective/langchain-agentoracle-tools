# langchain-agentoracle

LangChain integration for [AgentOracle](https://agentoracle.co) — trust verification tools for AI agents.

## Installation

```bash
pip install langchain-agentoracle
```

## Feedback & Support

Built something with AgentOracle? Hit an issue? We want to hear from you.

- **GitHub Discussions** — questions, ideas, show and tell: [github.com/TKCollective/x402-research-skill/discussions](https://github.com/TKCollective/x402-research-skill/discussions)
- **X / Twitter** — [@AgentOracle_AI](https://x.com/AgentOracle_AI)
- **Issues** — bugs and feature requests: open an issue in this repo

If you're evaluating AgentOracle for a project, drop a note in Discussions — we respond fast and can help with integration.

## Quick Start

```python
from langchain_agentoracle import AgentOracleVerifyTool, AgentOracleEvaluateTool

# Free preview verification
verify = AgentOracleVerifyTool()
result = verify.invoke("GPT-5 was released in March 2026")

# Full multi-source evaluation ($0.02 USDC via x402)
evaluate = AgentOracleEvaluateTool()
result = evaluate.invoke({
    "content": "Bitcoin reached $100K in 2025. Ethereum moved to proof-of-stake in 2022."
})
```

## Use with LangChain Agents

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_agentoracle import AgentOracleVerifyTool, AgentOraclePreviewTool

llm = ChatOpenAI(model="gpt-4")
tools = [AgentOracleVerifyTool(), AgentOraclePreviewTool()]

# Agent can now verify claims before acting on them
```

## Tools

| Tool | Description | Price |
|------|-------------|-------|
| `AgentOracleVerifyTool` | Quick claim verification via /preview | Free (10/hr) |
| `AgentOracleEvaluateTool` | Full 4-source verification with per-claim analysis | $0.02 USDC |
| `AgentOraclePreviewTool` | Research preview with confidence scoring | Free (10/hr) |

## How It Works

AgentOracle uses 4-source verification:
1. **Sonar** — Primary research
2. **Sonar Pro** — Deep research
3. **Adversarial** — Actively tries to disprove claims
4. **Gemma 4** — Claim decomposition and confidence calibration

Returns per-claim verdicts (supported/refuted/unverifiable) with confidence scores (0.00-1.00).

## Links

- **API**: [agentoracle.co](https://agentoracle.co)
- **npm SDK**: [agentoracle-verify](https://www.npmjs.com/package/agentoracle-verify)
- **GitHub**: [TKCollective/x402-research-skill](https://github.com/TKCollective/x402-research-skill)

## License

MIT
