"""
LLM-as-a-Judge evaluation framework for the Sovereign Venture Engine.

This module provides a quality gate that evaluates agent output using
a separate LLM (the "Judge"). Think of it as unit tests for your AI
agents — they run before merging code changes and catch regressions
in output quality.

Evaluation criteria:
- Hallucination: did the agent fabricate facts?
- Tone: is the output professional and on-brand?
- Relevance: did it use the prospect's actual data (tech stack, etc.)?
- Compliance: does it comply with CAN-SPAM / GDPR rules?

Usage:
    from tests.evals.base_eval import BaseAgentEval

    class OutreachEmailEval(BaseAgentEval):
        eval_name = "outreach_email"
        agent_type = "outreach"

        def get_test_scenarios(self):
            return [
                {
                    "input": {"lead": {...}},
                    "criteria": ["professional_tone", "mentions_tech_stack"],
                }
            ]

Run evals:
    pytest tests/evals/ -v
"""
