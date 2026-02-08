"""
Base class for LLM-as-a-Judge evaluations.

Provides a standard interface for evaluating agent outputs against
quality criteria. The Judge is a separate, lightweight LLM that
scores outputs on a 0-100 scale.

Design principles:
- Deterministic when possible (temperature=0 for the judge)
- Structured output (JSON scores, not free-text)
- Composable criteria (mix and match per agent type)
- Offline-capable (can run with mock judge for CI/CD)

The evaluation framework supports two modes:
1. Live mode: uses a real LLM (Anthropic/OpenAI) to judge outputs
2. Mock mode: uses a stub judge for fast CI tests (ENCLAVE_EVAL_MOCK=1)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Evaluation Criteria ────────────────────────────────────────────


@dataclass
class EvalCriterion:
    """
    A single quality criterion for the judge to evaluate.

    Args:
        name: Short identifier (e.g., "professional_tone").
        description: What the judge should look for.
        weight: Relative importance (0.0-1.0). Weights across
                all criteria for a scenario should sum to 1.0.
        threshold: Minimum acceptable score (0-100) for this criterion.
    """

    name: str
    description: str
    weight: float = 1.0
    threshold: int = 60


# Built-in criteria library (agents can use these or define custom ones)
BUILTIN_CRITERIA = {
    "hallucination": EvalCriterion(
        name="hallucination",
        description=(
            "Score 0-100 on factual accuracy. 100 = all facts are "
            "verifiable from the input data, 0 = entirely fabricated. "
            "Deduct points for any claims not supported by the input."
        ),
        weight=0.3,
        threshold=70,
    ),
    "professional_tone": EvalCriterion(
        name="professional_tone",
        description=(
            "Score 0-100 on professionalism. 100 = polished, "
            "business-appropriate language. 0 = casual, sloppy, or "
            "inappropriate tone. Consider formality, clarity, grammar."
        ),
        weight=0.25,
        threshold=70,
    ),
    "relevance": EvalCriterion(
        name="relevance",
        description=(
            "Score 0-100 on relevance to the prospect. 100 = deeply "
            "personalized, mentions specific company data. 0 = generic, "
            "could be sent to anyone. Check for tech stack, industry, "
            "company name, and role-specific references."
        ),
        weight=0.25,
        threshold=60,
    ),
    "compliance": EvalCriterion(
        name="compliance",
        description=(
            "Score 0-100 on regulatory compliance. 100 = fully "
            "compliant with CAN-SPAM and GDPR. 0 = multiple violations. "
            "Check for: opt-out mechanism, sender identity, no deceptive "
            "subject lines, appropriate data handling."
        ),
        weight=0.2,
        threshold=80,
    ),
    "mentions_tech_stack": EvalCriterion(
        name="mentions_tech_stack",
        description=(
            "Score 0-100 on whether the output references the "
            "prospect's actual technology stack from the input data. "
            "100 = specifically names their technologies. 0 = no "
            "technology references at all."
        ),
        weight=0.2,
        threshold=50,
    ),
}


# ─── Evaluation Results ─────────────────────────────────────────────


@dataclass
class CriterionScore:
    """Score for a single criterion."""

    criterion: str
    score: int  # 0-100
    reasoning: str = ""
    passed: bool = False


@dataclass
class EvalResult:
    """Complete evaluation result for one scenario."""

    scenario_name: str
    agent_type: str
    overall_score: float  # Weighted average, 0-100
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    passed: bool = False
    evaluated_at: str = ""
    judge_model: str = ""
    input_summary: str = ""
    output_summary: str = ""

    def __post_init__(self):
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()


# ─── Judge Interface ────────────────────────────────────────────────


class BaseJudge(ABC):
    """Abstract interface for an LLM judge."""

    @abstractmethod
    def evaluate(
        self,
        agent_input: dict[str, Any],
        agent_output: dict[str, Any],
        criteria: list[EvalCriterion],
    ) -> list[CriterionScore]:
        """
        Evaluate an agent's output against criteria.

        Args:
            agent_input: The input given to the agent.
            agent_output: The agent's output to evaluate.
            criteria: List of criteria to score against.

        Returns:
            List of CriterionScore objects.
        """
        ...


class MockJudge(BaseJudge):
    """
    Mock judge for CI/CD testing.

    Returns configurable scores without calling any LLM.
    Useful for testing the eval framework itself.
    """

    def __init__(self, default_score: int = 75):
        self.default_score = default_score
        self._custom_scores: dict[str, int] = {}

    def set_score(self, criterion_name: str, score: int) -> None:
        """Set a custom score for a specific criterion."""
        self._custom_scores[criterion_name] = score

    def evaluate(
        self,
        agent_input: dict[str, Any],
        agent_output: dict[str, Any],
        criteria: list[EvalCriterion],
    ) -> list[CriterionScore]:
        scores = []
        for c in criteria:
            score = self._custom_scores.get(c.name, self.default_score)
            scores.append(
                CriterionScore(
                    criterion=c.name,
                    score=score,
                    reasoning=f"Mock score: {score}/100",
                    passed=score >= c.threshold,
                )
            )
        return scores


class AnthropicJudge(BaseJudge):
    """
    LLM judge using Claude (Anthropic API).

    Uses a lightweight model (claude-3-haiku or claude-sonnet-4-20250514)
    with temperature=0 for deterministic scoring.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package required for AnthropicJudge. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def evaluate(
        self,
        agent_input: dict[str, Any],
        agent_output: dict[str, Any],
        criteria: list[EvalCriterion],
    ) -> list[CriterionScore]:
        client = self._get_client()

        criteria_descriptions = "\n".join(
            f"- **{c.name}**: {c.description}" for c in criteria
        )

        prompt = f"""You are an AI quality evaluator. Score the following agent output against each criterion.

## Agent Input (summarized):
{json.dumps(agent_input, indent=2, default=str)[:2000]}

## Agent Output (summarized):
{json.dumps(agent_output, indent=2, default=str)[:2000]}

## Criteria to evaluate:
{criteria_descriptions}

## Instructions:
For each criterion, provide:
1. A score from 0 to 100
2. A brief reasoning (1-2 sentences)

Respond in JSON format:
{{
  "scores": [
    {{"criterion": "name", "score": 75, "reasoning": "..."}},
    ...
  ]
}}"""

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the response
            text = response.content[0].text
            # Extract JSON from response (may be wrapped in markdown)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            scores = []
            criteria_map = {c.name: c for c in criteria}
            for entry in data.get("scores", []):
                crit_name = entry.get("criterion", "")
                crit = criteria_map.get(crit_name)
                score_val = int(entry.get("score", 0))
                scores.append(
                    CriterionScore(
                        criterion=crit_name,
                        score=score_val,
                        reasoning=entry.get("reasoning", ""),
                        passed=score_val >= (crit.threshold if crit else 60),
                    )
                )
            return scores

        except Exception as err:
            logger.warning(f"AnthropicJudge evaluation failed: {err}")
            # Return zero scores on failure
            return [
                CriterionScore(
                    criterion=c.name,
                    score=0,
                    reasoning=f"Evaluation failed: {err}",
                    passed=False,
                )
                for c in criteria
            ]


# ─── Base Eval Class ────────────────────────────────────────────────


class BaseAgentEval(ABC):
    """
    Base class for agent evaluations.

    Subclass this to create evals for specific agent types.
    Each eval defines test scenarios and criteria, then runs
    them through a judge.

    Usage:
        class OutreachEmailEval(BaseAgentEval):
            eval_name = "outreach_email"
            agent_type = "outreach"

            def get_test_scenarios(self):
                return [{"input": {...}, "criteria": [...]}]

            def get_criteria(self, criteria_names):
                return [BUILTIN_CRITERIA[n] for n in criteria_names]
    """

    eval_name: str = ""
    agent_type: str = ""

    def __init__(self, judge: Optional[BaseJudge] = None):
        """
        Initialize with a judge.

        If no judge is provided:
        - ENCLAVE_EVAL_MOCK=1 → MockJudge
        - Otherwise → AnthropicJudge (requires ANTHROPIC_API_KEY)
        """
        if judge is not None:
            self.judge = judge
        elif os.environ.get("ENCLAVE_EVAL_MOCK", "").strip() in ("1", "true"):
            self.judge = MockJudge()
        else:
            self.judge = AnthropicJudge()

    @abstractmethod
    def get_test_scenarios(self) -> list[dict[str, Any]]:
        """
        Return test scenarios for this eval.

        Each scenario is a dict with:
        - "name": Human-readable scenario name
        - "input": The agent input dict
        - "output": The agent output dict to evaluate
        - "criteria": List of criterion names (strings)

        Example:
            [
                {
                    "name": "cold_email_to_fintech_cto",
                    "input": {"lead": {"contact": {...}, "company": {...}}},
                    "output": {"draft_email_subject": "...", "draft_email_body": "..."},
                    "criteria": ["hallucination", "professional_tone", "relevance"],
                }
            ]
        """
        ...

    def get_criteria(self, criteria_names: list[str]) -> list[EvalCriterion]:
        """
        Resolve criterion names to EvalCriterion objects.

        Looks up names in BUILTIN_CRITERIA. Override to add custom criteria.
        """
        criteria = []
        for name in criteria_names:
            if name in BUILTIN_CRITERIA:
                criteria.append(BUILTIN_CRITERIA[name])
            else:
                logger.warning(
                    f"Unknown criterion '{name}' — skipping. "
                    f"Available: {list(BUILTIN_CRITERIA.keys())}"
                )
        return criteria

    def evaluate_scenario(
        self, scenario: dict[str, Any]
    ) -> EvalResult:
        """
        Evaluate a single test scenario.

        Args:
            scenario: Dict with 'name', 'input', 'output', 'criteria'.

        Returns:
            EvalResult with scores and pass/fail status.
        """
        name = scenario.get("name", "unnamed")
        agent_input = scenario.get("input", {})
        agent_output = scenario.get("output", {})
        criteria_names = scenario.get("criteria", [])

        criteria = self.get_criteria(criteria_names)
        if not criteria:
            return EvalResult(
                scenario_name=name,
                agent_type=self.agent_type,
                overall_score=0.0,
                passed=False,
                judge_model=getattr(self.judge, "model", "mock"),
            )

        # Run the judge
        criterion_scores = self.judge.evaluate(
            agent_input, agent_output, criteria
        )

        # Calculate weighted overall score
        total_weight = sum(c.weight for c in criteria)
        if total_weight > 0:
            criteria_map = {c.name: c for c in criteria}
            weighted_sum = 0.0
            for cs in criterion_scores:
                crit = criteria_map.get(cs.criterion)
                weight = crit.weight if crit else 1.0
                weighted_sum += cs.score * weight
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0

        # Check if all criteria passed
        all_passed = all(cs.passed for cs in criterion_scores)

        return EvalResult(
            scenario_name=name,
            agent_type=self.agent_type,
            overall_score=overall_score,
            criterion_scores=criterion_scores,
            passed=all_passed,
            judge_model=getattr(self.judge, "model", "mock"),
            input_summary=json.dumps(agent_input, default=str)[:200],
            output_summary=json.dumps(agent_output, default=str)[:200],
        )

    def run_all(self) -> list[EvalResult]:
        """
        Run all test scenarios and return results.

        Returns a list of EvalResult objects.
        """
        scenarios = self.get_test_scenarios()
        results = []

        for scenario in scenarios:
            result = self.evaluate_scenario(scenario)
            results.append(result)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(
                f"[{self.eval_name}] {scenario.get('name', '?')}: "
                f"{status} (score={result.overall_score:.1f})"
            )

        return results

    def assert_all_pass(self) -> None:
        """
        Run all scenarios and assert that all pass.

        Use in pytest:
            def test_outreach_quality():
                OutreachEmailEval().assert_all_pass()
        """
        results = self.run_all()
        failures = [r for r in results if not r.passed]

        if failures:
            failure_details = "\n".join(
                f"  - {r.scenario_name}: {r.overall_score:.1f}/100 "
                f"(failed: {[cs.criterion for cs in r.criterion_scores if not cs.passed]})"
                for r in failures
            )
            raise AssertionError(
                f"{len(failures)}/{len(results)} eval scenarios failed:\n"
                f"{failure_details}"
            )
