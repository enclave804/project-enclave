"""
Unit tests for the Enterprise Governance Layer.

Tests the three pillars:
1. Distributed Tracing (LangFuse integration with silent fail)
2. LLM-as-a-Judge quality gate (eval framework)
3. Native Feature Flags (deterministic rollouts via DB config)
"""

import asyncio
import hashlib
import os

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from core.agents.base import BaseAgent, _hash_to_bucket
from core.agents.state import BaseAgentState
from core.config.agent_schema import AgentInstanceConfig
from core.observability.tracing import (
    get_tracer,
    create_trace,
    create_span,
    end_span,
    flush_tracer,
    NoOpTrace,
    NoOpSpan,
    traced_operation,
)
from tests.evals.base_eval import (
    BaseAgentEval,
    MockJudge,
    EvalCriterion,
    EvalResult,
    CriterionScore,
    BUILTIN_CRITERIA,
)


# ─── Test Agent Implementation ────────────────────────────────────


class _StubAgent(BaseAgent):
    """Minimal agent for testing base class behavior."""

    agent_type = "_stub"

    def build_graph(self):
        return MagicMock()

    def get_tools(self):
        return []

    def get_state_class(self):
        return BaseAgentState


def _make_agent(**kwargs) -> _StubAgent:
    """Helper to create a stub agent with optional overrides."""
    config_kwargs = {
        "agent_id": "test_stub",
        "agent_type": "_stub",
        "name": "Test Stub",
        "vertical_id": "test_vertical",
    }
    config_kwargs.update(kwargs)
    config = AgentInstanceConfig(**config_kwargs)
    return _StubAgent(
        config=config,
        db=MagicMock(),
        embedder=MagicMock(),
        anthropic_client=MagicMock(),
    )


# ─── Distributed Tracing Tests ──────────────────────────────────────


class TestNoOpTrace:
    """Tests for the no-op trace/span fallbacks."""

    def test_noop_trace_returns_noop_span(self):
        trace = NoOpTrace()
        span = trace.span(name="test")
        assert isinstance(span, NoOpSpan)

    def test_noop_span_returns_noop_span(self):
        span = NoOpSpan()
        child = span.span(name="child")
        assert isinstance(child, NoOpSpan)

    def test_noop_trace_methods_dont_raise(self):
        trace = NoOpTrace()
        trace.update(output={"test": True})
        trace.event(name="test")
        trace.end()
        # No assertions needed — just verify no exceptions

    def test_noop_span_methods_dont_raise(self):
        span = NoOpSpan()
        span.end(output={"test": True})
        span.update(name="updated")
        span.event(name="test")
        span.generation(name="gen")

    def test_create_trace_without_tracer_returns_noop(self):
        trace = create_trace(
            None,
            name="test",
            agent_id="test",
            vertical_id="v",
            run_id="r1",
        )
        assert isinstance(trace, NoOpTrace)

    def test_create_span_on_noop_trace_returns_noop(self):
        trace = NoOpTrace()
        span = create_span(trace, name="test")
        assert isinstance(span, NoOpSpan)

    def test_end_span_on_noop_is_safe(self):
        span = NoOpSpan()
        end_span(span, output_data={"test": True})
        # No assertions needed

    def test_flush_none_tracer_is_safe(self):
        flush_tracer(None)
        # No assertions needed


class TestTracerInitialization:
    """Tests for get_tracer() with silent fail behavior."""

    def test_no_keys_returns_none(self):
        """Without LANGFUSE keys, tracer should return None."""
        import core.observability.tracing as tracing_module

        # Reset the singleton
        tracing_module._langfuse_client = None
        tracing_module._langfuse_initialized = False

        with patch.dict(os.environ, {}, clear=True):
            # Ensure keys are not set
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            result = get_tracer()

        assert result is None

        # Reset for other tests
        tracing_module._langfuse_initialized = False
        tracing_module._langfuse_client = None

    def test_is_langfuse_configured_false_without_keys(self):
        from core.observability.tracing import _is_langfuse_configured

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            assert _is_langfuse_configured() is False

    def test_is_langfuse_configured_true_with_keys(self):
        from core.observability.tracing import _is_langfuse_configured

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
            },
        ):
            assert _is_langfuse_configured() is True


class TestTracedOperation:
    """Tests for the traced_operation context manager."""

    def test_traced_operation_with_noop(self):
        trace = NoOpTrace()
        with traced_operation(trace, "test_op") as span:
            assert isinstance(span, NoOpSpan)

    def test_traced_operation_exception_handling(self):
        trace = NoOpTrace()
        with pytest.raises(ValueError, match="test error"):
            with traced_operation(trace, "failing_op"):
                raise ValueError("test error")


class TestTracingInBaseAgent:
    """Tests that tracing integrates correctly with BaseAgent.run()."""

    def test_run_creates_trace(self):
        """run() should create a trace even when LangFuse is not configured."""
        agent = _make_agent()
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "test"})
        agent._graph = mock_graph

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                agent.run({"task": "test"})
            )
        finally:
            loop.close()

        # Should complete successfully with NoOpTrace
        assert result["output"] == "test"

    def test_run_sets_current_trace(self):
        """run() should set _current_trace on the agent."""
        agent = _make_agent()
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "test"})
        agent._graph = mock_graph

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.run({"task": "test"}))
        finally:
            loop.close()

        assert hasattr(agent, "_current_trace")


# ─── Feature Flags Tests ────────────────────────────────────────────


class TestHashToBucket:
    """Tests for the deterministic hash bucketing function."""

    def test_returns_int_in_range(self):
        result = _hash_to_bucket("test_key")
        assert isinstance(result, int)
        assert 0 <= result < 100

    def test_deterministic(self):
        """Same key should always produce the same bucket."""
        a = _hash_to_bucket("agent:flag:lead-123")
        b = _hash_to_bucket("agent:flag:lead-123")
        assert a == b

    def test_different_keys_different_buckets(self):
        """Different keys should produce different buckets (usually)."""
        buckets = set()
        for i in range(50):
            buckets.add(_hash_to_bucket(f"key-{i}"))
        # With 50 random keys, we expect at least 10 unique buckets
        assert len(buckets) > 10

    def test_custom_bucket_count(self):
        result = _hash_to_bucket("test", buckets=10)
        assert 0 <= result < 10


class TestFeatureFlags:
    """Tests for BaseAgent.check_feature_flag()."""

    def test_missing_flag_returns_false(self):
        agent = _make_agent()
        assert agent.check_feature_flag("nonexistent_flag") is False

    def test_boolean_flag_true(self):
        agent = _make_agent(params={"flags": {"dark_mode": True}})
        assert agent.check_feature_flag("dark_mode") is True

    def test_boolean_flag_false(self):
        agent = _make_agent(params={"flags": {"dark_mode": False}})
        assert agent.check_feature_flag("dark_mode") is False

    def test_structured_flag_enabled(self):
        agent = _make_agent(
            params={
                "flags": {
                    "aggressive_mode": {"enabled": True}
                }
            }
        )
        assert agent.check_feature_flag("aggressive_mode") is True

    def test_structured_flag_disabled(self):
        agent = _make_agent(
            params={
                "flags": {
                    "aggressive_mode": {"enabled": False}
                }
            }
        )
        assert agent.check_feature_flag("aggressive_mode") is False

    def test_rollout_percent_deterministic(self):
        """Same lead_id should always get the same result."""
        agent = _make_agent(
            params={
                "flags": {
                    "new_template": {
                        "enabled": True,
                        "rollout_percent": 50,
                    }
                }
            }
        )
        context = {"lead_id": "lead-abc-123"}
        result1 = agent.check_feature_flag("new_template", context)
        result2 = agent.check_feature_flag("new_template", context)
        assert result1 == result2

    def test_rollout_percent_0_always_false(self):
        agent = _make_agent(
            params={
                "flags": {
                    "risky_feature": {
                        "enabled": True,
                        "rollout_percent": 0,
                    }
                }
            }
        )
        # Test with many different leads — all should be False
        for i in range(20):
            result = agent.check_feature_flag(
                "risky_feature", {"lead_id": f"lead-{i}"}
            )
            assert result is False

    def test_rollout_percent_100_always_true(self):
        agent = _make_agent(
            params={
                "flags": {
                    "safe_feature": {
                        "enabled": True,
                        "rollout_percent": 100,
                    }
                }
            }
        )
        # Test with many different leads — all should be True
        for i in range(20):
            result = agent.check_feature_flag(
                "safe_feature", {"lead_id": f"lead-{i}"}
            )
            assert result is True

    def test_rollout_without_lead_id_returns_false(self):
        """If no deterministic key in context, default to False."""
        agent = _make_agent(
            params={
                "flags": {
                    "partial_rollout": {
                        "enabled": True,
                        "rollout_percent": 50,
                    }
                }
            }
        )
        assert agent.check_feature_flag("partial_rollout", {}) is False

    def test_rollout_uses_entity_id_fallback(self):
        """Should use entity_id if lead_id is not available."""
        agent = _make_agent(
            params={
                "flags": {
                    "feature_x": {
                        "enabled": True,
                        "rollout_percent": 100,
                    }
                }
            }
        )
        result = agent.check_feature_flag(
            "feature_x", {"entity_id": "entity-456"}
        )
        assert result is True

    def test_db_fallback_when_no_local_flags(self):
        """Should try loading flags from DB when not in params."""
        agent = _make_agent()
        agent.db.get_agent = MagicMock(
            return_value={
                "config": {
                    "flags": {
                        "db_feature": True,
                    }
                }
            }
        )
        assert agent.check_feature_flag("db_feature") is True
        agent.db.get_agent.assert_called_once()

    def test_db_failure_returns_false(self):
        """DB errors should not crash — just return False."""
        agent = _make_agent()
        agent.db.get_agent = MagicMock(side_effect=Exception("DB down"))
        assert agent.check_feature_flag("any_flag") is False

    def test_invalid_flag_type_returns_false(self):
        """Non-bool, non-dict flag values should return False."""
        agent = _make_agent(
            params={"flags": {"weird_flag": "string_value"}}
        )
        assert agent.check_feature_flag("weird_flag") is False


# ─── LLM-as-a-Judge Eval Framework Tests ────────────────────────────


class TestMockJudge:
    """Tests for the MockJudge used in CI/CD testing."""

    def test_default_scores(self):
        judge = MockJudge(default_score=80)
        criteria = [
            EvalCriterion(name="tone", description="test", threshold=70),
            EvalCriterion(name="relevance", description="test", threshold=60),
        ]
        scores = judge.evaluate({}, {}, criteria)
        assert len(scores) == 2
        assert all(s.score == 80 for s in scores)
        assert all(s.passed for s in scores)

    def test_custom_scores(self):
        judge = MockJudge(default_score=50)
        judge.set_score("tone", 90)
        criteria = [
            EvalCriterion(name="tone", description="test", threshold=70),
            EvalCriterion(name="relevance", description="test", threshold=60),
        ]
        scores = judge.evaluate({}, {}, criteria)
        assert scores[0].score == 90  # tone (custom)
        assert scores[0].passed is True
        assert scores[1].score == 50  # relevance (default)
        assert scores[1].passed is False  # 50 < 60

    def test_below_threshold_fails(self):
        judge = MockJudge(default_score=40)
        criteria = [
            EvalCriterion(name="quality", description="test", threshold=60),
        ]
        scores = judge.evaluate({}, {}, criteria)
        assert scores[0].passed is False


class TestBuiltinCriteria:
    """Tests for the built-in evaluation criteria library."""

    def test_hallucination_exists(self):
        assert "hallucination" in BUILTIN_CRITERIA
        assert BUILTIN_CRITERIA["hallucination"].threshold == 70

    def test_professional_tone_exists(self):
        assert "professional_tone" in BUILTIN_CRITERIA

    def test_relevance_exists(self):
        assert "relevance" in BUILTIN_CRITERIA

    def test_compliance_exists(self):
        assert "compliance" in BUILTIN_CRITERIA
        assert BUILTIN_CRITERIA["compliance"].threshold == 80

    def test_all_criteria_have_descriptions(self):
        for name, criterion in BUILTIN_CRITERIA.items():
            assert criterion.description, f"{name} missing description"
            assert criterion.weight > 0, f"{name} has zero weight"


class _TestEval(BaseAgentEval):
    """Concrete eval subclass for testing."""

    eval_name = "test_eval"
    agent_type = "test"

    def __init__(self, scenarios=None, **kwargs):
        super().__init__(**kwargs)
        self._scenarios = scenarios or []

    def get_test_scenarios(self):
        return self._scenarios


class TestBaseAgentEval:
    """Tests for the BaseAgentEval framework."""

    def test_evaluate_scenario_with_mock(self):
        judge = MockJudge(default_score=85)
        eval_instance = _TestEval(
            judge=judge,
            scenarios=[
                {
                    "name": "basic_test",
                    "input": {"lead": "test"},
                    "output": {"email": "Hi there"},
                    "criteria": ["hallucination", "professional_tone"],
                }
            ],
        )
        result = eval_instance.evaluate_scenario(
            eval_instance.get_test_scenarios()[0]
        )
        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.overall_score > 0
        assert result.scenario_name == "basic_test"
        assert result.agent_type == "test"

    def test_run_all_returns_results(self):
        judge = MockJudge(default_score=85)
        eval_instance = _TestEval(
            judge=judge,
            scenarios=[
                {
                    "name": "scenario_1",
                    "input": {},
                    "output": {},
                    "criteria": ["hallucination"],
                },
                {
                    "name": "scenario_2",
                    "input": {},
                    "output": {},
                    "criteria": ["compliance"],
                },
            ],
        )
        results = eval_instance.run_all()
        assert len(results) == 2

    def test_assert_all_pass_succeeds(self):
        judge = MockJudge(default_score=90)
        eval_instance = _TestEval(
            judge=judge,
            scenarios=[
                {
                    "name": "passing_test",
                    "input": {},
                    "output": {},
                    "criteria": ["hallucination"],
                }
            ],
        )
        # Should not raise
        eval_instance.assert_all_pass()

    def test_assert_all_pass_fails(self):
        judge = MockJudge(default_score=30)
        eval_instance = _TestEval(
            judge=judge,
            scenarios=[
                {
                    "name": "failing_test",
                    "input": {},
                    "output": {},
                    "criteria": ["hallucination"],
                }
            ],
        )
        with pytest.raises(AssertionError, match="1/1 eval scenarios failed"):
            eval_instance.assert_all_pass()

    def test_empty_criteria_returns_zero_score(self):
        judge = MockJudge()
        eval_instance = _TestEval(judge=judge)
        result = eval_instance.evaluate_scenario(
            {"name": "empty", "input": {}, "output": {}, "criteria": []}
        )
        assert result.overall_score == 0.0
        assert result.passed is False

    def test_get_criteria_resolves_builtin(self):
        eval_instance = _TestEval(judge=MockJudge())
        criteria = eval_instance.get_criteria(["hallucination", "compliance"])
        assert len(criteria) == 2
        assert criteria[0].name == "hallucination"
        assert criteria[1].name == "compliance"

    def test_get_criteria_skips_unknown(self):
        eval_instance = _TestEval(judge=MockJudge())
        criteria = eval_instance.get_criteria(["hallucination", "nonexistent"])
        assert len(criteria) == 1


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_auto_timestamp(self):
        result = EvalResult(
            scenario_name="test",
            agent_type="outreach",
            overall_score=85.0,
        )
        assert result.evaluated_at  # Should be auto-populated

    def test_manual_timestamp(self):
        result = EvalResult(
            scenario_name="test",
            agent_type="outreach",
            overall_score=85.0,
            evaluated_at="2024-01-01T00:00:00Z",
        )
        assert result.evaluated_at == "2024-01-01T00:00:00Z"
