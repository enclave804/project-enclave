"""
Tests for overseer_agent.py — The SRE Meta-Agent.

Tests:
- Registration: @register_agent_type("overseer") works
- State initialization: OverseerAgentState populated correctly
- Graph building: LangGraph compiles with correct nodes/edges
- Collect metrics: all system tools called, state populated
- Diagnose (healthy): skips LLM when system is healthy
- Diagnose (degraded): calls LLM for root cause analysis
- Route by severity: routes to plan_actions or report
- Plan actions: converts recommendations to executable plan
- Execute actions: handles each action type
- Report generation: produces human-readable report
- Knowledge writing: stores insights when issues detected
- Full lifecycle: end-to-end with mocked LLM
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.state import OverseerAgentState
from core.config.agent_schema import AgentInstanceConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_overseer_config(**overrides) -> AgentInstanceConfig:
    """Create a minimal Overseer agent config."""
    defaults = {
        "agent_id": "overseer_test",
        "agent_type": "overseer",
        "name": "Test Overseer",
        "vertical_id": "test_vertical",
    }
    defaults.update(overrides)
    return AgentInstanceConfig(**defaults)


def _make_mock_db(
    runs: list | None = None,
    pending_tasks: int = 0,
    zombie_count: int = 0,
):
    """Create a mock DB with configurable responses."""
    db = MagicMock()
    db.get_agent_runs.return_value = runs or []
    db.count_pending_tasks.return_value = pending_tasks
    db.recover_zombie_tasks.return_value = zombie_count
    db.reset_agent_errors.return_value = None
    db.record_agent_error.return_value = None
    db.log_agent_run.return_value = None
    db.store_insight.return_value = {"id": "insight_123"}
    return db


def _make_mock_llm():
    """Create a mock Anthropic client."""
    client = MagicMock()

    def create_response(**kwargs):
        resp = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "diagnosis": "Elevated error rate in outreach agent",
            "root_causes": ["API timeout from Apollo service"],
            "recommended_actions": [
                {
                    "action": "recover_zombie_tasks",
                    "target": "task_queue",
                    "priority": "high",
                    "reasoning": "3 zombie tasks detected",
                },
                {
                    "action": "alert_team",
                    "target": "outreach",
                    "priority": "medium",
                    "reasoning": "40% failure rate needs investigation",
                },
            ],
        })
        resp.content = [text_block]
        resp.usage.input_tokens = 500
        resp.usage.output_tokens = 200
        return resp

    client.messages.create.side_effect = create_response
    return client


# ---------------------------------------------------------------------------
# Registration Tests
# ---------------------------------------------------------------------------

class TestOverseerRegistration:
    """Tests for agent type registration."""

    def test_registered_as_overseer(self):
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        # Import triggers @register_agent_type
        from core.agents.implementations.overseer_agent import OverseerAgent

        assert "overseer" in AGENT_IMPLEMENTATIONS
        assert AGENT_IMPLEMENTATIONS["overseer"] is OverseerAgent

    def test_agent_type_set(self):
        from core.agents.implementations.overseer_agent import OverseerAgent
        assert OverseerAgent.agent_type == "overseer"


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------

class TestOverseerState:
    """Tests for OverseerAgentState initialization."""

    def test_state_has_all_fields(self):
        """OverseerAgentState should have health, diagnostic, and action fields."""
        annotations = OverseerAgentState.__annotations__
        expected_fields = [
            "system_health", "health_status",
            "error_logs", "agent_error_rates",
            "task_queue_status", "cache_performance",
            "issues", "issue_count", "critical_count",
            "diagnosis", "root_causes", "recommended_actions",
            "actions_planned", "actions_approved",
            "actions_executed", "actions_failed",
            "report_summary", "report_generated_at",
        ]
        for field in expected_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_inherits_base_state(self):
        """Should inherit BaseAgentState fields."""
        annotations = OverseerAgentState.__annotations__
        base_fields = ["agent_id", "vertical_id", "run_id", "task_input", "error"]
        for field in base_fields:
            assert field in annotations, f"Missing base field: {field}"


# ---------------------------------------------------------------------------
# Graph Tests
# ---------------------------------------------------------------------------

class TestOverseerGraph:
    """Tests for the Overseer's LangGraph state machine."""

    def test_graph_builds(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        graph = agent.build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        graph = agent.build_graph()
        # Verify all nodes exist by checking the graph's node dictionary
        nodes = graph.nodes
        expected_nodes = [
            "collect_metrics", "diagnose", "plan_actions",
            "human_review", "execute_actions", "report",
        ]
        for node_name in expected_nodes:
            assert node_name in nodes, f"Missing node: {node_name}"

    def test_get_tools_returns_empty(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )
        assert agent.get_tools() == []

    def test_get_state_class(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )
        assert agent.get_state_class() is OverseerAgentState


# ---------------------------------------------------------------------------
# Node Tests
# ---------------------------------------------------------------------------

class TestCollectMetrics:
    """Tests for the collect_metrics node."""

    @pytest.mark.asyncio
    async def test_collects_all_metrics(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = _make_mock_db(
            runs=[
                {"agent_id": "outreach", "status": "completed"},
                {"agent_id": "outreach", "status": "failed", "error_message": "timeout"},
            ],
            pending_tasks=5,
        )

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state: OverseerAgentState = {
            "agent_id": "overseer_test",
            "vertical_id": "test_vertical",
        }

        result = await agent._node_collect_metrics(state)

        assert "system_health" in result
        assert "health_status" in result
        assert "error_logs" in result
        assert "agent_error_rates" in result
        assert "task_queue_status" in result
        assert "issue_count" in result

    @pytest.mark.asyncio
    async def test_handles_db_failures(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = MagicMock()
        mock_db.get_agent_runs.side_effect = Exception("DB down")
        mock_db.count_pending_tasks.side_effect = Exception("DB down")
        mock_db.recover_zombie_tasks.side_effect = Exception("DB down")
        mock_db.reset_agent_errors.return_value = None
        mock_db.record_agent_error.return_value = None

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state: OverseerAgentState = {
            "agent_id": "overseer_test",
            "vertical_id": "test_vertical",
        }

        # Should not raise
        result = await agent._node_collect_metrics(state)
        assert "health_status" in result


class TestDiagnose:
    """Tests for the diagnose node."""

    @pytest.mark.asyncio
    async def test_healthy_system_skips_llm(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state: OverseerAgentState = {
            "health_status": "healthy",
            "issues": [],
            "agent_error_rates": {},
            "error_logs": [],
            "task_queue_status": {},
        }

        result = await agent._node_diagnose(state)

        assert "All systems nominal" in result["diagnosis"]
        assert result["root_causes"] == []
        assert result["recommended_actions"] == []
        # LLM should NOT have been called
        agent.llm.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_degraded_system_calls_llm(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state: OverseerAgentState = {
            "health_status": "degraded",
            "issues": [{"severity": "warning", "component": "outreach", "message": "high errors"}],
            "agent_error_rates": {"outreach": {"failure_rate": 0.4}},
            "error_logs": [{"level": "ERROR", "message": "timeout"}],
            "task_queue_status": {"pending_count": 10},
        }

        result = await agent._node_diagnose(state)

        assert result["diagnosis"]
        assert len(result["recommended_actions"]) > 0

    @pytest.mark.asyncio
    async def test_handles_non_json_llm_response(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_llm = MagicMock()
        resp = MagicMock()
        text_block = MagicMock()
        text_block.text = "This is not valid JSON but a natural language response"
        resp.content = [text_block]
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        mock_llm.messages.create.return_value = resp

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=mock_llm,
        )

        state: OverseerAgentState = {
            "health_status": "degraded",
            "issues": [{"severity": "warning", "component": "test", "message": "test"}],
            "agent_error_rates": {},
            "error_logs": [],
            "task_queue_status": {},
        }

        result = await agent._node_diagnose(state)

        # Should handle gracefully — use raw text as diagnosis
        assert result["diagnosis"]
        assert len(result["root_causes"]) > 0


# ---------------------------------------------------------------------------
# Routing Tests
# ---------------------------------------------------------------------------

class TestRouting:
    """Tests for Overseer routing logic."""

    def test_route_needs_action(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {"recommended_actions": [{"action": "recover_zombie_tasks"}]}
        assert agent._route_by_severity(state) == "needs_action"

    def test_route_healthy(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {"recommended_actions": []}
        assert agent._route_by_severity(state) == "healthy"

    def test_route_after_review_approved(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {"actions_approved": True}
        assert agent._route_after_review(state) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {"actions_approved": False}
        assert agent._route_after_review(state) == "rejected"


# ---------------------------------------------------------------------------
# Action Execution Tests
# ---------------------------------------------------------------------------

class TestExecuteActions:
    """Tests for action execution."""

    @pytest.mark.asyncio
    async def test_recover_zombies(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = _make_mock_db(zombie_count=3)

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "actions_planned": [
                {"action": "recover_zombie_tasks", "target": "task_queue", "priority": "high"},
            ],
        }

        result = await agent._node_execute_actions(state)

        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "executed"
        assert "3 zombie" in result["actions_executed"][0]["result"]
        mock_db.recover_zombie_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )
        # Initialize cache with some entries
        agent.cache.put("key1", "value1")
        agent.cache.put("key2", "value2")

        state = {
            "actions_planned": [
                {"action": "clear_cache", "target": "cache", "priority": "low"},
            ],
        }

        result = await agent._node_execute_actions(state)

        assert len(result["actions_executed"]) == 1
        assert "2 cache" in result["actions_executed"][0]["result"]
        assert agent.cache.size == 0

    @pytest.mark.asyncio
    async def test_alert_team(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "actions_planned": [
                {
                    "action": "alert_team",
                    "target": "outreach",
                    "priority": "medium",
                    "reasoning": "High failure rate",
                },
            ],
        }

        result = await agent._node_execute_actions(state)

        assert len(result["actions_executed"]) == 1
        assert result["actions_executed"][0]["status"] == "executed"

    @pytest.mark.asyncio
    async def test_escalation(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "actions_planned": [
                {"action": "escalate_to_human", "target": "system", "priority": "high"},
            ],
        }

        result = await agent._node_execute_actions(state)
        assert result["actions_executed"][0]["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_manual_action(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "actions_planned": [
                {"action": "restart_agent", "target": "outreach", "priority": "high"},
            ],
        }

        result = await agent._node_execute_actions(state)
        assert result["actions_executed"][0]["status"] == "requires_manual"

    @pytest.mark.asyncio
    async def test_action_failure_handling(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = _make_mock_db()
        mock_db.recover_zombie_tasks.side_effect = Exception("DB locked")

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "actions_planned": [
                {"action": "recover_zombie_tasks", "target": "queue", "priority": "high"},
            ],
        }

        result = await agent._node_execute_actions(state)
        assert len(result["actions_failed"]) == 1
        assert "DB locked" in result["actions_failed"][0]["error"]


# ---------------------------------------------------------------------------
# Report Tests
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for report generation."""

    @pytest.mark.asyncio
    async def test_healthy_report(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "health_status": "healthy",
            "diagnosis": "All systems nominal",
            "issues": [],
            "root_causes": [],
            "actions_executed": [],
            "actions_failed": [],
            "agent_error_rates": {},
        }

        result = await agent._node_report(state)

        assert "Overseer Status Report" in result["report_summary"]
        assert "HEALTHY" in result["report_summary"]
        assert result["report_generated_at"]

    @pytest.mark.asyncio
    async def test_degraded_report_includes_issues(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "health_status": "critical",
            "diagnosis": "Multiple agent failures detected",
            "issues": [
                {"severity": "critical", "component": "outreach", "message": "50% failure rate"},
            ],
            "root_causes": ["Apollo API down"],
            "actions_executed": [
                {"action": "recover_zombie_tasks", "status": "executed", "result": "3 recovered"},
            ],
            "actions_failed": [],
            "agent_error_rates": {
                "outreach": {"failure_rate": 0.5, "total_runs": 10, "risk_level": "critical"},
            },
        }

        result = await agent._node_report(state)

        report = result["report_summary"]
        assert "CRITICAL" in report
        assert "Issues Detected" in report
        assert "Root Causes" in report
        assert "Actions Taken" in report
        assert "Agent Health" in report


# ---------------------------------------------------------------------------
# Knowledge Writing Tests
# ---------------------------------------------------------------------------

class TestKnowledgeWriting:
    """Tests for write_knowledge integration."""

    @pytest.mark.asyncio
    async def test_stores_insight_on_issues(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = _make_mock_db()

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        result = {
            "health_status": "degraded",
            "issues": [{"severity": "warning", "component": "x", "message": "y"}],
            "diagnosis": "System showing elevated error rates",
            "report_generated_at": "2024-01-15T10:00:00Z",
            "critical_count": 0,
        }

        await agent.write_knowledge(result)

        # Should have stored an insight
        mock_db.store_insight.assert_called_once()
        call_kwargs = mock_db.store_insight.call_args
        assert call_kwargs[1]["insight_type"] == "system_health"

    @pytest.mark.asyncio
    async def test_skips_insight_when_healthy(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        mock_db = _make_mock_db()

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        result = {
            "health_status": "healthy",
            "issues": [],
            "diagnosis": "All good",
        }

        await agent.write_knowledge(result)

        # Should NOT store anything when healthy
        mock_db.store_insight.assert_not_called()


# ---------------------------------------------------------------------------
# Plan Actions Tests
# ---------------------------------------------------------------------------

class TestPlanActions:
    """Tests for the plan_actions node."""

    @pytest.mark.asyncio
    async def test_converts_recommendations_to_plan(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = {
            "diagnosis": "Issues found",
            "recommended_actions": [
                {
                    "action": "recover_zombie_tasks",
                    "target": "task_queue",
                    "priority": "high",
                    "reasoning": "3 zombies detected",
                },
                {
                    "action": "restart_agent",
                    "target": "outreach",
                    "priority": "critical",
                    "reasoning": "Agent disabled by circuit breaker",
                },
            ],
        }

        result = await agent._node_plan_actions(state)

        assert len(result["actions_planned"]) == 2
        assert result["requires_human_approval"] is True
        # recover_zombie_tasks should be marked executable
        assert result["actions_planned"][0]["executable"] is True
        # restart_agent should NOT be executable (manual)
        assert result["actions_planned"][1]["executable"] is False


# ---------------------------------------------------------------------------
# Prepare State Tests
# ---------------------------------------------------------------------------

class TestPrepareState:
    """Tests for state preparation."""

    def test_prepare_initial_state(self):
        from core.agents.implementations.overseer_agent import OverseerAgent

        agent = OverseerAgent(
            config=_make_overseer_config(),
            db=_make_mock_db(),
            embedder=MagicMock(),
            anthropic_client=_make_mock_llm(),
        )

        state = agent._prepare_initial_state({"mode": "full_check"}, "run_123")

        assert state["agent_id"] == "overseer_test"
        assert state["vertical_id"] == "test_vertical"
        assert state["run_id"] == "run_123"
        assert state["system_health"] == {}
        assert state["health_status"] == "unknown"
        assert state["issues"] == []
        assert state["actions_planned"] == []
        assert state["actions_approved"] is False
