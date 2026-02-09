"""
Tests for AutopilotAgent — Phase 15.

Tests state construction, agent registration, node functions,
routing logic, graph structure, and session management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.state import AutopilotAgentState


# ── Mock Helpers ──────────────────────────────────────────────


class MockTable:
    def __init__(self, data=None):
        self._data = data or []
        self._filters = {}

    def select(self, cols="*"):
        return self

    def insert(self, data):
        if isinstance(data, dict):
            data["id"] = f"mock-{len(self._data) + 1}"
            self._data.append(data)
        return self

    def update(self, data):
        self._update_data = data
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def gte(self, col, val):
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        return self

    def execute(self):
        result = MagicMock()
        filtered = self._data
        for col, val in self._filters.items():
            filtered = [r for r in filtered if r.get(col) == val]
        result.data = filtered
        self._filters = {}
        return result


class MockDB:
    def __init__(self, tables=None):
        self._tables = tables or {}
        self.vertical_id = "test_vertical"
        self.client = self

    def table(self, name):
        return MockTable(self._tables.get(name, []))


# ── AutopilotAgentState Tests ─────────────────────────────────


class TestAutopilotAgentState:
    def test_state_is_typeddict(self):
        assert hasattr(AutopilotAgentState, "__annotations__")

    def test_state_has_performance_snapshot(self):
        annotations = AutopilotAgentState.__annotations__
        assert "performance_snapshot" in annotations

    def test_state_has_budget_snapshot(self):
        annotations = AutopilotAgentState.__annotations__
        assert "budget_snapshot" in annotations

    def test_state_has_detected_issues(self):
        annotations = AutopilotAgentState.__annotations__
        assert "detected_issues" in annotations

    def test_state_has_health_scores(self):
        annotations = AutopilotAgentState.__annotations__
        assert "health_scores" in annotations

    def test_state_has_healing_actions(self):
        annotations = AutopilotAgentState.__annotations__
        assert "healing_actions" in annotations

    def test_state_has_budget_actions(self):
        annotations = AutopilotAgentState.__annotations__
        assert "budget_actions" in annotations

    def test_state_has_experiment_proposals(self):
        annotations = AutopilotAgentState.__annotations__
        assert "experiment_proposals" in annotations

    def test_state_has_strategy_summary(self):
        annotations = AutopilotAgentState.__annotations__
        assert "strategy_summary" in annotations

    def test_state_has_strategy_confidence(self):
        annotations = AutopilotAgentState.__annotations__
        assert "strategy_confidence" in annotations

    def test_state_has_actions_planned(self):
        annotations = AutopilotAgentState.__annotations__
        assert "actions_planned" in annotations

    def test_state_has_actions_approved(self):
        annotations = AutopilotAgentState.__annotations__
        assert "actions_approved" in annotations

    def test_state_has_session_id(self):
        annotations = AutopilotAgentState.__annotations__
        assert "session_id" in annotations

    def test_state_has_session_type(self):
        annotations = AutopilotAgentState.__annotations__
        assert "session_type" in annotations

    def test_state_has_report_fields(self):
        annotations = AutopilotAgentState.__annotations__
        assert "report_summary" in annotations
        assert "report_generated_at" in annotations

    def test_state_creation(self):
        state: AutopilotAgentState = {
            "agent_id": "autopilot_v1",
            "vertical_id": "test",
            "performance_snapshot": {},
            "detected_issues": [],
            "health_scores": {},
            "strategy_confidence": 0.0,
            "actions_approved": False,
        }
        assert state["agent_id"] == "autopilot_v1"

    def test_full_state_creation(self):
        state: AutopilotAgentState = {
            "agent_id": "autopilot_v1",
            "vertical_id": "test",
            "run_id": "run-123",
            "performance_snapshot": {"agents": []},
            "budget_snapshot": {"total_spend": 0},
            "experiment_snapshot": [],
            "detected_issues": [],
            "health_scores": {"agent_a": 0.9},
            "optimization_opportunities": [],
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [],
            "strategy_summary": "All good",
            "strategy_confidence": 0.8,
            "actions_planned": [],
            "actions_approved": True,
            "actions_executed": [],
            "actions_failed": [],
            "session_id": "s-123",
            "session_type": "full_analysis",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["strategy_confidence"] == 0.8
        assert state["actions_approved"] is True


# ── Agent Registration ────────────────────────────────────────


class TestRegistration:
    def test_autopilot_importable(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        assert AutopilotAgent is not None

    def test_agent_type_is_autopilot(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        assert AutopilotAgent.agent_type == "autopilot"

    def test_registered_in_registry(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "autopilot" in AGENT_IMPLEMENTATIONS

    def test_system_prompt_defined(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "Autopilot" in AUTOPILOT_SYSTEM_PROMPT
        assert "SelfHealer" in AUTOPILOT_SYSTEM_PROMPT
        assert "BudgetManager" in AUTOPILOT_SYSTEM_PROMPT
        assert "Strategist" in AUTOPILOT_SYSTEM_PROMPT

    def test_session_types(self):
        from core.agents.implementations.autopilot_agent import (
            SESSION_FULL, SESSION_HEALING, SESSION_BUDGET, SESSION_STRATEGY,
            VALID_SESSION_TYPES,
        )
        assert SESSION_FULL == "full_analysis"
        assert SESSION_HEALING == "healing"
        assert SESSION_BUDGET == "budget"
        assert SESSION_STRATEGY == "strategy"
        assert len(VALID_SESSION_TYPES) == 4


# ── Routing Functions ─────────────────────────────────────────


class TestRouting:
    def test_route_by_issues_with_issues(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {
            "detected_issues": [{"agent_id": "a1", "severity": "critical"}],
            "optimization_opportunities": [],
        }
        result = AutopilotAgent._route_by_issues(state)
        assert result == "needs_strategy"

    def test_route_by_issues_with_opportunities(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {
            "detected_issues": [],
            "optimization_opportunities": [{"type": "high_performer"}],
        }
        result = AutopilotAgent._route_by_issues(state)
        assert result == "needs_strategy"

    def test_route_by_issues_all_clear(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {
            "detected_issues": [],
            "optimization_opportunities": [],
        }
        result = AutopilotAgent._route_by_issues(state)
        assert result == "all_clear"

    def test_route_by_approval_approved(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {"actions_approved": True}
        result = AutopilotAgent._route_by_approval(state)
        assert result == "approved"

    def test_route_by_approval_rejected(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {"actions_approved": False}
        result = AutopilotAgent._route_by_approval(state)
        assert result == "rejected"

    def test_route_by_approval_missing_key(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        state = {}
        result = AutopilotAgent._route_by_approval(state)
        assert result == "rejected"


# ── Rule-Based Strategy ───────────────────────────────────────


class TestRuleBasedStrategy:
    def test_fallback_with_issues(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent

        # Create minimal mock
        mock_config = MagicMock()
        mock_config.agent_id = "autopilot_v1"
        mock_config.vertical_id = "test"
        mock_config.name = "Test"
        mock_config.params = {}
        mock_config.human_gates.enabled = False
        mock_config.human_gates.gate_before = []

        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        context = {
            "detected_issues": [
                {"issue_type": "low_health", "agent_id": "a1", "details": {"message": "Low health"}},
            ],
            "opportunities": [],
        }
        result = agent._rule_based_strategy(context)
        assert "summary" in result
        assert "priority_actions" in result
        assert len(result["priority_actions"]) >= 1

    def test_fallback_with_budget_issue(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent

        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        context = {
            "detected_issues": [
                {"issue_type": "budget_inefficiency", "agent_id": "budget", "details": {"message": "Low ROAS"}},
            ],
            "opportunities": [],
        }
        result = agent._rule_based_strategy(context)
        categories = [a["category"] for a in result["priority_actions"]]
        assert "budget" in categories

    def test_fallback_with_opportunities(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent

        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        context = {
            "detected_issues": [],
            "opportunities": [
                {
                    "potential_impact": "high",
                    "suggested_action": "Do something",
                    "description": "Important opportunity",
                    "confidence": 0.8,
                    "details": {"agent_id": "a1"},
                },
            ],
        }
        result = agent._rule_based_strategy(context)
        assert len(result["priority_actions"]) >= 1

    def test_fallback_empty(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent

        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        result = agent._rule_based_strategy({"detected_issues": [], "opportunities": []})
        assert result["strategy_confidence"] == 0.6
        assert len(result["priority_actions"]) == 0


# ── Report Building ───────────────────────────────────────────


class TestReportBuilding:
    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        return agent

    def test_empty_report(self):
        agent = self._make_agent()
        report = agent._build_report(
            issues=[], opportunities=[], executed=[], failed=[],
            summary="", confidence=0, health_scores={},
        )
        assert "Autopilot Report" in report

    def test_report_with_issues(self):
        agent = self._make_agent()
        issues = [
            {"agent_id": "a1", "severity": "critical", "details": {"message": "Down"}},
        ]
        report = agent._build_report(
            issues=issues, opportunities=[], executed=[], failed=[],
            summary="Issues found", confidence=0.7, health_scores={},
        )
        assert "Issues Detected" in report
        assert "CRITICAL" in report

    def test_report_with_actions(self):
        agent = self._make_agent()
        executed = [
            {"category": "healing", "action": "Fixed config", "status": "success"},
        ]
        report = agent._build_report(
            issues=[], opportunities=[], executed=executed, failed=[],
            summary="Actions taken", confidence=0.8, health_scores={},
        )
        assert "Actions Taken" in report

    def test_report_with_failed_actions(self):
        agent = self._make_agent()
        failed = [
            {"category": "budget", "action": "Reallocation", "error": "Timeout"},
        ]
        report = agent._build_report(
            issues=[], opportunities=[], executed=[], failed=failed,
            summary="", confidence=0, health_scores={},
        )
        assert "Actions Failed" in report

    def test_report_with_health_scores(self):
        agent = self._make_agent()
        health_scores = {"a1": 0.9, "a2": 0.3}
        report = agent._build_report(
            issues=[], opportunities=[], executed=[], failed=[],
            summary="", confidence=0, health_scores=health_scores,
        )
        assert "Health Scores" in report
        assert "healthy" in report
        assert "critical" in report

    def test_report_with_strategy_summary(self):
        agent = self._make_agent()
        report = agent._build_report(
            issues=[], opportunities=[], executed=[], failed=[],
            summary="The system is performing well overall.",
            confidence=0.85, health_scores={},
        )
        assert "Strategy Summary" in report
        assert "85%" in report

    def test_report_with_opportunities(self):
        agent = self._make_agent()
        opps = [{"potential_impact": "high", "description": "Big chance"}]
        report = agent._build_report(
            issues=[], opportunities=opps, executed=[], failed=[],
            summary="", confidence=0, health_scores={},
        )
        assert "Opportunities" in report


# ── Graph Structure ───────────────────────────────────────────


class TestGraphStructure:
    def test_build_graph_importable(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        assert hasattr(AutopilotAgent, "build_graph")

    def test_get_tools_empty(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        tools = agent.get_tools()
        assert tools == []

    def test_get_state_class(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        state_class = agent.get_state_class()
        assert state_class is AutopilotAgentState


# ── System Prompt ─────────────────────────────────────────────


class TestSystemPrompt:
    def test_prompt_mentions_safety(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "NEVER" in AUTOPILOT_SYSTEM_PROMPT
        # Should mention no code-level changes
        assert "code" in AUTOPILOT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_budget_constraint(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "20%" in AUTOPILOT_SYSTEM_PROMPT

    def test_prompt_mentions_human_review(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "human" in AUTOPILOT_SYSTEM_PROMPT.lower()

    def test_prompt_expects_json(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "JSON" in AUTOPILOT_SYSTEM_PROMPT

    def test_prompt_mentions_total_spend(self):
        from core.agents.implementations.autopilot_agent import AUTOPILOT_SYSTEM_PROMPT
        assert "total" in AUTOPILOT_SYSTEM_PROMPT.lower()
        assert "spend" in AUTOPILOT_SYSTEM_PROMPT.lower()


# ── Node Functions ───────────────────────────────────────────


class TestNodeDetectIssues:
    """Test the _node_detect_issues logic."""

    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        mock_config.agent_id = "autopilot_v1"
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.db = MockDB()
        agent.embedder = None
        agent._hive = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_detect_low_health_agents(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "performance_snapshot": {"agents": []},
            "health_scores": {"agent_a": 0.2, "agent_b": 0.9},
            "budget_snapshot": {"roas": 2.0, "total_spend": 0},
        }
        result = await agent._node_detect_issues(state)
        issues = result["detected_issues"]
        # agent_a has health < 0.5, should be flagged
        assert any(i["agent_id"] == "agent_a" for i in issues)
        # agent_b has health >= 0.5, should NOT be flagged
        assert not any(i["agent_id"] == "agent_b" for i in issues)

    @pytest.mark.asyncio
    async def test_detect_budget_inefficiency(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "performance_snapshot": {"agents": []},
            "health_scores": {},
            "budget_snapshot": {"roas": 0.5, "total_spend": 5000},
        }
        result = await agent._node_detect_issues(state)
        issues = result["detected_issues"]
        assert any(i["issue_type"] == "budget_inefficiency" for i in issues)

    @pytest.mark.asyncio
    async def test_detect_no_issues_when_healthy(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "performance_snapshot": {"agents": []},
            "health_scores": {"a1": 0.95},
            "budget_snapshot": {"roas": 3.0, "total_spend": 0},
        }
        result = await agent._node_detect_issues(state)
        issues = result["detected_issues"]
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_detect_critical_vs_warning_threshold(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "performance_snapshot": {"agents": []},
            "health_scores": {"crit_agent": 0.2, "warn_agent": 0.4},
            "budget_snapshot": {},
        }
        result = await agent._node_detect_issues(state)
        issues = result["detected_issues"]
        crit = [i for i in issues if i["agent_id"] == "crit_agent"]
        warn = [i for i in issues if i["agent_id"] == "warn_agent"]
        assert crit[0]["severity"] == "critical"
        assert warn[0]["severity"] == "warning"


class TestNodeHumanReview:
    """Test the _node_human_review gate."""

    def _make_agent(self, auto_approve=0.0):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {"auto_approve_threshold": auto_approve}
        mock_config.agent_id = "autopilot_v1"
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        return agent

    @pytest.mark.asyncio
    async def test_no_auto_approve_by_default(self):
        agent = self._make_agent(auto_approve=0.0)
        state = {
            "actions_planned": [{"action": "test"}],
            "strategy_confidence": 0.99,
            "strategy_summary": "Good",
            "actions_approved": False,
        }
        result = await agent._node_human_review(state)
        assert result["actions_approved"] is False

    @pytest.mark.asyncio
    async def test_auto_approve_when_threshold_met(self):
        agent = self._make_agent(auto_approve=0.7)
        state = {
            "actions_planned": [{"action": "test"}],
            "strategy_confidence": 0.9,
            "strategy_summary": "High confidence",
            "actions_approved": False,
        }
        result = await agent._node_human_review(state)
        assert result["actions_approved"] is True

    @pytest.mark.asyncio
    async def test_auto_approve_not_met_below_threshold(self):
        agent = self._make_agent(auto_approve=0.9)
        state = {
            "actions_planned": [{"action": "test"}],
            "strategy_confidence": 0.5,
            "strategy_summary": "Low confidence",
            "actions_approved": False,
        }
        result = await agent._node_human_review(state)
        assert result["actions_approved"] is False

    @pytest.mark.asyncio
    async def test_preserves_manual_approval(self):
        agent = self._make_agent(auto_approve=0.0)
        state = {
            "actions_planned": [],
            "strategy_confidence": 0.3,
            "strategy_summary": "",
            "actions_approved": True,  # Manually set
        }
        result = await agent._node_human_review(state)
        assert result["actions_approved"] is True


class TestNodeReport:
    """Test the _node_report output."""

    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        mock_config.agent_id = "autopilot_v1"
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.db = MockDB()
        return agent

    @pytest.mark.asyncio
    async def test_report_generates_timestamp(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "detected_issues": [],
            "optimization_opportunities": [],
            "actions_executed": [],
            "actions_failed": [],
            "strategy_summary": "",
            "strategy_confidence": 0.0,
            "health_scores": {},
            "session_id": "",
            "performance_snapshot": {},
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [],
        }
        result = await agent._node_report(state)
        assert result["report_generated_at"] != ""
        assert result["report_summary"] != ""

    @pytest.mark.asyncio
    async def test_report_includes_all_sections(self):
        agent = self._make_agent()
        state = {
            "vertical_id": "test",
            "detected_issues": [
                {"agent_id": "a1", "severity": "critical", "details": {"message": "Crash"}},
            ],
            "optimization_opportunities": [
                {"potential_impact": "high", "description": "Scale up"},
            ],
            "actions_executed": [
                {"category": "healing", "action": "Fixed", "status": "success"},
            ],
            "actions_failed": [
                {"category": "budget", "error": "Timeout"},
            ],
            "strategy_summary": "We need to fix things",
            "strategy_confidence": 0.75,
            "health_scores": {"a1": 0.3},
            "session_id": "",
            "performance_snapshot": {},
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [],
        }
        result = await agent._node_report(state)
        report = result["report_summary"]
        assert "Issues Detected" in report
        assert "Opportunities" in report
        assert "Actions Taken" in report
        assert "Actions Failed" in report
        assert "Health Scores" in report
        assert "Strategy Summary" in report


class TestNodeExecuteStrategy:
    """Test the _node_execute_strategy logic."""

    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.db = MockDB()
        agent.llm = None
        agent.embedder = None
        agent._hive = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_execute_healing_action_uses_update_config(self):
        """Verify healing actions use 'update_config' action type (not 'config_fix')."""
        agent = self._make_agent()
        state = {
            "session_id": "s-1",
            "healing_actions": [
                {
                    "category": "healing",
                    "action": "Fix error threshold",
                    "target": "test_agent",
                    "parameter": "max_consecutive_errors",
                    "new_value": 10,
                    "reasoning": "Increase tolerance",
                },
            ],
            "budget_actions": [],
            "experiment_proposals": [],
        }
        result = await agent._node_execute_strategy(state)
        executed = result["actions_executed"]
        # Should have executed (not skipped) — verifies "update_config" is used
        assert len(executed) >= 1
        # Status should not be "skipped" (which would happen with wrong action type)
        for ex in executed:
            if ex["category"] == "healing":
                assert ex["status"] != "skipped"

    @pytest.mark.asyncio
    async def test_execute_budget_actions_queued(self):
        agent = self._make_agent()
        state = {
            "session_id": "s-1",
            "healing_actions": [],
            "budget_actions": [
                {"category": "budget", "action": "Shift budget", "target": "campaign_a"},
            ],
            "experiment_proposals": [],
        }
        result = await agent._node_execute_strategy(state)
        executed = result["actions_executed"]
        assert any(a["status"] == "queued_for_review" for a in executed)

    @pytest.mark.asyncio
    async def test_execute_experiment_proposals(self):
        agent = self._make_agent()
        state = {
            "session_id": "s-1",
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [
                {
                    "category": "strategy",
                    "action": "Launch test",
                    "experiment_spec": {
                        "name": "Subject Line Test",
                        "agent_id": "outreach",
                    },
                },
            ],
        }
        result = await agent._node_execute_strategy(state)
        executed = result["actions_executed"]
        assert any(a["status"] == "proposed" for a in executed)

    @pytest.mark.asyncio
    async def test_execute_empty_actions(self):
        agent = self._make_agent()
        state = {
            "session_id": "s-1",
            "healing_actions": [],
            "budget_actions": [],
            "experiment_proposals": [],
        }
        result = await agent._node_execute_strategy(state)
        assert result["actions_executed"] == []
        assert result["actions_failed"] == []


# ── Session Management ───────────────────────────────────────


class TestSessionManagement:
    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.db = MockDB()
        return agent

    def test_create_session_returns_id(self):
        agent = self._make_agent()
        session_id = agent._create_session("test_vertical", "full_analysis")
        assert session_id.startswith("mock-")

    def test_complete_session_noop_empty_id(self):
        agent = self._make_agent()
        # Should not raise
        agent._complete_session("", status="completed")

    def test_complete_session_with_data(self):
        agent = self._make_agent()
        session_id = agent._create_session("test", "healing")
        # Should not raise
        agent._complete_session(
            session_id,
            status="completed",
            metrics={"agents": []},
            issues=[{"agent_id": "a1"}],
            strategy={"summary": "All good"},
            actions=[],
        )


# ── Subsystem Factory Tests ──────────────────────────────────


class TestSubsystemFactories:
    def _make_agent(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.db = MockDB()
        agent.llm = None
        agent.embedder = None
        agent._hive = MagicMock()
        return agent

    def test_get_healer(self):
        from core.optimization.healer import SelfHealer
        agent = self._make_agent()
        healer = agent._get_healer()
        assert isinstance(healer, SelfHealer)

    def test_get_budget_manager(self):
        from core.optimization.budget_manager import BudgetManager
        agent = self._make_agent()
        mgr = agent._get_budget_manager()
        assert isinstance(mgr, BudgetManager)

    def test_get_strategist(self):
        from core.genesis.strategist import Strategist
        agent = self._make_agent()
        strat = agent._get_strategist()
        assert isinstance(strat, Strategist)


# ── Prepare Initial State ────────────────────────────────────


class TestPrepareInitialState:
    def test_default_state_fields(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        mock_config.agent_id = "autopilot_v1"
        mock_config.vertical_id = "test"
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        state = agent._prepare_initial_state(
            task={"mode": "healing"},
            run_id="r-1",
        )
        assert state["session_type"] == "healing"
        assert state["strategy_confidence"] == 0.0
        assert state["actions_approved"] is False
        assert state["detected_issues"] == []
        assert state["healing_actions"] == []

    def test_default_session_type_full(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        mock_config.agent_id = "autopilot_v1"
        mock_config.vertical_id = "test"
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config

        state = agent._prepare_initial_state(task={}, run_id="r-2")
        assert state["session_type"] == "full_analysis"


# ── Knowledge Writing ────────────────────────────────────────


class TestKnowledgeWriting:
    @pytest.mark.asyncio
    async def test_write_knowledge_skips_empty(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.store_insight = MagicMock()

        await agent.write_knowledge({
            "detected_issues": [],
            "actions_executed": [],
        })
        agent.store_insight.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_knowledge_calls_store(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        mock_config = MagicMock()
        mock_config.params = {}
        agent = AutopilotAgent.__new__(AutopilotAgent)
        agent.config = mock_config
        agent.store_insight = MagicMock()

        await agent.write_knowledge({
            "detected_issues": [{"agent_id": "a1"}],
            "actions_executed": [{"category": "healing"}],
            "strategy_summary": "Fixed things",
            "report_generated_at": "2025-02-09T00:00:00",
        })
        agent.store_insight.assert_called_once()
