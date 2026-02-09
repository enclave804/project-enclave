"""
Tests for SalesPipelineAgent — Phase 16: Sales Pipeline.

Covers:
    - SalesPipelineAgentState TypedDict
    - SalesPipelineAgent registration, construction, state class
    - Initial state preparation
    - Constants (MODES, PIPELINE_STAGES, STAGE_ORDER, thresholds)
    - All 6 nodes: scan_pipeline, analyze_deals, recommend_actions,
      human_review, execute_actions, report
    - Graph construction and routing
    - System prompt
    - YAML config (pipeline.yaml)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
# State Tests
# ══════════════════════════════════════════════════════════════════════


class TestSalesPipelineState:
    """Tests for SalesPipelineAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import SalesPipelineAgentState
        assert SalesPipelineAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import SalesPipelineAgentState
        state: SalesPipelineAgentState = {
            "agent_id": "pipeline_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "pipeline_v1"

    def test_create_full(self):
        from core.agents.state import SalesPipelineAgentState
        state: SalesPipelineAgentState = {
            "agent_id": "pipeline_v1",
            "vertical_id": "enclave_guard",
            "opportunities": [{"id": "opp_1", "stage": "qualified"}],
            "stage_metrics": {"qualified": {"count": 1, "value_cents": 150000}},
            "total_pipeline_value": 1500.0,
            "stalled_deals": [],
            "at_risk_deals": [],
            "hot_deals": [{"id": "opp_1"}],
            "recommended_actions": [],
            "actions_approved": False,
            "actions_executed": [],
            "actions_failed": [],
            "deals_moved": 0,
            "deals_won": 0,
            "deals_lost": 0,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_pipeline_value"] == 1500.0
        assert len(state["opportunities"]) == 1
        assert len(state["hot_deals"]) == 1

    def test_action_tracking_fields(self):
        from core.agents.state import SalesPipelineAgentState
        state: SalesPipelineAgentState = {
            "deals_moved": 3,
            "deals_won": 1,
            "deals_lost": 1,
            "actions_executed": [
                {"deal_id": "opp_1", "action": "advance_stage"},
            ],
            "actions_failed": [],
        }
        assert state["deals_moved"] == 3
        assert state["deals_won"] == 1
        assert len(state["actions_executed"]) == 1

    def test_stalled_deals_field(self):
        from core.agents.state import SalesPipelineAgentState
        state: SalesPipelineAgentState = {
            "stalled_deals": [
                {"id": "opp_1", "days_inactive": 10, "stage": "proposal"},
            ],
            "at_risk_deals": [
                {"id": "opp_2", "days_inactive": 20, "stage": "qualified"},
            ],
        }
        assert state["stalled_deals"][0]["days_inactive"] == 10
        assert state["at_risk_deals"][0]["days_inactive"] == 20


# ══════════════════════════════════════════════════════════════════════
# Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestSalesPipelineAgent:
    """Tests for SalesPipelineAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a SalesPipelineAgent with mocked dependencies."""
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="pipeline_v1",
            agent_type="sales_pipeline",
            name="Sales Pipeline Agent",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "stale_days": 7,
                "at_risk_days": 14,
            },
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return SalesPipelineAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "sales_pipeline" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        assert SalesPipelineAgent.agent_type == "sales_pipeline"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import SalesPipelineAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is SalesPipelineAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "SalesPipelineAgent" in repr_str
        assert "pipeline_v1" in repr_str

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "full_scan"}, "run-123"
        )
        assert state["opportunities"] == []
        assert state["stage_metrics"] == {}
        assert state["total_pipeline_value"] == 0.0
        assert state["stalled_deals"] == []
        assert state["at_risk_deals"] == []
        assert state["hot_deals"] == []
        assert state["recommended_actions"] == []
        assert state["actions_approved"] is False
        assert state["actions_executed"] == []
        assert state["actions_failed"] == []
        assert state["deals_moved"] == 0
        assert state["deals_won"] == 0
        assert state["deals_lost"] == 0
        assert state["report_summary"] == ""

    # ─── Constants ──────────────────────────────────────────────────

    def test_constants_modes(self):
        from core.agents.implementations import pipeline_agent
        assert "full_scan" in pipeline_agent.MODES
        assert "stalled_check" in pipeline_agent.MODES
        assert "forecast" in pipeline_agent.MODES
        assert "stage_update" in pipeline_agent.MODES

    def test_constants_pipeline_stages(self):
        from core.agents.implementations import pipeline_agent
        stages = pipeline_agent.PIPELINE_STAGES
        assert stages == [
            "prospect", "qualified", "proposal",
            "negotiation", "closed_won", "closed_lost",
        ]

    def test_constants_stage_order(self):
        from core.agents.implementations import pipeline_agent
        order = pipeline_agent.STAGE_ORDER
        assert order["prospect"] == 0
        assert order["closed_won"] == 4
        assert order["closed_lost"] == 5

    def test_constants_default_thresholds(self):
        from core.agents.implementations import pipeline_agent
        assert pipeline_agent.DEFAULT_STALE_DAYS == 7
        assert pipeline_agent.DEFAULT_AT_RISK_DAYS == 14

    def test_system_prompt_template(self):
        from core.agents.implementations import pipeline_agent
        prompt = pipeline_agent.PIPELINE_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{stale_days}" in prompt
        assert "{at_risk_days}" in prompt

    # ─── Node 1: Scan Pipeline ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_pipeline_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        result = await agent._node_scan_pipeline(state)

        assert result["current_node"] == "scan_pipeline"
        assert result["opportunities"] == []
        assert result["total_pipeline_value"] == 0.0
        assert isinstance(result["stage_metrics"], dict)

    @pytest.mark.asyncio
    async def test_node_scan_pipeline_with_deals(self):
        agent = self._make_agent()
        now = datetime.now(timezone.utc)
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "opp_1",
                "stage": "qualified",
                "value_cents": 150000,
                "created_at": (now - timedelta(days=5)).isoformat(),
            },
            {
                "id": "opp_2",
                "stage": "proposal",
                "value_cents": 250000,
                "created_at": (now - timedelta(days=10)).isoformat(),
            },
            {
                "id": "opp_3",
                "stage": "closed_won",
                "value_cents": 100000,
                "created_at": (now - timedelta(days=30)).isoformat(),
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        result = await agent._node_scan_pipeline(state)

        assert len(result["opportunities"]) == 3
        assert result["total_pipeline_value"] > 0
        assert "qualified" in result["stage_metrics"]
        assert result["stage_metrics"]["qualified"]["count"] == 1

    @pytest.mark.asyncio
    async def test_node_scan_pipeline_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.side_effect = Exception("DB error")

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        result = await agent._node_scan_pipeline(state)

        assert result["current_node"] == "scan_pipeline"
        assert result["opportunities"] == []

    # ─── Node 2: Analyze Deals ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_deals_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["opportunities"] = []

        result = await agent._node_analyze_deals(state)

        assert result["current_node"] == "analyze_deals"
        assert result["stalled_deals"] == []
        assert result["at_risk_deals"] == []
        assert result["hot_deals"] == []

    @pytest.mark.asyncio
    async def test_node_analyze_deals_stalled(self):
        agent = self._make_agent()
        now = datetime.now(timezone.utc)
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["opportunities"] = [
            {
                "id": "opp_1",
                "stage": "qualified",
                "value_cents": 150000,
                "updated_at": (now - timedelta(days=10)).isoformat(),
            },
        ]

        result = await agent._node_analyze_deals(state)

        assert len(result["stalled_deals"]) == 1
        assert result["stalled_deals"][0]["days_inactive"] >= 10

    @pytest.mark.asyncio
    async def test_node_analyze_deals_at_risk(self):
        agent = self._make_agent()
        now = datetime.now(timezone.utc)
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["opportunities"] = [
            {
                "id": "opp_2",
                "stage": "proposal",
                "value_cents": 200000,
                "updated_at": (now - timedelta(days=20)).isoformat(),
            },
        ]

        result = await agent._node_analyze_deals(state)

        assert len(result["at_risk_deals"]) == 1
        assert result["at_risk_deals"][0]["days_inactive"] >= 14

    @pytest.mark.asyncio
    async def test_node_analyze_deals_hot(self):
        agent = self._make_agent()
        now = datetime.now(timezone.utc)
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["opportunities"] = [
            {
                "id": "opp_hot",
                "stage": "negotiation",
                "value_cents": 500000,
                "updated_at": (now - timedelta(days=2)).isoformat(),
            },
        ]

        result = await agent._node_analyze_deals(state)

        assert len(result["hot_deals"]) == 1

    @pytest.mark.asyncio
    async def test_node_analyze_deals_skips_closed(self):
        agent = self._make_agent()
        now = datetime.now(timezone.utc)
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["opportunities"] = [
            {
                "id": "opp_won",
                "stage": "closed_won",
                "value_cents": 100000,
                "updated_at": (now - timedelta(days=30)).isoformat(),
            },
            {
                "id": "opp_lost",
                "stage": "closed_lost",
                "value_cents": 50000,
                "updated_at": (now - timedelta(days=60)).isoformat(),
            },
        ]

        result = await agent._node_analyze_deals(state)

        assert result["stalled_deals"] == []
        assert result["at_risk_deals"] == []
        assert result["hot_deals"] == []

    # ─── Node 3: Recommend Actions ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_recommend_actions_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["stalled_deals"] = []
        state["at_risk_deals"] = []
        state["hot_deals"] = []
        state["stage_metrics"] = {}

        result = await agent._node_recommend_actions(state)

        assert result["current_node"] == "recommend_actions"
        assert result["recommended_actions"] == []

    @pytest.mark.asyncio
    async def test_node_recommend_actions_at_risk(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["at_risk_deals"] = [
            {
                "id": "opp_1",
                "company_id": "comp_1",
                "stage": "qualified",
                "days_inactive": 20,
                "value_cents": 150000,
            },
        ]
        state["stalled_deals"] = []
        state["hot_deals"] = []
        state["stage_metrics"] = {}

        result = await agent._node_recommend_actions(state)

        assert len(result["recommended_actions"]) == 1
        rec = result["recommended_actions"][0]
        assert rec["recommended_action"] == "send_followup"
        assert rec["priority"] == "high"

    @pytest.mark.asyncio
    async def test_node_recommend_actions_stalled_qualified(self):
        """Stalled qualified deal should get schedule_meeting recommendation."""
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["stalled_deals"] = [
            {
                "id": "opp_q",
                "company_id": "comp_q",
                "stage": "qualified",
                "days_inactive": 10,
                "value_cents": 100000,
            },
        ]
        state["at_risk_deals"] = []
        state["hot_deals"] = []
        state["stage_metrics"] = {}

        result = await agent._node_recommend_actions(state)

        assert len(result["recommended_actions"]) == 1
        assert result["recommended_actions"][0]["recommended_action"] == "schedule_meeting"

    @pytest.mark.asyncio
    async def test_node_recommend_actions_hot_negotiation(self):
        """Hot deal in negotiation should get advance_stage recommendation."""
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["hot_deals"] = [
            {
                "id": "opp_h",
                "company_id": "comp_h",
                "stage": "negotiation",
                "days_inactive": 2,
                "value_cents": 500000,
            },
        ]
        state["stalled_deals"] = []
        state["at_risk_deals"] = []
        state["stage_metrics"] = {}

        result = await agent._node_recommend_actions(state)

        assert len(result["recommended_actions"]) == 1
        rec = result["recommended_actions"][0]
        assert rec["recommended_action"] == "advance_stage"
        assert rec["next_stage"] == "closed_won"

    # ─── Node 4: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "recommended_actions": [
                {"deal_id": "opp_1", "recommended_action": "advance_stage"},
            ],
        }
        result = await agent._node_human_review(state)

        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Execute Actions ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_execute_actions_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = []

        result = await agent._node_execute_actions(state)

        assert result["current_node"] == "execute_actions"
        assert result["actions_executed"] == []
        assert result["deals_moved"] == 0

    @pytest.mark.asyncio
    async def test_node_execute_actions_advance_stage(self):
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = [
            {
                "deal_id": "opp_1",
                "recommended_action": "advance_stage",
                "next_stage": "proposal",
                "current_stage": "qualified",
            },
        ]

        result = await agent._node_execute_actions(state)

        assert result["deals_moved"] == 1
        assert len(result["actions_executed"]) == 1
        assert result["actions_approved"] is True
        assert result["knowledge_written"] is True

    @pytest.mark.asyncio
    async def test_node_execute_actions_close_won(self):
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = [
            {
                "deal_id": "opp_h",
                "recommended_action": "advance_stage",
                "next_stage": "closed_won",
                "current_stage": "negotiation",
            },
        ]

        result = await agent._node_execute_actions(state)

        assert result["deals_won"] == 1
        assert result["deals_moved"] == 1

    @pytest.mark.asyncio
    async def test_node_execute_actions_close_lost(self):
        agent = self._make_agent()
        mock_execute = MagicMock()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = [
            {
                "deal_id": "opp_l",
                "recommended_action": "advance_stage",
                "next_stage": "closed_lost",
                "reasoning": "No budget",
            },
        ]

        result = await agent._node_execute_actions(state)

        assert result["deals_lost"] == 1
        assert result["deals_moved"] == 1

    @pytest.mark.asyncio
    async def test_node_execute_actions_send_followup(self):
        """Non-stage-change actions dispatch and execute without DB update."""
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = [
            {
                "deal_id": "opp_1",
                "recommended_action": "send_followup",
                "current_stage": "qualified",
            },
        ]

        result = await agent._node_execute_actions(state)

        assert len(result["actions_executed"]) == 1
        assert result["deals_moved"] == 0

    @pytest.mark.asyncio
    async def test_node_execute_actions_db_error(self):
        """DB errors for stage transitions are caught and logged as failed."""
        agent = self._make_agent()
        agent.db.client.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB write fail")

        state = agent._prepare_initial_state({"mode": "full_scan"}, "run-1")
        state["recommended_actions"] = [
            {
                "deal_id": "opp_fail",
                "recommended_action": "advance_stage",
                "next_stage": "proposal",
            },
        ]

        result = await agent._node_execute_actions(state)

        assert len(result["actions_failed"]) == 1
        assert result["actions_failed"][0]["status"] == "failed"

    # ─── Node 6: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "total_pipeline_value": 5000.0,
            "opportunities": [{"id": "opp_1"}],
            "stage_metrics": {
                "qualified": {"count": 1, "value_cents": 150000, "avg_age_days": 5.0},
            },
            "stalled_deals": [],
            "at_risk_deals": [],
            "hot_deals": [{"id": "opp_1"}],
            "recommended_actions": [{"deal_id": "opp_1"}],
            "actions_executed": [{"deal_id": "opp_1", "status": "executed"}],
            "actions_failed": [],
            "deals_moved": 1,
            "deals_won": 0,
            "deals_lost": 0,
        }
        result = await agent._node_report(state)

        assert result["current_node"] == "report"
        assert result["report_summary"] != ""
        assert result["report_generated_at"] != ""
        assert "Sales Pipeline Report" in result["report_summary"]
        assert "Pipeline Overview" in result["report_summary"]

    @pytest.mark.asyncio
    async def test_node_report_empty_pipeline(self):
        agent = self._make_agent()
        state = {
            "total_pipeline_value": 0.0,
            "opportunities": [],
            "stage_metrics": {},
            "stalled_deals": [],
            "at_risk_deals": [],
            "hot_deals": [],
            "recommended_actions": [],
            "actions_executed": [],
            "actions_failed": [],
            "deals_moved": 0,
            "deals_won": 0,
            "deals_lost": 0,
        }
        result = await agent._node_report(state)

        assert result["current_node"] == "report"
        assert "Sales Pipeline Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        assert SalesPipelineAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        assert SalesPipelineAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        assert SalesPipelineAgent._route_after_review({}) == "approved"

    # ─── System Prompt ───────────────────────────────────────────────

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "pipeline" in prompt.lower()

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestPipelineYAML:
    """Tests for pipeline.yaml configuration."""

    def _load_config(self):
        import yaml
        config_path = (
            Path(__file__).parent.parent.parent
            / "verticals"
            / "enclave_guard"
            / "agents"
            / "pipeline.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_agent_id(self):
        config = self._load_config()
        assert config["agent_id"] == "pipeline_v1"

    def test_agent_type(self):
        config = self._load_config()
        assert config["agent_type"] == "sales_pipeline"

    def test_enabled(self):
        config = self._load_config()
        assert config["enabled"] is True

    def test_model_provider(self):
        config = self._load_config()
        assert config["model"]["provider"] == "anthropic"

    def test_human_gates_enabled(self):
        config = self._load_config()
        assert config["human_gates"]["enabled"] is True

    def test_schedule_trigger(self):
        config = self._load_config()
        assert config["schedule"]["trigger"] == "scheduled"

    def test_params_stale_days(self):
        config = self._load_config()
        assert config["params"]["stale_days"] == 7

    def test_params_at_risk_days(self):
        config = self._load_config()
        assert config["params"]["at_risk_days"] == 14

    def test_params_company_name(self):
        config = self._load_config()
        assert config["params"]["company_name"] == "Enclave Guard"
