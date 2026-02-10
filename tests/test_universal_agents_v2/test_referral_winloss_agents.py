"""
Tests for Universal Business Agents v2 — Phase 21 (Part 1).

Covers 2 cross-vertical business operations agents:
    1. ReferralAgent (referral)
    2. WinLossAgent (win_loss)

Each agent tests:
    - State TypedDict import and creation
    - Agent registration, construction, state class
    - Initial state preparation
    - Module-level constants and system prompts
    - All graph nodes (async, mocked DB/LLM)
    - Graph construction and routing
    - __repr__ and write_knowledge
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
#  1.  ReferralAgent
# ======================================================================


class TestReferralAgentState:
    """Tests for ReferralAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ReferralAgentState
        assert ReferralAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ReferralAgentState
        state: ReferralAgentState = {
            "agent_id": "referral_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "referral_v1"

    def test_create_full(self):
        from core.agents.state import ReferralAgentState
        state: ReferralAgentState = {
            "agent_id": "referral_v1",
            "vertical_id": "enclave_guard",
            "happy_clients": [],
            "referral_candidates": [],
            "referral_requests": [],
            "requests_approved": False,
            "active_referrals": [],
            "total_referrals": 0,
            "converted_referrals": 0,
            "total_commission_cents": 0,
            "requests_sent": 0,
            "referrals_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_referrals"] == 0
        assert state["requests_approved"] is False
        assert state["referrals_saved"] is False

    def test_client_identification_fields(self):
        from core.agents.state import ReferralAgentState
        state: ReferralAgentState = {
            "happy_clients": [{"client_id": "c1", "nps_score": 10}],
            "referral_candidates": [{"client_id": "c1"}],
        }
        assert len(state["happy_clients"]) == 1
        assert state["happy_clients"][0]["nps_score"] == 10

    def test_tracking_fields(self):
        from core.agents.state import ReferralAgentState
        state: ReferralAgentState = {
            "active_referrals": [{"id": "ref_1", "status": "submitted"}],
            "total_referrals": 5,
            "converted_referrals": 2,
            "total_commission_cents": 50000,
            "requests_sent": 3,
        }
        assert state["total_referrals"] == 5
        assert state["converted_referrals"] == 2
        assert state["total_commission_cents"] == 50000


class TestReferralAgent:
    """Tests for ReferralAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a ReferralAgent with mocked dependencies."""
        from core.agents.implementations.referral_agent import ReferralAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="referral_v1",
            agent_type="referral",
            name="Referral Agent",
            vertical_id="enclave_guard",
            params={"company_name": "Test Corp"},
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

        return ReferralAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # --- Registration & Construction ------------------------------------

    def test_registration(self):
        from core.agents.implementations.referral_agent import ReferralAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "referral" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.referral_agent import ReferralAgent
        assert ReferralAgent.agent_type == "referral"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ReferralAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ReferralAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ReferralAgent" in r
        assert "referral_v1" in r

    # --- Initial State ---------------------------------------------------

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"min_nps": 9, "campaign_name": "Q1 Drive"}, "run-1"
        )
        assert state["happy_clients"] == []
        assert state["referral_candidates"] == []
        assert state["referral_requests"] == []
        assert state["requests_approved"] is False
        assert state["active_referrals"] == []
        assert state["total_referrals"] == 0
        assert state["converted_referrals"] == 0
        assert state["total_commission_cents"] == 0
        assert state["requests_sent"] == 0
        assert state["referrals_saved"] is False
        assert state["report_summary"] == ""

    # --- Constants -------------------------------------------------------

    def test_min_nps_for_referral(self):
        from core.agents.implementations import referral_agent
        assert referral_agent.MIN_NPS_FOR_REFERRAL == 9

    def test_default_commission_percent(self):
        from core.agents.implementations import referral_agent
        assert referral_agent.DEFAULT_COMMISSION_PERCENT == 10.0

    def test_referral_statuses(self):
        from core.agents.implementations import referral_agent
        statuses = referral_agent.REFERRAL_STATUSES
        assert "submitted" in statuses
        assert "contacted" in statuses
        assert "qualified" in statuses
        assert "converted" in statuses
        assert "lost" in statuses
        assert "expired" in statuses

    def test_referral_ask_prompt(self):
        from core.agents.implementations import referral_agent
        prompt = referral_agent.REFERRAL_ASK_PROMPT
        assert "{client_json}" in prompt
        assert "{vertical_id}" in prompt
        assert "{services}" in prompt
        assert "{relationship_months}" in prompt
        assert "{commission_percent}" in prompt

    # --- Node 1: Identify Candidates ------------------------------------

    @pytest.mark.asyncio
    async def test_node_identify_candidates_no_clients(self):
        agent = self._make_agent()
        mock_nps = MagicMock()
        mock_nps.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_nps

        state = agent._prepare_initial_state({"min_nps": 9}, "run-1")
        result = await agent._node_identify_candidates(state)
        assert result["current_node"] == "identify_candidates"
        assert result["happy_clients"] == []
        assert result["referral_candidates"] == []

    @pytest.mark.asyncio
    async def test_node_identify_candidates_with_clients(self):
        agent = self._make_agent()
        mock_nps = MagicMock()
        mock_nps.data = [
            {"client_id": "c1", "nps_score": 10, "comment": "Great!", "created_at": "2024-01-01"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_nps

        # Company enrichment
        mock_companies = MagicMock()
        mock_companies.data = [
            {"id": "c1", "name": "Acme", "domain": "acme.com", "primary_email": "j@acme.com", "primary_contact": "Jane", "industry": "tech"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.return_value = mock_companies

        # Active referrals check
        mock_active = MagicMock()
        mock_active.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.return_value = mock_active

        state = agent._prepare_initial_state({"min_nps": 9}, "run-1")
        result = await agent._node_identify_candidates(state)
        assert result["current_node"] == "identify_candidates"
        assert len(result["happy_clients"]) >= 1

    @pytest.mark.asyncio
    async def test_node_identify_candidates_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.gte.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({"min_nps": 9}, "run-1")
        result = await agent._node_identify_candidates(state)
        assert result["current_node"] == "identify_candidates"
        assert result["happy_clients"] == []

    # --- Node 2: Generate Referral Asks ----------------------------------

    @pytest.mark.asyncio
    async def test_node_generate_referral_asks_no_candidates(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["referral_candidates"] = []
        result = await agent._node_generate_referral_asks(state)
        assert result["current_node"] == "generate_referral_asks"
        assert result["referral_requests"] == []

    @pytest.mark.asyncio
    async def test_node_generate_referral_asks_llm_success(self):
        agent = self._make_agent()
        asks = [
            {
                "client_id": "c1",
                "client_name": "Acme Corp",
                "email": "jane@acme.com",
                "ask_message": "Hi Jane, we loved working with Acme...",
                "reasoning": "NPS 10, long relationship",
                "ask_type": "email",
                "confidence": 0.9,
            },
        ]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(asks)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["referral_candidates"] = [
            {
                "client_id": "c1",
                "company_name": "Acme Corp",
                "contact_name": "Jane",
                "contact_email": "jane@acme.com",
                "nps_score": 10,
                "response_date": "2024-01-01T00:00:00Z",
            },
        ]
        result = await agent._node_generate_referral_asks(state)
        assert result["current_node"] == "generate_referral_asks"
        assert len(result["referral_requests"]) == 1
        assert result["referral_requests"][0]["client_id"] == "c1"

    @pytest.mark.asyncio
    async def test_node_generate_referral_asks_llm_error_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["referral_candidates"] = [
            {
                "client_id": "c2",
                "company_name": "Beta Inc",
                "contact_name": "Bob",
                "contact_email": "bob@beta.com",
                "nps_score": 9,
                "response_date": "",
            },
        ]
        result = await agent._node_generate_referral_asks(state)
        assert result["current_node"] == "generate_referral_asks"
        # Fallback should NOT produce requests since the outer exception is caught
        # and no more batches processed

    # --- Node 3: Human Review -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "referral_requests": [{"client_id": "c1"}],
            "referral_candidates": [{"client_id": "c1"}],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # --- Node 4: Execute Referrals --------------------------------------

    @pytest.mark.asyncio
    async def test_node_execute_referrals_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "ref_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["referral_requests"] = [
            {
                "client_id": "c1",
                "client_name": "Acme Corp",
                "email": "jane@acme.com",
                "ask_message": "Hi Jane...",
                "ask_type": "email",
                "confidence": 0.9,
            },
        ]
        result = await agent._node_execute_referrals(state)
        assert result["current_node"] == "execute_referrals"
        assert len(result["active_referrals"]) == 1
        assert result["total_referrals"] == 1
        assert result["requests_sent"] == 1
        assert result["requests_approved"] is True
        assert result["referrals_saved"] is True

    @pytest.mark.asyncio
    async def test_node_execute_referrals_empty_requests(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["referral_requests"] = []
        result = await agent._node_execute_referrals(state)
        assert result["current_node"] == "execute_referrals"
        assert result["active_referrals"] == []
        assert result["total_referrals"] == 0
        assert result["requests_sent"] == 0
        assert result["referrals_saved"] is False

    @pytest.mark.asyncio
    async def test_node_execute_referrals_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({}, "run-1")
        state["referral_requests"] = [
            {"client_id": "c1", "client_name": "Acme", "email": "j@a.com",
             "ask_message": "Hi", "ask_type": "email", "confidence": 0.8},
        ]
        result = await agent._node_execute_referrals(state)
        assert result["active_referrals"] == []
        assert result["referrals_saved"] is False

    # --- Node 5: Report -------------------------------------------------

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["happy_clients"] = [
            {"client_id": "c1", "company_name": "Acme", "nps_score": 10, "contact_name": "Jane"},
        ]
        state["referral_candidates"] = [{"client_id": "c1"}]
        state["referral_requests"] = [{"client_name": "Acme", "reasoning": "NPS 10"}]
        state["active_referrals"] = [{"id": "ref_1"}]
        state["requests_sent"] = 1
        state["human_approval_status"] = "approved"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Referral Campaign Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_clients(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["happy_clients"] = []
        state["referral_candidates"] = []
        state["referral_requests"] = []
        state["active_referrals"] = []
        state["requests_sent"] = 0
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["report_generated_at"] != ""

    # --- Routing ---------------------------------------------------------

    def test_route_approved(self):
        from core.agents.implementations.referral_agent import ReferralAgent
        state = {"human_approval_status": "approved"}
        assert ReferralAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.referral_agent import ReferralAgent
        state = {"human_approval_status": "rejected"}
        assert ReferralAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.referral_agent import ReferralAgent
        state = {}
        assert ReferralAgent._route_after_review(state) == "approved"

    # --- write_knowledge -------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"report_summary": "test"})
        assert result is None


# ======================================================================
#  2.  WinLossAgent
# ======================================================================


class TestWinLossAgentState:
    """Tests for WinLossAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import WinLossAgentState
        assert WinLossAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import WinLossAgentState
        state: WinLossAgentState = {
            "agent_id": "win_loss_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "win_loss_v1"

    def test_create_full(self):
        from core.agents.state import WinLossAgentState
        state: WinLossAgentState = {
            "agent_id": "win_loss_v1",
            "vertical_id": "enclave_guard",
            "recent_deals": [],
            "analysis_period_days": 90,
            "total_won": 0,
            "total_lost": 0,
            "win_factors": [],
            "loss_factors": [],
            "competitive_gaps": [],
            "avg_sales_cycle_days": 0.0,
            "recommendations": [],
            "recommendations_count": 0,
            "analyses_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["analysis_period_days"] == 90
        assert state["avg_sales_cycle_days"] == 0.0
        assert state["analyses_saved"] is False

    def test_deal_data_fields(self):
        from core.agents.state import WinLossAgentState
        state: WinLossAgentState = {
            "recent_deals": [
                {"stage": "closed_won", "value_cents": 50000},
                {"stage": "closed_lost", "value_cents": 30000},
            ],
            "total_won": 1,
            "total_lost": 1,
        }
        assert len(state["recent_deals"]) == 2
        assert state["total_won"] == 1

    def test_analysis_fields(self):
        from core.agents.state import WinLossAgentState
        state: WinLossAgentState = {
            "win_factors": [{"factor": "pricing", "impact": "high"}],
            "loss_factors": [{"factor": "slow response", "impact": "medium"}],
            "competitive_gaps": [{"competitor": "Rival", "gap": "Feature X"}],
            "recommendations": [{"recommendation": "Improve response time"}],
            "recommendations_count": 1,
        }
        assert len(state["win_factors"]) == 1
        assert state["recommendations_count"] == 1


class TestWinLossAgent:
    """Tests for WinLossAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a WinLossAgent with mocked dependencies."""
        from core.agents.implementations.win_loss_agent import WinLossAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="win_loss_v1",
            agent_type="win_loss",
            name="Win Loss Agent",
            vertical_id="enclave_guard",
            params={"company_name": "Test Corp"},
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

        return WinLossAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # --- Registration & Construction ------------------------------------

    def test_registration(self):
        from core.agents.implementations.win_loss_agent import WinLossAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "win_loss" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.win_loss_agent import WinLossAgent
        assert WinLossAgent.agent_type == "win_loss"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import WinLossAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is WinLossAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "WinLossAgent" in r
        assert "win_loss_v1" in r

    # --- Initial State ---------------------------------------------------

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"analysis_period_days": 60}, "run-1"
        )
        assert state["recent_deals"] == []
        assert state["analysis_period_days"] == 90  # Default, not task override
        assert state["total_won"] == 0
        assert state["total_lost"] == 0
        assert state["win_factors"] == []
        assert state["loss_factors"] == []
        assert state["competitive_gaps"] == []
        assert state["avg_sales_cycle_days"] == 0.0
        assert state["recommendations"] == []
        assert state["recommendations_count"] == 0
        assert state["analyses_saved"] is False
        assert state["report_summary"] == ""

    # --- Constants -------------------------------------------------------

    def test_deal_factor_categories(self):
        from core.agents.implementations import win_loss_agent
        cats = win_loss_agent.DEAL_FACTOR_CATEGORIES
        assert "pricing" in cats
        assert "technical_capability" in cats
        assert "response_time" in cats
        assert "trust" in cats
        assert "competition" in cats
        assert "timing" in cats
        assert "budget" in cats
        assert "champion_strength" in cats

    def test_analysis_period_days_constant(self):
        from core.agents.implementations import win_loss_agent
        assert win_loss_agent.ANALYSIS_PERIOD_DAYS == 90

    def test_win_loss_analysis_prompt(self):
        from core.agents.implementations import win_loss_agent
        prompt = win_loss_agent.WIN_LOSS_ANALYSIS_PROMPT
        assert "{won_count}" in prompt
        assert "{lost_count}" in prompt
        assert "{won_deals_json}" in prompt
        assert "{lost_deals_json}" in prompt
        assert "{period_days}" in prompt
        assert "{factor_categories}" in prompt

    def test_recommendation_prompt(self):
        from core.agents.implementations import win_loss_agent
        prompt = win_loss_agent.RECOMMENDATION_PROMPT
        assert "{win_factors_json}" in prompt
        assert "{loss_factors_json}" in prompt
        assert "{competitive_gaps_json}" in prompt
        assert "{win_rate_pct}" in prompt
        assert "{avg_cycle_days}" in prompt

    # --- Node 1: Load Deals ----------------------------------------------

    @pytest.mark.asyncio
    async def test_node_load_deals_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"analysis_period_days": 90}, "run-1"
        )
        result = await agent._node_load_deals(state)
        assert result["current_node"] == "load_deals"
        assert result["recent_deals"] == []
        assert result["total_won"] == 0
        assert result["total_lost"] == 0
        assert result["avg_sales_cycle_days"] == 0.0

    @pytest.mark.asyncio
    async def test_node_load_deals_with_data(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"stage": "closed_won", "value_cents": 50000,
             "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-02-01T00:00:00Z"},
            {"stage": "closed_lost", "value_cents": 30000,
             "created_at": "2024-01-15T00:00:00Z", "updated_at": "2024-02-15T00:00:00Z"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"analysis_period_days": 90}, "run-1"
        )
        result = await agent._node_load_deals(state)
        assert result["current_node"] == "load_deals"
        assert len(result["recent_deals"]) == 2
        assert result["total_won"] == 1
        assert result["total_lost"] == 1
        assert result["analysis_period_days"] == 90
        assert result["avg_sales_cycle_days"] > 0

    @pytest.mark.asyncio
    async def test_node_load_deals_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"analysis_period_days": 90}, "run-1"
        )
        result = await agent._node_load_deals(state)
        assert result["current_node"] == "load_deals"
        assert result["recent_deals"] == []

    # --- Node 2: Analyze Patterns ----------------------------------------

    @pytest.mark.asyncio
    async def test_node_analyze_patterns_no_deals(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["recent_deals"] = []
        result = await agent._node_analyze_patterns(state)
        assert result["current_node"] == "analyze_patterns"
        assert result["win_factors"] == []
        assert result["loss_factors"] == []
        assert result["competitive_gaps"] == []

    @pytest.mark.asyncio
    async def test_node_analyze_patterns_llm_success(self):
        agent = self._make_agent()
        analysis = {
            "win_factors": [
                {"factor": "Strong pricing", "category": "pricing",
                 "frequency": 0.7, "impact": "high", "example_deals": ["Deal A"]},
            ],
            "loss_factors": [
                {"factor": "Slow response time", "category": "response_time",
                 "frequency": 0.5, "impact": "medium", "example_deals": ["Deal B"]},
            ],
            "competitive_gaps": [
                {"competitor": "Rival Corp", "gap": "Feature X missing",
                 "frequency": 0.3, "deals_affected": 2},
            ],
            "pattern_summary": "Strong pricing wins deals.",
        }
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(analysis)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["recent_deals"] = [
            {"stage": "closed_won", "value_cents": 50000, "industry": "tech"},
            {"stage": "closed_lost", "value_cents": 30000, "loss_reason": "price"},
        ]
        state["total_won"] = 1
        state["total_lost"] = 1
        result = await agent._node_analyze_patterns(state)
        assert result["current_node"] == "analyze_patterns"
        assert len(result["win_factors"]) == 1
        assert len(result["loss_factors"]) == 1
        assert len(result["competitive_gaps"]) == 1

    @pytest.mark.asyncio
    async def test_node_analyze_patterns_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["recent_deals"] = [{"stage": "closed_won"}]
        state["total_won"] = 1
        result = await agent._node_analyze_patterns(state)
        assert result["win_factors"] == []
        assert result["loss_factors"] == []
        assert result["competitive_gaps"] == []

    # --- Node 3: Generate Recommendations --------------------------------

    @pytest.mark.asyncio
    async def test_node_generate_recommendations_no_factors(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["win_factors"] = []
        state["loss_factors"] = []
        state["competitive_gaps"] = []
        state["total_won"] = 0
        state["total_lost"] = 0
        result = await agent._node_generate_recommendations(state)
        assert result["current_node"] == "generate_recommendations"
        assert result["recommendations"] == []
        assert result["recommendations_count"] == 0

    @pytest.mark.asyncio
    async def test_node_generate_recommendations_llm_success(self):
        agent = self._make_agent()
        recs = [
            {
                "recommendation": "Reduce proposal turnaround time to 24h",
                "priority": "critical",
                "expected_impact": "Win 15% more deals",
                "category": "process",
                "effort": "medium",
                "timeline_weeks": 4,
            },
        ]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(recs)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["win_factors"] = [{"factor": "Good pricing", "impact": "high"}]
        state["loss_factors"] = [{"factor": "Slow response", "impact": "high"}]
        state["competitive_gaps"] = []
        state["total_won"] = 8
        state["total_lost"] = 2
        state["avg_sales_cycle_days"] = 30.0
        result = await agent._node_generate_recommendations(state)
        assert result["current_node"] == "generate_recommendations"
        assert len(result["recommendations"]) == 1
        assert result["recommendations_count"] == 1

    @pytest.mark.asyncio
    async def test_node_generate_recommendations_llm_error_fallback(self):
        agent = self._make_agent()
        # Create LLM response that returns invalid JSON to trigger fallback
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "not valid json"
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["win_factors"] = [{"factor": "Good pricing", "impact": "high"}]
        state["loss_factors"] = [
            {"factor": "Slow response", "impact": "high"},
            {"factor": "Lack of features", "impact": "medium"},
        ]
        state["competitive_gaps"] = []
        state["total_won"] = 5
        state["total_lost"] = 5
        state["avg_sales_cycle_days"] = 30.0
        result = await agent._node_generate_recommendations(state)
        assert result["current_node"] == "generate_recommendations"
        # Fallback generates recommendations from loss factors
        assert len(result["recommendations"]) >= 1

    # --- Node 4: Human Review -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "recommendations": [{"recommendation": "Speed up proposals"}],
            "total_won": 5,
            "total_lost": 3,
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # --- Node 5: Report --------------------------------------------------

    @pytest.mark.asyncio
    async def test_node_report_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "analysis_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["total_won"] = 8
        state["total_lost"] = 2
        state["win_factors"] = [{"factor": "Pricing", "category": "pricing", "impact": "high"}]
        state["loss_factors"] = [{"factor": "Slow", "category": "response_time", "impact": "medium"}]
        state["competitive_gaps"] = [{"competitor": "Rival", "gap": "Feature X"}]
        state["recommendations"] = [{"recommendation": "Speed up", "priority": "high"}]
        state["avg_sales_cycle_days"] = 30.0
        state["analysis_period_days"] = 90
        state["human_approval_status"] = "approved"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["analyses_saved"] is True
        assert "Win/Loss Analysis Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["total_won"] = 3
        state["total_lost"] = 7
        state["win_factors"] = []
        state["loss_factors"] = []
        state["competitive_gaps"] = []
        state["recommendations"] = []
        state["avg_sales_cycle_days"] = 0.0
        state["analysis_period_days"] = 90
        result = await agent._node_report(state)
        assert result["analyses_saved"] is False

    @pytest.mark.asyncio
    async def test_node_report_no_deals(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["total_won"] = 0
        state["total_lost"] = 0
        state["win_factors"] = []
        state["loss_factors"] = []
        state["competitive_gaps"] = []
        state["recommendations"] = []
        state["avg_sales_cycle_days"] = 0.0
        state["analysis_period_days"] = 90
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["report_generated_at"] != ""

    # --- Routing ---------------------------------------------------------

    def test_route_approved(self):
        from core.agents.implementations.win_loss_agent import WinLossAgent
        state = {"human_approval_status": "approved"}
        assert WinLossAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.win_loss_agent import WinLossAgent
        state = {"human_approval_status": "rejected"}
        assert WinLossAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.win_loss_agent import WinLossAgent
        state = {}
        assert WinLossAgent._route_after_review(state) == "approved"

    # --- write_knowledge -------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"report_summary": "test"})
        assert result is None
