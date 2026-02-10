"""
Tests for Universal Business Agents v2 — Phase 21 (Knowledge Base + Feedback).

Covers 2 cross-vertical business operations agents:
    1. KnowledgeBaseAgent (knowledge_base)
    2. FeedbackAgent (feedback)

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


# ══════════════════════════════════════════════════════════════════════
#  1.  KnowledgeBaseAgent
# ══════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseAgentState:
    """Tests for KnowledgeBaseAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import KnowledgeBaseAgentState
        assert KnowledgeBaseAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import KnowledgeBaseAgentState
        state: KnowledgeBaseAgentState = {
            "agent_id": "knowledge_base_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "knowledge_base_v1"

    def test_create_full(self):
        from core.agents.state import KnowledgeBaseAgentState
        state: KnowledgeBaseAgentState = {
            "agent_id": "knowledge_base_v1",
            "vertical_id": "enclave_guard",
            "resolved_tickets": [{"ticket_id": "t1", "subject": "Login issue"}],
            "hive_insights": [{"topic": "auth", "content": "Common auth issues"}],
            "total_sources_scanned": 15,
            "topic_clusters": [{"topic": "Login", "ticket_count": 5}],
            "identified_gaps": [{"topic": "Login", "priority": "high"}],
            "total_gaps": 1,
            "draft_articles": [{"title": "Login FAQ", "category": "troubleshooting"}],
            "articles_generated": 1,
            "articles_saved": 0,
            "articles_published": 0,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["total_sources_scanned"] == 15
        assert len(state["resolved_tickets"]) == 1
        assert state["total_gaps"] == 1

    def test_ticket_and_insight_fields(self):
        from core.agents.state import KnowledgeBaseAgentState
        state: KnowledgeBaseAgentState = {
            "resolved_tickets": [
                {"ticket_id": "t1", "subject": "Login issue", "category": "troubleshooting"},
                {"ticket_id": "t2", "subject": "Billing question", "category": "billing"},
            ],
            "hive_insights": [{"topic": "auth", "confidence": 0.9}],
            "total_sources_scanned": 3,
        }
        assert len(state["resolved_tickets"]) == 2
        assert state["hive_insights"][0]["confidence"] == 0.9

    def test_article_and_gap_fields(self):
        from core.agents.state import KnowledgeBaseAgentState
        state: KnowledgeBaseAgentState = {
            "topic_clusters": [{"topic": "Auth"}, {"topic": "Billing"}],
            "identified_gaps": [{"topic": "Auth", "gap_description": "No KB article"}],
            "draft_articles": [{"title": "Auth FAQ", "content": "..."}],
            "articles_generated": 1,
            "articles_saved": 0,
        }
        assert len(state["topic_clusters"]) == 2
        assert state["articles_generated"] == 1

    def test_report_fields(self):
        from core.agents.state import KnowledgeBaseAgentState
        state: KnowledgeBaseAgentState = {
            "report_summary": "# Knowledge Base Report\n...",
            "report_generated_at": "2024-02-01T12:00:00Z",
            "articles_published": 3,
        }
        assert "Knowledge Base" in state["report_summary"]
        assert state["articles_published"] == 3


class TestKnowledgeBaseAgent:
    """Tests for KnowledgeBaseAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a KnowledgeBaseAgent with mocked dependencies."""
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="knowledge_base_v1",
            agent_type="knowledge_base",
            name="Knowledge Base Agent",
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

        return KnowledgeBaseAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "knowledge_base" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent
        assert KnowledgeBaseAgent.agent_type == "knowledge_base"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import KnowledgeBaseAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is KnowledgeBaseAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "KnowledgeBaseAgent" in r
        assert "knowledge_base_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"lookback_days": 30}, "run-1"
        )
        assert state["resolved_tickets"] == []
        assert state["hive_insights"] == []
        assert state["total_sources_scanned"] == 0
        assert state["topic_clusters"] == []
        assert state["identified_gaps"] == []
        assert state["total_gaps"] == 0
        assert state["draft_articles"] == []
        assert state["articles_generated"] == 0
        assert state["articles_saved"] == 0
        assert state["articles_published"] == 0
        assert state["report_summary"] == ""
        assert state["report_generated_at"] == ""

    def test_prepare_initial_state_common_keys(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"lookback_days": 30}, "run-1"
        )
        assert state["agent_id"] == "knowledge_base_v1"
        assert state["vertical_id"] == "enclave_guard"
        assert state["run_id"] == "run-1"
        assert state["current_node"] == "start"
        assert state["error"] is None
        assert state["retry_count"] == 0
        assert state["requires_human_approval"] is False
        assert state["knowledge_written"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_article_categories(self):
        from core.agents.implementations import knowledge_base_agent
        cats = knowledge_base_agent.ARTICLE_CATEGORIES
        assert isinstance(cats, list)
        assert "getting_started" in cats
        assert "troubleshooting" in cats
        assert "billing" in cats
        assert "integration" in cats
        assert "security" in cats

    def test_min_ticket_cluster_size(self):
        from core.agents.implementations import knowledge_base_agent
        assert knowledge_base_agent.MIN_TICKET_CLUSTER_SIZE == 3

    def test_gap_analysis_prompt(self):
        from core.agents.implementations import knowledge_base_agent
        prompt = knowledge_base_agent.GAP_ANALYSIS_PROMPT
        assert "{lookback_days}" in prompt
        assert "{tickets_json}" in prompt
        assert "{categories}" in prompt
        assert "{hive_insights_json}" in prompt
        assert "{min_cluster_size}" in prompt

    def test_article_generation_prompt(self):
        from core.agents.implementations import knowledge_base_agent
        prompt = knowledge_base_agent.ARTICLE_GENERATION_PROMPT
        assert "{topic}" in prompt
        assert "{category}" in prompt
        assert "{gap_description}" in prompt
        assert "{sample_questions_json}" in prompt
        assert "{ticket_summaries_json}" in prompt

    # ─── Node 1: Scan Sources ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_sources_success(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "t1",
                "subject": "Login issues",
                "category": "troubleshooting",
                "resolution_summary": "Reset password",
                "created_at": "2024-01-15T10:00:00Z",
            },
            {
                "id": "t2",
                "subject": "Billing question",
                "category": "billing",
                "resolution_summary": "Updated invoice",
                "created_at": "2024-01-20T10:00:00Z",
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
        agent.consult_hive = MagicMock(return_value=[
            {"topic": "Common Auth Issues", "content": "Users often forget passwords", "confidence": 0.8},
        ])

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_sources(state)
        assert result["current_node"] == "scan_sources"
        assert len(result["resolved_tickets"]) == 2
        assert len(result["hive_insights"]) == 1
        assert result["total_sources_scanned"] >= 3

    @pytest.mark.asyncio
    async def test_node_scan_sources_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.side_effect = Exception("DB fail")
        agent.consult_hive = MagicMock(return_value=[])

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_sources(state)
        assert result["current_node"] == "scan_sources"
        assert result["resolved_tickets"] == []

    @pytest.mark.asyncio
    async def test_node_scan_sources_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
        agent.consult_hive = MagicMock(return_value=[])

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_sources(state)
        assert result["resolved_tickets"] == []
        assert result["hive_insights"] == []
        assert result["total_sources_scanned"] == 0

    @pytest.mark.asyncio
    async def test_node_scan_sources_hive_error(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"id": "t1", "subject": "Test", "category": "general", "resolution_summary": "Fixed", "created_at": "2024-01-01"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
        agent.consult_hive = MagicMock(side_effect=Exception("Hive down"))

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_sources(state)
        assert len(result["resolved_tickets"]) == 1
        assert result["hive_insights"] == []

    # ─── Node 2: Analyze Gaps ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_gaps_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps([
            {
                "topic": "Password Reset",
                "category": "troubleshooting",
                "ticket_count": 5,
                "sample_questions": ["How do I reset my password?"],
                "gap_description": "No KB article for password reset",
                "priority": "high",
                "suggested_article_title": "How to Reset Your Password",
            },
            {
                "topic": "Invoice Clarification",
                "category": "billing",
                "ticket_count": 2,
                "sample_questions": ["Why was I charged extra?"],
                "gap_description": "Billing FAQ missing",
                "priority": "medium",
                "suggested_article_title": "Understanding Your Invoice",
            },
        ])
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["resolved_tickets"] = [
            {"subject": "Password reset help", "category": "troubleshooting", "resolution_summary": "Walk through reset steps"},
        ]
        state["hive_insights"] = []
        result = await agent._node_analyze_gaps(state)
        assert result["current_node"] == "analyze_gaps"
        assert len(result["topic_clusters"]) == 2
        assert len(result["identified_gaps"]) >= 1
        assert result["total_gaps"] >= 1

    @pytest.mark.asyncio
    async def test_node_analyze_gaps_no_sources(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["resolved_tickets"] = []
        state["hive_insights"] = []
        result = await agent._node_analyze_gaps(state)
        assert result["current_node"] == "analyze_gaps"
        assert result["topic_clusters"] == []
        assert result["identified_gaps"] == []
        assert result["total_gaps"] == 0

    @pytest.mark.asyncio
    async def test_node_analyze_gaps_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["resolved_tickets"] = [
            {"subject": "Test ticket", "category": "general", "resolution_summary": "Resolved"},
        ]
        state["hive_insights"] = []
        result = await agent._node_analyze_gaps(state)
        assert result["current_node"] == "analyze_gaps"
        assert result["topic_clusters"] == []
        assert result["total_gaps"] == 0

    @pytest.mark.asyncio
    async def test_node_analyze_gaps_parse_error(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "Not valid JSON at all"
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["resolved_tickets"] = [
            {"subject": "Test", "category": "general", "resolution_summary": "Fixed"},
        ]
        state["hive_insights"] = []
        result = await agent._node_analyze_gaps(state)
        assert result["topic_clusters"] == []
        assert result["total_gaps"] == 0

    # ─── Node 3: Generate Articles ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_articles_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "title": "How to Reset Your Password",
            "category": "troubleshooting",
            "summary": "Step-by-step password reset guide",
            "content": "# How to Reset Your Password\n\n1. Go to login page...",
            "tags": ["password", "login", "reset"],
            "related_topics": ["Account Security"],
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["identified_gaps"] = [
            {
                "topic": "Password Reset",
                "category": "troubleshooting",
                "gap_description": "No KB article on password reset",
                "sample_questions": ["How do I reset my password?"],
                "ticket_count": 5,
                "priority": "high",
                "suggested_article_title": "How to Reset Your Password",
            },
        ]
        state["resolved_tickets"] = [
            {"subject": "Password reset needed", "resolution_summary": "Walked through reset steps"},
        ]
        result = await agent._node_generate_articles(state)
        assert result["current_node"] == "generate_articles"
        assert len(result["draft_articles"]) == 1
        assert result["articles_generated"] == 1
        assert result["draft_articles"][0]["title"] == "How to Reset Your Password"

    @pytest.mark.asyncio
    async def test_node_generate_articles_no_gaps(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["identified_gaps"] = []
        state["resolved_tickets"] = []
        result = await agent._node_generate_articles(state)
        assert result["draft_articles"] == []
        assert result["articles_generated"] == 0

    @pytest.mark.asyncio
    async def test_node_generate_articles_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["identified_gaps"] = [
            {"topic": "Auth", "category": "security", "gap_description": "No article", "sample_questions": [], "priority": "high"},
        ]
        state["resolved_tickets"] = []
        result = await agent._node_generate_articles(state)
        assert result["current_node"] == "generate_articles"
        assert result["articles_generated"] == 0

    # ─── Node 4: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "draft_articles": [{"title": "Test Article"}],
            "total_gaps": 3,
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report_approved(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "art_1"}])

        state = agent._prepare_initial_state({}, "run-1")
        state["human_approval_status"] = "approved"
        state["draft_articles"] = [
            {"title": "Password FAQ", "category": "troubleshooting", "summary": "Reset steps",
             "content": "Guide content", "tags": ["password"], "related_topics": [],
             "source_gap": "Password Reset", "source_ticket_count": 5, "priority": "high"},
        ]
        state["topic_clusters"] = [{"topic": "Password Reset"}]
        state["identified_gaps"] = [{"topic": "Password Reset", "priority": "high", "gap_description": "Missing article"}]
        state["total_sources_scanned"] = 10
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["articles_saved"] >= 1
        assert "Knowledge Base Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_rejected(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["human_approval_status"] = "rejected"
        state["draft_articles"] = [
            {"title": "Rejected Article", "category": "general", "summary": "...",
             "content": "...", "tags": [], "related_topics": [],
             "source_gap": "Gap", "source_ticket_count": 2, "priority": "low"},
        ]
        state["topic_clusters"] = []
        state["identified_gaps"] = []
        state["total_sources_scanned"] = 5
        result = await agent._node_report(state)
        assert result["articles_saved"] == 0
        assert result["articles_published"] == 0

    @pytest.mark.asyncio
    async def test_node_report_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({}, "run-1")
        state["human_approval_status"] = "approved"
        state["draft_articles"] = [
            {"title": "Error Article", "category": "general", "summary": "...", "content": "...",
             "tags": [], "related_topics": [], "source_gap": "Gap", "source_ticket_count": 1, "priority": "medium"},
        ]
        state["topic_clusters"] = []
        state["identified_gaps"] = [{"topic": "Gap"}]
        state["total_sources_scanned"] = 1
        result = await agent._node_report(state)
        assert result["articles_saved"] == 0
        assert result["current_node"] == "report"

    @pytest.mark.asyncio
    async def test_node_report_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["draft_articles"] = []
        state["topic_clusters"] = []
        state["identified_gaps"] = []
        state["total_sources_scanned"] = 0
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Knowledge Base Report" in result["report_summary"]

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent
        state = {"human_approval_status": "approved"}
        assert KnowledgeBaseAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent
        state = {"human_approval_status": "rejected"}
        assert KnowledgeBaseAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.knowledge_base_agent import KnowledgeBaseAgent
        state = {}
        assert KnowledgeBaseAgent._route_after_review(state) == "approved"

    # ─── write_knowledge ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"articles_generated": 3})
        assert result is None


# ══════════════════════════════════════════════════════════════════════
#  2.  FeedbackAgent
# ══════════════════════════════════════════════════════════════════════


class TestFeedbackAgentState:
    """Tests for FeedbackAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import FeedbackAgentState
        assert FeedbackAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import FeedbackAgentState
        state: FeedbackAgentState = {
            "agent_id": "feedback_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "feedback_v1"

    def test_create_full(self):
        from core.agents.state import FeedbackAgentState
        state: FeedbackAgentState = {
            "agent_id": "feedback_v1",
            "vertical_id": "enclave_guard",
            "target_touchpoint": "post_project",
            "eligible_clients": [{"client_id": "c1"}],
            "total_eligible": 1,
            "feedback_responses": [{"client_id": "c1", "score": 9}],
            "total_responses": 1,
            "nps_score": 100.0,
            "promoters": [{"client_id": "c1", "score": 9}],
            "passives": [],
            "detractors": [],
            "avg_sentiment": 0.9,
            "critical_feedback": [],
            "critical_count": 0,
            "escalation_actions": [],
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["nps_score"] == 100.0
        assert state["target_touchpoint"] == "post_project"
        assert state["total_eligible"] == 1
        assert len(state["promoters"]) == 1

    def test_nps_grouping_fields(self):
        from core.agents.state import FeedbackAgentState
        state: FeedbackAgentState = {
            "promoters": [{"score": 10}, {"score": 9}],
            "passives": [{"score": 8}],
            "detractors": [{"score": 5}, {"score": 3}],
            "nps_score": 0.0,
            "avg_sentiment": 0.5,
        }
        assert len(state["promoters"]) == 2
        assert len(state["detractors"]) == 2

    def test_critical_and_escalation_fields(self):
        from core.agents.state import FeedbackAgentState
        state: FeedbackAgentState = {
            "critical_feedback": [{"client_id": "c1", "score": 2, "sentiment": "very_negative"}],
            "critical_count": 1,
            "escalation_actions": [{"client_id": "c1", "action": "Call immediately", "urgency": "high"}],
        }
        assert state["critical_count"] == 1
        assert state["escalation_actions"][0]["urgency"] == "high"


class TestFeedbackAgent:
    """Tests for FeedbackAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a FeedbackAgent with mocked dependencies."""
        from core.agents.implementations.feedback_agent import FeedbackAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="feedback_v1",
            agent_type="feedback",
            name="Feedback Agent",
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

        return FeedbackAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.feedback_agent import FeedbackAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "feedback" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.feedback_agent import FeedbackAgent
        assert FeedbackAgent.agent_type == "feedback"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import FeedbackAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is FeedbackAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "FeedbackAgent" in r
        assert "feedback_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"touchpoint": "post_project"}, "run-1"
        )
        assert state["target_touchpoint"] == ""
        assert state["eligible_clients"] == []
        assert state["total_eligible"] == 0
        assert state["feedback_responses"] == []
        assert state["total_responses"] == 0
        assert state["nps_score"] == 0.0
        assert state["promoters"] == []
        assert state["passives"] == []
        assert state["detractors"] == []
        assert state["avg_sentiment"] == 0.5
        assert state["critical_feedback"] == []
        assert state["critical_count"] == 0
        assert state["escalation_actions"] == []
        assert state["report_summary"] == ""
        assert state["report_generated_at"] == ""

    def test_prepare_initial_state_common_keys(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"touchpoint": "quarterly"}, "run-1"
        )
        assert state["agent_id"] == "feedback_v1"
        assert state["vertical_id"] == "enclave_guard"
        assert state["run_id"] == "run-1"
        assert state["current_node"] == "start"
        assert state["error"] is None
        assert state["retry_count"] == 0
        assert state["requires_human_approval"] is False
        assert state["knowledge_written"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_nps_categories(self):
        from core.agents.implementations import feedback_agent
        cats = feedback_agent.NPS_CATEGORIES
        assert "promoter" in cats
        assert "passive" in cats
        assert "detractor" in cats
        assert cats["promoter"]["min"] == 9
        assert cats["promoter"]["max"] == 10
        assert cats["passive"]["min"] == 7
        assert cats["detractor"]["max"] == 6

    def test_survey_types(self):
        from core.agents.implementations import feedback_agent
        types = feedback_agent.SURVEY_TYPES
        assert isinstance(types, list)
        assert "nps" in types
        assert "csat" in types
        assert "ces" in types

    def test_touchpoints(self):
        from core.agents.implementations import feedback_agent
        tp = feedback_agent.TOUCHPOINTS
        assert isinstance(tp, list)
        assert "post_project" in tp
        assert "quarterly" in tp
        assert "post_support" in tp
        assert "onboarding_complete" in tp

    def test_touchpoint_lookback_days(self):
        from core.agents.implementations import feedback_agent
        lookback = feedback_agent.TOUCHPOINT_LOOKBACK_DAYS
        assert isinstance(lookback, dict)
        assert lookback["post_project"] == 14
        assert lookback["quarterly"] == 90
        assert lookback["post_support"] == 7
        assert lookback["onboarding_complete"] == 30

    def test_sentiment_analysis_prompt(self):
        from core.agents.implementations import feedback_agent
        prompt = feedback_agent.SENTIMENT_ANALYSIS_PROMPT
        assert "{responses_json}" in prompt
        assert "{survey_type}" in prompt
        assert "{touchpoint}" in prompt

    def test_feedback_summary_prompt(self):
        from core.agents.implementations import feedback_agent
        prompt = feedback_agent.FEEDBACK_SUMMARY_PROMPT
        assert "{nps_score}" in prompt
        assert "{total_responses}" in prompt
        assert "{promoter_count}" in prompt
        assert "{detractor_count}" in prompt

    # ─── Module helpers ──────────────────────────────────────────────

    def test_calculate_nps_all_promoters(self):
        from core.agents.implementations.feedback_agent import _calculate_nps
        score = _calculate_nps([10, 10, 9, 9, 10])
        assert score == 100.0

    def test_calculate_nps_all_detractors(self):
        from core.agents.implementations.feedback_agent import _calculate_nps
        score = _calculate_nps([1, 2, 3, 4, 5])
        assert score == -100.0

    def test_calculate_nps_mixed(self):
        from core.agents.implementations.feedback_agent import _calculate_nps
        # 2 promoters (10,9), 1 passive (8), 2 detractors (5,3) => (2-2)/5*100 = 0
        score = _calculate_nps([10, 9, 8, 5, 3])
        assert score == 0.0

    def test_calculate_nps_empty(self):
        from core.agents.implementations.feedback_agent import _calculate_nps
        score = _calculate_nps([])
        assert score == 0.0

    def test_classify_nps(self):
        from core.agents.implementations.feedback_agent import _classify_nps
        assert _classify_nps(10) == "promoter"
        assert _classify_nps(9) == "promoter"
        assert _classify_nps(8) == "passive"
        assert _classify_nps(7) == "passive"
        assert _classify_nps(6) == "detractor"
        assert _classify_nps(0) == "detractor"

    # ─── Node 1: Identify Audience ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_identify_audience_post_project(self):
        agent = self._make_agent()
        mock_proposals = MagicMock()
        mock_proposals.data = [
            {"id": "p1", "company_name": "Acme Corp", "contact_email": "jane@acme.com", "contact_name": "Jane"},
        ]
        mock_feedback = MagicMock()
        mock_feedback.data = [
            {"client_id": "c1", "client_name": "Acme", "score": 9, "comment": "Great work!", "survey_type": "nps", "touchpoint": "post_project", "created_at": "2024-02-01"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.side_effect = [mock_proposals, mock_feedback]

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {"touchpoint": "post_project", "include_existing_responses": True}
        result = await agent._node_identify_audience(state)
        assert result["current_node"] == "identify_audience"
        assert result["target_touchpoint"] == "post_project"
        assert result["total_eligible"] >= 1
        assert result["total_responses"] >= 1

    @pytest.mark.asyncio
    async def test_node_identify_audience_quarterly(self):
        agent = self._make_agent()
        mock_companies = MagicMock()
        mock_companies.data = [
            {"id": "c1", "name": "Acme Corp", "domain": "acme.com"},
            {"id": "c2", "name": "Big Corp", "domain": "big.com"},
        ]
        mock_feedback = MagicMock()
        mock_feedback.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_companies
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_feedback

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {"touchpoint": "quarterly", "include_existing_responses": True}
        result = await agent._node_identify_audience(state)
        assert result["target_touchpoint"] == "quarterly"
        assert result["total_eligible"] >= 2

    @pytest.mark.asyncio
    async def test_node_identify_audience_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {"touchpoint": "post_project"}
        result = await agent._node_identify_audience(state)
        assert result["current_node"] == "identify_audience"

    @pytest.mark.asyncio
    async def test_node_identify_audience_invalid_touchpoint_fallback(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {"touchpoint": "nonexistent_touchpoint"}
        result = await agent._node_identify_audience(state)
        assert result["target_touchpoint"] == "quarterly"

    # ─── Node 2: Analyze Responses ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_responses_with_data(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps([
            {
                "client_id": "c1",
                "client_name": "Acme",
                "score": 9,
                "sentiment": "positive",
                "themes": ["great service"],
                "actionability": "no_action",
                "key_quote": "Everything was great!",
                "suggested_action": "Send thank you note",
            },
            {
                "client_id": "c2",
                "client_name": "Bad Corp",
                "score": 3,
                "sentiment": "very_negative",
                "themes": ["slow response"],
                "actionability": "immediate_action",
                "key_quote": "Response time was unacceptable",
                "suggested_action": "Schedule call",
            },
        ])
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "post_project"
        state["feedback_responses"] = [
            {"client_id": "c1", "client_name": "Acme", "score": 9, "comment": "Everything was great!"},
            {"client_id": "c2", "client_name": "Bad Corp", "score": 3, "comment": "Response time was unacceptable"},
        ]
        result = await agent._node_analyze_responses(state)
        assert result["current_node"] == "analyze_responses"
        assert result["total_responses"] == 2
        assert len(result["promoters"]) >= 1
        assert len(result["detractors"]) >= 1
        assert isinstance(result["nps_score"], float)  # 1 promoter, 1 detractor out of 2 => 0.0
        assert isinstance(result["avg_sentiment"], float)

    @pytest.mark.asyncio
    async def test_node_analyze_responses_no_responses(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "quarterly"
        state["feedback_responses"] = []
        result = await agent._node_analyze_responses(state)
        assert result["current_node"] == "analyze_responses"
        assert result["total_responses"] == 0
        assert result["nps_score"] == 0.0
        assert result["promoters"] == []
        assert result["passives"] == []
        assert result["detractors"] == []

    @pytest.mark.asyncio
    async def test_node_analyze_responses_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "quarterly"
        state["feedback_responses"] = [
            {"client_id": "c1", "score": 10, "comment": "Great!"},
        ]
        result = await agent._node_analyze_responses(state)
        assert result["current_node"] == "analyze_responses"
        assert result["total_responses"] == 1
        assert len(result["promoters"]) == 1

    # ─── Node 3: Route Critical ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_route_critical_with_detractors(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["detractors"] = [
            {"client_id": "c1", "client_name": "Unhappy Client", "score": 2, "sentiment": "very_negative", "actionability": "immediate_action", "key_quote": "Terrible service", "suggested_action": "Call immediately"},
        ]
        state["feedback_responses"] = state["detractors"]
        result = await agent._node_route_critical(state)
        assert result["current_node"] == "route_critical"
        assert result["critical_count"] >= 1
        assert len(result["critical_feedback"]) >= 1
        assert len(result["escalation_actions"]) >= 1

    @pytest.mark.asyncio
    async def test_node_route_critical_no_detractors(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["detractors"] = []
        state["feedback_responses"] = []
        result = await agent._node_route_critical(state)
        assert result["critical_count"] == 0
        assert result["critical_feedback"] == []
        assert result["escalation_actions"] == []

    @pytest.mark.asyncio
    async def test_node_route_critical_mixed_scores(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["detractors"] = [
            {"client_id": "c1", "client_name": "Client A", "score": 6, "sentiment": "negative", "actionability": "monitor"},
            {"client_id": "c2", "client_name": "Client B", "score": 3, "sentiment": "very_negative", "actionability": "immediate_action", "key_quote": "Worst experience", "suggested_action": "Urgent call"},
        ]
        state["feedback_responses"] = state["detractors"]
        result = await agent._node_route_critical(state)
        assert result["critical_count"] >= 1
        # Client B (score 3) should be critical
        critical_ids = [c.get("client_id") for c in result["critical_feedback"]]
        assert "c2" in critical_ids

    # ─── Node 4: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"critical_count": 2, "nps_score": -10.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report_with_data(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "fa_1"}])

        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "post_project"
        state["nps_score"] = 40.0
        state["total_responses"] = 10
        state["promoters"] = [{"score": 10}, {"score": 9}, {"score": 9}, {"score": 10}, {"score": 9}, {"score": 10}, {"score": 9}]
        state["passives"] = [{"score": 8}]
        state["detractors"] = [{"score": 5}, {"score": 4}]
        state["critical_feedback"] = [{"client_name": "Sad Corp", "score": 4, "sentiment": "negative", "key_quote": "Needs improvement"}]
        state["escalation_actions"] = [{"client_name": "Sad Corp", "urgency": "medium", "action": "Follow up"}]
        state["avg_sentiment"] = 0.7
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Customer Feedback Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_empty(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])

        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "quarterly"
        state["nps_score"] = 0.0
        state["total_responses"] = 0
        state["promoters"] = []
        state["passives"] = []
        state["detractors"] = []
        state["critical_feedback"] = []
        state["escalation_actions"] = []
        state["avg_sentiment"] = 0.5
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Customer Feedback Report" in result["report_summary"]

    @pytest.mark.asyncio
    async def test_node_report_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["target_touchpoint"] = "quarterly"
        state["nps_score"] = 50.0
        state["total_responses"] = 5
        state["promoters"] = [{"score": 10}]
        state["passives"] = []
        state["detractors"] = []
        state["critical_feedback"] = []
        state["escalation_actions"] = []
        state["avg_sentiment"] = 0.9
        result = await agent._node_report(state)
        assert result["current_node"] == "report"

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.feedback_agent import FeedbackAgent
        state = {"human_approval_status": "approved"}
        assert FeedbackAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.feedback_agent import FeedbackAgent
        state = {"human_approval_status": "rejected"}
        assert FeedbackAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.feedback_agent import FeedbackAgent
        state = {}
        assert FeedbackAgent._route_after_review(state) == "approved"

    # ─── write_knowledge ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"nps_score": 45.0})
        assert result is None
