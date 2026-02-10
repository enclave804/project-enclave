"""
Tests for Universal Business Agents — Phase 20.

Covers 4 cross-vertical business operations agents:
    1. ContractManagerAgent (contract_manager)
    2. SupportAgentImpl (support_agent)
    3. CompetitiveIntelAgent (competitive_intel)
    4. ReportingAgent (reporting)

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
#  1.  ContractManagerAgent
# ══════════════════════════════════════════════════════════════════════


class TestContractManagerState:
    """Tests for ContractManagerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ContractManagerAgentState
        assert ContractManagerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ContractManagerAgentState
        state: ContractManagerAgentState = {
            "agent_id": "contract_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "contract_v1"

    def test_create_full(self):
        from core.agents.state import ContractManagerAgentState
        state: ContractManagerAgentState = {
            "agent_id": "contract_v1",
            "vertical_id": "enclave_guard",
            "expiring_contracts": [],
            "total_expiring": 0,
            "renewal_window_days": 30,
            "contract_template": "",
            "company_name": "Acme Inc",
            "company_domain": "acme.com",
            "contact_name": "Jane Doe",
            "contact_email": "jane@acme.com",
            "opportunity_id": "opp_123",
            "contract_terms": {},
            "draft_contract": "",
            "contract_type": "msa",
            "contract_id": "",
            "contract_saved": False,
            "contract_sent": False,
            "sent_at": "",
            "signature_status": "pending",
            "signed_at": "",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["company_name"] == "Acme Inc"
        assert state["renewal_window_days"] == 30
        assert state["contract_saved"] is False

    def test_contract_type_field(self):
        from core.agents.state import ContractManagerAgentState
        state: ContractManagerAgentState = {
            "contract_type": "nda",
            "draft_contract": "# NDA\n\nAgreement...",
        }
        assert state["contract_type"] == "nda"
        assert "NDA" in state["draft_contract"]

    def test_signature_tracking_fields(self):
        from core.agents.state import ContractManagerAgentState
        state: ContractManagerAgentState = {
            "signature_status": "signed",
            "signed_at": "2024-01-15T10:00:00Z",
            "contract_sent": True,
            "sent_at": "2024-01-10T10:00:00Z",
        }
        assert state["signature_status"] == "signed"
        assert state["contract_sent"] is True


class TestContractManagerAgent:
    """Tests for ContractManagerAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a ContractManagerAgent with mocked dependencies."""
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="contract_v1",
            agent_type="contract_manager",
            name="Contract Manager",
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

        return ContractManagerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "contract_manager" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent
        assert ContractManagerAgent.agent_type == "contract_manager"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ContractManagerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ContractManagerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ContractManagerAgent" in r
        assert "contract_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_name": "Acme", "contract_type": "msa"}, "run-1"
        )
        assert state["expiring_contracts"] == []
        assert state["total_expiring"] == 0
        assert state["renewal_window_days"] == 30
        assert state["contract_template"] == ""
        assert state["company_name"] == ""
        assert state["draft_contract"] == ""
        assert state["contract_saved"] is False
        assert state["contract_sent"] is False
        assert state["signature_status"] == "pending"
        assert state["report_summary"] == ""

    # ─── Constants ──────────────────────────────────────────────────

    def test_contract_templates(self):
        from core.agents.implementations import contract_manager_agent
        templates = contract_manager_agent.CONTRACT_TEMPLATES
        assert "service_agreement" in templates
        assert "msa" in templates
        assert "nda" in templates
        assert "sow" in templates
        assert templates["msa"]["typical_duration_months"] == 24

    def test_renewal_window_days(self):
        from core.agents.implementations import contract_manager_agent
        assert contract_manager_agent.RENEWAL_WINDOW_DAYS == 30

    def test_system_prompt(self):
        from core.agents.implementations import contract_manager_agent
        prompt = contract_manager_agent.CONTRACT_SYSTEM_PROMPT
        assert "{contract_type}" in prompt
        assert "{company_name}" in prompt
        assert "{contact_name}" in prompt
        assert "{sections_list}" in prompt

    # ─── Node 1: Check Renewals ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_check_renewals_no_expiring(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.lte.return_value.gte.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"opportunity_id": "opp_1", "contract_type": "nda"}, "run-1"
        )
        result = await agent._node_check_renewals(state)
        assert result["current_node"] == "check_renewals"
        assert result["expiring_contracts"] == []
        assert result["total_expiring"] == 0
        assert result["contract_type"] == "nda"

    @pytest.mark.asyncio
    async def test_node_check_renewals_with_expiring(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"id": "c1", "company": "Acme", "status": "active"}]
        agent.db.client.table.return_value.select.return_value.lte.return_value.gte.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"contract_type": "msa", "company_name": "Acme"}, "run-1"
        )
        result = await agent._node_check_renewals(state)
        assert result["total_expiring"] == 1
        assert len(result["expiring_contracts"]) == 1

    @pytest.mark.asyncio
    async def test_node_check_renewals_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.lte.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"contract_type": "sow"}, "run-1"
        )
        result = await agent._node_check_renewals(state)
        assert result["current_node"] == "check_renewals"
        assert result["expiring_contracts"] == []

    # ─── Node 2: Generate Contract ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_contract_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "# Master Service Agreement\n\nBetween parties..."
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "msa"
        state["company_name"] = "Test Corp"
        state["contact_name"] = "Jane Doe"
        result = await agent._node_generate_contract(state)
        assert result["current_node"] == "generate_contract"
        assert "Master Service Agreement" in result["draft_contract"]
        assert result["contract_template"] == "msa"

    @pytest.mark.asyncio
    async def test_node_generate_contract_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "nda"
        state["company_name"] = "Fallback Corp"
        result = await agent._node_generate_contract(state)
        assert result["current_node"] == "generate_contract"
        assert "Fallback Corp" in result["draft_contract"]
        assert "Non-Disclosure Agreement" in result["draft_contract"]

    @pytest.mark.asyncio
    async def test_node_generate_contract_unknown_type_defaults(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "unknown_type"
        result = await agent._node_generate_contract(state)
        assert result["current_node"] == "generate_contract"
        assert "Service Agreement" in result["draft_contract"]

    # ─── Node 3: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"contract_type": "msa", "company_name": "Test"}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 4: Send Contract ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_send_contract_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "contract_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "msa"
        state["company_name"] = "Acme"
        state["draft_contract"] = "# Contract..."
        result = await agent._node_send_contract(state)
        assert result["current_node"] == "send_contract"
        assert result["contract_saved"] is True
        assert result["contract_sent"] is True
        assert result["contract_id"] == "contract_1"
        assert result["sent_at"] != ""

    @pytest.mark.asyncio
    async def test_node_send_contract_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "sow"
        result = await agent._node_send_contract(state)
        assert result["contract_saved"] is False
        assert result["contract_id"] == ""

    # ─── Node 5: Track Signatures ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_track_signatures_pending(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"signature_status": "pending", "signed_at": ""}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = {"contract_id": "c_1"}
        result = await agent._node_track_signatures(state)
        assert result["current_node"] == "track_signatures"
        assert result["signature_status"] == "pending"
        assert result["signed_at"] == ""

    @pytest.mark.asyncio
    async def test_node_track_signatures_signed(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{"signature_status": "signed", "signed_at": "2024-01-15T12:00:00Z"}]
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        state = {"contract_id": "c_2"}
        result = await agent._node_track_signatures(state)
        assert result["signature_status"] == "signed"
        assert result["signed_at"] == "2024-01-15T12:00:00Z"

    # ─── Node 6: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["contract_type"] = "msa"
        state["company_name"] = "Acme"
        state["total_expiring"] = 2
        state["contract_saved"] = True
        state["contract_sent"] = True
        state["signature_status"] = "pending"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Contract Management Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ──────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent
        state = {"human_approval_status": "approved"}
        assert ContractManagerAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent
        state = {"human_approval_status": "rejected"}
        assert ContractManagerAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.contract_manager_agent import ContractManagerAgent
        state = {}
        assert ContractManagerAgent._route_after_review(state) == "approved"


# ══════════════════════════════════════════════════════════════════════
#  2.  SupportAgentImpl
# ══════════════════════════════════════════════════════════════════════


class TestSupportAgentState:
    """Tests for SupportAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import SupportAgentState
        assert SupportAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import SupportAgentState
        state: SupportAgentState = {
            "agent_id": "support_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "support_v1"

    def test_create_full(self):
        from core.agents.state import SupportAgentState
        state: SupportAgentState = {
            "agent_id": "support_v1",
            "vertical_id": "print_biz",
            "ticket_id": "tk_001",
            "ticket_subject": "Can't login",
            "ticket_body": "I've been locked out since yesterday",
            "customer_email": "jane@acme.com",
            "customer_name": "Jane Doe",
            "customer_id": "cust_123",
            "category": "account_access",
            "priority": "high",
            "sentiment": "negative",
            "escalation_needed": False,
            "classification_reasoning": "",
            "relevant_articles": [],
            "knowledge_sources_checked": 0,
            "draft_response": "",
            "response_tone": "empathetic",
            "suggested_actions": [],
            "response_sent": False,
            "ticket_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["ticket_subject"] == "Can't login"
        assert state["priority"] == "high"
        assert state["escalation_needed"] is False

    def test_classification_fields(self):
        from core.agents.state import SupportAgentState
        state: SupportAgentState = {
            "category": "billing",
            "priority": "critical",
            "sentiment": "angry",
            "escalation_needed": True,
        }
        assert state["category"] == "billing"
        assert state["escalation_needed"] is True

    def test_response_fields(self):
        from core.agents.state import SupportAgentState
        state: SupportAgentState = {
            "draft_response": "Dear Customer...",
            "response_tone": "professional",
            "suggested_actions": ["Escalate to manager"],
        }
        assert len(state["suggested_actions"]) == 1


class TestSupportAgentImpl:
    """Tests for SupportAgentImpl implementation."""

    def _make_agent(self, **kwargs):
        """Create a SupportAgentImpl with mocked dependencies."""
        from core.agents.implementations.support_agent_impl import SupportAgentImpl
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="support_v1",
            agent_type="support_agent",
            name="Support Agent",
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

        return SupportAgentImpl(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.support_agent_impl import SupportAgentImpl  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "support_agent" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.support_agent_impl import SupportAgentImpl
        assert SupportAgentImpl.agent_type == "support_agent"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import SupportAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is SupportAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "SupportAgentImpl" in r
        assert "support_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"ticket_id": "tk_1", "ticket_subject": "Help"}, "run-1"
        )
        assert state["ticket_id"] == ""
        assert state["ticket_subject"] == ""
        assert state["customer_email"] == ""
        assert state["category"] == ""
        assert state["priority"] == ""
        assert state["escalation_needed"] is False
        assert state["relevant_articles"] == []
        assert state["knowledge_sources_checked"] == 0
        assert state["draft_response"] == ""
        assert state["response_sent"] is False
        assert state["ticket_saved"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_categories(self):
        from core.agents.implementations import support_agent_impl
        cats = support_agent_impl.CATEGORIES
        assert "billing" in cats
        assert "technical" in cats
        assert "bug" in cats
        assert "general" in cats

    def test_priority_rules(self):
        from core.agents.implementations import support_agent_impl
        rules = support_agent_impl.PRIORITY_RULES
        assert "critical" in rules
        assert "high" in rules
        assert "medium" in rules
        assert "low" in rules
        assert "keywords" in rules["critical"]

    def test_escalation_keywords(self):
        from core.agents.implementations import support_agent_impl
        kw = support_agent_impl.ESCALATION_KEYWORDS
        assert "lawyer" in kw
        assert "refund" in kw
        assert "escalate" in kw

    def test_sentiment_indicators(self):
        from core.agents.implementations import support_agent_impl
        si = support_agent_impl.SENTIMENT_INDICATORS
        assert "angry" in si
        assert "negative" in si
        assert "neutral" in si
        assert "positive" in si

    def test_classify_prompt(self):
        from core.agents.implementations import support_agent_impl
        prompt = support_agent_impl.SUPPORT_CLASSIFY_PROMPT
        assert "{subject}" in prompt
        assert "{body}" in prompt

    def test_response_prompt(self):
        from core.agents.implementations import support_agent_impl
        prompt = support_agent_impl.SUPPORT_RESPONSE_PROMPT
        assert "{customer_name}" in prompt
        assert "{knowledge_articles}" in prompt

    # ─── Node 1: Classify Ticket ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_classify_ticket_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "category": "billing",
            "priority": "high",
            "sentiment": "negative",
            "escalation_needed": False,
            "reasoning": "Billing dispute",
            "summary": "Customer has billing question",
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state(
            {
                "ticket_id": "tk_1",
                "ticket_subject": "Billing issue",
                "ticket_body": "I was charged twice",
                "customer_email": "jane@x.com",
            },
            "run-1",
        )
        result = await agent._node_classify_ticket(state)
        assert result["current_node"] == "classify_ticket"
        assert result["category"] == "billing"
        assert result["ticket_id"] == "tk_1"

    @pytest.mark.asyncio
    async def test_node_classify_ticket_escalation_keywords(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state(
            {
                "ticket_id": "tk_2",
                "ticket_subject": "I want to sue",
                "ticket_body": "This is unacceptable, I need a lawyer",
            },
            "run-1",
        )
        result = await agent._node_classify_ticket(state)
        assert result["escalation_needed"] is True
        assert result["priority"] in ("critical", "high")

    @pytest.mark.asyncio
    async def test_node_classify_ticket_llm_error_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = agent._prepare_initial_state(
            {
                "ticket_id": "tk_3",
                "ticket_subject": "Question about feature",
                "ticket_body": "I'm wondering how to do X",
            },
            "run-1",
        )
        result = await agent._node_classify_ticket(state)
        assert result["current_node"] == "classify_ticket"
        assert result["category"] == "general"

    # ─── Node 2: Search Knowledge ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_search_knowledge_found(self):
        agent = self._make_agent()
        mock_kb = MagicMock()
        mock_kb.data = [{"title": "FAQ", "content": "Reset your password..."}]
        mock_insights = MagicMock()
        mock_insights.data = []
        agent.db.client.table.return_value.select.return_value.ilike.return_value.limit.return_value.execute.side_effect = [
            mock_kb, mock_insights,
        ]

        state = {
            "ticket_subject": "Can't login",
            "ticket_body": "I'm locked out",
            "category": "account_access",
        }
        result = await agent._node_search_knowledge(state)
        assert result["current_node"] == "search_knowledge"
        assert len(result["relevant_articles"]) >= 1
        assert result["knowledge_sources_checked"] >= 1

    @pytest.mark.asyncio
    async def test_node_search_knowledge_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.ilike.side_effect = Exception("DB error")

        state = {"ticket_subject": "Help", "ticket_body": "Need help", "category": "general"}
        result = await agent._node_search_knowledge(state)
        assert result["current_node"] == "search_knowledge"
        assert result["relevant_articles"] == []

    # ─── Node 3: Draft Response ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_draft_response_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "Dear Customer, thank you for reaching out..."
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = {
            "ticket_subject": "Billing question",
            "ticket_body": "I was charged twice",
            "category": "billing",
            "priority": "high",
            "sentiment": "negative",
            "customer_name": "Jane",
            "customer_email": "jane@x.com",
            "relevant_articles": [],
            "escalation_needed": False,
        }
        result = await agent._node_draft_response(state)
        assert result["current_node"] == "draft_response"
        assert "Dear Customer" in result["draft_response"]
        assert result["response_tone"] == "empathetic"

    @pytest.mark.asyncio
    async def test_node_draft_response_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = {
            "ticket_subject": "Need help",
            "ticket_body": "Cannot access dashboard",
            "category": "technical",
            "priority": "medium",
            "sentiment": "neutral",
            "customer_name": "Bob",
            "customer_email": "bob@x.com",
            "relevant_articles": [],
            "escalation_needed": False,
        }
        result = await agent._node_draft_response(state)
        assert "Bob" in result["draft_response"]
        assert "Need help" in result["draft_response"]

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"priority": "high", "category": "billing", "escalation_needed": True}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "tk_saved"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["ticket_id"] = "tk_1"
        state["ticket_subject"] = "Help"
        state["category"] = "general"
        state["priority"] = "medium"
        state["sentiment"] = "neutral"
        state["relevant_articles"] = []
        state["suggested_actions"] = []
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["ticket_saved"] is True
        assert "Support Ticket Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["ticket_id"] = "tk_2"
        state["category"] = "bug"
        state["priority"] = "high"
        state["sentiment"] = "negative"
        state["relevant_articles"] = []
        state["suggested_actions"] = []
        result = await agent._node_report(state)
        assert result["ticket_saved"] is False

    # ─── Routing ──────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.support_agent_impl import SupportAgentImpl
        state = {"human_approval_status": "approved"}
        assert SupportAgentImpl._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.support_agent_impl import SupportAgentImpl
        state = {"human_approval_status": "rejected"}
        assert SupportAgentImpl._route_after_review(state) == "rejected"


# ══════════════════════════════════════════════════════════════════════
#  3.  CompetitiveIntelAgent
# ══════════════════════════════════════════════════════════════════════


class TestCompetitiveIntelState:
    """Tests for CompetitiveIntelAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import CompetitiveIntelAgentState
        assert CompetitiveIntelAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import CompetitiveIntelAgentState
        state: CompetitiveIntelAgentState = {
            "agent_id": "intel_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "intel_v1"

    def test_create_full(self):
        from core.agents.state import CompetitiveIntelAgentState
        state: CompetitiveIntelAgentState = {
            "agent_id": "intel_v1",
            "vertical_id": "enclave_guard",
            "monitored_competitors": [],
            "scan_results": [],
            "intel_findings": [],
            "finding_count": 0,
            "critical_findings": 0,
            "threat_score": 0.0,
            "alerts": [],
            "alerts_approved": False,
            "alerts_sent": 0,
            "intel_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["threat_score"] == 0.0
        assert state["critical_findings"] == 0
        assert state["alerts_approved"] is False

    def test_scan_fields(self):
        from core.agents.state import CompetitiveIntelAgentState
        state: CompetitiveIntelAgentState = {
            "monitored_competitors": [{"name": "Rival", "domain": "rival.com"}],
            "scan_results": [{"finding": "Price drop"}],
        }
        assert len(state["monitored_competitors"]) == 1
        assert state["scan_results"][0]["finding"] == "Price drop"

    def test_alerts_fields(self):
        from core.agents.state import CompetitiveIntelAgentState
        state: CompetitiveIntelAgentState = {
            "alerts": [{"message": "New product launch", "severity": "high"}],
            "alerts_sent": 5,
        }
        assert len(state["alerts"]) == 1
        assert state["alerts_sent"] == 5


class TestCompetitiveIntelAgent:
    """Tests for CompetitiveIntelAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a CompetitiveIntelAgent with mocked dependencies."""
        from core.agents.implementations.competitive_intel_agent import CompetitiveIntelAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="intel_v1",
            agent_type="competitive_intel",
            name="Competitive Intel",
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

        return CompetitiveIntelAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.competitive_intel_agent import CompetitiveIntelAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "competitive_intel" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.competitive_intel_agent import CompetitiveIntelAgent
        assert CompetitiveIntelAgent.agent_type == "competitive_intel"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import CompetitiveIntelAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is CompetitiveIntelAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "CompetitiveIntelAgent" in r
        assert "intel_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"competitors": ["rival.com"]}, "run-1"
        )
        assert state["monitored_competitors"] == []
        assert state["scan_results"] == []
        assert state["intel_findings"] == []
        assert state["finding_count"] == 0
        assert state["critical_findings"] == 0
        assert state["threat_score"] == 0.0
        assert state["alerts"] == []
        assert state["alerts_approved"] is False
        assert state["alerts_sent"] == 0
        assert state["intel_saved"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_intel_types(self):
        from core.agents.implementations import competitive_intel_agent
        types = competitive_intel_agent.INTEL_TYPES
        assert "pricing_change" in types
        assert "product_launch" in types
        assert "acquisition" in types

    def test_severity_levels(self):
        from core.agents.implementations import competitive_intel_agent
        levels = competitive_intel_agent.SEVERITY_LEVELS
        assert "critical" in levels
        assert "high" in levels
        assert "medium" in levels
        assert "low" in levels
        assert levels["critical"]["score_min"] == 8.0

    def test_intel_analysis_prompt(self):
        from core.agents.implementations import competitive_intel_agent
        prompt = competitive_intel_agent.INTEL_ANALYSIS_PROMPT
        assert "{competitors_json}" in prompt
        assert "{scan_results_json}" in prompt

    def test_alert_generation_prompt(self):
        from core.agents.implementations import competitive_intel_agent
        prompt = competitive_intel_agent.ALERT_GENERATION_PROMPT
        assert "{findings_json}" in prompt

    # ─── Node 1: Scan Competitors ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_competitors_from_task(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"competitors": ["rival.com", "other.io"]}, "run-1"
        )
        result = await agent._node_scan_competitors(state)
        assert result["current_node"] == "scan_competitors"
        assert len(result["monitored_competitors"]) == 2
        assert result["monitored_competitors"][0]["domain"] == "rival.com"

    @pytest.mark.asyncio
    async def test_node_scan_competitors_db_results(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"competitor_name": "Rival", "competitor_domain": "rival.com",
             "intel_type": "pricing", "finding": "Price drop 20%", "created_at": "2024-01-01"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"competitors": ["rival.com"]}, "run-1")
        result = await agent._node_scan_competitors(state)
        assert len(result["scan_results"]) >= 1

    @pytest.mark.asyncio
    async def test_node_scan_competitors_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.order.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({"competitors": ["x.com"]}, "run-1")
        result = await agent._node_scan_competitors(state)
        assert result["current_node"] == "scan_competitors"

    # ─── Node 2: Analyze Intel ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_intel_no_results(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["scan_results"] = []
        result = await agent._node_analyze_intel(state)
        assert result["current_node"] == "analyze_intel"
        assert result["intel_findings"] == []
        assert result["threat_score"] == 0.0

    @pytest.mark.asyncio
    async def test_node_analyze_intel_llm_success(self):
        agent = self._make_agent()
        findings = [
            {"competitor": "Rival", "intel_type": "pricing_change",
             "finding": "Price dropped 20%", "severity": "high",
             "threat_score": 7.5, "recommended_action": "Match price",
             "confidence": 0.8, "source": "web"},
        ]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(findings)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["scan_results"] = [{"finding": "test"}]
        state["monitored_competitors"] = [{"name": "Rival", "domain": "rival.com"}]
        result = await agent._node_analyze_intel(state)
        assert result["finding_count"] == 1
        assert result["threat_score"] == 7.5

    @pytest.mark.asyncio
    async def test_node_analyze_intel_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["scan_results"] = [{"finding": "test"}]
        result = await agent._node_analyze_intel(state)
        assert result["intel_findings"] == []
        assert result["threat_score"] == 0.0

    # ─── Node 3: Generate Alerts ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_alerts_no_significant(self):
        agent = self._make_agent()
        state = {"intel_findings": [{"severity": "low"}]}
        result = await agent._node_generate_alerts(state)
        assert result["current_node"] == "generate_alerts"
        assert result["alerts"] == []

    @pytest.mark.asyncio
    async def test_node_generate_alerts_llm_success(self):
        agent = self._make_agent()
        alerts = [
            {"competitor": "Rival", "alert_type": "threat",
             "message": "Major price drop", "severity": "high",
             "recommended_action": "Review pricing", "urgency_hours": 72},
        ]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(alerts)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = {
            "intel_findings": [
                {"severity": "high", "competitor": "Rival", "finding": "Price drop"},
            ],
        }
        result = await agent._node_generate_alerts(state)
        assert len(result["alerts"]) == 1

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"alerts": [{"msg": "test"}], "threat_score": 7.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "intel_1"}])

        state = agent._prepare_initial_state({}, "run-1")
        state["intel_findings"] = [{"competitor": "Rival", "severity": "high", "finding": "Test"}]
        state["alerts"] = [{"message": "Alert"}]
        state["threat_score"] = 7.0
        state["monitored_competitors"] = [{"name": "Rival"}]
        state["finding_count"] = 1
        state["critical_findings"] = 0
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["intel_saved"] is True
        assert "Competitive Intelligence Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ──────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.competitive_intel_agent import CompetitiveIntelAgent
        state = {"human_approval_status": "approved"}
        assert CompetitiveIntelAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.competitive_intel_agent import CompetitiveIntelAgent
        state = {"human_approval_status": "rejected"}
        assert CompetitiveIntelAgent._route_after_review(state) == "rejected"


# ══════════════════════════════════════════════════════════════════════
#  4.  ReportingAgent
# ══════════════════════════════════════════════════════════════════════


class TestReportingAgentState:
    """Tests for ReportingAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ReportingAgentState
        assert ReportingAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ReportingAgentState
        state: ReportingAgentState = {
            "agent_id": "reporting_v1",
            "vertical_id": "print_biz",
        }
        assert state["agent_id"] == "reporting_v1"

    def test_create_full(self):
        from core.agents.state import ReportingAgentState
        state: ReportingAgentState = {
            "agent_id": "reporting_v1",
            "vertical_id": "print_biz",
            "pipeline_metrics": {},
            "revenue_metrics": {},
            "outreach_metrics": {},
            "client_metrics": {},
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "trends": [],
            "forecasts": [],
            "anomalies": [],
            "report_format": "executive",
            "report_sections": [],
            "report_document": "",
            "report_id": "",
            "report_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["period_start"] == "2024-01-01"
        assert state["report_format"] == "executive"
        assert state["report_saved"] is False

    def test_metrics_fields(self):
        from core.agents.state import ReportingAgentState
        state: ReportingAgentState = {
            "pipeline_metrics": {"total_opportunities": 50},
            "revenue_metrics": {"total_revenue_cents": 500000},
        }
        assert state["pipeline_metrics"]["total_opportunities"] == 50

    def test_analysis_fields(self):
        from core.agents.state import ReportingAgentState
        state: ReportingAgentState = {
            "trends": [{"metric": "revenue", "direction": "up"}],
            "forecasts": [{"metric": "revenue", "forecast_value": 600000}],
            "anomalies": [{"metric": "churn", "severity": "high"}],
        }
        assert len(state["trends"]) == 1
        assert state["anomalies"][0]["severity"] == "high"


class TestReportingAgent:
    """Tests for ReportingAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a ReportingAgent with mocked dependencies."""
        from core.agents.implementations.reporting_agent import ReportingAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="reporting_v1",
            agent_type="reporting",
            name="Reporting Agent",
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

        return ReportingAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.reporting_agent import ReportingAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "reporting" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.reporting_agent import ReportingAgent
        assert ReportingAgent.agent_type == "reporting"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ReportingAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ReportingAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ReportingAgent" in r
        assert "reporting_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"report_format": "weekly", "period_start": "2024-01-01"}, "run-1"
        )
        assert state["pipeline_metrics"] == {}
        assert state["revenue_metrics"] == {}
        assert state["outreach_metrics"] == {}
        assert state["client_metrics"] == {}
        assert state["period_start"] == ""
        assert state["period_end"] == ""
        assert state["trends"] == []
        assert state["forecasts"] == []
        assert state["anomalies"] == []
        assert state["report_format"] == "executive"
        assert state["report_sections"] == []
        assert state["report_document"] == ""
        assert state["report_saved"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_report_sections(self):
        from core.agents.implementations import reporting_agent
        sections = reporting_agent.REPORT_SECTIONS
        assert "executive_summary" in sections
        assert "pipeline_overview" in sections
        assert "revenue_analysis" in sections
        assert "anomalies" in sections

    def test_metric_queries(self):
        from core.agents.implementations import reporting_agent
        queries = reporting_agent.METRIC_QUERIES
        assert "pipeline" in queries
        assert "revenue" in queries
        assert "outreach" in queries
        assert "client" in queries
        assert "table" in queries["pipeline"]

    def test_report_formats(self):
        from core.agents.implementations import reporting_agent
        formats = reporting_agent.REPORT_FORMATS
        assert "executive" in formats
        assert "detailed" in formats
        assert "weekly" in formats
        assert "monthly" in formats
        assert formats["executive"]["max_length"] == 2000

    def test_trend_analysis_prompt(self):
        from core.agents.implementations import reporting_agent
        prompt = reporting_agent.TREND_ANALYSIS_PROMPT
        assert "{metrics_json}" in prompt
        assert "{period_start}" in prompt

    def test_report_generation_prompt(self):
        from core.agents.implementations import reporting_agent
        prompt = reporting_agent.REPORT_GENERATION_PROMPT
        assert "{report_format}" in prompt
        assert "{pipeline_json}" in prompt
        assert "{revenue_json}" in prompt

    # ─── Node 1: Collect Metrics ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_collect_metrics_empty_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.gte.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"report_format": "executive", "period_start": "2024-01-01", "period_end": "2024-01-31"},
            "run-1",
        )
        result = await agent._node_collect_metrics(state)
        assert result["current_node"] == "collect_metrics"
        assert result["report_format"] == "executive"
        assert result["period_start"] == "2024-01-01"
        assert result["pipeline_metrics"]["total_opportunities"] == 0
        assert result["revenue_metrics"]["total_revenue_cents"] == 0

    @pytest.mark.asyncio
    async def test_node_collect_metrics_with_data(self):
        agent = self._make_agent()

        mock_opps = MagicMock()
        mock_opps.data = [
            {"value_cents": 10000, "stage": "closed_won"},
            {"value_cents": 5000, "stage": "proposal"},
        ]
        mock_invoices = MagicMock()
        mock_invoices.data = [{"status": "paid", "total_amount_cents": 10000}]
        mock_events = MagicMock()
        mock_events.data = [{"event_type": "sent"}, {"event_type": "opened"}]
        mock_clients = MagicMock()
        mock_clients.data = [{"status": "active", "onboarded_at": "2024-01-15"}]

        agent.db.client.table.return_value.select.return_value.gte.return_value.execute.side_effect = [
            mock_opps, mock_invoices, mock_events,
        ]
        agent.db.client.table.return_value.select.return_value.execute.return_value = mock_clients

        state = agent._prepare_initial_state(
            {"report_format": "detailed", "period_start": "2024-01-01", "period_end": "2024-01-31"},
            "run-1",
        )
        result = await agent._node_collect_metrics(state)
        assert result["pipeline_metrics"]["total_opportunities"] == 2
        assert result["pipeline_metrics"]["total_value_cents"] == 15000

    @pytest.mark.asyncio
    async def test_node_collect_metrics_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.gte.side_effect = Exception("DB fail")
        agent.db.client.table.return_value.select.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"report_format": "weekly"}, "run-1"
        )
        result = await agent._node_collect_metrics(state)
        assert result["current_node"] == "collect_metrics"
        assert result["pipeline_metrics"]["total_opportunities"] == 0

    # ─── Node 2: Analyze Trends ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_trends_llm_success(self):
        agent = self._make_agent()
        analysis = {
            "trends": [{"metric": "revenue", "direction": "up", "magnitude": 0.15, "insight": "Growing"}],
            "forecasts": [{"metric": "revenue", "forecast_value": 50000, "confidence": 0.8}],
            "anomalies": [{"metric": "churn", "expected": 5, "actual": 15, "severity": "high"}],
            "executive_summary": "Business is growing.",
        }
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(analysis)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = {
            "pipeline_metrics": {},
            "revenue_metrics": {},
            "outreach_metrics": {},
            "client_metrics": {},
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
        }
        result = await agent._node_analyze_trends(state)
        assert result["current_node"] == "analyze_trends"
        assert len(result["trends"]) == 1
        assert len(result["forecasts"]) == 1
        assert len(result["anomalies"]) == 1

    @pytest.mark.asyncio
    async def test_node_analyze_trends_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = {
            "pipeline_metrics": {},
            "revenue_metrics": {},
            "outreach_metrics": {},
            "client_metrics": {},
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
        }
        result = await agent._node_analyze_trends(state)
        assert result["trends"] == []
        assert result["forecasts"] == []
        assert result["anomalies"] == []

    # ─── Node 3: Generate Report ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_report_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "# Business Report\n\n## Executive Summary\nRevenue is up..."
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = {
            "report_format": "executive",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "pipeline_metrics": {},
            "revenue_metrics": {},
            "outreach_metrics": {},
            "client_metrics": {},
            "trends": [],
            "anomalies": [],
        }
        result = await agent._node_generate_report(state)
        assert result["current_node"] == "generate_report"
        assert "Business Report" in result["report_document"]
        assert len(result["report_sections"]) > 0

    @pytest.mark.asyncio
    async def test_node_generate_report_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("timeout")

        state = {
            "report_format": "executive",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "pipeline_metrics": {"total_opportunities": 10, "total_value_cents": 50000},
            "revenue_metrics": {"total_revenue_cents": 30000, "outstanding_cents": 5000},
            "outreach_metrics": {},
            "client_metrics": {},
            "trends": [],
            "anomalies": [],
        }
        result = await agent._node_generate_report(state)
        assert "Business Report" in result["report_document"]
        assert "Pipeline" in result["report_document"]

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"report_format": "executive", "anomalies": []}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "rpt_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["report_format"] = "executive"
        state["period_start"] = "2024-01-01"
        state["period_end"] = "2024-01-31"
        state["pipeline_metrics"] = {"total_opportunities": 5, "total_value_cents": 10000}
        state["revenue_metrics"] = {"total_revenue_cents": 8000}
        state["outreach_metrics"] = {}
        state["client_metrics"] = {}
        state["trends"] = [{"metric": "revenue", "direction": "up"}]
        state["anomalies"] = []
        state["report_document"] = "# Report"
        state["forecasts"] = []
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["report_saved"] is True
        assert result["report_id"] == "rpt_1"
        assert "Report Generation Summary" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["report_format"] = "weekly"
        state["pipeline_metrics"] = {"total_opportunities": 0, "total_value_cents": 0}
        state["revenue_metrics"] = {"total_revenue_cents": 0}
        state["outreach_metrics"] = {}
        state["client_metrics"] = {}
        state["trends"] = []
        state["anomalies"] = []
        state["forecasts"] = []
        state["report_document"] = ""
        result = await agent._node_report(state)
        assert result["report_saved"] is False
        assert result["report_id"] == ""

    # ─── Routing ──────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.reporting_agent import ReportingAgent
        state = {"human_approval_status": "approved"}
        assert ReportingAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.reporting_agent import ReportingAgent
        state = {"human_approval_status": "rejected"}
        assert ReportingAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.reporting_agent import ReportingAgent
        state = {}
        assert ReportingAgent._route_after_review(state) == "approved"
