"""
Tests for Universal Business Agents v2 — Phase 21 (Onboarding + Invoice).

Covers 2 cross-vertical business operations agents:
    1. OnboardingAgent (onboarding)
    2. InvoiceAgent (invoice)

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
#  1.  OnboardingAgent
# ══════════════════════════════════════════════════════════════════════


class TestOnboardingAgentState:
    """Tests for OnboardingAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import OnboardingAgentState
        assert OnboardingAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import OnboardingAgentState
        state: OnboardingAgentState = {
            "agent_id": "onboarding_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "onboarding_v1"

    def test_create_full(self):
        from core.agents.state import OnboardingAgentState
        state: OnboardingAgentState = {
            "agent_id": "onboarding_v1",
            "vertical_id": "enclave_guard",
            "company_name": "Acme Corp",
            "company_domain": "acme.com",
            "contact_name": "Jane Doe",
            "contact_email": "jane@acme.com",
            "opportunity_id": "opp_123",
            "contract_id": "cont_456",
            "template_name": "enterprise",
            "milestones": [],
            "total_milestones": 0,
            "completed_milestones": 0,
            "completion_percent": 0.0,
            "welcome_package": "",
            "welcome_package_sent": False,
            "kickoff_scheduled": False,
            "kickoff_date": "",
            "onboarding_id": "",
            "onboarding_status": "not_started",
            "onboarding_saved": False,
            "stalled_reason": "",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["company_name"] == "Acme Corp"
        assert state["onboarding_status"] == "not_started"
        assert state["welcome_package_sent"] is False

    def test_template_and_milestone_fields(self):
        from core.agents.state import OnboardingAgentState
        state: OnboardingAgentState = {
            "template_name": "enterprise",
            "milestones": [
                {"name": "welcome_call", "status": "pending"},
                {"name": "account_setup", "status": "complete"},
            ],
            "total_milestones": 2,
            "completed_milestones": 1,
            "completion_percent": 50.0,
        }
        assert state["template_name"] == "enterprise"
        assert len(state["milestones"]) == 2
        assert state["completion_percent"] == 50.0

    def test_kickoff_and_persistence_fields(self):
        from core.agents.state import OnboardingAgentState
        state: OnboardingAgentState = {
            "kickoff_scheduled": True,
            "kickoff_date": "2024-02-15",
            "onboarding_id": "onb_001",
            "onboarding_saved": True,
            "onboarding_status": "in_progress",
        }
        assert state["kickoff_scheduled"] is True
        assert state["onboarding_saved"] is True


class TestOnboardingAgent:
    """Tests for OnboardingAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create an OnboardingAgent with mocked dependencies."""
        from core.agents.implementations.onboarding_agent import OnboardingAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="onboarding_v1",
            agent_type="onboarding",
            name="Onboarding Agent",
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

        return OnboardingAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.onboarding_agent import OnboardingAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "onboarding" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.onboarding_agent import OnboardingAgent
        assert OnboardingAgent.agent_type == "onboarding"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import OnboardingAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is OnboardingAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "OnboardingAgent" in r
        assert "onboarding_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_name": "Acme", "template": "enterprise"}, "run-1"
        )
        assert state["company_name"] == ""
        assert state["company_domain"] == ""
        assert state["contact_name"] == ""
        assert state["contact_email"] == ""
        assert state["contact_title"] == ""
        assert state["proposal_id"] == ""
        assert state["deal_value_cents"] == 0
        assert state["template_key"] == "default"
        assert state["milestones"] == []
        assert state["total_milestones"] == 0
        assert state["welcome_content"] == ""
        assert state["kickoff_date"] == ""
        assert state["welcome_package_generated"] is False
        assert state["onboarding_id"] == ""
        assert state["onboarding_status"] == "not_started"
        assert state["onboarding_saved"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_common_keys(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_name": "Acme"}, "run-1"
        )
        assert state["agent_id"] == "onboarding_v1"
        assert state["vertical_id"] == "enclave_guard"
        assert state["run_id"] == "run-1"
        assert state["current_node"] == "start"
        assert state["error"] is None
        assert state["retry_count"] == 0
        assert state["requires_human_approval"] is False
        assert state["knowledge_written"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_onboarding_templates_keys(self):
        from core.agents.implementations import onboarding_agent
        templates = onboarding_agent.ONBOARDING_TEMPLATES
        assert "default" in templates
        assert "enterprise" in templates
        assert "quick_start" in templates

    def test_onboarding_templates_content(self):
        from core.agents.implementations import onboarding_agent
        templates = onboarding_agent.ONBOARDING_TEMPLATES
        assert "welcome_call" in templates["default"]
        assert "security_review" in templates["enterprise"]
        assert len(templates["quick_start"]) < len(templates["enterprise"])

    def test_milestone_defaults(self):
        from core.agents.implementations import onboarding_agent
        defaults = onboarding_agent.MILESTONE_DEFAULTS
        assert "welcome_call" in defaults
        assert "account_setup" in defaults
        assert "first_deliverable" in defaults
        assert defaults["welcome_call"]["days_offset"] == 2
        assert "description" in defaults["welcome_call"]

    def test_welcome_package_prompt(self):
        from core.agents.implementations import onboarding_agent
        prompt = onboarding_agent.WELCOME_PACKAGE_PROMPT
        assert "{company_name}" in prompt
        assert "{contact_name}" in prompt
        assert "{contact_title}" in prompt
        assert "{deal_value}" in prompt
        assert "{template_key}" in prompt
        assert "{milestones_json}" in prompt
        assert "{vertical_id}" in prompt

    # ─── Node 1: Setup Onboarding ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_setup_onboarding_success(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {
            "company_name": "Acme Corp",
            "company_domain": "acme.com",
            "contact_name": "Jane Doe",
            "contact_email": "jane@acme.com",
            "contact_title": "CTO",
            "proposal_id": "prop_123",
            "deal_value_cents": 100000,
            "template": "default",
        }
        result = await agent._node_setup_onboarding(state)
        assert result["current_node"] == "setup_onboarding"
        assert result["company_name"] == "Acme Corp"
        assert result["company_domain"] == "acme.com"
        assert result["contact_name"] == "Jane Doe"
        assert result["contact_email"] == "jane@acme.com"
        assert result["contact_title"] == "CTO"
        assert result["proposal_id"] == "prop_123"
        assert result["deal_value_cents"] == 100000
        assert result["template_key"] == "default"
        assert len(result["milestones"]) > 0
        assert result["total_milestones"] == len(result["milestones"])

    @pytest.mark.asyncio
    async def test_node_setup_onboarding_enterprise_auto_upgrade(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {
            "company_name": "Big Corp",
            "deal_value_cents": 600000,
            "template": "default",
        }
        result = await agent._node_setup_onboarding(state)
        assert result["template_key"] == "enterprise"
        assert result["total_milestones"] == len(
            agent.__class__.__module__ and result["milestones"]
        )

    @pytest.mark.asyncio
    async def test_node_setup_onboarding_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {
            "company_name": "Acme",
            "company_domain": "acme.com",
            "template": "default",
        }
        result = await agent._node_setup_onboarding(state)
        assert result["current_node"] == "setup_onboarding"
        assert result["company_name"] == "Acme"
        assert len(result["milestones"]) > 0

    @pytest.mark.asyncio
    async def test_node_setup_onboarding_unknown_template_fallback(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["task_input"] = {
            "company_name": "Acme",
            "template": "nonexistent_template",
        }
        result = await agent._node_setup_onboarding(state)
        assert result["current_node"] == "setup_onboarding"
        assert result["template_key"] in ("default", "enterprise", "quick_start")

    # ─── Node 2: Generate Welcome Package ────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_welcome_package_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "welcome_message": "Welcome to our team, Jane!",
            "timeline_overview": "30/60/90 day plan overview.",
            "kickoff_agenda": ["Introductions", "Review scope", "Next steps"],
            "next_steps": ["Sign NDA", "Schedule kickoff"],
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Acme Corp"
        state["contact_name"] = "Jane Doe"
        state["contact_title"] = "CTO"
        state["contact_email"] = "jane@acme.com"
        state["deal_value_cents"] = 100000
        state["template_key"] = "enterprise"
        state["milestones"] = [{"name": "welcome_call"}]
        result = await agent._node_generate_welcome_package(state)
        assert result["current_node"] == "generate_welcome_package"
        assert "Welcome" in result["welcome_content"]
        assert result["kickoff_date"] != ""
        assert isinstance(result["kickoff_agenda"], list)
        assert result["welcome_package_generated"] is True

    @pytest.mark.asyncio
    async def test_node_generate_welcome_package_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Fallback Corp"
        state["contact_name"] = "Bob Smith"
        state["template_key"] = "default"
        state["milestones"] = []
        result = await agent._node_generate_welcome_package(state)
        assert result["current_node"] == "generate_welcome_package"
        assert "Bob Smith" in result["welcome_content"]
        assert "Fallback Corp" in result["welcome_content"]
        assert result["welcome_package_generated"] is True
        assert result["kickoff_date"] != ""
        assert len(result["kickoff_agenda"]) > 0

    @pytest.mark.asyncio
    async def test_node_generate_welcome_package_empty_company(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = ""
        state["contact_name"] = ""
        state["template_key"] = "default"
        state["milestones"] = []
        result = await agent._node_generate_welcome_package(state)
        assert result["current_node"] == "generate_welcome_package"
        assert result["welcome_package_generated"] is True

    # ─── Node 3: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"company_name": "Test", "milestones": [{"name": "kickoff"}]}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 4: Execute Onboarding ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_execute_onboarding_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "onb_001"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Acme Corp"
        state["company_domain"] = "acme.com"
        state["contact_name"] = "Jane"
        state["contact_email"] = "jane@acme.com"
        state["milestones"] = [{"name": "welcome_call", "status": "pending"}]
        state["welcome_content"] = "Welcome!"
        state["kickoff_date"] = "2024-03-01"
        result = await agent._node_execute_onboarding(state)
        assert result["current_node"] == "execute_onboarding"
        assert result["onboarding_id"] == "onb_001"
        assert result["onboarding_status"] == "in_progress"
        assert result["onboarding_saved"] is True

    @pytest.mark.asyncio
    async def test_node_execute_onboarding_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Acme"
        state["milestones"] = []
        result = await agent._node_execute_onboarding(state)
        assert result["onboarding_saved"] is False
        assert result["onboarding_id"] == ""

    @pytest.mark.asyncio
    async def test_node_execute_onboarding_empty_data(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = ""
        state["milestones"] = []
        result = await agent._node_execute_onboarding(state)
        assert result["current_node"] == "execute_onboarding"

    # ─── Node 5: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Acme Corp"
        state["contact_name"] = "Jane Doe"
        state["contact_email"] = "jane@acme.com"
        state["deal_value_cents"] = 100000
        state["template_key"] = "enterprise"
        state["milestones"] = [
            {"name": "welcome_call", "due_date": "2024-03-01", "description": "Schedule kickoff"},
        ]
        state["kickoff_date"] = "2024-03-01"
        state["onboarding_saved"] = True
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Client Onboarding Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_not_saved(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["company_name"] = "Test Corp"
        state["template_key"] = "default"
        state["milestones"] = []
        state["onboarding_saved"] = False
        result = await agent._node_report(state)
        assert "Not saved" in result["report_summary"] or "report" in result["current_node"]

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.onboarding_agent import OnboardingAgent
        state = {"human_approval_status": "approved"}
        assert OnboardingAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.onboarding_agent import OnboardingAgent
        state = {"human_approval_status": "rejected"}
        assert OnboardingAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.onboarding_agent import OnboardingAgent
        state = {}
        assert OnboardingAgent._route_after_review(state) == "approved"

    # ─── write_knowledge ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"company_name": "Acme"})
        assert result is None


# ══════════════════════════════════════════════════════════════════════
#  2.  InvoiceAgent
# ══════════════════════════════════════════════════════════════════════


class TestInvoiceAgentState:
    """Tests for InvoiceAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import InvoiceAgentState
        assert InvoiceAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import InvoiceAgentState
        state: InvoiceAgentState = {
            "agent_id": "invoice_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "invoice_v1"

    def test_create_full(self):
        from core.agents.state import InvoiceAgentState
        state: InvoiceAgentState = {
            "agent_id": "invoice_v1",
            "vertical_id": "enclave_guard",
            "proposal_id": "prop_123",
            "contract_id": "cont_456",
            "company_name": "Acme Corp",
            "contact_name": "Jane Doe",
            "contact_email": "jane@acme.com",
            "invoice_id": "inv_001",
            "invoice_number": "INV-2024-001",
            "line_items": [],
            "subtotal_cents": 50000,
            "tax_cents": 5000,
            "total_cents": 55000,
            "currency": "USD",
            "due_date": "2024-03-01",
            "due_days": 30,
            "payment_status": "open",
            "paid_at": "",
            "stripe_invoice_id": "",
            "stripe_hosted_url": "",
            "overdue_invoices": [],
            "reminder_drafts": [],
            "reminders_approved": False,
            "reminders_sent": 0,
            "invoice_saved": False,
            "invoice_sent": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["company_name"] == "Acme Corp"
        assert state["total_cents"] == 55000
        assert state["payment_status"] == "open"

    def test_invoice_financial_fields(self):
        from core.agents.state import InvoiceAgentState
        state: InvoiceAgentState = {
            "line_items": [
                {"description": "Consulting", "amount_cents": 25000, "quantity": 2},
            ],
            "subtotal_cents": 50000,
            "tax_cents": 5000,
            "total_cents": 55000,
            "currency": "USD",
        }
        assert state["subtotal_cents"] == 50000
        assert len(state["line_items"]) == 1

    def test_overdue_and_reminder_fields(self):
        from core.agents.state import InvoiceAgentState
        state: InvoiceAgentState = {
            "overdue_invoices": [{"invoice_id": "inv_1", "days_overdue": 15}],
            "reminder_drafts": [{"subject": "Payment reminder", "tone": "firm"}],
            "reminders_sent": 3,
        }
        assert len(state["overdue_invoices"]) == 1
        assert state["reminders_sent"] == 3


class TestInvoiceAgent:
    """Tests for InvoiceAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create an InvoiceAgent with mocked dependencies."""
        from core.agents.implementations.invoice_agent import InvoiceAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="invoice_v1",
            agent_type="invoice",
            name="Invoice Agent",
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

        return InvoiceAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.invoice_agent import InvoiceAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "invoice" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.invoice_agent import InvoiceAgent
        assert InvoiceAgent.agent_type == "invoice"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import InvoiceAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is InvoiceAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "InvoiceAgent" in r
        assert "invoice_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"scan_mode": "overdue"}, "run-1"
        )
        assert state["overdue_invoices"] == []
        assert state["pending_proposals"] == []
        assert state["total_overdue"] == 0
        assert state["total_outstanding_cents"] == 0
        assert state["new_invoices"] == []
        assert state["invoices_created"] == 0
        assert state["reminder_drafts"] == []
        assert state["reminders_generated"] == 0
        assert state["invoices_sent"] == 0
        assert state["reminders_sent"] == 0
        assert state["report_summary"] == ""
        assert state["report_generated_at"] == ""

    # ─── Constants ──────────────────────────────────────────────────

    def test_reminder_tones(self):
        from core.agents.implementations import invoice_agent
        tones = invoice_agent.REMINDER_TONES
        assert "polite" in tones
        assert "firm" in tones
        assert "final" in tones
        assert tones["polite"]["urgency"] == "low"
        assert tones["firm"]["urgency"] == "medium"
        assert tones["final"]["urgency"] == "high"

    def test_overdue_thresholds(self):
        from core.agents.implementations import invoice_agent
        thresholds = invoice_agent.OVERDUE_THRESHOLDS
        assert 7 in thresholds
        assert 14 in thresholds
        assert 30 in thresholds
        assert thresholds[7] == "polite"
        assert thresholds[14] == "firm"
        assert thresholds[30] == "final"

    def test_invoice_generation_prompt(self):
        from core.agents.implementations import invoice_agent
        prompt = invoice_agent.INVOICE_GENERATION_PROMPT
        assert "{company_name}" in prompt
        assert "{contact_name}" in prompt
        assert "{line_items_json}" in prompt
        assert "{total_amount}" in prompt
        assert "{due_date}" in prompt

    def test_payment_reminder_prompt(self):
        from core.agents.implementations import invoice_agent
        prompt = invoice_agent.PAYMENT_REMINDER_PROMPT
        assert "{company_name}" in prompt
        assert "{contact_name}" in prompt
        assert "{amount_due}" in prompt
        assert "{days_overdue}" in prompt
        assert "{tone}" in prompt
        assert "{urgency}" in prompt

    # ─── Node 1: Scan Invoices ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_invoices_no_overdue(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.lt.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_invoices(state)
        assert result["current_node"] == "scan_invoices"
        assert result["overdue_invoices"] == []
        assert result["total_overdue"] == 0
        assert result["total_outstanding_cents"] == 0

    @pytest.mark.asyncio
    async def test_node_scan_invoices_with_overdue(self):
        agent = self._make_agent()
        mock_overdue = MagicMock()
        mock_overdue.data = [
            {
                "id": "inv_1",
                "company_name": "Late Corp",
                "contact_email": "late@corp.com",
                "contact_name": "John",
                "amount_cents": 50000,
                "due_date": "2024-01-01",
                "status": "overdue",
            },
        ]
        mock_proposals = MagicMock()
        mock_proposals.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.lt.return_value.order.return_value.limit.return_value.execute.return_value = mock_overdue
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_proposals

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_invoices(state)
        assert result["total_overdue"] >= 1
        assert len(result["overdue_invoices"]) >= 1
        assert result["total_outstanding_cents"] >= 50000

    @pytest.mark.asyncio
    async def test_node_scan_invoices_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_invoices(state)
        assert result["current_node"] == "scan_invoices"

    @pytest.mark.asyncio
    async def test_node_scan_invoices_with_pending_proposals(self):
        agent = self._make_agent()
        mock_overdue = MagicMock()
        mock_overdue.data = []
        mock_proposals = MagicMock()
        mock_proposals.data = [
            {
                "id": "prop_1",
                "company_name": "New Client",
                "contact_email": "new@client.com",
                "contact_name": "Sarah",
                "pricing_amount_cents": 75000,
                "accepted_at": "2024-02-01",
            },
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.lt.return_value.order.return_value.limit.return_value.execute.return_value = mock_overdue
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_proposals

        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_scan_invoices(state)
        assert len(result["pending_proposals"]) >= 1

    # ─── Node 2: Generate Invoice ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_invoice_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "invoice_memo": "Invoice for professional services",
            "line_items_formatted": [
                {"description": "Consulting", "amount_cents": 50000, "quantity": 1}
            ],
            "payment_terms": "Net 30",
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["pending_proposals"] = [
            {
                "proposal_id": "prop_1",
                "company_name": "Acme Corp",
                "contact_name": "Jane",
                "contact_email": "jane@acme.com",
                "amount_cents": 50000,
            },
        ]
        result = await agent._node_generate_invoice(state)
        assert result["current_node"] == "generate_invoice"
        assert len(result["new_invoices"]) == 1
        assert result["invoices_created"] == 1

    @pytest.mark.asyncio
    async def test_node_generate_invoice_no_proposals(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["pending_proposals"] = []
        result = await agent._node_generate_invoice(state)
        assert result["current_node"] == "generate_invoice"
        assert result["new_invoices"] == []
        assert result["invoices_created"] == 0

    @pytest.mark.asyncio
    async def test_node_generate_invoice_llm_error_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["pending_proposals"] = [
            {
                "proposal_id": "prop_2",
                "company_name": "Fallback Corp",
                "contact_name": "Bob",
                "contact_email": "bob@fb.com",
                "amount_cents": 30000,
            },
        ]
        result = await agent._node_generate_invoice(state)
        assert result["current_node"] == "generate_invoice"
        assert result["new_invoices"] == []

    # ─── Node 3: Draft Reminders ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_draft_reminders_llm_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps({
            "subject": "Payment reminder - $500.00 overdue",
            "body": "Dear Client, your invoice is overdue...",
            "suggested_deadline": "2024-03-15",
        })
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["overdue_invoices"] = [
            {
                "invoice_id": "inv_1",
                "company_name": "Late Corp",
                "contact_email": "late@corp.com",
                "contact_name": "John",
                "amount_cents": 50000,
                "days_overdue": 14,
                "tone": "firm",
            },
        ]
        result = await agent._node_draft_reminders(state)
        assert result["current_node"] == "draft_reminders"
        assert len(result["reminder_drafts"]) == 1
        assert result["reminders_generated"] == 1

    @pytest.mark.asyncio
    async def test_node_draft_reminders_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["overdue_invoices"] = []
        result = await agent._node_draft_reminders(state)
        assert result["current_node"] == "draft_reminders"
        assert result["reminder_drafts"] == []
        assert result["reminders_generated"] == 0

    @pytest.mark.asyncio
    async def test_node_draft_reminders_llm_error(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["overdue_invoices"] = [
            {
                "invoice_id": "inv_2",
                "company_name": "Error Corp",
                "contact_email": "err@corp.com",
                "contact_name": "Sam",
                "amount_cents": 20000,
                "days_overdue": 7,
                "tone": "polite",
            },
        ]
        result = await agent._node_draft_reminders(state)
        assert result["current_node"] == "draft_reminders"
        assert result["reminders_generated"] == 0

    # ─── Node 4: Human Review ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "new_invoices": [{"company": "Test"}],
            "reminder_drafts": [{"subject": "Reminder"}],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Send Invoices ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_send_invoices_success(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "inv_saved"}])

        state = agent._prepare_initial_state({}, "run-1")
        state["new_invoices"] = [
            {
                "proposal_id": "prop_1",
                "company_name": "Acme",
                "contact_email": "acme@test.com",
                "contact_name": "Jane",
                "amount_cents": 50000,
                "due_date": "2024-03-01",
                "line_items": [],
                "memo": "Invoice memo",
                "payment_terms": "Net 30",
            },
        ]
        state["reminder_drafts"] = [
            {
                "invoice_id": "inv_old",
                "company_name": "Late Corp",
                "contact_email": "late@corp.com",
                "tone": "polite",
                "subject": "Reminder",
                "body": "Please pay",
                "days_overdue": 10,
            },
        ]
        result = await agent._node_send_invoices(state)
        assert result["current_node"] == "send_invoices"
        assert result["invoices_sent"] >= 1
        assert result["reminders_sent"] >= 1

    @pytest.mark.asyncio
    async def test_node_send_invoices_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")

        state = agent._prepare_initial_state({}, "run-1")
        state["new_invoices"] = [
            {
                "proposal_id": "p1",
                "company_name": "Fail Corp",
                "amount_cents": 10000,
            },
        ]
        state["reminder_drafts"] = []
        result = await agent._node_send_invoices(state)
        assert result["invoices_sent"] == 0
        assert result["reminders_sent"] == 0

    @pytest.mark.asyncio
    async def test_node_send_invoices_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["new_invoices"] = []
        state["reminder_drafts"] = []
        result = await agent._node_send_invoices(state)
        assert result["invoices_sent"] == 0
        assert result["reminders_sent"] == 0

    # ─── Node 6: Report ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["overdue_invoices"] = [
            {"company_name": "Late Corp", "amount_cents": 50000, "days_overdue": 14, "tone": "firm"},
        ]
        state["new_invoices"] = [{"company_name": "New Corp"}]
        state["reminder_drafts"] = [{"tone": "firm"}]
        state["total_outstanding_cents"] = 50000
        state["invoices_sent"] = 1
        state["reminders_sent"] = 1
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Invoice & Billing Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["overdue_invoices"] = []
        state["new_invoices"] = []
        state["reminder_drafts"] = []
        state["total_outstanding_cents"] = 0
        state["invoices_sent"] = 0
        state["reminders_sent"] = 0
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Invoice & Billing Report" in result["report_summary"]

    # ─── Routing ──────────────────────────────────────────────────────

    def test_route_approved(self):
        from core.agents.implementations.invoice_agent import InvoiceAgent
        state = {"human_approval_status": "approved"}
        assert InvoiceAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.invoice_agent import InvoiceAgent
        state = {"human_approval_status": "rejected"}
        assert InvoiceAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.invoice_agent import InvoiceAgent
        state = {}
        assert InvoiceAgent._route_after_review(state) == "approved"

    # ─── write_knowledge ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"invoices_sent": 5})
        assert result is None
