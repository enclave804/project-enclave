"""
Tests for Phase 12: Operations Agents.

Covers:
    - Stripe Invoice Client (mock mode, customer ops, invoice ops)
    - Finance MCP tools (invoice generation, overdue check, reminders, P&L)
    - Finance Agent (state, contracts, graph, nodes, routing)
    - Customer Success Agent (state, contracts, graph, nodes, routing)
    - Operations Dashboard helpers (invoice, reminder, client, interaction stats)
    - YAML configs (finance.yaml, cs.yaml)
    - DB migration schema (009_operations.sql)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
# State Tests
# ══════════════════════════════════════════════════════════════════════


class TestFinanceState:
    """Tests for FinanceAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import FinanceAgentState
        assert FinanceAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import FinanceAgentState
        state: FinanceAgentState = {
            "agent_id": "finance_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "finance_v1"

    def test_create_full(self):
        from core.agents.state import FinanceAgentState
        state: FinanceAgentState = {
            "agent_id": "finance_v1",
            "vertical_id": "enclave_guard",
            "active_invoices": [{"id": "inv_1"}],
            "overdue_invoices": [],
            "paid_invoices": [{"id": "inv_2", "total": 5000}],
            "new_proposal_ids": ["prop_1"],
            "invoices_to_create": [],
            "invoices_created": [],
            "invoices_sent": 1,
            "reminder_drafts": [],
            "reminders_approved": False,
            "reminders_sent": 0,
            "total_revenue": 5000.0,
            "accounts_receivable": 1500.0,
            "monthly_recurring": 0.0,
            "service_revenue": 5000.0,
            "commerce_revenue": 0.0,
            "total_costs": 500.0,
            "net_profit": 4500.0,
            "pnl_data": {"period": "2025-01"},
        }
        assert state["total_revenue"] == 5000.0
        assert state["net_profit"] == 4500.0
        assert len(state["active_invoices"]) == 1

    def test_invoice_tracking_fields(self):
        from core.agents.state import FinanceAgentState
        state: FinanceAgentState = {
            "overdue_invoices": [
                {"id": "inv_3", "days_overdue": 15}
            ],
            "reminder_drafts": [
                {"invoice_id": "inv_3", "tone": "firm"}
            ],
        }
        assert state["overdue_invoices"][0]["days_overdue"] == 15
        assert state["reminder_drafts"][0]["tone"] == "firm"

    def test_pnl_fields(self):
        from core.agents.state import FinanceAgentState
        state: FinanceAgentState = {
            "service_revenue": 10000.0,
            "commerce_revenue": 2000.0,
            "total_costs": 3000.0,
            "net_profit": 9000.0,
            "pnl_data": {"service": 10000, "commerce": 2000},
        }
        assert state["pnl_data"]["service"] == 10000


class TestCustomerSuccessState:
    """Tests for CustomerSuccessAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import CustomerSuccessAgentState
        assert CustomerSuccessAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import CustomerSuccessAgentState
        state: CustomerSuccessAgentState = {
            "agent_id": "cs_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "cs_v1"

    def test_create_full(self):
        from core.agents.state import CustomerSuccessAgentState
        state: CustomerSuccessAgentState = {
            "agent_id": "cs_v1",
            "vertical_id": "enclave_guard",
            "active_clients": [{"company": "Acme"}],
            "new_clients": [{"company": "NewCo"}],
            "at_risk_clients": [],
            "client_id": "client_1",
            "onboarding_email_draft": "Welcome!",
            "onboarding_checklist": ["Step 1"],
            "kickoff_meeting_requested": True,
            "kickoff_meeting_scheduled": False,
            "welcome_packet_sent": True,
            "checkin_type": "30_day",
            "checkin_drafts": [{"to": "jane@acme.com", "body": "Hi"}],
            "last_contact_dates": {"acme.com": "2025-01-01"},
            "sentiment_scores": {"acme.com": 0.8},
            "churn_risk_score": 0.3,
            "retention_actions": ["schedule_call"],
            "cs_actions_approved": False,
            "human_edits": [],
            "emails_sent": 0,
            "meetings_scheduled": 0,
        }
        assert state["checkin_type"] == "30_day"
        assert state["churn_risk_score"] == 0.3
        assert len(state["active_clients"]) == 1

    def test_churn_risk_field(self):
        from core.agents.state import CustomerSuccessAgentState
        state: CustomerSuccessAgentState = {
            "churn_risk_score": 0.7,
            "at_risk_clients": [{"company": "RiskyCo", "risk": 0.7}],
        }
        assert state["churn_risk_score"] == 0.7

    def test_onboarding_fields(self):
        from core.agents.state import CustomerSuccessAgentState
        state: CustomerSuccessAgentState = {
            "onboarding_email_draft": "Welcome to Enclave Guard!",
            "onboarding_checklist": ["Welcome email", "Kickoff call"],
            "welcome_packet_sent": False,
        }
        assert len(state["onboarding_checklist"]) == 2


# ══════════════════════════════════════════════════════════════════════
# Contract Tests
# ══════════════════════════════════════════════════════════════════════


class TestFinanceContracts:
    """Tests for Finance-related contracts."""

    def test_invoice_request(self):
        from core.agents.contracts import InvoiceRequest
        req = InvoiceRequest(
            proposal_id="prop_1",
            company_name="Acme Corp",
            contact_email="billing@acme.com",
            contact_name="Jane Doe",
            line_items=[{"description": "Assessment", "amount": 1500000, "quantity": 1}],
            total_amount=15000.0,
            currency="usd",
            due_days=30,
        )
        assert req.proposal_id == "prop_1"
        assert req.total_amount == 15000.0
        assert len(req.line_items) == 1

    def test_invoice_request_defaults(self):
        from core.agents.contracts import InvoiceRequest
        req = InvoiceRequest(
            proposal_id="prop_1",
            company_name="Acme",
            contact_email="jane@acme.com",
        )
        assert req.currency == "usd"
        assert req.due_days == 30
        assert req.line_items == []
        assert req.total_amount == 0.0

    def test_invoice_data(self):
        from core.agents.contracts import InvoiceData
        inv = InvoiceData(
            invoice_id="inv_test",
            proposal_id="prop_1",
            company_name="Acme Corp",
            contact_email="billing@acme.com",
            total_amount=15000.0,
            status="open",
            stripe_url="https://invoice.stripe.com/i/test",
        )
        assert inv.status == "open"
        assert inv.total_amount == 15000.0

    def test_invoice_data_status_options(self):
        from core.agents.contracts import InvoiceData
        for status in ["draft", "open", "paid", "overdue", "void"]:
            inv = InvoiceData(
                invoice_id="inv_test",
                company_name="Test",
                contact_email="test@test.com",
                status=status,
            )
            assert inv.status == status

    def test_payment_reminder(self):
        from core.agents.contracts import PaymentReminder
        rem = PaymentReminder(
            invoice_id="inv_1",
            company_name="Acme",
            contact_email="billing@acme.com",
            amount_due=15000.0,
            days_overdue=14,
            tone="polite",
            draft_text="Hi, this is a reminder...",
        )
        assert rem.tone == "polite"
        assert rem.days_overdue == 14

    def test_payment_reminder_tones(self):
        from core.agents.contracts import PaymentReminder
        for tone in ["polite", "firm", "final"]:
            rem = PaymentReminder(
                invoice_id="inv_1",
                company_name="Test Co",
                contact_email="test@test.com",
                tone=tone,
                draft_text="Test",
            )
            assert rem.tone == tone

    def test_client_record(self):
        from core.agents.contracts import ClientRecord
        rec = ClientRecord(
            client_id="client_1",
            company_name="Acme Corp",
            contact_email="jane@acme.com",
            contact_name="Jane Doe",
            status="active",
            sentiment_score=0.85,
        )
        assert rec.status == "active"
        assert rec.sentiment_score == 0.85

    def test_client_record_statuses(self):
        from core.agents.contracts import ClientRecord
        for status in ["active", "at_risk", "churned"]:
            rec = ClientRecord(
                client_id="test",
                company_name="Test",
                contact_email="test@test.com",
                status=status,
            )
            assert rec.status == status


# ══════════════════════════════════════════════════════════════════════
# Stripe Invoice Client Tests
# ══════════════════════════════════════════════════════════════════════


class TestStripeInvoiceClient:
    """Tests for StripeInvoiceClient in mock mode."""

    def _make_client(self):
        from core.integrations.finance.stripe_invoice_client import StripeInvoiceClient
        return StripeInvoiceClient(api_key=None, mock_mode=True)

    def test_from_env_mock_mode(self):
        """Without STRIPE_SECRET_KEY, client enters mock mode."""
        from core.integrations.finance.stripe_invoice_client import StripeInvoiceClient
        with patch.dict(os.environ, {}, clear=True):
            client = StripeInvoiceClient.from_env()
            assert client.mock_mode is True

    def test_from_env_live_mode(self):
        """With STRIPE_SECRET_KEY and stripe package, client uses live mode."""
        from core.integrations.finance.stripe_invoice_client import StripeInvoiceClient
        # If stripe package is not installed, it falls back to mock mode
        # So we just verify the from_env method works without error
        with patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_test_123"}):
            client = StripeInvoiceClient.from_env()
            # Will be mock_mode=True if stripe not installed, False if installed
            assert isinstance(client.mock_mode, bool)

    @pytest.mark.asyncio
    async def test_create_customer(self):
        client = self._make_client()
        result = await client.create_customer(
            email="jane@acme.com",
            name="Jane Doe",
            company="Acme Corp",
        )
        assert "id" in result
        assert result["id"].startswith("cus_mock_")
        assert result["email"] == "jane@acme.com"
        assert result["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_create_customer_metadata(self):
        client = self._make_client()
        result = await client.create_customer(
            email="test@test.com",
            metadata={"proposal_id": "prop_1"},
        )
        assert result["metadata"]["proposal_id"] == "prop_1"

    @pytest.mark.asyncio
    async def test_get_customer(self):
        client = self._make_client()
        created = await client.create_customer(email="jane@acme.com")
        fetched = await client.get_customer(created["id"])
        assert fetched["id"] == created["id"]
        assert fetched["email"] == "jane@acme.com"

    @pytest.mark.asyncio
    async def test_create_invoice(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@acme.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[
                {"description": "Assessment", "amount": 1500000, "quantity": 1},
            ],
            due_days=30,
            currency="usd",
        )
        assert "id" in invoice
        assert invoice["id"].startswith("inv_mock_")
        assert invoice["customer_id"] == customer["id"]
        assert invoice["total_amount"] == 1500000
        assert invoice["status"] == "draft"

    @pytest.mark.asyncio
    async def test_create_invoice_multiple_items(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[
                {"description": "Item 1", "amount": 500000, "quantity": 1},
                {"description": "Item 2", "amount": 300000, "quantity": 2},
            ],
        )
        assert invoice["total_amount"] == 500000 + 300000 * 2

    @pytest.mark.asyncio
    async def test_finalize_invoice(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@acme.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test", "amount": 1000, "quantity": 1}],
        )
        finalized = await client.finalize_invoice(invoice["id"])
        assert finalized["status"] == "open"
        assert "hosted_invoice_url" in finalized

    @pytest.mark.asyncio
    async def test_get_invoice(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test", "amount": 1000, "quantity": 1}],
        )
        fetched = await client.get_invoice(invoice["id"])
        assert fetched["id"] == invoice["id"]

    @pytest.mark.asyncio
    async def test_list_invoices_all(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test 1", "amount": 1000, "quantity": 1}],
        )
        await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test 2", "amount": 2000, "quantity": 1}],
        )
        all_invoices = await client.list_invoices()
        assert len(all_invoices) >= 2

    @pytest.mark.asyncio
    async def test_list_invoices_by_status(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        inv = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test", "amount": 1000, "quantity": 1}],
        )
        await client.finalize_invoice(inv["id"])
        open_invs = await client.list_invoices(status="open")
        assert all(i["status"] == "open" for i in open_invs)

    @pytest.mark.asyncio
    async def test_void_invoice(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test", "amount": 1000, "quantity": 1}],
        )
        await client.finalize_invoice(invoice["id"])
        voided = await client.void_invoice(invoice["id"])
        assert voided["status"] == "void"

    @pytest.mark.asyncio
    async def test_invoice_memo_and_metadata(self):
        client = self._make_client()
        customer = await client.create_customer(email="test@test.com")
        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=[{"description": "Test", "amount": 1000, "quantity": 1}],
            memo="For proposal prop_123",
            metadata={"proposal_id": "prop_123"},
        )
        assert invoice["memo"] == "For proposal prop_123"
        assert invoice["metadata"]["proposal_id"] == "prop_123"


# ══════════════════════════════════════════════════════════════════════
# Finance MCP Tools Tests
# ══════════════════════════════════════════════════════════════════════


class TestFinanceMCPTools:
    """Tests for finance MCP tool functions."""

    @pytest.mark.asyncio
    async def test_generate_invoice_from_proposal(self):
        from core.mcp.tools.finance_tools import generate_invoice_from_proposal
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.create_customer = AsyncMock(return_value={"id": "cus_1"})
            mock_client.create_invoice = AsyncMock(return_value={
                "id": "inv_1",
                "hosted_invoice_url": "https://pay.test",
            })
            mock_client.finalize_invoice = AsyncMock(return_value={
                "status": "open",
                "hosted_invoice_url": "https://pay.test",
            })
            mock_get.return_value = mock_client

            result_json = await generate_invoice_from_proposal(
                proposal_id="prop_1",
                company_name="Acme Corp",
                contact_email="billing@acme.com",
                amount_cents=1500000,
                description="Security Assessment",
            )
            result = json.loads(result_json)
            assert result["status"] == "success"
            assert result["invoice_id"] == "inv_1"
            assert result["total_amount_cents"] == 1500000

    @pytest.mark.asyncio
    async def test_generate_invoice_error(self):
        from core.mcp.tools.finance_tools import generate_invoice_from_proposal
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.create_customer = AsyncMock(side_effect=Exception("Stripe down"))
            mock_get.return_value = mock_client

            result_json = await generate_invoice_from_proposal(
                proposal_id="prop_1",
                company_name="Test",
                contact_email="test@test.com",
            )
            result = json.loads(result_json)
            assert result["status"] == "error"
            assert "Stripe down" in result["error"]

    @pytest.mark.asyncio
    async def test_check_overdue_invoices(self):
        from core.mcp.tools.finance_tools import check_overdue_invoices
        past_due = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.list_invoices = AsyncMock(return_value=[
                {"id": "inv_1", "due_date": past_due, "amount_due": 150000},
                {"id": "inv_2", "due_date": datetime.now(timezone.utc).isoformat(), "amount_due": 50000},
            ])
            mock_get.return_value = mock_client

            result_json = await check_overdue_invoices(days_overdue=7)
            result = json.loads(result_json)
            assert result["status"] == "success"
            assert result["overdue_count"] == 1
            assert result["total_overdue_cents"] == 150000

    @pytest.mark.asyncio
    async def test_check_overdue_no_invoices(self):
        from core.mcp.tools.finance_tools import check_overdue_invoices
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.list_invoices = AsyncMock(return_value=[])
            mock_get.return_value = mock_client

            result_json = await check_overdue_invoices()
            result = json.loads(result_json)
            assert result["overdue_count"] == 0

    @pytest.mark.asyncio
    async def test_send_payment_reminder_polite(self):
        from core.mcp.tools.finance_tools import send_payment_reminder
        result_json = await send_payment_reminder(
            invoice_id="inv_1",
            contact_email="billing@acme.com",
            company_name="Acme Corp",
            amount_display="$1,500.00",
            days_overdue=10,
            tone="polite",
        )
        result = json.loads(result_json)
        assert result["status"] == "draft_ready"
        assert result["tone"] == "polite"
        assert result["requires_human_review"] is True
        assert "friendly reminder" in result["draft_text"]

    @pytest.mark.asyncio
    async def test_send_payment_reminder_firm(self):
        from core.mcp.tools.finance_tools import send_payment_reminder
        result_json = await send_payment_reminder(
            invoice_id="inv_1",
            contact_email="billing@acme.com",
            company_name="Acme Corp",
            days_overdue=20,
            tone="firm",
        )
        result = json.loads(result_json)
        assert result["tone"] == "firm"
        assert "service interruption" in result["draft_text"]

    @pytest.mark.asyncio
    async def test_send_payment_reminder_final(self):
        from core.mcp.tools.finance_tools import send_payment_reminder
        result_json = await send_payment_reminder(
            invoice_id="inv_1",
            contact_email="billing@acme.com",
            company_name="Acme Corp",
            days_overdue=35,
            tone="final",
        )
        result = json.loads(result_json)
        assert result["tone"] == "final"
        assert "FINAL NOTICE" in result["draft_text"]

    @pytest.mark.asyncio
    async def test_get_monthly_pnl(self):
        from core.mcp.tools.finance_tools import get_monthly_pnl
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.list_invoices = AsyncMock(side_effect=[
                [{"total_amount": 1500000}],  # paid
                [{"amount_due": 500000}],      # open
            ])
            mock_get.return_value = mock_client

            result_json = await get_monthly_pnl()
            result = json.loads(result_json)
            assert result["status"] == "success"
            assert result["service_revenue_cents"] == 1500000
            assert result["accounts_receivable_cents"] == 500000

    @pytest.mark.asyncio
    async def test_get_monthly_pnl_error(self):
        from core.mcp.tools.finance_tools import get_monthly_pnl
        with patch("core.mcp.tools.finance_tools._get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.list_invoices = AsyncMock(side_effect=Exception("DB error"))
            mock_get.return_value = mock_client

            result_json = await get_monthly_pnl()
            result = json.loads(result_json)
            assert result["status"] == "error"


# ══════════════════════════════════════════════════════════════════════
# Finance Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestFinanceAgent:
    """Tests for FinanceAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a FinanceAgent with mocked dependencies."""
        from core.agents.implementations.finance_agent import FinanceAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="finance_v1",
            agent_type="finance",
            name="Finance Agent",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "default_currency": "usd",
                "default_due_days": 30,
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

        return FinanceAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    def test_registration(self):
        from core.agents.implementations.finance_agent import FinanceAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "finance" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.finance_agent import FinanceAgent
        assert FinanceAgent.agent_type == "finance"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import FinanceAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is FinanceAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "full_cycle", "proposal_ids": ["prop_1"]},
            "run-123",
        )
        assert state["active_invoices"] == []
        assert state["overdue_invoices"] == []
        assert state["reminder_drafts"] == []
        assert state["reminders_approved"] is False
        assert state["total_revenue"] == 0.0

    def test_prepare_initial_state_proposal_ids(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"proposal_ids": ["prop_1", "prop_2"]},
            "run-1",
        )
        assert state["new_proposal_ids"] == ["prop_1", "prop_2"]

    def test_constants(self):
        from core.agents.implementations import finance_agent
        assert finance_agent.MAX_INVOICE_AMOUNT == 100_000_00
        assert "full_cycle" in finance_agent.MODES
        assert "invoice_only" in finance_agent.MODES
        assert "overdue_check" in finance_agent.MODES
        assert "pnl_report" in finance_agent.MODES

    def test_reminder_tones(self):
        from core.agents.implementations import finance_agent
        tones = finance_agent.REMINDER_TONES
        # Should have polite, firm, final
        tone_values = set(tones.values())
        assert "polite" in tone_values
        assert "firm" in tone_values
        assert "final" in tone_values

    @pytest.mark.asyncio
    async def test_scan_invoices_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_cycle"}, "run-1")
        with patch("core.mcp.tools.finance_tools.check_overdue_invoices",
                    new_callable=AsyncMock,
                    return_value='{"status": "success", "overdue_count": 0, "invoices": []}'):
            with patch("core.mcp.tools.finance_tools.get_monthly_pnl",
                        new_callable=AsyncMock,
                        return_value='{"status": "success", "service_revenue_cents": 0}'):
                result = await agent._node_scan_invoices(state)
        assert result["current_node"] == "scan_invoices"

    @pytest.mark.asyncio
    async def test_identify_overdue_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_cycle"}, "run-1")
        with patch("core.mcp.tools.finance_tools.check_overdue_invoices",
                    new_callable=AsyncMock,
                    return_value=json.dumps({
                        "status": "success",
                        "overdue_count": 1,
                        "invoices": [{"id": "inv_1", "days_overdue": 15, "amount_due": 150000}],
                    })):
            result = await agent._node_identify_overdue(state)
        assert result["current_node"] == "identify_overdue"
        assert len(result["overdue_invoices"]) == 1

    @pytest.mark.asyncio
    async def test_draft_reminders_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_cycle"}, "run-1")
        state["overdue_invoices"] = [
            {
                "id": "inv_1",
                "days_overdue": 10,
                "amount_due": 150000,
                "customer_id": "cus_1",
            }
        ]
        with patch("core.mcp.tools.finance_tools.send_payment_reminder",
                    new_callable=AsyncMock,
                    return_value=json.dumps({
                        "status": "draft_ready",
                        "draft_text": "Hi, friendly reminder...",
                        "tone": "polite",
                    })):
            result = await agent._node_draft_reminders(state)
        assert result["current_node"] == "draft_reminders"
        assert len(result["reminder_drafts"]) >= 1

    @pytest.mark.asyncio
    async def test_human_review_node(self):
        agent = self._make_agent()
        state = {
            "reminder_drafts": [{"invoice_id": "inv_1", "tone": "polite"}],
            "overdue_invoices": [{"id": "inv_1"}],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    def test_route_after_review_approved(self):
        from core.agents.implementations.finance_agent import FinanceAgent
        assert FinanceAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.finance_agent import FinanceAgent
        assert FinanceAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    @pytest.mark.asyncio
    async def test_generate_pnl_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"mode": "full_cycle"}, "run-1")
        with patch("core.mcp.tools.finance_tools.get_monthly_pnl",
                    new_callable=AsyncMock,
                    return_value=json.dumps({
                        "status": "success",
                        "service_revenue_cents": 5000000,
                        "accounts_receivable_cents": 1000000,
                        "total_costs_cents": 500000,
                        "net_profit_cents": 4500000,
                        "period": "2025-01",
                    })):
            result = await agent._node_generate_pnl(state)
        assert result["current_node"] == "generate_pnl"
        assert result["service_revenue"] > 0

    @pytest.mark.asyncio
    async def test_report_node(self):
        agent = self._make_agent()
        state = {
            "active_invoices": [{"id": "inv_1"}],
            "overdue_invoices": [],
            "paid_invoices": [],
            "reminders_sent": 0,
            "invoices_sent": 0,
            "total_revenue": 5000.0,
            "accounts_receivable": 1000.0,
            "net_profit": 4000.0,
            "pnl_data": {},
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["report_summary"] != ""
        assert result["report_generated_at"] != ""

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "FinanceAgent" in repr_str
        assert "finance_v1" in repr_str


# ══════════════════════════════════════════════════════════════════════
# Customer Success Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestCustomerSuccessAgent:
    """Tests for CustomerSuccessAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a CustomerSuccessAgent with mocked dependencies."""
        from core.agents.implementations.cs_agent import CustomerSuccessAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="cs_v1",
            agent_type="cs",
            name="Customer Success Agent",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "default_sender": "success@enclaveguard.com",
            },
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()
        db.store_training_example = MagicMock(return_value={"id": "rlhf"})

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return CustomerSuccessAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    def test_registration(self):
        from core.agents.implementations.cs_agent import CustomerSuccessAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "cs" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.cs_agent import CustomerSuccessAgent
        assert CustomerSuccessAgent.agent_type == "cs"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import CustomerSuccessAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is CustomerSuccessAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "onboarding", "client_id": "client_1"},
            "run-123",
        )
        assert state["active_clients"] == []
        assert state["new_clients"] == []
        assert state["at_risk_clients"] == []
        assert state["checkin_drafts"] == []
        assert state["emails_sent"] == 0
        assert state["meetings_scheduled"] == 0

    def test_prepare_initial_state_checkin_type(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "quarterly"},
            "run-1",
        )
        assert state["checkin_type"] == "quarterly"

    def test_constants(self):
        from core.agents.implementations import cs_agent
        assert "onboarding" in cs_agent.CHECKIN_TYPES
        assert "30_day" in cs_agent.CHECKIN_TYPES
        assert "60_day" in cs_agent.CHECKIN_TYPES
        assert "quarterly" in cs_agent.CHECKIN_TYPES
        assert "health_check" in cs_agent.CHECKIN_TYPES

    def test_risk_thresholds(self):
        from core.agents.implementations import cs_agent
        thresholds = cs_agent.RISK_THRESHOLDS
        assert thresholds[30] == 0.3
        assert thresholds[45] == 0.5
        assert thresholds[60] == 0.7
        assert thresholds[90] == 0.9

    def test_onboarding_checklist(self):
        from core.agents.implementations import cs_agent
        checklist = cs_agent.ONBOARDING_CHECKLIST
        assert len(checklist) == 6
        assert any("welcome" in item.lower() for item in checklist)
        assert any("kickoff" in item.lower() for item in checklist)

    @pytest.mark.asyncio
    async def test_detect_signals_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "health_check", "clients": [{"company": "Acme", "domain": "acme.com"}]},
            "run-1",
        )
        result = await agent._node_detect_signals(state)
        assert result["current_node"] == "detect_signals"

    @pytest.mark.asyncio
    async def test_generate_outreach_node(self):
        agent = self._make_agent()
        # Mock LLM for email generation
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="Hi Acme team, checking in on your security assessment progress...")]
        ))
        state = agent._prepare_initial_state(
            {"mode": "30_day"},
            "run-1",
        )
        state["active_clients"] = [
            {"company_name": "Acme", "contact_email": "jane@acme.com", "contact_name": "Jane"},
        ]
        result = await agent._node_generate_outreach(state)
        assert result["current_node"] == "generate_outreach"

    @pytest.mark.asyncio
    async def test_generate_outreach_onboarding(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"mode": "onboarding"},
            "run-1",
        )
        state["new_clients"] = [
            {"company_name": "NewCo", "contact_email": "jane@newco.com", "contact_name": "Jane"},
        ]
        result = await agent._node_generate_outreach(state)
        assert result["current_node"] == "generate_outreach"

    @pytest.mark.asyncio
    async def test_schedule_followup_node(self):
        agent = self._make_agent()
        state = {
            "checkin_type": "onboarding",
            "active_clients": [{"company_name": "Acme"}],
            "new_clients": [{"company_name": "NewCo"}],
            "checkin_drafts": [{"to": "jane@acme.com"}],
        }
        result = await agent._node_schedule_followup(state)
        assert result["current_node"] == "schedule_followup"

    @pytest.mark.asyncio
    async def test_human_review_node(self):
        agent = self._make_agent()
        state = {
            "checkin_drafts": [{"to": "jane@acme.com", "body": "Hi"}],
            "at_risk_clients": [],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    def test_route_after_review_approved(self):
        from core.agents.implementations.cs_agent import CustomerSuccessAgent
        assert CustomerSuccessAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.cs_agent import CustomerSuccessAgent
        assert CustomerSuccessAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    @pytest.mark.asyncio
    async def test_report_node(self):
        agent = self._make_agent()
        state = {
            "active_clients": [{"company_name": "Acme"}],
            "new_clients": [],
            "at_risk_clients": [],
            "emails_sent": 1,
            "meetings_scheduled": 0,
            "checkin_type": "health_check",
            "checkin_drafts": [],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["report_summary"] != ""
        assert result["report_generated_at"] != ""

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "CustomerSuccessAgent" in repr_str
        assert "cs_v1" in repr_str


# ══════════════════════════════════════════════════════════════════════
# Dashboard Helper Tests
# ══════════════════════════════════════════════════════════════════════


class TestInvoiceHelpers:
    """Tests for invoice dashboard helpers."""

    def test_compute_invoice_stats_empty(self):
        from dashboard.pages._operations_helpers import compute_invoice_stats
        stats = compute_invoice_stats([])
        assert stats["total"] == 0
        assert stats["collection_rate"] == 0.0
        assert stats["avg_invoice"] == 0.0

    def test_compute_invoice_stats(self):
        from dashboard.pages._operations_helpers import compute_invoice_stats
        invoices = [
            {"status": "paid", "amount_cents": 1500000},
            {"status": "paid", "amount_cents": 500000},
            {"status": "open", "amount_cents": 300000},
            {"status": "overdue", "amount_cents": 200000},
            {"status": "void", "amount_cents": 100000},
        ]
        stats = compute_invoice_stats(invoices)
        assert stats["total"] == 5
        assert stats["paid"] == 2
        assert stats["open"] == 1
        assert stats["overdue"] == 1
        assert stats["void"] == 1
        assert stats["paid_amount"] == 2000000
        assert stats["overdue_amount"] == 200000

    def test_collection_rate(self):
        from dashboard.pages._operations_helpers import compute_invoice_stats
        invoices = [
            {"status": "paid", "amount_cents": 1000},
            {"status": "paid", "amount_cents": 1000},
            {"status": "void", "amount_cents": 1000},
        ]
        stats = compute_invoice_stats(invoices)
        # 2 paid / (2 paid + 1 void) = 66.67%
        assert abs(stats["collection_rate"] - 66.67) < 1

    def test_format_invoice_status(self):
        from dashboard.pages._operations_helpers import format_invoice_status
        text, color = format_invoice_status("paid")
        assert text == "Paid"
        assert color == "#10B981"

    def test_format_invoice_status_overdue(self):
        from dashboard.pages._operations_helpers import format_invoice_status
        text, color = format_invoice_status("overdue")
        assert text == "Overdue"
        assert color == "#EF4444"

    def test_format_invoice_status_unknown(self):
        from dashboard.pages._operations_helpers import format_invoice_status
        text, color = format_invoice_status("unknown_status")
        assert text == "Unknown_Status"

    def test_format_amount_cents(self):
        from dashboard.pages._operations_helpers import format_amount_cents
        assert format_amount_cents(150000) == "$1,500.00"
        assert format_amount_cents(0) == "$0.00"
        assert format_amount_cents(99) == "$0.99"

    def test_format_amount_cents_currency(self):
        from dashboard.pages._operations_helpers import format_amount_cents
        result = format_amount_cents(100000, "eur")
        assert "\u20ac" in result


class TestReminderHelpers:
    """Tests for reminder dashboard helpers."""

    def test_compute_reminder_stats_empty(self):
        from dashboard.pages._operations_helpers import compute_reminder_stats
        stats = compute_reminder_stats([])
        assert stats["total"] == 0
        assert stats["sent"] == 0

    def test_compute_reminder_stats(self):
        from dashboard.pages._operations_helpers import compute_reminder_stats
        reminders = [
            {"status": "sent", "tone": "polite"},
            {"status": "sent", "tone": "firm"},
            {"status": "draft", "tone": "polite"},
            {"status": "approved", "tone": "final"},
        ]
        stats = compute_reminder_stats(reminders)
        assert stats["total"] == 4
        assert stats["sent"] == 2
        assert stats["pending"] == 2  # draft + approved
        assert stats["by_tone"]["polite"] == 2
        assert stats["by_tone"]["firm"] == 1
        assert stats["by_tone"]["final"] == 1

    def test_format_reminder_tone(self):
        from dashboard.pages._operations_helpers import format_reminder_tone
        text, color = format_reminder_tone("polite")
        assert text == "Polite"
        text, color = format_reminder_tone("final")
        assert text == "Final Notice"
        assert color == "#EF4444"


class TestClientHelpers:
    """Tests for client dashboard helpers."""

    def test_compute_client_stats_empty(self):
        from dashboard.pages._operations_helpers import compute_client_stats
        stats = compute_client_stats([])
        assert stats["total"] == 0
        assert stats["avg_sentiment"] == 0.0
        assert stats["avg_churn_risk"] == 0.0

    def test_compute_client_stats(self):
        from dashboard.pages._operations_helpers import compute_client_stats
        clients = [
            {"status": "active", "sentiment_score": 0.9, "churn_risk": 0.1, "onboarding_complete": True},
            {"status": "active", "sentiment_score": 0.7, "churn_risk": 0.3, "onboarding_complete": True},
            {"status": "at_risk", "sentiment_score": 0.4, "churn_risk": 0.7, "onboarding_complete": True},
            {"status": "onboarding", "sentiment_score": 0.0, "churn_risk": 0.0, "onboarding_complete": False},
            {"status": "churned", "sentiment_score": 0.0, "churn_risk": 0.0, "onboarding_complete": True},
        ]
        stats = compute_client_stats(clients)
        assert stats["total"] == 5
        assert stats["active"] == 2
        assert stats["at_risk"] == 1
        assert stats["onboarding"] == 1
        assert stats["churned"] == 1
        # avg_sentiment for active + at_risk only
        assert abs(stats["avg_sentiment"] - (0.9 + 0.7 + 0.4) / 3) < 0.01

    def test_format_client_status(self):
        from dashboard.pages._operations_helpers import format_client_status
        text, color = format_client_status("active")
        assert text == "Active"
        assert color == "#10B981"
        text, color = format_client_status("at_risk")
        assert text == "At Risk"
        assert color == "#EF4444"

    def test_format_churn_risk(self):
        from dashboard.pages._operations_helpers import format_churn_risk
        text, color = format_churn_risk(0.2)
        assert text == "Low"
        text, color = format_churn_risk(0.4)
        assert text == "Moderate"
        text, color = format_churn_risk(0.55)
        assert text == "High"
        text, color = format_churn_risk(0.8)
        assert text == "Critical"
        assert color == "#EF4444"

    def test_onboarding_rate(self):
        from dashboard.pages._operations_helpers import compute_client_stats
        clients = [
            {"status": "active", "onboarding_complete": True},
            {"status": "active", "onboarding_complete": True},
            {"status": "onboarding", "onboarding_complete": False},
        ]
        stats = compute_client_stats(clients)
        # 2 out of 3 completed
        assert abs(stats["onboarding_rate"] - 66.67) < 1


class TestInteractionHelpers:
    """Tests for CS interaction dashboard helpers."""

    def test_compute_interaction_stats_empty(self):
        from dashboard.pages._operations_helpers import compute_interaction_stats
        stats = compute_interaction_stats([])
        assert stats["total"] == 0

    def test_compute_interaction_stats(self):
        from dashboard.pages._operations_helpers import compute_interaction_stats
        interactions = [
            {"status": "sent", "checkin_type": "onboarding"},
            {"status": "sent", "checkin_type": "30_day"},
            {"status": "draft", "checkin_type": "health_check"},
            {"status": "approved", "checkin_type": "quarterly"},
        ]
        stats = compute_interaction_stats(interactions)
        assert stats["total"] == 4
        assert stats["sent"] == 2
        assert stats["pending"] == 2
        assert stats["by_type"]["onboarding"] == 1
        assert stats["by_type"]["30_day"] == 1
        assert stats["by_type"]["health_check"] == 1
        assert stats["by_type"]["quarterly"] == 1

    def test_format_interaction_type(self):
        from dashboard.pages._operations_helpers import format_interaction_type
        text, color = format_interaction_type("onboarding")
        assert text == "Onboarding"
        text, color = format_interaction_type("health_check")
        assert text == "Health Check"
        assert color == "#EF4444"


class TestOperationsScore:
    """Tests for composite operations health score."""

    def test_perfect_score(self):
        from dashboard.pages._operations_helpers import compute_operations_score
        score = compute_operations_score(
            {"collection_rate": 100, "total": 10},
            {"avg_churn_risk": 0, "total": 5},
            {"total": 5, "sent": 5},
        )
        assert score["score"] == 100.0
        assert score["grade"] == "A"

    def test_zero_score(self):
        from dashboard.pages._operations_helpers import compute_operations_score
        score = compute_operations_score(
            {"collection_rate": 0, "total": 10},
            {"avg_churn_risk": 1.0, "total": 5},
            {"total": 10, "sent": 0},
        )
        assert score["score"] == 0.0
        assert score["grade"] == "F"

    def test_mixed_score(self):
        from dashboard.pages._operations_helpers import compute_operations_score
        score = compute_operations_score(
            {"collection_rate": 80, "total": 10},
            {"avg_churn_risk": 0.3, "total": 5},
            {"total": 10, "sent": 7},
        )
        assert 50 < score["score"] < 90

    def test_empty_data(self):
        from dashboard.pages._operations_helpers import compute_operations_score
        score = compute_operations_score(
            {"collection_rate": 0, "total": 0},
            {"avg_churn_risk": 0, "total": 0},
            {"total": 0, "sent": 0},
        )
        # Should give neutral scores for empty data
        assert score["score"] > 0

    def test_grades(self):
        from dashboard.pages._operations_helpers import compute_operations_score
        # A grade: high collection, low risk, high responsiveness
        score_a = compute_operations_score(
            {"collection_rate": 95}, {"avg_churn_risk": 0.05, "total": 5}, {"total": 10, "sent": 9},
        )
        assert score_a["grade"] in ("A", "B")  # 95*0.4 + 95*0.3 + 90*0.3 = 38+28.5+27=93.5

        # Low grade: poor everything
        score_low = compute_operations_score(
            {"collection_rate": 30}, {"avg_churn_risk": 0.8, "total": 5}, {"total": 10, "sent": 2},
        )
        assert score_low["grade"] in ("D", "F")


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestFinanceYAML:
    """Tests for finance.yaml config validation."""

    def _load_config(self):
        import yaml
        config_path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "finance.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_loads(self):
        config = self._load_config()
        assert config is not None

    def test_agent_id(self):
        config = self._load_config()
        assert config["agent_id"] == "finance_v1"

    def test_agent_type(self):
        config = self._load_config()
        assert config["agent_type"] == "finance"

    def test_model_config(self):
        config = self._load_config()
        assert config["model"]["provider"] == "anthropic"
        assert config["model"]["temperature"] == 0.3

    def test_human_gates(self):
        config = self._load_config()
        assert config["human_gates"]["enabled"] is True
        assert "human_review" in config["human_gates"]["gate_before"]

    def test_params(self):
        config = self._load_config()
        assert config["params"]["default_currency"] == "usd"
        assert config["params"]["default_due_days"] == 30
        assert config["params"]["max_invoice_amount"] == 10000000

    def test_pydantic_validation(self):
        from core.config.agent_schema import AgentInstanceConfig
        config = self._load_config()
        parsed = AgentInstanceConfig(**config)
        assert parsed.agent_type == "finance"
        assert parsed.enabled is True


class TestCSYAML:
    """Tests for cs.yaml config validation."""

    def _load_config(self):
        import yaml
        config_path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "cs.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_loads(self):
        config = self._load_config()
        assert config is not None

    def test_agent_id(self):
        config = self._load_config()
        assert config["agent_id"] == "cs_v1"

    def test_agent_type(self):
        config = self._load_config()
        assert config["agent_type"] == "cs"

    def test_model_config(self):
        config = self._load_config()
        assert config["model"]["provider"] == "anthropic"
        assert config["model"]["temperature"] == 0.6

    def test_human_gates(self):
        config = self._load_config()
        assert config["human_gates"]["enabled"] is True
        assert "human_review" in config["human_gates"]["gate_before"]

    def test_params(self):
        config = self._load_config()
        params = config["params"]
        assert params["company_name"] == "Enclave Guard"
        assert len(params["onboarding_checklist"]) == 6
        assert params["risk_thresholds"]["critical"] == 90

    def test_pydantic_validation(self):
        from core.config.agent_schema import AgentInstanceConfig
        config = self._load_config()
        parsed = AgentInstanceConfig(**config)
        assert parsed.agent_type == "cs"
        assert parsed.enabled is True


# ══════════════════════════════════════════════════════════════════════
# Migration Schema Tests
# ══════════════════════════════════════════════════════════════════════


class TestMigration009:
    """Tests for 009_operations.sql migration schema."""

    def _load_sql(self):
        sql_path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "009_operations.sql"
        return sql_path.read_text()

    def test_migration_exists(self):
        sql = self._load_sql()
        assert len(sql) > 0

    def test_invoices_table(self):
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS invoices" in sql
        assert "stripe_invoice_id" in sql
        assert "amount_cents" in sql
        assert "status" in sql
        assert "'draft'" in sql
        assert "'open'" in sql
        assert "'paid'" in sql
        assert "'overdue'" in sql
        assert "'void'" in sql

    def test_payment_reminders_table(self):
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS payment_reminders" in sql
        assert "tone" in sql
        assert "'polite'" in sql
        assert "'firm'" in sql
        assert "'final'" in sql
        assert "draft_text" in sql

    def test_client_records_table(self):
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS client_records" in sql
        assert "churn_risk" in sql
        assert "sentiment_score" in sql
        assert "onboarding_complete" in sql
        assert "'onboarding'" in sql
        assert "'active'" in sql
        assert "'at_risk'" in sql
        assert "'churned'" in sql

    def test_cs_interactions_table(self):
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS cs_interactions" in sql
        assert "interaction_type" in sql
        assert "checkin_type" in sql
        assert "'health_check'" in sql

    def test_indexes(self):
        sql = self._load_sql()
        assert "idx_invoices_vertical" in sql
        assert "idx_invoices_status" in sql
        assert "idx_clients_vertical" in sql
        assert "idx_clients_risk" in sql
        assert "idx_cs_interactions_client" in sql
        assert "idx_reminders_invoice" in sql

    def test_rpc_function(self):
        sql = self._load_sql()
        assert "get_operations_stats" in sql
        assert "RETURNS JSONB" in sql
        assert "p_vertical_id TEXT" in sql

    def test_foreign_keys(self):
        sql = self._load_sql()
        assert "REFERENCES invoices(id)" in sql
        assert "REFERENCES client_records(id)" in sql


# ══════════════════════════════════════════════════════════════════════
# Agent Registry Tests
# ══════════════════════════════════════════════════════════════════════


class TestAgentRegistration:
    """Tests that operations agents are properly registered."""

    def test_finance_registered(self):
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        from core.agents.implementations.finance_agent import FinanceAgent
        assert AGENT_IMPLEMENTATIONS.get("finance") is FinanceAgent

    def test_cs_registered(self):
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        from core.agents.implementations.cs_agent import CustomerSuccessAgent
        assert AGENT_IMPLEMENTATIONS.get("cs") is CustomerSuccessAgent

    def test_all_agents_count(self):
        """Ensure we have all expected agent types registered."""
        # Import all agent implementations to trigger registration
        from core.agents.implementations.finance_agent import FinanceAgent  # noqa: F401
        from core.agents.implementations.cs_agent import CustomerSuccessAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        # At minimum finance and cs should be registered
        assert "finance" in AGENT_IMPLEMENTATIONS
        assert "cs" in AGENT_IMPLEMENTATIONS
        assert len(AGENT_IMPLEMENTATIONS) >= 2
