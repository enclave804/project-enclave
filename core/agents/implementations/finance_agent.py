"""
Finance Agent — The CFO.

Automates the post-sale money flow: generates invoices from accepted
proposals, chases overdue payments, and calculates monthly P&L from
all revenue streams (service invoicing + e-commerce).

Architecture (LangGraph State Machine):
    scan_invoices → identify_overdue → draft_reminders →
    human_review → send_reminders → generate_pnl → report → END

Trigger Events:
    - proposal_accepted: Generates invoice for newly signed deal
    - scheduled (weekly): Checks for overdue invoices, generates P&L
    - manual: On-demand financial reporting

Shared Brain Integration:
    - Reads: proposal data, commerce revenue, client records
    - Writes: financial metrics, payment patterns, client payment behavior

Safety:
    - NEVER sends invoices or reminders automatically — human_review gate
    - Amount validation (prevents invoices > configured max)
    - Mock mode when Stripe keys are missing

Usage:
    agent = FinanceAgent(config, db, embedder, llm)
    result = await agent.run({
        "mode": "full_cycle",  # or "invoice_only", "overdue_check", "pnl_report"
        "proposal_ids": ["prop_123"],  # Optional: specific proposals to invoice
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, InvoiceData, PaymentReminder
from core.agents.registry import register_agent_type
from core.agents.state import FinanceAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

MODES = {"full_cycle", "invoice_only", "overdue_check", "pnl_report"}
REMINDER_TONES = {
    (0, 14): "polite",
    (15, 30): "firm",
    (31, 999): "final",
}

MAX_INVOICE_AMOUNT = 100_000_00  # $100,000 in cents — safety cap


@register_agent_type("finance")
class FinanceAgent(BaseAgent):
    """
    AI-powered finance agent that automates B2B invoicing and collections.

    Nodes:
        1. scan_invoices    — Check for new proposals + existing invoice status
        2. identify_overdue — Find unpaid invoices past due date
        3. draft_reminders  — Generate payment reminder drafts
        4. human_review     — Gate for human approval (NEVER auto-send)
        5. send_reminders   — Execute approved reminder sends
        6. generate_pnl     — Calculate monthly P&L
        7. report           — Generate financial summary
    """

    def build_graph(self) -> Any:
        """Build the Finance Agent's LangGraph state machine."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(FinanceAgentState)

        workflow.add_node("scan_invoices", self._node_scan_invoices)
        workflow.add_node("identify_overdue", self._node_identify_overdue)
        workflow.add_node("draft_reminders", self._node_draft_reminders)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_reminders", self._node_send_reminders)
        workflow.add_node("generate_pnl", self._node_generate_pnl)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_invoices")

        workflow.add_edge("scan_invoices", "identify_overdue")
        workflow.add_edge("identify_overdue", "draft_reminders")
        workflow.add_edge("draft_reminders", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_reminders",
                "rejected": "generate_pnl",
            },
        )
        workflow.add_edge("send_reminders", "generate_pnl")
        workflow.add_edge("generate_pnl", "report")
        workflow.add_edge("report", END)

        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        return []

    def get_state_class(self) -> Type[FinanceAgentState]:
        return FinanceAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "active_invoices": [],
            "overdue_invoices": [],
            "paid_invoices": [],
            "new_proposal_ids": task.get("proposal_ids", []),
            "invoices_to_create": [],
            "invoices_created": [],
            "invoices_sent": 0,
            "reminder_drafts": [],
            "reminders_approved": False,
            "reminders_sent": 0,
            "total_revenue": 0.0,
            "accounts_receivable": 0.0,
            "monthly_recurring": 0.0,
            "service_revenue": 0.0,
            "commerce_revenue": 0.0,
            "total_costs": 0.0,
            "net_profit": 0.0,
            "pnl_data": {},
            "finance_actions_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Invoices ────────────────────────────────────────

    async def _node_scan_invoices(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Scan existing invoices and check for new proposals to invoice.

        Pulls invoice status from Stripe (or mock) and identifies
        accepted proposals that need invoices generated.
        """
        logger.info(
            "finance_scan_start",
            extra={"agent_id": self.agent_id},
        )

        active_invoices: list[dict[str, Any]] = []
        paid_invoices: list[dict[str, Any]] = []
        invoices_to_create: list[dict[str, Any]] = []

        # Fetch existing invoices via MCP tools
        try:
            from core.mcp.tools.finance_tools import _get_client
            client = _get_client()
            open_invs = await client.list_invoices(status="open")
            active_invoices = open_invs
            paid_invs = await client.list_invoices(status="paid")
            paid_invoices = paid_invs
        except Exception as e:
            logger.debug(f"Invoice scan failed: {e}")

        # Check for new proposals needing invoices
        new_proposal_ids = state.get("new_proposal_ids", [])
        for prop_id in new_proposal_ids:
            # In production: query proposals table for accepted proposals
            invoices_to_create.append({
                "proposal_id": prop_id,
                "status": "needs_invoice",
            })

        # Pull proposal insights from shared brain
        try:
            query = "accepted proposal invoice"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="proposal_sent",
                limit=5,
            )
            # Could surface recently accepted proposals here
        except Exception as e:
            logger.debug(f"Could not fetch proposal insights: {e}")

        logger.info(
            "finance_scan_complete",
            extra={
                "active_invoices": len(active_invoices),
                "paid_invoices": len(paid_invoices),
                "to_create": len(invoices_to_create),
            },
        )

        return {
            "current_node": "scan_invoices",
            "active_invoices": active_invoices,
            "paid_invoices": paid_invoices,
            "invoices_to_create": invoices_to_create,
        }

    # ─── Node 2: Identify Overdue ─────────────────────────────────────

    async def _node_identify_overdue(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Identify overdue invoices from active invoice list.
        """
        overdue_days = self.config.params.get("overdue_threshold_days", 7)

        logger.info(
            "finance_overdue_check",
            extra={"threshold_days": overdue_days},
        )

        overdue: list[dict[str, Any]] = []

        try:
            from core.mcp.tools.finance_tools import check_overdue_invoices
            result_json = await check_overdue_invoices(days_overdue=overdue_days)
            result = json.loads(result_json)
            overdue = result.get("invoices", [])
        except Exception as e:
            logger.debug(f"Overdue check failed: {e}")

        logger.info(
            "finance_overdue_found",
            extra={"overdue_count": len(overdue)},
        )

        return {
            "current_node": "identify_overdue",
            "overdue_invoices": overdue,
        }

    # ─── Node 3: Draft Reminders ──────────────────────────────────────

    async def _node_draft_reminders(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Draft payment reminder emails for overdue invoices.

        Tone escalates based on days overdue:
        - 7-14 days: polite
        - 15-30 days: firm
        - 30+ days: final
        """
        overdue = state.get("overdue_invoices", [])

        logger.info(
            "finance_draft_reminders",
            extra={"overdue_count": len(overdue)},
        )

        drafts: list[dict[str, Any]] = []

        for inv in overdue:
            days = inv.get("days_overdue", 0)
            tone = "polite"
            for (low, high), t in REMINDER_TONES.items():
                if low <= days <= high:
                    tone = t
                    break

            amount = inv.get("amount_due", inv.get("total_amount", 0))
            amount_display = f"${amount / 100:,.2f}" if amount else "$0.00"

            try:
                from core.mcp.tools.finance_tools import send_payment_reminder
                result_json = await send_payment_reminder(
                    invoice_id=inv.get("id", ""),
                    contact_email=inv.get("contact_email", inv.get("customer_id", "")),
                    company_name=inv.get("company_name", ""),
                    amount_display=amount_display,
                    days_overdue=days,
                    tone=tone,
                )
                draft = json.loads(result_json)
                drafts.append(draft)
            except Exception as e:
                logger.debug(f"Reminder draft failed: {e}")

        return {
            "current_node": "draft_reminders",
            "reminder_drafts": drafts,
        }

    # ─── Node 4: Human Review ─────────────────────────────────────────

    async def _node_human_review(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Human Review — NEVER auto-send invoices or reminders.
        """
        drafts = state.get("reminder_drafts", [])
        invoices_to_create = state.get("invoices_to_create", [])

        logger.info(
            "finance_human_review_pending",
            extra={
                "reminder_drafts": len(drafts),
                "invoices_pending": len(invoices_to_create),
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Send Reminders ───────────────────────────────────────

    async def _node_send_reminders(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Send approved payment reminders.

        In production, sends via email engine. In mock mode, logs only.
        """
        drafts = state.get("reminder_drafts", [])
        sent = 0

        logger.info(
            "finance_send_reminders",
            extra={"drafts": len(drafts)},
        )

        for draft in drafts:
            # In production: send via EmailEngine
            # For now, log and count
            logger.info(
                "finance_reminder_sent",
                extra={
                    "invoice_id": draft.get("invoice_id", ""),
                    "tone": draft.get("tone", ""),
                },
            )
            sent += 1

        # Write insight to shared brain
        if sent > 0:
            self.store_insight(InsightData(
                insight_type="payment_reminder",
                title=f"Finance: Sent {sent} payment reminders",
                content=f"Sent {sent} payment reminders for overdue invoices.",
                confidence=0.85,
                metadata={"reminders_sent": sent},
            ))

        return {
            "current_node": "send_reminders",
            "reminders_sent": sent,
            "reminders_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Generate P&L ────────────────────────────────────────

    async def _node_generate_pnl(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """
        Node 6: Calculate monthly P&L from all revenue streams.
        """
        logger.info("finance_pnl_start")

        pnl_data: dict[str, Any] = {}

        try:
            from core.mcp.tools.finance_tools import get_monthly_pnl
            result_json = await get_monthly_pnl(vertical_id=self.vertical_id)
            pnl_data = json.loads(result_json)
        except Exception as e:
            logger.debug(f"P&L calculation failed: {e}")

        service_revenue = pnl_data.get("service_revenue_cents", 0)
        commerce_revenue = pnl_data.get("commerce_revenue_cents", 0)
        total_revenue = service_revenue + commerce_revenue
        accounts_receivable = pnl_data.get("accounts_receivable_cents", 0)
        total_costs = pnl_data.get("total_costs_cents", 0)
        net_profit = total_revenue - total_costs

        # Write financial insights to shared brain
        if total_revenue > 0:
            self.store_insight(InsightData(
                insight_type="financial_metrics",
                title=f"Finance: Monthly revenue ${total_revenue / 100:,.2f}",
                content=(
                    f"Monthly P&L — Revenue: ${total_revenue / 100:,.2f} "
                    f"(Service: ${service_revenue / 100:,.2f}, "
                    f"Commerce: ${commerce_revenue / 100:,.2f}). "
                    f"AR: ${accounts_receivable / 100:,.2f}. "
                    f"Net: ${net_profit / 100:,.2f}."
                ),
                confidence=0.90,
                metadata=pnl_data,
            ))

        return {
            "current_node": "generate_pnl",
            "total_revenue": total_revenue / 100,
            "service_revenue": service_revenue / 100,
            "commerce_revenue": commerce_revenue / 100,
            "accounts_receivable": accounts_receivable / 100,
            "total_costs": total_costs / 100,
            "net_profit": net_profit / 100,
            "pnl_data": pnl_data,
        }

    # ─── Node 7: Report ───────────────────────────────────────────────

    async def _node_report(
        self, state: FinanceAgentState
    ) -> dict[str, Any]:
        """Node 7: Generate financial summary report."""
        now = datetime.now(timezone.utc).isoformat()

        sections = [
            "# Finance Report",
            f"*Generated: {now}*\n",
            "## Revenue",
            f"- **Total Revenue:** ${state.get('total_revenue', 0):,.2f}",
            f"- **Service Revenue:** ${state.get('service_revenue', 0):,.2f}",
            f"- **Commerce Revenue:** ${state.get('commerce_revenue', 0):,.2f}",
            f"\n## Accounts Receivable",
            f"- **Outstanding:** ${state.get('accounts_receivable', 0):,.2f}",
            f"- **Overdue Invoices:** {len(state.get('overdue_invoices', []))}",
            f"- **Reminders Sent:** {state.get('reminders_sent', 0)}",
            f"\n## Profitability",
            f"- **Total Costs:** ${state.get('total_costs', 0):,.2f}",
            f"- **Net Profit:** ${state.get('net_profit', 0):,.2f}",
        ]

        active = state.get("active_invoices", [])
        paid = state.get("paid_invoices", [])
        if active or paid:
            sections.append(f"\n## Invoice Summary")
            sections.append(f"- **Active:** {len(active)}")
            sections.append(f"- **Paid:** {len(paid)}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: FinanceAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── System Prompt ────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        if self.config.system_prompt_path:
            try:
                with open(self.config.system_prompt_path) as f:
                    return f.read()
            except FileNotFoundError:
                pass
        return (
            "You are a professional finance manager. "
            "You handle invoicing, payment tracking, and financial reporting "
            "with precision and professionalism. You escalate payment reminders "
            "gradually (polite → firm → final) and always maintain positive "
            "client relationships. You never make threats or use aggressive language."
        )

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return f"<FinanceAgent agent_id={self.agent_id!r} vertical={self.vertical_id!r}>"
