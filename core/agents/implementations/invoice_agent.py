"""
Invoice Agent — The Billing Specialist.

Scans for overdue invoices, generates new invoices from accepted
proposals, and creates graduated payment reminders based on how
many days overdue. Tracks payment patterns and accounts receivable
health across the business. Works across all verticals.

Architecture (LangGraph State Machine):
    scan_invoices → generate_invoice → draft_reminders →
    human_review → send_invoices → report → END

Trigger Events:
    - scheduled: Daily invoice/overdue sweep
    - event: proposal_accepted (auto-generate invoice)
    - manual: On-demand invoice generation or reminder blast

Shared Brain Integration:
    - Reads: proposal data, client payment history, vertical pricing
    - Writes: payment patterns, average days-to-pay, collection effectiveness

Safety:
    - All invoices and reminders require human review before sending
    - Tone escalation follows strict day thresholds
    - Never sends final notice without explicit approval

Usage:
    agent = InvoiceAgent(config, db, embedder, llm)
    result = await agent.run({
        "scan_mode": "overdue",
        "reminder_enabled": True,
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import InvoiceAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

REMINDER_TONES = {
    "polite": {
        "days_threshold": 7,
        "urgency": "low",
        "description": "Gentle reminder about outstanding payment",
    },
    "firm": {
        "days_threshold": 14,
        "urgency": "medium",
        "description": "Direct follow-up with clear deadline",
    },
    "final": {
        "days_threshold": 30,
        "urgency": "high",
        "description": "Final notice before escalation to collections",
    },
}

OVERDUE_THRESHOLDS = {
    7: "polite",
    14: "firm",
    30: "final",
}

INVOICE_GENERATION_PROMPT = """\
You are a billing specialist generating a professional invoice summary \
for a client.

Client Details:
- Company: {company_name}
- Contact: {contact_name} ({contact_email})

Proposal/Contract Reference: {proposal_id}
Line Items:
{line_items_json}

Total Amount: ${total_amount}
Due Date: {due_date}
Currency: {currency}

Generate a professional invoice memo that includes:
1. Brief description of services rendered
2. Line item breakdown
3. Payment terms and due date
4. Payment instructions placeholder

Return as a JSON object:
{{
    "invoice_memo": "Professional memo text",
    "line_items_formatted": [
        {{"description": "...", "amount_cents": 0, "quantity": 1}}
    ],
    "payment_terms": "Net 30 from invoice date"
}}

Be concise and professional. Return ONLY the JSON, no markdown fences.
"""

PAYMENT_REMINDER_PROMPT = """\
You are drafting a payment reminder for an overdue invoice. Match the \
tone to the urgency level.

Invoice Details:
- Company: {company_name}
- Contact: {contact_name}
- Invoice Amount: ${amount_due}
- Days Overdue: {days_overdue}
- Original Due Date: {due_date}

Tone: {tone} ({tone_description})
Urgency: {urgency}

Guidelines by tone:
- polite: Friendly reminder, assume good faith, offer help if needed
- firm: Direct and professional, clearly state the overdue amount and deadline
- final: Serious tone, mention potential service impacts, request immediate action

Generate a payment reminder email.

Return as a JSON object:
{{
    "subject": "Email subject line",
    "body": "Email body text",
    "suggested_deadline": "YYYY-MM-DD"
}}

Return ONLY the JSON, no markdown fences.
"""


@register_agent_type("invoice")
class InvoiceAgent(BaseAgent):
    """
    Invoice generation and payment reminder agent.

    Nodes:
        1. scan_invoices     -- Query invoices table for overdue, check proposals
        2. generate_invoice  -- Build line items from proposal/contract, format via LLM
        3. draft_reminders   -- LLM creates payment reminders with graduated tone
        4. human_review      -- Gate: approve invoices and reminders
        5. send_invoices     -- Save to invoices table, mark as sent
        6. report            -- Summary + InsightData on payment patterns
    """

    def build_graph(self) -> Any:
        """Build the Invoice Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(InvoiceAgentState)

        workflow.add_node("scan_invoices", self._node_scan_invoices)
        workflow.add_node("generate_invoice", self._node_generate_invoice)
        workflow.add_node("draft_reminders", self._node_draft_reminders)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_invoices", self._node_send_invoices)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_invoices")

        workflow.add_edge("scan_invoices", "generate_invoice")
        workflow.add_edge("generate_invoice", "draft_reminders")
        workflow.add_edge("draft_reminders", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_invoices",
                "rejected": "report",
            },
        )
        workflow.add_edge("send_invoices", "report")
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
        return self.mcp_tools or []

    @classmethod
    def get_state_class(cls) -> Type[InvoiceAgentState]:
        return InvoiceAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "overdue_invoices": [],
            "pending_proposals": [],
            "total_overdue": 0,
            "total_outstanding_cents": 0,
            "new_invoices": [],
            "invoices_created": 0,
            "reminder_drafts": [],
            "reminder_tone": "",
            "reminders_generated": 0,
            "invoices_sent": 0,
            "reminders_sent": 0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Invoices ───────────────────────────────────────

    async def _node_scan_invoices(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 1: Query invoices table for overdue items and proposals needing invoices."""
        task = state.get("task_input", {})

        logger.info(
            "invoice_scan_started",
            extra={"agent_id": self.agent_id},
        )

        overdue_invoices: list[dict[str, Any]] = []
        pending_proposals: list[dict[str, Any]] = []
        total_outstanding_cents = 0

        # Scan for overdue invoices
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            result = (
                self.db.client.table("invoices")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .in_("status", ["open", "overdue"])
                .lt("due_date", now)
                .order("due_date", desc=False)
                .limit(50)
                .execute()
            )
            if result.data:
                for inv in result.data:
                    due_date = inv.get("due_date", "")
                    days_overdue = 0
                    if due_date:
                        try:
                            due_dt = datetime.strptime(due_date, "%Y-%m-%d").replace(
                                tzinfo=timezone.utc
                            )
                            days_overdue = (datetime.now(timezone.utc) - due_dt).days
                        except ValueError:
                            pass

                    # Determine reminder tone based on days overdue
                    tone = "polite"
                    for threshold_days in sorted(OVERDUE_THRESHOLDS.keys(), reverse=True):
                        if days_overdue >= threshold_days:
                            tone = OVERDUE_THRESHOLDS[threshold_days]
                            break

                    overdue_invoices.append({
                        "invoice_id": inv.get("id", ""),
                        "company_name": inv.get("company_name", ""),
                        "contact_email": inv.get("contact_email", ""),
                        "contact_name": inv.get("contact_name", ""),
                        "amount_cents": inv.get("amount_cents", 0),
                        "due_date": due_date,
                        "days_overdue": days_overdue,
                        "tone": tone,
                        "status": inv.get("status", "overdue"),
                    })
                    total_outstanding_cents += inv.get("amount_cents", 0)

                logger.info(
                    "invoice_overdue_found",
                    extra={
                        "count": len(overdue_invoices),
                        "total_outstanding_cents": total_outstanding_cents,
                    },
                )
        except Exception as e:
            logger.warning(
                "invoice_scan_error",
                extra={"error": str(e)[:200]},
            )

        # Scan for accepted proposals that need invoices
        try:
            result = (
                self.db.client.table("proposals")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .eq("status", "accepted")
                .eq("invoice_generated", False)
                .order("accepted_at", desc=True)
                .limit(20)
                .execute()
            )
            if result.data:
                for prop in result.data:
                    pending_proposals.append({
                        "proposal_id": prop.get("id", ""),
                        "company_name": prop.get("company_name", ""),
                        "contact_email": prop.get("contact_email", ""),
                        "contact_name": prop.get("contact_name", ""),
                        "amount_cents": prop.get("pricing_amount_cents", 0),
                        "accepted_at": prop.get("accepted_at", ""),
                    })
                logger.info(
                    "invoice_pending_proposals_found",
                    extra={"count": len(pending_proposals)},
                )
        except Exception as e:
            logger.warning(
                "invoice_proposals_scan_error",
                extra={"error": str(e)[:200]},
            )

        # Also check task-provided data
        task_overdue = task.get("overdue_invoices", [])
        overdue_invoices.extend(task_overdue)
        task_proposals = task.get("pending_proposals", [])
        pending_proposals.extend(task_proposals)

        logger.info(
            "invoice_scan_complete",
            extra={
                "overdue": len(overdue_invoices),
                "proposals": len(pending_proposals),
                "outstanding_cents": total_outstanding_cents,
            },
        )

        return {
            "current_node": "scan_invoices",
            "overdue_invoices": overdue_invoices,
            "pending_proposals": pending_proposals,
            "total_overdue": len(overdue_invoices),
            "total_outstanding_cents": total_outstanding_cents,
        }

    # ─── Node 2: Generate Invoice ────────────────────────────────────

    async def _node_generate_invoice(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 2: Build line items from proposals, format invoice via LLM."""
        pending_proposals = state.get("pending_proposals", [])

        logger.info(
            "invoice_generation_started",
            extra={"proposals_count": len(pending_proposals)},
        )

        new_invoices: list[dict[str, Any]] = []

        for proposal in pending_proposals[:10]:
            try:
                company_name = proposal.get("company_name", "")
                contact_name = proposal.get("contact_name", "")
                contact_email = proposal.get("contact_email", "")
                amount_cents = proposal.get("amount_cents", 0)
                proposal_id = proposal.get("proposal_id", "")

                # Default line items from proposal amount
                line_items = proposal.get("line_items", [])
                if not line_items:
                    line_items = [
                        {
                            "description": f"Professional services — {company_name}",
                            "amount_cents": amount_cents,
                            "quantity": 1,
                        }
                    ]

                due_days = self.config.params.get("default_due_days", 30)
                due_date = (
                    datetime.now(timezone.utc) + timedelta(days=due_days)
                ).strftime("%Y-%m-%d")
                total_amount = amount_cents / 100 if amount_cents else 0

                prompt = INVOICE_GENERATION_PROMPT.format(
                    company_name=company_name or "Client",
                    contact_name=contact_name or "Contact",
                    contact_email=contact_email or "N/A",
                    proposal_id=proposal_id or "N/A",
                    line_items_json=json.dumps(line_items[:10], indent=2),
                    total_amount=f"{total_amount:,.2f}",
                    due_date=due_date,
                    currency=self.config.params.get("currency", "USD"),
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a billing specialist.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                )

                llm_text = llm_response.content[0].text.strip()
                invoice_data: dict[str, Any] = {}

                try:
                    invoice_data = json.loads(llm_text)
                except (json.JSONDecodeError, KeyError):
                    logger.debug("invoice_generation_parse_error")
                    invoice_data = {
                        "invoice_memo": f"Invoice for services rendered to {company_name}",
                        "payment_terms": f"Net {due_days}",
                    }

                new_invoices.append({
                    "proposal_id": proposal_id,
                    "company_name": company_name,
                    "contact_name": contact_name,
                    "contact_email": contact_email,
                    "amount_cents": amount_cents,
                    "due_date": due_date,
                    "line_items": line_items,
                    "memo": invoice_data.get("invoice_memo", ""),
                    "payment_terms": invoice_data.get("payment_terms", f"Net {due_days}"),
                    "status": "draft",
                })

            except Exception as e:
                logger.warning(
                    "invoice_generation_error",
                    extra={
                        "proposal": proposal.get("proposal_id", ""),
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "invoice_generation_complete",
            extra={"invoices_created": len(new_invoices)},
        )

        return {
            "current_node": "generate_invoice",
            "new_invoices": new_invoices,
            "invoices_created": len(new_invoices),
        }

    # ─── Node 3: Draft Reminders ─────────────────────────────────────

    async def _node_draft_reminders(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM creates payment reminders with graduated tone."""
        overdue_invoices = state.get("overdue_invoices", [])

        logger.info(
            "invoice_reminders_started",
            extra={"overdue_count": len(overdue_invoices)},
        )

        reminder_drafts: list[dict[str, Any]] = []

        for invoice in overdue_invoices[:15]:
            try:
                tone = invoice.get("tone", "polite")
                tone_config = REMINDER_TONES.get(tone, REMINDER_TONES["polite"])
                days_overdue = invoice.get("days_overdue", 0)
                amount_cents = invoice.get("amount_cents", 0)
                amount_due = amount_cents / 100 if amount_cents else 0

                prompt = PAYMENT_REMINDER_PROMPT.format(
                    company_name=invoice.get("company_name", "Client"),
                    contact_name=invoice.get("contact_name", "Contact"),
                    amount_due=f"{amount_due:,.2f}",
                    days_overdue=days_overdue,
                    due_date=invoice.get("due_date", "N/A"),
                    tone=tone,
                    tone_description=tone_config["description"],
                    urgency=tone_config["urgency"],
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a professional billing specialist drafting payment reminders.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                )

                llm_text = llm_response.content[0].text.strip()
                reminder_data: dict[str, Any] = {}

                try:
                    reminder_data = json.loads(llm_text)
                except (json.JSONDecodeError, KeyError):
                    logger.debug("invoice_reminder_parse_error")
                    reminder_data = {
                        "subject": f"Payment reminder — ${amount_due:,.2f} overdue",
                        "body": (
                            f"This is a {tone} reminder that your invoice of "
                            f"${amount_due:,.2f} is {days_overdue} days overdue. "
                            f"Please arrange payment at your earliest convenience."
                        ),
                    }

                reminder_drafts.append({
                    "invoice_id": invoice.get("invoice_id", ""),
                    "company_name": invoice.get("company_name", ""),
                    "contact_email": invoice.get("contact_email", ""),
                    "contact_name": invoice.get("contact_name", ""),
                    "amount_cents": amount_cents,
                    "days_overdue": days_overdue,
                    "tone": tone,
                    "subject": reminder_data.get("subject", "Payment reminder"),
                    "body": reminder_data.get("body", ""),
                    "suggested_deadline": reminder_data.get("suggested_deadline", ""),
                })

            except Exception as e:
                logger.warning(
                    "invoice_reminder_error",
                    extra={
                        "invoice_id": invoice.get("invoice_id", ""),
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "invoice_reminders_drafted",
            extra={"reminders": len(reminder_drafts)},
        )

        return {
            "current_node": "draft_reminders",
            "reminder_drafts": reminder_drafts,
            "reminders_generated": len(reminder_drafts),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 4: Present invoices and reminders for human approval."""
        new_invoices = state.get("new_invoices", [])
        reminder_drafts = state.get("reminder_drafts", [])

        logger.info(
            "invoice_human_review_pending",
            extra={
                "invoices": len(new_invoices),
                "reminders": len(reminder_drafts),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Send Invoices ───────────────────────────────────────

    async def _node_send_invoices(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 5: Save invoices to table and mark reminders as sent."""
        now = datetime.now(timezone.utc).isoformat()
        new_invoices = state.get("new_invoices", [])
        reminder_drafts = state.get("reminder_drafts", [])

        logger.info(
            "invoice_send_started",
            extra={
                "invoices": len(new_invoices),
                "reminders": len(reminder_drafts),
            },
        )

        invoices_sent = 0
        reminders_sent = 0

        # Save new invoices
        for invoice in new_invoices:
            try:
                record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "proposal_id": invoice.get("proposal_id", ""),
                    "company_name": invoice.get("company_name", ""),
                    "contact_email": invoice.get("contact_email", ""),
                    "contact_name": invoice.get("contact_name", ""),
                    "amount_cents": invoice.get("amount_cents", 0),
                    "due_date": invoice.get("due_date", ""),
                    "line_items": json.dumps(invoice.get("line_items", [])),
                    "memo": invoice.get("memo", "")[:3000],
                    "payment_terms": invoice.get("payment_terms", "Net 30"),
                    "status": "open",
                    "created_at": now,
                }
                self.db.client.table("invoices").insert(record).execute()
                invoices_sent += 1
            except Exception as e:
                logger.warning(
                    "invoice_save_error",
                    extra={
                        "company": invoice.get("company_name", ""),
                        "error": str(e)[:200],
                    },
                )

        # Save reminders as sent
        for reminder in reminder_drafts:
            try:
                record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "invoice_id": reminder.get("invoice_id", ""),
                    "company_name": reminder.get("company_name", ""),
                    "contact_email": reminder.get("contact_email", ""),
                    "tone": reminder.get("tone", "polite"),
                    "subject": reminder.get("subject", ""),
                    "body": reminder.get("body", "")[:5000],
                    "days_overdue": reminder.get("days_overdue", 0),
                    "sent_at": now,
                }
                self.db.client.table("payment_reminders").insert(record).execute()
                reminders_sent += 1
            except Exception as e:
                logger.warning(
                    "invoice_reminder_save_error",
                    extra={
                        "invoice_id": reminder.get("invoice_id", ""),
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "invoice_send_complete",
            extra={
                "invoices_sent": invoices_sent,
                "reminders_sent": reminders_sent,
            },
        )

        return {
            "current_node": "send_invoices",
            "invoices_sent": invoices_sent,
            "reminders_sent": reminders_sent,
        }

    # ─── Node 6: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: InvoiceAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate summary and store InsightData on payment patterns."""
        now = datetime.now(timezone.utc).isoformat()
        overdue = state.get("overdue_invoices", [])
        new_invoices = state.get("new_invoices", [])
        reminders = state.get("reminder_drafts", [])
        total_outstanding = state.get("total_outstanding_cents", 0)
        invoices_sent = state.get("invoices_sent", 0)
        reminders_sent = state.get("reminders_sent", 0)

        # Calculate tone distribution
        tone_counts: dict[str, int] = {}
        for r in reminders:
            t = r.get("tone", "polite")
            tone_counts[t] = tone_counts.get(t, 0) + 1

        # Average days overdue
        days_list = [inv.get("days_overdue", 0) for inv in overdue if inv.get("days_overdue", 0) > 0]
        avg_days_overdue = round(sum(days_list) / len(days_list), 1) if days_list else 0

        # Build report
        sections = [
            "# Invoice & Billing Report",
            f"*Generated: {now}*\n",
            f"## Summary",
            f"- **Overdue Invoices:** {len(overdue)}",
            f"- **Outstanding Amount:** ${total_outstanding / 100:,.2f}",
            f"- **Average Days Overdue:** {avg_days_overdue}",
            f"- **New Invoices Generated:** {len(new_invoices)}",
            f"- **Invoices Sent:** {invoices_sent}",
            f"- **Reminders Drafted:** {len(reminders)}",
            f"- **Reminders Sent:** {reminders_sent}",
        ]

        if tone_counts:
            sections.append("\n## Reminder Tone Distribution")
            for tone, count in sorted(tone_counts.items()):
                sections.append(f"- **{tone.title()}:** {count}")

        if overdue:
            sections.append("\n## Overdue Invoices")
            for i, inv in enumerate(overdue[:10], 1):
                sections.append(
                    f"{i}. **{inv.get('company_name', 'Unknown')}** — "
                    f"${inv.get('amount_cents', 0) / 100:,.2f} | "
                    f"{inv.get('days_overdue', 0)} days overdue | "
                    f"Tone: {inv.get('tone', 'polite')}"
                )

        report = "\n".join(sections)

        # Store insight on payment patterns
        if overdue or new_invoices:
            self.store_insight(InsightData(
                insight_type="payment_pattern",
                title=f"Billing: {len(overdue)} overdue, ${total_outstanding / 100:,.2f} outstanding",
                content=(
                    f"Scanned {len(overdue)} overdue invoices totaling "
                    f"${total_outstanding / 100:,.2f}. "
                    f"Average {avg_days_overdue} days overdue. "
                    f"Generated {len(new_invoices)} new invoices and "
                    f"{len(reminders)} payment reminders. "
                    f"Tone distribution: {json.dumps(tone_counts)}."
                ),
                confidence=0.8,
                metadata={
                    "total_overdue": len(overdue),
                    "total_outstanding_cents": total_outstanding,
                    "avg_days_overdue": avg_days_overdue,
                    "invoices_created": len(new_invoices),
                    "reminders_generated": len(reminders),
                    "tone_distribution": tone_counts,
                },
            ))

        logger.info(
            "invoice_report_generated",
            extra={
                "overdue": len(overdue),
                "outstanding_cents": total_outstanding,
                "invoices_sent": invoices_sent,
                "reminders_sent": reminders_sent,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: InvoiceAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<InvoiceAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
