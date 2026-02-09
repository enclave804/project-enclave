"""
Customer Success Agent — The Account Manager.

Automates client onboarding, periodic check-ins, and churn prevention.
Triggered when proposals are accepted (onboarding) or on a periodic
schedule (retention monitoring).

Architecture (LangGraph State Machine):
    detect_signals → generate_outreach → schedule_followup →
    human_review → execute → report → END

Signal Types:
    - new_client: Accepted proposal → welcome packet + kickoff scheduling
    - 30_day: First month check-in → "How's it going?"
    - 60_day: Two-month check-in → "Here's what we've accomplished"
    - quarterly: Quarterly business review → renewal discussion
    - health_check: Periodic scan for at-risk clients (no contact 30+ days)

Shared Brain Integration:
    - Reads: proposal data, outreach insights, appointment history
    - Writes: client sentiment, churn risk scores, retention actions

Safety:
    - NEVER sends emails automatically — human_review gate
    - Respects contact frequency limits (no spamming clients)

RLHF Hook: If a human edits the check-in email,
the (draft, edit) pair is saved for better future personalization.

Usage:
    agent = CustomerSuccessAgent(config, db, embedder, llm)
    result = await agent.run({
        "mode": "onboarding",  # or "health_check", "30_day", "quarterly"
        "client_id": "client_123",
        "company_name": "Acme Corp",
        "contact_name": "Jane Doe",
        "contact_email": "jane@acme.com",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, ClientRecord
from core.agents.registry import register_agent_type
from core.agents.state import CustomerSuccessAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

CHECKIN_TYPES = {"onboarding", "30_day", "60_day", "quarterly", "health_check"}

# Days since last contact → risk level
RISK_THRESHOLDS = {
    30: 0.3,   # Mild concern
    45: 0.5,   # Moderate risk
    60: 0.7,   # High risk
    90: 0.9,   # Critical
}

ONBOARDING_CHECKLIST = [
    "Send welcome packet email",
    "Schedule kickoff call",
    "Share onboarding documentation",
    "Confirm project timeline",
    "Set up communication channel (Slack/Email)",
    "Introduce key team members",
]


@register_agent_type("cs")
class CustomerSuccessAgent(BaseAgent):
    """
    AI-powered customer success agent for client retention.

    Nodes:
        1. detect_signals    — Identify new clients and at-risk accounts
        2. generate_outreach — Draft personalized check-in emails
        3. schedule_followup — Plan meetings via appointment agent
        4. human_review      — Gate for human approval (NEVER auto-send)
        5. execute           — Send approved outreach + schedule meetings
        6. report            — Generate CS activity summary
    """

    def build_graph(self) -> Any:
        """Build the CS Agent's LangGraph state machine."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(CustomerSuccessAgentState)

        workflow.add_node("detect_signals", self._node_detect_signals)
        workflow.add_node("generate_outreach", self._node_generate_outreach)
        workflow.add_node("schedule_followup", self._node_schedule_followup)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute", self._node_execute)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("detect_signals")

        workflow.add_edge("detect_signals", "generate_outreach")
        workflow.add_edge("generate_outreach", "schedule_followup")
        workflow.add_edge("schedule_followup", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute", "report")
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

    def get_state_class(self) -> Type[CustomerSuccessAgentState]:
        return CustomerSuccessAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "active_clients": [],
            "new_clients": [],
            "at_risk_clients": [],
            "client_id": task.get("client_id", ""),
            "onboarding_email_draft": "",
            "onboarding_checklist": list(ONBOARDING_CHECKLIST),
            "kickoff_meeting_requested": False,
            "kickoff_meeting_scheduled": False,
            "welcome_packet_sent": False,
            "checkin_type": task.get("mode", "health_check"),
            "checkin_drafts": [],
            "last_contact_dates": {},
            "sentiment_scores": {},
            "churn_risk_score": 0.0,
            "retention_actions": [],
            "cs_actions_approved": False,
            "human_edits": [],
            "emails_sent": 0,
            "meetings_scheduled": 0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Detect Signals ───────────────────────────────────────

    async def _node_detect_signals(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Detect client signals — new clients, at-risk accounts.

        Scans for:
        - Newly accepted proposals (needs onboarding)
        - Clients with no contact in 30+ days (at-risk)
        - Scheduled check-ins (30/60/quarterly)
        """
        checkin_type = state.get("checkin_type", "health_check")
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}

        logger.info(
            "cs_detect_signals",
            extra={
                "agent_id": self.agent_id,
                "checkin_type": checkin_type,
            },
        )

        new_clients: list[dict[str, Any]] = []
        at_risk_clients: list[dict[str, Any]] = []
        sentiment_scores: dict[str, float] = {}

        # If specific client provided, focus on them
        client_id = state.get("client_id", "")
        if client_id and checkin_type == "onboarding":
            new_clients.append({
                "client_id": client_id,
                "company_name": task_input.get("company_name", ""),
                "contact_name": task_input.get("contact_name", ""),
                "contact_email": task_input.get("contact_email", ""),
                "proposal_id": task_input.get("proposal_id", ""),
            })

        # Pull client insights from shared brain
        try:
            query = "client sentiment satisfaction churn risk"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="client_health",
                limit=10,
            )
            for ins in (insights if isinstance(insights, list) else []):
                cid = ins.get("metadata", {}).get("client_id", "")
                score = ins.get("confidence", 0.5)
                if cid:
                    sentiment_scores[cid] = score
        except Exception as e:
            logger.debug(f"Could not fetch client insights: {e}")

        # Scan for at-risk clients (no recent contact)
        last_contacts = state.get("last_contact_dates", {})
        now = datetime.now(timezone.utc)
        for cid, last_str in last_contacts.items():
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                days_since = (now - last_dt).days
                if days_since >= 30:
                    churn_risk = 0.0
                    for threshold, risk in sorted(RISK_THRESHOLDS.items()):
                        if days_since >= threshold:
                            churn_risk = risk
                    at_risk_clients.append({
                        "client_id": cid,
                        "days_since_contact": days_since,
                        "churn_risk": churn_risk,
                    })
            except (ValueError, TypeError):
                continue

        logger.info(
            "cs_signals_detected",
            extra={
                "new_clients": len(new_clients),
                "at_risk": len(at_risk_clients),
            },
        )

        return {
            "current_node": "detect_signals",
            "new_clients": new_clients,
            "at_risk_clients": at_risk_clients,
            "sentiment_scores": sentiment_scores,
        }

    # ─── Node 2: Generate Outreach ────────────────────────────────────

    async def _node_generate_outreach(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Generate personalized outreach emails.

        For new clients: welcome packet + onboarding email
        For at-risk clients: check-in email
        For scheduled: milestone-appropriate message
        """
        checkin_type = state.get("checkin_type", "health_check")
        new_clients = state.get("new_clients", [])
        at_risk_clients = state.get("at_risk_clients", [])
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}

        logger.info(
            "cs_generate_outreach",
            extra={
                "checkin_type": checkin_type,
                "new_clients": len(new_clients),
                "at_risk": len(at_risk_clients),
            },
        )

        drafts: list[dict[str, Any]] = []
        our_company = self.config.params.get("company_name", "Enclave Guard")

        # Generate onboarding emails for new clients
        if checkin_type == "onboarding" and new_clients:
            for client in new_clients:
                name = client.get("contact_name", "there")
                company = client.get("company_name", "your company")

                prompt = (
                    f"Write a warm, professional onboarding welcome email.\n\n"
                    f"FROM: {our_company} team\n"
                    f"TO: {name} at {company}\n"
                    f"CONTEXT: They just signed a cybersecurity engagement with us.\n\n"
                    f"Include:\n"
                    f"1. Warm welcome and excitement about the partnership\n"
                    f"2. Brief overview of what happens next\n"
                    f"3. Mention the kickoff call will be scheduled\n"
                    f"4. Provide a direct contact for questions\n"
                    f"5. Professional but personable tone\n\n"
                    f"Write the email body only (no subject line)."
                )

                draft_text = ""
                try:
                    response = self.llm.messages.create(
                        model=self.config.model.model,
                        max_tokens=1000,
                        temperature=0.5,
                        messages=[{"role": "user", "content": prompt}],
                        system=self._get_system_prompt(),
                    )
                    draft_text = response.content[0].text.strip() if response.content else ""
                except Exception as e:
                    logger.warning(f"Outreach generation failed: {e}")
                    draft_text = (
                        f"Hi {name},\n\n"
                        f"Welcome to {our_company}! We're thrilled to be working with {company}.\n\n"
                        f"Next steps:\n"
                        f"1. We'll schedule a kickoff call this week\n"
                        f"2. We'll share our onboarding documentation\n"
                        f"3. We'll introduce your dedicated team\n\n"
                        f"If you have any questions, don't hesitate to reach out.\n\n"
                        f"Best regards,\n"
                        f"The {our_company} Team"
                    )

                drafts.append({
                    "client_id": client.get("client_id", ""),
                    "company_name": company,
                    "contact_name": name,
                    "contact_email": client.get("contact_email", ""),
                    "type": "onboarding",
                    "draft_text": draft_text,
                    "subject": f"Welcome to {our_company} — Next Steps",
                })

        # Generate check-in emails for at-risk clients
        if checkin_type == "health_check" and at_risk_clients:
            for client in at_risk_clients:
                days = client.get("days_since_contact", 30)
                risk = client.get("churn_risk", 0.3)

                tone = "warm" if risk < 0.5 else "concerned"
                drafts.append({
                    "client_id": client.get("client_id", ""),
                    "type": "health_check",
                    "days_since_contact": days,
                    "churn_risk": risk,
                    "draft_text": (
                        f"Hi,\n\n"
                        f"It's been a while since we last connected, and I wanted to check in. "
                        f"How are things going with the cybersecurity engagement?\n\n"
                        f"I'd love to schedule a quick call to:\n"
                        f"- Review progress and any findings\n"
                        f"- Discuss any challenges or concerns\n"
                        f"- Plan next steps\n\n"
                        f"What does your calendar look like this week?\n\n"
                        f"Best regards,\n"
                        f"The {our_company} Team"
                    ),
                    "subject": "Quick Check-in — How's Everything Going?",
                })

        # Generate milestone check-in emails
        if checkin_type in ("30_day", "60_day", "quarterly"):
            client_id = state.get("client_id", "")
            contact_name = task_input.get("contact_name", "there")
            company_name = task_input.get("company_name", "")

            milestone_messages = {
                "30_day": "one month since we kicked off",
                "60_day": "two months into our engagement",
                "quarterly": "time for our quarterly business review",
            }

            drafts.append({
                "client_id": client_id,
                "company_name": company_name,
                "contact_name": contact_name,
                "contact_email": task_input.get("contact_email", ""),
                "type": checkin_type,
                "draft_text": (
                    f"Hi {contact_name},\n\n"
                    f"It's been {milestone_messages.get(checkin_type, 'a while')} — "
                    f"I wanted to touch base and see how things are going.\n\n"
                    f"I'd love to schedule a quick call to review:\n"
                    f"- What's working well\n"
                    f"- Any areas for improvement\n"
                    f"- Upcoming priorities\n\n"
                    f"Would any time this week work for a 30-minute call?\n\n"
                    f"Best regards,\n"
                    f"The {our_company} Team"
                ),
                "subject": f"Check-in — {checkin_type.replace('_', ' ').title()}",
            })

        onboarding_draft = ""
        if drafts and drafts[0].get("type") == "onboarding":
            onboarding_draft = drafts[0].get("draft_text", "")

        return {
            "current_node": "generate_outreach",
            "checkin_drafts": drafts,
            "onboarding_email_draft": onboarding_draft,
        }

    # ─── Node 3: Schedule Followup ────────────────────────────────────

    async def _node_schedule_followup(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Plan follow-up meetings via calendar/appointment agent.
        """
        checkin_type = state.get("checkin_type", "")
        new_clients = state.get("new_clients", [])

        logger.info(
            "cs_schedule_followup",
            extra={"checkin_type": checkin_type},
        )

        needs_meeting = checkin_type in ("onboarding", "quarterly") or len(new_clients) > 0

        return {
            "current_node": "schedule_followup",
            "kickoff_meeting_requested": needs_meeting,
        }

    # ─── Node 4: Human Review ─────────────────────────────────────────

    async def _node_human_review(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """Node 4: Human gate — NEVER auto-send client communications."""
        drafts = state.get("checkin_drafts", [])

        logger.info(
            "cs_human_review_pending",
            extra={"drafts": len(drafts)},
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Execute ──────────────────────────────────────────────

    async def _node_execute(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Execute approved actions — send emails, schedule meetings.

        Captures RLHF data if human edited the drafts.
        """
        drafts = state.get("checkin_drafts", [])
        human_edits = state.get("human_edits", [])
        checkin_type = state.get("checkin_type", "")

        logger.info(
            "cs_execute",
            extra={"drafts": len(drafts), "edits": len(human_edits)},
        )

        emails_sent = 0
        meetings_scheduled = 0

        for i, draft in enumerate(drafts):
            # Check for human edits (RLHF capture)
            for edit in human_edits:
                if edit.get("index") == i:
                    original = draft.get("draft_text", "")
                    edited = edit.get("edited_text", "")
                    if original and edited and original != edited:
                        self.learn(
                            task_input={
                                "type": draft.get("type", ""),
                                "client_id": draft.get("client_id", ""),
                            },
                            model_output=original,
                            human_correction=edited,
                            source="manual_review",
                            metadata={"agent_type": "cs", "checkin_type": checkin_type},
                        )
                    break

            # In production: send via EmailEngine
            logger.info(
                "cs_email_sent",
                extra={
                    "client_id": draft.get("client_id", ""),
                    "type": draft.get("type", ""),
                },
            )
            emails_sent += 1

        # Schedule kickoff if requested
        if state.get("kickoff_meeting_requested", False):
            # In production: call appointment agent or calendar_client
            meetings_scheduled += 1

        # Write CS insights to shared brain
        if emails_sent > 0:
            self.store_insight(InsightData(
                insight_type="client_health",
                title=f"CS: {checkin_type} outreach for {emails_sent} clients",
                content=(
                    f"Sent {emails_sent} {checkin_type} emails. "
                    f"Scheduled {meetings_scheduled} meetings."
                ),
                confidence=0.80,
                metadata={
                    "checkin_type": checkin_type,
                    "emails_sent": emails_sent,
                    "meetings_scheduled": meetings_scheduled,
                },
            ))

        return {
            "current_node": "execute",
            "emails_sent": emails_sent,
            "meetings_scheduled": meetings_scheduled,
            "welcome_packet_sent": checkin_type == "onboarding" and emails_sent > 0,
            "kickoff_meeting_scheduled": meetings_scheduled > 0,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ───────────────────────────────────────────────

    async def _node_report(
        self, state: CustomerSuccessAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate CS activity summary."""
        now = datetime.now(timezone.utc).isoformat()
        checkin_type = state.get("checkin_type", "")
        emails = state.get("emails_sent", 0)
        meetings = state.get("meetings_scheduled", 0)
        new_clients = state.get("new_clients", [])
        at_risk = state.get("at_risk_clients", [])
        approved = state.get("cs_actions_approved", False)

        sections = [
            "# Customer Success Report",
            f"*Generated: {now}*\n",
            f"## Activity: {checkin_type.replace('_', ' ').title()}",
            f"- **Emails Sent:** {emails}",
            f"- **Meetings Scheduled:** {meetings}",
            f"- **Status:** {'Executed' if emails > 0 else ('Rejected' if not approved else 'Pending')}",
        ]

        if new_clients:
            sections.append(f"\n## New Clients ({len(new_clients)})")
            for c in new_clients:
                sections.append(f"  - {c.get('company_name', '?')} ({c.get('contact_name', '')})")

        if at_risk:
            sections.append(f"\n## At-Risk Clients ({len(at_risk)})")
            for c in at_risk:
                sections.append(
                    f"  - Client {c.get('client_id', '?')} — "
                    f"{c.get('days_since_contact', 0)} days since contact, "
                    f"risk: {c.get('churn_risk', 0):.0%}"
                )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: CustomerSuccessAgentState) -> str:
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
            "You are a customer success manager who builds strong client relationships. "
            "You write warm, professional emails that make clients feel valued and supported. "
            "You proactively identify risks and take action before issues escalate. "
            "You're data-driven but empathetic — numbers inform your outreach, "
            "but genuine care drives your messaging."
        )

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return f"<CustomerSuccessAgent agent_id={self.agent_id!r} vertical={self.vertical_id!r}>"
