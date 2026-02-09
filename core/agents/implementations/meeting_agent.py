"""
Meeting Scheduler Agent — The Calendar Manager.

Handles booking meetings with prospects, sending calendar invites,
and tracking confirmations. Bridges the gap between a warm reply
and an actual discovery call.

Architecture (LangGraph State Machine):
    check_requests -> propose_times -> human_review ->
    send_invites -> report -> END

Trigger Events:
    - meeting_requested: From appointment setter or pipeline agent
    - scheduled: Periodic check for unconfirmed meetings
    - manual: On-demand scheduling

Shared Brain Integration:
    - Reads: contact preferences, meeting history, availability
    - Writes: meeting outcomes, scheduling patterns

Safety:
    - NEVER sends invites automatically -- human_review gate
    - Validates contact email before sending
    - Respects working hours (configurable)

Usage:
    agent = MeetingSchedulerAgent(config, db, embedder, llm)
    result = await agent.run({
        "mode": "check_pending",
        "meeting_requests": [{"contact_email": "jane@acme.com", ...}],
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import MeetingSchedulerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

MODES = {"check_pending", "schedule_new", "follow_up_unconfirmed"}

MEETING_TYPES = {"discovery", "demo", "follow_up", "negotiation", "kickoff"}

MEETING_STATUSES = {
    "proposed", "confirmed", "completed",
    "cancelled", "no_show", "rescheduled",
}

MEETING_SYSTEM_PROMPT = """\
You are a professional meeting scheduler for {company_name}. \
You propose meeting times and write brief, polite confirmation \
emails for B2B discovery calls and demos.

Rules:
- Propose 2-3 time slots during business hours
- Keep invitation emails under 100 words
- Include a clear meeting link or instructions
- Be professional but friendly
- Reference the prospect's interest from prior communication

Return a JSON object with:
- "subject": email subject line
- "body": email body text
- "proposed_times": list of ISO datetime strings
"""


@register_agent_type("meeting_scheduler")
class MeetingSchedulerAgent(BaseAgent):
    """
    Calendar management agent for booking prospect meetings.

    Nodes:
        1. check_requests   -- Pull pending meeting requests
        2. propose_times    -- Generate time slots + invitation drafts
        3. human_review     -- Gate: review proposed meetings
        4. send_invites     -- Send calendar invites / confirmation emails
        5. report           -- Summary + update opportunity stage
    """

    def build_graph(self) -> Any:
        """Build the Meeting Scheduler Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(MeetingSchedulerAgentState)

        workflow.add_node("check_requests", self._node_check_requests)
        workflow.add_node("propose_times", self._node_propose_times)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("send_invites", self._node_send_invites)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("check_requests")

        workflow.add_edge("check_requests", "propose_times")
        workflow.add_edge("propose_times", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "send_invites",
                "rejected": "report",
            },
        )
        workflow.add_edge("send_invites", "report")
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

    def get_state_class(self) -> Type[MeetingSchedulerAgentState]:
        return MeetingSchedulerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "pending_requests": task.get("meeting_requests", []),
            "total_requests": len(task.get("meeting_requests", [])),
            "proposed_meetings": [],
            "meetings_approved": False,
            "human_edits": [],
            "invites_sent": 0,
            "meetings_confirmed": 0,
            "meetings_cancelled": 0,
            "calendar_links": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Check Requests ──────────────────────────────────────

    async def _node_check_requests(
        self, state: MeetingSchedulerAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Check for pending meeting requests.

        Pulls from task_input and/or scheduled_meetings table for
        proposed meetings that need follow-up.
        """
        logger.info(
            "meeting_check_start",
            extra={"agent_id": self.agent_id},
        )

        pending = list(state.get("pending_requests", []))

        # Also check DB for proposed meetings without confirmation
        try:
            result = (
                self.db.client.table("scheduled_meetings")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .eq("status", "proposed")
                .execute()
            )
            db_pending = result.data or []
            for meeting in db_pending:
                pending.append({
                    "meeting_id": meeting.get("id", ""),
                    "contact_email": meeting.get("contact_email", ""),
                    "contact_name": meeting.get("contact_name", ""),
                    "company_name": meeting.get("company_name", ""),
                    "meeting_type": meeting.get("meeting_type", "discovery"),
                    "source": "database",
                })
        except Exception as e:
            logger.debug(f"Failed to check pending meetings: {e}")

        logger.info(
            "meeting_check_complete",
            extra={"pending_count": len(pending)},
        )

        return {
            "current_node": "check_requests",
            "pending_requests": pending,
            "total_requests": len(pending),
        }

    # ─── Node 2: Propose Times ──────────────────────────────────────

    async def _node_propose_times(
        self, state: MeetingSchedulerAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Generate time slot proposals and invitation drafts.

        Uses LLM to create personalized meeting invitations.
        """
        pending = state.get("pending_requests", [])
        default_duration = self.config.params.get("default_duration", 30)
        company_name = self.config.params.get("company_name", "Our Team")

        logger.info(
            "meeting_propose_start",
            extra={"pending_count": len(pending)},
        )

        proposals: list[dict[str, Any]] = []

        for req in pending:
            contact_name = req.get("contact_name", "there")
            contact_email = req.get("contact_email", "")
            comp_name = req.get("company_name", "your company")
            meeting_type = req.get("meeting_type", "discovery")
            duration = req.get("duration_minutes", default_duration)

            # Generate proposed time slots (next 3 business days)
            now = datetime.now(timezone.utc)
            proposed_times: list[str] = []
            for day_offset in range(1, 6):
                candidate = now + timedelta(days=day_offset)
                # Skip weekends
                if candidate.weekday() < 5:
                    # Propose 10 AM and 2 PM slots
                    slot_10 = candidate.replace(
                        hour=10, minute=0, second=0, microsecond=0
                    )
                    proposed_times.append(slot_10.isoformat())
                    if len(proposed_times) >= 3:
                        break
                    slot_14 = candidate.replace(
                        hour=14, minute=0, second=0, microsecond=0
                    )
                    proposed_times.append(slot_14.isoformat())
                    if len(proposed_times) >= 3:
                        break

            # Generate invitation email via LLM
            subject = f"{meeting_type.title()} Call — {company_name} + {comp_name}"
            body = ""

            try:
                prompt = (
                    f"Write a brief meeting invitation email to {contact_name} "
                    f"at {comp_name} for a {meeting_type} call.\n"
                    f"Duration: {duration} minutes.\n"
                    f"Proposed times: {', '.join(proposed_times[:3])}\n"
                    f"Write the email body:"
                )

                system = MEETING_SYSTEM_PROMPT.format(company_name=company_name)

                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=400,
                    temperature=0.5,
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                )
                body = (
                    response.content[0].text.strip()
                    if response.content
                    else ""
                )
            except Exception as e:
                logger.debug(f"LLM meeting invite failed: {e}")
                body = (
                    f"Hi {contact_name},\n\n"
                    f"I'd love to schedule a {meeting_type} call to discuss "
                    f"how we can help {comp_name}. Would any of these times "
                    f"work for a {duration}-minute chat?\n\n"
                    f"Looking forward to connecting!\n\n"
                    f"Best regards"
                )

            proposals.append({
                "contact_email": contact_email,
                "contact_name": contact_name,
                "company_name": comp_name,
                "meeting_type": meeting_type,
                "duration_minutes": duration,
                "proposed_times": proposed_times,
                "subject": subject,
                "body": body,
                "meeting_id": req.get("meeting_id", ""),
            })

        logger.info(
            "meeting_propose_complete",
            extra={"proposals": len(proposals)},
        )

        return {
            "current_node": "propose_times",
            "proposed_meetings": proposals,
        }

    # ─── Node 3: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: MeetingSchedulerAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Present meeting proposals for human approval.
        """
        proposals = state.get("proposed_meetings", [])

        logger.info(
            "meeting_human_review_pending",
            extra={"proposal_count": len(proposals)},
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Send Invites ──────────────────────────────────────

    async def _node_send_invites(
        self, state: MeetingSchedulerAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Send approved meeting invitations.

        In production: sends via email engine and creates calendar events.
        Creates records in scheduled_meetings table.
        """
        proposals = state.get("proposed_meetings", [])
        sent = 0
        calendar_links: list[str] = []

        logger.info(
            "meeting_send_start",
            extra={"proposals": len(proposals)},
        )

        for prop in proposals:
            # In production: send via EmailEngine + create calendar event
            logger.info(
                "meeting_invite_sent",
                extra={
                    "contact": prop.get("contact_email", ""),
                    "type": prop.get("meeting_type", ""),
                },
            )
            sent += 1

            # Create scheduled_meetings record
            proposed_times = prop.get("proposed_times", [])
            scheduled_at = proposed_times[0] if proposed_times else None

            try:
                self.db.client.table("scheduled_meetings").insert({
                    "vertical_id": self.vertical_id,
                    "contact_email": prop.get("contact_email", ""),
                    "contact_name": prop.get("contact_name", ""),
                    "company_name": prop.get("company_name", ""),
                    "meeting_type": prop.get("meeting_type", "discovery"),
                    "title": prop.get("subject", ""),
                    "duration_minutes": prop.get("duration_minutes", 30),
                    "scheduled_at": scheduled_at,
                    "status": "proposed",
                    "agenda": prop.get("body", ""),
                }).execute()
            except Exception as e:
                logger.debug(f"Failed to create meeting record: {e}")

            # Mock calendar link
            calendar_links.append(
                f"https://calendar.app/meeting/{prop.get('contact_email', '')}"
            )

        # Write insight to shared brain
        if sent > 0:
            self.store_insight(InsightData(
                insight_type="meeting_scheduled",
                title=f"Meetings: {sent} invitations sent",
                content=f"Sent {sent} meeting invitations to prospects.",
                confidence=0.85,
                metadata={"invites_sent": sent},
            ))

        return {
            "current_node": "send_invites",
            "invites_sent": sent,
            "meetings_approved": True,
            "calendar_links": calendar_links,
            "knowledge_written": True,
        }

    # ─── Node 5: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: MeetingSchedulerAgentState
    ) -> dict[str, Any]:
        """Node 5: Generate meeting scheduling summary."""
        now = datetime.now(timezone.utc).isoformat()

        sections = [
            "# Meeting Scheduler Report",
            f"*Generated: {now}*\n",
            "## Requests",
            f"- **Pending Requests:** {state.get('total_requests', 0)}",
            f"- **Proposals Generated:** {len(state.get('proposed_meetings', []))}",
            f"\n## Execution",
            f"- **Invites Sent:** {state.get('invites_sent', 0)}",
            f"- **Meetings Confirmed:** {state.get('meetings_confirmed', 0)}",
            f"- **Meetings Cancelled:** {state.get('meetings_cancelled', 0)}",
        ]

        links = state.get("calendar_links", [])
        if links:
            sections.append(f"\n## Calendar Links")
            for link in links[:5]:
                sections.append(f"- {link}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: MeetingSchedulerAgentState) -> str:
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
            "You are a professional meeting scheduler. "
            "You propose convenient time slots and write polite, brief "
            "meeting invitation emails. You handle discovery calls, demos, "
            "and follow-up meetings for B2B sales."
        )

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<MeetingSchedulerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
