"""
Onboarding Agent — The Welcome Concierge.

Manages the first 90 days of a new client relationship: generates
personalized welcome packages, schedules kickoff calls, creates
milestone-based onboarding plans, and tracks progress through
completion. Works across all verticals.

Architecture (LangGraph State Machine):
    setup_onboarding → generate_welcome_package →
    human_review → execute_onboarding → report → END

Trigger Events:
    - event: proposal_accepted (new client signed)
    - scheduled: Weekly onboarding progress check
    - manual: On-demand onboarding setup

Shared Brain Integration:
    - Reads: deal context, proposal history, vertical-specific templates
    - Writes: onboarding patterns, time-to-first-value metrics

Safety:
    - Welcome packages require human review before sending
    - Client data is scoped to the vertical
    - No external communications without approval

Usage:
    agent = OnboardingAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "contact_email": "jane@acme.com",
        "contact_name": "Jane Doe",
        "template": "enterprise",
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
from core.agents.state import OnboardingAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

ONBOARDING_TEMPLATES = {
    "default": [
        "welcome_call",
        "account_setup",
        "service_overview",
        "first_deliverable",
        "check_in",
    ],
    "enterprise": [
        "welcome_call",
        "security_review",
        "account_setup",
        "team_training",
        "integration",
        "first_deliverable",
        "executive_review",
        "quarterly_plan",
    ],
    "quick_start": [
        "welcome_call",
        "account_setup",
        "first_deliverable",
    ],
}

MILESTONE_DEFAULTS = {
    "welcome_call": {
        "description": "Schedule and conduct introductory kickoff call",
        "days_offset": 2,
    },
    "security_review": {
        "description": "Complete security questionnaire and initial review",
        "days_offset": 5,
    },
    "account_setup": {
        "description": "Provision accounts, credentials, and access",
        "days_offset": 7,
    },
    "team_training": {
        "description": "Conduct team onboarding and training session",
        "days_offset": 14,
    },
    "integration": {
        "description": "Set up integrations and data connections",
        "days_offset": 21,
    },
    "service_overview": {
        "description": "Walk through service capabilities and scope",
        "days_offset": 10,
    },
    "first_deliverable": {
        "description": "Deliver first tangible output or report",
        "days_offset": 30,
    },
    "executive_review": {
        "description": "Executive-level progress review and alignment",
        "days_offset": 45,
    },
    "quarterly_plan": {
        "description": "Build and present the quarterly strategic plan",
        "days_offset": 60,
    },
    "check_in": {
        "description": "30-day check-in on satisfaction and progress",
        "days_offset": 30,
    },
}

WELCOME_PACKAGE_PROMPT = """\
You are a client onboarding specialist creating a personalized welcome \
package for a new client.

Client Details:
- Company: {company_name}
- Contact: {contact_name} ({contact_title})
- Email: {contact_email}
- Deal Value: ${deal_value}

Onboarding Template: {template_key}
Milestones:
{milestones_json}

Vertical Context: {vertical_id}

Generate a warm, professional welcome package that includes:
1. A personalized welcome message (2-3 paragraphs)
2. Overview of what to expect in the first 30/60/90 days
3. Key milestones and their timeline
4. A suggested kickoff agenda (5-7 bullet points)
5. Next steps and action items

Return as a JSON object:
{{
    "welcome_message": "The personalized welcome text",
    "timeline_overview": "30/60/90 day overview",
    "kickoff_agenda": ["item1", "item2", ...],
    "next_steps": ["step1", "step2", ...]
}}

Be specific to the company and avoid generic platitudes. Reference \
their industry where possible. Return ONLY the JSON, no markdown fences.
"""


@register_agent_type("onboarding")
class OnboardingAgent(BaseAgent):
    """
    Client onboarding management agent.

    Nodes:
        1. setup_onboarding     -- Load client data, select template, build milestones
        2. generate_welcome_package -- LLM creates personalized welcome content
        3. human_review         -- Gate: approve welcome package
        4. execute_onboarding   -- Save to client_onboarding table, mark in_progress
        5. report               -- Build summary + InsightData for onboarding patterns
    """

    def build_graph(self) -> Any:
        """Build the Onboarding Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(OnboardingAgentState)

        workflow.add_node("setup_onboarding", self._node_setup_onboarding)
        workflow.add_node("generate_welcome_package", self._node_generate_welcome_package)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_onboarding", self._node_execute_onboarding)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("setup_onboarding")

        workflow.add_edge("setup_onboarding", "generate_welcome_package")
        workflow.add_edge("generate_welcome_package", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute_onboarding",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute_onboarding", "report")
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
    def get_state_class(cls) -> Type[OnboardingAgentState]:
        return OnboardingAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_name": "",
            "company_domain": "",
            "contact_name": "",
            "contact_email": "",
            "contact_title": "",
            "proposal_id": "",
            "deal_value_cents": 0,
            "template_key": "default",
            "milestones": [],
            "total_milestones": 0,
            "milestones_completed": 0,
            "welcome_content": "",
            "kickoff_date": "",
            "kickoff_agenda": [],
            "welcome_package_generated": False,
            "onboarding_id": "",
            "onboarding_status": "not_started",
            "onboarding_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Setup Onboarding ────────────────────────────────────

    async def _node_setup_onboarding(
        self, state: OnboardingAgentState
    ) -> dict[str, Any]:
        """Node 1: Load client data from task_input, select template, generate milestones."""
        task = state.get("task_input", {})

        logger.info(
            "onboarding_setup_started",
            extra={"agent_id": self.agent_id},
        )

        # Extract client data from task
        company_name = task.get("company_name", "")
        company_domain = task.get("company_domain", "")
        contact_name = task.get("contact_name", "")
        contact_email = task.get("contact_email", "")
        contact_title = task.get("contact_title", "")
        proposal_id = task.get("proposal_id", "")
        deal_value_cents = task.get("deal_value_cents", 0)

        # Select onboarding template from task or config
        template_key = task.get("template", "default")
        if template_key not in ONBOARDING_TEMPLATES:
            config_template = self.config.params.get("default_template", "default")
            logger.info(
                "onboarding_template_fallback",
                extra={
                    "requested": template_key,
                    "fallback": config_template,
                },
            )
            template_key = config_template

        # Auto-select enterprise template for high-value deals
        enterprise_threshold = self.config.params.get(
            "enterprise_threshold_cents", 500000
        )
        if deal_value_cents >= enterprise_threshold and template_key == "default":
            template_key = "enterprise"
            logger.info(
                "onboarding_auto_enterprise",
                extra={
                    "deal_value_cents": deal_value_cents,
                    "threshold": enterprise_threshold,
                },
            )

        # Build milestone list with due dates
        template_steps = ONBOARDING_TEMPLATES.get(template_key, ONBOARDING_TEMPLATES["default"])
        now = datetime.now(timezone.utc)
        milestones: list[dict[str, Any]] = []

        for step_name in template_steps:
            defaults = MILESTONE_DEFAULTS.get(step_name, {})
            days_offset = defaults.get("days_offset", 7)
            due_date = (now + timedelta(days=days_offset)).strftime("%Y-%m-%d")

            milestones.append({
                "name": step_name,
                "description": defaults.get("description", step_name.replace("_", " ").title()),
                "due_date": due_date,
                "status": "pending",
                "days_offset": days_offset,
            })

        # Try to load existing client data from database
        try:
            result = (
                self.db.client.table("companies")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .eq("domain", company_domain)
                .limit(1)
                .execute()
            )
            if result.data:
                company_record = result.data[0]
                if not company_name:
                    company_name = company_record.get("name", "")
                logger.info(
                    "onboarding_company_loaded",
                    extra={"company": company_name},
                )
        except Exception as e:
            logger.warning(
                "onboarding_company_lookup_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "onboarding_setup_complete",
            extra={
                "company": company_name,
                "template": template_key,
                "milestones": len(milestones),
            },
        )

        return {
            "current_node": "setup_onboarding",
            "company_name": company_name,
            "company_domain": company_domain,
            "contact_name": contact_name,
            "contact_email": contact_email,
            "contact_title": contact_title,
            "proposal_id": proposal_id,
            "deal_value_cents": deal_value_cents,
            "template_key": template_key,
            "milestones": milestones,
            "total_milestones": len(milestones),
        }

    # ─── Node 2: Generate Welcome Package ────────────────────────────

    async def _node_generate_welcome_package(
        self, state: OnboardingAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM creates personalized welcome content and kickoff agenda."""
        company_name = state.get("company_name", "")
        contact_name = state.get("contact_name", "")
        contact_title = state.get("contact_title", "")
        contact_email = state.get("contact_email", "")
        deal_value_cents = state.get("deal_value_cents", 0)
        template_key = state.get("template_key", "default")
        milestones = state.get("milestones", [])

        logger.info(
            "onboarding_welcome_generation_started",
            extra={
                "company": company_name,
                "template": template_key,
            },
        )

        welcome_content = ""
        kickoff_agenda: list[str] = []
        kickoff_date = ""

        try:
            deal_value_display = f"{deal_value_cents / 100:,.2f}" if deal_value_cents else "N/A"

            prompt = WELCOME_PACKAGE_PROMPT.format(
                company_name=company_name or "New Client",
                contact_name=contact_name or "Client Contact",
                contact_title=contact_title or "Decision Maker",
                contact_email=contact_email or "N/A",
                deal_value=deal_value_display,
                template_key=template_key,
                milestones_json=json.dumps(milestones[:10], indent=2),
                vertical_id=self.vertical_id,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a client onboarding specialist.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                package_data = json.loads(llm_text)
                if isinstance(package_data, dict):
                    welcome_content = package_data.get("welcome_message", "")
                    timeline_overview = package_data.get("timeline_overview", "")
                    if timeline_overview:
                        welcome_content += f"\n\n## Timeline\n{timeline_overview}"
                    kickoff_agenda = package_data.get("kickoff_agenda", [])
                    if not isinstance(kickoff_agenda, list):
                        kickoff_agenda = []
            except (json.JSONDecodeError, KeyError):
                logger.warning("onboarding_welcome_parse_error")
                welcome_content = llm_text[:2000]

        except Exception as e:
            logger.error(
                "onboarding_welcome_llm_error",
                extra={"error": str(e)[:200]},
            )
            welcome_content = (
                f"Welcome to our team, {contact_name}! "
                f"We are excited to begin working with {company_name}. "
                f"Your dedicated team will be in touch shortly to schedule "
                f"a kickoff call and get your onboarding underway."
            )
            kickoff_agenda = [
                "Introductions and team alignment",
                "Review project scope and deliverables",
                "Discuss timeline and milestones",
                "Identify key stakeholders",
                "Next steps and action items",
            ]

        # Schedule kickoff 2 business days from now
        kickoff_offset = self.config.params.get("kickoff_days_offset", 2)
        kickoff_dt = datetime.now(timezone.utc) + timedelta(days=kickoff_offset)
        kickoff_date = kickoff_dt.strftime("%Y-%m-%d")

        welcome_generated = bool(welcome_content)

        logger.info(
            "onboarding_welcome_generated",
            extra={
                "content_length": len(welcome_content),
                "agenda_items": len(kickoff_agenda),
                "kickoff_date": kickoff_date,
            },
        )

        return {
            "current_node": "generate_welcome_package",
            "welcome_content": welcome_content,
            "kickoff_date": kickoff_date,
            "kickoff_agenda": kickoff_agenda,
            "welcome_package_generated": welcome_generated,
        }

    # ─── Node 3: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: OnboardingAgentState
    ) -> dict[str, Any]:
        """Node 3: Present welcome package for human approval."""
        company_name = state.get("company_name", "Unknown")
        milestones = state.get("milestones", [])

        logger.info(
            "onboarding_human_review_pending",
            extra={
                "company": company_name,
                "milestones": len(milestones),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 4: Execute Onboarding ──────────────────────────────────

    async def _node_execute_onboarding(
        self, state: OnboardingAgentState
    ) -> dict[str, Any]:
        """Node 4: Save onboarding record to client_onboarding table."""
        now = datetime.now(timezone.utc).isoformat()
        company_name = state.get("company_name", "")
        contact_email = state.get("contact_email", "")
        milestones = state.get("milestones", [])

        logger.info(
            "onboarding_execute_started",
            extra={"company": company_name},
        )

        onboarding_saved = False
        onboarding_id = ""

        try:
            record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "company_name": company_name,
                "company_domain": state.get("company_domain", ""),
                "contact_name": state.get("contact_name", ""),
                "contact_email": contact_email,
                "contact_title": state.get("contact_title", ""),
                "proposal_id": state.get("proposal_id", ""),
                "deal_value_cents": state.get("deal_value_cents", 0),
                "template_key": state.get("template_key", "default"),
                "milestones": json.dumps(milestones),
                "welcome_content": state.get("welcome_content", "")[:5000],
                "kickoff_date": state.get("kickoff_date", ""),
                "kickoff_agenda": json.dumps(state.get("kickoff_agenda", [])),
                "status": "in_progress",
                "created_at": now,
            }
            result = (
                self.db.client.table("client_onboarding")
                .insert(record)
                .execute()
            )
            if result.data:
                onboarding_id = result.data[0].get("id", "")
                onboarding_saved = True
                logger.info(
                    "onboarding_record_saved",
                    extra={
                        "onboarding_id": onboarding_id,
                        "company": company_name,
                    },
                )
        except Exception as e:
            logger.error(
                "onboarding_save_error",
                extra={"error": str(e)[:200]},
            )

        return {
            "current_node": "execute_onboarding",
            "onboarding_id": onboarding_id,
            "onboarding_status": "in_progress",
            "onboarding_saved": onboarding_saved,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: OnboardingAgentState
    ) -> dict[str, Any]:
        """Node 5: Build summary and store InsightData for onboarding patterns."""
        now = datetime.now(timezone.utc).isoformat()
        company_name = state.get("company_name", "Unknown")
        template_key = state.get("template_key", "default")
        milestones = state.get("milestones", [])
        onboarding_saved = state.get("onboarding_saved", False)
        approval_status = state.get("human_approval_status", "approved")

        # Build report
        sections = [
            "# Client Onboarding Report",
            f"*Generated: {now}*\n",
            f"## Client",
            f"- **Company:** {company_name}",
            f"- **Contact:** {state.get('contact_name', 'N/A')} ({state.get('contact_email', 'N/A')})",
            f"- **Deal Value:** ${state.get('deal_value_cents', 0) / 100:,.2f}",
            f"\n## Onboarding Plan",
            f"- **Template:** {template_key}",
            f"- **Milestones:** {len(milestones)}",
            f"- **Kickoff Date:** {state.get('kickoff_date', 'TBD')}",
            f"- **Status:** {'Saved' if onboarding_saved else 'Not saved'}",
            f"- **Approval:** {approval_status}",
        ]

        if milestones:
            sections.append("\n## Milestones")
            for i, m in enumerate(milestones, 1):
                sections.append(
                    f"{i}. **{m.get('name', 'N/A')}** — "
                    f"Due: {m.get('due_date', 'TBD')} | "
                    f"{m.get('description', '')[:80]}"
                )

        kickoff_agenda = state.get("kickoff_agenda", [])
        if kickoff_agenda:
            sections.append("\n## Kickoff Agenda")
            for item in kickoff_agenda:
                sections.append(f"- {item}")

        report = "\n".join(sections)

        # Store insight for onboarding patterns
        if onboarding_saved:
            self.store_insight(InsightData(
                insight_type="onboarding_pattern",
                title=f"Onboarding: {company_name} ({template_key} template)",
                content=(
                    f"Onboarded {company_name} using {template_key} template "
                    f"with {len(milestones)} milestones. "
                    f"Deal value: ${state.get('deal_value_cents', 0) / 100:,.2f}. "
                    f"Kickoff scheduled: {state.get('kickoff_date', 'TBD')}."
                ),
                confidence=0.8,
                metadata={
                    "template_key": template_key,
                    "milestone_count": len(milestones),
                    "deal_value_cents": state.get("deal_value_cents", 0),
                    "company_name": company_name,
                },
            ))

        logger.info(
            "onboarding_report_generated",
            extra={
                "company": company_name,
                "milestones": len(milestones),
                "saved": onboarding_saved,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: OnboardingAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<OnboardingAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
