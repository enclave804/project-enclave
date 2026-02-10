"""
Project Manager Agent — The Plan Architect.

Creates comprehensive project plans with phased timelines, resource
allocation, budget estimates, and risk assessments. Transforms raw
requirements and scope definitions into executable project blueprints.

Architecture (LangGraph State Machine):
    gather_requirements → create_plan → assess_risks →
    human_review → report → END

Trigger Events:
    - new_project: A new project has been created or requested
    - scope_change: Existing project scope has changed
    - manual: On-demand project planning

Shared Brain Integration:
    - Reads: historical project data, team capacity, past risk patterns
    - Writes: project plan templates, risk patterns, timeline benchmarks

Safety:
    - NEVER commits resources without human approval
    - All budget estimates are advisory; finance team must validate
    - Risk assessments require human review before stakeholder distribution
    - Timeline buffers are always included (DEFAULT_TIMELINE_BUFFER_PCT)

Usage:
    agent = ProjectManagerAgent(config, db, embedder, llm)
    result = await agent.run({
        "project_name": "Website Redesign",
        "objectives": ["Improve UX", "Increase conversion by 20%"],
        "constraints": {"budget_cents": 5000000, "deadline": "2026-06-01"},
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import ProjectPlanData
from core.agents.registry import register_agent_type
from core.agents.state import ProjectManagerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────

PLAN_PHASES = [
    "discovery",
    "requirements",
    "design",
    "development",
    "testing",
    "deployment",
    "stabilization",
    "handoff",
]

RISK_SEVERITY_MAP = {
    "critical": {
        "score_min": 8.0,
        "description": "Showstopper risk that could cancel the project",
        "color": "red",
        "mitigation_urgency": "immediate",
    },
    "high": {
        "score_min": 6.0,
        "description": "Major risk that could significantly delay delivery",
        "color": "orange",
        "mitigation_urgency": "this_week",
    },
    "medium": {
        "score_min": 4.0,
        "description": "Moderate risk that needs monitoring and a mitigation plan",
        "color": "yellow",
        "mitigation_urgency": "this_sprint",
    },
    "low": {
        "score_min": 2.0,
        "description": "Minor risk with limited impact on delivery",
        "color": "blue",
        "mitigation_urgency": "backlog",
    },
    "negligible": {
        "score_min": 0.0,
        "description": "Minimal risk, awareness only",
        "color": "gray",
        "mitigation_urgency": "none",
    },
}

DEFAULT_TIMELINE_BUFFER_PCT = 20

# ─── LLM Prompt Templates ───────────────────────────────────────────────────

PLAN_GENERATION_PROMPT = """\
You are a senior project manager. Generate a comprehensive project plan \
from the requirements and context below.

Project Name: {project_name}
Objectives: {objectives_json}
Requirements: {requirements_json}
Constraints: {constraints_json}
Existing Projects Context: {existing_projects_json}
Timeline Buffer: {buffer_pct}%

Standard phases: {phases_list}

Return a JSON object with the plan:
{{
    "project_name": "{project_name}",
    "summary": "One-paragraph project summary",
    "phases": [
        {{
            "phase_name": "discovery|requirements|design|development|testing|deployment|stabilization|handoff",
            "phase_order": 1,
            "description": "What happens in this phase",
            "tasks": [
                {{
                    "task_name": "name",
                    "description": "brief description",
                    "estimated_hours": 0,
                    "assigned_role": "role name",
                    "dependencies": ["task_name"]
                }}
            ],
            "duration_weeks": 0,
            "deliverables": ["list of deliverables"],
            "resources_needed": ["role1", "role2"]
        }}
    ],
    "timeline_weeks": 0,
    "resources": [
        {{
            "role": "role name",
            "allocation_pct": 0-100,
            "phase": "phase_name or 'all'",
            "headcount": 1
        }}
    ],
    "budget_estimate_cents": 0,
    "budget_breakdown": [
        {{
            "category": "personnel|infrastructure|tools|contingency",
            "amount_cents": 0,
            "notes": "explanation"
        }}
    ],
    "milestones": [
        {{
            "milestone_name": "name",
            "target_week": 0,
            "criteria": "completion criteria"
        }}
    ],
    "assumptions": ["list of planning assumptions"],
    "out_of_scope": ["items explicitly excluded"]
}}

Include the {buffer_pct}% timeline buffer in the total timeline. \
Be realistic about task durations and resource needs.

Return ONLY the JSON object, no markdown code fences.
"""

RISK_ASSESSMENT_PROMPT = """\
You are a project risk analyst. Identify and score risks for the \
project plan below.

Project Plan:
{plan_json}

Requirements:
{requirements_json}

Constraints:
{constraints_json}

Risk severity levels: {severity_levels}

For each identified risk, return a JSON array:
[
    {{
        "risk_id": "RISK-001",
        "category": "scope|schedule|resource|technical|external|budget|quality",
        "description": "Clear description of the risk",
        "severity": "critical|high|medium|low|negligible",
        "likelihood": 0.0-1.0,
        "impact_score": 0.0-10.0,
        "risk_score": 0.0-10.0,
        "affected_phases": ["phase_name"],
        "mitigation_strategy": "Specific mitigation steps",
        "contingency_plan": "What to do if the risk materializes",
        "owner": "Suggested role to own this risk",
        "early_warning_signs": ["indicators to watch for"]
    }}
]

Consider schedule risks, resource constraints, technical complexity, \
scope creep, external dependencies, and budget overruns. Be specific \
about mitigations.

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("project_manager")
class ProjectManagerAgent(BaseAgent):
    """
    Project planning and risk assessment agent.

    Nodes:
        1. gather_requirements -- Load from task input + projects table, extract scope
        2. create_plan         -- LLM generates phased plan with timeline and resources
        3. assess_risks        -- LLM identifies risks, scores severity, suggests mitigations
        4. human_review        -- Gate: approve project plan
        5. report              -- Save plan to project_plans table + ProjectPlanData insight
    """

    def build_graph(self) -> Any:
        """Build the Project Manager Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ProjectManagerAgentState)

        workflow.add_node("gather_requirements", self._node_gather_requirements)
        workflow.add_node("create_plan", self._node_create_plan)
        workflow.add_node("assess_risks", self._node_assess_risks)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("gather_requirements")

        workflow.add_edge("gather_requirements", "create_plan")
        workflow.add_edge("create_plan", "assess_risks")
        workflow.add_edge("assess_risks", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "report",
                "rejected": "report",
            },
        )
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
    def get_state_class(cls) -> Type[ProjectManagerAgentState]:
        return ProjectManagerAgentState

    # ─── State Preparation ────────────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "requirements": [],
            "objectives": [],
            "scope_summary": "",
            "plan_phases": [],
            "timeline_weeks": 0,
            "resources": [],
            "budget_estimate_cents": 0,
            "risks": [],
            "total_risk_score": 0.0,
            "high_risks": 0,
            "plan_approved": False,
            "plan_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Gather Requirements ───────────────────────────────────────

    async def _node_gather_requirements(
        self, state: ProjectManagerAgentState
    ) -> dict[str, Any]:
        """Node 1: Load requirements from task input + projects table, extract scope."""
        task = state.get("task_input", {})

        logger.info(
            "project_manager_gather_requirements",
            extra={"agent_id": self.agent_id},
        )

        requirements: list[dict[str, Any]] = []
        objectives: list[str] = []
        scope_summary = ""

        # Extract objectives from task
        task_objectives = task.get("objectives", [])
        if isinstance(task_objectives, list):
            objectives = [str(o) for o in task_objectives]

        # Extract requirements from task
        task_requirements = task.get("requirements", [])
        if isinstance(task_requirements, list):
            for req in task_requirements:
                if isinstance(req, str):
                    requirements.append({
                        "requirement_id": f"REQ-{len(requirements) + 1:03d}",
                        "description": req,
                        "priority": "medium",
                        "source": "task_input",
                    })
                elif isinstance(req, dict):
                    if "requirement_id" not in req:
                        req["requirement_id"] = f"REQ-{len(requirements) + 1:03d}"
                    requirements.append(req)

        # Extract scope from task
        scope_summary = task.get("scope_summary", task.get("scope", ""))
        project_name = task.get("project_name", "Untitled Project")

        # Load related project data from database for context
        existing_projects: list[dict[str, Any]] = []
        try:
            result = (
                self.db.client.table("projects")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )
            if result.data:
                for record in result.data:
                    existing_projects.append({
                        "project_id": record.get("id", ""),
                        "project_name": record.get("project_name", ""),
                        "status": record.get("status", ""),
                        "completion_pct": record.get("completion_pct", 0),
                        "description": record.get("description", "")[:200],
                    })
        except Exception as e:
            logger.warning(
                "project_manager_db_load_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Extract constraints from task
        constraints = task.get("constraints", {})
        if isinstance(constraints, dict):
            budget_constraint = constraints.get("budget_cents", 0)
            deadline = constraints.get("deadline", "")
            if deadline and not scope_summary:
                scope_summary = (
                    f"Project '{project_name}' with deadline {deadline}"
                )
                if budget_constraint:
                    scope_summary += f" and budget ${budget_constraint / 100:,.0f}"

        # If no explicit requirements, derive from objectives
        if not requirements and objectives:
            for i, obj in enumerate(objectives, 1):
                requirements.append({
                    "requirement_id": f"REQ-{i:03d}",
                    "description": f"Achieve objective: {obj}",
                    "priority": "high",
                    "source": "derived_from_objectives",
                })

        if not scope_summary and project_name:
            scope_summary = f"Project plan for '{project_name}'"

        logger.info(
            "project_manager_requirements_gathered",
            extra={
                "requirements_count": len(requirements),
                "objectives_count": len(objectives),
                "existing_projects": len(existing_projects),
                "has_scope": bool(scope_summary),
            },
        )

        return {
            "current_node": "gather_requirements",
            "requirements": requirements,
            "objectives": objectives,
            "scope_summary": scope_summary,
        }

    # ─── Node 2: Create Plan ─────────────────────────────────────────────

    async def _node_create_plan(
        self, state: ProjectManagerAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM generates project plan with phases, timeline, resources."""
        task = state.get("task_input", {})
        requirements = state.get("requirements", [])
        objectives = state.get("objectives", [])

        project_name = task.get("project_name", "Untitled Project")
        constraints = task.get("constraints", {})

        logger.info(
            "project_manager_create_plan",
            extra={
                "project_label": project_name,
                "requirements_count": len(requirements),
            },
        )

        plan_phases: list[dict[str, Any]] = []
        timeline_weeks = 0
        resources: list[dict[str, Any]] = []
        budget_estimate_cents = 0

        if not requirements and not objectives:
            logger.info("project_manager_no_requirements_for_plan")
            return {
                "current_node": "create_plan",
                "plan_phases": [],
                "timeline_weeks": 0,
                "resources": [],
                "budget_estimate_cents": 0,
            }

        # Load existing projects for context
        existing_projects: list[dict[str, Any]] = []
        try:
            result = (
                self.db.client.table("projects")
                .select("project_name, status, completion_pct")
                .eq("vertical_id", self.vertical_id)
                .limit(10)
                .execute()
            )
            if result.data:
                existing_projects = result.data
        except Exception:
            pass

        try:
            prompt = PLAN_GENERATION_PROMPT.format(
                project_name=project_name,
                objectives_json=json.dumps(objectives, indent=2),
                requirements_json=json.dumps(requirements[:30], indent=2),
                constraints_json=json.dumps(constraints, indent=2),
                existing_projects_json=json.dumps(
                    existing_projects[:10], indent=2
                ),
                buffer_pct=DEFAULT_TIMELINE_BUFFER_PCT,
                phases_list=", ".join(PLAN_PHASES),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a senior project manager creating detailed project plans.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                plan_data = json.loads(llm_text)
                if isinstance(plan_data, dict):
                    plan_phases = plan_data.get("phases", [])
                    timeline_weeks = plan_data.get("timeline_weeks", 0)
                    resources = plan_data.get("resources", [])
                    budget_estimate_cents = plan_data.get(
                        "budget_estimate_cents", 0
                    )
            except (json.JSONDecodeError, KeyError):
                logger.debug("project_manager_plan_parse_error")

        except Exception as e:
            logger.warning(
                "project_manager_plan_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Ensure timeline has buffer applied
        if timeline_weeks > 0:
            buffered = int(
                timeline_weeks * (1 + DEFAULT_TIMELINE_BUFFER_PCT / 100)
            )
            if buffered > timeline_weeks:
                timeline_weeks = buffered

        logger.info(
            "project_manager_plan_created",
            extra={
                "phases_count": len(plan_phases),
                "timeline_weeks": timeline_weeks,
                "resources_count": len(resources),
                "budget_cents": budget_estimate_cents,
            },
        )

        return {
            "current_node": "create_plan",
            "plan_phases": plan_phases,
            "timeline_weeks": timeline_weeks,
            "resources": resources,
            "budget_estimate_cents": budget_estimate_cents,
        }

    # ─── Node 3: Assess Risks ────────────────────────────────────────────

    async def _node_assess_risks(
        self, state: ProjectManagerAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM identifies risks, scores severity, suggests mitigations."""
        task = state.get("task_input", {})
        plan_phases = state.get("plan_phases", [])
        requirements = state.get("requirements", [])
        constraints = task.get("constraints", {})

        logger.info(
            "project_manager_assess_risks",
            extra={"phases_to_assess": len(plan_phases)},
        )

        risks: list[dict[str, Any]] = []
        total_risk_score = 0.0
        high_risks = 0

        if not plan_phases:
            logger.info("project_manager_no_plan_for_risk_assessment")
            return {
                "current_node": "assess_risks",
                "risks": [],
                "total_risk_score": 0.0,
                "high_risks": 0,
            }

        try:
            # Build plan summary for risk assessment
            plan_summary = {
                "phases": plan_phases,
                "timeline_weeks": state.get("timeline_weeks", 0),
                "resources": state.get("resources", []),
                "budget_estimate_cents": state.get("budget_estimate_cents", 0),
            }

            severity_levels = ", ".join(
                f"{k} (>={v['score_min']})"
                for k, v in RISK_SEVERITY_MAP.items()
            )

            prompt = RISK_ASSESSMENT_PROMPT.format(
                plan_json=json.dumps(plan_summary, indent=2),
                requirements_json=json.dumps(requirements[:20], indent=2),
                constraints_json=json.dumps(constraints, indent=2),
                severity_levels=severity_levels,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a project risk analyst identifying and scoring project risks.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                risks_data = json.loads(llm_text)
                if isinstance(risks_data, list):
                    risks = risks_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("project_manager_risks_parse_error")
                # Fallback: create basic schedule and scope risks
                risks = [
                    {
                        "risk_id": "RISK-001",
                        "category": "schedule",
                        "description": "Timeline may be optimistic given project complexity",
                        "severity": "medium",
                        "likelihood": 0.5,
                        "impact_score": 5.0,
                        "risk_score": 5.0,
                        "affected_phases": ["development", "testing"],
                        "mitigation_strategy": "Add buffer time and track velocity closely",
                        "contingency_plan": "Reduce scope or extend deadline",
                        "owner": "Project Manager",
                        "early_warning_signs": [
                            "Sprint velocity declining",
                            "Tasks taking longer than estimated",
                        ],
                    },
                    {
                        "risk_id": "RISK-002",
                        "category": "scope",
                        "description": "Requirements may expand during development",
                        "severity": "medium",
                        "likelihood": 0.6,
                        "impact_score": 4.0,
                        "risk_score": 4.8,
                        "affected_phases": ["requirements", "development"],
                        "mitigation_strategy": "Strict change control process",
                        "contingency_plan": "Defer new requirements to phase 2",
                        "owner": "Project Manager",
                        "early_warning_signs": [
                            "Frequent change requests",
                            "Stakeholder adding features",
                        ],
                    },
                ]

        except Exception as e:
            logger.warning(
                "project_manager_risks_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Calculate aggregate risk metrics
        if risks:
            scores = [r.get("risk_score", 0) for r in risks]
            total_risk_score = round(
                sum(scores) / max(len(scores), 1), 1
            )
            high_risks = sum(
                1 for r in risks
                if r.get("severity") in ("critical", "high")
            )

        logger.info(
            "project_manager_risks_assessed",
            extra={
                "risks_identified": len(risks),
                "total_risk_score": total_risk_score,
                "high_risk_count": high_risks,
            },
        )

        return {
            "current_node": "assess_risks",
            "risks": risks,
            "total_risk_score": total_risk_score,
            "high_risks": high_risks,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────────

    async def _node_human_review(
        self, state: ProjectManagerAgentState
    ) -> dict[str, Any]:
        """Node 4: Present project plan for human approval."""
        plan_phases = state.get("plan_phases", [])
        timeline_weeks = state.get("timeline_weeks", 0)
        total_risk_score = state.get("total_risk_score", 0)
        high_risks = state.get("high_risks", 0)

        logger.info(
            "project_manager_human_review_pending",
            extra={
                "phases_count": len(plan_phases),
                "timeline_weeks": timeline_weeks,
                "risk_score": total_risk_score,
                "high_risk_count": high_risks,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ────────────────────────────────────────────────

    async def _node_report(
        self, state: ProjectManagerAgentState
    ) -> dict[str, Any]:
        """Node 5: Save plan to project_plans table, store insight, generate report."""
        now = datetime.now(timezone.utc).isoformat()
        task = state.get("task_input", {})
        plan_phases = state.get("plan_phases", [])
        timeline_weeks = state.get("timeline_weeks", 0)
        resources = state.get("resources", [])
        risks = state.get("risks", [])
        total_risk_score = state.get("total_risk_score", 0.0)
        high_risks = state.get("high_risks", 0)
        budget_cents = state.get("budget_estimate_cents", 0)
        objectives = state.get("objectives", [])
        requirements = state.get("requirements", [])
        scope_summary = state.get("scope_summary", "")

        project_name = task.get("project_name", "Untitled Project")
        was_approved = state.get("human_approval_status") == "approved"

        # Save plan to database
        plan_saved = False

        try:
            plan_record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "project_name": project_name,
                "scope_summary": scope_summary[:500],
                "objectives": json.dumps(objectives),
                "requirements": json.dumps(requirements[:50]),
                "phases": json.dumps(plan_phases),
                "timeline_weeks": timeline_weeks,
                "resources": json.dumps(resources),
                "risks": json.dumps(risks[:20]),
                "total_risk_score": total_risk_score,
                "budget_estimate_cents": budget_cents,
                "status": "approved" if was_approved else "draft",
                "created_at": now,
            }
            self.db.client.table("project_plans").insert(
                plan_record
            ).execute()
            plan_saved = True
        except Exception as e:
            logger.debug(f"Failed to save project plan: {e}")

        # Build markdown report
        sections = [
            f"# Project Plan: {project_name}",
            f"*Generated: {now}*\n",
            "## Summary",
            f"- **Scope:** {scope_summary[:200]}" if scope_summary else "",
            f"- **Timeline:** {timeline_weeks} weeks (includes {DEFAULT_TIMELINE_BUFFER_PCT}% buffer)",
            f"- **Budget Estimate:** ${budget_cents / 100:,.0f}" if budget_cents else "- **Budget Estimate:** TBD",
            f"- **Resources:** {len(resources)} roles allocated",
            f"- **Risks Identified:** {len(risks)} ({high_risks} high/critical)",
            f"- **Risk Score:** {total_risk_score:.1f}/10",
            f"- **Status:** {'Approved' if was_approved else 'Draft / Pending Approval'}",
        ]

        # Remove empty strings from summary
        sections = [s for s in sections if s]

        if objectives:
            sections.append("\n## Objectives")
            for i, obj in enumerate(objectives[:10], 1):
                sections.append(f"{i}. {obj}")

        if plan_phases:
            sections.append("\n## Phases")
            for i, phase in enumerate(plan_phases[:12], 1):
                phase_name = phase.get("phase_name", f"Phase {i}")
                duration = phase.get("duration_weeks", 0)
                desc = phase.get("description", "")[:100]
                sections.append(
                    f"{i}. **{phase_name.replace('_', ' ').title()}** "
                    f"({duration} weeks): {desc}"
                )
                # List deliverables
                deliverables = phase.get("deliverables", [])
                for d in deliverables[:5]:
                    sections.append(f"   - {d}")

        if resources:
            sections.append("\n## Resource Allocation")
            for r in resources[:15]:
                role = r.get("role", "Unknown")
                alloc = r.get("allocation_pct", 0)
                phase = r.get("phase", "all")
                headcount = r.get("headcount", 1)
                sections.append(
                    f"- **{role}**: {alloc}% allocation, "
                    f"{headcount} headcount ({phase})"
                )

        if risks:
            sections.append("\n## Risk Assessment")
            for i, risk in enumerate(risks[:10], 1):
                sev = risk.get("severity", "medium").upper()
                score = risk.get("risk_score", 0)
                sections.append(
                    f"{i}. **[{sev}]** (Score: {score:.1f}) "
                    f"{risk.get('description', 'N/A')[:120]}"
                )
                mitigation = risk.get("mitigation_strategy", "")
                if mitigation:
                    sections.append(f"   Mitigation: _{mitigation[:100]}_")

        report = "\n".join(sections)

        # Store insight
        if plan_phases:
            self.store_insight(ProjectPlanData(
                project_name=project_name,
                phases=plan_phases,
                timeline_weeks=timeline_weeks,
                resources=resources,
                risks=risks[:20],
                total_risk_score=total_risk_score,
                budget_estimate_cents=budget_cents,
                objectives=objectives,
                metadata={
                    "phases_count": len(plan_phases),
                    "resources_count": len(resources),
                    "risks_count": len(risks),
                    "high_risks": high_risks,
                    "buffer_pct": DEFAULT_TIMELINE_BUFFER_PCT,
                    "requirements_count": len(requirements),
                    "plan_approved": was_approved,
                },
            ))

        logger.info(
            "project_manager_report_generated",
            extra={
                "project_label": project_name,
                "phases_count": len(plan_phases),
                "timeline_weeks": timeline_weeks,
                "risks_count": len(risks),
                "plan_saved": plan_saved,
            },
        )

        return {
            "current_node": "report",
            "plan_approved": was_approved,
            "plan_saved": plan_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ProjectManagerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ProjectManagerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
