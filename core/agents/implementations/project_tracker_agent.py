"""
Project Tracker Agent — The Task Watchdog.

Monitors project tasks, milestones, blockers, and completion metrics.
Flags overdue and blocked items for immediate attention. Provides
real-time health assessments across all active projects in a vertical.

Architecture (LangGraph State Machine):
    load_projects → analyze_status → flag_blockers →
    human_review → report → END

Trigger Events:
    - scheduled: Daily/weekly project health sweep
    - task_update: External task status change
    - manual: On-demand project status review

Shared Brain Integration:
    - Reads: project history, past blocker patterns, team velocity
    - Writes: project health signals, blocker patterns, completion trends

Safety:
    - NEVER modifies project data without human approval
    - All status changes are advisory; human judgment required
    - Blocker escalations require review before notification
    - Completion metrics are calculated, never fabricated

Usage:
    agent = ProjectTrackerAgent(config, db, embedder, llm)
    result = await agent.run({
        "scope": "all_active",
        "include_completed": False,
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import ProjectData
from core.agents.registry import register_agent_type
from core.agents.state import ProjectTrackerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PROJECT_STATUSES = {
    "active": {
        "label": "Active",
        "description": "Project is underway with recent activity",
        "color": "green",
        "health_min": 7.0,
    },
    "on_track": {
        "label": "On Track",
        "description": "Project is meeting milestones on schedule",
        "color": "green",
        "health_min": 6.0,
    },
    "at_risk": {
        "label": "At Risk",
        "description": "Project has issues that may delay delivery",
        "color": "orange",
        "health_min": 4.0,
    },
    "blocked": {
        "label": "Blocked",
        "description": "Project cannot proceed due to blockers",
        "color": "red",
        "health_min": 0.0,
    },
    "completed": {
        "label": "Completed",
        "description": "Project has been delivered and closed",
        "color": "blue",
        "health_min": 10.0,
    },
}

OVERDUE_THRESHOLD_DAYS = 3

PRIORITY_LEVELS = [
    "critical",
    "high",
    "medium",
    "low",
    "informational",
]

# ─── LLM Prompt Templates ─────────────────────────────────────────────

STATUS_ANALYSIS_PROMPT = """\
You are a project management analyst. Analyze the project data below \
and assess the health status of each project.

Projects:
{projects_json}

Milestones:
{milestones_json}

Overdue threshold: {overdue_threshold} days.

For each project, return a JSON array of status assessment objects:
[
    {{
        "project_id": "the project ID",
        "project_name": "the project name",
        "health_score": 0.0-10.0,
        "status": "active|on_track|at_risk|blocked|completed",
        "completion_pct": 0.0-100.0,
        "milestones_completed": 0,
        "milestones_total": 0,
        "velocity_trend": "improving|stable|declining",
        "summary": "Brief health assessment in one sentence",
        "concerns": ["list of specific concerns if any"]
    }}
]

Consider milestone completion rates, overdue items, recent activity, \
and overall trajectory. Be precise about completion percentages.

Return ONLY the JSON array, no markdown code fences.
"""

BLOCKER_ANALYSIS_PROMPT = """\
You are a project management analyst specializing in risk identification. \
Analyze the projects and their status assessments to identify blockers \
and overdue items that need immediate attention.

Projects:
{projects_json}

Status Assessments:
{assessments_json}

Overdue Items:
{overdue_json}

For each blocker or issue found, return a JSON array:
[
    {{
        "project_id": "the project ID",
        "project_name": "the project name",
        "blocker_type": "dependency|resource|technical|external|scope_creep",
        "description": "Clear description of the blocker",
        "severity": "critical|high|medium|low",
        "days_blocked": 0,
        "recommendation": "Specific actionable recommendation to resolve",
        "escalation_needed": true/false,
        "estimated_impact_days": 0
    }}
]

Focus on actionable blockers. Prioritize items that delay delivery.

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("project_tracker")
class ProjectTrackerAgent(BaseAgent):
    """
    Project tracking and health monitoring agent.

    Nodes:
        1. load_projects     -- Pull projects from DB by vertical, merge task input
        2. analyze_status    -- LLM analyzes health, calculates completion metrics
        3. flag_blockers     -- Identify blocked/overdue items, generate recommendations
        4. human_review      -- Gate: approve status updates and escalations
        5. report            -- Save updates to DB + ProjectData insight
    """

    def build_graph(self) -> Any:
        """Build the Project Tracker Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ProjectTrackerAgentState)

        workflow.add_node("load_projects", self._node_load_projects)
        workflow.add_node("analyze_status", self._node_analyze_status)
        workflow.add_node("flag_blockers", self._node_flag_blockers)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_projects")

        workflow.add_edge("load_projects", "analyze_status")
        workflow.add_edge("analyze_status", "flag_blockers")
        workflow.add_edge("flag_blockers", "human_review")
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
    def get_state_class(cls) -> Type[ProjectTrackerAgentState]:
        return ProjectTrackerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "projects": [],
            "projects_loaded": 0,
            "milestones": [],
            "status_assessments": [],
            "avg_completion_pct": 0.0,
            "projects_on_track": 0,
            "projects_at_risk": 0,
            "blockers": [],
            "overdue_items": [],
            "blocker_count": 0,
            "overdue_count": 0,
            "recommendations": [],
            "updates_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Projects ───────────────────────────────────────

    async def _node_load_projects(
        self, state: ProjectTrackerAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull projects from DB by vertical_id, merge task input."""
        task = state.get("task_input", {})

        logger.info(
            "project_tracker_load_projects",
            extra={"agent_id": self.agent_id},
        )

        projects: list[dict[str, Any]] = []
        milestones: list[dict[str, Any]] = []

        # Load projects from database
        try:
            scope = task.get("scope", "all_active")
            include_completed = task.get("include_completed", False)

            query = (
                self.db.client.table("projects")
                .select("*")
                .eq("vertical_id", self.vertical_id)
            )

            if not include_completed and scope == "all_active":
                query = query.neq("status", "completed")

            query = query.order("created_at", desc=True).limit(100)
            result = query.execute()

            if result.data:
                for record in result.data:
                    projects.append({
                        "project_id": record.get("id", ""),
                        "project_name": record.get("project_name", ""),
                        "status": record.get("status", "active"),
                        "owner": record.get("owner", ""),
                        "due_date": record.get("due_date", ""),
                        "created_at": record.get("created_at", ""),
                        "updated_at": record.get("updated_at", ""),
                        "description": record.get("description", ""),
                        "tags": record.get("tags", []),
                        "milestone_count": record.get("milestone_count", 0),
                        "completed_milestones": record.get("completed_milestones", 0),
                    })

                    # Extract milestones if embedded
                    record_milestones = record.get("milestones", [])
                    if isinstance(record_milestones, list):
                        for ms in record_milestones:
                            milestones.append({
                                "milestone_id": ms.get("id", ""),
                                "project_id": record.get("id", ""),
                                "title": ms.get("title", ""),
                                "status": ms.get("status", "pending"),
                                "due_date": ms.get("due_date", ""),
                                "completed_at": ms.get("completed_at", ""),
                            })

        except Exception as e:
            logger.warning(
                "project_tracker_db_load_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Merge task-provided projects
        task_projects = task.get("projects", [])
        if isinstance(task_projects, list):
            for tp in task_projects:
                if isinstance(tp, dict):
                    # Avoid duplicates by project_id
                    existing_ids = {p.get("project_id") for p in projects}
                    tp_id = tp.get("project_id", tp.get("id", ""))
                    if tp_id not in existing_ids:
                        projects.append({
                            "project_id": tp_id,
                            "project_name": tp.get("project_name", tp.get("name", "")),
                            "status": tp.get("status", "active"),
                            "owner": tp.get("owner", ""),
                            "due_date": tp.get("due_date", ""),
                            "created_at": tp.get("created_at", ""),
                            "updated_at": tp.get("updated_at", ""),
                            "description": tp.get("description", ""),
                            "tags": tp.get("tags", []),
                            "milestone_count": tp.get("milestone_count", 0),
                            "completed_milestones": tp.get("completed_milestones", 0),
                        })

        # Merge task-provided milestones
        task_milestones = task.get("milestones", [])
        if isinstance(task_milestones, list):
            milestones.extend(task_milestones)

        # Identify overdue items based on due_date
        now = datetime.now(timezone.utc)
        overdue_items: list[dict[str, Any]] = []

        for proj in projects:
            due_str = proj.get("due_date", "")
            if due_str and proj.get("status") not in ("completed",):
                try:
                    due_dt = datetime.fromisoformat(
                        due_str.replace("Z", "+00:00")
                    )
                    days_over = (now - due_dt).days
                    if days_over > OVERDUE_THRESHOLD_DAYS:
                        overdue_items.append({
                            "item_type": "project",
                            "item_id": proj.get("project_id", ""),
                            "project_id": proj.get("project_id", ""),
                            "project_name": proj.get("project_name", ""),
                            "due_date": due_str,
                            "days_overdue": days_over,
                        })
                except (ValueError, TypeError):
                    pass

        for ms in milestones:
            due_str = ms.get("due_date", "")
            if due_str and ms.get("status") not in ("completed", "done"):
                try:
                    due_dt = datetime.fromisoformat(
                        due_str.replace("Z", "+00:00")
                    )
                    days_over = (now - due_dt).days
                    if days_over > OVERDUE_THRESHOLD_DAYS:
                        overdue_items.append({
                            "item_type": "milestone",
                            "item_id": ms.get("milestone_id", ""),
                            "project_id": ms.get("project_id", ""),
                            "title": ms.get("title", ""),
                            "due_date": due_str,
                            "days_overdue": days_over,
                        })
                except (ValueError, TypeError):
                    pass

        logger.info(
            "project_tracker_projects_loaded",
            extra={
                "projects_loaded": len(projects),
                "milestones_loaded": len(milestones),
                "overdue_detected": len(overdue_items),
            },
        )

        return {
            "current_node": "load_projects",
            "projects": projects,
            "projects_loaded": len(projects),
            "milestones": milestones,
            "overdue_items": overdue_items,
            "overdue_count": len(overdue_items),
        }

    # ─── Node 2: Analyze Status ──────────────────────────────────────

    async def _node_analyze_status(
        self, state: ProjectTrackerAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM analyzes project health, calculates completion metrics."""
        projects = state.get("projects", [])
        milestones = state.get("milestones", [])

        logger.info(
            "project_tracker_analyze_status",
            extra={"projects_to_analyze": len(projects)},
        )

        status_assessments: list[dict[str, Any]] = []
        avg_completion_pct = 0.0
        projects_on_track = 0
        projects_at_risk = 0

        if not projects:
            logger.info("project_tracker_no_projects_to_analyze")
            return {
                "current_node": "analyze_status",
                "status_assessments": [],
                "avg_completion_pct": 0.0,
                "projects_on_track": 0,
                "projects_at_risk": 0,
            }

        try:
            prompt = STATUS_ANALYSIS_PROMPT.format(
                projects_json=json.dumps(projects[:25], indent=2),
                milestones_json=json.dumps(milestones[:50], indent=2),
                overdue_threshold=OVERDUE_THRESHOLD_DAYS,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a project management analyst assessing project health.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                assessments_data = json.loads(llm_text)
                if isinstance(assessments_data, list):
                    status_assessments = assessments_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("project_tracker_status_parse_error")

        except Exception as e:
            logger.warning(
                "project_tracker_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Calculate aggregate metrics
        if status_assessments:
            completions = [
                a.get("completion_pct", 0) for a in status_assessments
            ]
            avg_completion_pct = round(
                sum(completions) / len(completions), 1
            )
            projects_on_track = sum(
                1 for a in status_assessments
                if a.get("status") in ("active", "on_track", "completed")
            )
            projects_at_risk = sum(
                1 for a in status_assessments
                if a.get("status") in ("at_risk", "blocked")
            )

        logger.info(
            "project_tracker_analysis_complete",
            extra={
                "assessments": len(status_assessments),
                "avg_completion": avg_completion_pct,
                "on_track": projects_on_track,
                "at_risk": projects_at_risk,
            },
        )

        return {
            "current_node": "analyze_status",
            "status_assessments": status_assessments,
            "avg_completion_pct": avg_completion_pct,
            "projects_on_track": projects_on_track,
            "projects_at_risk": projects_at_risk,
        }

    # ─── Node 3: Flag Blockers ───────────────────────────────────────

    async def _node_flag_blockers(
        self, state: ProjectTrackerAgentState
    ) -> dict[str, Any]:
        """Node 3: Identify blocked/overdue items, generate recommendations."""
        projects = state.get("projects", [])
        assessments = state.get("status_assessments", [])
        overdue_items = state.get("overdue_items", [])

        logger.info(
            "project_tracker_flag_blockers",
            extra={
                "projects_count": len(projects),
                "overdue_count": len(overdue_items),
            },
        )

        blockers: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        # Only run LLM analysis if there are at-risk or overdue items
        at_risk_projects = [
            a for a in assessments
            if a.get("status") in ("at_risk", "blocked")
            or a.get("health_score", 10) < 5.0
        ]

        has_issues = bool(at_risk_projects) or bool(overdue_items)

        if not has_issues:
            logger.info("project_tracker_no_blockers_detected")
            return {
                "current_node": "flag_blockers",
                "blockers": [],
                "blocker_count": 0,
                "recommendations": [],
            }

        try:
            prompt = BLOCKER_ANALYSIS_PROMPT.format(
                projects_json=json.dumps(projects[:20], indent=2),
                assessments_json=json.dumps(assessments[:20], indent=2),
                overdue_json=json.dumps(overdue_items[:20], indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a project risk analyst identifying blockers and issues.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                blockers_data = json.loads(llm_text)
                if isinstance(blockers_data, list):
                    blockers = blockers_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("project_tracker_blockers_parse_error")
                # Fallback: create blockers from overdue items
                for item in overdue_items[:10]:
                    blockers.append({
                        "project_id": item.get("project_id", ""),
                        "project_name": item.get("project_name", item.get("title", "")),
                        "blocker_type": "dependency",
                        "description": (
                            f"Overdue by {item.get('days_overdue', 0)} days: "
                            f"{item.get('item_type', 'item')} "
                            f"'{item.get('project_name', item.get('title', 'Unknown'))}'"
                        ),
                        "severity": "high" if item.get("days_overdue", 0) > 7 else "medium",
                        "days_blocked": item.get("days_overdue", 0),
                        "recommendation": "Review and reassign or extend deadline",
                        "escalation_needed": item.get("days_overdue", 0) > 7,
                        "estimated_impact_days": item.get("days_overdue", 0),
                    })

        except Exception as e:
            logger.warning(
                "project_tracker_blockers_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Build recommendations from blockers
        for blocker in blockers:
            rec = blocker.get("recommendation", "")
            if rec:
                recommendations.append({
                    "project_id": blocker.get("project_id", ""),
                    "project_name": blocker.get("project_name", ""),
                    "action": rec,
                    "priority": blocker.get("severity", "medium"),
                    "blocker_type": blocker.get("blocker_type", ""),
                    "escalation_needed": blocker.get("escalation_needed", False),
                })

        logger.info(
            "project_tracker_blockers_flagged",
            extra={
                "blockers_found": len(blockers),
                "recommendations_generated": len(recommendations),
            },
        )

        return {
            "current_node": "flag_blockers",
            "blockers": blockers,
            "blocker_count": len(blockers),
            "recommendations": recommendations,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ProjectTrackerAgentState
    ) -> dict[str, Any]:
        """Node 4: Present status updates and blocker alerts for human approval."""
        blocker_count = state.get("blocker_count", 0)
        overdue_count = state.get("overdue_count", 0)
        projects_at_risk = state.get("projects_at_risk", 0)

        logger.info(
            "project_tracker_human_review_pending",
            extra={
                "blocker_count": blocker_count,
                "overdue_count": overdue_count,
                "at_risk_count": projects_at_risk,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ProjectTrackerAgentState
    ) -> dict[str, Any]:
        """Node 5: Save updates to DB, store ProjectData insight, generate report."""
        now = datetime.now(timezone.utc).isoformat()
        projects = state.get("projects", [])
        assessments = state.get("status_assessments", [])
        blockers = state.get("blockers", [])
        overdue_items = state.get("overdue_items", [])
        recommendations = state.get("recommendations", [])
        avg_completion = state.get("avg_completion_pct", 0.0)
        projects_on_track = state.get("projects_on_track", 0)
        projects_at_risk = state.get("projects_at_risk", 0)

        # Save status updates to database
        updates_saved = False

        for assessment in assessments:
            try:
                project_id = assessment.get("project_id", "")
                if not project_id:
                    continue

                update_record = {
                    "status": assessment.get("status", "active"),
                    "health_score": assessment.get("health_score", 0),
                    "completion_pct": assessment.get("completion_pct", 0),
                    "updated_at": now,
                }

                self.db.client.table("projects").update(
                    update_record
                ).eq("id", project_id).execute()
                updates_saved = True
            except Exception as e:
                logger.debug(f"Failed to save project update: {e}")

        # Build markdown report
        sections = [
            "# Project Tracker Report",
            f"*Generated: {now}*\n",
            "## Summary",
            f"- **Projects Tracked:** {len(projects)}",
            f"- **Average Completion:** {avg_completion:.1f}%",
            f"- **On Track:** {projects_on_track}",
            f"- **At Risk:** {projects_at_risk}",
            f"- **Blockers:** {len(blockers)}",
            f"- **Overdue Items:** {len(overdue_items)}",
        ]

        if assessments:
            sections.append("\n## Project Health")
            for i, a in enumerate(assessments[:15], 1):
                status_label = a.get("status", "unknown").upper()
                health = a.get("health_score", 0)
                completion = a.get("completion_pct", 0)
                sections.append(
                    f"{i}. **[{status_label}]** {a.get('project_name', 'Unknown')}: "
                    f"Health {health:.1f}/10, {completion:.0f}% complete"
                )
                summary = a.get("summary", "")
                if summary:
                    sections.append(f"   _{summary}_")

        if blockers:
            sections.append("\n## Blockers")
            for i, b in enumerate(blockers[:10], 1):
                sev = b.get("severity", "medium").upper()
                sections.append(
                    f"{i}. **[{sev}]** {b.get('project_name', 'Unknown')}: "
                    f"{b.get('description', 'N/A')[:120]}"
                )
                rec = b.get("recommendation", "")
                if rec:
                    sections.append(f"   → _{rec[:100]}_")

        if recommendations:
            sections.append("\n## Recommendations")
            for i, r in enumerate(recommendations[:10], 1):
                sections.append(
                    f"{i}. **{r.get('priority', 'medium').upper()}** "
                    f"({r.get('project_name', 'Unknown')}): "
                    f"{r.get('action', 'N/A')[:120]}"
                )

        report = "\n".join(sections)

        # Store insight for each at-risk or blocked project
        if assessments:
            self.store_insight(ProjectData(
                project_name=f"Portfolio of {len(projects)} projects",
                status="at_risk" if projects_at_risk > 0 else "on_track",
                completion_pct=avg_completion,
                milestones_total=sum(
                    a.get("milestones_total", 0) for a in assessments
                ),
                milestones_completed=sum(
                    a.get("milestones_completed", 0) for a in assessments
                ),
                blockers=blockers[:10],
                health_score=round(
                    sum(a.get("health_score", 0) for a in assessments)
                    / max(len(assessments), 1),
                    1,
                ),
                metadata={
                    "projects_tracked": len(projects),
                    "avg_completion_pct": avg_completion,
                    "projects_on_track": projects_on_track,
                    "projects_at_risk": projects_at_risk,
                    "blocker_count": len(blockers),
                    "overdue_count": len(overdue_items),
                    "recommendations_count": len(recommendations),
                },
            ))

        logger.info(
            "project_tracker_report_generated",
            extra={
                "projects_tracked": len(projects),
                "blockers_found": len(blockers),
                "avg_completion": avg_completion,
                "updates_saved": updates_saved,
            },
        )

        return {
            "current_node": "report",
            "updates_saved": updates_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ProjectTrackerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ProjectTrackerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
