"""
Remediation Guide Agent — The Fix-It Coordinator.

Aggregates findings from all security domain agents (vuln scanner,
network analyst, IAM analyst, cloud security, etc.), prioritizes them
by risk, generates step-by-step remediation instructions, creates
actionable tasks, and verifies that fixes were applied.

Architecture (LangGraph State Machine):
    load_findings → prioritize → generate_steps →
    human_review → create_tasks → verify → report → END

Trigger Events:
    - assessment_complete: All domain agents have finished their scans
    - manual: On-demand remediation planning
    - finding_added: New critical finding requires remediation plan

Shared Brain Integration:
    - Reads: All security findings from domain agents, compliance gaps
    - Writes: Remediation progress, fix effectiveness, time-to-remediate

Safety:
    - NEVER applies fixes automatically — generates instructions only
    - All remediation plans require human_review before task creation
    - Verification is advisory (checks reported status, not live systems)
    - Estimated effort is a guideline, not a commitment

Usage:
    agent = RemediationGuideAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "source_agents": ["vuln_scanner", "iam_analyst", "cloud_security"],
    })
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import RemediationGuideAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PRIORITY_WEIGHTS = {
    "severity": 0.40,       # How bad is the finding?
    "exploitability": 0.25, # How easy to exploit?
    "business_impact": 0.20,# What's the business consequence?
    "fix_complexity": 0.15, # How hard is the fix? (inverse — easy fixes rise)
}

REMEDIATION_CATEGORIES = {
    "configuration": {
        "label": "Configuration Change",
        "avg_hours": 2,
        "description": "Modify system, service, or application settings",
    },
    "patch": {
        "label": "Patch / Update",
        "avg_hours": 4,
        "description": "Apply security patch or software update",
    },
    "architecture": {
        "label": "Architecture Change",
        "avg_hours": 40,
        "description": "Modify system architecture or deployment",
    },
    "policy": {
        "label": "Policy / Process Update",
        "avg_hours": 8,
        "description": "Create or update security policies and procedures",
    },
    "training": {
        "label": "Training & Awareness",
        "avg_hours": 4,
        "description": "Staff training or awareness program update",
    },
    "monitoring": {
        "label": "Monitoring / Detection",
        "avg_hours": 6,
        "description": "Add or improve monitoring, alerting, or logging",
    },
}

VERIFICATION_METHODS = {
    "rescan": {"label": "Re-scan", "description": "Run the original scan again to verify fix"},
    "config_review": {"label": "Config Review", "description": "Review configuration manually"},
    "log_check": {"label": "Log Verification", "description": "Check logs for expected behavior"},
    "test_case": {"label": "Test Case", "description": "Execute specific test to confirm fix"},
    "attestation": {"label": "Attestation", "description": "Owner attests that fix was applied"},
}

REMEDIATION_SYSTEM_PROMPT = """\
You are a cybersecurity remediation specialist. Given the security \
findings below, generate step-by-step remediation instructions.

For each finding, produce a JSON object:
{{
    "finding_id": "...",
    "steps": [
        {{"step_number": 1, "action": "...", "detail": "...", "command": "..."}}
    ],
    "category": "configuration|patch|architecture|policy|training|monitoring",
    "estimated_hours": N,
    "verification_method": "rescan|config_review|log_check|test_case|attestation",
    "prerequisites": ["..."],
    "rollback_plan": "..."
}}

Company: {company_name}
Findings count: {finding_count}

Prioritize fixes that reduce the most risk with the least effort.
Return ONLY a JSON array of remediation plans, no markdown code fences.
"""


@register_agent_type("remediation_guide")
class RemediationGuideAgent(BaseAgent):
    """
    Cross-domain remediation coordination agent for Enclave Guard.

    Nodes:
        1. load_findings    -- Aggregate findings from all security domain agents
        2. prioritize       -- Score and rank findings by risk and fix effort
        3. generate_steps   -- Create step-by-step remediation instructions
        4. human_review     -- Gate: approve plans before task creation
        5. create_tasks     -- Create remediation tasks in tracking system
        6. verify           -- Check remediation status against reported fixes
        7. report           -- Generate remediation progress report
    """

    def build_graph(self) -> Any:
        """Build the Remediation Guide Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(RemediationGuideAgentState)

        workflow.add_node("load_findings", self._node_load_findings)
        workflow.add_node("prioritize", self._node_prioritize)
        workflow.add_node("generate_steps", self._node_generate_steps)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("create_tasks", self._node_create_tasks)
        workflow.add_node("verify", self._node_verify)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_findings")

        workflow.add_edge("load_findings", "prioritize")
        workflow.add_edge("prioritize", "generate_steps")
        workflow.add_edge("generate_steps", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "create_tasks",
                "rejected": "report",
            },
        )
        workflow.add_edge("create_tasks", "verify")
        workflow.add_edge("verify", "report")
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
    def get_state_class(cls) -> Type[RemediationGuideAgentState]:
        return RemediationGuideAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "source_findings": [],
            "source_agents": (task or {}).get("source_agents", []),
            "finding_count": 0,
            "prioritized_findings": [],
            "priority_matrix": {},
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "remediation_plans": [],
            "total_steps": 0,
            "estimated_total_hours": 0.0,
            "tasks_created": [],
            "tasks_count": 0,
            "verification_results": [],
            "verified_count": 0,
            "failed_verification_count": 0,
            "plans_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Findings ──────────────────────────────────────

    async def _node_load_findings(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 1: Aggregate findings from all security domain agents."""
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")
        source_agents = state.get("source_agents", [])

        logger.info(
            "remediation_load_findings",
            extra={"company": company, "sources": source_agents},
        )

        all_findings: list[dict[str, Any]] = []

        # Load findings from DB (security_findings table)
        try:
            query = self.db.client.table("security_findings").select("*").eq(
                "vertical_id", self.vertical_id
            )
            if company and company != "Unknown":
                query = query.eq("company_name", company)

            result = query.execute()
            all_findings = result.data or []
        except Exception as e:
            logger.debug(f"Failed to load findings from DB: {e}")

        # If no DB findings, check task input for inline findings
        if not all_findings:
            inline = task.get("findings", [])
            if isinstance(inline, list):
                all_findings = inline

        # Deduplicate by finding text
        seen = set()
        deduped: list[dict[str, Any]] = []
        for f in all_findings:
            key = f.get("finding", "")[:100]
            if key and key not in seen:
                seen.add(key)
                deduped.append(f)

        logger.info(
            "remediation_findings_loaded",
            extra={
                "total": len(all_findings),
                "deduped": len(deduped),
            },
        )

        return {
            "current_node": "load_findings",
            "source_findings": deduped,
            "finding_count": len(deduped),
        }

    # ─── Node 2: Prioritize ─────────────────────────────────────────

    async def _node_prioritize(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 2: Score and rank findings by risk-weighted priority."""
        findings = state.get("source_findings", [])

        logger.info("remediation_prioritize", extra={"findings": len(findings)})

        severity_scores = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        priority_matrix: dict[str, Any] = {}

        scored: list[dict[str, Any]] = []
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for finding in findings:
            finding_id = finding.get("id", str(uuid.uuid4())[:8])
            sev = finding.get("severity", "low")
            counts[sev] = counts.get(sev, 0) + 1

            # Base severity score
            sev_score = severity_scores.get(sev, 1)

            # Exploitability: network-accessible findings are more exploitable
            domain = finding.get("domain", finding.get("category", ""))
            exploit_score = 7  # Default medium
            if domain in ("network", "storage", "cloud"):
                exploit_score = 9  # Externally exploitable
            elif domain in ("iam", "password"):
                exploit_score = 6  # Requires auth context

            # Business impact: data-related findings have higher impact
            impact_score = 5  # Default
            if sev == "critical":
                impact_score = 9
            elif sev == "high":
                impact_score = 7

            # Fix complexity (inverse: easy fixes = higher priority)
            category = finding.get("category", "configuration")
            fix_info = REMEDIATION_CATEGORIES.get(category, REMEDIATION_CATEGORIES["configuration"])
            fix_hours = fix_info["avg_hours"]
            complexity_score = max(1, 10 - (fix_hours / 5))  # Lower hours = higher score

            # Weighted priority score
            priority_score = (
                sev_score * PRIORITY_WEIGHTS["severity"]
                + exploit_score * PRIORITY_WEIGHTS["exploitability"]
                + impact_score * PRIORITY_WEIGHTS["business_impact"]
                + complexity_score * PRIORITY_WEIGHTS["fix_complexity"]
            )

            priority_matrix[finding_id] = {
                "score": round(priority_score, 2),
                "severity": sev_score,
                "exploitability": exploit_score,
                "business_impact": impact_score,
                "fix_complexity": round(complexity_score, 1),
            }

            scored.append({
                **finding,
                "finding_id": finding_id,
                "priority_score": round(priority_score, 2),
            })

        # Sort by priority (highest first)
        scored.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

        return {
            "current_node": "prioritize",
            "prioritized_findings": scored,
            "priority_matrix": priority_matrix,
            "critical_count": counts.get("critical", 0),
            "high_count": counts.get("high", 0),
            "medium_count": counts.get("medium", 0),
            "low_count": counts.get("low", 0),
        }

    # ─── Node 3: Generate Steps ──────────────────────────────────────

    async def _node_generate_steps(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 3: Generate step-by-step remediation instructions per finding."""
        findings = state.get("prioritized_findings", [])

        logger.info("remediation_generate_steps", extra={"findings": len(findings)})

        plans: list[dict[str, Any]] = []
        total_steps = 0
        total_hours = 0.0

        for finding in findings:
            finding_id = finding.get("finding_id", "")
            sev = finding.get("severity", "low")
            category = finding.get("category", "configuration")
            recommendation = finding.get("recommendation", "")
            domain = finding.get("domain", "")

            # Determine remediation category
            if "patch" in recommendation.lower() or "update" in recommendation.lower():
                rem_category = "patch"
            elif "policy" in recommendation.lower() or "process" in recommendation.lower():
                rem_category = "policy"
            elif "monitor" in recommendation.lower() or "log" in recommendation.lower():
                rem_category = "monitoring"
            elif "train" in recommendation.lower():
                rem_category = "training"
            elif "architecture" in category.lower():
                rem_category = "architecture"
            else:
                rem_category = "configuration"

            cat_info = REMEDIATION_CATEGORIES.get(rem_category, REMEDIATION_CATEGORIES["configuration"])
            est_hours = cat_info["avg_hours"]

            # Scale estimate by severity
            if sev == "critical":
                est_hours *= 1.5
            elif sev == "low":
                est_hours *= 0.5

            # Determine verification method
            if domain in ("network", "cloud", "storage"):
                verify_method = "rescan"
            elif domain in ("iam", "password"):
                verify_method = "config_review"
            elif domain in ("logging",):
                verify_method = "log_check"
            else:
                verify_method = "config_review"

            # Generate steps from recommendation
            steps = []
            step_num = 1

            # Step 1: always assess current state
            steps.append({
                "step_number": step_num,
                "action": "Assess current state",
                "detail": f"Document the current configuration for {finding.get('finding', 'this finding')}",
            })
            step_num += 1

            # Step 2: apply the fix
            if recommendation:
                steps.append({
                    "step_number": step_num,
                    "action": "Apply remediation",
                    "detail": recommendation,
                })
                step_num += 1

            # Step 3: verify
            verify_info = VERIFICATION_METHODS.get(verify_method, VERIFICATION_METHODS["config_review"])
            steps.append({
                "step_number": step_num,
                "action": f"Verify fix ({verify_info['label']})",
                "detail": verify_info["description"],
            })
            step_num += 1

            # Step 4: document
            steps.append({
                "step_number": step_num,
                "action": "Document changes",
                "detail": "Record all changes made, update configuration management records",
            })

            total_steps += len(steps)
            total_hours += est_hours

            plans.append({
                "finding_id": finding_id,
                "finding": finding.get("finding", ""),
                "severity": sev,
                "category": rem_category,
                "steps": steps,
                "estimated_hours": round(est_hours, 1),
                "verification_method": verify_method,
                "prerequisites": [],
                "rollback_plan": f"Revert to documented previous configuration state",
            })

        return {
            "current_node": "generate_steps",
            "remediation_plans": plans,
            "total_steps": total_steps,
            "estimated_total_hours": round(total_hours, 1),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 4: Present remediation plans for human approval."""
        plans = state.get("remediation_plans", [])
        logger.info(
            "remediation_human_review_pending",
            extra={
                "plans": len(plans),
                "total_hours": state.get("estimated_total_hours", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Create Tasks ───────────────────────────────────────

    async def _node_create_tasks(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 5: Create remediation tasks in the tracking system."""
        plans = state.get("remediation_plans", [])
        task_input = state.get("task_input", {})
        company = task_input.get("company_name", "Unknown")

        logger.info("remediation_create_tasks", extra={"plans": len(plans)})

        tasks: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for plan in plans:
            sev = plan.get("severity", "low")
            # Set due date based on severity
            if sev == "critical":
                due_delta = timedelta(days=7)
            elif sev == "high":
                due_delta = timedelta(days=14)
            elif sev == "medium":
                due_delta = timedelta(days=30)
            else:
                due_delta = timedelta(days=60)

            task_id = str(uuid.uuid4())[:8]
            task_record = {
                "task_id": task_id,
                "title": f"[{sev.upper()}] Remediate: "
                         f"{plan.get('finding', '')[:80]}",
                "finding_id": plan.get("finding_id", ""),
                "severity": sev,
                "category": plan.get("category", "configuration"),
                "status": "open",
                "estimated_hours": plan.get("estimated_hours", 0),
                "due_date": (now + due_delta).date().isoformat(),
                "created_at": now.isoformat(),
                "steps": plan.get("steps", []),
                "verification_method": plan.get("verification_method", ""),
            }

            # Persist to DB
            try:
                self.db.client.table("remediation_tasks").insert({
                    "id": task_id,
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "company_name": company,
                    "finding_id": plan.get("finding_id", ""),
                    "title": task_record["title"],
                    "severity": sev,
                    "category": plan.get("category", ""),
                    "status": "open",
                    "estimated_hours": plan.get("estimated_hours", 0),
                    "due_date": task_record["due_date"],
                    "steps": plan.get("steps", []),
                    "created_at": now.isoformat(),
                }).execute()
            except Exception as e:
                logger.debug(f"Failed to persist task {task_id}: {e}")

            tasks.append(task_record)

        return {
            "current_node": "create_tasks",
            "tasks_created": tasks,
            "tasks_count": len(tasks),
            "plans_approved": True,
        }

    # ─── Node 6: Verify ────────────────────────────────────────────

    async def _node_verify(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 6: Check remediation status against reported fixes."""
        tasks = state.get("tasks_created", [])
        task_input = state.get("task_input", {})
        company = task_input.get("company_name", "Unknown")

        logger.info("remediation_verify", extra={"tasks": len(tasks)})

        verification_results: list[dict[str, Any]] = []
        verified = 0
        failed = 0

        # Check if any tasks have already been marked as fixed
        for task in tasks:
            finding_id = task.get("finding_id", "")
            method = task.get("verification_method", "attestation")

            # In production: check status from DB or external systems
            # For now, all newly created tasks are "pending verification"
            result = {
                "finding_id": finding_id,
                "task_id": task.get("task_id", ""),
                "verified": False,
                "method": method,
                "notes": "Task created — awaiting remediation",
                "status": "pending",
            }
            verification_results.append(result)

        # Write insight to shared brain
        self.store_insight(InsightData(
            insight_type="remediation_plan",
            title=f"Remediation: {company} — "
                  f"{len(tasks)} tasks created",
            content=(
                f"Remediation plan for {company}: "
                f"{len(tasks)} tasks created. "
                f"Critical: {state.get('critical_count', 0)}, "
                f"High: {state.get('high_count', 0)}, "
                f"Medium: {state.get('medium_count', 0)}, "
                f"Low: {state.get('low_count', 0)}. "
                f"Estimated effort: {state.get('estimated_total_hours', 0)} hours."
            ),
            confidence=0.85,
            metadata={
                "company": company,
                "task_count": len(tasks),
                "total_hours": state.get("estimated_total_hours", 0),
            },
        ))

        return {
            "current_node": "verify",
            "verification_results": verification_results,
            "verified_count": verified,
            "failed_verification_count": failed,
            "knowledge_written": True,
        }

    # ─── Node 7: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: RemediationGuideAgentState
    ) -> dict[str, Any]:
        """Node 7: Generate remediation progress report."""
        now = datetime.now(timezone.utc).isoformat()
        task_input = state.get("task_input", {})
        company = task_input.get("company_name", "Unknown")
        plans = state.get("remediation_plans", [])
        tasks = state.get("tasks_created", [])

        # Count by category
        by_category: dict[str, int] = {}
        for plan in plans:
            cat = plan.get("category", "other")
            by_category[cat] = by_category.get(cat, 0) + 1

        sections = [
            "# Remediation Guide Report",
            f"*Generated: {now}*\n",
            f"## Organization: {company}",
            f"- **Source Agents:** {', '.join(state.get('source_agents', [])) or 'All'}",
            f"\n## Findings Summary",
            f"- **Total Findings:** {state.get('finding_count', 0)}",
            f"- **Critical:** {state.get('critical_count', 0)}",
            f"- **High:** {state.get('high_count', 0)}",
            f"- **Medium:** {state.get('medium_count', 0)}",
            f"- **Low:** {state.get('low_count', 0)}",
            f"\n## Remediation Plans",
            f"- **Plans Generated:** {len(plans)}",
            f"- **Total Steps:** {state.get('total_steps', 0)}",
            f"- **Estimated Effort:** {state.get('estimated_total_hours', 0)} hours",
        ]

        sections.append("\n## By Remediation Category")
        for cat_id, cat_info in REMEDIATION_CATEGORIES.items():
            count = by_category.get(cat_id, 0)
            if count > 0:
                sections.append(f"- **{cat_info['label']}:** {count}")

        sections.append(f"\n## Task Status")
        sections.append(f"- **Tasks Created:** {len(tasks)}")
        sections.append(f"- **Verified:** {state.get('verified_count', 0)}")
        sections.append(
            f"- **Failed Verification:** "
            f"{state.get('failed_verification_count', 0)}"
        )
        sections.append(
            f"- **Pending:** "
            f"{len(tasks) - state.get('verified_count', 0) - state.get('failed_verification_count', 0)}"
        )

        # Top priority items
        prioritized = state.get("prioritized_findings", [])
        top_5 = prioritized[:5]
        if top_5:
            sections.append("\n## Top Priority Remediation Items")
            for i, item in enumerate(top_5, 1):
                sections.append(
                    f"  {i}. [{item.get('severity', 'N/A').upper()}] "
                    f"{item.get('finding', 'N/A')[:100]}"
                )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: RemediationGuideAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<RemediationGuideAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
