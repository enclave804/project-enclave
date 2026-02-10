"""
Incident Readiness Agent — The Crisis Planner.

Evaluates incident response plans, backup strategies, recovery procedures,
and communication plans. Produces a readiness score (0-100) and letter
grade (A-F) to quantify organizational preparedness for cyber incidents.

Architecture (LangGraph State Machine):
    assess_ir_plan → evaluate_backups → check_communication →
    human_review → score_readiness → report → END

Trigger Events:
    - security_assessment: Full IR readiness evaluation
    - questionnaire_submitted: Client completes IR questionnaire
    - manual: On-demand readiness check

Shared Brain Integration:
    - Reads: compliance requirements, industry benchmarks, peer assessments
    - Writes: readiness patterns, backup gap insights, IR maturity data

Safety:
    - NEVER tests actual disaster recovery (assessment only)
    - All findings require human_review gate before scoring
    - Readiness scores are advisory, not certification
    - Does not access or validate actual backup systems

Usage:
    agent = IncidentReadinessAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "questionnaire": {...},
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import IncidentReadinessAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

READINESS_GRADES = {
    "A": {"min_score": 90, "label": "Excellent", "description": "Well-prepared for incidents"},
    "B": {"min_score": 75, "label": "Good", "description": "Minor gaps to address"},
    "C": {"min_score": 60, "label": "Fair", "description": "Significant gaps exist"},
    "D": {"min_score": 40, "label": "Poor", "description": "Major preparedness gaps"},
    "F": {"min_score": 0, "label": "Failing", "description": "Not prepared for incidents"},
}

IR_PLAN_REQUIREMENTS = {
    "plan_exists": {"weight": 15, "description": "Documented IR plan exists"},
    "roles_defined": {"weight": 10, "description": "IR team roles and responsibilities defined"},
    "escalation_procedures": {"weight": 10, "description": "Clear escalation procedures"},
    "runbooks": {"weight": 10, "description": "Incident-specific runbooks available"},
    "tested_annually": {"weight": 15, "description": "IR plan tested within past year"},
    "lessons_learned": {"weight": 5, "description": "Post-incident review process"},
    "retainer": {"weight": 5, "description": "External IR retainer in place"},
}

BACKUP_CRITERIA = {
    "strategy_exists": {"weight": 10, "description": "Documented backup strategy"},
    "offsite_or_cloud": {"weight": 5, "description": "Offsite or cloud backups"},
    "encrypted": {"weight": 5, "description": "Backup encryption at rest"},
    "tested": {"weight": 10, "description": "Regular backup restoration tests"},
    "immutable": {"weight": 5, "description": "Immutable/WORM backups for ransomware protection"},
    "rto_defined": {"weight": 3, "description": "Recovery Time Objective defined"},
    "rpo_defined": {"weight": 3, "description": "Recovery Point Objective defined"},
}

COMMUNICATION_CRITERIA = {
    "plan_exists": {"weight": 5, "description": "Communication plan documented"},
    "stakeholder_list": {"weight": 3, "description": "Stakeholder contact list maintained"},
    "notification_templates": {"weight": 3, "description": "Pre-written notification templates"},
    "regulatory_process": {"weight": 5, "description": "Regulatory notification process"},
    "media_response": {"weight": 3, "description": "Media/PR response plan"},
}

IR_SYSTEM_PROMPT = """\
You are a cybersecurity incident response readiness assessor. Given the \
questionnaire responses below, evaluate the organization's preparedness \
for cybersecurity incidents.

Produce a JSON object with:
{{
    "ir_plan_findings": [
        {{"finding": "...", "severity": "critical|high|medium|low", \
"requirement": "...", "recommendation": "..."}}
    ],
    "backup_findings": [...],
    "communication_findings": [...],
    "critical_gaps": ["..."],
    "summary": "Brief executive summary"
}}

Company: {company_name}
Industry: {industry}
Employee Count: {employee_count}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("incident_readiness")
class IncidentReadinessAgent(BaseAgent):
    """
    Incident response readiness assessment agent for Enclave Guard.

    Nodes:
        1. assess_ir_plan       -- Evaluate IR plan completeness and testing
        2. evaluate_backups     -- Check backup strategy, testing, encryption
        3. check_communication  -- Assess communication and notification plans
        4. human_review         -- Gate: approve findings before scoring
        5. score_readiness      -- Compute readiness score (0-100) and grade (A-F)
        6. report               -- Generate readiness assessment report
    """

    def build_graph(self) -> Any:
        """Build the Incident Readiness Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(IncidentReadinessAgentState)

        workflow.add_node("assess_ir_plan", self._node_assess_ir_plan)
        workflow.add_node("evaluate_backups", self._node_evaluate_backups)
        workflow.add_node("check_communication", self._node_check_communication)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("score_readiness", self._node_score_readiness)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("assess_ir_plan")

        workflow.add_edge("assess_ir_plan", "evaluate_backups")
        workflow.add_edge("evaluate_backups", "check_communication")
        workflow.add_edge("check_communication", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "score_readiness",
                "rejected": "report",
            },
        )
        workflow.add_edge("score_readiness", "report")
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
    def get_state_class(cls) -> Type[IncidentReadinessAgentState]:
        return IncidentReadinessAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "ir_plan_exists": False,
            "ir_plan_findings": [],
            "ir_plan_last_tested": None,
            "ir_plan_owner": "",
            "ir_roles_defined": False,
            "escalation_procedures": False,
            "runbook_count": 0,
            "backup_findings": [],
            "backup_strategy": "none",
            "backup_frequency": "none",
            "backup_tested": False,
            "backup_encrypted": False,
            "rto_hours": None,
            "rpo_hours": None,
            "immutable_backups": False,
            "communication_findings": [],
            "communication_plan_exists": False,
            "stakeholder_list_defined": False,
            "notification_templates": False,
            "regulatory_notification_process": False,
            "media_response_plan": False,
            "readiness_score": 0.0,
            "readiness_grade": "F",
            "all_findings": [],
            "critical_gaps": [],
            "findings_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Assess IR Plan ──────────────────────────────────────

    async def _node_assess_ir_plan(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 1: Evaluate incident response plan completeness and testing."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})
        company = task.get("company_name", "Unknown")

        logger.info("ir_assess_plan", extra={"company": company})

        findings: list[dict[str, Any]] = []

        plan_exists = bool(responses.get("ir_plan_exists", False))
        roles_defined = bool(responses.get("ir_roles_defined", False))
        escalation = bool(responses.get("escalation_procedures", False))
        runbook_count = int(responses.get("runbook_count", 0))
        plan_owner = responses.get("ir_plan_owner", "")
        last_tested = responses.get("ir_plan_last_tested", None)

        if not plan_exists:
            findings.append({
                "finding": "No documented incident response plan exists",
                "severity": "critical",
                "requirement": "plan_exists",
                "recommendation": "Create a comprehensive IR plan covering "
                                  "detection, containment, eradication, recovery, "
                                  "and lessons learned phases.",
            })
        else:
            if not roles_defined:
                findings.append({
                    "finding": "IR team roles and responsibilities not defined",
                    "severity": "high",
                    "requirement": "roles_defined",
                    "recommendation": "Define IR Commander, Communications Lead, "
                                      "Technical Lead, and Legal Liaison roles.",
                })

            if not escalation:
                findings.append({
                    "finding": "No escalation procedures documented",
                    "severity": "high",
                    "requirement": "escalation_procedures",
                    "recommendation": "Document severity-based escalation paths "
                                      "with clear thresholds and contacts.",
                })

            if runbook_count == 0:
                findings.append({
                    "finding": "No incident-specific runbooks available",
                    "severity": "medium",
                    "requirement": "runbooks",
                    "recommendation": "Create runbooks for top incident types: "
                                      "ransomware, data breach, DDoS, insider threat.",
                })

            # Test recency check
            if last_tested:
                try:
                    tested_dt = datetime.fromisoformat(
                        last_tested.replace("Z", "+00:00")
                    )
                    days_since = (datetime.now(timezone.utc) - tested_dt).days
                    if days_since > 365:
                        findings.append({
                            "finding": f"IR plan last tested {days_since} days ago "
                                       f"(over 1 year)",
                            "severity": "high",
                            "requirement": "tested_annually",
                            "recommendation": "Conduct tabletop exercise or full "
                                              "simulation within the next quarter.",
                        })
                except (ValueError, TypeError):
                    pass
            elif plan_exists:
                findings.append({
                    "finding": "IR plan has never been tested",
                    "severity": "critical",
                    "requirement": "tested_annually",
                    "recommendation": "Schedule an immediate tabletop exercise "
                                      "to validate the IR plan.",
                })

        return {
            "current_node": "assess_ir_plan",
            "ir_plan_exists": plan_exists,
            "ir_plan_findings": findings,
            "ir_plan_last_tested": last_tested,
            "ir_plan_owner": plan_owner,
            "ir_roles_defined": roles_defined,
            "escalation_procedures": escalation,
            "runbook_count": runbook_count,
        }

    # ─── Node 2: Evaluate Backups ────────────────────────────────────

    async def _node_evaluate_backups(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 2: Evaluate backup strategy, testing, and recovery readiness."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})

        logger.info("ir_evaluate_backups", extra={"agent_id": self.agent_id})

        findings: list[dict[str, Any]] = []

        strategy = responses.get("backup_strategy", "none")
        frequency = responses.get("backup_frequency", "none")
        tested = bool(responses.get("backup_tested", False))
        encrypted = bool(responses.get("backup_encrypted", False))
        immutable = bool(responses.get("immutable_backups", False))
        rto = responses.get("rto_hours")
        rpo = responses.get("rpo_hours")

        if strategy == "none":
            findings.append({
                "finding": "No backup strategy in place",
                "severity": "critical",
                "requirement": "strategy_exists",
                "recommendation": "Implement 3-2-1 backup strategy: 3 copies, "
                                  "2 different media, 1 offsite.",
            })
        elif strategy == "local":
            findings.append({
                "finding": "Backups are local-only (no offsite/cloud)",
                "severity": "high",
                "requirement": "offsite_or_cloud",
                "recommendation": "Add offsite or cloud backup tier. Local-only "
                                  "backups are vulnerable to site disasters.",
            })

        if not tested:
            findings.append({
                "finding": "Backup restoration has not been tested",
                "severity": "critical",
                "requirement": "tested",
                "recommendation": "Conduct quarterly backup restoration tests. "
                                  "Untested backups may fail when needed most.",
            })

        if not encrypted:
            findings.append({
                "finding": "Backups are not encrypted at rest",
                "severity": "high",
                "requirement": "encrypted",
                "recommendation": "Enable AES-256 encryption for all backup data "
                                  "at rest and in transit.",
            })

        if not immutable:
            findings.append({
                "finding": "No immutable/WORM backups configured",
                "severity": "high",
                "requirement": "immutable",
                "recommendation": "Deploy immutable backups to protect against "
                                  "ransomware that targets backup systems.",
            })

        if rto is None:
            findings.append({
                "finding": "Recovery Time Objective (RTO) not defined",
                "severity": "medium",
                "requirement": "rto_defined",
                "recommendation": "Define RTO for each critical system. This "
                                  "drives backup architecture decisions.",
            })

        if rpo is None:
            findings.append({
                "finding": "Recovery Point Objective (RPO) not defined",
                "severity": "medium",
                "requirement": "rpo_defined",
                "recommendation": "Define RPO per system to determine backup "
                                  "frequency requirements.",
            })

        return {
            "current_node": "evaluate_backups",
            "backup_findings": findings,
            "backup_strategy": strategy,
            "backup_frequency": frequency,
            "backup_tested": tested,
            "backup_encrypted": encrypted,
            "rto_hours": int(rto) if rto is not None else None,
            "rpo_hours": int(rpo) if rpo is not None else None,
            "immutable_backups": immutable,
        }

    # ─── Node 3: Check Communication ────────────────────────────────

    async def _node_check_communication(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 3: Assess incident communication and notification plans."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})

        logger.info("ir_check_communication", extra={"agent_id": self.agent_id})

        findings: list[dict[str, Any]] = []

        comm_plan = bool(responses.get("communication_plan_exists", False))
        stakeholders = bool(responses.get("stakeholder_list_defined", False))
        templates = bool(responses.get("notification_templates", False))
        regulatory = bool(responses.get("regulatory_notification_process", False))
        media = bool(responses.get("media_response_plan", False))

        if not comm_plan:
            findings.append({
                "finding": "No incident communication plan exists",
                "severity": "high",
                "requirement": "plan_exists",
                "recommendation": "Create communication plan covering internal "
                                  "stakeholders, customers, regulators, and media.",
            })

        if not stakeholders:
            findings.append({
                "finding": "No stakeholder contact list maintained",
                "severity": "medium",
                "requirement": "stakeholder_list",
                "recommendation": "Maintain an up-to-date contact list for all "
                                  "incident stakeholders (exec, legal, PR, IT).",
            })

        if not regulatory:
            findings.append({
                "finding": "No regulatory notification process documented",
                "severity": "high",
                "requirement": "regulatory_process",
                "recommendation": "Document breach notification requirements per "
                                  "jurisdiction (GDPR 72hr, state laws, etc.).",
            })

        if not templates:
            findings.append({
                "finding": "No pre-written notification templates available",
                "severity": "low",
                "requirement": "notification_templates",
                "recommendation": "Prepare template communications for common "
                                  "incident scenarios to reduce response time.",
            })

        return {
            "current_node": "check_communication",
            "communication_findings": findings,
            "communication_plan_exists": comm_plan,
            "stakeholder_list_defined": stakeholders,
            "notification_templates": templates,
            "regulatory_notification_process": regulatory,
            "media_response_plan": media,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 4: Present all findings for human approval before scoring."""
        ir_findings = state.get("ir_plan_findings", [])
        backup_findings = state.get("backup_findings", [])
        comm_findings = state.get("communication_findings", [])
        total = len(ir_findings) + len(backup_findings) + len(comm_findings)

        logger.info(
            "ir_human_review_pending",
            extra={"total_findings": total},
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Score Readiness ─────────────────────────────────────

    async def _node_score_readiness(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 5: Compute readiness score (0-100) and letter grade (A-F)."""
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")

        logger.info("ir_score_readiness", extra={"company": company})

        # Collect all findings
        all_findings = (
            state.get("ir_plan_findings", [])
            + state.get("backup_findings", [])
            + state.get("communication_findings", [])
        )

        # Start at 100, subtract for each finding based on severity
        score = 100.0
        severity_penalties = {"critical": 20, "high": 12, "medium": 6, "low": 2}

        for finding in all_findings:
            sev = finding.get("severity", "low")
            score -= severity_penalties.get(sev, 2)

        score = max(0.0, min(100.0, score))

        # Determine grade
        grade = "F"
        for g, cfg in READINESS_GRADES.items():
            if score >= cfg["min_score"]:
                grade = g
                break

        # Identify critical gaps
        critical_gaps = [
            f.get("finding", "")
            for f in all_findings
            if f.get("severity") == "critical"
        ]

        # Write insight to shared brain
        self.store_insight(InsightData(
            insight_type="incident_readiness",
            title=f"IR Readiness: {company} — Grade {grade} ({score:.0f}/100)",
            content=(
                f"Incident readiness assessment for {company}: "
                f"score {score:.0f}/100 (grade {grade}). "
                f"{len(all_findings)} findings, "
                f"{len(critical_gaps)} critical gaps. "
                f"IR plan: {'exists' if state.get('ir_plan_exists') else 'missing'}. "
                f"Backups: {state.get('backup_strategy', 'none')}."
            ),
            confidence=0.85,
            metadata={
                "company": company,
                "score": score,
                "grade": grade,
                "finding_count": len(all_findings),
                "critical_gaps": len(critical_gaps),
            },
        ))

        return {
            "current_node": "score_readiness",
            "readiness_score": round(score, 1),
            "readiness_grade": grade,
            "all_findings": all_findings,
            "critical_gaps": critical_gaps,
            "findings_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: IncidentReadinessAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate incident readiness assessment report."""
        now = datetime.now(timezone.utc).isoformat()
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")
        findings = state.get("all_findings", [])
        grade = state.get("readiness_grade", "F")
        grade_info = READINESS_GRADES.get(grade, READINESS_GRADES["F"])

        sections = [
            "# Incident Readiness Assessment Report",
            f"*Generated: {now}*\n",
            f"## Organization: {company}",
            f"\n## Readiness Score",
            f"- **Score:** {state.get('readiness_score', 0):.0f}/100",
            f"- **Grade:** {grade} — {grade_info['label']}",
            f"- **Assessment:** {grade_info['description']}",
            f"- **Total Findings:** {len(findings)}",
            f"\n## IR Plan Status",
            f"- **Plan Exists:** {'Yes' if state.get('ir_plan_exists') else 'No'}",
            f"- **Roles Defined:** {'Yes' if state.get('ir_roles_defined') else 'No'}",
            f"- **Escalation Procedures:** {'Yes' if state.get('escalation_procedures') else 'No'}",
            f"- **Runbooks:** {state.get('runbook_count', 0)}",
            f"- **Last Tested:** {state.get('ir_plan_last_tested') or 'Never'}",
            f"\n## Backup Status",
            f"- **Strategy:** {state.get('backup_strategy', 'none')}",
            f"- **Frequency:** {state.get('backup_frequency', 'none')}",
            f"- **Tested:** {'Yes' if state.get('backup_tested') else 'No'}",
            f"- **Encrypted:** {'Yes' if state.get('backup_encrypted') else 'No'}",
            f"- **Immutable:** {'Yes' if state.get('immutable_backups') else 'No'}",
            f"- **RTO:** {state.get('rto_hours', 'N/A')} hours",
            f"- **RPO:** {state.get('rpo_hours', 'N/A')} hours",
            f"\n## Communication Plan",
            f"- **Plan Exists:** {'Yes' if state.get('communication_plan_exists') else 'No'}",
            f"- **Stakeholder List:** {'Yes' if state.get('stakeholder_list_defined') else 'No'}",
            f"- **Regulatory Process:** {'Yes' if state.get('regulatory_notification_process') else 'No'}",
        ]

        critical_gaps = state.get("critical_gaps", [])
        if critical_gaps:
            sections.append("\n## Critical Gaps (Immediate Action Required)")
            for i, gap in enumerate(critical_gaps, 1):
                sections.append(f"  {i}. {gap}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: IncidentReadinessAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<IncidentReadinessAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
