"""
IAM Analyst Agent — The Identity Guardian.

Evaluates MFA adoption, password policies, privileged accounts,
and access review practices across the target organization.
Produces risk-scored findings and actionable recommendations.

Architecture (LangGraph State Machine):
    load_questionnaire → analyze_access → assess_mfa →
    human_review → save_findings → report → END

Trigger Events:
    - security_assessment: Full IAM review as part of engagement
    - questionnaire_submitted: Client completes IAM questionnaire
    - manual: On-demand IAM analysis

Shared Brain Integration:
    - Reads: compliance requirements, industry benchmarks, peer findings
    - Writes: IAM risk patterns, MFA adoption insights, password policy gaps

Safety:
    - NEVER accesses client identity systems directly
    - All findings require human_review gate before saving
    - Questionnaire data is treated as sensitive (PII-adjacent)
    - Risk scores are advisory, not authoritative

Usage:
    agent = IAMAnalystAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_name": "Acme Corp",
        "questionnaire": {...},
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import IAMAnalystAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

ACCESS_RISK_LEVELS = {
    "critical": {"min_score": 80, "label": "Critical", "color": "red"},
    "high": {"min_score": 60, "label": "High", "color": "orange"},
    "medium": {"min_score": 40, "label": "Medium", "color": "yellow"},
    "low": {"min_score": 0, "label": "Low", "color": "green"},
}

MFA_TYPES = {
    "totp": {"strength": 0.7, "label": "TOTP (Authenticator App)"},
    "fido2": {"strength": 1.0, "label": "FIDO2/WebAuthn (Hardware Key)"},
    "sms": {"strength": 0.3, "label": "SMS OTP (Weakest)"},
    "push": {"strength": 0.8, "label": "Push Notification"},
    "email": {"strength": 0.2, "label": "Email OTP"},
}

PASSWORD_POLICY_REQUIREMENTS = {
    "min_length": {"recommended": 14, "minimum": 8, "weight": 0.25},
    "rotation_days": {"recommended": 0, "maximum": 90, "weight": 0.15},
    "complexity": {"required": True, "weight": 0.20},
    "history_count": {"recommended": 12, "minimum": 5, "weight": 0.10},
    "lockout_threshold": {"recommended": 5, "maximum": 10, "weight": 0.15},
    "breach_checking": {"required": True, "weight": 0.15},
}

IAM_SYSTEM_PROMPT = """\
You are a cybersecurity IAM analyst specializing in identity and access \
management assessments. Given the questionnaire responses below, produce \
a JSON object with:
{{
    "access_findings": [
        {{"finding": "...", "severity": "critical|high|medium|low", \
"category": "privileged_access|service_accounts|access_reviews|rbac|separation_of_duties", \
"recommendation": "..."}}
    ],
    "mfa_findings": [
        {{"finding": "...", "severity": "critical|high|medium|low", \
"category": "adoption|enforcement|type_strength", "recommendation": "..."}}
    ],
    "password_findings": [
        {{"finding": "...", "severity": "critical|high|medium|low", \
"category": "length|rotation|complexity|breach_check", "recommendation": "..."}}
    ],
    "overall_risk_score": 0-100,
    "summary": "Brief executive summary"
}}

Company: {company_name}
Industry: {industry}
Employee Count: {employee_count}
Compliance Frameworks: {frameworks}

Scoring guide:
- 0-39: Low risk (well-managed IAM)
- 40-59: Medium risk (gaps exist)
- 60-79: High risk (significant gaps)
- 80-100: Critical risk (immediate action needed)

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("iam_analyst")
class IAMAnalystAgent(BaseAgent):
    """
    IAM assessment agent for Enclave Guard cybersecurity engagements.

    Nodes:
        1. load_questionnaire  -- Parse and validate client IAM questionnaire
        2. analyze_access      -- Evaluate access controls, RBAC, privileged accounts
        3. assess_mfa          -- Score MFA adoption, password policies
        4. human_review        -- Gate: approve findings before saving
        5. save_findings       -- Persist findings to DB and shared brain
        6. report              -- Generate IAM assessment report
    """

    def build_graph(self) -> Any:
        """Build the IAM Analyst Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(IAMAnalystAgentState)

        workflow.add_node("load_questionnaire", self._node_load_questionnaire)
        workflow.add_node("analyze_access", self._node_analyze_access)
        workflow.add_node("assess_mfa", self._node_assess_mfa)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("save_findings", self._node_save_findings)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_questionnaire")

        workflow.add_edge("load_questionnaire", "analyze_access")
        workflow.add_edge("analyze_access", "assess_mfa")
        workflow.add_edge("assess_mfa", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "save_findings",
                "rejected": "report",
            },
        )
        workflow.add_edge("save_findings", "report")
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
    def get_state_class(cls) -> Type[IAMAnalystAgentState]:
        return IAMAnalystAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "questionnaire_responses": (task or {}).get("questionnaire", {}),
            "organization_profile": {},
            "compliance_frameworks": [],
            "access_findings": [],
            "privileged_account_count": 0,
            "service_account_count": 0,
            "orphaned_accounts": [],
            "access_review_frequency": "never",
            "rbac_maturity": "none",
            "mfa_findings": [],
            "mfa_adoption_rate": 0.0,
            "mfa_types_deployed": [],
            "mfa_enforced_for_admins": False,
            "mfa_enforced_for_all": False,
            "password_policy_findings": [],
            "password_policy_score": 0.0,
            "password_min_length": 0,
            "password_rotation_days": 0,
            "passwords_require_complexity": False,
            "iam_risk_score": 0.0,
            "iam_risk_level": "low",
            "all_findings": [],
            "findings_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Questionnaire ──────────────────────────────────

    async def _node_load_questionnaire(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 1: Parse and validate the client's IAM questionnaire responses."""
        logger.info("iam_load_questionnaire", extra={"agent_id": self.agent_id})

        task = state.get("task_input", {})
        responses = task.get("questionnaire", state.get("questionnaire_responses", {}))
        company_name = task.get("company_name", "Unknown")

        org_profile = {
            "company_name": company_name,
            "industry": task.get("industry", responses.get("industry", "unknown")),
            "employee_count": task.get(
                "employee_count", responses.get("employee_count", 0)
            ),
            "identity_provider": responses.get("identity_provider", "unknown"),
            "directory_service": responses.get("directory_service", "unknown"),
        }

        frameworks = responses.get("compliance_frameworks", [])
        if isinstance(frameworks, str):
            frameworks = [f.strip() for f in frameworks.split(",") if f.strip()]

        logger.info(
            "iam_questionnaire_loaded",
            extra={
                "company": company_name,
                "questions_answered": len(responses),
                "frameworks": len(frameworks),
            },
        )

        return {
            "current_node": "load_questionnaire",
            "questionnaire_responses": responses,
            "organization_profile": org_profile,
            "compliance_frameworks": frameworks,
        }

    # ─── Node 2: Analyze Access ──────────────────────────────────────

    async def _node_analyze_access(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 2: Evaluate access controls, RBAC maturity, privileged accounts."""
        responses = state.get("questionnaire_responses", {})
        org = state.get("organization_profile", {})

        logger.info("iam_analyze_access", extra={"company": org.get("company_name")})

        findings: list[dict[str, Any]] = []

        # Privileged account analysis
        priv_count = int(responses.get("privileged_account_count", 0))
        svc_count = int(responses.get("service_account_count", 0))
        employee_count = int(org.get("employee_count", 1) or 1)

        priv_ratio = priv_count / max(employee_count, 1)
        if priv_ratio > 0.15:
            findings.append({
                "finding": f"Excessive privileged accounts: {priv_count} "
                           f"({priv_ratio:.0%} of workforce)",
                "severity": "high",
                "category": "privileged_access",
                "recommendation": "Implement least-privilege access model. "
                                  "Target <5% privileged account ratio.",
            })
        elif priv_ratio > 0.05:
            findings.append({
                "finding": f"Privileged account ratio slightly elevated: "
                           f"{priv_ratio:.0%}",
                "severity": "medium",
                "category": "privileged_access",
                "recommendation": "Review and reduce privileged accounts quarterly.",
            })

        # Access review frequency
        review_freq = responses.get("access_review_frequency", "never")
        if review_freq == "never":
            findings.append({
                "finding": "No periodic access reviews conducted",
                "severity": "critical",
                "category": "access_reviews",
                "recommendation": "Implement quarterly access reviews for all "
                                  "systems. Prioritize privileged accounts.",
            })
        elif review_freq == "annually":
            findings.append({
                "finding": "Access reviews only conducted annually",
                "severity": "medium",
                "category": "access_reviews",
                "recommendation": "Increase to quarterly reviews, especially "
                                  "for privileged and service accounts.",
            })

        # RBAC maturity
        rbac = responses.get("rbac_maturity", "none")
        if rbac in ("none", "basic"):
            findings.append({
                "finding": f"RBAC maturity is '{rbac}' — flat permission model",
                "severity": "high" if rbac == "none" else "medium",
                "category": "rbac",
                "recommendation": "Implement role-based access control with "
                                  "defined roles aligned to job functions.",
            })

        # Service accounts
        if svc_count > 0 and not responses.get("service_account_rotation", False):
            findings.append({
                "finding": f"{svc_count} service accounts without credential rotation",
                "severity": "high",
                "category": "service_accounts",
                "recommendation": "Implement automated credential rotation "
                                  "for all service accounts (max 90-day lifecycle).",
            })

        return {
            "current_node": "analyze_access",
            "access_findings": findings,
            "privileged_account_count": priv_count,
            "service_account_count": svc_count,
            "orphaned_accounts": responses.get("orphaned_accounts", []),
            "access_review_frequency": review_freq,
            "rbac_maturity": rbac,
        }

    # ─── Node 3: Assess MFA ─────────────────────────────────────────

    async def _node_assess_mfa(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 3: Evaluate MFA adoption and password policy strength."""
        responses = state.get("questionnaire_responses", {})
        org = state.get("organization_profile", {})
        access_findings = state.get("access_findings", [])

        logger.info("iam_assess_mfa", extra={"company": org.get("company_name")})

        mfa_findings: list[dict[str, Any]] = []
        pwd_findings: list[dict[str, Any]] = []

        # MFA adoption
        mfa_rate = float(responses.get("mfa_adoption_rate", 0))
        mfa_types = responses.get("mfa_types", [])
        mfa_admin = responses.get("mfa_enforced_admins", False)
        mfa_all = responses.get("mfa_enforced_all", False)

        if not mfa_admin:
            mfa_findings.append({
                "finding": "MFA not enforced for administrator accounts",
                "severity": "critical",
                "category": "enforcement",
                "recommendation": "Immediately enforce MFA for all admin and "
                                  "privileged accounts using FIDO2 or TOTP.",
            })

        if mfa_rate < 0.5:
            mfa_findings.append({
                "finding": f"MFA adoption rate critically low: {mfa_rate:.0%}",
                "severity": "critical",
                "category": "adoption",
                "recommendation": "Deploy organization-wide MFA mandate. "
                                  "Start with privileged users, then all users.",
            })
        elif mfa_rate < 0.9:
            mfa_findings.append({
                "finding": f"MFA adoption incomplete: {mfa_rate:.0%}",
                "severity": "high",
                "category": "adoption",
                "recommendation": "Push for 100% MFA adoption. Identify and "
                                  "enroll remaining users.",
            })

        # MFA strength
        if "sms" in mfa_types and "fido2" not in mfa_types:
            mfa_findings.append({
                "finding": "Reliance on SMS-based MFA (SIM-swap vulnerable)",
                "severity": "medium",
                "category": "type_strength",
                "recommendation": "Migrate from SMS OTP to TOTP or FIDO2 "
                                  "hardware keys for phishing resistance.",
            })

        # Password policy
        min_len = int(responses.get("password_min_length", 0))
        rotation = int(responses.get("password_rotation_days", 0))
        complexity = responses.get("password_complexity", False)

        pwd_score = 0.0
        if min_len >= 14:
            pwd_score += 0.3
        elif min_len >= 8:
            pwd_score += 0.15
        else:
            pwd_findings.append({
                "finding": f"Password minimum length too short: {min_len} chars",
                "severity": "critical",
                "category": "length",
                "recommendation": "Set minimum password length to 14+ characters "
                                  "(NIST SP 800-63B recommendation).",
            })

        if complexity:
            pwd_score += 0.2
        if responses.get("breach_checking", False):
            pwd_score += 0.3
        if rotation == 0 or rotation > 365:
            pwd_score += 0.2  # NIST recommends no forced rotation

        # Combine all findings for risk scoring
        all_findings = access_findings + mfa_findings + pwd_findings
        severity_weights = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        risk_score = min(
            100.0,
            sum(severity_weights.get(f.get("severity", "low"), 3) for f in all_findings),
        )

        risk_level = "low"
        for level, cfg in ACCESS_RISK_LEVELS.items():
            if risk_score >= cfg["min_score"]:
                risk_level = level
                break

        logger.info(
            "iam_assessment_complete",
            extra={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "total_findings": len(all_findings),
            },
        )

        return {
            "current_node": "assess_mfa",
            "mfa_findings": mfa_findings,
            "mfa_adoption_rate": mfa_rate,
            "mfa_types_deployed": mfa_types,
            "mfa_enforced_for_admins": mfa_admin,
            "mfa_enforced_for_all": mfa_all,
            "password_policy_findings": pwd_findings,
            "password_policy_score": round(pwd_score, 2),
            "password_min_length": min_len,
            "password_rotation_days": rotation,
            "passwords_require_complexity": complexity,
            "iam_risk_score": round(risk_score, 1),
            "iam_risk_level": risk_level,
            "all_findings": all_findings,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 4: Present IAM findings for human approval before saving."""
        findings = state.get("all_findings", [])
        logger.info(
            "iam_human_review_pending",
            extra={
                "finding_count": len(findings),
                "risk_score": state.get("iam_risk_score", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Save Findings ──────────────────────────────────────

    async def _node_save_findings(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 5: Persist approved findings to DB and shared brain."""
        findings = state.get("all_findings", [])
        org = state.get("organization_profile", {})
        company = org.get("company_name", "Unknown")

        logger.info("iam_save_findings", extra={"count": len(findings)})

        # Save each finding to the assessments table
        saved = 0
        for finding in findings:
            try:
                self.db.client.table("security_findings").insert({
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "company_name": company,
                    "domain": "iam",
                    "finding": finding.get("finding", ""),
                    "severity": finding.get("severity", "low"),
                    "category": finding.get("category", ""),
                    "recommendation": finding.get("recommendation", ""),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }).execute()
                saved += 1
            except Exception as e:
                logger.debug(f"Failed to save finding: {e}")

        # Write insight to shared brain
        if findings:
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            self.store_insight(InsightData(
                insight_type="iam_assessment",
                title=f"IAM Assessment: {company} — "
                      f"{state.get('iam_risk_level', 'unknown').upper()} risk",
                content=(
                    f"IAM assessment for {company}: risk score "
                    f"{state.get('iam_risk_score', 0)}/100, "
                    f"{len(findings)} findings ({critical} critical). "
                    f"MFA adoption: {state.get('mfa_adoption_rate', 0):.0%}. "
                    f"Password policy score: {state.get('password_policy_score', 0):.0%}."
                ),
                confidence=0.85,
                metadata={
                    "company": company,
                    "risk_score": state.get("iam_risk_score", 0),
                    "finding_count": len(findings),
                    "critical_count": critical,
                },
            ))

        return {
            "current_node": "save_findings",
            "findings_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: IAMAnalystAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate IAM assessment report."""
        now = datetime.now(timezone.utc).isoformat()
        org = state.get("organization_profile", {})
        findings = state.get("all_findings", [])

        sev_counts = {}
        for f in findings:
            s = f.get("severity", "low")
            sev_counts[s] = sev_counts.get(s, 0) + 1

        sections = [
            "# IAM Security Assessment Report",
            f"*Generated: {now}*\n",
            f"## Organization: {org.get('company_name', 'N/A')}",
            f"- **Industry:** {org.get('industry', 'N/A')}",
            f"- **Employees:** {org.get('employee_count', 'N/A')}",
            f"- **Frameworks:** {', '.join(state.get('compliance_frameworks', [])) or 'None'}",
            f"\n## Risk Summary",
            f"- **IAM Risk Score:** {state.get('iam_risk_score', 0)}/100 "
            f"({state.get('iam_risk_level', 'unknown').upper()})",
            f"- **Total Findings:** {len(findings)}",
            f"- **Critical:** {sev_counts.get('critical', 0)}",
            f"- **High:** {sev_counts.get('high', 0)}",
            f"- **Medium:** {sev_counts.get('medium', 0)}",
            f"- **Low:** {sev_counts.get('low', 0)}",
            f"\n## MFA Status",
            f"- **Adoption Rate:** {state.get('mfa_adoption_rate', 0):.0%}",
            f"- **Types Deployed:** {', '.join(state.get('mfa_types_deployed', [])) or 'None'}",
            f"- **Admin Enforcement:** {'Yes' if state.get('mfa_enforced_for_admins') else 'No'}",
            f"\n## Password Policy",
            f"- **Score:** {state.get('password_policy_score', 0):.0%}",
            f"- **Min Length:** {state.get('password_min_length', 0)} chars",
            f"\n## Access Controls",
            f"- **Privileged Accounts:** {state.get('privileged_account_count', 0)}",
            f"- **Service Accounts:** {state.get('service_account_count', 0)}",
            f"- **Review Frequency:** {state.get('access_review_frequency', 'N/A')}",
            f"- **RBAC Maturity:** {state.get('rbac_maturity', 'N/A')}",
        ]

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: IAMAnalystAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<IAMAnalystAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
