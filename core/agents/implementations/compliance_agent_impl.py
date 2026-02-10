"""
Compliance Agent — The Regulator.

Audits contact records for consent compliance across GDPR, CCPA,
CAN-SPAM, HIPAA, SOX, and PCI-DSS regulations. Checks data retention
periods, generates regulation-specific compliance reports with scores,
and flags violations for remediation. Works across all verticals.

Architecture (LangGraph State Machine):
    audit_consent → check_retention → generate_compliance_report →
    human_review → report → END

Trigger Events:
    - scheduled: Monthly compliance audit sweep
    - regulation_change: New regulation or policy update
    - manual: On-demand compliance audit

Shared Brain Integration:
    - Reads: consent records, data processing agreements, retention policies
    - Writes: compliance posture trends, common violation patterns

Safety:
    - NEVER deletes or modifies data without human approval
    - Retention actions are flagged, not executed automatically
    - Compliance scores are advisory, not legal determinations
    - Human review required before any remediation actions

Usage:
    agent = ComplianceAgentImpl(config, db, embedder, llm)
    result = await agent.run({
        "regulations": ["gdpr", "ccpa", "can_spam"],
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
from core.agents.state import ComplianceAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SUPPORTED_REGULATIONS = [
    "gdpr",
    "ccpa",
    "can_spam",
    "hipaa",
    "sox",
    "pci_dss",
]

CONSENT_TYPES = [
    "marketing_email",
    "data_processing",
    "analytics",
    "third_party_sharing",
]

RETENTION_DEFAULTS = {
    "gdpr": 365 * 3,       # 3 years
    "ccpa": 365 * 2,       # 2 years
    "hipaa": 365 * 6,      # 6 years
    "sox": 365 * 7,        # 7 years
    "pci_dss": 365 * 1,    # 1 year
    "can_spam": 365 * 3,   # 3 years
}

REGULATION_REQUIREMENTS = {
    "gdpr": {
        "label": "GDPR (EU General Data Protection Regulation)",
        "consent_required": ["data_processing", "marketing_email", "analytics"],
        "right_to_erasure": True,
        "breach_notification_hours": 72,
        "dpo_required": True,
    },
    "ccpa": {
        "label": "CCPA (California Consumer Privacy Act)",
        "consent_required": ["data_processing", "third_party_sharing"],
        "right_to_erasure": True,
        "opt_out_required": True,
        "sale_disclosure": True,
    },
    "can_spam": {
        "label": "CAN-SPAM Act",
        "consent_required": ["marketing_email"],
        "unsubscribe_required": True,
        "physical_address_required": True,
        "opt_out_processing_days": 10,
    },
    "hipaa": {
        "label": "HIPAA (Health Insurance Portability and Accountability Act)",
        "consent_required": ["data_processing"],
        "phi_protection": True,
        "minimum_necessary": True,
        "breach_notification_days": 60,
    },
    "sox": {
        "label": "SOX (Sarbanes-Oxley Act)",
        "consent_required": [],
        "financial_record_retention": True,
        "audit_trail_required": True,
    },
    "pci_dss": {
        "label": "PCI DSS (Payment Card Industry Data Security Standard)",
        "consent_required": ["data_processing"],
        "cardholder_data_protection": True,
        "access_control_required": True,
    },
}

COMPLIANCE_AUDIT_PROMPT = """\
You are a regulatory compliance auditor. Analyze the consent records \
and data handling practices below against active regulations.

Active Regulations: {regulations}

Consent Records Summary:
{consent_summary_json}

Missing Consent:
{missing_consent_json}

Retention Violations:
{retention_json}

For each regulation, return a JSON object:
{{
    "findings": [
        {{
            "regulation": "regulation_code",
            "finding_type": "missing_consent|expired_consent|retention_violation|policy_gap",
            "severity": "critical|high|medium|low",
            "description": "Detailed finding description",
            "affected_records": 0,
            "recommended_action": "Specific remediation step",
            "compliance_risk": "Description of the compliance risk"
        }}
    ],
    "regulation_scores": {{
        "regulation_code": 0-100
    }},
    "overall_score": 0-100,
    "executive_summary": "2-3 paragraph compliance posture summary"
}}

Be thorough but fair. Only flag genuine compliance gaps. \
Consider industry standards and best practices.

Return ONLY the JSON object, no markdown code fences.
"""

COMPLIANCE_REPORT_PROMPT = """\
Generate a compliance status report based on the audit findings below.

Findings:
{findings_json}

Regulation Scores:
{scores_json}

Overall Score: {overall_score}/100

Generate a professional Markdown compliance report with:
1. Executive Summary
2. Regulation-by-regulation breakdown
3. Critical findings requiring immediate action
4. Recommended remediation timeline
5. Overall compliance posture assessment

Guidelines:
- Be specific about risks and remediation steps
- Prioritize critical and high-severity findings
- Include estimated effort for each remediation
- Note any regulatory deadlines or windows

Generate the Markdown report directly, no code fences.
"""


@register_agent_type("compliance")
class ComplianceAgentImpl(BaseAgent):
    """
    Regulatory compliance auditing and reporting agent.

    Nodes:
        1. audit_consent              -- Scan contacts for missing/expired consent
        2. check_retention            -- Identify data past retention periods
        3. generate_compliance_report -- LLM produces regulation-specific report
        4. human_review               -- Gate: approve compliance findings
        5. report                     -- Save to compliance_records + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Compliance Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ComplianceAgentState)

        workflow.add_node("audit_consent", self._node_audit_consent)
        workflow.add_node("check_retention", self._node_check_retention)
        workflow.add_node("generate_compliance_report", self._node_generate_compliance_report)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("audit_consent")

        workflow.add_edge("audit_consent", "check_retention")
        workflow.add_edge("check_retention", "generate_compliance_report")
        workflow.add_edge("generate_compliance_report", "human_review")
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
    def get_state_class(cls) -> Type[ComplianceAgentState]:
        return ComplianceAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "active_regulations": [],
            "regulation_requirements": [],
            "consent_records": [],
            "missing_consent": [],
            "total_records_audited": 0,
            "consent_gaps": 0,
            "expiring_records": [],
            "retention_actions": [],
            "compliance_score": 0.0,
            "findings": [],
            "findings_count": 0,
            "actions_approved": False,
            "records_saved": False,
            "report_document": "",
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Audit Consent ───────────────────────────────────────

    async def _node_audit_consent(
        self, state: ComplianceAgentState
    ) -> dict[str, Any]:
        """Node 1: Scan contacts for missing or expired consent records."""
        task = state.get("task_input", {})
        regulations = task.get(
            "regulations",
            self.config.params.get("regulations", ["gdpr", "can_spam"]),
        )

        # Filter to supported regulations only
        active_regulations = [
            r for r in regulations if r in SUPPORTED_REGULATIONS
        ]

        logger.info(
            "compliance_audit_consent",
            extra={
                "agent_id": self.agent_id,
                "regulations": active_regulations,
            },
        )

        consent_records: list[dict[str, Any]] = []
        missing_consent: list[dict[str, Any]] = []
        total_audited = 0
        consent_gaps = 0

        # Build required consent types from active regulations
        required_consents: set[str] = set()
        regulation_reqs: list[dict[str, Any]] = []
        for reg in active_regulations:
            req = REGULATION_REQUIREMENTS.get(reg, {})
            regulation_reqs.append({
                "regulation": reg,
                "label": req.get("label", reg.upper()),
                "required_consents": req.get("consent_required", []),
            })
            required_consents.update(req.get("consent_required", []))

        # Query contacts and check consent records
        try:
            contacts_result = (
                self.db.client.table("contacts")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .limit(500)
                .execute()
            )

            if contacts_result.data:
                total_audited = len(contacts_result.data)

                # Batch fetch consent records
                contact_ids = [
                    c.get("id", "") for c in contacts_result.data if c.get("id")
                ]

                existing_consents: dict[str, list[str]] = {}
                if contact_ids:
                    try:
                        consent_result = (
                            self.db.client.table("consent_records")
                            .select("*")
                            .eq("vertical_id", self.vertical_id)
                            .in_("contact_id", contact_ids[:200])
                            .execute()
                        )

                        if consent_result.data:
                            for cr in consent_result.data:
                                cid = cr.get("contact_id", "")
                                ctype = cr.get("consent_type", "")
                                status = cr.get("status", "")
                                expires = cr.get("expires_at", "")

                                if cid not in existing_consents:
                                    existing_consents[cid] = []

                                # Check if consent is still valid
                                if status == "granted":
                                    if expires:
                                        try:
                                            exp_dt = datetime.fromisoformat(
                                                expires.replace("Z", "+00:00")
                                            )
                                            if exp_dt < datetime.now(timezone.utc):
                                                consent_records.append({
                                                    "contact_id": cid,
                                                    "consent_type": ctype,
                                                    "status": "expired",
                                                    "expires_at": expires,
                                                })
                                                continue
                                        except (ValueError, TypeError):
                                            pass

                                    existing_consents[cid].append(ctype)
                                    consent_records.append({
                                        "contact_id": cid,
                                        "consent_type": ctype,
                                        "status": "active",
                                        "expires_at": expires,
                                    })

                    except Exception as e:
                        logger.warning(
                            "compliance_consent_query_error",
                            extra={"error": str(e)[:200]},
                        )

                # Check each contact for missing consents
                for contact in contacts_result.data:
                    cid = contact.get("id", "")
                    contact_consents = set(existing_consents.get(cid, []))

                    for consent_type in required_consents:
                        if consent_type not in contact_consents:
                            missing_consent.append({
                                "contact_id": cid,
                                "contact_name": contact.get("name", ""),
                                "contact_email": contact.get("email", ""),
                                "consent_type": consent_type,
                                "regulations": [
                                    r for r in active_regulations
                                    if consent_type in
                                    REGULATION_REQUIREMENTS.get(r, {}).get(
                                        "consent_required", []
                                    )
                                ],
                            })
                            consent_gaps += 1

        except Exception as e:
            logger.warning(
                "compliance_contacts_query_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "compliance_consent_audit_complete",
            extra={
                "total_audited": total_audited,
                "consent_records": len(consent_records),
                "missing_consent": len(missing_consent),
                "consent_gaps": consent_gaps,
            },
        )

        return {
            "current_node": "audit_consent",
            "active_regulations": active_regulations,
            "regulation_requirements": regulation_reqs,
            "consent_records": consent_records,
            "missing_consent": missing_consent,
            "total_records_audited": total_audited,
            "consent_gaps": consent_gaps,
        }

    # ─── Node 2: Check Retention ─────────────────────────────────────

    async def _node_check_retention(
        self, state: ComplianceAgentState
    ) -> dict[str, Any]:
        """Node 2: Identify data past retention period, flag for action."""
        active_regulations = state.get("active_regulations", [])
        now = datetime.now(timezone.utc)

        logger.info(
            "compliance_check_retention",
            extra={"regulations": active_regulations},
        )

        expiring_records: list[dict[str, Any]] = []
        retention_actions: list[dict[str, Any]] = []

        # Determine the strictest retention period
        min_retention_days = min(
            (RETENTION_DEFAULTS.get(r, 365 * 10) for r in active_regulations),
            default=365 * 3,
        )

        retention_cutoff = (
            now - timedelta(days=min_retention_days)
        ).isoformat()

        # Check contacts table for old records
        tables_to_check = ["contacts", "companies"]

        for table_name in tables_to_check:
            try:
                result = (
                    self.db.client.table(table_name)
                    .select("id, created_at, updated_at, name, email")
                    .eq("vertical_id", self.vertical_id)
                    .lt("created_at", retention_cutoff)
                    .limit(200)
                    .execute()
                )

                if result.data:
                    for record in result.data:
                        record_id = record.get("id", "")
                        created = record.get("created_at", "")

                        # Determine which regulations this violates
                        age_days = 0
                        if created:
                            try:
                                created_dt = datetime.fromisoformat(
                                    created.replace("Z", "+00:00")
                                )
                                age_days = (now - created_dt).days
                            except (ValueError, TypeError):
                                age_days = min_retention_days + 1

                        violated_regulations = []
                        for reg in active_regulations:
                            max_days = RETENTION_DEFAULTS.get(reg, 365 * 10)
                            if age_days > max_days:
                                violated_regulations.append(reg)

                        if violated_regulations:
                            expiring_records.append({
                                "record_id": record_id,
                                "table": table_name,
                                "created_at": created,
                                "age_days": age_days,
                                "regulations_violated": violated_regulations,
                            })

                            # Determine appropriate action
                            action = "archive" if age_days < min_retention_days * 1.5 else "delete"
                            retention_actions.append({
                                "record_id": record_id,
                                "table": table_name,
                                "action": action,
                                "reason": (
                                    f"Record is {age_days} days old, exceeds "
                                    f"retention for: {', '.join(violated_regulations)}"
                                ),
                                "regulations": violated_regulations,
                            })

            except Exception as e:
                logger.warning(
                    "compliance_retention_check_error",
                    extra={
                        "table": table_name,
                        "error": str(e)[:200],
                    },
                )

        logger.info(
            "compliance_retention_check_complete",
            extra={
                "expiring_records": len(expiring_records),
                "retention_actions": len(retention_actions),
            },
        )

        return {
            "current_node": "check_retention",
            "expiring_records": expiring_records,
            "retention_actions": retention_actions,
        }

    # ─── Node 3: Generate Compliance Report ──────────────────────────

    async def _node_generate_compliance_report(
        self, state: ComplianceAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM produces regulation-specific compliance report."""
        active_regs = state.get("active_regulations", [])
        consent_records = state.get("consent_records", [])
        missing_consent = state.get("missing_consent", [])
        expiring_records = state.get("expiring_records", [])
        total_audited = state.get("total_records_audited", 0)

        logger.info(
            "compliance_generate_report",
            extra={"regulations": active_regs},
        )

        findings: list[dict[str, Any]] = []
        regulation_scores: dict[str, float] = {}
        compliance_score = 100.0
        report_document = ""

        # Build consent summary
        consent_summary = {
            "total_contacts_audited": total_audited,
            "active_consents": len([c for c in consent_records if c.get("status") == "active"]),
            "expired_consents": len([c for c in consent_records if c.get("status") == "expired"]),
            "missing_consents": len(missing_consent),
        }

        try:
            audit_prompt = COMPLIANCE_AUDIT_PROMPT.format(
                regulations=", ".join(active_regs),
                consent_summary_json=json.dumps(consent_summary, indent=2),
                missing_consent_json=json.dumps(missing_consent[:20], indent=2),
                retention_json=json.dumps(expiring_records[:20], indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a regulatory compliance auditor specializing in data privacy.",
                messages=[{"role": "user", "content": audit_prompt}],
                max_tokens=4000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                audit_result = json.loads(llm_text)
                findings = audit_result.get("findings", [])
                regulation_scores = audit_result.get("regulation_scores", {})
                compliance_score = audit_result.get("overall_score", 0.0)
            except (json.JSONDecodeError, KeyError):
                logger.debug("compliance_audit_parse_error")
                # Fallback: calculate basic score
                if total_audited > 0:
                    consent_rate = 1.0 - (
                        len(missing_consent) / max(total_audited * len(CONSENT_TYPES), 1)
                    )
                    compliance_score = max(0, min(100, consent_rate * 100))
                    for reg in active_regs:
                        regulation_scores[reg] = compliance_score

        except Exception as e:
            logger.warning(
                "compliance_audit_llm_error",
                extra={"error": str(e)[:200]},
            )

        # Generate the formatted report
        try:
            report_prompt = COMPLIANCE_REPORT_PROMPT.format(
                findings_json=json.dumps(findings[:15], indent=2),
                scores_json=json.dumps(regulation_scores, indent=2),
                overall_score=compliance_score,
            )

            report_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a compliance reporting specialist.",
                messages=[{"role": "user", "content": report_prompt}],
                max_tokens=3000,
            )

            report_document = report_response.content[0].text.strip()

        except Exception as e:
            logger.warning(
                "compliance_report_llm_error",
                extra={"error": str(e)[:200]},
            )
            # Fallback report
            report_document = (
                f"# Compliance Report\n\n"
                f"**Overall Score:** {compliance_score:.0f}/100\n\n"
                f"**Regulations Audited:** {', '.join(active_regs)}\n\n"
                f"**Records Audited:** {total_audited}\n\n"
                f"**Findings:** {len(findings)}\n\n"
                f"**Missing Consent Records:** {len(missing_consent)}\n\n"
                f"**Retention Violations:** {len(expiring_records)}\n"
            )

        logger.info(
            "compliance_report_generated",
            extra={
                "compliance_score": compliance_score,
                "findings": len(findings),
            },
        )

        return {
            "current_node": "generate_compliance_report",
            "compliance_score": compliance_score,
            "findings": findings,
            "findings_count": len(findings),
            "report_document": report_document,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ComplianceAgentState
    ) -> dict[str, Any]:
        """Node 4: Present compliance findings for human review."""
        compliance_score = state.get("compliance_score", 0.0)
        findings = state.get("findings", [])

        logger.info(
            "compliance_human_review_pending",
            extra={
                "compliance_score": compliance_score,
                "findings_count": len(findings),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ComplianceAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to compliance_records table + InsightData on posture."""
        now = datetime.now(timezone.utc).isoformat()
        active_regs = state.get("active_regulations", [])
        findings = state.get("findings", [])
        compliance_score = state.get("compliance_score", 0.0)
        missing_consent = state.get("missing_consent", [])
        expiring_records = state.get("expiring_records", [])
        retention_actions = state.get("retention_actions", [])
        total_audited = state.get("total_records_audited", 0)
        approved = state.get("human_approval_status") == "approved"

        # Save compliance records
        records_saved = False
        try:
            compliance_record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "active_regulations": json.dumps(active_regs),
                "compliance_score": compliance_score,
                "total_contacts_audited": total_audited,
                "missing_consent_count": len(missing_consent),
                "retention_violations": len(expiring_records),
                "findings": json.dumps(findings),
                "findings_count": len(findings),
                "report_document": state.get("report_document", ""),
                "status": "approved" if approved else "draft",
                "created_at": now,
            }

            result = (
                self.db.client.table("compliance_records")
                .insert(compliance_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                records_saved = True
                logger.info(
                    "compliance_record_saved",
                    extra={"id": result.data[0].get("id", "")},
                )
        except Exception as e:
            logger.warning(
                "compliance_save_error",
                extra={"error": str(e)[:200]},
            )

        # Count findings by severity
        severity_counts: dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Build report summary
        sections = [
            "# Compliance Audit Summary",
            f"*Generated: {now}*\n",
            f"## Overall Score: {compliance_score:.0f}/100",
            f"\n## Audit Scope",
            f"- **Regulations:** {', '.join(r.upper() for r in active_regs)}",
            f"- **Records Audited:** {total_audited}",
            f"- **Total Findings:** {len(findings)}",
            f"- **Missing Consent:** {len(missing_consent)}",
            f"- **Retention Violations:** {len(expiring_records)}",
            f"- **Actions Pending:** {len(retention_actions)}",
            f"- **Status:** {'Approved' if approved else 'Draft'}",
        ]

        if severity_counts:
            sections.append("\n## Findings by Severity")
            for sev in ["critical", "high", "medium", "low"]:
                count = severity_counts.get(sev, 0)
                if count > 0:
                    sections.append(f"- **{sev.upper()}:** {count}")

        if findings:
            sections.append("\n## Key Findings")
            for i, f in enumerate(findings[:5], 1):
                sections.append(
                    f"{i}. **[{f.get('severity', 'N/A').upper()}]** "
                    f"({f.get('regulation', 'N/A').upper()}) "
                    f"{f.get('description', 'N/A')[:100]}"
                )

        report = "\n".join(sections)

        # Store insight
        self.store_insight(InsightData(
            insight_type="compliance_posture",
            title=f"Compliance Audit: {compliance_score:.0f}/100 across {len(active_regs)} regulations",
            content=(
                f"Audited {total_audited} records against "
                f"{', '.join(r.upper() for r in active_regs)}. "
                f"Overall compliance score: {compliance_score:.0f}/100. "
                f"Found {len(findings)} findings "
                f"({severity_counts.get('critical', 0)} critical). "
                f"{len(missing_consent)} missing consent records. "
                f"{len(expiring_records)} retention violations."
            ),
            confidence=0.85,
            metadata={
                "compliance_score": compliance_score,
                "regulations": active_regs,
                "total_audited": total_audited,
                "findings_count": len(findings),
                "missing_consent": len(missing_consent),
                "retention_violations": len(expiring_records),
                "severity_distribution": severity_counts,
            },
        ))

        logger.info(
            "compliance_report_summary_generated",
            extra={
                "score": compliance_score,
                "findings": len(findings),
                "saved": records_saved,
            },
        )

        return {
            "current_node": "report",
            "records_saved": records_saved,
            "actions_approved": approved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ComplianceAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ComplianceAgentImpl agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
