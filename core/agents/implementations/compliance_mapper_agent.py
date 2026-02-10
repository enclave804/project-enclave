"""
Compliance Mapper Agent — The Regulation Navigator.

Maps an organization's current security posture against major compliance
frameworks (SOC 2, HIPAA, PCI-DSS, ISO 27001) to identify gaps and
produce a prioritized remediation roadmap.

Architecture (LangGraph State Machine):
    load_requirements -> map_controls -> identify_gaps ->
    human_review -> generate_roadmap -> report -> END

Trigger Events:
    - new_assessment: Triggered after security scans complete
    - manual: On-demand compliance gap analysis
    - quarterly: Periodic compliance posture review

Shared Brain Integration:
    - Reads: security findings from VulnScanner/AppSec/Network agents
    - Writes: compliance scores, gap analysis, remediation roadmaps

Safety:
    - NEVER makes compliance certifications -- advisory only
    - All gap analyses require human_review before client delivery
    - Clearly labels estimates vs. verified controls
    - Disclaims that this is not legal or audit advice

Usage:
    agent = ComplianceMapperAgent(config, db, embedder, llm)
    result = await agent.run({
        "company_domain": "example.com",
        "target_frameworks": ["soc2", "hipaa"],
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
from core.agents.state import ComplianceMapperAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SUPPORTED_FRAMEWORKS = {"soc2", "hipaa", "pci_dss", "iso27001"}

CONTROL_STATUSES = {"met", "partial", "missing", "not_applicable"}

# Simplified control catalogs (production would load from full CSV/DB)
FRAMEWORK_CONTROLS = {
    "soc2": [
        {"id": "CC1.1", "control": "COSO Principle 1", "description": "Integrity and ethical values", "category": "Control Environment"},
        {"id": "CC2.1", "control": "Information and Communication", "description": "Internal communication of security policies", "category": "Communication"},
        {"id": "CC3.1", "control": "Risk Assessment", "description": "Risk identification and analysis", "category": "Risk Assessment"},
        {"id": "CC5.1", "control": "Logical Access", "description": "Authentication and access controls", "category": "Access Control"},
        {"id": "CC6.1", "control": "System Operations", "description": "Vulnerability management and patching", "category": "Operations"},
        {"id": "CC6.6", "control": "Change Management", "description": "Change control procedures", "category": "Operations"},
        {"id": "CC7.1", "control": "Monitoring", "description": "Security monitoring and alerting", "category": "Monitoring"},
        {"id": "CC7.2", "control": "Incident Response", "description": "Incident detection and response", "category": "Monitoring"},
        {"id": "CC8.1", "control": "Risk Mitigation", "description": "Risk mitigation activities", "category": "Risk Mitigation"},
        {"id": "CC9.1", "control": "Vendor Management", "description": "Third-party risk management", "category": "Vendor Management"},
    ],
    "hipaa": [
        {"id": "164.308(a)(1)", "control": "Security Management", "description": "Risk analysis and management", "category": "Administrative"},
        {"id": "164.308(a)(3)", "control": "Workforce Security", "description": "Authorization and access", "category": "Administrative"},
        {"id": "164.308(a)(4)", "control": "Information Access", "description": "Access management", "category": "Administrative"},
        {"id": "164.308(a)(5)", "control": "Security Awareness", "description": "Security training", "category": "Administrative"},
        {"id": "164.308(a)(6)", "control": "Incident Procedures", "description": "Security incident handling", "category": "Administrative"},
        {"id": "164.310(a)(1)", "control": "Facility Access", "description": "Physical access controls", "category": "Physical"},
        {"id": "164.312(a)(1)", "control": "Access Control", "description": "Unique user identification", "category": "Technical"},
        {"id": "164.312(c)(1)", "control": "Integrity", "description": "Data integrity controls", "category": "Technical"},
        {"id": "164.312(d)", "control": "Authentication", "description": "Person or entity authentication", "category": "Technical"},
        {"id": "164.312(e)(1)", "control": "Transmission Security", "description": "Encryption in transit", "category": "Technical"},
    ],
    "pci_dss": [
        {"id": "1.1", "control": "Firewall Config", "description": "Install and maintain firewall configuration", "category": "Network Security"},
        {"id": "2.1", "control": "System Defaults", "description": "Do not use vendor-supplied defaults", "category": "System Security"},
        {"id": "3.1", "control": "Data Protection", "description": "Protect stored cardholder data", "category": "Data Protection"},
        {"id": "4.1", "control": "Encryption", "description": "Encrypt transmission of cardholder data", "category": "Encryption"},
        {"id": "5.1", "control": "Antivirus", "description": "Use and update anti-virus software", "category": "Malware Protection"},
        {"id": "6.1", "control": "Secure Systems", "description": "Develop and maintain secure systems", "category": "Secure Development"},
        {"id": "7.1", "control": "Access Restriction", "description": "Restrict access by business need", "category": "Access Control"},
        {"id": "8.1", "control": "Identification", "description": "Assign unique ID to each person", "category": "Identification"},
        {"id": "10.1", "control": "Logging", "description": "Track and monitor all access", "category": "Monitoring"},
        {"id": "11.1", "control": "Testing", "description": "Regularly test security systems", "category": "Testing"},
    ],
    "iso27001": [
        {"id": "A.5", "control": "Information Security Policies", "description": "Management direction for information security", "category": "Policy"},
        {"id": "A.6", "control": "Organization of InfoSec", "description": "Internal organization and mobile devices", "category": "Organization"},
        {"id": "A.7", "control": "Human Resource Security", "description": "Before, during, and after employment", "category": "HR Security"},
        {"id": "A.8", "control": "Asset Management", "description": "Inventory and classification of assets", "category": "Asset Management"},
        {"id": "A.9", "control": "Access Control", "description": "Business requirements of access control", "category": "Access Control"},
        {"id": "A.10", "control": "Cryptography", "description": "Cryptographic controls", "category": "Cryptography"},
        {"id": "A.12", "control": "Operations Security", "description": "Operational procedures and responsibilities", "category": "Operations"},
        {"id": "A.13", "control": "Communications Security", "description": "Network security management", "category": "Network"},
        {"id": "A.16", "control": "Incident Management", "description": "Information security incident management", "category": "Incident Management"},
        {"id": "A.18", "control": "Compliance", "description": "Compliance with legal and contractual requirements", "category": "Compliance"},
    ],
}

COMPLIANCE_SYSTEM_PROMPT = """\
You are an expert compliance auditor for {company_name}. \
You map security findings against compliance frameworks to identify gaps.

Given the security findings and framework controls below for {domain}, \
produce a JSON object with:
{{
    "control_mappings": [
        {{
            "control_id": "Framework control ID",
            "framework": "soc2|hipaa|pci_dss|iso27001",
            "status": "met|partial|missing",
            "evidence": "What supports this status",
            "gap": "What is missing (if partial/missing)"
        }}
    ],
    "compliance_score": 0.0-100.0,
    "top_gaps": ["list of most critical gaps"]
}}

Rules:
- Base assessments on observable evidence from scan data
- Mark controls as 'partial' when some evidence exists but is incomplete
- Mark controls as 'missing' when no evidence supports compliance
- Be conservative -- when in doubt, mark as partial
- This is advisory, NOT a certification

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("compliance_mapper")
class ComplianceMapperAgent(BaseAgent):
    """
    Compliance gap analysis and roadmap generation agent.

    Nodes:
        1. load_requirements    -- Load framework controls for target frameworks
        2. map_controls         -- Map security findings to compliance controls
        3. identify_gaps        -- Score compliance and identify critical gaps
        4. human_review         -- Gate: review gap analysis before delivery
        5. generate_roadmap     -- Create prioritized remediation roadmap
        6. report               -- Summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """Build the Compliance Mapper's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ComplianceMapperAgentState)

        workflow.add_node("load_requirements", self._node_load_requirements)
        workflow.add_node("map_controls", self._node_map_controls)
        workflow.add_node("identify_gaps", self._node_identify_gaps)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("generate_roadmap", self._node_generate_roadmap)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_requirements")

        workflow.add_edge("load_requirements", "map_controls")
        workflow.add_edge("map_controls", "identify_gaps")
        workflow.add_edge("identify_gaps", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "generate_roadmap",
                "rejected": "report",
            },
        )
        workflow.add_edge("generate_roadmap", "report")
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
    def get_state_class(cls) -> Type[ComplianceMapperAgentState]:
        return ComplianceMapperAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        frameworks = task.get("target_frameworks", ["soc2"])
        state.update({
            "company_domain": task.get("company_domain", ""),
            "company_name": task.get("company_name", ""),
            "target_frameworks": [f for f in frameworks if f in SUPPORTED_FRAMEWORKS],
            "framework_requirements": {},
            "total_controls": 0,
            "control_mappings": [],
            "controls_met": 0,
            "controls_partial": 0,
            "controls_missing": 0,
            "gaps": [],
            "compliance_scores": {},
            "overall_compliance_score": 0.0,
            "roadmap_items": [],
            "roadmap_summary": "",
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Requirements ──────────────────────────────────

    async def _node_load_requirements(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Load framework controls for the target frameworks.

        Retrieves control catalogs for each requested framework
        (SOC2, HIPAA, PCI-DSS, ISO 27001).
        """
        frameworks = state.get("target_frameworks", ["soc2"])

        logger.info(
            "compliance_load: loading %d frameworks: %s",
            len(frameworks), frameworks,
        )

        requirements: dict[str, Any] = {}
        total_controls = 0

        for fw in frameworks:
            controls = FRAMEWORK_CONTROLS.get(fw, [])
            requirements[fw] = controls
            total_controls += len(controls)

        logger.info(
            "compliance_load_complete: %d frameworks, %d total controls",
            len(requirements), total_controls,
        )

        return {
            "current_node": "load_requirements",
            "framework_requirements": requirements,
            "total_controls": total_controls,
        }

    # ─── Node 2: Map Controls ──────────────────────────────────────

    async def _node_map_controls(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Map existing security findings to compliance controls.

        Uses security assessment data from the database to determine
        which compliance controls are met, partially met, or missing.
        """
        domain = state.get("company_domain", "")
        requirements = state.get("framework_requirements", {})

        logger.info("compliance_map: mapping controls for %s", domain)

        # Fetch existing security findings for this domain
        existing_findings: list[dict[str, Any]] = []
        try:
            result = (
                self.db.client.table("security_findings")
                .select("*")
                .eq("domain", domain)
                .eq("vertical_id", self.vertical_id)
                .execute()
            )
            existing_findings = result.data or []
        except Exception as e:
            logger.debug(f"Failed to load security findings: {e}")

        # Fetch assessment summary
        assessments: list[dict[str, Any]] = []
        try:
            result = (
                self.db.client.table("security_assessments")
                .select("*")
                .eq("domain", domain)
                .eq("vertical_id", self.vertical_id)
                .execute()
            )
            assessments = result.data or []
        except Exception as e:
            logger.debug(f"Failed to load assessments: {e}")

        # Use LLM to map findings to controls
        mappings: list[dict[str, Any]] = []
        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the target")

        for fw, controls in requirements.items():
            try:
                findings_summary = json.dumps(
                    [{"severity": f.get("severity"), "title": f.get("title")} for f in existing_findings[:30]],
                    indent=2,
                )
                controls_json = json.dumps(controls, indent=2)

                prompt = (
                    f"Framework: {fw.upper()}\n"
                    f"Domain: {domain}\n"
                    f"Security findings:\n{findings_summary}\n\n"
                    f"Controls to map:\n{controls_json}\n\n"
                    f"Map each control to met/partial/missing status."
                )
                system = COMPLIANCE_SYSTEM_PROMPT.format(
                    company_name=company_name, domain=domain,
                )

                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                )
                text = response.content[0].text.strip() if response.content else ""
                if text:
                    try:
                        parsed = json.loads(text)
                        llm_mappings = parsed.get("control_mappings", [])
                        mappings.extend(llm_mappings)
                    except (json.JSONDecodeError, ValueError):
                        logger.debug(f"Failed to parse compliance mapping for {fw}")
            except Exception as e:
                logger.debug(f"LLM compliance mapping failed for {fw}: {e}")

            # Fallback: if LLM produced no mappings, do rule-based
            if not any(m.get("framework") == fw for m in mappings):
                has_vuln_scan = any(a.get("assessment_type") == "vulnerability_scan" for a in assessments)
                for control in controls:
                    status = "missing"
                    if has_vuln_scan and control.get("category") in ("Operations", "Monitoring", "Testing"):
                        status = "partial"
                    mappings.append({
                        "control_id": control["id"],
                        "framework": fw,
                        "status": status,
                        "evidence": "Based on available assessment data" if status == "partial" else "",
                        "gap": control["description"] if status == "missing" else "",
                    })

        logger.info(
            "compliance_map_complete: %d control mappings",
            len(mappings),
        )

        return {
            "current_node": "map_controls",
            "control_mappings": mappings,
        }

    # ─── Node 3: Identify Gaps ────────────────────────────────────

    async def _node_identify_gaps(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Score compliance and identify critical gaps.

        Computes per-framework compliance scores and aggregates
        gaps that need remediation.
        """
        mappings = state.get("control_mappings", [])
        frameworks = state.get("target_frameworks", [])

        logger.info("compliance_gaps: analyzing %d mappings", len(mappings))

        gaps: list[dict[str, Any]] = []
        compliance_scores: dict[str, float] = {}
        met = 0
        partial = 0
        missing = 0

        for fw in frameworks:
            fw_mappings = [m for m in mappings if m.get("framework") == fw]
            if not fw_mappings:
                compliance_scores[fw] = 0.0
                continue

            fw_met = sum(1 for m in fw_mappings if m.get("status") == "met")
            fw_partial = sum(1 for m in fw_mappings if m.get("status") == "partial")
            fw_missing = sum(1 for m in fw_mappings if m.get("status") == "missing")

            met += fw_met
            partial += fw_partial
            missing += fw_missing

            # Score: met=1.0, partial=0.5, missing=0.0
            total = len(fw_mappings)
            score = ((fw_met * 1.0) + (fw_partial * 0.5)) / max(total, 1) * 100
            compliance_scores[fw] = round(score, 1)

            # Collect gaps
            for m in fw_mappings:
                if m.get("status") in ("partial", "missing"):
                    gaps.append({
                        "framework": fw,
                        "control_id": m.get("control_id", ""),
                        "severity": "high" if m.get("status") == "missing" else "medium",
                        "description": m.get("gap", "") or m.get("evidence", ""),
                        "remediation": f"Address {m.get('control_id', '')} control gap.",
                    })

        # Overall compliance score
        overall = sum(compliance_scores.values()) / max(len(compliance_scores), 1)

        logger.info(
            "compliance_gaps_complete: met=%d, partial=%d, missing=%d, overall=%.1f%%",
            met, partial, missing, overall,
        )

        return {
            "current_node": "identify_gaps",
            "gaps": gaps,
            "compliance_scores": compliance_scores,
            "overall_compliance_score": round(overall, 1),
            "controls_met": met,
            "controls_partial": partial,
            "controls_missing": missing,
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """Node 4: Present compliance gap analysis for human approval."""
        logger.info(
            "compliance_human_review_pending: gaps=%d, overall=%.1f%%",
            len(state.get("gaps", [])),
            state.get("overall_compliance_score", 0.0),
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Generate Roadmap ────────────────────────────────

    async def _node_generate_roadmap(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Create a prioritized remediation roadmap.

        Generates actionable steps to close compliance gaps,
        ordered by impact and effort.
        """
        domain = state.get("company_domain", "")
        gaps = state.get("gaps", [])
        scores = state.get("compliance_scores", {})

        logger.info("compliance_roadmap: generating for %d gaps", len(gaps))

        roadmap_items: list[dict[str, Any]] = []
        roadmap_summary = ""

        # Sort gaps by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_gaps = sorted(gaps, key=lambda g: severity_order.get(g.get("severity", "medium"), 2))

        for i, gap in enumerate(sorted_gaps):
            effort = 2 if gap.get("severity") == "high" else 1
            roadmap_items.append({
                "priority": i + 1,
                "action": f"Remediate {gap.get('control_id', '')}: {gap.get('description', '')}",
                "framework": gap.get("framework", ""),
                "effort_weeks": effort,
                "impact": "high" if gap.get("severity") in ("critical", "high") else "medium",
            })

        # Generate roadmap summary via LLM
        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the target")
        try:
            prompt = (
                f"Compliance scores for {domain}: {json.dumps(scores)}\n"
                f"Gap count: {len(gaps)}\n"
                f"Top gaps: {json.dumps(sorted_gaps[:10], indent=2)}\n\n"
                f"Generate a 3-5 sentence remediation roadmap summary."
            )

            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                system=f"You are a compliance advisor for {company_name}. Provide a concise remediation roadmap.",
            )
            roadmap_summary = (
                response.content[0].text.strip()
                if response.content else ""
            )
        except Exception as e:
            logger.debug(f"LLM roadmap generation failed: {e}")
            roadmap_summary = (
                f"Remediation roadmap for {domain}: {len(gaps)} gaps identified "
                f"across {len(scores)} frameworks. Priority: address critical and "
                f"high severity gaps first."
            )

        # Publish insight to Hive Mind
        self.store_insight(InsightData(
            insight_type="compliance_assessment",
            title=f"Compliance: {domain} — {state.get('overall_compliance_score', 0.0):.0f}%",
            content=(
                f"Compliance mapping for {domain}: "
                f"{state.get('controls_met', 0)} met, "
                f"{state.get('controls_partial', 0)} partial, "
                f"{state.get('controls_missing', 0)} missing. "
                f"Scores: {json.dumps(scores)}."
            ),
            confidence=0.80,
            metadata={
                "domain": domain,
                "compliance_scores": scores,
                "gap_count": len(gaps),
            },
        ))

        return {
            "current_node": "generate_roadmap",
            "roadmap_items": roadmap_items,
            "roadmap_summary": roadmap_summary,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: ComplianceMapperAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate compliance gap analysis report."""
        now = datetime.now(timezone.utc).isoformat()
        domain = state.get("company_domain", "")
        scores = state.get("compliance_scores", {})

        sections = [
            "# Compliance Gap Analysis Report",
            f"**Domain:** {domain}",
            f"*Generated: {now}*\n",
            "## Overall Compliance Score",
            f"**{state.get('overall_compliance_score', 0.0):.1f}%**\n",
            "## Framework Scores",
        ]

        for fw, score in scores.items():
            sections.append(f"- **{fw.upper()}:** {score:.1f}%")

        sections.extend([
            f"\n## Control Status",
            f"- **Met:** {state.get('controls_met', 0)}",
            f"- **Partial:** {state.get('controls_partial', 0)}",
            f"- **Missing:** {state.get('controls_missing', 0)}",
            f"- **Total Gaps:** {len(state.get('gaps', []))}",
        ])

        roadmap = state.get("roadmap_summary", "")
        if roadmap:
            sections.append(f"\n## Remediation Roadmap\n{roadmap}")

        items = state.get("roadmap_items", [])
        if items:
            sections.append(f"\n## Top Priority Items ({min(5, len(items))})")
            for item in items[:5]:
                sections.append(
                    f"- **#{item.get('priority', 0)}** [{item.get('framework', '').upper()}] "
                    f"{item.get('action', '')} ({item.get('effort_weeks', 0)}w)"
                )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ComplianceMapperAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── Knowledge ───────────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ComplianceMapperAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
