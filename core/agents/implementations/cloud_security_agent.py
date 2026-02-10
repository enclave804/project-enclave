"""
Cloud Security Agent — The Cloud Auditor.

Evaluates cloud infrastructure configurations across AWS, Azure, and GCP.
Checks storage bucket permissions, IAM roles, security groups, encryption
settings, and audit logging to identify misconfigurations and compliance gaps.

Architecture (LangGraph State Machine):
    identify_cloud → scan_configs → analyze_misconfigs →
    human_review → save_findings → report → END

Trigger Events:
    - security_assessment: Full cloud security audit
    - questionnaire_submitted: Client completes cloud questionnaire
    - manual: On-demand cloud review

Shared Brain Integration:
    - Reads: compliance requirements, CIS benchmarks, peer cloud findings
    - Writes: misconfiguration patterns, cloud risk insights

Safety:
    - NEVER accesses client cloud accounts directly
    - Assessment based on questionnaire and self-reported configurations
    - All findings require human_review gate before saving
    - Does not modify cloud resources

Usage:
    agent = CloudSecurityAgent(config, db, embedder, llm)
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
from core.agents.state import CloudSecurityAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

CLOUD_PROVIDERS = {
    "aws": {"label": "Amazon Web Services", "benchmarks": ["CIS AWS", "AWS Well-Architected"]},
    "azure": {"label": "Microsoft Azure", "benchmarks": ["CIS Azure", "Azure Security Benchmark"]},
    "gcp": {"label": "Google Cloud Platform", "benchmarks": ["CIS GCP", "Google Cloud Security"]},
}

MISCONFIG_CATEGORIES = {
    "storage": {"label": "Storage & Data", "weight": 0.20},
    "iam": {"label": "Identity & Access", "weight": 0.25},
    "network": {"label": "Network Security", "weight": 0.20},
    "encryption": {"label": "Encryption", "weight": 0.15},
    "logging": {"label": "Logging & Monitoring", "weight": 0.10},
    "compute": {"label": "Compute Security", "weight": 0.10},
}

CLOUD_BENCHMARKS = {
    "cis_level_1": {"label": "CIS Level 1 (Essential)", "min_compliance": 0.90},
    "cis_level_2": {"label": "CIS Level 2 (Hardened)", "min_compliance": 0.75},
    "well_architected": {"label": "Well-Architected Security Pillar", "min_compliance": 0.80},
}

CLOUD_SYSTEM_PROMPT = """\
You are a cloud security assessor specializing in AWS, Azure, and GCP \
security configurations. Given the questionnaire responses below, identify \
misconfigurations and produce a JSON object with:
{{
    "misconfig_findings": [
        {{"finding": "...", "severity": "critical|high|medium|low", \
"category": "storage|iam|network|encryption|logging|compute", \
"resource": "...", "recommendation": "...", "benchmark_ref": "..."}}
    ],
    "benchmark_compliance": {{
        "cis_level_1": 0.0-1.0,
        "cis_level_2": 0.0-1.0
    }},
    "summary": "Brief executive summary"
}}

Cloud Providers: {providers}
Company: {company_name}
Industry: {industry}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("cloud_security")
class CloudSecurityAgent(BaseAgent):
    """
    Cloud security assessment agent for Enclave Guard engagements.

    Nodes:
        1. identify_cloud     -- Identify cloud providers and services in use
        2. scan_configs       -- Evaluate storage, IAM, network, encryption configs
        3. analyze_misconfigs -- Score misconfigurations against CIS benchmarks
        4. human_review       -- Gate: approve findings before saving
        5. save_findings      -- Persist findings to DB and shared brain
        6. report             -- Generate cloud security assessment report
    """

    def build_graph(self) -> Any:
        """Build the Cloud Security Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(CloudSecurityAgentState)

        workflow.add_node("identify_cloud", self._node_identify_cloud)
        workflow.add_node("scan_configs", self._node_scan_configs)
        workflow.add_node("analyze_misconfigs", self._node_analyze_misconfigs)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("save_findings", self._node_save_findings)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("identify_cloud")

        workflow.add_edge("identify_cloud", "scan_configs")
        workflow.add_edge("scan_configs", "analyze_misconfigs")
        workflow.add_edge("analyze_misconfigs", "human_review")
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
    def get_state_class(cls) -> Type[CloudSecurityAgentState]:
        return CloudSecurityAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "cloud_providers": [],
            "cloud_services_in_use": [],
            "cloud_accounts": [],
            "scan_results": [],
            "public_buckets": [],
            "open_security_groups": [],
            "unencrypted_resources": [],
            "logging_gaps": [],
            "misconfig_findings": [],
            "misconfig_by_category": {},
            "misconfig_by_severity": {},
            "benchmark_compliance": {},
            "cloud_risk_score": 0.0,
            "cloud_risk_level": "low",
            "all_findings": [],
            "findings_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Identify Cloud ──────────────────────────────────────

    async def _node_identify_cloud(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 1: Identify cloud providers, accounts, and services in use."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})
        company = task.get("company_name", "Unknown")

        logger.info("cloud_identify", extra={"company": company})

        providers = responses.get("cloud_providers", [])
        if isinstance(providers, str):
            providers = [p.strip().lower() for p in providers.split(",") if p.strip()]

        services = responses.get("cloud_services", [])
        if isinstance(services, str):
            services = [s.strip() for s in services.split(",") if s.strip()]

        accounts = responses.get("cloud_accounts", [])
        if not isinstance(accounts, list):
            accounts = []

        logger.info(
            "cloud_identified",
            extra={
                "providers": providers,
                "service_count": len(services),
                "account_count": len(accounts),
            },
        )

        return {
            "current_node": "identify_cloud",
            "cloud_providers": providers,
            "cloud_services_in_use": services,
            "cloud_accounts": accounts,
        }

    # ─── Node 2: Scan Configs ────────────────────────────────────────

    async def _node_scan_configs(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 2: Evaluate storage, IAM, network, and encryption configurations."""
        task = state.get("task_input", {})
        responses = task.get("questionnaire", {})
        providers = state.get("cloud_providers", [])

        logger.info("cloud_scan_configs", extra={"providers": providers})

        public_buckets: list[dict[str, Any]] = []
        open_sgs: list[dict[str, Any]] = []
        unencrypted: list[dict[str, Any]] = []
        logging_gaps: list[dict[str, Any]] = []
        scan_results: list[dict[str, Any]] = []

        # Storage analysis
        public_storage = responses.get("public_storage_buckets", [])
        if isinstance(public_storage, list):
            for bucket in public_storage:
                if isinstance(bucket, dict):
                    public_buckets.append(bucket)
                else:
                    public_buckets.append({"name": str(bucket), "provider": providers[0] if providers else "unknown"})
        elif responses.get("has_public_buckets", False):
            public_buckets.append({
                "name": "unspecified_public_bucket",
                "provider": providers[0] if providers else "unknown",
            })

        # Security group analysis
        open_ports = responses.get("open_security_groups", [])
        if isinstance(open_ports, list):
            for sg in open_ports:
                if isinstance(sg, dict):
                    open_sgs.append(sg)
        elif responses.get("has_open_security_groups", False):
            open_sgs.append({
                "rule": "0.0.0.0/0 inbound",
                "provider": providers[0] if providers else "unknown",
            })

        # Encryption checks
        if not responses.get("storage_encryption_enabled", False):
            unencrypted.append({
                "resource_type": "storage",
                "provider": providers[0] if providers else "unknown",
                "detail": "Storage volumes not encrypted at rest",
            })
        if not responses.get("database_encryption_enabled", False):
            unencrypted.append({
                "resource_type": "database",
                "provider": providers[0] if providers else "unknown",
                "detail": "Database instances not encrypted at rest",
            })
        if not responses.get("transit_encryption_enabled", False):
            unencrypted.append({
                "resource_type": "network",
                "provider": providers[0] if providers else "unknown",
                "detail": "Data in transit not encrypted (no TLS enforcement)",
            })

        # Logging checks
        if not responses.get("cloud_trail_enabled", False):
            logging_gaps.append({
                "service": "audit_trail",
                "provider": providers[0] if providers else "unknown",
                "detail": "Cloud audit trail / activity logging not enabled",
            })
        if not responses.get("flow_logs_enabled", False):
            logging_gaps.append({
                "service": "flow_logs",
                "provider": providers[0] if providers else "unknown",
                "detail": "VPC/network flow logs not enabled",
            })

        return {
            "current_node": "scan_configs",
            "scan_results": scan_results,
            "public_buckets": public_buckets,
            "open_security_groups": open_sgs,
            "unencrypted_resources": unencrypted,
            "logging_gaps": logging_gaps,
        }

    # ─── Node 3: Analyze Misconfigs ──────────────────────────────────

    async def _node_analyze_misconfigs(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 3: Score misconfigurations against CIS benchmarks."""
        public_buckets = state.get("public_buckets", [])
        open_sgs = state.get("open_security_groups", [])
        unencrypted = state.get("unencrypted_resources", [])
        logging_gaps = state.get("logging_gaps", [])

        logger.info("cloud_analyze_misconfigs", extra={"agent_id": self.agent_id})

        findings: list[dict[str, Any]] = []

        # Public storage findings
        for bucket in public_buckets:
            findings.append({
                "finding": f"Publicly accessible storage: {bucket.get('name', 'unknown')}",
                "severity": "critical",
                "category": "storage",
                "resource": bucket.get("name", ""),
                "recommendation": "Restrict public access immediately. Enable "
                                  "S3 Block Public Access or equivalent.",
                "benchmark_ref": "CIS 2.1.1",
            })

        # Open security groups
        for sg in open_sgs:
            findings.append({
                "finding": f"Overly permissive security group: {sg.get('rule', 'open')}",
                "severity": "high",
                "category": "network",
                "resource": sg.get("group_id", sg.get("rule", "")),
                "recommendation": "Restrict inbound rules to specific IP ranges "
                                  "and required ports only.",
                "benchmark_ref": "CIS 4.1",
            })

        # Unencrypted resources
        for res in unencrypted:
            sev = "critical" if res.get("resource_type") == "database" else "high"
            findings.append({
                "finding": f"Unencrypted {res.get('resource_type', 'resource')}: "
                           f"{res.get('detail', '')}",
                "severity": sev,
                "category": "encryption",
                "resource": res.get("resource_type", ""),
                "recommendation": "Enable encryption at rest using KMS or "
                                  "provider-managed keys.",
                "benchmark_ref": "CIS 2.2",
            })

        # Logging gaps
        for gap in logging_gaps:
            findings.append({
                "finding": f"Missing logging: {gap.get('detail', '')}",
                "severity": "high",
                "category": "logging",
                "resource": gap.get("service", ""),
                "recommendation": "Enable audit logging and centralize logs "
                                  "to SIEM for monitoring.",
                "benchmark_ref": "CIS 3.1",
            })

        # Categorize findings
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for f in findings:
            cat = f.get("category", "other")
            sev = f.get("severity", "low")
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Risk score calculation
        severity_weights = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        risk_score = min(
            100.0,
            sum(severity_weights.get(f.get("severity", "low"), 3) for f in findings),
        )

        risk_level = "low"
        if risk_score >= 80:
            risk_level = "critical"
        elif risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"

        # Estimate benchmark compliance
        total_checks = max(len(findings) + 10, 20)  # Estimate total applicable checks
        passed = total_checks - len(findings)
        benchmark_compliance = {
            "cis_level_1": round(max(0.0, passed / total_checks), 2),
            "cis_level_2": round(max(0.0, (passed - 3) / total_checks), 2),
        }

        logger.info(
            "cloud_analysis_complete",
            extra={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "findings": len(findings),
            },
        )

        return {
            "current_node": "analyze_misconfigs",
            "misconfig_findings": findings,
            "misconfig_by_category": by_category,
            "misconfig_by_severity": by_severity,
            "benchmark_compliance": benchmark_compliance,
            "cloud_risk_score": round(risk_score, 1),
            "cloud_risk_level": risk_level,
            "all_findings": findings,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 4: Present cloud findings for human approval."""
        findings = state.get("all_findings", [])
        logger.info(
            "cloud_human_review_pending",
            extra={
                "finding_count": len(findings),
                "risk_score": state.get("cloud_risk_score", 0),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Save Findings ──────────────────────────────────────

    async def _node_save_findings(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 5: Persist approved findings to DB and shared brain."""
        findings = state.get("all_findings", [])
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")
        providers = state.get("cloud_providers", [])

        logger.info("cloud_save_findings", extra={"count": len(findings)})

        saved = 0
        for finding in findings:
            try:
                self.db.client.table("security_findings").insert({
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "company_name": company,
                    "domain": "cloud",
                    "finding": finding.get("finding", ""),
                    "severity": finding.get("severity", "low"),
                    "category": finding.get("category", ""),
                    "recommendation": finding.get("recommendation", ""),
                    "benchmark_ref": finding.get("benchmark_ref", ""),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }).execute()
                saved += 1
            except Exception as e:
                logger.debug(f"Failed to save cloud finding: {e}")

        if findings:
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            self.store_insight(InsightData(
                insight_type="cloud_security_assessment",
                title=f"Cloud Security: {company} — "
                      f"{state.get('cloud_risk_level', 'unknown').upper()} risk",
                content=(
                    f"Cloud security assessment for {company} "
                    f"({', '.join(providers)}): risk score "
                    f"{state.get('cloud_risk_score', 0)}/100, "
                    f"{len(findings)} misconfigurations ({critical} critical). "
                    f"CIS L1 compliance: "
                    f"{state.get('benchmark_compliance', {}).get('cis_level_1', 0):.0%}."
                ),
                confidence=0.85,
                metadata={
                    "company": company,
                    "providers": providers,
                    "risk_score": state.get("cloud_risk_score", 0),
                    "finding_count": len(findings),
                },
            ))

        return {
            "current_node": "save_findings",
            "findings_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: CloudSecurityAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate cloud security assessment report."""
        now = datetime.now(timezone.utc).isoformat()
        task = state.get("task_input", {})
        company = task.get("company_name", "Unknown")
        findings = state.get("all_findings", [])
        by_sev = state.get("misconfig_by_severity", {})
        by_cat = state.get("misconfig_by_category", {})
        benchmarks = state.get("benchmark_compliance", {})

        sections = [
            "# Cloud Security Assessment Report",
            f"*Generated: {now}*\n",
            f"## Organization: {company}",
            f"- **Cloud Providers:** {', '.join(state.get('cloud_providers', [])) or 'N/A'}",
            f"- **Services in Use:** {len(state.get('cloud_services_in_use', []))}",
            f"\n## Risk Summary",
            f"- **Cloud Risk Score:** {state.get('cloud_risk_score', 0)}/100 "
            f"({state.get('cloud_risk_level', 'unknown').upper()})",
            f"- **Total Misconfigurations:** {len(findings)}",
            f"- **Critical:** {by_sev.get('critical', 0)}",
            f"- **High:** {by_sev.get('high', 0)}",
            f"- **Medium:** {by_sev.get('medium', 0)}",
            f"- **Low:** {by_sev.get('low', 0)}",
            f"\n## Findings by Category",
        ]

        for cat, info in MISCONFIG_CATEGORIES.items():
            count = by_cat.get(cat, 0)
            sections.append(f"- **{info['label']}:** {count} findings")

        sections.append("\n## CIS Benchmark Compliance")
        for bench_id, bench_info in CLOUD_BENCHMARKS.items():
            pct = benchmarks.get(bench_id, 0)
            target = bench_info["min_compliance"]
            status = "PASS" if pct >= target else "FAIL"
            sections.append(
                f"- **{bench_info['label']}:** {pct:.0%} "
                f"(target: {target:.0%}) [{status}]"
            )

        sections.append(f"\n## Key Issues")
        sections.append(
            f"- **Public Buckets:** {len(state.get('public_buckets', []))}"
        )
        sections.append(
            f"- **Open Security Groups:** {len(state.get('open_security_groups', []))}"
        )
        sections.append(
            f"- **Unencrypted Resources:** {len(state.get('unencrypted_resources', []))}"
        )
        sections.append(
            f"- **Logging Gaps:** {len(state.get('logging_gaps', []))}"
        )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: CloudSecurityAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<CloudSecurityAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
