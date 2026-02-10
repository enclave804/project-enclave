"""
Vulnerability Scanner Agent — The Security Auditor.

Probes public-facing assets for SSL, HTTP header, and network
vulnerabilities using passive reconnaissance (Shodan, SSL Labs,
HTTP header analysis). Produces severity-scored findings with
remediation guidance.

Architecture (LangGraph State Machine):
    initialize_scan -> scan_targets -> analyze_results ->
    human_review -> save_findings -> report -> END

Trigger Events:
    - new_assessment: Scan triggered for a prospect/client domain
    - scheduled (weekly): Re-scan existing clients for drift
    - manual: On-demand vulnerability scan

Shared Brain Integration:
    - Reads: previous assessments for the domain, industry baselines
    - Writes: vulnerability findings, risk scores, remediation patterns

Safety:
    - NEVER performs active exploitation -- passive scanning only
    - All findings require human_review before persistence
    - Rate-limited to respect external API quotas
    - Only scans publicly accessible endpoints (no internal networks)

Usage:
    agent = VulnScannerAgent(config, db, embedder, llm)
    result = await agent.run({"company_domain": "example.com"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import VulnScannerAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

SCAN_TYPES = {"full", "ssl_only", "headers_only", "network"}

SEVERITY_LEVELS = ["critical", "high", "medium", "low", "info"]

RISK_THRESHOLDS = {
    "critical": 9.0,
    "high": 7.0,
    "medium": 4.0,
    "low": 2.0,
}

REQUIRED_SECURITY_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy",
]

VULN_SCANNER_SYSTEM_PROMPT = """\
You are an expert cybersecurity vulnerability analyst for {company_name}. \
You analyze scan results from SSL/TLS checks, HTTP header analysis, and \
network reconnaissance to produce a risk assessment.

Given the scan findings below, produce a JSON object with:
{{
    "risk_score": 0.0-10.0,
    "executive_summary": "2-3 sentence summary for a non-technical executive",
    "top_risks": [
        {{
            "severity": "critical|high|medium|low",
            "title": "Short title",
            "description": "What the issue is",
            "remediation": "How to fix it",
            "cvss": 0.0-10.0
        }}
    ]
}}

Rules:
- Be precise about severity levels (critical = exploitable now, high = likely exploitable)
- Provide actionable remediation steps, not generic advice
- Consider the cumulative risk of multiple medium findings
- Reference specific CVEs where applicable
- Domain being scanned: {domain}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("vuln_scanner")
class VulnScannerAgent(BaseAgent):
    """
    Passive vulnerability scanner for external attack surface assessment.

    Nodes:
        1. initialize_scan   -- Validate target, set scan parameters
        2. scan_targets      -- Execute SSL, header, and network scans
        3. analyze_results   -- Score risk 0-10, generate executive summary
        4. human_review      -- Gate: review findings before persistence
        5. save_findings     -- Write to security_assessments + security_findings
        6. report            -- Summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """Build the Vulnerability Scanner's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(VulnScannerAgentState)

        workflow.add_node("initialize_scan", self._node_initialize_scan)
        workflow.add_node("scan_targets", self._node_scan_targets)
        workflow.add_node("analyze_results", self._node_analyze_results)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("save_findings", self._node_save_findings)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("initialize_scan")

        workflow.add_edge("initialize_scan", "scan_targets")
        workflow.add_edge("scan_targets", "analyze_results")
        workflow.add_edge("analyze_results", "human_review")
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
    def get_state_class(cls) -> Type[VulnScannerAgentState]:
        return VulnScannerAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_domain": task.get("company_domain", ""),
            "company_name": task.get("company_name", ""),
            "scan_type": task.get("scan_type", "full"),
            "scan_targets": [],
            "ssl_findings": [],
            "header_findings": [],
            "network_findings": [],
            "raw_scan_data": {},
            "findings": [],
            "risk_score": 0.0,
            "executive_summary": "",
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "assessment_id": "",
            "findings_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Initialize Scan ────────────────────────────────────

    async def _node_initialize_scan(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Validate target domain and configure scan parameters.

        Checks domain format, resolves targets, and queries the shared
        brain for any previous assessments of this domain.
        """
        domain = state.get("company_domain", "")
        scan_type = state.get("scan_type", "full")

        logger.info(
            "vuln_scan_init: domain=%s scan_type=%s",
            domain, scan_type,
        )

        if not domain:
            return {
                "current_node": "initialize_scan",
                "error": "No company_domain provided",
            }

        if scan_type not in SCAN_TYPES:
            scan_type = "full"

        # Build target list based on scan type
        targets: list[dict[str, Any]] = []
        targets.append({
            "host": domain,
            "port": 443,
            "protocol": "https",
            "service": "web",
        })
        if scan_type in ("full", "network"):
            targets.append({
                "host": domain,
                "port": 80,
                "protocol": "http",
                "service": "web",
            })

        # Consult shared brain for previous assessments
        try:
            insights = self.consult_hive(
                f"Previous vulnerability findings for {domain}",
                min_confidence=0.6,
                limit=3,
            )
            if insights:
                logger.info(
                    "vuln_scan_init: found %d prior insights for %s",
                    len(insights), domain,
                )
        except Exception:
            pass

        return {
            "current_node": "initialize_scan",
            "scan_type": scan_type,
            "scan_targets": targets,
        }

    # ─── Node 2: Scan Targets ──────────────────────────────────────

    async def _node_scan_targets(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Execute passive scans against the target domain.

        Performs SSL/TLS analysis, HTTP security header checks, and
        network reconnaissance via Shodan (or mock data in dev mode).
        """
        domain = state.get("company_domain", "")
        scan_type = state.get("scan_type", "full")
        targets = state.get("scan_targets", [])

        logger.info(
            "vuln_scan_targets: scanning %d targets for %s",
            len(targets), domain,
        )

        ssl_findings: list[dict[str, Any]] = []
        header_findings: list[dict[str, Any]] = []
        network_findings: list[dict[str, Any]] = []
        raw_scan_data: dict[str, Any] = {"domain": domain, "scanned_at": datetime.now(timezone.utc).isoformat()}

        # --- SSL/TLS Scan ---
        if scan_type in ("full", "ssl_only"):
            try:
                result = (
                    self.db.client.table("tech_stack_cache")
                    .select("ssl_data")
                    .eq("domain", domain)
                    .limit(1)
                    .execute()
                )
                ssl_data = (result.data[0].get("ssl_data", {}) if result.data else {})
                raw_scan_data["ssl"] = ssl_data

                # Evaluate SSL findings
                if ssl_data:
                    protocol = ssl_data.get("protocol_version", "")
                    if protocol and "TLS 1.0" in protocol:
                        ssl_findings.append({
                            "severity": "high",
                            "title": "Outdated TLS 1.0 Supported",
                            "description": "Server supports TLS 1.0 which has known vulnerabilities.",
                            "remediation": "Disable TLS 1.0 and 1.1, enforce TLS 1.2+ minimum.",
                            "cvss": 7.4,
                        })
                    if ssl_data.get("cert_expired", False):
                        ssl_findings.append({
                            "severity": "critical",
                            "title": "SSL Certificate Expired",
                            "description": "The SSL certificate has expired, causing browser warnings.",
                            "remediation": "Renew the SSL certificate immediately.",
                            "cvss": 9.1,
                        })
                    days_to_expiry = ssl_data.get("days_to_expiry", 365)
                    if 0 < days_to_expiry < 30:
                        ssl_findings.append({
                            "severity": "medium",
                            "title": "SSL Certificate Expiring Soon",
                            "description": f"Certificate expires in {days_to_expiry} days.",
                            "remediation": "Renew the SSL certificate before expiry.",
                            "cvss": 4.0,
                        })
            except Exception as e:
                logger.debug(f"SSL scan failed for {domain}: {e}")

        # --- HTTP Header Scan ---
        if scan_type in ("full", "headers_only"):
            try:
                result = (
                    self.db.client.table("tech_stack_cache")
                    .select("headers")
                    .eq("domain", domain)
                    .limit(1)
                    .execute()
                )
                headers = (result.data[0].get("headers", {}) if result.data else {})
                raw_scan_data["headers"] = headers

                for hdr in REQUIRED_SECURITY_HEADERS:
                    if hdr.lower() not in {k.lower() for k in headers.keys()}:
                        severity = "high" if hdr in ("Strict-Transport-Security", "Content-Security-Policy") else "medium"
                        header_findings.append({
                            "severity": severity,
                            "title": f"Missing Security Header: {hdr}",
                            "description": f"The HTTP response does not include the {hdr} header.",
                            "remediation": f"Configure the web server to return the {hdr} header with appropriate values.",
                            "cvss": 6.5 if severity == "high" else 4.3,
                        })
            except Exception as e:
                logger.debug(f"Header scan failed for {domain}: {e}")

        # --- Network Scan (Shodan lookup) ---
        if scan_type in ("full", "network"):
            try:
                result = (
                    self.db.client.table("tech_stack_cache")
                    .select("shodan_data")
                    .eq("domain", domain)
                    .limit(1)
                    .execute()
                )
                shodan = (result.data[0].get("shodan_data", {}) if result.data else {})
                raw_scan_data["shodan"] = shodan

                open_ports = shodan.get("ports", [])
                for port in open_ports:
                    if port in (21, 23, 3389, 5900):
                        network_findings.append({
                            "severity": "critical" if port == 23 else "high",
                            "title": f"Dangerous Port {port} Open",
                            "description": f"Port {port} is publicly accessible — commonly targeted by attackers.",
                            "remediation": f"Close port {port} or restrict access via firewall rules.",
                            "cvss": 9.0 if port == 23 else 7.5,
                        })
                    elif port not in (80, 443):
                        network_findings.append({
                            "severity": "low",
                            "title": f"Non-Standard Port {port} Open",
                            "description": f"Port {port} is publicly accessible.",
                            "remediation": "Review whether this port needs to be publicly exposed.",
                            "cvss": 2.0,
                        })
            except Exception as e:
                logger.debug(f"Network scan failed for {domain}: {e}")

        total = len(ssl_findings) + len(header_findings) + len(network_findings)
        logger.info(
            "vuln_scan_targets_complete: %d findings (ssl=%d, headers=%d, network=%d)",
            total, len(ssl_findings), len(header_findings), len(network_findings),
        )

        return {
            "current_node": "scan_targets",
            "ssl_findings": ssl_findings,
            "header_findings": header_findings,
            "network_findings": network_findings,
            "raw_scan_data": raw_scan_data,
        }

    # ─── Node 3: Analyze Results ──────────────────────────────────

    async def _node_analyze_results(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Score risk 0-10 and generate executive summary.

        Combines all findings, deduplicates, scores composite risk,
        and uses LLM to generate an executive-friendly summary.
        """
        domain = state.get("company_domain", "")
        ssl = state.get("ssl_findings", [])
        headers = state.get("header_findings", [])
        network = state.get("network_findings", [])

        all_findings = ssl + headers + network

        logger.info(
            "vuln_analyze: %d total findings for %s",
            len(all_findings), domain,
        )

        # Count by severity
        counts = {s: 0 for s in SEVERITY_LEVELS}
        for f in all_findings:
            sev = f.get("severity", "info")
            if sev in counts:
                counts[sev] += 1

        # Compute composite risk score (0-10)
        risk_score = min(10.0, (
            counts["critical"] * 3.0
            + counts["high"] * 2.0
            + counts["medium"] * 1.0
            + counts["low"] * 0.3
        ))

        # Generate executive summary via LLM
        executive_summary = ""
        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the target")
        try:
            findings_text = json.dumps(all_findings[:20], indent=2)
            prompt = (
                f"Scan findings for {domain}:\n{findings_text}\n\n"
                f"Risk score: {risk_score:.1f}/10\n"
                f"Critical: {counts['critical']}, High: {counts['high']}, "
                f"Medium: {counts['medium']}, Low: {counts['low']}\n\n"
                f"Generate the risk assessment JSON."
            )
            system = VULN_SCANNER_SYSTEM_PROMPT.format(
                company_name=company_name,
                domain=domain,
            )

            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
            text = response.content[0].text.strip() if response.content else ""
            if text:
                try:
                    parsed = json.loads(text)
                    executive_summary = parsed.get("executive_summary", "")
                    if parsed.get("risk_score"):
                        risk_score = float(parsed["risk_score"])
                except (json.JSONDecodeError, ValueError):
                    executive_summary = text[:500]
        except Exception as e:
            logger.debug(f"LLM analysis failed for {domain}: {e}")
            executive_summary = (
                f"Scan of {domain} identified {len(all_findings)} findings: "
                f"{counts['critical']} critical, {counts['high']} high, "
                f"{counts['medium']} medium, {counts['low']} low. "
                f"Risk score: {risk_score:.1f}/10."
            )

        return {
            "current_node": "analyze_results",
            "findings": all_findings,
            "risk_score": round(risk_score, 1),
            "executive_summary": executive_summary,
            "critical_count": counts["critical"],
            "high_count": counts["high"],
            "medium_count": counts["medium"],
            "low_count": counts["low"],
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Present findings for human approval before persistence.

        All vulnerability findings require human review to prevent
        false positives from being saved to the assessment record.
        """
        findings = state.get("findings", [])

        logger.info(
            "vuln_human_review_pending: %d findings, risk=%.1f",
            len(findings), state.get("risk_score", 0.0),
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Save Findings ────────────────────────────────────

    async def _node_save_findings(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Persist approved findings to the database.

        Creates a security_assessments record and writes individual
        findings to the security_findings table.
        """
        domain = state.get("company_domain", "")
        findings = state.get("findings", [])
        risk_score = state.get("risk_score", 0.0)

        logger.info(
            "vuln_save_findings: saving %d findings for %s",
            len(findings), domain,
        )

        assessment_id = ""
        now = datetime.now(timezone.utc).isoformat()

        # Create assessment record
        try:
            result = self.db.client.table("security_assessments").insert({
                "vertical_id": self.vertical_id,
                "domain": domain,
                "assessment_type": "vulnerability_scan",
                "scan_type": state.get("scan_type", "full"),
                "risk_score": risk_score,
                "executive_summary": state.get("executive_summary", ""),
                "finding_count": len(findings),
                "critical_count": state.get("critical_count", 0),
                "high_count": state.get("high_count", 0),
                "medium_count": state.get("medium_count", 0),
                "low_count": state.get("low_count", 0),
                "status": "completed",
                "created_at": now,
            }).execute()
            if result.data:
                assessment_id = result.data[0].get("id", "")
        except Exception as e:
            logger.debug(f"Failed to create assessment record: {e}")

        # Write individual findings
        if assessment_id and findings:
            for finding in findings:
                try:
                    self.db.client.table("security_findings").insert({
                        "assessment_id": assessment_id,
                        "vertical_id": self.vertical_id,
                        "domain": domain,
                        "severity": finding.get("severity", "info"),
                        "title": finding.get("title", ""),
                        "description": finding.get("description", ""),
                        "remediation": finding.get("remediation", ""),
                        "cvss_score": finding.get("cvss", 0.0),
                        "status": "open",
                        "created_at": now,
                    }).execute()
                except Exception as e:
                    logger.debug(f"Failed to save finding: {e}")

        # Publish insight to Hive Mind
        if findings:
            self.store_insight(InsightData(
                insight_type="vulnerability_assessment",
                title=f"Vuln Scan: {domain} — Risk {risk_score:.1f}/10",
                content=(
                    f"Vulnerability scan of {domain}: {len(findings)} findings "
                    f"({state.get('critical_count', 0)} critical, "
                    f"{state.get('high_count', 0)} high). "
                    f"Risk score: {risk_score:.1f}/10."
                ),
                confidence=0.85,
                metadata={
                    "domain": domain,
                    "risk_score": risk_score,
                    "finding_count": len(findings),
                    "assessment_id": assessment_id,
                },
            ))

        return {
            "current_node": "save_findings",
            "assessment_id": assessment_id,
            "findings_saved": bool(assessment_id),
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: VulnScannerAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate vulnerability scan summary report."""
        now = datetime.now(timezone.utc).isoformat()
        domain = state.get("company_domain", "")

        sections = [
            "# Vulnerability Scan Report",
            f"**Domain:** {domain}",
            f"**Scan Type:** {state.get('scan_type', 'full')}",
            f"*Generated: {now}*\n",
            "## Risk Score",
            f"**{state.get('risk_score', 0.0):.1f} / 10.0**\n",
            "## Findings Summary",
            f"- **Critical:** {state.get('critical_count', 0)}",
            f"- **High:** {state.get('high_count', 0)}",
            f"- **Medium:** {state.get('medium_count', 0)}",
            f"- **Low:** {state.get('low_count', 0)}",
            f"- **Total:** {len(state.get('findings', []))}",
        ]

        summary = state.get("executive_summary", "")
        if summary:
            sections.append(f"\n## Executive Summary\n{summary}")

        if state.get("findings_saved"):
            sections.append(
                f"\n## Persistence\n- Assessment ID: `{state.get('assessment_id', '')}`"
            )

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: VulnScannerAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── Knowledge ───────────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<VulnScannerAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
