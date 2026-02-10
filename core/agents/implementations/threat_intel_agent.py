"""
Threat Intel Agent -- The Cyber Watchdog.

Gathers threat intelligence from feeds, analyzes CVEs and IOCs,
prioritizes risks by severity and exploitability, and generates
security advisories with mitigation recommendations.

Architecture (LangGraph State Machine):
    gather_feeds -> analyze_threats -> prioritize_risks ->
    human_review -> report -> END

Trigger Events:
    - scheduled: Hourly/daily threat feed ingestion
    - cve_published: New CVE matching monitored technologies
    - ioc_detected: Indicator of compromise flagged
    - manual: On-demand threat analysis

Shared Brain Integration:
    - Reads: technology stack, asset inventory, previous advisories
    - Writes: threat patterns, risk scores, mitigation playbooks

Safety:
    - NEVER executes exploit code or vulnerability proofs
    - All intelligence gathered from public feeds only
    - Advisories require human review before distribution
    - Risk scores are advisory; security team judgment required

Usage:
    agent = ThreatIntelAgent(config, db, embedder, llm)
    result = await agent.run({
        "feeds": ["nvd", "cisa_kev", "mitre_att&ck"],
        "technology_filter": ["python", "nodejs", "postgresql"],
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import ThreatIntelData
from core.agents.registry import register_agent_type
from core.agents.state import ThreatIntelAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

THREAT_TYPES = [
    "vulnerability",
    "malware",
    "phishing",
    "data_breach",
    "insider_threat",
    "apt",
    "ddos",
    "ransomware",
]

SEVERITY_MAP = {
    "critical": {
        "score_min": 9.0,
        "description": "Actively exploited or trivially exploitable with severe impact",
        "color": "red",
        "response_window": "4 hours",
    },
    "high": {
        "score_min": 7.0,
        "description": "Exploitable with significant impact, patch urgently",
        "color": "orange",
        "response_window": "24 hours",
    },
    "medium": {
        "score_min": 4.0,
        "description": "Exploitable under specific conditions, schedule patch",
        "color": "yellow",
        "response_window": "7 days",
    },
    "low": {
        "score_min": 0.1,
        "description": "Limited exploitability or minimal impact",
        "color": "blue",
        "response_window": "30 days",
    },
}

CVE_SCORE_THRESHOLDS = {
    "critical": 9.0,
    "high": 7.0,
    "medium": 4.0,
    "low": 0.1,
}

FEED_SOURCES = [
    {"name": "nvd", "description": "NIST National Vulnerability Database", "type": "cve"},
    {"name": "cisa_kev", "description": "CISA Known Exploited Vulnerabilities", "type": "cve"},
    {"name": "mitre_attck", "description": "MITRE ATT&CK Framework", "type": "ttp"},
    {"name": "abuse_ch", "description": "abuse.ch Malware Bazaar", "type": "ioc"},
    {"name": "otx_alienvault", "description": "AlienVault OTX Pulse Feed", "type": "ioc"},
    {"name": "phishtank", "description": "PhishTank Phishing URLs", "type": "phishing"},
    {"name": "cert_advisories", "description": "CERT/CC Advisories", "type": "advisory"},
    {"name": "github_advisories", "description": "GitHub Security Advisories", "type": "cve"},
]

# --- LLM Prompt Templates ---------------------------------------------------

THREAT_ANALYSIS_PROMPT = """You are a cyber threat intelligence analyst. Analyze the following threat data and classify each finding.

Technology Context:
{technology_context_json}

Threat Feed Data:
{feed_data_json}

For each threat, return a JSON array of objects:
[
    {{
        "threat_id": 0,
        "threat_type": "one of: {threat_types}",
        "title": "Concise threat title",
        "description": "Detailed threat description",
        "cve_id": "CVE-XXXX-XXXXX or empty string",
        "cvss_score": 0.0-10.0,
        "severity": "critical|high|medium|low",
        "ioc_indicators": ["IP", "hash", "domain"],
        "affected_systems": ["system1", "system2"],
        "exploit_available": true or false,
        "actively_exploited": true or false,
        "attack_vector": "network|local|physical|adjacent",
        "confidence": 0.0-1.0,
        "source_feed": "feed name"
    }}
]

Score CVSS from 0.0 to 10.0 per CVSS v3.1 guidelines.
Flag actively exploited vulnerabilities as critical regardless of base score.

Return ONLY the JSON array, no markdown code fences.
"""

RISK_PRIORITIZATION_PROMPT = """You are a cyber threat intelligence analyst. Prioritize the analyzed threats and generate mitigation recommendations.

Threat Findings:
{findings_json}

Severity Mapping:
{severity_map_json}

For each threat, return a JSON object with prioritized threats and mitigations:
{{
    "prioritized_threats": [
        {{
            "threat_id": 0,
            "priority_rank": 1,
            "risk_score": 0.0-10.0,
            "exploitability": "trivial|moderate|complex|theoretical",
            "business_impact": "Revenue loss, data exposure, etc.",
            "mitigation": "Specific mitigation steps",
            "mitigation_effort": "immediate|short_term|long_term",
            "references": ["URL1", "URL2"]
        }}
    ],
    "overall_risk_score": 0.0-10.0,
    "executive_summary": "Brief overall risk assessment"
}}

Rank by: actively exploited > CVSS score > exploit availability > affected systems.
Be specific about mitigations -- generic advice is not helpful.

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("threat_intel")
class ThreatIntelAgent(BaseAgent):
    """
    Threat intelligence gathering and risk prioritization agent.

    Nodes:
        1. gather_feeds       -- Pull data from threat_intelligence table
        2. analyze_threats    -- LLM classifies CVEs, extracts IOCs
        3. prioritize_risks   -- Score and rank threats, generate mitigations
        4. human_review       -- Gate: approve advisory
        5. report             -- Save to DB + store ThreatIntelData insight
    """

    def build_graph(self) -> Any:
        """Build the Threat Intel Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ThreatIntelAgentState)

        workflow.add_node("gather_feeds", self._node_gather_feeds)
        workflow.add_node("analyze_threats", self._node_analyze_threats)
        workflow.add_node("prioritize_risks", self._node_prioritize_risks)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("gather_feeds")

        workflow.add_edge("gather_feeds", "analyze_threats")
        workflow.add_edge("analyze_threats", "prioritize_risks")
        workflow.add_edge("prioritize_risks", "human_review")
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
    def get_state_class(cls) -> Type[ThreatIntelAgentState]:
        return ThreatIntelAgentState

    # --- State Preparation ---------------------------------------------------

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "feed_data": [],
            "feed_count": 0,
            "sources_queried": [],
            "threat_findings": [],
            "threat_count": 0,
            "cve_count": 0,
            "ioc_count": 0,
            "prioritized_threats": [],
            "critical_count": 0,
            "high_count": 0,
            "overall_risk_score": 0.0,
            "mitigations": [],
            "advisory_approved": False,
            "threats_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # --- Node 1: Gather Feeds ------------------------------------------------

    async def _node_gather_feeds(
        self, state: ThreatIntelAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull threat data from threat_intelligence table and feeds."""
        task = state.get("task_input", {})

        logger.info(
            "threat_intel_gather_feeds",
            extra={"agent_id": self.agent_id},
        )

        requested_feeds = task.get("feeds", [])
        technology_filter = task.get("technology_filter", [])

        if not requested_feeds:
            requested_feeds = self.config.params.get("feeds", [
                s["name"] for s in FEED_SOURCES
            ])

        feed_data: list[dict[str, Any]] = []
        sources_queried: list[str] = []

        # Load from threat_intelligence table
        try:
            result = (
                self.db.client.table("threat_intelligence")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .limit(100)
                .execute()
            )
            if result.data:
                for record in result.data:
                    source = record.get("source_feed", "database")
                    feed_data.append({
                        "threat_id": record.get("id", ""),
                        "threat_type": record.get("threat_type", ""),
                        "title": record.get("title", ""),
                        "description": record.get("description", ""),
                        "cve_id": record.get("cve_id", ""),
                        "cvss_score": record.get("cvss_score", 0.0),
                        "ioc_indicators": record.get("ioc_indicators", []),
                        "affected_systems": record.get("affected_systems", []),
                        "source_feed": source,
                        "detected_at": record.get("created_at", ""),
                    })
                    if source not in sources_queried:
                        sources_queried.append(source)
        except Exception as e:
            logger.warning(
                "threat_intel_db_feed_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Add task-provided feed data
        task_feeds = task.get("feed_data", [])
        for tf in task_feeds:
            feed_data.append(tf)
            src = tf.get("source_feed", "task_input")
            if src not in sources_queried:
                sources_queried.append(src)

        # Filter by requested feed sources
        if requested_feeds:
            valid_sources = set(requested_feeds) | {"database", "task_input"}
            feed_data = [
                fd for fd in feed_data
                if fd.get("source_feed", "") in valid_sources
                or not fd.get("source_feed")
            ]

        # Track which FEED_SOURCES were matched
        for fs in FEED_SOURCES:
            if fs["name"] in requested_feeds and fs["name"] not in sources_queried:
                sources_queried.append(fs["name"])

        logger.info(
            "threat_intel_feeds_gathered",
            extra={
                "feed_items": len(feed_data),
                "sources_count": len(sources_queried),
                "tech_filter": len(technology_filter),
            },
        )

        return {
            "current_node": "gather_feeds",
            "feed_data": feed_data,
            "feed_count": len(feed_data),
            "sources_queried": sources_queried,
        }

    # --- Node 2: Analyze Threats ---------------------------------------------

    async def _node_analyze_threats(
        self, state: ThreatIntelAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM analyzes threats, classifies CVEs, extracts IOCs."""
        feed_data = state.get("feed_data", [])
        task = state.get("task_input", {})
        technology_filter = task.get("technology_filter", [])

        if not technology_filter:
            technology_filter = self.config.params.get("technology_stack", [])

        logger.info(
            "threat_intel_analyze_threats",
            extra={"feed_items_to_analyze": len(feed_data)},
        )

        threat_findings: list[dict[str, Any]] = []
        cve_count = 0
        ioc_count = 0

        if not feed_data:
            logger.info("threat_intel_no_feed_data")
            return {
                "current_node": "analyze_threats",
                "threat_findings": [],
                "threat_count": 0,
                "cve_count": 0,
                "ioc_count": 0,
            }

        try:
            prompt = THREAT_ANALYSIS_PROMPT.format(
                technology_context_json=json.dumps(technology_filter, indent=2),
                feed_data_json=json.dumps(feed_data[:25], indent=2),
                threat_types=", ".join(THREAT_TYPES),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a cyber threat intelligence analyst.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                findings_data = json.loads(llm_text)
                if isinstance(findings_data, list):
                    threat_findings = findings_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("threat_intel_analysis_parse_error")

        except Exception as e:
            logger.warning(
                "threat_intel_llm_analysis_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Count CVEs and IOCs
        for tf in threat_findings:
            if tf.get("cve_id"):
                cve_count += 1
            iocs = tf.get("ioc_indicators", [])
            ioc_count += len(iocs)

        logger.info(
            "threat_intel_analysis_complete",
            extra={
                "threats_found": len(threat_findings),
                "cves_identified": cve_count,
                "iocs_extracted": ioc_count,
            },
        )

        return {
            "current_node": "analyze_threats",
            "threat_findings": threat_findings,
            "threat_count": len(threat_findings),
            "cve_count": cve_count,
            "ioc_count": ioc_count,
        }

    # --- Node 3: Prioritize Risks --------------------------------------------

    async def _node_prioritize_risks(
        self, state: ThreatIntelAgentState
    ) -> dict[str, Any]:
        """Node 3: Score and rank threats, generate mitigation recommendations."""
        threat_findings = state.get("threat_findings", [])

        logger.info(
            "threat_intel_prioritize_risks",
            extra={"threats_to_prioritize": len(threat_findings)},
        )

        prioritized_threats: list[dict[str, Any]] = []
        mitigations: list[dict[str, Any]] = []
        overall_risk_score = 0.0
        critical_count = 0
        high_count = 0

        if not threat_findings:
            logger.info("threat_intel_no_threats_to_prioritize")
            return {
                "current_node": "prioritize_risks",
                "prioritized_threats": [],
                "critical_count": 0,
                "high_count": 0,
                "overall_risk_score": 0.0,
                "mitigations": [],
            }

        try:
            prompt = RISK_PRIORITIZATION_PROMPT.format(
                findings_json=json.dumps(threat_findings[:20], indent=2),
                severity_map_json=json.dumps(SEVERITY_MAP, indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a cyber threat intelligence analyst prioritizing risks.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                prioritization_data = json.loads(llm_text)
                if isinstance(prioritization_data, dict):
                    prioritized_threats = prioritization_data.get(
                        "prioritized_threats", []
                    )
                    overall_risk_score = prioritization_data.get(
                        "overall_risk_score", 0.0
                    )
            except (json.JSONDecodeError, KeyError):
                logger.debug("threat_intel_prioritization_parse_error")
                # Fallback: sort by CVSS score
                sorted_threats = sorted(
                    threat_findings,
                    key=lambda t: t.get("cvss_score", 0.0),
                    reverse=True,
                )
                for rank, tf in enumerate(sorted_threats, 1):
                    prioritized_threats.append({
                        "threat_id": tf.get("threat_id", rank),
                        "priority_rank": rank,
                        "risk_score": tf.get("cvss_score", 0.0),
                        "exploitability": "moderate",
                        "business_impact": "Requires assessment",
                        "mitigation": "Apply vendor patches and monitor",
                        "mitigation_effort": "short_term",
                        "references": [],
                    })
                if sorted_threats:
                    overall_risk_score = sorted_threats[0].get("cvss_score", 0.0)

        except Exception as e:
            logger.warning(
                "threat_intel_prioritization_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Count severity levels
        for tf in threat_findings:
            severity = tf.get("severity", "low")
            if severity == "critical":
                critical_count += 1
            elif severity == "high":
                high_count += 1

        # Extract mitigations from prioritized threats
        for pt in prioritized_threats:
            if pt.get("mitigation"):
                mitigations.append({
                    "threat_id": pt.get("threat_id"),
                    "priority_rank": pt.get("priority_rank"),
                    "mitigation": pt.get("mitigation", ""),
                    "effort": pt.get("mitigation_effort", "short_term"),
                    "risk_score": pt.get("risk_score", 0.0),
                })

        logger.info(
            "threat_intel_risks_prioritized",
            extra={
                "prioritized": len(prioritized_threats),
                "critical_threats": critical_count,
                "high_threats": high_count,
                "risk_score": overall_risk_score,
                "mitigation_count": len(mitigations),
            },
        )

        return {
            "current_node": "prioritize_risks",
            "prioritized_threats": prioritized_threats,
            "critical_count": critical_count,
            "high_count": high_count,
            "overall_risk_score": round(overall_risk_score, 1),
            "mitigations": mitigations,
        }

    # --- Node 4: Human Review ------------------------------------------------

    async def _node_human_review(
        self, state: ThreatIntelAgentState
    ) -> dict[str, Any]:
        """Node 4: Present advisory for human approval before distribution."""
        prioritized_threats = state.get("prioritized_threats", [])
        overall_risk_score = state.get("overall_risk_score", 0.0)

        logger.info(
            "threat_intel_human_review_pending",
            extra={
                "threat_count": len(prioritized_threats),
                "risk_score": overall_risk_score,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # --- Node 5: Report ------------------------------------------------------

    async def _node_report(
        self, state: ThreatIntelAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to threat_intelligence table and generate advisory."""
        now = datetime.now(timezone.utc).isoformat()
        threat_findings = state.get("threat_findings", [])
        prioritized_threats = state.get("prioritized_threats", [])
        mitigations = state.get("mitigations", [])
        overall_risk_score = state.get("overall_risk_score", 0.0)
        sources_queried = state.get("sources_queried", [])

        threats_saved = False

        # Save threat findings to database
        for tf in threat_findings:
            try:
                record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "threat_type": tf.get("threat_type", ""),
                    "title": tf.get("title", ""),
                    "description": tf.get("description", "")[:1000],
                    "cve_id": tf.get("cve_id", ""),
                    "cvss_score": tf.get("cvss_score", 0.0),
                    "severity": tf.get("severity", "low"),
                    "ioc_indicators": tf.get("ioc_indicators", []),
                    "affected_systems": tf.get("affected_systems", []),
                    "source_feed": tf.get("source_feed", ""),
                    "exploit_available": tf.get("exploit_available", False),
                    "actively_exploited": tf.get("actively_exploited", False),
                    "created_at": now,
                }
                self.db.client.table("threat_intelligence").insert(record).execute()
                threats_saved = True
            except Exception as e:
                logger.debug(f"Failed to save threat finding: {e}")

        # Build advisory report
        sections = [
            "# Threat Intelligence Advisory",
            f"*Generated: {now}*
",
            "## Summary",
            f"- **Feed Sources Queried:** {len(sources_queried)}",
            f"- **Threats Identified:** {state.get('threat_count', 0)}",
            f"- **CVEs Found:** {state.get('cve_count', 0)}",
            f"- **IOCs Extracted:** {state.get('ioc_count', 0)}",
            f"- **Critical Threats:** {state.get('critical_count', 0)}",
            f"- **High Threats:** {state.get('high_count', 0)}",
            f"- **Overall Risk Score:** {overall_risk_score:.1f}/10",
        ]

        if prioritized_threats:
            sections.append("
## Prioritized Threats")
            for pt in prioritized_threats[:10]:
                rank = pt.get("priority_rank", "?")
                risk = pt.get("risk_score", 0.0)
                # Find matching finding for title
                tid = pt.get("threat_id")
                title = ""
                for tf in threat_findings:
                    if tf.get("threat_id") == tid:
                        title = tf.get("title", "")
                        break
                if not title:
                    title = f"Threat #{tid}"
                sections.append(
                    f"{rank}. **[{risk:.1f}]** {title[:80]} "
                    f"-- {pt.get('exploitability', 'unknown')}"
                )

        if mitigations:
            sections.append("
## Mitigations")
            for i, m in enumerate(mitigations[:10], 1):
                effort = m.get("effort", "short_term").replace("_", " ").title()
                sections.append(
                    f"{i}. **[{effort}]** {m.get('mitigation', 'N/A')[:100]}"
                )

        if threat_findings:
            cve_findings = [tf for tf in threat_findings if tf.get("cve_id")]
            if cve_findings:
                sections.append("
## CVE Details")
                for tf in cve_findings[:8]:
                    cvss = tf.get("cvss_score", 0.0)
                    sev = tf.get("severity", "low").upper()
                    sections.append(
                        f"- **{tf.get('cve_id', 'N/A')}** [{sev}] "
                        f"CVSS {cvss:.1f} -- {tf.get('title', 'N/A')[:60]}"
                    )

        report = "
".join(sections)

        # Store insight
        if threat_findings:
            top_cve = ""
            for tf in threat_findings:
                if tf.get("cve_id"):
                    top_cve = tf.get("cve_id", "")
                    break

            self.store_insight(ThreatIntelData(
                threat_type="multi" if len(set(
                    tf.get("threat_type", "") for tf in threat_findings
                )) > 1 else threat_findings[0].get("threat_type", "vulnerability"),
                severity="critical" if state.get("critical_count", 0) > 0
                    else "high" if state.get("high_count", 0) > 0
                    else "medium",
                severity_score=overall_risk_score,
                cve_id=top_cve,
                ioc_indicators=[
                    ioc
                    for tf in threat_findings
                    for ioc in tf.get("ioc_indicators", [])
                ][:20],
                affected_systems=list(set(
                    sys
                    for tf in threat_findings
                    for sys in tf.get("affected_systems", [])
                ))[:15],
                mitigation=(
                    mitigations[0].get("mitigation", "") if mitigations else ""
                ),
                source_feed=", ".join(sources_queried[:5]),
                detected_at=now,
                metadata={
                    "threat_count": len(threat_findings),
                    "cve_count": state.get("cve_count", 0),
                    "ioc_count": state.get("ioc_count", 0),
                    "critical_count": state.get("critical_count", 0),
                    "high_count": state.get("high_count", 0),
                    "overall_risk_score": overall_risk_score,
                    "sources": sources_queried,
                },
            ))

        logger.info(
            "threat_intel_report_generated",
            extra={
                "threats_total": len(threat_findings),
                "prioritized_total": len(prioritized_threats),
                "mitigations_total": len(mitigations),
                "risk_score": overall_risk_score,
            },
        )

        return {
            "current_node": "report",
            "advisory_approved": state.get("human_approval_status") == "approved",
            "threats_saved": threats_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # --- Routing -------------------------------------------------------------

    @staticmethod
    def _route_after_review(state: ThreatIntelAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ThreatIntelAgent agent_id={self.agent_id\!r} "
            f"vertical={self.vertical_id\!r}>"
        )
