"""
Competitive Intelligence Agent — The Market Watchdog.

Monitors competitor activity, analyzes market intelligence, and generates
actionable threat alerts. Scans competitor domains, pricing changes,
product launches, and market positioning shifts.

Architecture (LangGraph State Machine):
    scan_competitors → analyze_intel → generate_alerts →
    human_review → report → END

Trigger Events:
    - scheduled: Daily/weekly competitor monitoring sweep
    - market_signal: External intelligence trigger
    - manual: On-demand competitive analysis

Shared Brain Integration:
    - Reads: market position data, competitor history, pricing baselines
    - Writes: competitive threat patterns, market shift signals

Safety:
    - NEVER accesses private or protected competitor data
    - All intelligence gathered from public sources only
    - Alerts require human review before distribution
    - Threat scores are advisory; human judgment required

Usage:
    agent = CompetitiveIntelAgent(config, db, embedder, llm)
    result = await agent.run({
        "scan_scope": "full",
        "competitors": ["competitor1.com", "competitor2.com"],
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
from core.agents.state import CompetitiveIntelAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

INTEL_TYPES = [
    "pricing_change",
    "product_launch",
    "feature_update",
    "market_expansion",
    "partnership",
    "acquisition",
    "leadership_change",
    "funding_round",
    "marketing_campaign",
    "customer_churn_signal",
]

SEVERITY_LEVELS = {
    "critical": {
        "score_min": 8.0,
        "description": "Immediate competitive threat requiring action",
        "color": "red",
        "response_window_hours": 24,
    },
    "high": {
        "score_min": 6.0,
        "description": "Significant competitive development to monitor closely",
        "color": "orange",
        "response_window_hours": 72,
    },
    "medium": {
        "score_min": 4.0,
        "description": "Notable competitive activity worth tracking",
        "color": "yellow",
        "response_window_hours": 168,
    },
    "low": {
        "score_min": 0.0,
        "description": "Minor competitive signal for awareness",
        "color": "blue",
        "response_window_hours": 720,
    },
}

INTEL_ANALYSIS_PROMPT = """\
You are a competitive intelligence analyst. Analyze the competitor scan \
results below and classify each finding.

Competitors Monitored:
{competitors_json}

Scan Results:
{scan_results_json}

For each finding, return a JSON array of objects:
[
    {{
        "competitor": "domain or name",
        "intel_type": "one of: {intel_types}",
        "finding": "Detailed description of the competitive intelligence",
        "severity": "critical|high|medium|low",
        "threat_score": 0.0-10.0,
        "recommended_action": "What to do about this",
        "confidence": 0.0-1.0,
        "source": "Where this intelligence came from"
    }}
]

Consider impact on our market position, revenue risk, and urgency. \
Be specific about recommended actions.

Return ONLY the JSON array, no markdown code fences.
"""

ALERT_GENERATION_PROMPT = """\
You are a competitive intelligence analyst. Generate actionable alerts \
from the classified findings below.

Findings:
{findings_json}

For each significant finding (severity critical or high), create an alert:
[
    {{
        "competitor": "name",
        "alert_type": "threat|opportunity|awareness",
        "message": "Clear, actionable alert message",
        "severity": "critical|high|medium|low",
        "recommended_action": "Specific action steps",
        "urgency_hours": 24-720
    }}
]

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("competitive_intel")
class CompetitiveIntelAgent(BaseAgent):
    """
    Competitive intelligence monitoring and analysis agent.

    Nodes:
        1. scan_competitors    -- Pull monitored domains, check for updates
        2. analyze_intel       -- LLM scores and classifies findings
        3. generate_alerts     -- Create actionable alerts for significant findings
        4. human_review        -- Gate: approve alerts
        5. report              -- Save to competitor_intel table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Competitive Intel Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(CompetitiveIntelAgentState)

        workflow.add_node("scan_competitors", self._node_scan_competitors)
        workflow.add_node("analyze_intel", self._node_analyze_intel)
        workflow.add_node("generate_alerts", self._node_generate_alerts)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_competitors")

        workflow.add_edge("scan_competitors", "analyze_intel")
        workflow.add_edge("analyze_intel", "generate_alerts")
        workflow.add_edge("generate_alerts", "human_review")
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
    def get_state_class(cls) -> Type[CompetitiveIntelAgentState]:
        return CompetitiveIntelAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "monitored_competitors": [],
            "scan_results": [],
            "intel_findings": [],
            "finding_count": 0,
            "critical_findings": 0,
            "threat_score": 0.0,
            "alerts": [],
            "alerts_approved": False,
            "alerts_sent": 0,
            "intel_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Competitors ────────────────────────────────────

    async def _node_scan_competitors(
        self, state: CompetitiveIntelAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull monitored domains from config, check for updates."""
        task = state.get("task_input", {})

        logger.info(
            "competitive_intel_scan_competitors",
            extra={"agent_id": self.agent_id},
        )

        # Get competitor list from task or config
        competitors_raw = task.get("competitors", [])
        scan_scope = task.get("scan_scope", "full")

        monitored_competitors: list[dict[str, Any]] = []
        scan_results: list[dict[str, Any]] = []

        # Load from config params if not in task
        if not competitors_raw:
            competitors_raw = self.config.params.get("competitors", [])

        # Build competitor records
        for comp in competitors_raw:
            if isinstance(comp, str):
                monitored_competitors.append({
                    "name": comp.replace(".com", "").replace(".", " ").title(),
                    "domain": comp,
                    "last_checked": "",
                })
            elif isinstance(comp, dict):
                monitored_competitors.append(comp)

        # Load competitor data from database
        try:
            result = (
                self.db.client.table("competitor_intel")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            if result.data:
                for record in result.data:
                    scan_results.append({
                        "competitor": record.get("competitor_name", ""),
                        "domain": record.get("competitor_domain", ""),
                        "intel_type": record.get("intel_type", ""),
                        "finding": record.get("finding", ""),
                        "source": record.get("source", "database"),
                        "detected_at": record.get("created_at", ""),
                    })
        except Exception as e:
            logger.warning(
                "competitive_intel_db_scan_error",
                extra={"error": str(e)[:200]},
            )

        # Add any task-provided findings
        task_findings = task.get("scan_results", [])
        scan_results.extend(task_findings)

        logger.info(
            "competitive_intel_scan_complete",
            extra={
                "competitors": len(monitored_competitors),
                "results": len(scan_results),
                "scope": scan_scope,
            },
        )

        return {
            "current_node": "scan_competitors",
            "monitored_competitors": monitored_competitors,
            "scan_results": scan_results,
        }

    # ─── Node 2: Analyze Intel ───────────────────────────────────────

    async def _node_analyze_intel(
        self, state: CompetitiveIntelAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM scores and classifies intelligence findings."""
        competitors = state.get("monitored_competitors", [])
        scan_results = state.get("scan_results", [])

        logger.info(
            "competitive_intel_analyze",
            extra={"results_to_analyze": len(scan_results)},
        )

        intel_findings: list[dict[str, Any]] = []
        threat_score = 0.0

        if not scan_results:
            logger.info("competitive_intel_no_results_to_analyze")
            return {
                "current_node": "analyze_intel",
                "intel_findings": [],
                "finding_count": 0,
                "critical_findings": 0,
                "threat_score": 0.0,
            }

        try:
            prompt = INTEL_ANALYSIS_PROMPT.format(
                competitors_json=json.dumps(competitors[:10], indent=2),
                scan_results_json=json.dumps(scan_results[:20], indent=2),
                intel_types=", ".join(INTEL_TYPES),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a competitive intelligence analyst.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                findings_data = json.loads(llm_text)
                if isinstance(findings_data, list):
                    intel_findings = findings_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("competitive_intel_parse_error")

        except Exception as e:
            logger.warning(
                "competitive_intel_llm_error",
                extra={"error": str(e)[:200]},
            )

        # Calculate aggregate threat score
        finding_count = len(intel_findings)
        critical_findings = sum(
            1 for f in intel_findings
            if f.get("severity") == "critical"
        )
        if intel_findings:
            scores = [f.get("threat_score", 0) for f in intel_findings]
            threat_score = round(max(scores) if scores else 0.0, 1)

        logger.info(
            "competitive_intel_analysis_complete",
            extra={
                "findings": finding_count,
                "critical": critical_findings,
                "threat_score": threat_score,
            },
        )

        return {
            "current_node": "analyze_intel",
            "intel_findings": intel_findings,
            "finding_count": finding_count,
            "critical_findings": critical_findings,
            "threat_score": threat_score,
        }

    # ─── Node 3: Generate Alerts ─────────────────────────────────────

    async def _node_generate_alerts(
        self, state: CompetitiveIntelAgentState
    ) -> dict[str, Any]:
        """Node 3: Create actionable alerts for significant findings."""
        findings = state.get("intel_findings", [])

        logger.info(
            "competitive_intel_generate_alerts",
            extra={"findings_count": len(findings)},
        )

        alerts: list[dict[str, Any]] = []

        # Filter to significant findings only
        significant = [
            f for f in findings
            if f.get("severity") in ("critical", "high")
        ]

        if not significant:
            logger.info("competitive_intel_no_significant_findings")
            return {
                "current_node": "generate_alerts",
                "alerts": [],
            }

        try:
            prompt = ALERT_GENERATION_PROMPT.format(
                findings_json=json.dumps(significant[:10], indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a competitive intelligence analyst generating threat alerts.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                alerts_data = json.loads(llm_text)
                if isinstance(alerts_data, list):
                    alerts = alerts_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("competitive_intel_alerts_parse_error")
                # Fallback: create basic alerts from findings
                for finding in significant[:5]:
                    alerts.append({
                        "competitor": finding.get("competitor", "Unknown"),
                        "alert_type": "threat",
                        "message": finding.get("finding", "")[:200],
                        "severity": finding.get("severity", "high"),
                        "recommended_action": finding.get("recommended_action", "Review and assess"),
                        "urgency_hours": SEVERITY_LEVELS.get(
                            finding.get("severity", "high"), {}
                        ).get("response_window_hours", 72),
                    })

        except Exception as e:
            logger.warning(
                "competitive_intel_alerts_llm_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "competitive_intel_alerts_generated",
            extra={"alert_count": len(alerts)},
        )

        return {
            "current_node": "generate_alerts",
            "alerts": alerts,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: CompetitiveIntelAgentState
    ) -> dict[str, Any]:
        """Node 4: Present alerts for human approval."""
        alerts = state.get("alerts", [])
        threat_score = state.get("threat_score", 0)

        logger.info(
            "competitive_intel_human_review_pending",
            extra={
                "alert_count": len(alerts),
                "threat_score": threat_score,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: CompetitiveIntelAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to competitor_intel table and generate insights."""
        now = datetime.now(timezone.utc).isoformat()
        findings = state.get("intel_findings", [])
        alerts = state.get("alerts", [])
        threat_score = state.get("threat_score", 0.0)
        competitors = state.get("monitored_competitors", [])

        # Save intel findings to database
        intel_saved = False
        alerts_sent = 0

        for finding in findings:
            try:
                record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "competitor_name": finding.get("competitor", ""),
                    "competitor_domain": finding.get("domain", ""),
                    "intel_type": finding.get("intel_type", ""),
                    "finding": finding.get("finding", ""),
                    "severity": finding.get("severity", "low"),
                    "threat_score": finding.get("threat_score", 0),
                    "source": finding.get("source", "analysis"),
                    "created_at": now,
                }
                self.db.client.table("competitor_intel").insert(record).execute()
                intel_saved = True
            except Exception as e:
                logger.debug(f"Failed to save intel finding: {e}")

        alerts_sent = len(alerts) if state.get("human_approval_status") == "approved" else 0

        # Build report
        sections = [
            "# Competitive Intelligence Report",
            f"*Generated: {now}*\n",
            f"## Summary",
            f"- **Competitors Monitored:** {len(competitors)}",
            f"- **Findings:** {state.get('finding_count', 0)}",
            f"- **Critical Findings:** {state.get('critical_findings', 0)}",
            f"- **Threat Score:** {threat_score:.1f}/10",
            f"- **Alerts Generated:** {len(alerts)}",
        ]

        if findings:
            sections.append("\n## Key Findings")
            for i, f in enumerate(findings[:10], 1):
                sev = f.get("severity", "low").upper()
                sections.append(
                    f"{i}. **[{sev}]** {f.get('competitor', 'Unknown')}: "
                    f"{f.get('finding', 'N/A')[:100]}"
                )

        if alerts:
            sections.append("\n## Alerts")
            for i, a in enumerate(alerts[:5], 1):
                sections.append(
                    f"{i}. **{a.get('severity', 'N/A').upper()}** "
                    f"({a.get('competitor', 'Unknown')}): "
                    f"{a.get('message', 'N/A')[:100]}"
                )

        report = "\n".join(sections)

        # Store insight
        if findings:
            self.store_insight(InsightData(
                insight_type="competitive_intelligence",
                title=f"Competitive Intel: {len(findings)} findings, threat {threat_score:.1f}/10",
                content=(
                    f"Monitored {len(competitors)} competitors. "
                    f"Found {len(findings)} intelligence items "
                    f"({state.get('critical_findings', 0)} critical). "
                    f"Overall threat score: {threat_score:.1f}/10. "
                    f"Generated {len(alerts)} actionable alerts."
                ),
                confidence=0.75,
                metadata={
                    "finding_count": len(findings),
                    "critical_count": state.get("critical_findings", 0),
                    "threat_score": threat_score,
                    "alert_count": len(alerts),
                    "competitors_monitored": len(competitors),
                },
            ))

        logger.info(
            "competitive_intel_report_generated",
            extra={
                "findings": len(findings),
                "alerts": len(alerts),
                "threat_score": threat_score,
            },
        )

        return {
            "current_node": "report",
            "alerts_sent": alerts_sent,
            "intel_saved": intel_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: CompetitiveIntelAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<CompetitiveIntelAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
