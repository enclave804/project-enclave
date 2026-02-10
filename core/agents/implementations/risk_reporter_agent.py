"""
Risk Reporter Agent — The Executive Briefer.

Aggregates all security assessment data (vulnerability scans, network
analysis, AppSec reviews, compliance mappings), quantifies risk in
dollar terms, and produces executive-ready reports with a risk matrix.

Architecture (LangGraph State Machine):
    aggregate_data -> quantify_risk -> generate_report ->
    human_review -> deliver -> report -> END

Trigger Events:
    - assessment_complete: Triggered after all domain assessments finish
    - manual: On-demand executive risk report
    - quarterly: Periodic board-level risk briefing

Shared Brain Integration:
    - Reads: all assessment findings, compliance gaps, attack surface data
    - Writes: executive risk scores, risk quantification, remediation ROI

Safety:
    - NEVER makes binding risk commitments -- advisory estimates only
    - All reports require human_review before delivery to clients
    - Dollar estimates are clearly labeled as projections, not guarantees
    - Aggregation preserves source attribution for audit trail

Usage:
    agent = RiskReporterAgent(config, db, embedder, llm)
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
from core.agents.state import RiskReporterAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

RISK_CATEGORIES = [
    "vulnerability_exploitation",
    "data_breach",
    "ransomware",
    "compliance_penalty",
    "business_disruption",
    "reputation_damage",
]

# Industry average breach cost multipliers (simplified)
BREACH_COST_MULTIPLIERS = {
    "healthcare": 10.93,     # $M per breach (IBM Cost of Data Breach 2024)
    "financial": 5.90,
    "technology": 4.97,
    "education": 3.65,
    "default": 4.45,
}

LIKELIHOOD_LABELS = {
    (0.0, 0.2): "rare",
    (0.2, 0.4): "unlikely",
    (0.4, 0.6): "possible",
    (0.6, 0.8): "likely",
    (0.8, 1.01): "almost_certain",
}

IMPACT_LABELS = {
    (0.0, 0.2): "negligible",
    (0.2, 0.4): "minor",
    (0.4, 0.6): "moderate",
    (0.6, 0.8): "major",
    (0.8, 1.01): "catastrophic",
}

RISK_REPORT_SYSTEM_PROMPT = """\
You are a senior cybersecurity risk advisor preparing an executive \
risk briefing for {company_name}. You translate technical security \
findings into business risk language that C-suite executives understand.

Given the aggregated security data below for {domain}, produce a JSON object with:
{{
    "executive_summary": "3-5 sentence summary for the board",
    "risk_score": 0.0-10.0,
    "risk_matrix": [
        {{
            "category": "Risk category name",
            "likelihood": 0.0-1.0,
            "impact": 0.0-1.0,
            "risk_level": "critical|high|medium|low",
            "dollar_exposure": estimated annual dollar impact
        }}
    ],
    "recommendations": [
        {{
            "priority": 1-10 (1=highest),
            "action": "What to do",
            "cost_estimate": "Estimated implementation cost",
            "risk_reduction": "How much risk this reduces (percentage)"
        }}
    ]
}}

Rules:
- Translate CVE scores and technical jargon into business impact
- Dollar estimates should reflect industry-specific breach costs
- Recommendations should have clear ROI justification
- Risk matrix must cover all six standard risk categories
- Be honest about uncertainties — use ranges, not false precision
- Industry context: {industry}

Return ONLY the JSON object, no markdown code fences.
"""


@register_agent_type("risk_reporter")
class RiskReporterAgent(BaseAgent):
    """
    Executive risk quantification and reporting agent.

    Nodes:
        1. aggregate_data    -- Pull findings from all assessment agents
        2. quantify_risk     -- Score risk 0-10, estimate dollar exposure
        3. generate_report   -- Create executive-ready report with risk matrix
        4. human_review      -- Gate: review report before delivery
        5. deliver           -- Save report and optionally send to client
        6. report            -- Summary + write insights to Hive Mind
    """

    def build_graph(self) -> Any:
        """Build the Risk Reporter's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(RiskReporterAgentState)

        workflow.add_node("aggregate_data", self._node_aggregate_data)
        workflow.add_node("quantify_risk", self._node_quantify_risk)
        workflow.add_node("generate_report", self._node_generate_report)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("deliver", self._node_deliver)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("aggregate_data")

        workflow.add_edge("aggregate_data", "quantify_risk")
        workflow.add_edge("quantify_risk", "generate_report")
        workflow.add_edge("generate_report", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "deliver",
                "rejected": "report",
            },
        )
        workflow.add_edge("deliver", "report")
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
    def get_state_class(cls) -> Type[RiskReporterAgentState]:
        return RiskReporterAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "company_domain": task.get("company_domain", ""),
            "company_name": task.get("company_name", ""),
            "all_findings": [],
            "compliance_data": {},
            "network_data": {},
            "overall_risk_score": 0.0,
            "risk_by_category": {},
            "estimated_breach_cost": 0.0,
            "annualized_loss_expectancy": 0.0,
            "executive_summary": "",
            "detailed_sections": [],
            "risk_matrix": {},
            "priority_actions": [],
            "report_format": task.get("report_format", "markdown"),
            "report_approved": False,
            "report_delivered": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Aggregate Data ──────────────────────────────────────

    async def _node_aggregate_data(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Pull and aggregate findings from all assessment agents.

        Queries security_assessments and security_findings tables for all
        data related to the target domain, including vulnerability scans,
        network analysis, AppSec reviews, and compliance mappings.
        """
        domain = state.get("company_domain", "")

        logger.info("risk_aggregate: collecting data for %s", domain)

        all_findings: list[dict[str, Any]] = []
        compliance_data: dict[str, Any] = {}
        network_data: dict[str, Any] = {}
        assessment_ids: list[str] = []

        # --- Security Findings ---
        try:
            result = (
                self.db.client.table("security_findings")
                .select("*")
                .eq("domain", domain)
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .limit(200)
                .execute()
            )
            all_findings = result.data or []
        except Exception as e:
            logger.debug(f"Failed to load security findings: {e}")

        # --- Assessments (for metadata) ---
        try:
            result = (
                self.db.client.table("security_assessments")
                .select("*")
                .eq("domain", domain)
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .execute()
            )
            assessments = result.data or []
            assessment_ids = [a.get("id", "") for a in assessments if a.get("id")]

            # Extract compliance data
            for a in assessments:
                if a.get("assessment_type") == "compliance_mapping":
                    compliance_data = {
                        "overall_score": a.get("compliance_score", 0.0),
                        "frameworks": a.get("frameworks_assessed", []),
                        "gap_count": a.get("gap_count", 0),
                    }
                elif a.get("assessment_type") in ("network_analysis", "vulnerability_scan"):
                    if not network_data:
                        network_data = {
                            "risk_score": a.get("risk_score", 0.0),
                            "finding_count": a.get("finding_count", 0),
                        }
        except Exception as e:
            logger.debug(f"Failed to load assessments: {e}")

        # Consult shared brain for additional context
        try:
            insights = self.consult_hive(
                f"Security assessment findings for {domain}",
                min_confidence=0.7,
                limit=5,
            )
            if insights:
                for insight in insights:
                    content = insight.get("content", "")
                    if content:
                        all_findings.append({
                            "source": "hive_mind",
                            "severity": "info",
                            "title": insight.get("title", "Hive Mind Insight"),
                            "description": content,
                        })
        except Exception:
            pass

        logger.info(
            "risk_aggregate_complete: %d findings, %d assessments",
            len(all_findings), len(assessment_ids),
        )

        return {
            "current_node": "aggregate_data",
            "all_findings": all_findings,
            "compliance_data": compliance_data,
            "network_data": network_data,
            "assessment_id": assessment_ids[0] if assessment_ids else "",
        }

    # ─── Node 2: Quantify Risk ──────────────────────────────────────

    async def _node_quantify_risk(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Score risk 0-10 and estimate dollar exposure.

        Computes risk by category using finding severity, industry
        breach cost data, and annualized loss expectancy (ALE).
        """
        all_findings = state.get("all_findings", [])
        compliance_data = state.get("compliance_data", {})
        domain = state.get("company_domain", "")

        logger.info("risk_quantify: scoring %d findings", len(all_findings))

        industry = self.config.params.get("industry", "default")
        breach_multiplier = BREACH_COST_MULTIPLIERS.get(
            industry, BREACH_COST_MULTIPLIERS["default"]
        )

        # Count findings by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in all_findings:
            sev = f.get("severity", "info")
            if sev in severity_counts:
                severity_counts[sev] += 1

        # Compute overall risk score (0-10)
        overall_risk = min(10.0, (
            severity_counts["critical"] * 2.5
            + severity_counts["high"] * 1.5
            + severity_counts["medium"] * 0.5
            + severity_counts["low"] * 0.1
        ))

        # Risk by category
        risk_by_category: dict[str, float] = {}
        for cat in RISK_CATEGORIES:
            # Simplified: derive from overall risk with category weighting
            if cat == "vulnerability_exploitation":
                risk_by_category[cat] = min(10.0, overall_risk * 1.2)
            elif cat == "data_breach":
                risk_by_category[cat] = min(10.0, overall_risk * 1.0)
            elif cat == "ransomware":
                risk_by_category[cat] = min(10.0, overall_risk * 0.8)
            elif cat == "compliance_penalty":
                comp_score = compliance_data.get("overall_score", 50)
                risk_by_category[cat] = min(10.0, (100 - comp_score) / 10)
            elif cat == "business_disruption":
                risk_by_category[cat] = min(10.0, overall_risk * 0.6)
            elif cat == "reputation_damage":
                risk_by_category[cat] = min(10.0, overall_risk * 0.5)

        # Estimate breach cost
        # Single Loss Expectancy (SLE) = breach multiplier * $1M
        sle = breach_multiplier * 1_000_000

        # Annual Rate of Occurrence (ARO) based on risk score
        # Risk 10 = ~80% chance, Risk 0 = ~2% chance
        aro = min(0.8, max(0.02, overall_risk / 12.5))

        # Annualized Loss Expectancy (ALE) = SLE * ARO
        ale = sle * aro

        logger.info(
            "risk_quantify_complete: risk=%.1f, ALE=$%.0f, breach_cost=$%.0fM",
            overall_risk, ale, breach_multiplier,
        )

        return {
            "current_node": "quantify_risk",
            "overall_risk_score": round(overall_risk, 1),
            "risk_by_category": {k: round(v, 1) for k, v in risk_by_category.items()},
            "estimated_breach_cost": round(sle, 2),
            "annualized_loss_expectancy": round(ale, 2),
        }

    # ─── Node 3: Generate Report ──────────────────────────────────

    async def _node_generate_report(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Create executive-ready report with risk matrix.

        Uses LLM to generate an executive summary, risk matrix,
        and prioritized recommendations in business language.
        """
        domain = state.get("company_domain", "")
        all_findings = state.get("all_findings", [])
        risk_score = state.get("overall_risk_score", 0.0)
        risk_by_cat = state.get("risk_by_category", {})
        ale = state.get("annualized_loss_expectancy", 0.0)

        logger.info("risk_report: generating executive report for %s", domain)

        company_name = state.get("company_name", "") or self.config.params.get("company_name", "the organization")
        industry = self.config.params.get("industry", "technology")

        executive_summary = ""
        risk_matrix: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        # Generate via LLM
        try:
            # Summarize findings for LLM context
            finding_summary = {
                "total_findings": len(all_findings),
                "critical": sum(1 for f in all_findings if f.get("severity") == "critical"),
                "high": sum(1 for f in all_findings if f.get("severity") == "high"),
                "medium": sum(1 for f in all_findings if f.get("severity") == "medium"),
                "low": sum(1 for f in all_findings if f.get("severity") == "low"),
                "top_findings": [
                    {"title": f.get("title", ""), "severity": f.get("severity", "")}
                    for f in all_findings[:10]
                    if f.get("severity") in ("critical", "high")
                ],
            }

            prompt = (
                f"Risk data for {domain} ({company_name}):\n"
                f"Overall risk: {risk_score}/10\n"
                f"Risk by category: {json.dumps(risk_by_cat)}\n"
                f"Annualized loss expectancy: ${ale:,.0f}\n"
                f"Findings: {json.dumps(finding_summary, indent=2)}\n"
                f"Compliance data: {json.dumps(state.get('compliance_data', {}))}\n\n"
                f"Generate the executive risk briefing JSON."
            )
            system = RISK_REPORT_SYSTEM_PROMPT.format(
                company_name=company_name,
                domain=domain,
                industry=industry,
            )

            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
            text = response.content[0].text.strip() if response.content else ""
            if text:
                try:
                    parsed = json.loads(text)
                    executive_summary = parsed.get("executive_summary", "")
                    risk_matrix = parsed.get("risk_matrix", [])
                    recommendations = parsed.get("recommendations", [])
                except (json.JSONDecodeError, ValueError):
                    executive_summary = text[:500]
        except Exception as e:
            logger.debug(f"LLM report generation failed: {e}")

        # Fallback executive summary
        if not executive_summary:
            executive_summary = (
                f"Security assessment of {domain} reveals an overall risk score "
                f"of {risk_score:.1f}/10. We identified {len(all_findings)} security "
                f"findings across vulnerability scanning, network analysis, and "
                f"application security review. The estimated annualized loss "
                f"expectancy is ${ale:,.0f}. Immediate action is recommended for "
                f"all critical and high severity findings."
            )

        # Fallback risk matrix
        if not risk_matrix:
            for cat, cat_score in risk_by_cat.items():
                likelihood = min(1.0, cat_score / 10)
                impact = min(1.0, cat_score / 8)
                risk_level = "critical" if cat_score >= 8 else "high" if cat_score >= 6 else "medium" if cat_score >= 3 else "low"
                dollar_exposure = ale * (cat_score / max(sum(risk_by_cat.values()), 1))

                risk_matrix.append({
                    "category": cat.replace("_", " ").title(),
                    "likelihood": round(likelihood, 2),
                    "impact": round(impact, 2),
                    "risk_level": risk_level,
                    "dollar_exposure": round(dollar_exposure, 0),
                })

        # Build report sections
        now = datetime.now(timezone.utc).isoformat()
        sections = [
            {
                "title": "Executive Summary",
                "content": executive_summary,
                "order": 1,
            },
            {
                "title": "Risk Overview",
                "content": (
                    f"Overall Risk Score: {risk_score:.1f}/10\n"
                    f"Annualized Loss Expectancy: ${ale:,.0f}\n"
                    f"Total Findings: {len(all_findings)}"
                ),
                "order": 2,
            },
            {
                "title": "Risk Matrix",
                "content": json.dumps(risk_matrix, indent=2),
                "order": 3,
            },
            {
                "title": "Recommendations",
                "content": json.dumps(recommendations, indent=2),
                "order": 4,
            },
        ]

        logger.info(
            "risk_report_complete: %d sections, %d matrix entries, %d recs",
            len(sections), len(risk_matrix), len(recommendations),
        )

        return {
            "current_node": "generate_report",
            "executive_summary": executive_summary,
            "detailed_sections": sections,
            "risk_matrix": {"entries": risk_matrix, "generated_at": now},
            "priority_actions": recommendations,
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """Node 4: Present executive risk report for human approval."""
        logger.info(
            "risk_human_review_pending: risk=%.1f, ALE=$%.0f",
            state.get("overall_risk_score", 0.0),
            state.get("annualized_loss_expectancy", 0.0),
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Deliver ──────────────────────────────────────────

    async def _node_deliver(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Save the approved report and mark as delivered.

        Persists the executive report to the database and optionally
        triggers delivery (email, PDF generation, dashboard update).
        """
        domain = state.get("company_domain", "")
        now = datetime.now(timezone.utc).isoformat()

        logger.info("risk_deliver: saving report for %s", domain)

        delivered = False

        # Save report to DB
        try:
            self.db.client.table("security_assessments").insert({
                "vertical_id": self.vertical_id,
                "domain": domain,
                "assessment_type": "executive_risk_report",
                "risk_score": state.get("overall_risk_score", 0.0),
                "executive_summary": state.get("executive_summary", ""),
                "finding_count": len(state.get("all_findings", [])),
                "status": "delivered",
                "metadata": {
                    "annualized_loss_expectancy": state.get("annualized_loss_expectancy", 0.0),
                    "estimated_breach_cost": state.get("estimated_breach_cost", 0.0),
                    "risk_by_category": state.get("risk_by_category", {}),
                    "report_format": state.get("report_format", "markdown"),
                },
                "created_at": now,
            }).execute()
            delivered = True
        except Exception as e:
            logger.debug(f"Failed to save risk report: {e}")

        # Publish insight to Hive Mind
        self.store_insight(InsightData(
            insight_type="executive_risk_report",
            title=f"Risk Report: {domain} — {state.get('overall_risk_score', 0.0):.1f}/10",
            content=(
                f"Executive risk report for {domain}: risk score "
                f"{state.get('overall_risk_score', 0.0):.1f}/10, "
                f"ALE ${state.get('annualized_loss_expectancy', 0.0):,.0f}, "
                f"{len(state.get('all_findings', []))} total findings."
            ),
            confidence=0.90,
            metadata={
                "domain": domain,
                "risk_score": state.get("overall_risk_score", 0.0),
                "ale": state.get("annualized_loss_expectancy", 0.0),
            },
        ))

        return {
            "current_node": "deliver",
            "report_delivered": delivered,
            "report_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: RiskReporterAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate risk reporter activity summary."""
        now = datetime.now(timezone.utc).isoformat()
        domain = state.get("company_domain", "")
        risk_score = state.get("overall_risk_score", 0.0)
        ale = state.get("annualized_loss_expectancy", 0.0)
        risk_by_cat = state.get("risk_by_category", {})

        sections = [
            "# Executive Risk Report",
            f"**Domain:** {domain}",
            f"*Generated: {now}*\n",
            "## Overall Risk Score",
            f"**{risk_score:.1f} / 10.0**\n",
            "## Financial Impact",
            f"- **Estimated Breach Cost:** ${state.get('estimated_breach_cost', 0.0):,.0f}",
            f"- **Annualized Loss Expectancy:** ${ale:,.0f}",
            f"- **Total Findings:** {len(state.get('all_findings', []))}",
        ]

        # Risk by category
        if risk_by_cat:
            sections.append("\n## Risk by Category")
            for cat, score in sorted(risk_by_cat.items(), key=lambda x: -x[1]):
                label = cat.replace("_", " ").title()
                bar = "#" * int(score)
                sections.append(f"- **{label}:** {score:.1f}/10 {bar}")

        # Executive summary
        summary = state.get("executive_summary", "")
        if summary:
            sections.append(f"\n## Executive Summary\n{summary}")

        # Top recommendations
        recs = state.get("priority_actions", [])
        if recs:
            sections.append(f"\n## Top Recommendations ({min(5, len(recs))})")
            for i, rec in enumerate(recs[:5], 1):
                action = rec.get("action", "") if isinstance(rec, dict) else str(rec)
                sections.append(f"{i}. {action}")

        # Delivery status
        if state.get("report_delivered"):
            sections.append("\n## Delivery Status: DELIVERED")
        else:
            sections.append("\n## Delivery Status: NOT DELIVERED (rejected or pending)")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: RiskReporterAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── Knowledge ───────────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<RiskReporterAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
