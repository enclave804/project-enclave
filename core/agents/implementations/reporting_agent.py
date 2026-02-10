"""
Reporting Agent — The Business Intelligence Engine.

Collects metrics from pipeline, revenue, outreach, and client data sources.
Analyzes trends, generates forecasts, identifies anomalies, and produces
comprehensive business analytics reports.

Architecture (LangGraph State Machine):
    collect_metrics → analyze_trends → generate_report →
    human_review → report → END

Trigger Events:
    - scheduled: Weekly/monthly report generation
    - report_request: Manual report generation request
    - manual: On-demand business analytics

Shared Brain Integration:
    - Reads: all agent metrics, pipeline data, revenue streams
    - Writes: trend patterns, forecast accuracy, anomaly patterns

Safety:
    - NEVER exposes raw customer PII in reports
    - Financial data is aggregated, not individual transactions
    - Reports require human review before distribution
    - Forecasts are clearly marked as estimates with confidence levels

Usage:
    agent = ReportingAgent(config, db, embedder, llm)
    result = await agent.run({
        "report_format": "executive",
        "period_start": "2024-01-01",
        "period_end": "2024-01-31",
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
from core.agents.state import ReportingAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

REPORT_SECTIONS = [
    "executive_summary",
    "pipeline_overview",
    "revenue_analysis",
    "outreach_performance",
    "client_health",
    "trends_and_forecasts",
    "anomalies",
    "recommendations",
]

METRIC_QUERIES = {
    "pipeline": {
        "table": "opportunities",
        "metrics": ["total_value", "stage_distribution", "conversion_rate", "avg_deal_size"],
        "description": "Sales pipeline health and deal flow metrics.",
    },
    "revenue": {
        "table": "invoices",
        "metrics": ["total_revenue", "mrr", "arr", "accounts_receivable", "churn_rate"],
        "description": "Revenue streams, recurring revenue, and financial health.",
    },
    "outreach": {
        "table": "outreach_events",
        "metrics": ["emails_sent", "open_rate", "reply_rate", "meetings_booked", "conversion_rate"],
        "description": "Email outreach performance and engagement metrics.",
    },
    "client": {
        "table": "clients",
        "metrics": ["total_clients", "new_clients", "at_risk_clients", "nps_score", "retention_rate"],
        "description": "Client health, satisfaction, and retention metrics.",
    },
}

REPORT_FORMATS = {
    "executive": {
        "label": "Executive Summary",
        "sections": ["executive_summary", "revenue_analysis", "pipeline_overview", "recommendations"],
        "max_length": 2000,
    },
    "detailed": {
        "label": "Detailed Report",
        "sections": REPORT_SECTIONS,
        "max_length": 5000,
    },
    "weekly": {
        "label": "Weekly Digest",
        "sections": ["executive_summary", "outreach_performance", "pipeline_overview"],
        "max_length": 1500,
    },
    "monthly": {
        "label": "Monthly Report",
        "sections": REPORT_SECTIONS,
        "max_length": 4000,
    },
}

TREND_ANALYSIS_PROMPT = """\
You are a business analyst. Analyze the metrics below and identify:
1. Key trends (improving, declining, stable)
2. Forecasts for next period
3. Anomalies that require attention

Metrics Data:
{metrics_json}

Period: {period_start} to {period_end}

Return a JSON object:
{{
    "trends": [
        {{"metric": "name", "direction": "up|down|stable", "magnitude": 0.0-1.0, \
"insight": "description"}}
    ],
    "forecasts": [
        {{"metric": "name", "forecast_value": 0, "confidence": 0.0-1.0}}
    ],
    "anomalies": [
        {{"metric": "name", "expected": 0, "actual": 0, "severity": "high|medium|low"}}
    ],
    "executive_summary": "2-3 sentence overview of business health"
}}

Return ONLY the JSON object, no markdown code fences.
"""

REPORT_GENERATION_PROMPT = """\
You are a business analytics expert. Generate a comprehensive {report_format} \
report in Markdown format from the data below.

Report Period: {period_start} to {period_end}

Pipeline Metrics:
{pipeline_json}

Revenue Metrics:
{revenue_json}

Outreach Metrics:
{outreach_json}

Client Metrics:
{client_json}

Trends Analysis:
{trends_json}

Anomalies:
{anomalies_json}

Sections to include: {sections}

Guidelines:
- Use clear headers and bullet points
- Include percentage changes where available
- Highlight wins and areas needing attention
- End with actionable recommendations
- Maximum length: {max_length} words

Generate the Markdown report directly, no code fences.
"""


@register_agent_type("reporting")
class ReportingAgent(BaseAgent):
    """
    Business analytics and reporting agent.

    Nodes:
        1. collect_metrics   -- Pull data from pipeline, revenue, outreach, client tables
        2. analyze_trends    -- LLM identifies patterns and forecasts
        3. generate_report   -- Create formatted report with charts data
        4. human_review      -- Gate: approve report
        5. report            -- Save to business_reports table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Reporting Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ReportingAgentState)

        workflow.add_node("collect_metrics", self._node_collect_metrics)
        workflow.add_node("analyze_trends", self._node_analyze_trends)
        workflow.add_node("generate_report", self._node_generate_report)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("collect_metrics")

        workflow.add_edge("collect_metrics", "analyze_trends")
        workflow.add_edge("analyze_trends", "generate_report")
        workflow.add_edge("generate_report", "human_review")
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
    def get_state_class(cls) -> Type[ReportingAgentState]:
        return ReportingAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "pipeline_metrics": {},
            "revenue_metrics": {},
            "outreach_metrics": {},
            "client_metrics": {},
            "period_start": "",
            "period_end": "",
            "trends": [],
            "forecasts": [],
            "anomalies": [],
            "report_format": "executive",
            "report_sections": [],
            "report_document": "",
            "report_id": "",
            "report_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Collect Metrics ─────────────────────────────────────

    async def _node_collect_metrics(
        self, state: ReportingAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull data from pipeline, revenue, outreach, and client tables."""
        task = state.get("task_input", {})
        report_format = task.get("report_format", "executive")
        period_start = task.get("period_start", "")
        period_end = task.get("period_end", "")

        logger.info(
            "reporting_collect_metrics",
            extra={
                "report_format": report_format,
                "agent_id": self.agent_id,
            },
        )

        if not period_start:
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1).strftime("%Y-%m-%d")
            period_end = now.strftime("%Y-%m-%d")

        pipeline_metrics: dict[str, Any] = {
            "total_opportunities": 0,
            "total_value_cents": 0,
            "stages": {},
            "avg_deal_size_cents": 0,
            "won_count": 0,
            "lost_count": 0,
        }
        revenue_metrics: dict[str, Any] = {
            "total_revenue_cents": 0,
            "paid_invoices": 0,
            "outstanding_cents": 0,
            "overdue_cents": 0,
        }
        outreach_metrics: dict[str, Any] = {
            "emails_sent": 0,
            "emails_opened": 0,
            "replies_received": 0,
            "meetings_booked": 0,
        }
        client_metrics: dict[str, Any] = {
            "total_clients": 0,
            "new_clients": 0,
            "at_risk_clients": 0,
            "active_clients": 0,
        }

        # ── Pipeline Metrics ──
        try:
            result = (
                self.db.client.table("opportunities")
                .select("*")
                .gte("created_at", period_start)
                .execute()
            )
            if result.data:
                opps = result.data
                pipeline_metrics["total_opportunities"] = len(opps)
                total_val = sum(o.get("value_cents", 0) for o in opps)
                pipeline_metrics["total_value_cents"] = total_val
                if opps:
                    pipeline_metrics["avg_deal_size_cents"] = total_val // len(opps)

                # Stage distribution
                stages: dict[str, int] = {}
                won = 0
                lost = 0
                for opp in opps:
                    stage = opp.get("stage", "unknown")
                    stages[stage] = stages.get(stage, 0) + 1
                    if stage == "closed_won":
                        won += 1
                    elif stage == "closed_lost":
                        lost += 1
                pipeline_metrics["stages"] = stages
                pipeline_metrics["won_count"] = won
                pipeline_metrics["lost_count"] = lost
        except Exception as e:
            logger.warning(f"reporting_pipeline_error: {str(e)[:200]}")

        # ── Revenue Metrics ──
        try:
            result = (
                self.db.client.table("invoices")
                .select("*")
                .gte("created_at", period_start)
                .execute()
            )
            if result.data:
                invoices = result.data
                paid = [i for i in invoices if i.get("status") == "paid"]
                overdue = [i for i in invoices if i.get("status") == "overdue"]
                outstanding = [i for i in invoices if i.get("status") in ("open", "sent")]

                revenue_metrics["total_revenue_cents"] = sum(
                    i.get("total_amount_cents", 0) for i in paid
                )
                revenue_metrics["paid_invoices"] = len(paid)
                revenue_metrics["outstanding_cents"] = sum(
                    i.get("total_amount_cents", 0) for i in outstanding
                )
                revenue_metrics["overdue_cents"] = sum(
                    i.get("total_amount_cents", 0) for i in overdue
                )
        except Exception as e:
            logger.warning(f"reporting_revenue_error: {str(e)[:200]}")

        # ── Outreach Metrics ──
        try:
            result = (
                self.db.client.table("outreach_events")
                .select("*")
                .gte("created_at", period_start)
                .execute()
            )
            if result.data:
                events = result.data
                outreach_metrics["emails_sent"] = sum(
                    1 for e in events if e.get("event_type") == "sent"
                )
                outreach_metrics["emails_opened"] = sum(
                    1 for e in events if e.get("event_type") == "opened"
                )
                outreach_metrics["replies_received"] = sum(
                    1 for e in events if e.get("event_type") == "replied"
                )
                outreach_metrics["meetings_booked"] = sum(
                    1 for e in events if e.get("event_type") == "meeting_booked"
                )
        except Exception as e:
            logger.warning(f"reporting_outreach_error: {str(e)[:200]}")

        # ── Client Metrics ──
        try:
            result = (
                self.db.client.table("clients")
                .select("*")
                .execute()
            )
            if result.data:
                clients = result.data
                client_metrics["total_clients"] = len(clients)
                client_metrics["active_clients"] = sum(
                    1 for c in clients if c.get("status") == "active"
                )
                client_metrics["at_risk_clients"] = sum(
                    1 for c in clients if c.get("status") == "at_risk"
                )
                client_metrics["new_clients"] = sum(
                    1 for c in clients
                    if c.get("onboarded_at", "") >= period_start
                )
        except Exception as e:
            logger.warning(f"reporting_client_error: {str(e)[:200]}")

        logger.info(
            "reporting_metrics_collected",
            extra={
                "pipeline_opps": pipeline_metrics["total_opportunities"],
                "revenue_cents": revenue_metrics["total_revenue_cents"],
                "outreach_sent": outreach_metrics["emails_sent"],
                "total_clients": client_metrics["total_clients"],
            },
        )

        return {
            "current_node": "collect_metrics",
            "pipeline_metrics": pipeline_metrics,
            "revenue_metrics": revenue_metrics,
            "outreach_metrics": outreach_metrics,
            "client_metrics": client_metrics,
            "period_start": period_start,
            "period_end": period_end,
            "report_format": report_format,
        }

    # ─── Node 2: Analyze Trends ──────────────────────────────────────

    async def _node_analyze_trends(
        self, state: ReportingAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM identifies patterns, forecasts, and anomalies."""
        pipeline = state.get("pipeline_metrics", {})
        revenue = state.get("revenue_metrics", {})
        outreach = state.get("outreach_metrics", {})
        client = state.get("client_metrics", {})
        period_start = state.get("period_start", "")
        period_end = state.get("period_end", "")

        logger.info("reporting_analyze_trends")

        all_metrics = {
            "pipeline": pipeline,
            "revenue": revenue,
            "outreach": outreach,
            "client": client,
        }

        trends: list[dict[str, Any]] = []
        forecasts: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []

        try:
            prompt = TREND_ANALYSIS_PROMPT.format(
                metrics_json=json.dumps(all_metrics, indent=2),
                period_start=period_start,
                period_end=period_end,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a business analytics expert.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                analysis = json.loads(llm_text)
                trends = analysis.get("trends", [])
                forecasts = analysis.get("forecasts", [])
                anomalies = analysis.get("anomalies", [])
            except (json.JSONDecodeError, KeyError):
                logger.debug("reporting_trend_parse_error")

        except Exception as e:
            logger.warning(
                "reporting_trend_llm_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "reporting_trends_analyzed",
            extra={
                "trends": len(trends),
                "forecasts": len(forecasts),
                "anomalies": len(anomalies),
            },
        )

        return {
            "current_node": "analyze_trends",
            "trends": trends,
            "forecasts": forecasts,
            "anomalies": anomalies,
        }

    # ─── Node 3: Generate Report ─────────────────────────────────────

    async def _node_generate_report(
        self, state: ReportingAgentState
    ) -> dict[str, Any]:
        """Node 3: Create formatted report with data sections."""
        report_format = state.get("report_format", "executive")
        format_config = REPORT_FORMATS.get(report_format, REPORT_FORMATS["executive"])

        logger.info(
            "reporting_generate_report",
            extra={"format": report_format},
        )

        report_document = ""

        try:
            prompt = REPORT_GENERATION_PROMPT.format(
                report_format=format_config["label"],
                period_start=state.get("period_start", ""),
                period_end=state.get("period_end", ""),
                pipeline_json=json.dumps(state.get("pipeline_metrics", {}), indent=2),
                revenue_json=json.dumps(state.get("revenue_metrics", {}), indent=2),
                outreach_json=json.dumps(state.get("outreach_metrics", {}), indent=2),
                client_json=json.dumps(state.get("client_metrics", {}), indent=2),
                trends_json=json.dumps(state.get("trends", []), indent=2),
                anomalies_json=json.dumps(state.get("anomalies", []), indent=2),
                sections=", ".join(
                    s.replace("_", " ").title() for s in format_config["sections"]
                ),
                max_length=format_config["max_length"],
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a business analytics expert generating reports.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )

            report_document = llm_response.content[0].text.strip()

        except Exception as e:
            logger.warning(
                "reporting_generate_llm_error",
                extra={"error": str(e)[:200]},
            )
            # Fallback report
            pipeline = state.get("pipeline_metrics", {})
            revenue = state.get("revenue_metrics", {})
            report_document = (
                f"# Business Report — {format_config['label']}\n\n"
                f"**Period:** {state.get('period_start', '')} to {state.get('period_end', '')}\n\n"
                f"## Pipeline\n"
                f"- Opportunities: {pipeline.get('total_opportunities', 0)}\n"
                f"- Total Value: ${pipeline.get('total_value_cents', 0) / 100:.2f}\n\n"
                f"## Revenue\n"
                f"- Total: ${revenue.get('total_revenue_cents', 0) / 100:.2f}\n"
                f"- Outstanding: ${revenue.get('outstanding_cents', 0) / 100:.2f}\n"
            )

        # Build report sections metadata
        report_sections: list[dict[str, Any]] = []
        for section_key in format_config["sections"]:
            report_sections.append({
                "title": section_key.replace("_", " ").title(),
                "key": section_key,
            })

        logger.info(
            "reporting_report_generated",
            extra={
                "format": report_format,
                "length_chars": len(report_document),
            },
        )

        return {
            "current_node": "generate_report",
            "report_document": report_document,
            "report_sections": report_sections,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: ReportingAgentState
    ) -> dict[str, Any]:
        """Node 4: Present report for human approval."""
        report_format = state.get("report_format", "executive")
        anomalies = state.get("anomalies", [])

        logger.info(
            "reporting_human_review_pending",
            extra={
                "format": report_format,
                "anomaly_count": len(anomalies),
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: ReportingAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to business_reports table and generate insights."""
        now = datetime.now(timezone.utc).isoformat()
        report_format = state.get("report_format", "executive")
        pipeline = state.get("pipeline_metrics", {})
        revenue = state.get("revenue_metrics", {})

        # Save report to database
        report_id = ""
        report_saved = False

        try:
            report_record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "report_format": report_format,
                "period_start": state.get("period_start", ""),
                "period_end": state.get("period_end", ""),
                "report_document": state.get("report_document", ""),
                "pipeline_metrics": json.dumps(pipeline),
                "revenue_metrics": json.dumps(revenue),
                "outreach_metrics": json.dumps(state.get("outreach_metrics", {})),
                "client_metrics": json.dumps(state.get("client_metrics", {})),
                "trends": json.dumps(state.get("trends", [])),
                "forecasts": json.dumps(state.get("forecasts", [])),
                "anomalies": json.dumps(state.get("anomalies", [])),
                "status": "approved" if state.get("human_approval_status") == "approved" else "draft",
                "created_at": now,
            }

            result = (
                self.db.client.table("business_reports")
                .insert(report_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                report_id = result.data[0].get("id", "")
                report_saved = True
                logger.info(
                    "reporting_report_saved",
                    extra={"report_id": report_id},
                )
        except Exception as e:
            logger.warning(
                "reporting_save_error",
                extra={"error": str(e)[:200]},
            )

        # Build summary
        sections = [
            "# Report Generation Summary",
            f"*Generated: {now}*\n",
            f"## Report Details",
            f"- **Format:** {report_format}",
            f"- **Period:** {state.get('period_start', '')} to {state.get('period_end', '')}",
            f"- **Report ID:** {report_id or 'N/A'}",
            f"- **Saved:** {'Yes' if report_saved else 'No'}",
            f"\n## Key Metrics",
            f"- Pipeline: {pipeline.get('total_opportunities', 0)} opportunities "
            f"(${pipeline.get('total_value_cents', 0) / 100:.2f})",
            f"- Revenue: ${revenue.get('total_revenue_cents', 0) / 100:.2f}",
            f"- Trends Identified: {len(state.get('trends', []))}",
            f"- Anomalies Found: {len(state.get('anomalies', []))}",
        ]

        report_summary = "\n".join(sections)

        # Store insight
        total_revenue = revenue.get("total_revenue_cents", 0)
        total_pipeline = pipeline.get("total_value_cents", 0)

        self.store_insight(InsightData(
            insight_type="business_analytics",
            title=f"Report: {report_format} ({state.get('period_start', '')} to {state.get('period_end', '')})",
            content=(
                f"Generated {report_format} report. "
                f"Pipeline: {pipeline.get('total_opportunities', 0)} opportunities "
                f"worth ${total_pipeline / 100:.2f}. "
                f"Revenue: ${total_revenue / 100:.2f}. "
                f"Identified {len(state.get('trends', []))} trends and "
                f"{len(state.get('anomalies', []))} anomalies."
            ),
            confidence=0.80,
            metadata={
                "report_format": report_format,
                "total_pipeline_cents": total_pipeline,
                "total_revenue_cents": total_revenue,
                "trend_count": len(state.get("trends", [])),
                "anomaly_count": len(state.get("anomalies", [])),
            },
        ))

        logger.info(
            "reporting_summary_generated",
            extra={
                "report_format": report_format,
                "report_id": report_id,
                "report_saved": report_saved,
            },
        )

        return {
            "current_node": "report",
            "report_id": report_id,
            "report_saved": report_saved,
            "report_summary": report_summary,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: ReportingAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<ReportingAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
