"""
Brand Monitor Agent -- The Reputation Sentinel.

Scans platforms for brand mentions, analyzes sentiment across
social media, news, forums, and review sites, calculates brand
health scores, and generates alerts for negative or critical mentions.

Architecture (LangGraph State Machine):
    scan_mentions -> analyze_sentiment -> generate_alerts ->
    human_review -> report -> END

Trigger Events:
    - scheduled: Hourly/daily brand mention sweeps
    - mention_spike: Volume anomaly detected
    - manual: On-demand reputation analysis

Shared Brain Integration:
    - Reads: brand keywords, competitor names, alert history
    - Writes: sentiment trends, brand health scores, alert patterns

Safety:
    - NEVER posts or interacts on social platforms
    - All data gathered from public sources only
    - Alerts require human review before escalation
    - Sentiment scores are advisory; human judgment required

Usage:
    agent = BrandMonitorAgent(config, db, embedder, llm)
    result = await agent.run({
        "brand_keywords": ["Enclave", "EnclaveSec"],
        "scan_scope": "full",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import BrandMentionData
from core.agents.registry import register_agent_type
from core.agents.state import BrandMonitorAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

PLATFORMS = [
    "twitter",
    "linkedin",
    "reddit",
    "news",
    "blog",
    "forum",
    "review",
]

SENTIMENT_THRESHOLDS = {
    "positive": 0.6,
    "negative": -0.3,
    "alert": -0.5,
}

ALERT_TRIGGERS = {
    "negative_spike": {
        "description": "Sudden increase in negative mentions",
        "threshold_pct": 25.0,
        "severity": "high",
        "response_window_hours": 4,
    },
    "critical_mention": {
        "description": "Highly negative mention from influential source",
        "threshold_score": -0.8,
        "severity": "critical",
        "response_window_hours": 1,
    },
    "competitor_mention": {
        "description": "Brand mentioned alongside a competitor",
        "threshold_score": -0.2,
        "severity": "medium",
        "response_window_hours": 24,
    },
    "review_negative": {
        "description": "Negative review on a major platform",
        "threshold_score": -0.5,
        "severity": "high",
        "response_window_hours": 8,
    },
    "viral_risk": {
        "description": "Mention gaining traction rapidly",
        "threshold_engagement": 1000,
        "severity": "critical",
        "response_window_hours": 2,
    },
}

# --- LLM Prompt Templates ---------------------------------------------------

SENTIMENT_ANALYSIS_PROMPT = """You are a brand reputation analyst. Analyze the following brand mentions and score their sentiment.

Brand Keywords:
{brand_keywords_json}

Mentions to Analyze:
{mentions_json}

For each mention, return a JSON array of objects:
[
    {{
        "mention_id": 0,
        "platform": "twitter|linkedin|reddit|news|blog|forum|review",
        "author": "author name or handle",
        "text_snippet": "First 200 chars of mention",
        "sentiment": "positive|negative|neutral|mixed",
        "sentiment_score": -1.0 to 1.0,
        "topics": ["topic1", "topic2"],
        "competitor_mentioned": "name or empty string",
        "influence_score": 0.0 to 1.0,
        "risk_level": "none|low|medium|high|critical",
        "recommended_response": "Suggested action or empty string"
    }}
]

Score -1.0 = extremely negative, 0.0 = neutral, 1.0 = extremely positive.
Consider brand context, sarcasm, and nuance carefully.

Return ONLY the JSON array, no markdown code fences.
"""

ALERT_GENERATION_PROMPT = """You are a brand reputation analyst. Generate actionable alerts from the negative or critical brand mentions below.

Sentiment Results:
{sentiment_results_json}

Alert Trigger Thresholds:
{alert_triggers_json}

For each mention warranting an alert, create:
[
    {{
        "mention_id": 0,
        "alert_type": "negative_spike|critical_mention|competitor_mention|review_negative|viral_risk",
        "severity": "critical|high|medium|low",
        "platform": "platform name",
        "message": "Clear, actionable alert message",
        "recommended_action": "Specific response steps",
        "urgency_hours": 1-24,
        "author": "mention author",
        "sentiment_score": -1.0 to 0.0
    }}
]

Only generate alerts for mentions that meet severity thresholds.
Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("brand_monitor")
class BrandMonitorAgent(BaseAgent):
    """
    Brand monitoring and sentiment analysis agent.

    Nodes:
        1. scan_mentions       -- Pull mentions from brand_mentions table
        2. analyze_sentiment   -- LLM scores sentiment, calculates brand health
        3. generate_alerts     -- Create alerts for negative/critical mentions
        4. human_review        -- Gate: approve alerts
        5. report              -- Save to DB + store BrandMentionData insight
    """

    def build_graph(self) -> Any:
        """Build the Brand Monitor Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(BrandMonitorAgentState)

        workflow.add_node("scan_mentions", self._node_scan_mentions)
        workflow.add_node("analyze_sentiment", self._node_analyze_sentiment)
        workflow.add_node("generate_alerts", self._node_generate_alerts)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_mentions")

        workflow.add_edge("scan_mentions", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "generate_alerts")
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
    def get_state_class(cls) -> Type[BrandMonitorAgentState]:
        return BrandMonitorAgentState

    # --- State Preparation ---------------------------------------------------

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "mentions": [],
            "mention_count": 0,
            "platforms_scanned": [],
            "sentiment_results": [],
            "brand_health_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "alerts": [],
            "alert_count": 0,
            "alerts_approved": False,
            "mentions_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # --- Node 1: Scan Mentions -----------------------------------------------

    async def _node_scan_mentions(
        self, state: BrandMonitorAgentState
    ) -> dict[str, Any]:
        """Node 1: Pull brand mentions from brand_mentions table and task input."""
        task = state.get("task_input", {})

        logger.info(
            "brand_monitor_scan_mentions",
            extra={"agent_id": self.agent_id},
        )

        brand_keywords = task.get("brand_keywords", [])
        scan_scope = task.get("scan_scope", "full")
        platform_filter = task.get("platforms", PLATFORMS)

        if not brand_keywords:
            brand_keywords = self.config.params.get("brand_keywords", [])

        mentions: list[dict[str, Any]] = []
        platforms_scanned: list[str] = []

        # Load mentions from database
        try:
            result = (
                self.db.client.table("brand_mentions")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .order("created_at", desc=True)
                .limit(100)
                .execute()
            )
            if result.data:
                for record in result.data:
                    platform = record.get("platform", "unknown")
                    mentions.append({
                        "mention_id": record.get("id", ""),
                        "platform": platform,
                        "author": record.get("author", ""),
                        "text": record.get("mention_text", ""),
                        "source_url": record.get("source_url", ""),
                        "engagement": record.get("engagement_count", 0),
                        "detected_at": record.get("created_at", ""),
                    })
                    if platform not in platforms_scanned:
                        platforms_scanned.append(platform)
        except Exception as e:
            logger.warning(
                "brand_monitor_db_scan_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Add task-provided mentions
        task_mentions = task.get("mentions", [])
        for tm in task_mentions:
            mentions.append(tm)
            plat = tm.get("platform", "unknown")
            if plat not in platforms_scanned:
                platforms_scanned.append(plat)

        # Filter by requested platforms
        if scan_scope != "full" and platform_filter:
            mentions = [
                m for m in mentions
                if m.get("platform", "unknown") in platform_filter
            ]

        logger.info(
            "brand_monitor_scan_complete",
            extra={
                "mentions_scanned": len(mentions),
                "platforms": len(platforms_scanned),
                "scope": scan_scope,
                "keyword_count": len(brand_keywords),
            },
        )

        return {
            "current_node": "scan_mentions",
            "mentions": mentions,
            "mention_count": len(mentions),
            "platforms_scanned": platforms_scanned,
        }

    # --- Node 2: Analyze Sentiment -------------------------------------------

    async def _node_analyze_sentiment(
        self, state: BrandMonitorAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM scores sentiment, categorizes, calculates brand health."""
        mentions = state.get("mentions", [])
        task = state.get("task_input", {})
        brand_keywords = task.get("brand_keywords", [])

        if not brand_keywords:
            brand_keywords = self.config.params.get("brand_keywords", [])

        logger.info(
            "brand_monitor_analyze_sentiment",
            extra={"mentions_to_analyze": len(mentions)},
        )

        sentiment_results: list[dict[str, Any]] = []
        brand_health_score = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        if not mentions:
            logger.info("brand_monitor_no_mentions_to_analyze")
            return {
                "current_node": "analyze_sentiment",
                "sentiment_results": [],
                "brand_health_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }

        try:
            prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                brand_keywords_json=json.dumps(brand_keywords, indent=2),
                mentions_json=json.dumps(mentions[:30], indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a brand reputation analyst.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                results_data = json.loads(llm_text)
                if isinstance(results_data, list):
                    sentiment_results = results_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("brand_monitor_sentiment_parse_error")

        except Exception as e:
            logger.warning(
                "brand_monitor_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        # Calculate aggregate scores
        for sr in sentiment_results:
            sentiment = sr.get("sentiment", "neutral")
            if sentiment == "positive":
                positive_count += 1
            elif sentiment == "negative":
                negative_count += 1
            else:
                neutral_count += 1

        total = len(sentiment_results) or 1
        # Brand health: weighted score from -100 to +100
        scores = [sr.get("sentiment_score", 0.0) for sr in sentiment_results]
        avg_score = sum(scores) / total if scores else 0.0
        brand_health_score = round((avg_score + 1.0) * 50, 1)  # Normalize to 0-100

        logger.info(
            "brand_monitor_sentiment_complete",
            extra={
                "total_analyzed": len(sentiment_results),
                "positive_mentions": positive_count,
                "negative_mentions": negative_count,
                "neutral_mentions": neutral_count,
                "health_score": brand_health_score,
            },
        )

        return {
            "current_node": "analyze_sentiment",
            "sentiment_results": sentiment_results,
            "brand_health_score": brand_health_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
        }

    # --- Node 3: Generate Alerts ---------------------------------------------

    async def _node_generate_alerts(
        self, state: BrandMonitorAgentState
    ) -> dict[str, Any]:
        """Node 3: Create alerts for negative/critical mentions and competitor mentions."""
        sentiment_results = state.get("sentiment_results", [])

        logger.info(
            "brand_monitor_generate_alerts",
            extra={"sentiment_count": len(sentiment_results)},
        )

        alerts: list[dict[str, Any]] = []

        # Filter to negative or risky mentions
        negative_mentions = [
            sr for sr in sentiment_results
            if sr.get("sentiment_score", 0.0) <= SENTIMENT_THRESHOLDS["negative"]
            or sr.get("risk_level", "none") in ("high", "critical")
            or sr.get("competitor_mentioned", "")
        ]

        if not negative_mentions:
            logger.info("brand_monitor_no_alerts_needed")
            return {
                "current_node": "generate_alerts",
                "alerts": [],
                "alert_count": 0,
            }

        try:
            prompt = ALERT_GENERATION_PROMPT.format(
                sentiment_results_json=json.dumps(negative_mentions[:15], indent=2),
                alert_triggers_json=json.dumps(ALERT_TRIGGERS, indent=2),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a brand reputation analyst generating alerts.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                alerts_data = json.loads(llm_text)
                if isinstance(alerts_data, list):
                    alerts = alerts_data
            except (json.JSONDecodeError, KeyError):
                logger.debug("brand_monitor_alerts_parse_error")
                # Fallback: create basic alerts from negative mentions
                for mention in negative_mentions[:5]:
                    trigger_key = "critical_mention" if mention.get(
                        "sentiment_score", 0
                    ) <= SENTIMENT_THRESHOLDS["alert"] else "review_negative"
                    trigger = ALERT_TRIGGERS.get(trigger_key, {})
                    alerts.append({
                        "mention_id": mention.get("mention_id", 0),
                        "alert_type": trigger_key,
                        "severity": trigger.get("severity", "high"),
                        "platform": mention.get("platform", "unknown"),
                        "message": mention.get("text_snippet", "")[:200],
                        "recommended_action": mention.get(
                            "recommended_response", "Review and assess"
                        ),
                        "urgency_hours": trigger.get("response_window_hours", 8),
                        "author": mention.get("author", ""),
                        "sentiment_score": mention.get("sentiment_score", 0.0),
                    })

        except Exception as e:
            logger.warning(
                "brand_monitor_alerts_llm_error",
                extra={"error_detail": str(e)[:200]},
            )

        logger.info(
            "brand_monitor_alerts_generated",
            extra={"alert_count": len(alerts)},
        )

        return {
            "current_node": "generate_alerts",
            "alerts": alerts,
            "alert_count": len(alerts),
        }

    # --- Node 4: Human Review ------------------------------------------------

    async def _node_human_review(
        self, state: BrandMonitorAgentState
    ) -> dict[str, Any]:
        """Node 4: Present alerts for human approval before escalation."""
        alerts = state.get("alerts", [])
        brand_health_score = state.get("brand_health_score", 0.0)

        logger.info(
            "brand_monitor_human_review_pending",
            extra={
                "alert_count": len(alerts),
                "health_score": brand_health_score,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # --- Node 5: Report ------------------------------------------------------

    async def _node_report(
        self, state: BrandMonitorAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to brand_mentions table and generate report."""
        now = datetime.now(timezone.utc).isoformat()
        mentions = state.get("mentions", [])
        sentiment_results = state.get("sentiment_results", [])
        alerts = state.get("alerts", [])
        brand_health_score = state.get("brand_health_score", 0.0)
        platforms_scanned = state.get("platforms_scanned", [])

        mentions_saved = False
        alert_count = len(alerts) if state.get("human_approval_status") == "approved" else 0

        # Save sentiment results to database
        for sr in sentiment_results:
            try:
                record = {
                    "vertical_id": self.vertical_id,
                    "agent_id": self.agent_id,
                    "platform": sr.get("platform", ""),
                    "author": sr.get("author", ""),
                    "mention_text": sr.get("text_snippet", "")[:500],
                    "sentiment": sr.get("sentiment", "neutral"),
                    "sentiment_score": sr.get("sentiment_score", 0.0),
                    "risk_level": sr.get("risk_level", "none"),
                    "competitor_mentioned": sr.get("competitor_mentioned", ""),
                    "created_at": now,
                }
                self.db.client.table("brand_mentions").insert(record).execute()
                mentions_saved = True
            except Exception as e:
                logger.debug(f"Failed to save brand mention: {e}")

        # Build report
        sections = [
            "# Brand Monitoring Report",
            f"*Generated: {now}*
",
            "## Summary",
            f"- **Mentions Scanned:** {state.get('mention_count', 0)}",
            f"- **Platforms:** {', '.join(platforms_scanned)}",
            f"- **Brand Health Score:** {brand_health_score:.1f}/100",
            f"- **Positive Mentions:** {state.get('positive_count', 0)}",
            f"- **Negative Mentions:** {state.get('negative_count', 0)}",
            f"- **Neutral Mentions:** {state.get('neutral_count', 0)}",
            f"- **Alerts Generated:** {len(alerts)}",
        ]

        if sentiment_results:
            sections.append("
## Sentiment Breakdown")
            for i, sr in enumerate(sentiment_results[:15], 1):
                score = sr.get("sentiment_score", 0.0)
                sentiment = sr.get("sentiment", "neutral").upper()
                platform = sr.get("platform", "unknown")
                sections.append(
                    f"{i}. **[{sentiment}]** ({platform}) "
                    f"Score: {score:+.2f} -- "
                    f"{sr.get('text_snippet', 'N/A')[:80]}"
                )

        if alerts:
            sections.append("
## Alerts")
            for i, a in enumerate(alerts[:10], 1):
                sections.append(
                    f"{i}. **{a.get('severity', 'N/A').upper()}** "
                    f"({a.get('platform', 'unknown')}): "
                    f"{a.get('message', 'N/A')[:100]}"
                )

        report = "
".join(sections)

        # Store insight
        if sentiment_results:
            self.store_insight(BrandMentionData(
                platform="multi-platform",
                sentiment="mixed",
                sentiment_score=round(
                    sum(sr.get("sentiment_score", 0.0) for sr in sentiment_results)
                    / max(len(sentiment_results), 1),
                    2,
                ),
                brand_health_score=brand_health_score,
                alert_triggered=len(alerts) > 0,
                alert_type="multi" if len(alerts) > 1 else (
                    alerts[0].get("alert_type", "") if alerts else ""
                ),
                mention_text=(
                    f"Scanned {len(mentions)} mentions across "
                    f"{len(platforms_scanned)} platforms. "
                    f"Brand health: {brand_health_score:.1f}/100. "
                    f"{len(alerts)} alerts generated."
                ),
                detected_at=now,
                metadata={
                    "mention_count": len(mentions),
                    "positive_count": state.get("positive_count", 0),
                    "negative_count": state.get("negative_count", 0),
                    "alert_count": len(alerts),
                    "platforms": platforms_scanned,
                },
            ))

        logger.info(
            "brand_monitor_report_generated",
            extra={
                "mentions_total": len(mentions),
                "sentiment_analyzed": len(sentiment_results),
                "alerts_total": len(alerts),
                "health_score": brand_health_score,
            },
        )

        return {
            "current_node": "report",
            "alert_count": alert_count,
            "mentions_saved": mentions_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # --- Routing -------------------------------------------------------------

    @staticmethod
    def _route_after_review(state: BrandMonitorAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<BrandMonitorAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
