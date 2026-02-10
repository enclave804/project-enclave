"""
Feedback Agent — The Voice of the Customer.

Identifies clients at key touchpoints (post-project, quarterly,
post-support, onboarding complete), analyzes NPS/CSAT/CES survey
responses, calculates Net Promoter Score, and routes critical
negative feedback for immediate follow-up. Works across all verticals.

Architecture (LangGraph State Machine):
    identify_audience → analyze_responses → route_critical →
    human_review → report → END

Trigger Events:
    - scheduled: Monthly NPS sweep across all active clients
    - event: project_completed, onboarding_complete (touchpoint-based)
    - manual: On-demand feedback analysis

Shared Brain Integration:
    - Reads: client records, project history, onboarding status
    - Writes: satisfaction trends, NPS benchmarks, churn risk signals

Safety:
    - Escalation actions require human review before execution
    - Client feedback is treated as confidential within the vertical
    - NPS scores are advisory; human judgment required for follow-up

Usage:
    agent = FeedbackAgent(config, db, embedder, llm)
    result = await agent.run({
        "touchpoint": "post_project",
        "include_existing_responses": True,
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
from core.agents.state import FeedbackAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

NPS_CATEGORIES = {
    "promoter": {"min": 9, "max": 10},
    "passive": {"min": 7, "max": 8},
    "detractor": {"min": 0, "max": 6},
}

SURVEY_TYPES = ["nps", "csat", "ces", "custom"]

TOUCHPOINTS = [
    "post_project",
    "quarterly",
    "post_support",
    "onboarding_complete",
]

TOUCHPOINT_LOOKBACK_DAYS = {
    "post_project": 14,
    "quarterly": 90,
    "post_support": 7,
    "onboarding_complete": 30,
}

SENTIMENT_ANALYSIS_PROMPT = """\
You are a customer feedback analyst. Analyze the following survey \
responses and classify each by sentiment and actionability.

Feedback Responses:
{responses_json}

Survey Type: {survey_type}
Touchpoint: {touchpoint}

For each response, classify:
1. Sentiment (positive, neutral, negative, very_negative)
2. Key themes/topics mentioned
3. Actionability (immediate_action, monitor, no_action)
4. Suggested follow-up action if sentiment is negative

Return a JSON array:
[
    {{
        "client_id": "original client_id",
        "client_name": "original client_name",
        "score": 0,
        "sentiment": "positive|neutral|negative|very_negative",
        "themes": ["theme1", "theme2"],
        "actionability": "immediate_action|monitor|no_action",
        "key_quote": "Most impactful quote from their feedback",
        "suggested_action": "What to do about this feedback"
    }}
]

Be objective and evidence-based. Return ONLY the JSON array, \
no markdown code fences.
"""

FEEDBACK_SUMMARY_PROMPT = """\
Summarize the following customer feedback data into a concise \
executive brief.

NPS Score: {nps_score}
Total Responses: {total_responses}
Promoters: {promoter_count} ({promoter_pct}%)
Passives: {passive_count} ({passive_pct}%)
Detractors: {detractor_count} ({detractor_pct}%)

Top Positive Themes:
{positive_themes_json}

Top Negative Themes:
{negative_themes_json}

Critical Feedback Requiring Attention:
{critical_json}

Generate a 3-paragraph executive summary covering:
1. Overall satisfaction health (one paragraph)
2. Key wins and strengths identified (one paragraph)
3. Areas requiring immediate attention (one paragraph)

Return as a JSON object:
{{
    "executive_summary": "The full 3-paragraph summary",
    "health_rating": "excellent|good|fair|poor|critical",
    "top_risk": "Single biggest risk identified",
    "top_opportunity": "Single biggest opportunity identified"
}}

Return ONLY the JSON, no markdown fences.
"""


def _classify_nps(score: int) -> str:
    """Classify an NPS score into promoter/passive/detractor."""
    if score >= NPS_CATEGORIES["promoter"]["min"]:
        return "promoter"
    elif score >= NPS_CATEGORIES["passive"]["min"]:
        return "passive"
    return "detractor"


def _calculate_nps(scores: list[int]) -> float:
    """Calculate Net Promoter Score from a list of scores (0-10)."""
    if not scores:
        return 0.0
    promoters = sum(1 for s in scores if s >= 9)
    detractors = sum(1 for s in scores if s <= 6)
    total = len(scores)
    return round(((promoters - detractors) / total) * 100, 1)


@register_agent_type("feedback")
class FeedbackAgent(BaseAgent):
    """
    Customer feedback collection and NPS analysis agent.

    Nodes:
        1. identify_audience   -- Find clients at relevant touchpoints
        2. analyze_responses   -- LLM analyzes sentiment, calculate NPS
        3. route_critical      -- Flag negative feedback for follow-up
        4. human_review        -- Gate: approve escalation actions
        5. report              -- Save to feedback_responses + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Feedback Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(FeedbackAgentState)

        workflow.add_node("identify_audience", self._node_identify_audience)
        workflow.add_node("analyze_responses", self._node_analyze_responses)
        workflow.add_node("route_critical", self._node_route_critical)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("identify_audience")

        workflow.add_edge("identify_audience", "analyze_responses")
        workflow.add_edge("analyze_responses", "route_critical")
        workflow.add_edge("route_critical", "human_review")
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
    def get_state_class(cls) -> Type[FeedbackAgentState]:
        return FeedbackAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "target_touchpoint": "",
            "eligible_clients": [],
            "total_eligible": 0,
            "feedback_responses": [],
            "total_responses": 0,
            "nps_score": 0.0,
            "promoters": [],
            "passives": [],
            "detractors": [],
            "avg_sentiment": 0.5,
            "critical_feedback": [],
            "critical_count": 0,
            "escalation_actions": [],
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Identify Audience ───────────────────────────────────

    async def _node_identify_audience(
        self, state: FeedbackAgentState
    ) -> dict[str, Any]:
        """Node 1: Find clients at relevant touchpoints for feedback collection."""
        task = state.get("task_input", {})

        logger.info(
            "feedback_identify_audience_started",
            extra={"agent_id": self.agent_id},
        )

        touchpoint = task.get("touchpoint", self.config.params.get("default_touchpoint", "quarterly"))
        if touchpoint not in TOUCHPOINTS:
            logger.info(
                "feedback_touchpoint_fallback",
                extra={"requested": touchpoint, "fallback": "quarterly"},
            )
            touchpoint = "quarterly"

        lookback_days = TOUCHPOINT_LOOKBACK_DAYS.get(touchpoint, 30)
        cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()

        eligible_clients: list[dict[str, Any]] = []
        feedback_responses: list[dict[str, Any]] = []

        # Query clients based on touchpoint
        try:
            if touchpoint == "post_project":
                # Find clients with recently completed projects
                result = (
                    self.db.client.table("proposals")
                    .select("company_name, contact_email, contact_name, id")
                    .eq("vertical_id", self.vertical_id)
                    .eq("status", "completed")
                    .gte("completed_at", cutoff_date)
                    .order("completed_at", desc=True)
                    .limit(50)
                    .execute()
                )
                if result.data:
                    for record in result.data:
                        eligible_clients.append({
                            "client_id": record.get("id", ""),
                            "company_name": record.get("company_name", ""),
                            "contact_email": record.get("contact_email", ""),
                            "contact_name": record.get("contact_name", ""),
                            "touchpoint": touchpoint,
                        })
            elif touchpoint == "quarterly":
                # Find all active clients
                result = (
                    self.db.client.table("companies")
                    .select("id, name, domain")
                    .eq("vertical_id", self.vertical_id)
                    .eq("status", "active")
                    .limit(50)
                    .execute()
                )
                if result.data:
                    for record in result.data:
                        eligible_clients.append({
                            "client_id": record.get("id", ""),
                            "company_name": record.get("name", ""),
                            "company_domain": record.get("domain", ""),
                            "touchpoint": touchpoint,
                        })
            elif touchpoint == "onboarding_complete":
                # Find clients who recently completed onboarding
                result = (
                    self.db.client.table("client_onboarding")
                    .select("company_name, contact_email, contact_name, id")
                    .eq("vertical_id", self.vertical_id)
                    .eq("status", "completed")
                    .gte("created_at", cutoff_date)
                    .limit(50)
                    .execute()
                )
                if result.data:
                    for record in result.data:
                        eligible_clients.append({
                            "client_id": record.get("id", ""),
                            "company_name": record.get("company_name", ""),
                            "contact_email": record.get("contact_email", ""),
                            "contact_name": record.get("contact_name", ""),
                            "touchpoint": touchpoint,
                        })
            else:
                # post_support: recently resolved tickets
                result = (
                    self.db.client.table("support_tickets")
                    .select("customer_name, customer_email, customer_id, id")
                    .eq("vertical_id", self.vertical_id)
                    .eq("status", "resolved")
                    .gte("created_at", cutoff_date)
                    .limit(50)
                    .execute()
                )
                if result.data:
                    for record in result.data:
                        eligible_clients.append({
                            "client_id": record.get("customer_id", record.get("id", "")),
                            "contact_name": record.get("customer_name", ""),
                            "contact_email": record.get("customer_email", ""),
                            "touchpoint": touchpoint,
                        })

            logger.info(
                "feedback_eligible_clients_found",
                extra={
                    "touchpoint": touchpoint,
                    "count": len(eligible_clients),
                },
            )
        except Exception as e:
            logger.warning(
                "feedback_audience_query_error",
                extra={"error": str(e)[:200]},
            )

        # Load existing feedback responses if requested
        include_existing = task.get("include_existing_responses", True)
        if include_existing:
            try:
                result = (
                    self.db.client.table("feedback_responses")
                    .select("*")
                    .eq("vertical_id", self.vertical_id)
                    .eq("touchpoint", touchpoint)
                    .gte("created_at", cutoff_date)
                    .order("created_at", desc=True)
                    .limit(100)
                    .execute()
                )
                if result.data:
                    for resp in result.data:
                        feedback_responses.append({
                            "client_id": resp.get("client_id", ""),
                            "client_name": resp.get("client_name", ""),
                            "score": resp.get("score", 0),
                            "comment": resp.get("comment", ""),
                            "survey_type": resp.get("survey_type", "nps"),
                            "touchpoint": resp.get("touchpoint", touchpoint),
                            "created_at": resp.get("created_at", ""),
                        })
                    logger.info(
                        "feedback_existing_responses_loaded",
                        extra={"count": len(feedback_responses)},
                    )
            except Exception as e:
                logger.warning(
                    "feedback_responses_query_error",
                    extra={"error": str(e)[:200]},
                )

        # Add task-provided responses
        task_responses = task.get("feedback_responses", [])
        feedback_responses.extend(task_responses)

        # Add task-provided clients
        task_clients = task.get("eligible_clients", [])
        eligible_clients.extend(task_clients)

        logger.info(
            "feedback_audience_identified",
            extra={
                "touchpoint": touchpoint,
                "eligible": len(eligible_clients),
                "existing_responses": len(feedback_responses),
            },
        )

        return {
            "current_node": "identify_audience",
            "target_touchpoint": touchpoint,
            "eligible_clients": eligible_clients,
            "total_eligible": len(eligible_clients),
            "feedback_responses": feedback_responses,
            "total_responses": len(feedback_responses),
        }

    # ─── Node 2: Analyze Responses ───────────────────────────────────

    async def _node_analyze_responses(
        self, state: FeedbackAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM analyzes sentiment, calculate NPS, group categories."""
        feedback_responses = state.get("feedback_responses", [])
        touchpoint = state.get("target_touchpoint", "quarterly")
        task = state.get("task_input", {})
        survey_type = task.get("survey_type", "nps")

        logger.info(
            "feedback_analysis_started",
            extra={
                "responses": len(feedback_responses),
                "touchpoint": touchpoint,
            },
        )

        promoters: list[dict[str, Any]] = []
        passives: list[dict[str, Any]] = []
        detractors: list[dict[str, Any]] = []
        nps_score = 0.0
        avg_sentiment = 0.5

        if not feedback_responses:
            logger.info("feedback_no_responses_to_analyze")
            return {
                "current_node": "analyze_responses",
                "feedback_responses": [],
                "total_responses": 0,
                "nps_score": 0.0,
                "promoters": [],
                "passives": [],
                "detractors": [],
                "avg_sentiment": 0.5,
            }

        # Calculate NPS from raw scores
        scores = [r.get("score", 5) for r in feedback_responses if isinstance(r.get("score"), (int, float))]
        if scores:
            nps_score = _calculate_nps(scores)

        # Classify into NPS categories
        for resp in feedback_responses:
            score = resp.get("score", 5)
            category = _classify_nps(score)
            if category == "promoter":
                promoters.append(resp)
            elif category == "passive":
                passives.append(resp)
            else:
                detractors.append(resp)

        # LLM sentiment analysis for responses with comments
        responses_with_comments = [
            r for r in feedback_responses
            if r.get("comment", "").strip()
        ]

        analyzed_responses = list(feedback_responses)  # Start with all

        if responses_with_comments:
            try:
                # Anonymize before sending to LLM
                anonymized = []
                for r in responses_with_comments[:30]:
                    anonymized.append({
                        "client_id": r.get("client_id", ""),
                        "client_name": r.get("client_name", "Anonymous"),
                        "score": r.get("score", 0),
                        "comment": r.get("comment", "")[:500],
                    })

                prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                    responses_json=json.dumps(anonymized, indent=2),
                    survey_type=survey_type,
                    touchpoint=touchpoint,
                )

                llm_response = self.llm.messages.create(
                    model="claude-haiku-4-5-20250514",
                    system="You are a customer feedback analyst.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                )

                llm_text = llm_response.content[0].text.strip()

                try:
                    analysis_data = json.loads(llm_text)
                    if isinstance(analysis_data, list):
                        # Merge LLM analysis back into responses
                        analysis_map = {
                            a.get("client_id", ""): a
                            for a in analysis_data
                        }
                        for resp in analyzed_responses:
                            cid = resp.get("client_id", "")
                            if cid in analysis_map:
                                analysis = analysis_map[cid]
                                resp["sentiment"] = analysis.get("sentiment", "neutral")
                                resp["themes"] = analysis.get("themes", [])
                                resp["actionability"] = analysis.get("actionability", "no_action")
                                resp["key_quote"] = analysis.get("key_quote", "")
                                resp["suggested_action"] = analysis.get("suggested_action", "")
                except (json.JSONDecodeError, KeyError):
                    logger.warning("feedback_sentiment_parse_error")

            except Exception as e:
                logger.error(
                    "feedback_sentiment_llm_error",
                    extra={"error": str(e)[:200]},
                )

        # Calculate average sentiment score
        sentiment_values = {"positive": 0.9, "neutral": 0.5, "negative": 0.2, "very_negative": 0.1}
        sentiment_scores = [
            sentiment_values.get(r.get("sentiment", "neutral"), 0.5)
            for r in analyzed_responses
        ]
        if sentiment_scores:
            avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 2)

        logger.info(
            "feedback_analysis_complete",
            extra={
                "nps": nps_score,
                "promoters": len(promoters),
                "passives": len(passives),
                "detractors": len(detractors),
                "avg_sentiment": avg_sentiment,
            },
        )

        return {
            "current_node": "analyze_responses",
            "feedback_responses": analyzed_responses,
            "total_responses": len(analyzed_responses),
            "nps_score": nps_score,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "avg_sentiment": avg_sentiment,
        }

    # ─── Node 3: Route Critical ──────────────────────────────────────

    async def _node_route_critical(
        self, state: FeedbackAgentState
    ) -> dict[str, Any]:
        """Node 3: Flag negative feedback (NPS 0-6) for immediate follow-up."""
        detractors = state.get("detractors", [])
        feedback_responses = state.get("feedback_responses", [])

        logger.info(
            "feedback_critical_routing_started",
            extra={"detractors": len(detractors)},
        )

        critical_feedback: list[dict[str, Any]] = []
        escalation_actions: list[dict[str, Any]] = []

        # Identify critical feedback: detractors with actionable comments
        for resp in detractors:
            score = resp.get("score", 5)
            actionability = resp.get("actionability", "monitor")
            sentiment = resp.get("sentiment", "negative")

            # Critical: very low score OR marked as immediate_action
            is_critical = (
                score <= 4
                or actionability == "immediate_action"
                or sentiment == "very_negative"
            )

            if is_critical:
                critical_feedback.append(resp)

                # Build escalation action
                urgency = "high" if score <= 3 else "medium"
                suggested_action = resp.get(
                    "suggested_action",
                    "Schedule immediate follow-up call to address concerns",
                )

                escalation_actions.append({
                    "client_id": resp.get("client_id", ""),
                    "client_name": resp.get("client_name", "Unknown"),
                    "contact_email": resp.get("contact_email", ""),
                    "score": score,
                    "sentiment": sentiment,
                    "action": suggested_action,
                    "urgency": urgency,
                    "reasoning": (
                        f"NPS score {score}/10 with {sentiment} sentiment. "
                        f"Key concern: {resp.get('key_quote', 'N/A')[:100]}"
                    ),
                })

        # Also flag any responses with immediate_action regardless of score
        for resp in feedback_responses:
            if (
                resp.get("actionability") == "immediate_action"
                and resp.get("client_id") not in [
                    c.get("client_id") for c in critical_feedback
                ]
            ):
                critical_feedback.append(resp)
                escalation_actions.append({
                    "client_id": resp.get("client_id", ""),
                    "client_name": resp.get("client_name", "Unknown"),
                    "score": resp.get("score", 0),
                    "action": resp.get("suggested_action", "Review feedback immediately"),
                    "urgency": "medium",
                    "reasoning": f"Marked as immediate action: {resp.get('key_quote', '')[:100]}",
                })

        logger.info(
            "feedback_critical_routing_complete",
            extra={
                "critical_count": len(critical_feedback),
                "escalations": len(escalation_actions),
            },
        )

        return {
            "current_node": "route_critical",
            "critical_feedback": critical_feedback,
            "critical_count": len(critical_feedback),
            "escalation_actions": escalation_actions,
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: FeedbackAgentState
    ) -> dict[str, Any]:
        """Node 4: Present critical feedback and escalation actions for approval."""
        critical_count = state.get("critical_count", 0)
        nps_score = state.get("nps_score", 0)

        logger.info(
            "feedback_human_review_pending",
            extra={
                "critical_count": critical_count,
                "nps_score": nps_score,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: FeedbackAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to feedback_responses table and generate satisfaction insights."""
        now = datetime.now(timezone.utc).isoformat()
        touchpoint = state.get("target_touchpoint", "")
        nps_score = state.get("nps_score", 0.0)
        total_responses = state.get("total_responses", 0)
        promoters = state.get("promoters", [])
        passives = state.get("passives", [])
        detractors = state.get("detractors", [])
        critical_feedback = state.get("critical_feedback", [])
        escalation_actions = state.get("escalation_actions", [])
        avg_sentiment = state.get("avg_sentiment", 0.5)

        # Save feedback analysis to database
        try:
            record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "touchpoint": touchpoint,
                "nps_score": nps_score,
                "total_responses": total_responses,
                "promoter_count": len(promoters),
                "passive_count": len(passives),
                "detractor_count": len(detractors),
                "critical_count": len(critical_feedback),
                "avg_sentiment": avg_sentiment,
                "escalation_actions": json.dumps(escalation_actions[:20]),
                "created_at": now,
            }
            self.db.client.table("feedback_analysis").insert(record).execute()
            logger.info("feedback_analysis_saved")
        except Exception as e:
            logger.warning(
                "feedback_analysis_save_error",
                extra={"error": str(e)[:200]},
            )

        # Calculate percentages
        total = max(total_responses, 1)
        promoter_pct = round(len(promoters) / total * 100, 1)
        passive_pct = round(len(passives) / total * 100, 1)
        detractor_pct = round(len(detractors) / total * 100, 1)

        # Determine NPS health rating
        if nps_score >= 50:
            health_rating = "excellent"
        elif nps_score >= 20:
            health_rating = "good"
        elif nps_score >= 0:
            health_rating = "fair"
        elif nps_score >= -20:
            health_rating = "poor"
        else:
            health_rating = "critical"

        # Build report
        sections = [
            "# Customer Feedback Report",
            f"*Generated: {now}*\n",
            f"## NPS Summary",
            f"- **Net Promoter Score:** {nps_score} ({health_rating.upper()})",
            f"- **Total Responses:** {total_responses}",
            f"- **Touchpoint:** {touchpoint}",
            f"- **Average Sentiment:** {avg_sentiment:.2f}/1.0",
            f"\n## Distribution",
            f"- **Promoters (9-10):** {len(promoters)} ({promoter_pct}%)",
            f"- **Passives (7-8):** {len(passives)} ({passive_pct}%)",
            f"- **Detractors (0-6):** {len(detractors)} ({detractor_pct}%)",
        ]

        if critical_feedback:
            sections.append(f"\n## Critical Feedback ({len(critical_feedback)} items)")
            for i, cf in enumerate(critical_feedback[:5], 1):
                sections.append(
                    f"{i}. **{cf.get('client_name', 'Anonymous')}** — "
                    f"Score: {cf.get('score', 'N/A')}/10 | "
                    f"Sentiment: {cf.get('sentiment', 'N/A')} | "
                    f"{cf.get('key_quote', 'No comment')[:80]}"
                )

        if escalation_actions:
            sections.append(f"\n## Escalation Actions ({len(escalation_actions)} items)")
            for i, ea in enumerate(escalation_actions[:5], 1):
                sections.append(
                    f"{i}. **{ea.get('client_name', 'Unknown')}** "
                    f"[{ea.get('urgency', 'medium').upper()}] — "
                    f"{ea.get('action', 'N/A')[:80]}"
                )

        report = "\n".join(sections)

        # Store insight on satisfaction trends
        if total_responses > 0:
            self.store_insight(InsightData(
                insight_type="satisfaction_trend",
                title=f"NPS {nps_score} ({health_rating}) — {touchpoint} ({total_responses} responses)",
                content=(
                    f"NPS score: {nps_score} ({health_rating}). "
                    f"{len(promoters)} promoters, {len(passives)} passives, "
                    f"{len(detractors)} detractors from {total_responses} responses. "
                    f"Average sentiment: {avg_sentiment:.2f}. "
                    f"{len(critical_feedback)} critical items requiring follow-up."
                ),
                confidence=0.85 if total_responses >= 10 else 0.6,
                metadata={
                    "nps_score": nps_score,
                    "health_rating": health_rating,
                    "total_responses": total_responses,
                    "promoter_count": len(promoters),
                    "passive_count": len(passives),
                    "detractor_count": len(detractors),
                    "critical_count": len(critical_feedback),
                    "avg_sentiment": avg_sentiment,
                    "touchpoint": touchpoint,
                },
            ))

        logger.info(
            "feedback_report_generated",
            extra={
                "nps": nps_score,
                "health": health_rating,
                "responses": total_responses,
                "critical": len(critical_feedback),
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: FeedbackAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<FeedbackAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
