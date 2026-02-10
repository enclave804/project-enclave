"""
Win/Loss Agent — The Strategist.

Analyzes closed opportunities (won and lost) to identify recurring
success and failure patterns, competitive gaps, and strategic
recommendations for improving close rates. Works across all verticals.

Architecture (LangGraph State Machine):
    load_deals → analyze_patterns → generate_recommendations →
    human_review → report → END

Trigger Events:
    - scheduled: Weekly/monthly win-loss analysis
    - deal_closed: New deal closed (won or lost)
    - manual: On-demand strategy review

Shared Brain Integration:
    - Reads: deal history, competitor intel, pricing data
    - Writes: win/loss factor patterns, competitive gap insights

Safety:
    - NEVER exposes individual deal pricing in cross-client reports
    - Competitive intelligence is advisory, not actionable without review
    - Recommendations require human review before process changes
    - Client names aggregated in pattern analysis to protect relationships

Usage:
    agent = WinLossAgent(config, db, embedder, llm)
    result = await agent.run({
        "analysis_period_days": 90,
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
from core.agents.state import WinLossAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

DEAL_FACTOR_CATEGORIES = [
    "pricing",
    "technical_capability",
    "response_time",
    "trust",
    "competition",
    "timing",
    "budget",
    "champion_strength",
]

ANALYSIS_PERIOD_DAYS = 90

WIN_LOSS_ANALYSIS_PROMPT = """\
You are a sales strategy analyst. Analyze the closed deals below and \
identify win and loss factors.

Won Deals ({won_count}):
{won_deals_json}

Lost Deals ({lost_count}):
{lost_deals_json}

Analysis Period: Last {period_days} days

Factor Categories: {factor_categories}

For each pattern, return a JSON object:
{{
    "win_factors": [
        {{
            "factor": "Description of the winning factor",
            "category": "one of: {factor_categories}",
            "frequency": 0.0-1.0,
            "impact": "high|medium|low",
            "example_deals": ["brief anonymized deal reference"]
        }}
    ],
    "loss_factors": [
        {{
            "factor": "Description of the losing factor",
            "category": "one of: {factor_categories}",
            "frequency": 0.0-1.0,
            "impact": "high|medium|low",
            "example_deals": ["brief anonymized deal reference"]
        }}
    ],
    "competitive_gaps": [
        {{
            "competitor": "Competitor name or category",
            "gap": "Description of the competitive gap",
            "frequency": 0.0-1.0,
            "deals_affected": 0
        }}
    ],
    "pattern_summary": "2-3 paragraph executive summary of key patterns"
}}

Be specific about factors. Reference deal stages and sizes where relevant. \
Protect client confidentiality by anonymizing details.

Return ONLY the JSON object, no markdown code fences.
"""

RECOMMENDATION_PROMPT = """\
Based on win/loss patterns below, generate actionable strategy \
recommendations to improve win rates.

Win Factors:
{win_factors_json}

Loss Factors:
{loss_factors_json}

Competitive Gaps:
{competitive_gaps_json}

Current Win Rate: {win_rate_pct}%
Average Sales Cycle: {avg_cycle_days} days

Return a JSON array of recommendations:
[
    {{
        "recommendation": "Clear, actionable recommendation",
        "priority": "critical|high|medium|low",
        "expected_impact": "Description of expected improvement",
        "category": "process|messaging|pricing|product|training",
        "effort": "low|medium|high",
        "timeline_weeks": 1-12
    }}
]

Focus on high-impact, actionable changes. Prioritize quick wins first.

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("win_loss")
class WinLossAgent(BaseAgent):
    """
    Win/loss analysis and strategy recommendation agent.

    Nodes:
        1. load_deals                -- Query closed opportunities for analysis period
        2. analyze_patterns          -- LLM identifies win/loss factors, competitive gaps
        3. generate_recommendations  -- LLM produces strategy recommendations
        4. human_review              -- Gate: approve recommendations
        5. report                    -- Save to deal_analyses table + InsightData
    """

    def build_graph(self) -> Any:
        """Build the Win/Loss Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(WinLossAgentState)

        workflow.add_node("load_deals", self._node_load_deals)
        workflow.add_node("analyze_patterns", self._node_analyze_patterns)
        workflow.add_node("generate_recommendations", self._node_generate_recommendations)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("load_deals")

        workflow.add_edge("load_deals", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "human_review")
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
    def get_state_class(cls) -> Type[WinLossAgentState]:
        return WinLossAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any] | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "recent_deals": [],
            "analysis_period_days": ANALYSIS_PERIOD_DAYS,
            "total_won": 0,
            "total_lost": 0,
            "win_factors": [],
            "loss_factors": [],
            "competitive_gaps": [],
            "avg_sales_cycle_days": 0.0,
            "recommendations": [],
            "recommendations_count": 0,
            "analyses_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Load Deals ──────────────────────────────────────────

    async def _node_load_deals(
        self, state: WinLossAgentState
    ) -> dict[str, Any]:
        """Node 1: Query closed opportunities (won + lost) for analysis period."""
        task = state.get("task_input", {})
        period_days = task.get("analysis_period_days", ANALYSIS_PERIOD_DAYS)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=period_days)
        ).isoformat()

        logger.info(
            "win_loss_load_deals",
            extra={
                "agent_id": self.agent_id,
                "period_days": period_days,
            },
        )

        recent_deals: list[dict[str, Any]] = []
        total_won = 0
        total_lost = 0
        cycle_days_sum = 0.0
        cycle_count = 0

        try:
            result = (
                self.db.client.table("opportunities")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .in_("stage", ["closed_won", "closed_lost"])
                .gte("updated_at", cutoff)
                .order("updated_at", desc=True)
                .limit(200)
                .execute()
            )

            if result.data:
                for deal in result.data:
                    stage = deal.get("stage", "")
                    recent_deals.append(deal)

                    if stage == "closed_won":
                        total_won += 1
                    elif stage == "closed_lost":
                        total_lost += 1

                    # Calculate sales cycle
                    created = deal.get("created_at", "")
                    closed = deal.get("updated_at", "")
                    if created and closed:
                        try:
                            created_dt = datetime.fromisoformat(
                                created.replace("Z", "+00:00")
                            )
                            closed_dt = datetime.fromisoformat(
                                closed.replace("Z", "+00:00")
                            )
                            days = (closed_dt - created_dt).days
                            if days >= 0:
                                cycle_days_sum += days
                                cycle_count += 1
                        except (ValueError, TypeError):
                            pass

        except Exception as e:
            logger.warning(
                "win_loss_deal_load_error",
                extra={"error": str(e)[:200]},
            )

        # Add task-provided deals if any
        task_deals = task.get("deals", [])
        if task_deals:
            recent_deals.extend(task_deals)
            for d in task_deals:
                if d.get("stage") == "closed_won":
                    total_won += 1
                elif d.get("stage") == "closed_lost":
                    total_lost += 1

        avg_cycle = round(cycle_days_sum / max(cycle_count, 1), 1)

        logger.info(
            "win_loss_deals_loaded",
            extra={
                "total_deals": len(recent_deals),
                "won": total_won,
                "lost": total_lost,
                "avg_cycle_days": avg_cycle,
            },
        )

        return {
            "current_node": "load_deals",
            "recent_deals": recent_deals,
            "analysis_period_days": period_days,
            "total_won": total_won,
            "total_lost": total_lost,
            "avg_sales_cycle_days": avg_cycle,
        }

    # ─── Node 2: Analyze Patterns ────────────────────────────────────

    async def _node_analyze_patterns(
        self, state: WinLossAgentState
    ) -> dict[str, Any]:
        """Node 2: LLM identifies win/loss factors and competitive gaps."""
        deals = state.get("recent_deals", [])
        total_won = state.get("total_won", 0)
        total_lost = state.get("total_lost", 0)
        period_days = state.get("analysis_period_days", ANALYSIS_PERIOD_DAYS)

        logger.info(
            "win_loss_analyze_patterns",
            extra={"total_deals": len(deals)},
        )

        win_factors: list[dict[str, Any]] = []
        loss_factors: list[dict[str, Any]] = []
        competitive_gaps: list[dict[str, Any]] = []
        pattern_summary = ""

        if not deals:
            logger.info("win_loss_no_deals_to_analyze")
            return {
                "current_node": "analyze_patterns",
                "win_factors": [],
                "loss_factors": [],
                "competitive_gaps": [],
            }

        # Separate won and lost deals for analysis
        won_deals = [d for d in deals if d.get("stage") == "closed_won"]
        lost_deals = [d for d in deals if d.get("stage") == "closed_lost"]

        # Anonymize deal data for LLM — strip emails and full names
        def anonymize(deal_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
            sanitized = []
            for d in deal_list[:25]:
                sanitized.append({
                    "industry": d.get("industry", ""),
                    "company_size": d.get("company_size", ""),
                    "value_cents": d.get("value_cents", 0),
                    "stage": d.get("stage", ""),
                    "loss_reason": d.get("loss_reason", ""),
                    "win_reason": d.get("win_reason", ""),
                    "competitor": d.get("competitor", ""),
                    "deal_source": d.get("source", ""),
                    "notes": d.get("notes", "")[:200],
                })
            return sanitized

        try:
            prompt = WIN_LOSS_ANALYSIS_PROMPT.format(
                won_count=total_won,
                lost_count=total_lost,
                won_deals_json=json.dumps(anonymize(won_deals), indent=2),
                lost_deals_json=json.dumps(anonymize(lost_deals), indent=2),
                period_days=period_days,
                factor_categories=", ".join(DEAL_FACTOR_CATEGORIES),
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a B2B sales strategy analyst specializing in win/loss analysis.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                analysis = json.loads(llm_text)
                win_factors = analysis.get("win_factors", [])
                loss_factors = analysis.get("loss_factors", [])
                competitive_gaps = analysis.get("competitive_gaps", [])
                pattern_summary = analysis.get("pattern_summary", "")
            except (json.JSONDecodeError, KeyError):
                logger.debug("win_loss_analysis_parse_error")

        except Exception as e:
            logger.warning(
                "win_loss_analysis_llm_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "win_loss_patterns_analyzed",
            extra={
                "win_factors": len(win_factors),
                "loss_factors": len(loss_factors),
                "competitive_gaps": len(competitive_gaps),
            },
        )

        return {
            "current_node": "analyze_patterns",
            "win_factors": win_factors,
            "loss_factors": loss_factors,
            "competitive_gaps": competitive_gaps,
        }

    # ─── Node 3: Generate Recommendations ────────────────────────────

    async def _node_generate_recommendations(
        self, state: WinLossAgentState
    ) -> dict[str, Any]:
        """Node 3: LLM produces strategy recommendations based on patterns."""
        win_factors = state.get("win_factors", [])
        loss_factors = state.get("loss_factors", [])
        competitive_gaps = state.get("competitive_gaps", [])
        total_won = state.get("total_won", 0)
        total_lost = state.get("total_lost", 0)
        avg_cycle = state.get("avg_sales_cycle_days", 0.0)

        total_deals = total_won + total_lost
        win_rate_pct = round(
            (total_won / max(total_deals, 1)) * 100, 1
        )

        logger.info(
            "win_loss_generate_recommendations",
            extra={
                "win_rate_pct": win_rate_pct,
                "avg_cycle_days": avg_cycle,
            },
        )

        recommendations: list[dict[str, Any]] = []

        if not win_factors and not loss_factors:
            logger.info("win_loss_no_factors_for_recommendations")
            return {
                "current_node": "generate_recommendations",
                "recommendations": [],
                "recommendations_count": 0,
            }

        try:
            prompt = RECOMMENDATION_PROMPT.format(
                win_factors_json=json.dumps(win_factors[:10], indent=2),
                loss_factors_json=json.dumps(loss_factors[:10], indent=2),
                competitive_gaps_json=json.dumps(competitive_gaps[:10], indent=2),
                win_rate_pct=win_rate_pct,
                avg_cycle_days=avg_cycle,
            )

            llm_response = self.llm.messages.create(
                model="claude-haiku-4-5-20250514",
                system="You are a sales strategy consultant generating actionable recommendations.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )

            llm_text = llm_response.content[0].text.strip()

            try:
                recs = json.loads(llm_text)
                if isinstance(recs, list):
                    recommendations = recs
            except (json.JSONDecodeError, KeyError):
                logger.debug("win_loss_recommendations_parse_error")
                # Fallback: generate basic recommendations from factors
                if loss_factors:
                    for factor in loss_factors[:3]:
                        recommendations.append({
                            "recommendation": f"Address loss factor: {factor.get('factor', 'Unknown')[:100]}",
                            "priority": "high" if factor.get("impact") == "high" else "medium",
                            "expected_impact": "Reduce losses from this factor",
                            "category": "process",
                            "effort": "medium",
                            "timeline_weeks": 4,
                        })

        except Exception as e:
            logger.warning(
                "win_loss_recommendations_llm_error",
                extra={"error": str(e)[:200]},
            )

        logger.info(
            "win_loss_recommendations_generated",
            extra={"recommendation_count": len(recommendations)},
        )

        return {
            "current_node": "generate_recommendations",
            "recommendations": recommendations,
            "recommendations_count": len(recommendations),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: WinLossAgentState
    ) -> dict[str, Any]:
        """Node 4: Present analysis and recommendations for human review."""
        recommendations = state.get("recommendations", [])
        total_won = state.get("total_won", 0)
        total_lost = state.get("total_lost", 0)

        logger.info(
            "win_loss_human_review_pending",
            extra={
                "recommendations": len(recommendations),
                "deals_analyzed": total_won + total_lost,
            },
        )
        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: WinLossAgentState
    ) -> dict[str, Any]:
        """Node 5: Save to deal_analyses table + InsightData on deal patterns."""
        now = datetime.now(timezone.utc).isoformat()
        total_won = state.get("total_won", 0)
        total_lost = state.get("total_lost", 0)
        win_factors = state.get("win_factors", [])
        loss_factors = state.get("loss_factors", [])
        competitive_gaps = state.get("competitive_gaps", [])
        recommendations = state.get("recommendations", [])
        avg_cycle = state.get("avg_sales_cycle_days", 0.0)
        period_days = state.get("analysis_period_days", ANALYSIS_PERIOD_DAYS)

        total_deals = total_won + total_lost
        win_rate = round(
            (total_won / max(total_deals, 1)) * 100, 1
        )

        # Save analysis to database
        analyses_saved = False
        try:
            analysis_record = {
                "vertical_id": self.vertical_id,
                "agent_id": self.agent_id,
                "analysis_period_days": period_days,
                "total_deals": total_deals,
                "total_won": total_won,
                "total_lost": total_lost,
                "win_rate_pct": win_rate,
                "avg_sales_cycle_days": avg_cycle,
                "win_factors": json.dumps(win_factors),
                "loss_factors": json.dumps(loss_factors),
                "competitive_gaps": json.dumps(competitive_gaps),
                "recommendations": json.dumps(recommendations),
                "status": (
                    "approved"
                    if state.get("human_approval_status") == "approved"
                    else "draft"
                ),
                "created_at": now,
            }

            result = (
                self.db.client.table("deal_analyses")
                .insert(analysis_record)
                .execute()
            )
            if result.data and len(result.data) > 0:
                analyses_saved = True
                logger.info(
                    "win_loss_analysis_saved",
                    extra={"id": result.data[0].get("id", "")},
                )
        except Exception as e:
            logger.warning(
                "win_loss_save_error",
                extra={"error": str(e)[:200]},
            )

        # Build report
        sections = [
            "# Win/Loss Analysis Report",
            f"*Generated: {now}*\n",
            f"## Summary",
            f"- **Analysis Period:** Last {period_days} days",
            f"- **Total Deals:** {total_deals}",
            f"- **Won:** {total_won} | **Lost:** {total_lost}",
            f"- **Win Rate:** {win_rate}%",
            f"- **Avg Sales Cycle:** {avg_cycle:.0f} days",
        ]

        if win_factors:
            sections.append("\n## Top Win Factors")
            for i, f in enumerate(win_factors[:5], 1):
                sections.append(
                    f"{i}. **[{f.get('category', 'N/A').upper()}]** "
                    f"{f.get('factor', 'N/A')[:100]} "
                    f"(Impact: {f.get('impact', 'N/A')})"
                )

        if loss_factors:
            sections.append("\n## Top Loss Factors")
            for i, f in enumerate(loss_factors[:5], 1):
                sections.append(
                    f"{i}. **[{f.get('category', 'N/A').upper()}]** "
                    f"{f.get('factor', 'N/A')[:100]} "
                    f"(Impact: {f.get('impact', 'N/A')})"
                )

        if competitive_gaps:
            sections.append("\n## Competitive Gaps")
            for i, g in enumerate(competitive_gaps[:5], 1):
                sections.append(
                    f"{i}. **{g.get('competitor', 'Unknown')}**: "
                    f"{g.get('gap', 'N/A')[:100]}"
                )

        if recommendations:
            sections.append("\n## Recommendations")
            for i, r in enumerate(recommendations[:5], 1):
                sections.append(
                    f"{i}. **[{r.get('priority', 'N/A').upper()}]** "
                    f"{r.get('recommendation', 'N/A')[:100]}"
                )

        report = "\n".join(sections)

        # Store insight
        if total_deals > 0:
            self.store_insight(InsightData(
                insight_type="deal_pattern",
                title=f"Win/Loss: {win_rate}% win rate over {period_days} days",
                content=(
                    f"Analyzed {total_deals} closed deals ({total_won} won, "
                    f"{total_lost} lost). Win rate: {win_rate}%. "
                    f"Avg cycle: {avg_cycle:.0f} days. "
                    f"Identified {len(win_factors)} win factors, "
                    f"{len(loss_factors)} loss factors, and "
                    f"{len(competitive_gaps)} competitive gaps. "
                    f"Generated {len(recommendations)} recommendations."
                ),
                confidence=0.80,
                metadata={
                    "total_deals": total_deals,
                    "win_rate_pct": win_rate,
                    "avg_cycle_days": avg_cycle,
                    "win_factor_count": len(win_factors),
                    "loss_factor_count": len(loss_factors),
                    "recommendation_count": len(recommendations),
                },
            ))

        logger.info(
            "win_loss_report_generated",
            extra={
                "win_rate": win_rate,
                "total_deals": total_deals,
                "recommendations": len(recommendations),
                "saved": analyses_saved,
            },
        )

        return {
            "current_node": "report",
            "analyses_saved": analyses_saved,
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: WinLossAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<WinLossAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
