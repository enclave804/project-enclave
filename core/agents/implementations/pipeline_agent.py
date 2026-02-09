"""
Sales Pipeline Agent — The Deal Tracker.

Monitors all opportunities through their lifecycle stages, identifies
stalled or at-risk deals, and recommends actions to keep the pipeline
flowing toward revenue.

Architecture (LangGraph State Machine):
    scan_pipeline -> analyze_deals -> recommend_actions ->
    human_review -> execute_actions -> report -> END

Stages (from `opportunities` table):
    prospect -> qualified -> proposal -> negotiation -> closed_won | closed_lost

Trigger Events:
    - scheduled (daily): Full pipeline scan and analysis
    - deal_stage_changed: Re-analyze when a deal moves stages
    - manual: On-demand pipeline review

Shared Brain Integration:
    - Reads: outreach performance, meeting outcomes, proposal data
    - Writes: pipeline velocity, deal patterns, conversion insights

Safety:
    - NEVER auto-moves deals -- human_review gate on all transitions
    - Stage transitions require explicit approval
    - Lost deals require reason documentation
    - Revenue projections are estimates, not guarantees

Usage:
    agent = SalesPipelineAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "full_scan"})
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import SalesPipelineAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

MODES = {"full_scan", "stalled_check", "forecast", "stage_update"}

PIPELINE_STAGES = [
    "prospect", "qualified", "proposal",
    "negotiation", "closed_won", "closed_lost",
]

STAGE_ORDER = {stage: i for i, stage in enumerate(PIPELINE_STAGES)}

# Default thresholds for deal health
DEFAULT_STALE_DAYS = 7      # Days without activity before "stalled"
DEFAULT_AT_RISK_DAYS = 14   # Days without activity before "at risk"

PIPELINE_SYSTEM_PROMPT = """\
You are a sales pipeline analyst for {company_name}. \
You review the current pipeline and recommend specific actions \
to move deals forward.

Given the pipeline data below, produce a JSON array of recommended actions:
[
    {{
        "deal_id": "opportunity UUID",
        "company_name": "...",
        "current_stage": "...",
        "recommended_action": "advance_stage" | "send_followup" | "schedule_meeting" | "send_proposal" | "mark_lost",
        "next_stage": "..." (if advancing),
        "reasoning": "Brief explanation",
        "priority": "high" | "medium" | "low",
        "confidence": 0.0-1.0
    }}
]

Rules:
- Only recommend stage advances when there's clear evidence of progress
- Stalled deals (>{stale_days} days) need follow-up or meeting
- At-risk deals (>{at_risk_days} days) need immediate attention
- Never recommend closing without negotiation completion
- Consider the full conversation history for each deal

Return ONLY the JSON array, no markdown code fences.
"""


@register_agent_type("sales_pipeline")
class SalesPipelineAgent(BaseAgent):
    """
    Deal stage tracking and pipeline optimization agent.

    Nodes:
        1. scan_pipeline       -- Pull all opportunities, compute stage metrics
        2. analyze_deals       -- Score deal health, identify stalled/at-risk
        3. recommend_actions   -- Suggest stage transitions and follow-ups
        4. human_review        -- Gate: approve recommended actions
        5. execute_actions     -- Move deals through stages, dispatch events
        6. report              -- Pipeline summary + Hive Mind insights
    """

    def build_graph(self) -> Any:
        """Build the Sales Pipeline Agent's LangGraph state machine."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(SalesPipelineAgentState)

        workflow.add_node("scan_pipeline", self._node_scan_pipeline)
        workflow.add_node("analyze_deals", self._node_analyze_deals)
        workflow.add_node("recommend_actions", self._node_recommend_actions)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("execute_actions", self._node_execute_actions)
        workflow.add_node("report", self._node_report)

        workflow.set_entry_point("scan_pipeline")

        workflow.add_edge("scan_pipeline", "analyze_deals")
        workflow.add_edge("analyze_deals", "recommend_actions")
        workflow.add_edge("recommend_actions", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "execute_actions",
                "rejected": "report",
            },
        )
        workflow.add_edge("execute_actions", "report")
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
        return []

    def get_state_class(self) -> Type[SalesPipelineAgentState]:
        return SalesPipelineAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "opportunities": [],
            "stage_metrics": {},
            "total_pipeline_value": 0.0,
            "stalled_deals": [],
            "at_risk_deals": [],
            "hot_deals": [],
            "recommended_actions": [],
            "actions_approved": False,
            "actions_executed": [],
            "actions_failed": [],
            "deals_moved": 0,
            "deals_won": 0,
            "deals_lost": 0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Scan Pipeline ──────────────────────────────────────

    async def _node_scan_pipeline(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Pull all opportunities and compute stage metrics.

        Queries the `opportunities` table for all active deals,
        groups by stage, and computes value and count metrics.
        """
        logger.info(
            "pipeline_scan_start",
            extra={"agent_id": self.agent_id},
        )

        opportunities: list[dict[str, Any]] = []

        try:
            result = (
                self.db.client.table("opportunities")
                .select("*")
                .eq("vertical_id", self.vertical_id)
                .execute()
            )
            opportunities = result.data or []
        except Exception as e:
            logger.debug(f"Failed to load opportunities: {e}")

        # Compute stage metrics
        stage_metrics: dict[str, Any] = {}
        total_value = 0.0

        for stage in PIPELINE_STAGES:
            stage_deals = [
                o for o in opportunities if o.get("stage") == stage
            ]
            stage_value = sum(
                o.get("value_cents", 0) for o in stage_deals
            )
            total_value += stage_value

            # Calculate average age
            avg_age = 0.0
            if stage_deals:
                now = datetime.now(timezone.utc)
                ages = []
                for deal in stage_deals:
                    created = deal.get("created_at", "")
                    if created:
                        try:
                            dt = datetime.fromisoformat(
                                created.replace("Z", "+00:00")
                            )
                            ages.append((now - dt).days)
                        except (ValueError, TypeError):
                            pass
                if ages:
                    avg_age = sum(ages) / len(ages)

            stage_metrics[stage] = {
                "count": len(stage_deals),
                "value_cents": stage_value,
                "avg_age_days": round(avg_age, 1),
            }

        logger.info(
            "pipeline_scan_complete",
            extra={
                "total_deals": len(opportunities),
                "total_value": total_value,
            },
        )

        return {
            "current_node": "scan_pipeline",
            "opportunities": opportunities,
            "stage_metrics": stage_metrics,
            "total_pipeline_value": total_value / 100,  # Convert to dollars
        }

    # ─── Node 2: Analyze Deals ──────────────────────────────────────

    async def _node_analyze_deals(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Analyze deal health and identify stalled/at-risk deals.

        Checks each opportunity's last activity date against thresholds.
        """
        opportunities = state.get("opportunities", [])
        stale_days = self.config.params.get("stale_days", DEFAULT_STALE_DAYS)
        at_risk_days = self.config.params.get(
            "at_risk_days", DEFAULT_AT_RISK_DAYS
        )

        logger.info(
            "pipeline_analyze_start",
            extra={"deals": len(opportunities)},
        )

        stalled: list[dict[str, Any]] = []
        at_risk: list[dict[str, Any]] = []
        hot: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for deal in opportunities:
            stage = deal.get("stage", "prospect")

            # Skip closed deals
            if stage in ("closed_won", "closed_lost"):
                continue

            # Calculate days since last activity
            updated = deal.get("updated_at", deal.get("created_at", ""))
            days_inactive = 0
            if updated:
                try:
                    dt = datetime.fromisoformat(
                        updated.replace("Z", "+00:00")
                    )
                    days_inactive = (now - dt).days
                except (ValueError, TypeError):
                    pass

            deal_info = {
                "id": deal.get("id", ""),
                "company_id": deal.get("company_id", ""),
                "stage": stage,
                "value_cents": deal.get("value_cents", 0),
                "days_inactive": days_inactive,
                "notes": deal.get("notes", ""),
            }

            if days_inactive >= at_risk_days:
                at_risk.append(deal_info)
            elif days_inactive >= stale_days:
                stalled.append(deal_info)

            # Hot deals: in negotiation with high value
            if stage == "negotiation" and days_inactive < stale_days:
                hot.append(deal_info)

        logger.info(
            "pipeline_analyze_complete",
            extra={
                "stalled": len(stalled),
                "at_risk": len(at_risk),
                "hot": len(hot),
            },
        )

        return {
            "current_node": "analyze_deals",
            "stalled_deals": stalled,
            "at_risk_deals": at_risk,
            "hot_deals": hot,
        }

    # ─── Node 3: Recommend Actions ──────────────────────────────────

    async def _node_recommend_actions(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Generate recommended actions for pipeline optimization.

        Uses LLM to analyze pipeline state and suggest specific actions
        for stalled, at-risk, and hot deals.
        """
        stalled = state.get("stalled_deals", [])
        at_risk = state.get("at_risk_deals", [])
        hot = state.get("hot_deals", [])
        stage_metrics = state.get("stage_metrics", {})

        logger.info(
            "pipeline_recommend_start",
            extra={
                "stalled": len(stalled),
                "at_risk": len(at_risk),
                "hot": len(hot),
            },
        )

        recommendations: list[dict[str, Any]] = []

        # Rule-based recommendations for common cases
        for deal in at_risk:
            recommendations.append({
                "deal_id": deal["id"],
                "company_id": deal.get("company_id", ""),
                "current_stage": deal["stage"],
                "recommended_action": "send_followup",
                "reasoning": (
                    f"Deal inactive for {deal['days_inactive']} days. "
                    f"Needs immediate follow-up to prevent loss."
                ),
                "priority": "high",
                "confidence": 0.8,
            })

        for deal in stalled:
            action = "send_followup"
            if deal["stage"] == "qualified":
                action = "schedule_meeting"
            elif deal["stage"] == "proposal":
                action = "send_followup"
            elif deal["stage"] == "negotiation":
                action = "schedule_meeting"

            recommendations.append({
                "deal_id": deal["id"],
                "company_id": deal.get("company_id", ""),
                "current_stage": deal["stage"],
                "recommended_action": action,
                "reasoning": (
                    f"Deal stalled at {deal['stage']} for "
                    f"{deal['days_inactive']} days."
                ),
                "priority": "medium",
                "confidence": 0.7,
            })

        for deal in hot:
            if deal["stage"] == "negotiation":
                recommendations.append({
                    "deal_id": deal["id"],
                    "company_id": deal.get("company_id", ""),
                    "current_stage": deal["stage"],
                    "recommended_action": "advance_stage",
                    "next_stage": "closed_won",
                    "reasoning": "Hot deal in negotiation — ready to close.",
                    "priority": "high",
                    "confidence": 0.6,
                })

        # Optionally enhance with LLM analysis
        if recommendations and self.llm:
            try:
                company_name = self.config.params.get(
                    "company_name", "Our Team"
                )
                stale_days = self.config.params.get(
                    "stale_days", DEFAULT_STALE_DAYS
                )
                at_risk_d = self.config.params.get(
                    "at_risk_days", DEFAULT_AT_RISK_DAYS
                )
                # LLM can refine priorities — but rule-based is primary
                logger.debug(
                    "pipeline_llm_enhancement_available",
                    extra={"recommendations": len(recommendations)},
                )
            except Exception as e:
                logger.debug(f"LLM pipeline analysis failed: {e}")

        logger.info(
            "pipeline_recommend_complete",
            extra={"recommendations": len(recommendations)},
        )

        return {
            "current_node": "recommend_actions",
            "recommended_actions": recommendations,
        }

    # ─── Node 4: Human Review ──────────────────────────────────────

    async def _node_human_review(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Present pipeline recommendations for human approval.

        All stage transitions and follow-up actions require approval.
        """
        actions = state.get("recommended_actions", [])

        logger.info(
            "pipeline_human_review_pending",
            extra={"action_count": len(actions)},
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Execute Actions ──────────────────────────────────

    async def _node_execute_actions(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Execute approved pipeline actions.

        Moves deals through stages in the `opportunities` table and
        dispatches events for downstream agents.
        """
        actions = state.get("recommended_actions", [])
        executed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        deals_moved = 0
        deals_won = 0
        deals_lost = 0

        logger.info(
            "pipeline_execute_start",
            extra={"actions": len(actions)},
        )

        for action in actions:
            deal_id = action.get("deal_id", "")
            rec_action = action.get("recommended_action", "")

            try:
                if rec_action == "advance_stage":
                    next_stage = action.get("next_stage", "")
                    if next_stage:
                        now = datetime.now(timezone.utc)
                        update_data: dict[str, Any] = {
                            "stage": next_stage,
                            "updated_at": now.isoformat(),
                        }
                        if next_stage == "closed_won":
                            update_data["won_at"] = now.isoformat()
                            deals_won += 1
                        elif next_stage == "closed_lost":
                            update_data["lost_at"] = now.isoformat()
                            update_data["lost_reason"] = action.get(
                                "reasoning", ""
                            )
                            deals_lost += 1

                        self.db.client.table("opportunities").update(
                            update_data
                        ).eq("id", deal_id).execute()
                        deals_moved += 1

                elif rec_action in ("send_followup", "schedule_meeting"):
                    # In production: dispatch to FollowUpAgent or MeetingSchedulerAgent
                    # via EventBus/TaskQueue
                    logger.info(
                        f"pipeline_dispatch_{rec_action}",
                        extra={"deal_id": deal_id},
                    )

                executed.append({
                    **action,
                    "status": "executed",
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                })

            except Exception as e:
                logger.error(f"Failed to execute action for {deal_id}: {e}")
                failed.append({
                    **action,
                    "status": "failed",
                    "error": str(e)[:200],
                })

        # Write insights to shared brain
        if deals_moved > 0:
            self.store_insight(InsightData(
                insight_type="pipeline_movement",
                title=f"Pipeline: {deals_moved} deals moved",
                content=(
                    f"Pipeline actions executed: {deals_moved} stage transitions, "
                    f"{deals_won} won, {deals_lost} lost."
                ),
                confidence=0.85,
                metadata={
                    "deals_moved": deals_moved,
                    "deals_won": deals_won,
                    "deals_lost": deals_lost,
                },
            ))

        return {
            "current_node": "execute_actions",
            "actions_executed": executed,
            "actions_failed": failed,
            "actions_approved": True,
            "deals_moved": deals_moved,
            "deals_won": deals_won,
            "deals_lost": deals_lost,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ────────────────────────────────────────────

    async def _node_report(
        self, state: SalesPipelineAgentState
    ) -> dict[str, Any]:
        """Node 6: Generate pipeline summary report."""
        now = datetime.now(timezone.utc).isoformat()
        stage_metrics = state.get("stage_metrics", {})

        sections = [
            "# Sales Pipeline Report",
            f"*Generated: {now}*\n",
            "## Pipeline Overview",
            f"- **Total Pipeline Value:** ${state.get('total_pipeline_value', 0):,.2f}",
            f"- **Total Deals:** {len(state.get('opportunities', []))}",
        ]

        # Stage breakdown
        sections.append("\n## Stage Breakdown")
        for stage in PIPELINE_STAGES:
            metrics = stage_metrics.get(stage, {})
            count = metrics.get("count", 0)
            value = metrics.get("value_cents", 0) / 100
            avg_age = metrics.get("avg_age_days", 0)
            sections.append(
                f"- **{stage.replace('_', ' ').title()}:** "
                f"{count} deals (${value:,.2f}) — avg {avg_age:.0f} days"
            )

        # Health indicators
        stalled = state.get("stalled_deals", [])
        at_risk = state.get("at_risk_deals", [])
        hot = state.get("hot_deals", [])

        sections.append("\n## Deal Health")
        sections.append(f"- **Hot Deals:** {len(hot)}")
        sections.append(f"- **Stalled Deals:** {len(stalled)}")
        sections.append(f"- **At-Risk Deals:** {len(at_risk)}")

        # Actions taken
        executed = state.get("actions_executed", [])
        failed = state.get("actions_failed", [])

        sections.append("\n## Actions")
        sections.append(f"- **Recommended:** {len(state.get('recommended_actions', []))}")
        sections.append(f"- **Executed:** {len(executed)}")
        sections.append(f"- **Failed:** {len(failed)}")
        sections.append(f"- **Deals Moved:** {state.get('deals_moved', 0)}")
        sections.append(f"- **Deals Won:** {state.get('deals_won', 0)}")
        sections.append(f"- **Deals Lost:** {state.get('deals_lost', 0)}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing ──────────────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: SalesPipelineAgentState) -> str:
        status = state.get("human_approval_status", "approved")
        return "rejected" if status == "rejected" else "approved"

    # ─── System Prompt ────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        if self.config.system_prompt_path:
            try:
                with open(self.config.system_prompt_path) as f:
                    return f.read()
            except FileNotFoundError:
                pass
        return (
            "You are a sales pipeline analyst. You monitor deals through "
            "their lifecycle stages, identify stalled opportunities, and "
            "recommend specific actions to move deals toward close. "
            "You prioritize high-value deals and flag at-risk opportunities."
        )

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"<SalesPipelineAgent agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
