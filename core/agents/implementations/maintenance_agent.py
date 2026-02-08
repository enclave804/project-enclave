"""
Maintenance Agent — The Janitor.

Automated data hygiene for the shared brain (shared_insights table).
Runs on a weekly schedule to:

1. Scan for duplicate insights (cosine similarity > 0.95)
2. Merge duplicates into "Golden Insights" using LLM summarization
3. Prune decayed insights (old, low-usage entries)

This keeps the vector DB lean and relevant, preventing the "knowledge
garbage" problem where agents drown in stale or redundant context.

Architecture (LangGraph State Machine):
    scan_duplicates → merge_insights → prune_decayed → report

Usage:
    agent = MaintenanceAgent(config, db, embedder, llm)
    result = await agent.run({"mode": "full"})  # full scan + merge + prune
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData
from core.agents.registry import register_agent_type
from core.agents.state import BaseAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)


class MaintenanceAgentState(BaseAgentState, total=False):
    """State for the Maintenance Agent."""

    # ---- Scan Results ----
    duplicate_groups: list[list[dict[str, Any]]]
    duplicate_count: int

    # ---- Merge Results ----
    merged_insights: list[dict[str, Any]]
    merge_count: int

    # ---- Prune Results ----
    pruned_count: int
    pruned_ids: list[str]

    # ---- Report ----
    report_summary: str
    total_insights_before: int
    total_insights_after: int


@register_agent_type("maintenance")
class MaintenanceAgent(BaseAgent):
    """
    Automated data hygiene agent for the shared brain.

    Nodes:
        1. scan_duplicates — Find insights with cosine similarity > threshold
        2. merge_insights — LLM-powered deduplication into Golden Insights
        3. prune_decayed — Flag/remove old low-usage insights
        4. report — Generate maintenance summary
    """

    def build_graph(self) -> Any:
        """Build the Maintenance Agent's LangGraph state machine."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(MaintenanceAgentState)

        # Add nodes
        workflow.add_node("scan_duplicates", self._node_scan_duplicates)
        workflow.add_node("merge_insights", self._node_merge_insights)
        workflow.add_node("prune_decayed", self._node_prune_decayed)
        workflow.add_node("report", self._node_report)

        # Linear flow: scan → merge → prune → report
        workflow.set_entry_point("scan_duplicates")
        workflow.add_edge("scan_duplicates", "merge_insights")
        workflow.add_edge("merge_insights", "prune_decayed")
        workflow.add_edge("prune_decayed", "report")
        workflow.add_edge("report", END)

        # Compile (no human gates for maintenance)
        compile_kwargs: dict[str, Any] = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """Maintenance agent uses DB directly, no MCP tools."""
        return []

    def get_state_class(self) -> Type[MaintenanceAgentState]:
        return MaintenanceAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "duplicate_groups": [],
            "duplicate_count": 0,
            "merged_insights": [],
            "merge_count": 0,
            "pruned_count": 0,
            "pruned_ids": [],
            "report_summary": "",
            "total_insights_before": 0,
            "total_insights_after": 0,
        })
        return state

    # ─── Node 1: Scan Duplicates ──────────────────────────────────────

    async def _node_scan_duplicates(
        self, state: MaintenanceAgentState
    ) -> dict[str, Any]:
        """
        Scan shared_insights for near-duplicate entries.

        Uses cosine similarity threshold (default 0.95) to identify
        groups of insights that are semantically identical.
        """
        threshold = self.config.params.get("similarity_threshold", 0.95)
        max_groups = self.config.params.get("max_duplicate_groups", 50)

        logger.info(
            "maintenance_scan_started",
            extra={
                "agent_id": self.agent_id,
                "threshold": threshold,
            },
        )

        duplicate_groups: list[list[dict[str, Any]]] = []
        total_before = 0

        try:
            # Get all insights for this vertical
            all_insights = self.db.list_insights(
                vertical_id=self.vertical_id,
                limit=1000,
            )
            total_before = len(all_insights) if isinstance(all_insights, list) else 0

            # Group duplicates using embedding similarity
            # In production, this would use pgvector's cosine distance
            # For now, delegate to a DB RPC if available
            try:
                groups = self.db.find_duplicate_insights(
                    vertical_id=self.vertical_id,
                    similarity_threshold=threshold,
                    limit=max_groups,
                )
                duplicate_groups = groups if isinstance(groups, list) else []
            except (AttributeError, Exception) as e:
                logger.debug(
                    f"find_duplicate_insights not available: {e}. "
                    "Skipping duplicate scan (no RPC function)."
                )

        except Exception as e:
            logger.warning(
                "maintenance_scan_failed",
                extra={"error": str(e)[:200]},
            )

        duplicate_count = sum(len(g) for g in duplicate_groups)

        logger.info(
            "maintenance_scan_completed",
            extra={
                "agent_id": self.agent_id,
                "total_insights": total_before,
                "duplicate_groups": len(duplicate_groups),
                "duplicate_count": duplicate_count,
            },
        )

        return {
            "current_node": "scan_duplicates",
            "duplicate_groups": duplicate_groups,
            "duplicate_count": duplicate_count,
            "total_insights_before": total_before,
        }

    # ─── Node 2: Merge Insights ───────────────────────────────────────

    async def _node_merge_insights(
        self, state: MaintenanceAgentState
    ) -> dict[str, Any]:
        """
        Merge duplicate insight groups into Golden Insights using LLM.

        For each group of duplicates:
        1. Feed all duplicate contents to the LLM
        2. Generate one consolidated "Golden Insight"
        3. Store the golden insight
        4. Archive the originals
        """
        groups = state.get("duplicate_groups", [])
        max_merge_per_run = self.config.params.get("max_merges_per_run", 10)
        merged_insights: list[dict[str, Any]] = []

        if not groups:
            return {
                "current_node": "merge_insights",
                "merged_insights": [],
                "merge_count": 0,
            }

        logger.info(
            "maintenance_merge_started",
            extra={
                "agent_id": self.agent_id,
                "groups_to_merge": min(len(groups), max_merge_per_run),
            },
        )

        for group in groups[:max_merge_per_run]:
            if len(group) < 2:
                continue

            # Build merge prompt
            contents = "\n---\n".join(
                f"[{g.get('insight_type', '?')}] {g.get('content', '')[:300]}"
                for g in group
            )

            try:
                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=500,
                    temperature=0.2,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Consolidate these duplicate insights into ONE concise "
                            "golden insight. Preserve all unique information.\n\n"
                            f"Duplicates:\n{contents}\n\n"
                            "Return ONLY the consolidated insight text."
                        ),
                    }],
                )

                merged_text = (
                    response.content[0].text.strip() if response.content else ""
                )

                if merged_text:
                    # Store the golden insight
                    insight_type = group[0].get("insight_type", "winning_pattern")
                    try:
                        self.db.store_insight(
                            source_agent_id=self.agent_id,
                            insight_type=insight_type,
                            title=f"[Golden] {group[0].get('title', '')}",
                            content=merged_text,
                            confidence_score=0.9,
                            vertical_id=self.vertical_id,
                            metadata={"merged_from": len(group), "golden": True},
                        )
                    except Exception:
                        pass

                    merged_insights.append({
                        "original_count": len(group),
                        "merged_text": merged_text[:200],
                        "insight_type": insight_type,
                    })

                    # Archive originals (best-effort)
                    for original in group:
                        try:
                            self.db.archive_insight(original.get("id"))
                        except (AttributeError, Exception):
                            pass

            except Exception as e:
                logger.warning(
                    "maintenance_merge_group_failed",
                    extra={"error": str(e)[:200]},
                )

        logger.info(
            "maintenance_merge_completed",
            extra={
                "agent_id": self.agent_id,
                "merged_count": len(merged_insights),
            },
        )

        return {
            "current_node": "merge_insights",
            "merged_insights": merged_insights,
            "merge_count": len(merged_insights),
        }

    # ─── Node 3: Prune Decayed Insights ───────────────────────────────

    async def _node_prune_decayed(
        self, state: MaintenanceAgentState
    ) -> dict[str, Any]:
        """
        Flag or remove insights that are old and have low usage.

        Decay criteria:
        - Older than max_age_days (default: 90)
        - Low confidence score (< 0.3)
        """
        max_age_days = self.config.params.get("max_age_days", 90)
        min_confidence = self.config.params.get("min_confidence_for_retention", 0.3)
        cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat()

        pruned_ids: list[str] = []

        logger.info(
            "maintenance_prune_started",
            extra={
                "agent_id": self.agent_id,
                "max_age_days": max_age_days,
                "cutoff_date": cutoff_date,
            },
        )

        try:
            # Find stale insights
            stale = self.db.find_stale_insights(
                vertical_id=self.vertical_id,
                older_than=cutoff_date,
                min_confidence=min_confidence,
                limit=100,
            )

            if isinstance(stale, list):
                for insight in stale:
                    insight_id = insight.get("id")
                    if insight_id:
                        try:
                            self.db.archive_insight(insight_id)
                            pruned_ids.append(str(insight_id))
                        except (AttributeError, Exception):
                            pass

        except (AttributeError, Exception) as e:
            logger.debug(
                f"Prune functions not available: {e}. Skipping prune step."
            )

        logger.info(
            "maintenance_prune_completed",
            extra={
                "agent_id": self.agent_id,
                "pruned_count": len(pruned_ids),
            },
        )

        return {
            "current_node": "prune_decayed",
            "pruned_count": len(pruned_ids),
            "pruned_ids": pruned_ids,
        }

    # ─── Node 4: Report ───────────────────────────────────────────────

    async def _node_report(
        self, state: MaintenanceAgentState
    ) -> dict[str, Any]:
        """Generate a maintenance summary report."""
        total_before = state.get("total_insights_before", 0)
        duplicate_count = state.get("duplicate_count", 0)
        merge_count = state.get("merge_count", 0)
        pruned_count = state.get("pruned_count", 0)
        total_after = total_before - duplicate_count - pruned_count + merge_count

        report = (
            f"🧹 Maintenance Report\n"
            f"═══════════════════════\n"
            f"Total insights before: {total_before}\n"
            f"Duplicate groups found: {len(state.get('duplicate_groups', []))}\n"
            f"Duplicates merged: {merge_count} groups → golden insights\n"
            f"Stale insights pruned: {pruned_count}\n"
            f"Estimated insights after: {total_after}\n"
        )

        logger.info(
            "maintenance_report",
            extra={
                "agent_id": self.agent_id,
                "total_before": total_before,
                "merged": merge_count,
                "pruned": pruned_count,
                "total_after": total_after,
            },
        )

        return {
            "current_node": "report",
            "report_summary": report,
            "total_insights_after": max(total_after, 0),
            "knowledge_written": True,
        }

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """Maintenance agent doesn't write new knowledge — it cleans existing."""
        pass

    def __repr__(self) -> str:
        return (
            f"<MaintenanceAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
