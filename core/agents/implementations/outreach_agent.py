"""
Outreach Agent — Strangler-fig adapter for the legacy pipeline.

Wraps the existing `build_pipeline_graph()` from core/graph/workflow_engine.py
so it can be managed by the new agent framework without ANY changes to
the proven graph code.

This adapter:
- Delegates build_graph() to the existing build_pipeline_graph()
- Delegates run() to the existing process_lead() / process_batch()
- Maps LeadState <-> BaseAgentState transparently
- Respects existing human gates and compliance checks
- Zero changes to core/graph/

After all Phase 1 agents are stable, we can optionally rewrite outreach
as a tool-using ReAct agent. For now, wrapping is safest.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.registry import register_agent_type
from core.agents.state import OutreachAgentState
from core.config.agent_schema import AgentInstanceConfig
from core.exceptions import AgentConfigurationError, TaskExecutionError

logger = logging.getLogger(__name__)


@register_agent_type("outreach")
class OutreachAgent(BaseAgent):
    """
    Agent wrapper around the existing outreach pipeline.

    Adapter pattern: translates between the agent framework's run(task)
    interface and the legacy pipeline's process_lead(graph, lead_data, ...).
    """

    def __init__(
        self,
        config: AgentInstanceConfig,
        db: Any,
        embedder: Any,
        anthropic_client: Any,
        checkpointer: Any = None,
        browser_tool: Any = None,
        mcp_tools: Optional[list[Any]] = None,
        apollo: Any = None,
        vertical_config: Any = None,
    ):
        super().__init__(
            config=config,
            db=db,
            embedder=embedder,
            anthropic_client=anthropic_client,
            checkpointer=checkpointer,
            browser_tool=browser_tool,
            mcp_tools=mcp_tools,
        )
        # Extra dependencies for the legacy pipeline
        self._apollo = apollo
        self._vertical_config = vertical_config

    # ── Graph Construction ────────────────────────────────────────

    def build_graph(self) -> Any:
        """
        Build the graph by delegating to the legacy workflow engine.

        Requires apollo and vertical_config to be set (via set_legacy_deps
        or constructor). Raises AgentConfigurationError if not provided.
        """
        if self._vertical_config is None:
            raise AgentConfigurationError(
                "OutreachAgent requires vertical_config for graph construction. "
                "Call set_legacy_deps() or pass vertical_config= in constructor.",
                agent_id=self.agent_id,
            )

        from core.graph.workflow_engine import build_pipeline_graph

        return build_pipeline_graph(
            config=self._vertical_config,
            db=self.db,
            apollo=self._apollo,
            embedder=self.embedder,
            anthropic_client=self.llm,
            checkpointer=self.checkpointer,
            test_mode=False,
        )

    def get_tools(self) -> list[Any]:
        """
        The legacy pipeline doesn't use a tool-calling pattern.
        Tools are baked into the graph nodes themselves.
        """
        return []

    def get_state_class(self) -> Type[OutreachAgentState]:
        return OutreachAgentState

    # ── Configuration ─────────────────────────────────────────────

    def set_legacy_deps(
        self,
        apollo: Any,
        vertical_config: Any,
    ) -> None:
        """
        Set the legacy dependencies needed by build_pipeline_graph().

        Called by the CLI or orchestrator after basic construction.
        Invalidates any cached graph so it gets rebuilt on next run.
        """
        self._apollo = apollo
        self._vertical_config = vertical_config
        self._graph = None  # Force rebuild

    # ── Task Execution ────────────────────────────────────────────

    async def run(
        self,
        task: dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute an outreach task through the legacy pipeline.

        The task dict should contain lead data in one of these formats:
        1. Single lead: {"lead": {"contact": {...}, "company": {...}}}
        2. Batch: {"leads": [{"contact": {...}, "company": {...}}, ...]}

        Falls back to BaseAgent.run() for standard lifecycle logging
        and circuit breaker behavior.
        """
        # If task contains "leads" (batch mode), delegate to process_batch
        if "leads" in task:
            return await self._run_batch(task, thread_id)

        # Single lead mode: use BaseAgent.run() for lifecycle
        return await super().run(task, thread_id)

    async def _run_batch(
        self,
        task: dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process a batch of leads through the legacy pipeline."""
        from core.graph.workflow_engine import process_batch

        graph = self.get_graph()
        leads = task.get("leads", [])
        if not leads:
            logger.warning(f"Agent '{self.agent_id}': empty batch, nothing to do")
            return {"total": 0, "processed": 0, "sent": 0, "skipped": 0, "errors": 0}

        logger.info(
            f"Agent '{self.agent_id}' processing batch of {len(leads)} leads"
        )

        try:
            result = await process_batch(graph, leads, self.vertical_id)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(str(e)[:500])
            raise TaskExecutionError(
                f"Batch processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    # ── State Preparation (override for legacy compat) ────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """
        Prepare initial state for a single lead.

        Translates from the agent framework's task dict to the legacy
        pipeline's LeadState format via create_initial_lead_state().
        """
        from core.graph.state import create_initial_lead_state

        lead_data = task.get("lead", task)
        state = create_initial_lead_state(lead_data, self.vertical_id)
        state["pipeline_run_id"] = run_id
        return state

    # ── Knowledge Writing ─────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        No-op: the legacy pipeline's write_to_rag node handles this.

        The existing graph already writes knowledge chunks at the end
        of every run, so we don't double-write here.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"<OutreachAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r} "
            f"legacy_adapter=True>"
        )
