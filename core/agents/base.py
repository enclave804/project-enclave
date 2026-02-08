"""
Base agent class for the Sovereign Venture Engine.

All agents inherit from BaseAgent and implement:
- build_graph(): construct their LangGraph state machine
- get_tools(): return list of tools available to this agent

Design principles:
- Config-driven: behavior defined by YAML, not code
- Shared knowledge: all agents read/write the same RAG store
- Human-in-the-loop: configurable gates before critical actions
- Observable: every run logged to agent_runs table
- Circuit breaker: auto-disables after consecutive failures
- Confidence gating: only high-confidence insights enter the shared brain
- Neural router: cheap intent classification before waking the LLM
- Refinement loop: agents self-critique and improve output quality
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.contracts import InsightData
from core.agents.state import BaseAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all Sovereign Venture Engine agents.

    Lifecycle:
    1. __init__(): receives config, DB, and shared services
    2. build_graph(): constructs the agent's LangGraph state machine
    3. run(): route task → execute graph → refine output → write knowledge
    4. write_knowledge(): stores results back to shared RAG

    Safety:
    - Circuit breaker: tracks consecutive errors, auto-disables via DB
    - Confidence gating: insights below threshold are silently dropped

    Quality:
    - Neural router: _route_task() classifies intent before LLM invocation
    - Refinement loop: _run_refinement_loop() self-critiques output quality
    """

    agent_type: str = ""  # Override via @register_agent_type decorator

    def __init__(
        self,
        config: AgentInstanceConfig,
        db: Any,
        embedder: Any,
        anthropic_client: Any,
        checkpointer: Any = None,
        browser_tool: Any = None,
        mcp_tools: Optional[list[Any]] = None,
    ):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.llm = anthropic_client
        self.checkpointer = checkpointer
        self.browser_tool = browser_tool
        self.mcp_tools = mcp_tools or []
        self._graph: Any = None
        self._consecutive_errors: int = 0

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def vertical_id(self) -> str:
        return self.config.vertical_id

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def build_graph(self) -> Any:
        """Construct and return the compiled LangGraph for this agent."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """Return the list of tools this agent can use."""
        ...

    @abstractmethod
    def get_state_class(self) -> Type[BaseAgentState]:
        """Return the TypedDict state class for this agent's graph."""
        ...

    def get_graph(self) -> Any:
        """Get or build the agent's graph (lazy initialization)."""
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    async def run(
        self,
        task: dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute a task through this agent's graph.

        Includes circuit breaker logic: on success resets the error counter,
        on failure increments it and auto-disables if threshold is reached.

        Args:
            task: Task input data (agent-specific schema).
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final state after graph execution.
        """
        graph = self.get_graph()
        run_id = str(uuid.uuid4())
        thread_id = thread_id or run_id
        start_time = time.monotonic()

        # Log run start
        self._log_run(run_id, "started", input_data=task)
        logger.info(
            f"Agent '{self.agent_id}' starting run {run_id[:8]}... "
            f"(vertical={self.vertical_id})"
        )

        try:
            # Step 1: Neural Router — classify intent before LLM
            route_action = self._route_task(task)
            if route_action in ("sleep", "discard"):
                duration_ms = int((time.monotonic() - start_time) * 1000)
                self._log_run(
                    run_id,
                    "routed_away",
                    input_data=task,
                    output_data={"route_action": route_action},
                    duration_ms=duration_ms,
                )
                logger.info(
                    f"Agent '{self.agent_id}' task routed to '{route_action}' "
                    f"— skipping LLM execution ({duration_ms}ms)"
                )
                return {"route_action": route_action, "run_id": run_id}

            # Step 2: Execute graph
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = self._prepare_initial_state(task, run_id)
            result = await graph.ainvoke(initial_state, config=config)

            # Step 3: Refinement Loop — self-critique output quality
            result = await self._run_refinement_loop(result)

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Log completion
            self._log_run(
                run_id,
                "completed",
                output_data=self._sanitize_state(result),
                duration_ms=duration_ms,
            )
            logger.info(
                f"Agent '{self.agent_id}' completed run {run_id[:8]}... "
                f"({duration_ms}ms)"
            )

            # Circuit breaker: reset on success
            self._on_success()

            # Write knowledge back to shared RAG
            await self.write_knowledge(result)

            return result

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            self._log_run(
                run_id,
                "failed",
                error_message=str(e)[:500],
                duration_ms=duration_ms,
            )
            logger.error(
                f"Agent '{self.agent_id}' failed run {run_id[:8]}...: {e}"
            )

            # Circuit breaker: record failure, may auto-disable
            self._on_failure(str(e)[:500])

            raise

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        """Reset consecutive error counter on successful run."""
        self._consecutive_errors = 0
        try:
            self.db.reset_agent_errors(
                agent_id=self.agent_id,
                vertical_id=self.vertical_id,
            )
        except Exception as err:
            logger.debug(f"Could not reset agent errors in DB: {err}")

    def _on_failure(self, error_message: str) -> None:
        """
        Increment consecutive error counter.
        If threshold is reached, auto-disable via DB (circuit breaker).
        """
        self._consecutive_errors += 1
        try:
            result = self.db.record_agent_error(
                agent_id=self.agent_id,
                vertical_id=self.vertical_id,
                error_message=error_message,
            )
            if result and result.get("status") == "disabled":
                logger.critical(
                    f"CIRCUIT BREAKER: Agent '{self.agent_id}' auto-disabled "
                    f"after {self._consecutive_errors} consecutive errors. "
                    f"Last error: {error_message[:200]}"
                )
        except Exception as err:
            logger.debug(f"Could not record agent error in DB: {err}")

    # ------------------------------------------------------------------
    # Neural Router (cheap intent classification before LLM)
    # ------------------------------------------------------------------

    def _route_task(self, task: dict[str, Any]) -> str:
        """
        Route a task through the neural router if configured.

        Checks the task against the routing config's intent_actions map.
        Returns the action to take:
        - "proceed": continue to LLM (normal execution)
        - "sleep": skip this task silently
        - "discard": drop the task entirely
        - custom string: a handler name for the agent to dispatch

        If routing is disabled, always returns "proceed".

        Override in subclass to integrate with a real classifier
        (e.g., BERT intent model). The base implementation checks
        for a "classified_intent" key in the task dict.
        """
        routing = self.config.routing
        if not routing.enabled:
            return "proceed"

        # Look for a pre-classified intent in the task
        # (future: call a local BERT/YOLO model here)
        classified_intent = task.get("classified_intent", "")
        confidence = task.get("classification_confidence", 0.0)

        if not classified_intent:
            logger.debug(
                f"Agent '{self.agent_id}': no classified_intent in task, "
                f"falling back to '{routing.fallback_action}'"
            )
            return routing.fallback_action

        # Check if confidence meets threshold
        if confidence < routing.confidence_threshold:
            logger.info(
                f"Agent '{self.agent_id}': router confidence {confidence:.2f} "
                f"< threshold {routing.confidence_threshold:.2f}, "
                f"falling back to '{routing.fallback_action}'"
            )
            return routing.fallback_action

        # Look up the action for this intent
        action = routing.intent_actions.get(
            classified_intent, routing.fallback_action
        )
        logger.info(
            f"Agent '{self.agent_id}': routed intent '{classified_intent}' "
            f"(confidence={confidence:.2f}) → action '{action}'"
        )
        return action

    # ------------------------------------------------------------------
    # Refinement Loop (self-correction)
    # ------------------------------------------------------------------

    async def _run_refinement_loop(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run the self-correction refinement loop if configured.

        Evaluates the agent's output against the critic_prompt rubric.
        If quality is below threshold, loops back for refinement up to
        max_iterations times.

        Override in subclass to implement the actual critic call.
        The base implementation is a pass-through that logs whether
        refinement is enabled.

        Args:
            result: The agent's raw output state dict.

        Returns:
            The refined (or original) result state dict.
        """
        refinement = self.config.refinement
        if not refinement.enabled:
            return result

        if not refinement.critic_prompt:
            logger.warning(
                f"Agent '{self.agent_id}': refinement enabled but no "
                f"critic_prompt configured — skipping refinement"
            )
            return result

        logger.info(
            f"Agent '{self.agent_id}': refinement loop enabled "
            f"(max_iterations={refinement.max_iterations}, "
            f"quality_threshold={refinement.quality_threshold})"
        )

        # Placeholder: subclasses implement the actual critic → refine cycle.
        # When implemented, this will:
        # 1. Send result + critic_prompt to LLM
        # 2. Parse quality score from response
        # 3. If score < quality_threshold, refine and loop
        # 4. Return the best result after max_iterations

        return result

    # ------------------------------------------------------------------
    # Knowledge Writing (with confidence gating)
    # ------------------------------------------------------------------

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Write agent results back to the shared RAG store.
        Override in subclass to define what knowledge this agent produces.

        Subclasses should call `self.store_insight(insight)` which enforces
        the confidence threshold from config.
        """
        pass  # Default: no-op. Subclasses implement.

    def store_insight(self, insight: InsightData) -> Optional[dict]:
        """
        Store an insight in the shared brain, gated by confidence threshold.

        Returns the stored record if accepted, None if rejected.
        """
        threshold = self.config.rag_write_confidence_threshold
        if insight.confidence < threshold:
            logger.info(
                f"Insight dropped (confidence {insight.confidence:.2f} "
                f"< threshold {threshold:.2f}): {insight.title or insight.content[:60]}"
            )
            return None

        try:
            return self.db.store_insight(
                source_agent_id=self.agent_id,
                insight_type=insight.insight_type,
                title=insight.title,
                content=insight.content,
                confidence_score=insight.confidence,
                related_entity_id=insight.related_entity_id,
                related_entity_type=insight.related_entity_type,
                metadata=insight.metadata,
                vertical_id=self.vertical_id,
            )
        except Exception as err:
            logger.warning(f"Failed to store insight: {err}")
            return None

    # ------------------------------------------------------------------
    # State Preparation
    # ------------------------------------------------------------------

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """
        Prepare the initial state for graph invocation.
        Override in subclass for agent-specific initialization.
        """
        return {
            "agent_id": self.agent_id,
            "vertical_id": self.vertical_id,
            "run_id": run_id,
            "task_input": task,
            "current_node": "start",
            "error": None,
            "retry_count": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "requires_human_approval": False,
            "knowledge_written": False,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_run(
        self,
        run_id: str,
        status: str,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Log an agent run to the database (best-effort, never raises)."""
        try:
            self.db.log_agent_run(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                run_id=run_id,
                status=status,
                input_data=input_data,
                output_data=output_data,
                error_message=error_message,
                duration_ms=duration_ms,
                vertical_id=self.vertical_id,
            )
        except Exception as log_err:
            logger.warning(f"Failed to log agent run: {log_err}")

    def _sanitize_state(self, state: dict) -> dict:
        """Remove large/sensitive fields before logging to DB."""
        skip_keys = {
            "raw_apollo_data",
            "embedding",
            "browser_screenshot",
            "draft_email_body",
            "draft_content",
            "rag_context",
        }
        return {
            k: v
            for k, v in state.items()
            if k not in skip_keys and not isinstance(v, (bytes, memoryview))
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
