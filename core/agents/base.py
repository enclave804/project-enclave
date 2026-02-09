"""
Base agent class for the Sovereign Venture Engine.

All agents inherit from BaseAgent and implement:
- build_graph(): construct their LangGraph state machine
- get_tools(): return list of tools available to this agent

Design principles:
- Config-driven: behavior defined by YAML, not code
- Shared knowledge: all agents read/write the same RAG store
- Human-in-the-loop: configurable gates before critical actions
- Observable: every run logged to agent_runs table + LangFuse traces
- Circuit breaker: auto-disables after consecutive failures
- Confidence gating: only high-confidence insights enter the shared brain
- Neural router: cheap intent classification before waking the LLM
- Refinement loop: agents self-critique and improve output quality
- Feature flags: dynamic config from DB for safe rollouts
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.contracts import InsightData
from core.agents.state import BaseAgentState
from core.config.agent_schema import AgentInstanceConfig
from core.observability.tracing import (
    get_tracer,
    create_trace,
    create_span,
    end_span,
    flush_tracer,
    NoOpTrace,
)
from core.safety.input_guard import SecurityGuard, SecurityException, get_guard

logger = logging.getLogger(__name__)


def _hash_to_bucket(key: str, buckets: int = 100) -> int:
    """
    Deterministically hash a string to a 0-(buckets-1) integer.

    Used for feature flag percentage rollouts: the same key always
    maps to the same bucket, ensuring consistent behavior per lead.
    """
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16) % buckets


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

    Observability:
    - LangFuse tracing: full execution path visualized on a timeline
    - Feature flags: dynamic rollout via DB config (check_feature_flag)
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
        *,
        router: Any = None,
        openai_client: Any = None,
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

        # --- Phase 6: Multi-Model Router ---
        # If a router is provided, use it for intent-based model routing.
        # Otherwise, create one from the anthropic_client + optional openai_client.
        # self.llm remains available for backward compatibility.
        if router is not None:
            self.router = router
        else:
            from core.llm.router import ModelRouter
            self.router = ModelRouter(
                anthropic_client=anthropic_client,
                openai_client=openai_client,
            )

        # Vision client (lazy — uses router when called)
        self._vision: Any = None

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def vertical_id(self) -> str:
        return self.config.vertical_id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def vision(self) -> Any:
        """Lazy-initialized VisionClient for image analysis."""
        if self._vision is None:
            from core.llm.vision import VisionClient
            self._vision = VisionClient(router=self.router)
        return self._vision

    async def route_llm(
        self,
        intent: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> Any:
        """
        Convenience: route an LLM call through the ModelRouter.

        Instead of self.llm.messages.create(), agents can use:
            response = await self.route_llm(
                intent="creative_writing",
                system_prompt="You are a copywriter.",
                user_prompt="Write a tweet about AI.",
            )
            text = response.text

        Falls back to self.llm (Anthropic) if routing fails.
        """
        return await self.router.route(
            intent=intent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

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

        Full lifecycle: route → execute → refine → write knowledge.
        Includes circuit breaker, distributed tracing, and feature flags.

        Args:
            task: Task input data (agent-specific schema).
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final state after graph execution.
        """
        # ── Security Airlock: scan task input for prompt injection ──
        # Recursively validates all string values in the task dict.
        # Raises SecurityException if any injection pattern is detected.
        _guard = get_guard()
        for _key, _val in task.items():
            if isinstance(_val, str):
                _guard.validate(_val)
            elif isinstance(_val, dict):
                if not _guard.scan_dict(_val):
                    raise SecurityException(
                        f"Prompt injection detected in task field '{_key}'",
                        pattern_name="nested_injection",
                        input_preview=str(_val)[:100],
                    )

        graph = self.get_graph()
        run_id = str(uuid.uuid4())
        thread_id = thread_id or run_id
        start_time = time.monotonic()

        # Initialize distributed trace (no-op if LangFuse not configured)
        tracer = get_tracer()
        trace = create_trace(
            tracer,
            name=f"{self.agent_type}-run",
            agent_id=self.agent_id,
            vertical_id=self.vertical_id,
            run_id=run_id,
        )
        self._current_trace = trace

        # Log run start
        self._log_run(run_id, "started", input_data=task)
        logger.info(
            "agent_run_started",
            extra={
                "agent_id": self.agent_id,
                "run_id": run_id[:8],
                "vertical_id": self.vertical_id,
            },
        )

        try:
            # Step 1: Neural Router — classify intent before LLM
            route_span = create_span(
                trace, name="neural_router", input_data=task
            )
            route_action = self._route_task(task)
            end_span(route_span, output_data={"action": route_action})

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
                    "agent_task_routed",
                    extra={
                        "agent_id": self.agent_id,
                        "route_action": route_action,
                        "duration_ms": duration_ms,
                    },
                )
                trace.update(
                    output={"route_action": route_action, "run_id": run_id}
                )
                flush_tracer(tracer)
                return {"route_action": route_action, "run_id": run_id}

            # Step 2: Execute graph
            graph_span = create_span(
                trace,
                name="graph_execution",
                input_data={"thread_id": thread_id},
            )
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = self._prepare_initial_state(task, run_id)
            result = await graph.ainvoke(initial_state, config=config)
            end_span(graph_span, output_data=self._sanitize_state(result))

            # Step 3: Refinement Loop — self-critique output quality
            refine_span = create_span(trace, name="refinement_loop")
            result = await self._run_refinement_loop(result)
            end_span(refine_span, output_data={"refined": True})

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Log completion
            self._log_run(
                run_id,
                "completed",
                output_data=self._sanitize_state(result),
                duration_ms=duration_ms,
            )
            logger.info(
                "agent_run_completed",
                extra={
                    "agent_id": self.agent_id,
                    "run_id": run_id[:8],
                    "duration_ms": duration_ms,
                    "status": "completed",
                },
            )

            # Circuit breaker: reset on success
            self._on_success()

            # Write knowledge back to shared RAG
            knowledge_span = create_span(trace, name="write_knowledge")
            await self.write_knowledge(result)
            end_span(knowledge_span, output_data={"written": True})

            trace.update(
                output=self._sanitize_state(result),
                metadata={"duration_ms": duration_ms},
            )
            flush_tracer(tracer)

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
                "agent_run_failed",
                extra={
                    "agent_id": self.agent_id,
                    "run_id": run_id[:8],
                    "duration_ms": duration_ms,
                    "error": str(e)[:200],
                },
            )

            # Record error in trace
            trace.update(
                output={"error": str(e)[:500]},
                metadata={"duration_ms": duration_ms},
            )
            flush_tracer(tracer)

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
    # RLHF Data Collection (God Mode Lite)
    # ------------------------------------------------------------------

    def learn(
        self,
        task_input: dict[str, Any],
        model_output: str,
        human_correction: Optional[str] = None,
        score: Optional[int] = None,
        source: str = "manual_review",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        """
        Save a training example to the RLHF database.

        The Data Flywheel: every time a human corrects an agent's output,
        we save the (bad_draft, good_rewrite) pair. Over time this builds
        a fine-tuning dataset for prompt optimization or model training.

        This method is best-effort — it NEVER crashes the agent if the
        database write fails. Learning failures are logged but swallowed.

        Args:
            task_input: The context/prompt that produced the output.
            model_output: What the agent generated (the "candidate").
            human_correction: What the human rewrote (the "gold"). None
                if the human only scored but didn't rewrite.
            score: Quality score 0-100. None if not rated.
            source: How this example was collected:
                'manual_review', 'shadow_comparison', 'a_b_test', 'automated_eval'
            metadata: Optional metadata for filtering (e.g., lead industry).

        Returns:
            Stored training example record, or None on failure.
        """
        try:
            result = self.db.store_training_example(
                agent_id=self.agent_id,
                vertical_id=self.vertical_id,
                task_input=task_input,
                model_output=model_output,
                human_correction=human_correction,
                score=score,
                source=source,
                metadata=metadata or {},
            )
            logger.info(
                "rlhf_data_captured",
                extra={
                    "agent_id": self.agent_id,
                    "score": score,
                    "has_correction": human_correction is not None,
                    "source": source,
                },
            )
            return result
        except Exception as e:
            # Never crash the agent because learning failed
            logger.warning(
                "rlhf_learning_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            return None

    def get_training_examples(
        self,
        min_score: Optional[int] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve training examples for this agent.

        Used for few-shot prompt construction or training data export.

        Args:
            min_score: Only examples with score >= this value.
            source: Filter by collection source.
            limit: Max examples to return.

        Returns:
            List of training example records.
        """
        try:
            return self.db.get_training_examples(
                agent_id=self.agent_id,
                vertical_id=self.vertical_id,
                min_score=min_score,
                source=source,
                limit=limit,
            )
        except Exception as e:
            logger.warning(
                "rlhf_retrieval_failed",
                extra={
                    "agent_id": self.agent_id,
                    "error": str(e)[:200],
                },
            )
            return []

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
    # Feature Flags (dynamic config from DB)
    # ------------------------------------------------------------------

    def check_feature_flag(
        self, flag_name: str, context: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        Check if a feature flag is enabled for this agent.

        Uses the agent's config stored in the DB `agents` table.
        Supports percentage-based rollouts by hashing a deterministic
        key from the context (e.g., lead_id) so the same lead always
        gets the same decision.

        Flag structure in agent config JSON:
            {
                "flags": {
                    "aggressive_mode": {
                        "enabled": true,
                        "rollout_percent": 10
                    },
                    "new_email_template": {
                        "enabled": true
                    }
                }
            }

        Args:
            flag_name: Name of the feature flag.
            context: Optional context dict. If the flag has
                     rollout_percent, context['lead_id'] (or
                     context['entity_id']) is used as the hash key
                     for deterministic bucketing.

        Returns:
            True if the flag is enabled (and the context qualifies
            under the rollout percentage), False otherwise.
        """
        context = context or {}

        # Try reading flags from the agent's config params first
        # (available without a DB call, set via YAML or runtime)
        flags = self.config.params.get("flags", {})

        # If no local flags, try loading from DB (best-effort)
        if not flags:
            flags = self._load_flags_from_db()

        flag = flags.get(flag_name)
        if flag is None:
            return False

        # Simple boolean flag
        if isinstance(flag, bool):
            return flag

        # Structured flag: { "enabled": true, "rollout_percent": 10 }
        if not isinstance(flag, dict):
            return False

        if not flag.get("enabled", False):
            return False

        # Full rollout (no percentage specified)
        rollout_percent = flag.get("rollout_percent")
        if rollout_percent is None:
            return True

        # Percentage-based rollout: deterministic hash bucketing
        hash_key = (
            context.get("lead_id")
            or context.get("entity_id")
            or context.get("task_id")
            or ""
        )
        if not hash_key:
            # No deterministic key — fall back to False to be safe
            logger.debug(
                f"Feature flag '{flag_name}': no lead_id/entity_id in "
                f"context for rollout bucketing — defaulting to False"
            )
            return False

        # Hash the key to get a deterministic 0-99 bucket
        bucket = _hash_to_bucket(f"{self.agent_id}:{flag_name}:{hash_key}")
        result = bucket < rollout_percent

        logger.debug(
            f"Feature flag '{flag_name}': bucket={bucket}, "
            f"rollout={rollout_percent}% → {'enabled' if result else 'disabled'}"
        )
        return result

    def _load_flags_from_db(self) -> dict[str, Any]:
        """
        Load feature flags from the DB agent config (best-effort).

        Returns empty dict on any failure.
        """
        try:
            agent_record = self.db.get_agent(
                agent_id=self.agent_id,
                vertical_id=self.vertical_id,
            )
            if agent_record and isinstance(agent_record, dict):
                config = agent_record.get("config", {})
                if isinstance(config, dict):
                    return config.get("flags", {})
        except Exception as err:
            logger.debug(f"Could not load feature flags from DB: {err}")
        return {}

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
