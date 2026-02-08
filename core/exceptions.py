"""
Custom exception hierarchy for the Sovereign Venture Engine.

Structured error handling with clear categories:
- Configuration errors (caught at startup)
- Task execution errors (caught during agent runs)
- Retryable vs permanent failures
- Circuit breaker triggers
- Dependency failures (external APIs down)
- Human intervention required

Usage:
    from core.exceptions import TaskExecutionError, RetryableError

    try:
        result = await some_api_call()
    except httpx.TimeoutException as e:
        raise RetryableError("API timed out", retry_after_seconds=30) from e
"""

from __future__ import annotations

from typing import Optional


class EnclaveError(Exception):
    """
    Base exception for all Sovereign Venture Engine errors.

    All custom exceptions inherit from this, so you can catch
    `EnclaveError` to handle any platform-specific error.
    """

    def __init__(self, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}


# ── Configuration Errors ──────────────────────────────────────────


class AgentConfigurationError(EnclaveError):
    """
    Raised when an agent's YAML config or registry setup is invalid.

    Examples:
    - Missing required fields in YAML
    - agent_type not registered via @register_agent_type
    - Invalid parameter values
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        config_path: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.agent_id = agent_id
        self.config_path = config_path


class VerticalConfigError(EnclaveError):
    """
    Raised when a vertical's config.yaml is invalid or missing.
    """

    def __init__(
        self,
        message: str,
        *,
        vertical_id: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.vertical_id = vertical_id


# ── Task Execution Errors ─────────────────────────────────────────


class TaskExecutionError(EnclaveError):
    """
    Raised when an agent fails to execute a task.

    This is the general "something went wrong during agent run" error.
    For retryable failures, use RetryableError instead.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        node: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.agent_id = agent_id
        self.task_id = task_id
        self.node = node


class RetryableError(TaskExecutionError):
    """
    A task failure that can be retried (e.g., API timeout, rate limit).

    The task queue will re-enqueue the task with exponential backoff.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int = 60,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        node: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(
            message,
            agent_id=agent_id,
            task_id=task_id,
            node=node,
            details=details,
        )
        self.retry_after_seconds = retry_after_seconds


# ── Circuit Breaker ───────────────────────────────────────────────


class CircuitBreakerError(EnclaveError):
    """
    Raised when an agent has been auto-disabled due to too many
    consecutive failures.

    The agent must be manually re-enabled (via DB or dashboard)
    after the root cause is fixed.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        consecutive_errors: int = 0,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.agent_id = agent_id
        self.consecutive_errors = consecutive_errors


# ── Dependency Errors ─────────────────────────────────────────────


class DependencyError(EnclaveError):
    """
    Raised when an external dependency (API, database, service) is
    unavailable or returns an unexpected response.
    """

    def __init__(
        self,
        message: str,
        *,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.service = service
        self.status_code = status_code


# ── Human Intervention ────────────────────────────────────────────


class HumanInterventionRequired(EnclaveError):
    """
    Raised when a task requires human review before proceeding.

    This is NOT an error — it's a normal control flow signal
    used by human gates to pause the pipeline.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        gate_node: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.agent_id = agent_id
        self.task_id = task_id
        self.gate_node = gate_node


# ── Knowledge / RAG Errors ────────────────────────────────────────


class KnowledgeStoreError(EnclaveError):
    """
    Raised when reading from or writing to the RAG knowledge store fails.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details=details)
        self.operation = operation
