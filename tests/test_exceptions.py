"""
Unit tests for the custom exception hierarchy.

Validates exception creation, inheritance, and attribute storage.
"""

import pytest

from core.exceptions import (
    EnclaveError,
    AgentConfigurationError,
    VerticalConfigError,
    TaskExecutionError,
    RetryableError,
    CircuitBreakerError,
    DependencyError,
    HumanInterventionRequired,
    KnowledgeStoreError,
)


class TestEnclaveError:
    """Tests for the base exception class."""

    def test_basic_creation(self):
        err = EnclaveError("something broke")
        assert str(err) == "something broke"
        assert err.details == {}

    def test_with_details(self):
        err = EnclaveError("oops", details={"code": 42})
        assert err.details["code"] == 42

    def test_is_exception(self):
        assert issubclass(EnclaveError, Exception)


class TestAgentConfigurationError:
    """Tests for agent configuration errors."""

    def test_inherits_enclave_error(self):
        assert issubclass(AgentConfigurationError, EnclaveError)

    def test_stores_agent_id(self):
        err = AgentConfigurationError(
            "bad config", agent_id="outreach", config_path="agents/outreach.yaml"
        )
        assert err.agent_id == "outreach"
        assert err.config_path == "agents/outreach.yaml"

    def test_catchable_as_enclave_error(self):
        with pytest.raises(EnclaveError):
            raise AgentConfigurationError("invalid", agent_id="x")


class TestVerticalConfigError:
    """Tests for vertical config errors."""

    def test_stores_vertical_id(self):
        err = VerticalConfigError("missing yaml", vertical_id="enclave_guard")
        assert err.vertical_id == "enclave_guard"

    def test_inherits_enclave_error(self):
        assert issubclass(VerticalConfigError, EnclaveError)


class TestTaskExecutionError:
    """Tests for task execution errors."""

    def test_stores_context(self):
        err = TaskExecutionError(
            "node failed",
            agent_id="outreach",
            task_id="task-001",
            node="enrich_company",
        )
        assert err.agent_id == "outreach"
        assert err.task_id == "task-001"
        assert err.node == "enrich_company"

    def test_inherits_enclave_error(self):
        assert issubclass(TaskExecutionError, EnclaveError)


class TestRetryableError:
    """Tests for retryable errors."""

    def test_stores_retry_delay(self):
        err = RetryableError(
            "timeout", retry_after_seconds=30, agent_id="outreach"
        )
        assert err.retry_after_seconds == 30
        assert err.agent_id == "outreach"

    def test_default_retry_delay(self):
        err = RetryableError("timeout")
        assert err.retry_after_seconds == 60

    def test_inherits_task_execution_error(self):
        assert issubclass(RetryableError, TaskExecutionError)

    def test_catchable_as_task_execution_error(self):
        with pytest.raises(TaskExecutionError):
            raise RetryableError("API down")


class TestCircuitBreakerError:
    """Tests for circuit breaker errors."""

    def test_stores_error_count(self):
        err = CircuitBreakerError(
            "too many failures",
            agent_id="outreach",
            consecutive_errors=5,
        )
        assert err.agent_id == "outreach"
        assert err.consecutive_errors == 5

    def test_inherits_enclave_error(self):
        assert issubclass(CircuitBreakerError, EnclaveError)


class TestDependencyError:
    """Tests for dependency errors."""

    def test_stores_service_info(self):
        err = DependencyError(
            "Apollo API down", service="apollo", status_code=503
        )
        assert err.service == "apollo"
        assert err.status_code == 503

    def test_inherits_enclave_error(self):
        assert issubclass(DependencyError, EnclaveError)


class TestHumanInterventionRequired:
    """Tests for human intervention signal."""

    def test_stores_gate_info(self):
        err = HumanInterventionRequired(
            "Review required",
            agent_id="outreach",
            task_id="task-001",
            gate_node="send_outreach",
        )
        assert err.agent_id == "outreach"
        assert err.gate_node == "send_outreach"

    def test_inherits_enclave_error(self):
        assert issubclass(HumanInterventionRequired, EnclaveError)


class TestKnowledgeStoreError:
    """Tests for knowledge store errors."""

    def test_stores_operation(self):
        err = KnowledgeStoreError("write failed", operation="store_insight")
        assert err.operation == "store_insight"

    def test_inherits_enclave_error(self):
        assert issubclass(KnowledgeStoreError, EnclaveError)


class TestExceptionHierarchy:
    """Verify the full inheritance chain."""

    def test_all_inherit_from_enclave_error(self):
        """Every custom exception should be catchable as EnclaveError."""
        exceptions = [
            AgentConfigurationError,
            VerticalConfigError,
            TaskExecutionError,
            RetryableError,
            CircuitBreakerError,
            DependencyError,
            HumanInterventionRequired,
            KnowledgeStoreError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, EnclaveError), (
                f"{exc_class.__name__} does not inherit from EnclaveError"
            )

    def test_retryable_is_task_execution(self):
        """RetryableError should be catchable as TaskExecutionError."""
        assert issubclass(RetryableError, TaskExecutionError)
