"""
Tests for the structured logging configuration.

Validates:
- JSONFormatter produces valid JSON with required fields
- DevFormatter produces human-readable colored text
- ContextFilter injects trace_id from thread-local storage
- configure_logging() switches mode based on ENCLAVE_ENV
- Extra fields (agent_id, run_id) appear in JSON output
"""

from __future__ import annotations

import json
import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest

from core.observability.logging_config import (
    JSONFormatter,
    DevFormatter,
    ContextFilter,
    configure_logging,
    set_trace_id,
    get_trace_id,
    clear_trace_id,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _cleanup_trace_id():
    """Clear trace_id before and after each test."""
    clear_trace_id()
    yield
    clear_trace_id()


@pytest.fixture
def json_formatter():
    return JSONFormatter()


@pytest.fixture
def dev_formatter():
    return DevFormatter()


@pytest.fixture
def context_filter():
    return ContextFilter()


def _make_record(
    msg: str = "test message",
    level: int = logging.INFO,
    name: str = "test.logger",
    extra: dict | None = None,
) -> logging.LogRecord:
    """Create a LogRecord with optional extra fields."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=42,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


# ─── JSONFormatter Tests ──────────────────────────────────────────────


class TestJSONFormatter:
    """Tests for the production JSON log formatter."""

    def test_produces_valid_json(self, json_formatter):
        """JSONFormatter output must be valid JSON."""
        record = _make_record("hello world")
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_includes_required_fields(self, json_formatter):
        """JSON output includes timestamp, level, logger, message."""
        record = _make_record("test", level=logging.WARNING, name="core.agents")
        output = json_formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" in parsed
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "core.agents"
        assert parsed["message"] == "test"

    def test_includes_extra_fields(self, json_formatter):
        """Extra fields passed via logger.info(..., extra={}) appear in JSON."""
        record = _make_record(
            "agent_run_started",
            extra={"agent_id": "outreach", "run_id": "abc-123"},
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)

        assert parsed["agent_id"] == "outreach"
        assert parsed["run_id"] == "abc-123"

    def test_includes_trace_id(self, json_formatter):
        """trace_id from ContextFilter is included in JSON output."""
        record = _make_record("test", extra={"trace_id": "trace-xyz"})
        output = json_formatter.format(record)
        parsed = json.loads(output)

        assert parsed["trace_id"] == "trace-xyz"

    def test_timestamp_is_iso_format(self, json_formatter):
        """Timestamp should be ISO 8601 format."""
        record = _make_record("test")
        output = json_formatter.format(record)
        parsed = json.loads(output)

        # ISO format has 'T' separator and contains year
        timestamp = parsed["timestamp"]
        assert "T" in timestamp
        assert "20" in timestamp  # year

    def test_handles_exception_info(self, json_formatter):
        """Exception info is included in the JSON output."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = _make_record("error occurred")
            record.exc_info = sys.exc_info()

        output = json_formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "test error" in parsed["exception"]

    def test_non_serializable_extra_becomes_string(self, json_formatter):
        """Non-JSON-serializable extra values are converted to strings."""
        record = _make_record(
            "test",
            extra={"complex_obj": object()},
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)

        # Should be stringified, not crash
        assert "complex_obj" in parsed
        assert isinstance(parsed["complex_obj"], str)


# ─── DevFormatter Tests ───────────────────────────────────────────────


class TestDevFormatter:
    """Tests for the local development colored formatter."""

    def test_includes_message(self, dev_formatter):
        """DevFormatter includes the log message."""
        record = _make_record("hello dev world")
        output = dev_formatter.format(record)
        assert "hello dev world" in output

    def test_includes_level(self, dev_formatter):
        """DevFormatter includes the log level."""
        record = _make_record("test", level=logging.WARNING)
        output = dev_formatter.format(record)
        assert "WARNING" in output

    def test_includes_logger_name(self, dev_formatter):
        """DevFormatter includes the logger name."""
        record = _make_record("test", name="core.agents.base")
        output = dev_formatter.format(record)
        assert "core.agents.base" in output

    def test_includes_extra_fields_inline(self, dev_formatter):
        """DevFormatter shows known extra fields inline as key=value."""
        record = _make_record(
            "test",
            extra={"agent_id": "outreach", "trace_id": "trace-abc"},
        )
        output = dev_formatter.format(record)
        assert "agent_id=outreach" in output
        assert "trace_id=trace-abc" in output

    def test_color_codes_present_for_error(self, dev_formatter):
        """Error level should have red color codes."""
        record = _make_record("error!", level=logging.ERROR)
        output = dev_formatter.format(record)
        # ANSI red: \033[31m
        assert "\033[31m" in output


# ─── ContextFilter Tests ──────────────────────────────────────────────


class TestContextFilter:
    """Tests for the trace_id injection filter."""

    def test_injects_trace_id_when_set(self, context_filter):
        """ContextFilter adds trace_id to record when set in thread-local."""
        set_trace_id("trace-123")
        record = _make_record("test")
        context_filter.filter(record)
        assert hasattr(record, "trace_id")
        assert record.trace_id == "trace-123"  # type: ignore[attr-defined]

    def test_no_trace_id_when_not_set(self, context_filter):
        """ContextFilter doesn't add trace_id when not set."""
        record = _make_record("test")
        context_filter.filter(record)
        assert not hasattr(record, "trace_id")

    def test_always_returns_true(self, context_filter):
        """ContextFilter should never suppress log records."""
        record = _make_record("test")
        assert context_filter.filter(record) is True


# ─── Trace Context Helpers ────────────────────────────────────────────


class TestTraceContext:
    """Tests for set_trace_id / get_trace_id / clear_trace_id."""

    def test_set_and_get(self):
        """set_trace_id stores value retrievable by get_trace_id."""
        set_trace_id("my-trace")
        assert get_trace_id() == "my-trace"

    def test_clear(self):
        """clear_trace_id removes the stored trace_id."""
        set_trace_id("to-clear")
        clear_trace_id()
        assert get_trace_id() is None

    def test_get_returns_none_by_default(self):
        """get_trace_id returns None when nothing is set."""
        assert get_trace_id() is None


# ─── configure_logging Tests ──────────────────────────────────────────


class TestConfigureLogging:
    """Tests for the configure_logging() entry point."""

    def test_production_uses_json_formatter(self):
        """In production env, root logger uses JSONFormatter."""
        configure_logging(env="production")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, JSONFormatter)
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_development_uses_dev_formatter(self):
        """In development env, root logger uses DevFormatter."""
        configure_logging(env="development")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, DevFormatter)
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_reads_env_var(self):
        """configure_logging reads ENCLAVE_ENV when no arg given."""
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            configure_logging()
            root = logging.getLogger()
            formatter = root.handlers[0].formatter
            assert isinstance(formatter, JSONFormatter)
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_defaults_to_development(self):
        """Without ENCLAVE_ENV, defaults to development (DevFormatter)."""
        with patch.dict(os.environ, {}, clear=True):
            # Also clear ENCLAVE_ENV if it exists
            os.environ.pop("ENCLAVE_ENV", None)
            configure_logging()
            root = logging.getLogger()
            formatter = root.handlers[0].formatter
            assert isinstance(formatter, DevFormatter)
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_removes_existing_handlers(self):
        """configure_logging clears previous handlers (no duplicates)."""
        root = logging.getLogger()
        # Add a dummy handler
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)

        configure_logging(env="development")
        # Should have exactly 1 handler (the new one), not initial_count + 1
        assert len(root.handlers) == 1
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_context_filter_attached(self):
        """configure_logging attaches ContextFilter to the handler."""
        configure_logging(env="development")
        root = logging.getLogger()
        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert ContextFilter in filter_types
        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_json_output_end_to_end(self):
        """Full integration: logger.info with extra produces valid JSON."""
        # Set up production logging to a StringIO
        configure_logging(env="production")
        root = logging.getLogger()

        # Replace stdout handler with StringIO
        stream = StringIO()
        root.handlers[0].stream = stream

        test_logger = logging.getLogger("test.e2e")
        test_logger.info(
            "agent_run_started",
            extra={"agent_id": "outreach", "run_id": "xyz-789"},
        )

        output = stream.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["message"] == "agent_run_started"
        assert parsed["agent_id"] == "outreach"
        assert parsed["run_id"] == "xyz-789"
        assert parsed["level"] == "INFO"

        # Clean up
        for h in root.handlers[:]:
            root.removeHandler(h)
