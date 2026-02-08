"""
Structured logging configuration for the Sovereign Venture Engine.

Design decision: Use Python's built-in logging with a JSONFormatter
instead of structlog. This gives us the same structured JSON output
for production (LangFuse, Datadog, etc.) while keeping all existing
logging.getLogger() calls working with zero refactoring.

Environments:
- production: JSON to stdout (machine-readable)
- development/staging/test: Colored text to stderr (human-readable)

Usage:
    from core.observability.logging_config import configure_logging

    configure_logging()  # auto-detects from ENCLAVE_ENV

    # Existing code unchanged:
    logger = logging.getLogger(__name__)
    logger.info("something happened")

    # Structured fields via extra dict:
    logger.info("agent_run_started", extra={
        "agent_id": "outreach",
        "run_id": "abc-123",
        "vertical_id": "enclave_guard",
    })
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

# ─── Thread-local Trace Context ───────────────────────────────────────

_trace_context = threading.local()


def set_trace_id(trace_id: str) -> None:
    """
    Set the current LangFuse trace_id on the thread-local context.

    Called by the tracing layer when a new trace is started, so that
    all log records emitted during that trace carry the trace_id.
    """
    _trace_context.trace_id = trace_id


def get_trace_id() -> Optional[str]:
    """Get the current trace_id, or None if not in a traced context."""
    return getattr(_trace_context, "trace_id", None)


def clear_trace_id() -> None:
    """Clear the trace_id from the thread-local context."""
    _trace_context.trace_id = None


# ─── Context Filter ───────────────────────────────────────────────────


class ContextFilter(logging.Filter):
    """
    Injects trace_id into every log record from thread-local storage.

    This allows the JSONFormatter to include trace_id in every log line
    when a LangFuse trace is active, enabling log-to-trace correlation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = get_trace_id()
        if trace_id:
            record.trace_id = trace_id  # type: ignore[attr-defined]
        return True


# ─── JSON Formatter (Production) ──────────────────────────────────────


# Fields we want to extract from the record's extra dict
_STANDARD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})


class JSONFormatter(logging.Formatter):
    """
    Outputs log records as single-line JSON objects.

    Includes standard fields (timestamp, level, logger, message) plus
    any extra fields passed via `logger.info("msg", extra={...})`.

    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "core.agents.base",
         "message": "agent_run_started", "agent_id": "outreach", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        # Build the base log entry
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace_id if present (from ContextFilter)
        if hasattr(record, "trace_id"):
            entry["trace_id"] = record.trace_id

        # Extract extra fields (anything not in the standard set)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                if key == "trace_id":
                    continue  # already handled above
                try:
                    # Ensure the value is JSON-serializable
                    json.dumps(value)
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        # Add exception info if present
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ─── Dev Formatter (Local Development) ────────────────────────────────


class DevFormatter(logging.Formatter):
    """
    Colorful, human-readable logs for local development.

    Format: [HH:MM:SS] LEVEL logger: message [key=value key=value]
    """

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    # Known extra fields to display inline
    _EXTRA_KEYS = (
        "agent_id", "vertical_id", "run_id", "trace_id",
        "tool_name", "duration_ms", "status", "route_action",
    )

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Collect extra fields for inline display
        extras = []
        for key in self._EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")

        extra_str = f" [{' '.join(extras)}]" if extras else ""

        formatted = (
            f"{self.RESET}[{timestamp}] "
            f"{color}{record.levelname:<8}{self.RESET} "
            f"{record.name}: {record.getMessage()}{extra_str}"
        )

        # Add exception info if present
        if record.exc_info and record.exc_info[1]:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


# ─── Configuration ────────────────────────────────────────────────────


def configure_logging(
    env: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure the root logger based on environment.

    Args:
        env: Override environment. If None, reads from ENCLAVE_ENV
             (defaults to "development").
        level: Log level (default: INFO).

    Behavior:
        - production → JSONFormatter to stdout
        - everything else → DevFormatter to stderr
    """
    env = env or os.environ.get("ENCLAVE_ENV", "development").lower().strip()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create the handler
    if env == "production":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(DevFormatter())

    # Add the context filter for trace_id injection
    handler.addFilter(ContextFilter())
    root_logger.addHandler(handler)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("supabase").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)
