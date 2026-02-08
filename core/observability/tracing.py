"""
Distributed tracing for the Sovereign Venture Engine.

Uses LangFuse for end-to-end visibility into agent execution paths.
When Agent A passes data to Agent B and Agent B fails, you can see
the exact chain of events on a visual timeline.

Design principles:
- Silent fail: if LANGFUSE_PUBLIC_KEY is not set, everything is a no-op
- Zero overhead: no-op wrappers add negligible performance cost
- Tagged traces: every span carries agent_id, vertical_id, run_id
- Composable: works with existing logging, doesn't replace it

Usage:
    from core.observability.tracing import get_tracer, create_trace

    tracer = get_tracer()  # None if not configured
    trace = create_trace(
        tracer=tracer,
        name="outreach-run",
        agent_id="outreach",
        vertical_id="enclave_guard",
        run_id="abc-123",
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── LangFuse Client (lazy, singleton) ──────────────────────────────

_langfuse_client: Any = None
_langfuse_initialized: bool = False


def _is_langfuse_configured() -> bool:
    """Check if LangFuse environment variables are set."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_tracer() -> Any:
    """
    Get the LangFuse client singleton, or None if not configured.

    Returns None (not raises) when:
    - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing
    - langfuse package is not installed
    - Connection to LangFuse fails

    This ensures local dev never breaks due to missing tracing.
    """
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True

    if not _is_langfuse_configured():
        logger.debug(
            "LangFuse not configured (LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY not set) — tracing disabled"
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            ),
        )
        logger.info("LangFuse tracing initialized successfully")
        return _langfuse_client

    except ImportError:
        logger.debug(
            "langfuse package not installed — tracing disabled. "
            "Install with: pip install langfuse"
        )
        return None
    except Exception as err:
        logger.warning(f"Failed to initialize LangFuse: {err}")
        return None


def create_trace(
    tracer: Any,
    *,
    name: str,
    agent_id: str,
    vertical_id: str,
    run_id: str,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> Any:
    """
    Create a new trace for an agent run.

    Returns a LangFuse trace object, or a NoOpTrace if tracing
    is not configured.

    Args:
        tracer: LangFuse client (from get_tracer()). None → NoOpTrace.
        name: Trace name (e.g., "outreach-run").
        agent_id: The agent's unique ID.
        vertical_id: The vertical this agent belongs to.
        run_id: Unique run identifier.
        metadata: Additional metadata to attach.
        tags: Additional tags for filtering.

    Returns:
        A trace object with .span() method, or NoOpTrace.
    """
    if tracer is None:
        return NoOpTrace()

    try:
        trace_tags = [
            f"agent:{agent_id}",
            f"vertical:{vertical_id}",
        ]
        if tags:
            trace_tags.extend(tags)

        trace_metadata = {
            "agent_id": agent_id,
            "vertical_id": vertical_id,
            "run_id": run_id,
            **(metadata or {}),
        }

        return tracer.trace(
            id=run_id,
            name=name,
            metadata=trace_metadata,
            tags=trace_tags,
        )
    except Exception as err:
        logger.debug(f"Failed to create LangFuse trace: {err}")
        return NoOpTrace()


def create_span(
    trace: Any,
    *,
    name: str,
    input_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Any:
    """
    Create a span (sub-operation) within a trace.

    Returns a LangFuse span, or a NoOpSpan if trace is a NoOpTrace.
    """
    if isinstance(trace, NoOpTrace):
        return NoOpSpan()

    try:
        return trace.span(
            name=name,
            input=input_data,
            metadata=metadata,
        )
    except Exception as err:
        logger.debug(f"Failed to create LangFuse span: {err}")
        return NoOpSpan()


def end_span(
    span: Any,
    *,
    output_data: Optional[dict] = None,
    status: str = "ok",
    level: str = "DEFAULT",
) -> None:
    """
    End a span, recording its output and status.

    Safe to call on NoOpSpan — silently does nothing.
    """
    if isinstance(span, NoOpSpan):
        return

    try:
        span.end(
            output=output_data,
            level=level if status == "ok" else "ERROR",
            status_message=status if status != "ok" else None,
        )
    except Exception as err:
        logger.debug(f"Failed to end LangFuse span: {err}")


def flush_tracer(tracer: Any) -> None:
    """Flush any pending traces. Safe to call with None."""
    if tracer is None:
        return
    try:
        tracer.flush()
    except Exception as err:
        logger.debug(f"Failed to flush LangFuse: {err}")


# ─── No-Op Fallbacks ────────────────────────────────────────────────


class NoOpSpan:
    """No-op span that silently swallows all calls."""

    def end(self, **kwargs: Any) -> None:
        pass

    def span(self, **kwargs: Any) -> "NoOpSpan":
        return NoOpSpan()

    def generation(self, **kwargs: Any) -> "NoOpSpan":
        return NoOpSpan()

    def event(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class NoOpTrace:
    """No-op trace that silently swallows all calls."""

    def span(self, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def generation(self, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def event(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass


@contextmanager
def traced_operation(
    trace: Any,
    name: str,
    input_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Context manager for tracing a block of code.

    Usage:
        with traced_operation(trace, "enrich_company", input_data={...}) as span:
            result = do_enrichment()
            end_span(span, output_data=result)
    """
    span = create_span(trace, name=name, input_data=input_data, metadata=metadata)
    try:
        yield span
    except Exception as err:
        end_span(span, status=f"error: {str(err)[:200]}", level="ERROR")
        raise
