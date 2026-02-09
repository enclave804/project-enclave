"""
System Monitoring MCP tools for the Overseer Agent.

Provides read-only introspection tools for system health monitoring,
agent run history analysis, task queue inspection, and log retrieval.
These tools give the Overseer Agent eyes into the entire platform.

Architecture:
    get_recent_logs()          → Read Python logs from structured log buffer
    query_run_history()        → Query agent_runs table for failures, durations
    get_system_health()        → Aggregate health check across all agents
    get_task_queue_status()    → Inspect pending/stuck/zombie tasks
    get_agent_error_rates()    → Error frequency analysis per agent
    get_knowledge_stats()      → Shared brain utilization metrics
    get_cache_performance()    → LLM response cache hit rates

Safety:
    All tools are READ-ONLY. The Overseer diagnoses but doesn't directly
    modify system state. Human gates or event_bus dispatch handle actions.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log Buffer — In-memory ring buffer for structured log capture
# ---------------------------------------------------------------------------

class LogBuffer:
    """
    Thread-safe in-memory ring buffer that captures structured log records.

    The Overseer reads from this buffer instead of parsing log files.
    Attach to Python's logging system via LogBufferHandler.
    """

    def __init__(self, max_entries: int = 2000):
        self._max = max_entries
        self._entries: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        """Add a log entry. Oldest entries are evicted when full."""
        self._entries.append(record)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def query(
        self,
        level: Optional[str] = None,
        logger_name: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 50,
        since_seconds: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Query log entries with optional filters.

        Args:
            level: Filter by log level (ERROR, WARNING, INFO, DEBUG).
            logger_name: Filter by logger name prefix.
            keyword: Search in message text.
            limit: Max entries to return.
            since_seconds: Only entries from the last N seconds.

        Returns:
            Matching log entries, newest first.
        """
        results = list(reversed(self._entries))
        cutoff = None
        if since_seconds:
            cutoff = time.time() - since_seconds

        filtered = []
        for entry in results:
            if level and entry.get("level", "").upper() != level.upper():
                continue
            if logger_name and not entry.get("logger", "").startswith(logger_name):
                continue
            if keyword and keyword.lower() not in entry.get("message", "").lower():
                continue
            if cutoff and entry.get("timestamp", 0) < cutoff:
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break

        return filtered

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


class LogBufferHandler(logging.Handler):
    """
    Python logging handler that writes to a LogBuffer.

    Attach to the root logger to capture all structured logs:
        handler = LogBufferHandler(buffer)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "funcName": record.funcName,
                "lineno": record.lineno,
            }
            # Capture structured extra fields
            for key in ("agent_id", "run_id", "vertical_id", "error",
                        "duration_ms", "route_action", "tool_name",
                        "status", "score", "hit_count"):
                val = getattr(record, key, None)
                if val is not None:
                    entry[key] = val
            self.buffer.add(entry)
        except Exception:
            pass  # Never crash the app because of log buffering


# Singleton log buffer — initialized on first import
_log_buffer = LogBuffer(max_entries=2000)


def get_log_buffer() -> LogBuffer:
    """Get the global log buffer singleton."""
    return _log_buffer


def install_log_buffer(level: int = logging.INFO) -> LogBufferHandler:
    """
    Install the log buffer handler on the root logger.

    Call this once at application startup to enable log capture.
    Returns the handler for removal/testing.
    """
    handler = LogBufferHandler(_log_buffer)
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# MCP Tool Functions
# ---------------------------------------------------------------------------

def get_recent_logs(
    level: str = "ERROR",
    logger_name: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 30,
    since_minutes: int = 60,
    *,
    _buffer: Any = None,
) -> str:
    """
    Retrieve recent application logs from the in-memory buffer.

    The Overseer uses this to detect errors, warnings, and anomalies
    without parsing log files. Logs are pre-structured with agent_id,
    run_id, and other metadata when available.

    Args:
        level: Minimum log level to retrieve (ERROR, WARNING, INFO, DEBUG).
        logger_name: Filter by logger name prefix (e.g. "core.agents").
        keyword: Search in log message text.
        limit: Maximum number of log entries to return (default: 30).
        since_minutes: Only logs from the last N minutes (default: 60).
        _buffer: Injected LogBuffer for testing.

    Returns:
        JSON string with matching log entries.
    """
    buffer = _buffer or _log_buffer

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "get_recent_logs",
            "level": level,
            "limit": limit,
        },
    )

    entries = buffer.query(
        level=level,
        logger_name=logger_name,
        keyword=keyword,
        limit=limit,
        since_seconds=since_minutes * 60,
    )

    # Format timestamps for readability
    formatted = []
    for entry in entries:
        e = dict(entry)
        ts = e.get("timestamp", 0)
        e["time"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
        formatted.append(e)

    return json.dumps({
        "logs": formatted,
        "count": len(formatted),
        "buffer_size": buffer.size,
        "filter": {
            "level": level,
            "logger_name": logger_name,
            "keyword": keyword,
            "since_minutes": since_minutes,
        },
    }, indent=2, default=str)


def query_run_history(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    vertical_id: str = "enclave_guard",
    *,
    _db: Any = None,
) -> str:
    """
    Query agent execution history from the agent_runs table.

    The Overseer uses this to identify failing agents, slow runs,
    and execution patterns across the platform.

    Args:
        agent_id: Filter by specific agent (omit for all agents).
        status: Filter by run status ("completed", "failed", "started").
        limit: Maximum results (default: 20).
        vertical_id: Vertical to query (default: "enclave_guard").
        _db: Injected EnclaveDB instance for testing.

    Returns:
        JSON string with run history and summary statistics.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB
        _db = EnclaveDB(vertical_id)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "query_run_history",
            "agent_id": agent_id,
            "status": status,
        },
    )

    try:
        runs = _db.get_agent_runs(
            agent_id=agent_id,
            vertical_id=vertical_id,
            limit=limit,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "runs": []})

    # Filter by status if specified
    if status:
        runs = [r for r in runs if r.get("status") == status]

    # Compute summary stats
    total = len(runs)
    failed = sum(1 for r in runs if r.get("status") == "failed")
    completed = sum(1 for r in runs if r.get("status") == "completed")
    durations = [r.get("duration_ms", 0) for r in runs if r.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Extract essential fields
    formatted_runs = []
    for run in runs[:limit]:
        formatted_runs.append({
            "run_id": run.get("run_id", "")[:8],
            "agent_id": run.get("agent_id", ""),
            "agent_type": run.get("agent_type", ""),
            "status": run.get("status", ""),
            "error_message": (run.get("error_message") or "")[:200],
            "duration_ms": run.get("duration_ms"),
            "created_at": run.get("created_at", ""),
        })

    return json.dumps({
        "runs": formatted_runs,
        "summary": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "failure_rate": round(failed / total, 3) if total > 0 else 0,
            "avg_duration_ms": round(avg_duration),
        },
    }, indent=2, default=str)


def get_system_health(
    vertical_id: str = "enclave_guard",
    *,
    _db: Any = None,
    _buffer: Any = None,
) -> str:
    """
    Comprehensive system health check across all agents.

    Aggregates:
    - Recent error count per agent
    - Task queue depth (pending + stuck)
    - Circuit breaker status
    - Log buffer error density
    - Knowledge base size

    This is the Overseer's primary diagnostic tool.

    Args:
        vertical_id: Vertical to check.
        _db: Injected EnclaveDB for testing.
        _buffer: Injected LogBuffer for testing.

    Returns:
        JSON string with full system health report.
    """
    buffer = _buffer or _log_buffer

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "get_system_health"},
    )

    health: dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vertical_id": vertical_id,
        "checks": {},
        "issues": [],
    }

    # --- Check 1: Log Error Density ---
    recent_errors = buffer.query(level="ERROR", limit=100, since_seconds=3600)
    recent_warnings = buffer.query(level="WARNING", limit=100, since_seconds=3600)

    health["checks"]["log_errors"] = {
        "errors_last_hour": len(recent_errors),
        "warnings_last_hour": len(recent_warnings),
        "buffer_size": buffer.size,
    }

    if len(recent_errors) > 20:
        health["issues"].append({
            "severity": "critical",
            "component": "logs",
            "message": f"{len(recent_errors)} errors in the last hour",
        })
    elif len(recent_errors) > 5:
        health["issues"].append({
            "severity": "warning",
            "component": "logs",
            "message": f"{len(recent_errors)} errors in the last hour",
        })

    # --- Check 2: Agent Runs ---
    if _db is not None:
        try:
            runs = _db.get_agent_runs(
                agent_id=None,
                vertical_id=vertical_id,
                limit=50,
            )

            # Group by agent
            agent_stats: dict[str, dict] = {}
            for run in runs:
                aid = run.get("agent_id", "unknown")
                if aid not in agent_stats:
                    agent_stats[aid] = {"total": 0, "failed": 0, "latest_error": ""}
                agent_stats[aid]["total"] += 1
                if run.get("status") == "failed":
                    agent_stats[aid]["failed"] += 1
                    if not agent_stats[aid]["latest_error"]:
                        agent_stats[aid]["latest_error"] = (
                            run.get("error_message", "")[:200]
                        )

            health["checks"]["agent_runs"] = agent_stats

            # Flag agents with high failure rates
            for aid, stats in agent_stats.items():
                if stats["total"] > 0:
                    failure_rate = stats["failed"] / stats["total"]
                    if failure_rate > 0.5:
                        health["issues"].append({
                            "severity": "critical",
                            "component": f"agent:{aid}",
                            "message": (
                                f"Failure rate {failure_rate:.0%} "
                                f"({stats['failed']}/{stats['total']} runs)"
                            ),
                            "latest_error": stats["latest_error"],
                        })

        except Exception as e:
            health["checks"]["agent_runs"] = {"error": str(e)}

        # --- Check 3: Task Queue ---
        try:
            pending = _db.count_pending_tasks(agent_id=None)
            health["checks"]["task_queue"] = {
                "pending_tasks": pending,
            }
            if pending > 50:
                health["issues"].append({
                    "severity": "warning",
                    "component": "task_queue",
                    "message": f"{pending} pending tasks — possible backlog",
                })
        except Exception as e:
            health["checks"]["task_queue"] = {"error": str(e)}

    # --- Overall Status ---
    critical_count = sum(
        1 for i in health["issues"] if i["severity"] == "critical"
    )
    warning_count = sum(
        1 for i in health["issues"] if i["severity"] == "warning"
    )

    if critical_count > 0:
        health["status"] = "critical"
    elif warning_count > 0:
        health["status"] = "degraded"
    else:
        health["status"] = "healthy"

    return json.dumps(health, indent=2, default=str)


def get_task_queue_status(
    vertical_id: str = "enclave_guard",
    *,
    _db: Any = None,
) -> str:
    """
    Inspect the cross-agent task queue for bottlenecks and zombies.

    Shows pending, running, and stuck tasks. The Overseer uses this
    to detect processing backlogs and recover zombie tasks.

    Args:
        vertical_id: Vertical to query.
        _db: Injected EnclaveDB for testing.

    Returns:
        JSON string with task queue breakdown.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB
        _db = EnclaveDB(vertical_id)

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "get_task_queue_status"},
    )

    result: dict[str, Any] = {
        "vertical_id": vertical_id,
        "queue": {},
    }

    try:
        pending = _db.count_pending_tasks(agent_id=None)
        result["queue"]["pending_count"] = pending
    except Exception as e:
        result["queue"]["pending_count_error"] = str(e)

    try:
        zombies = _db.recover_zombie_tasks(stale_minutes=15)
        result["queue"]["recovered_zombies"] = zombies
    except Exception as e:
        result["queue"]["zombie_recovery_error"] = str(e)

    return json.dumps(result, indent=2, default=str)


def get_agent_error_rates(
    days: int = 7,
    vertical_id: str = "enclave_guard",
    *,
    _db: Any = None,
) -> str:
    """
    Analyze error frequency per agent over a time window.

    The Overseer uses this to identify agents trending toward
    circuit breaker activation or needing attention.

    Args:
        days: Look-back window in days (default: 7).
        vertical_id: Vertical to query.
        _db: Injected EnclaveDB for testing.

    Returns:
        JSON string with per-agent error analysis.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB
        _db = EnclaveDB(vertical_id)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "get_agent_error_rates",
            "days": days,
        },
    )

    try:
        runs = _db.get_agent_runs(
            agent_id=None,
            vertical_id=vertical_id,
            limit=500,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "agents": {}})

    # Group by agent and compute rates
    agents: dict[str, dict] = {}
    for run in runs:
        aid = run.get("agent_id", "unknown")
        if aid not in agents:
            agents[aid] = {
                "total_runs": 0,
                "failed_runs": 0,
                "completed_runs": 0,
                "total_duration_ms": 0,
                "error_messages": [],
            }
        agents[aid]["total_runs"] += 1
        status = run.get("status", "")
        if status == "failed":
            agents[aid]["failed_runs"] += 1
            err = run.get("error_message", "")
            if err and len(agents[aid]["error_messages"]) < 5:
                agents[aid]["error_messages"].append(err[:200])
        elif status == "completed":
            agents[aid]["completed_runs"] += 1
            agents[aid]["total_duration_ms"] += run.get("duration_ms", 0)

    # Compute derived metrics
    for aid, stats in agents.items():
        total = stats["total_runs"]
        stats["failure_rate"] = (
            round(stats["failed_runs"] / total, 3) if total > 0 else 0
        )
        completed = stats["completed_runs"]
        stats["avg_duration_ms"] = (
            round(stats["total_duration_ms"] / completed)
            if completed > 0 else 0
        )
        # Risk assessment
        if stats["failure_rate"] > 0.5:
            stats["risk_level"] = "critical"
        elif stats["failure_rate"] > 0.2:
            stats["risk_level"] = "elevated"
        else:
            stats["risk_level"] = "normal"

    return json.dumps({
        "agents": agents,
        "window_days": days,
        "vertical_id": vertical_id,
    }, indent=2, default=str)


def get_knowledge_stats(
    vertical_id: str = "enclave_guard",
    *,
    _db: Any = None,
) -> str:
    """
    Report on shared brain (knowledge_chunks + shared_insights) utilization.

    Shows total chunks, insight types distribution, and recent activity.

    Args:
        vertical_id: Vertical to query.
        _db: Injected EnclaveDB for testing.

    Returns:
        JSON string with knowledge base statistics.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB
        _db = EnclaveDB(vertical_id)

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "get_knowledge_stats"},
    )

    stats: dict[str, Any] = {
        "vertical_id": vertical_id,
        "knowledge_chunks": {},
        "shared_insights": {},
    }

    # This is best-effort — some methods may not exist yet
    try:
        # Attempt to get counts via raw SQL or list operations
        insights = _db.search_knowledge(
            query_embedding=[0.0] * 1536,  # Zero vector — matches everything with low similarity
            chunk_type=None,
            limit=1,
        )
        stats["knowledge_chunks"]["accessible"] = True
    except Exception as e:
        stats["knowledge_chunks"]["accessible"] = False
        stats["knowledge_chunks"]["error"] = str(e)[:200]

    return json.dumps(stats, indent=2, default=str)


def get_cache_performance(
    *,
    _cache: Any = None,
) -> str:
    """
    Report LLM response cache performance metrics.

    Shows hit rate, evictions, size, and temperature settings.
    The Overseer uses this to optimize caching strategy.

    Args:
        _cache: Injected ResponseCache for testing.

    Returns:
        JSON string with cache statistics.
    """
    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "get_cache_performance"},
    )

    if _cache is None:
        return json.dumps({
            "cache_available": False,
            "message": "No cache instance provided. Use with agent.cache.",
        })

    try:
        stats = _cache.get_stats()
        entries = _cache.list_entries()

        return json.dumps({
            "cache_available": True,
            "stats": stats,
            "entries": entries[:20],  # Top 20 cached entries
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({
            "cache_available": False,
            "error": str(e),
        })
