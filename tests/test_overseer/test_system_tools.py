"""
Tests for system_tools.py — Overseer Agent's monitoring MCP tools.

Tests:
- LogBuffer: ring buffer operations, query filtering, size limits
- LogBufferHandler: Python logging integration
- get_recent_logs: log retrieval with filters
- query_run_history: agent run history analysis
- get_system_health: comprehensive health check
- get_task_queue_status: queue inspection
- get_agent_error_rates: per-agent failure analysis
- get_knowledge_stats: shared brain utilization
- get_cache_performance: LLM cache metrics
"""

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from core.mcp.tools.system_tools import (
    LogBuffer,
    LogBufferHandler,
    get_log_buffer,
    install_log_buffer,
    get_recent_logs,
    query_run_history,
    get_system_health,
    get_task_queue_status,
    get_agent_error_rates,
    get_knowledge_stats,
    get_cache_performance,
)


# ===========================================================================
# LogBuffer Tests
# ===========================================================================

class TestLogBuffer:
    """Tests for the in-memory log ring buffer."""

    def test_add_and_query(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"level": "ERROR", "message": "test error", "timestamp": time.time()})
        buf.add({"level": "INFO", "message": "test info", "timestamp": time.time()})

        results = buf.query(limit=10)
        assert len(results) == 2

    def test_ring_buffer_eviction(self):
        buf = LogBuffer(max_entries=5)
        for i in range(10):
            buf.add({"message": f"entry_{i}", "timestamp": time.time()})

        assert buf.size == 5
        results = buf.query(limit=10)
        # Most recent entries should be preserved
        assert results[0]["message"] == "entry_9"
        assert results[4]["message"] == "entry_5"

    def test_query_by_level(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"level": "ERROR", "message": "fail", "timestamp": time.time()})
        buf.add({"level": "INFO", "message": "ok", "timestamp": time.time()})
        buf.add({"level": "ERROR", "message": "fail2", "timestamp": time.time()})

        errors = buf.query(level="ERROR")
        assert len(errors) == 2
        assert all(e["level"] == "ERROR" for e in errors)

    def test_query_by_logger_name(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"logger": "core.agents.base", "message": "a", "timestamp": time.time()})
        buf.add({"logger": "core.llm.router", "message": "b", "timestamp": time.time()})
        buf.add({"logger": "core.agents.overseer", "message": "c", "timestamp": time.time()})

        results = buf.query(logger_name="core.agents")
        assert len(results) == 2

    def test_query_by_keyword(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"message": "circuit breaker tripped", "timestamp": time.time()})
        buf.add({"message": "normal operation", "timestamp": time.time()})

        results = buf.query(keyword="circuit")
        assert len(results) == 1
        assert "circuit" in results[0]["message"]

    def test_query_since_seconds(self):
        buf = LogBuffer(max_entries=100)
        old_time = time.time() - 7200  # 2 hours ago
        buf.add({"message": "old", "timestamp": old_time})
        buf.add({"message": "recent", "timestamp": time.time()})

        results = buf.query(since_seconds=3600)  # Last hour
        assert len(results) == 1
        assert results[0]["message"] == "recent"

    def test_query_limit(self):
        buf = LogBuffer(max_entries=100)
        for i in range(20):
            buf.add({"message": f"entry_{i}", "timestamp": time.time()})

        results = buf.query(limit=5)
        assert len(results) == 5

    def test_query_newest_first(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"message": "first", "timestamp": time.time()})
        buf.add({"message": "second", "timestamp": time.time()})
        buf.add({"message": "third", "timestamp": time.time()})

        results = buf.query()
        assert results[0]["message"] == "third"
        assert results[2]["message"] == "first"

    def test_clear(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"message": "a"})
        buf.add({"message": "b"})
        cleared = buf.clear()
        assert cleared == 2
        assert buf.size == 0

    def test_empty_query(self):
        buf = LogBuffer(max_entries=100)
        results = buf.query()
        assert results == []


# ===========================================================================
# LogBufferHandler Tests
# ===========================================================================

class TestLogBufferHandler:
    """Tests for the Python logging integration."""

    def test_handler_captures_log_records(self):
        buf = LogBuffer(max_entries=100)
        handler = LogBufferHandler(buf)

        logger = logging.getLogger("test.handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            logger.error("Test error message")
            logger.info("Test info message")

            assert buf.size == 2
            entries = buf.query()
            assert entries[0]["level"] == "INFO"
            assert entries[1]["level"] == "ERROR"
        finally:
            logger.removeHandler(handler)

    def test_handler_captures_extra_fields(self):
        buf = LogBuffer(max_entries=100)
        handler = LogBufferHandler(buf)

        logger = logging.getLogger("test.extra")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            logger.info(
                "agent_run_started",
                extra={"agent_id": "outreach", "run_id": "abc123"},
            )

            entries = buf.query()
            assert len(entries) == 1
            assert entries[0]["agent_id"] == "outreach"
            assert entries[0]["run_id"] == "abc123"
        finally:
            logger.removeHandler(handler)

    def test_handler_never_crashes(self):
        """Handler should never raise, even with malformed records."""
        buf = LogBuffer(max_entries=100)
        handler = LogBufferHandler(buf)

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="test", args=None, exc_info=None,
        )
        # Should not raise
        handler.emit(record)
        assert buf.size == 1


class TestGetLogBuffer:
    """Tests for the global log buffer singleton."""

    def test_singleton(self):
        buf1 = get_log_buffer()
        buf2 = get_log_buffer()
        assert buf1 is buf2


# ===========================================================================
# MCP Tool Function Tests
# ===========================================================================

class TestGetRecentLogs:
    """Tests for the get_recent_logs() MCP tool."""

    def test_returns_json(self):
        buf = LogBuffer(max_entries=100)
        buf.add({
            "level": "ERROR",
            "message": "test error",
            "timestamp": time.time(),
            "logger": "core.agents",
        })

        result = get_recent_logs(level="ERROR", limit=10, _buffer=buf)
        parsed = json.loads(result)

        assert "logs" in parsed
        assert "count" in parsed
        assert parsed["count"] == 1
        assert parsed["logs"][0]["message"] == "test error"

    def test_filters_by_level(self):
        buf = LogBuffer(max_entries=100)
        buf.add({"level": "ERROR", "message": "err", "timestamp": time.time()})
        buf.add({"level": "INFO", "message": "info", "timestamp": time.time()})

        result = json.loads(get_recent_logs(level="ERROR", _buffer=buf))
        assert result["count"] == 1

    def test_includes_filter_metadata(self):
        buf = LogBuffer(max_entries=100)
        result = json.loads(get_recent_logs(
            level="WARNING",
            logger_name="core.agents",
            keyword="test",
            since_minutes=30,
            _buffer=buf,
        ))
        assert result["filter"]["level"] == "WARNING"
        assert result["filter"]["logger_name"] == "core.agents"
        assert result["filter"]["since_minutes"] == 30

    def test_empty_buffer(self):
        buf = LogBuffer(max_entries=100)
        result = json.loads(get_recent_logs(_buffer=buf))
        assert result["count"] == 0
        assert result["logs"] == []


class TestQueryRunHistory:
    """Tests for the query_run_history() MCP tool."""

    def test_returns_run_history(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {
                "run_id": "abc12345-6789",
                "agent_id": "outreach",
                "agent_type": "outreach",
                "status": "completed",
                "duration_ms": 1500,
                "created_at": "2024-01-15T10:00:00Z",
            },
            {
                "run_id": "def12345-6789",
                "agent_id": "outreach",
                "agent_type": "outreach",
                "status": "failed",
                "error_message": "API timeout",
                "duration_ms": 5000,
                "created_at": "2024-01-15T09:00:00Z",
            },
        ]

        result = json.loads(query_run_history(
            agent_id="outreach",
            vertical_id="enclave_guard",
            _db=mock_db,
        ))

        assert result["summary"]["total"] == 2
        assert result["summary"]["completed"] == 1
        assert result["summary"]["failed"] == 1
        assert result["summary"]["failure_rate"] == 0.5
        assert result["summary"]["avg_duration_ms"] > 0

    def test_filters_by_status(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {"run_id": "a", "status": "completed", "agent_id": "x"},
            {"run_id": "b", "status": "failed", "agent_id": "x"},
        ]

        result = json.loads(query_run_history(
            status="failed", _db=mock_db,
        ))
        assert result["summary"]["total"] == 1

    def test_handles_db_error(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.side_effect = Exception("DB down")

        result = json.loads(query_run_history(_db=mock_db))
        assert "error" in result


class TestGetSystemHealth:
    """Tests for the get_system_health() MCP tool."""

    def test_healthy_system(self):
        buf = LogBuffer(max_entries=100)  # Empty buffer = no errors
        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {"agent_id": "outreach", "status": "completed"},
            {"agent_id": "outreach", "status": "completed"},
        ]
        mock_db.count_pending_tasks.return_value = 3

        result = json.loads(get_system_health(
            _db=mock_db, _buffer=buf,
        ))

        assert result["status"] == "healthy"
        assert result["checks"]["log_errors"]["errors_last_hour"] == 0
        assert result["checks"]["task_queue"]["pending_tasks"] == 3

    def test_degraded_system(self):
        buf = LogBuffer(max_entries=100)
        # Add 10 errors
        for i in range(10):
            buf.add({"level": "ERROR", "message": f"err_{i}", "timestamp": time.time()})

        result = json.loads(get_system_health(_buffer=buf))

        assert result["status"] in ("degraded", "critical")
        assert len(result["issues"]) > 0

    def test_critical_system(self):
        buf = LogBuffer(max_entries=100)
        # Add 25 errors in the last hour
        for i in range(25):
            buf.add({"level": "ERROR", "message": f"err_{i}", "timestamp": time.time()})

        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {"agent_id": "outreach", "status": "failed", "error_message": "crash"},
        ] * 5
        mock_db.count_pending_tasks.return_value = 100

        result = json.loads(get_system_health(
            _db=mock_db, _buffer=buf,
        ))

        assert result["status"] == "critical"
        critical_issues = [
            i for i in result["issues"] if i["severity"] == "critical"
        ]
        assert len(critical_issues) > 0

    def test_handles_db_errors_gracefully(self):
        buf = LogBuffer(max_entries=100)
        mock_db = MagicMock()
        mock_db.get_agent_runs.side_effect = Exception("DB down")
        mock_db.count_pending_tasks.side_effect = Exception("DB down")

        # Should not raise
        result = json.loads(get_system_health(
            _db=mock_db, _buffer=buf,
        ))
        assert "checks" in result


class TestGetTaskQueueStatus:
    """Tests for the get_task_queue_status() MCP tool."""

    def test_returns_queue_status(self):
        mock_db = MagicMock()
        mock_db.count_pending_tasks.return_value = 12
        mock_db.recover_zombie_tasks.return_value = 2

        result = json.loads(get_task_queue_status(_db=mock_db))

        assert result["queue"]["pending_count"] == 12
        assert result["queue"]["recovered_zombies"] == 2

    def test_handles_errors(self):
        mock_db = MagicMock()
        mock_db.count_pending_tasks.side_effect = Exception("fail")
        mock_db.recover_zombie_tasks.side_effect = Exception("fail")

        result = json.loads(get_task_queue_status(_db=mock_db))
        assert "pending_count_error" in result["queue"]


class TestGetAgentErrorRates:
    """Tests for the get_agent_error_rates() MCP tool."""

    def test_computes_error_rates(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {"agent_id": "outreach", "status": "completed", "duration_ms": 1000},
            {"agent_id": "outreach", "status": "completed", "duration_ms": 2000},
            {"agent_id": "outreach", "status": "failed", "error_message": "timeout"},
            {"agent_id": "seo", "status": "completed", "duration_ms": 3000},
        ]

        result = json.loads(get_agent_error_rates(_db=mock_db))

        outreach = result["agents"]["outreach"]
        assert outreach["total_runs"] == 3
        assert outreach["failed_runs"] == 1
        assert outreach["failure_rate"] == pytest.approx(0.333, abs=0.01)
        assert outreach["risk_level"] == "elevated"

        seo = result["agents"]["seo"]
        assert seo["failure_rate"] == 0.0
        assert seo["risk_level"] == "normal"

    def test_critical_risk_level(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.return_value = [
            {"agent_id": "broken", "status": "failed", "error_message": "crash"},
            {"agent_id": "broken", "status": "failed", "error_message": "crash2"},
        ]

        result = json.loads(get_agent_error_rates(_db=mock_db))
        assert result["agents"]["broken"]["risk_level"] == "critical"

    def test_handles_db_error(self):
        mock_db = MagicMock()
        mock_db.get_agent_runs.side_effect = Exception("DB down")

        result = json.loads(get_agent_error_rates(_db=mock_db))
        assert "error" in result


class TestGetKnowledgeStats:
    """Tests for the get_knowledge_stats() MCP tool."""

    def test_returns_stats(self):
        mock_db = MagicMock()
        mock_db.search_knowledge.return_value = []

        result = json.loads(get_knowledge_stats(_db=mock_db))
        assert result["knowledge_chunks"]["accessible"] is True

    def test_handles_db_error(self):
        mock_db = MagicMock()
        mock_db.search_knowledge.side_effect = Exception("fail")

        result = json.loads(get_knowledge_stats(_db=mock_db))
        assert result["knowledge_chunks"]["accessible"] is False


class TestGetCachePerformance:
    """Tests for the get_cache_performance() MCP tool."""

    def test_returns_cache_stats(self):
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {
            "size": 10,
            "hits": 50,
            "misses": 5,
            "hit_rate": 0.909,
        }
        mock_cache.list_entries.return_value = [
            {"key": "abc...", "provider": "anthropic"},
        ]

        result = json.loads(get_cache_performance(_cache=mock_cache))
        assert result["cache_available"] is True
        assert result["stats"]["hit_rate"] == 0.909

    def test_no_cache_instance(self):
        result = json.loads(get_cache_performance())
        assert result["cache_available"] is False

    def test_handles_cache_error(self):
        mock_cache = MagicMock()
        mock_cache.get_stats.side_effect = Exception("broken")

        result = json.loads(get_cache_performance(_cache=mock_cache))
        assert result["cache_available"] is False
