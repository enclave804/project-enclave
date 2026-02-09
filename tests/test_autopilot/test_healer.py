"""
Tests for SelfHealer — Phase 15.

Tests crash analysis, config fix suggestions, safe key validation,
health score computation, and healing history retrieval.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.optimization.healer import (
    DESTRUCTIVE_KEYS,
    ERROR_PATTERNS,
    SAFE_CONFIG_KEYS,
    SelfHealer,
)


# ── Mock DB ──────────────────────────────────────────────────


class MockTable:
    """Chainable mock for Supabase table operations."""

    def __init__(self, data=None):
        self._data = data or []
        self._filters = {}

    def select(self, cols="*"):
        return self

    def insert(self, data):
        self._data.append(data)
        return self

    def update(self, data):
        self._update_data = data
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def gte(self, col, val):
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        return self

    def execute(self):
        result = MagicMock()
        filtered = self._data
        for col, val in self._filters.items():
            filtered = [r for r in filtered if r.get(col) == val]
        result.data = filtered
        self._filters = {}
        return result


class MockDB:
    def __init__(self, tables=None):
        self._tables = tables or {}
        self.vertical_id = "test_vertical"
        self.client = self

    def table(self, name):
        return MockTable(self._tables.get(name, []))


@pytest.fixture
def mock_db():
    return MockDB()


@pytest.fixture
def healer(mock_db):
    return SelfHealer(db=mock_db)


# ── SelfHealer Construction ──────────────────────────────────


class TestSelfHealerConstruction:
    def test_init_with_db(self, healer):
        assert healer.db is not None
        assert healer.llm is None

    def test_init_with_llm(self, mock_db):
        llm = MagicMock()
        h = SelfHealer(db=mock_db, llm_client=llm)
        assert h.llm is llm


# ── Crash Analysis ────────────────────────────────────────────


class TestAnalyzeCrash:
    def test_analyze_rate_limit(self, healer):
        result = healer.analyze_crash(
            "outreach_v1",
            "HTTPError: 429 Too Many Requests — rate limit exceeded",
        )
        assert result["severity"] in ("medium", "high", "critical")
        assert "category" in result
        assert result["category"] == "rate_limit"

    def test_analyze_auth_error(self, healer):
        result = healer.analyze_crash(
            "outreach_v1",
            "AuthenticationError: Invalid API key provided",
        )
        assert result["category"] == "auth"

    def test_analyze_network_error(self, healer):
        result = healer.analyze_crash(
            "ads_v1",
            "ConnectionError: timed out connecting to api.example.com",
        )
        assert result["category"] == "network"

    def test_analyze_circuit_breaker(self, healer):
        result = healer.analyze_crash(
            "seo_v1",
            "circuit_breaker tripped: Agent disabled after 5 consecutive_errors",
        )
        assert result["category"] == "circuit_breaker"

    def test_analyze_database_error(self, healer):
        result = healer.analyze_crash(
            "finance_v1",
            "DatabaseError: relation 'invoices' does not exist",
        )
        assert result["category"] == "database"

    def test_analyze_unknown_error(self, healer):
        result = healer.analyze_crash(
            "test_v1",
            "SomethingWeirdHappened: no pattern match",
        )
        assert result["category"] == "unknown"
        assert result["severity"] == "medium"

    def test_analyze_returns_agent_id(self, healer):
        result = healer.analyze_crash("my_agent", "rate limit exceeded")
        assert result["agent_id"] == "my_agent"

    def test_analyze_empty_error(self, healer):
        result = healer.analyze_crash("agent_x", "")
        assert result["category"] == "unknown"

    def test_analyze_includes_recommended_action(self, healer):
        result = healer.analyze_crash("agent", "rate limit exceeded")
        assert "recommended_action" in result


# ── Fix Suggestions ───────────────────────────────────────────


class TestSuggestFix:
    def test_suggest_fix_rate_limit(self, healer):
        diagnosis = {
            "agent_id": "outreach_v1",
            "category": "rate_limit",
            "severity": "warning",
        }
        fixes = healer.suggest_fix("outreach_v1", diagnosis)
        assert isinstance(fixes, list)
        assert len(fixes) > 0
        assert all("parameter" in f for f in fixes)

    def test_suggest_fix_circuit_breaker(self, healer):
        diagnosis = {
            "agent_id": "seo_v1",
            "category": "circuit_breaker",
            "severity": "critical",
        }
        fixes = healer.suggest_fix("seo_v1", diagnosis)
        assert len(fixes) > 0

    def test_suggest_fix_auth(self, healer):
        diagnosis = {
            "agent_id": "outreach_v1",
            "category": "auth",
            "severity": "critical",
        }
        fixes = healer.suggest_fix("outreach_v1", diagnosis)
        assert len(fixes) > 0

    def test_suggest_fix_unknown(self, healer):
        diagnosis = {
            "agent_id": "test",
            "category": "unknown",
            "severity": "info",
        }
        fixes = healer.suggest_fix("test", diagnosis)
        assert isinstance(fixes, list)

    def test_fix_has_reasoning(self, healer):
        diagnosis = {"agent_id": "a", "category": "rate_limit", "severity": "warning"}
        fixes = healer.suggest_fix("a", diagnosis)
        if fixes:
            assert "reasoning" in fixes[0]


# ── Config Fix Application ────────────────────────────────────


class TestApplyConfigFix:
    def test_apply_safe_key(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "max_consecutive_errors",
            "old_value": 3,
            "new_value": 5,
            "reasoning": "Increase tolerance",
        }
        result = healer.apply_config_fix("outreach_v1", fix)
        # Without a real DB get_agent_record, this may fail; key validation passes
        assert result["status"] in ("applied", "logged", "success", "failed")

    def test_reject_unsafe_key(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "source_code.main",
            "old_value": "",
            "new_value": "import os; os.system('rm -rf /')",
            "reasoning": "Hack attempt",
        }
        result = healer.apply_config_fix("outreach_v1", fix)
        assert result["status"] == "rejected"

    def test_apply_params_wildcard(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "params.daily_lead_limit",
            "old_value": 50,
            "new_value": 30,
            "reasoning": "Reduce load",
        }
        result = healer.apply_config_fix("outreach_v1", fix)
        assert result["status"] != "rejected"

    def test_destructive_key_requires_approval(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "enabled",
            "old_value": True,
            "new_value": False,
            "reasoning": "Disable crashing agent",
        }
        result = healer.apply_config_fix("outreach_v1", fix)
        # Destructive key → pending_approval status
        assert result["status"] == "pending_approval"

    def test_apply_with_session_id(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "schedule.cron",
            "old_value": "0 */6 * * *",
            "new_value": "0 */12 * * *",
            "reasoning": "Reduce frequency",
        }
        result = healer.apply_config_fix(
            "outreach_v1", fix, session_id="session-123",
        )
        assert result["status"] != "rejected"

    def test_apply_schedule_trigger(self, healer):
        fix = {
            "action": "update_config",
            "parameter": "schedule.trigger",
            "old_value": "cron",
            "new_value": "manual",
            "reasoning": "Switch to manual trigger",
        }
        result = healer.apply_config_fix("agent_x", fix)
        assert result["status"] != "rejected"


# ── Safe Key Validation ───────────────────────────────────────


class TestSafeKeyValidation:
    def test_safe_config_keys_frozenset(self):
        assert isinstance(SAFE_CONFIG_KEYS, frozenset)

    def test_enabled_is_safe(self, healer):
        assert healer._is_safe_key("enabled") is True

    def test_schedule_cron_is_safe(self, healer):
        assert healer._is_safe_key("schedule.cron") is True

    def test_params_wildcard_is_safe(self, healer):
        assert healer._is_safe_key("params.daily_lead_limit") is True
        assert healer._is_safe_key("params.max_daily_budget") is True

    def test_code_is_not_safe(self, healer):
        assert healer._is_safe_key("source_code") is False

    def test_random_key_not_safe(self, healer):
        assert healer._is_safe_key("admin_password") is False

    def test_destructive_keys_subset(self):
        assert DESTRUCTIVE_KEYS.issubset(SAFE_CONFIG_KEYS)

    def test_enabled_is_destructive(self):
        assert "enabled" in DESTRUCTIVE_KEYS


# ── Health Score ──────────────────────────────────────────────


class TestHealthScore:
    def test_health_score_no_data(self, healer):
        score = healer.get_agent_health_score("nonexistent_agent")
        # Should return a default score, not crash
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_health_score_with_data(self):
        runs = [
            {"agent_id": "test_agent", "status": "completed", "created_at": datetime.now(timezone.utc).isoformat()},
            {"agent_id": "test_agent", "status": "completed", "created_at": datetime.now(timezone.utc).isoformat()},
            {"agent_id": "test_agent", "status": "failed", "created_at": datetime.now(timezone.utc).isoformat()},
        ]
        db = MockDB(tables={"agent_runs": runs})
        h = SelfHealer(db=db)
        score = h.get_agent_health_score("test_agent")
        assert 0.0 <= score <= 1.0

    def test_health_score_all_success(self):
        runs = [
            {"agent_id": "perfect", "status": "completed", "created_at": datetime.now(timezone.utc).isoformat()}
            for _ in range(10)
        ]
        db = MockDB(tables={"agent_runs": runs})
        h = SelfHealer(db=db)
        score = h.get_agent_health_score("perfect")
        assert score >= 0.5  # Should be high

    def test_health_score_all_failures(self):
        runs = [
            {"agent_id": "broken", "status": "failed", "created_at": datetime.now(timezone.utc).isoformat()}
            for _ in range(10)
        ]
        db = MockDB(tables={"agent_runs": runs})
        h = SelfHealer(db=db)
        score = h.get_agent_health_score("broken")
        assert score <= 0.5  # Should be low


# ── Healing History ───────────────────────────────────────────


class TestHealingHistory:
    def test_empty_history(self, healer):
        history = healer.get_healing_history("test_vertical")
        assert history == []

    def test_history_returns_list(self):
        actions = [
            {
                "vertical_id": "test",
                "action_type": "config_fix",
                "target": "agent_x",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        db = MockDB(tables={"optimization_actions": actions})
        h = SelfHealer(db=db)
        history = h.get_healing_history("test")
        assert isinstance(history, list)


# ── Error Patterns ────────────────────────────────────────────


class TestErrorPatterns:
    def test_error_patterns_dict(self):
        assert isinstance(ERROR_PATTERNS, dict)
        assert len(ERROR_PATTERNS) > 0

    def test_all_patterns_have_severity(self):
        valid_severities = ("critical", "high", "medium", "low")
        for key, value in ERROR_PATTERNS.items():
            # ERROR_PATTERNS maps pattern -> severity string directly
            assert isinstance(value, str)
            assert value in valid_severities
