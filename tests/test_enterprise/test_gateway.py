"""
Tests for Enterprise API Gateway — Phase 14.

Tests rate limiter, API key middleware, scope checking,
FastAPI endpoints, and org-scoped data isolation.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.enterprise.gateway import (
    RateLimiter,
    create_auth_dependency,
    get_health_response,
    require_scope,
)
from core.enterprise.models import (
    AuthContext,
    OrgRole,
    PlanTier,
    PLAN_TIER_LIMITS,
)


# ── Rate Limiter Tests ───────────────────────────────────────


class TestRateLimiter:
    """Tests for the in-memory sliding window rate limiter."""

    def test_basic_allow(self):
        limiter = RateLimiter(default_limit=10)
        assert limiter.is_allowed("org-1") is True

    def test_increments_count(self):
        limiter = RateLimiter(default_limit=10)
        for _ in range(5):
            limiter.is_allowed("org-1")
        remaining = limiter.get_remaining("org-1", limit=10)
        assert remaining == 5

    def test_blocks_at_limit(self):
        limiter = RateLimiter(default_limit=3)
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is False

    def test_per_org_isolation(self):
        limiter = RateLimiter(default_limit=2)
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is False
        # org-2 should still be allowed
        assert limiter.is_allowed("org-2") is True

    def test_custom_limit_override(self):
        limiter = RateLimiter(default_limit=100)
        # Override with a lower limit
        assert limiter.is_allowed("org-1", limit=1) is True
        assert limiter.is_allowed("org-1", limit=1) is False

    def test_zero_limit_blocks_all(self):
        limiter = RateLimiter(default_limit=0)
        assert limiter.is_allowed("org-1") is False

    def test_window_slides(self):
        limiter = RateLimiter(default_limit=2, window_seconds=1)
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is True
        assert limiter.is_allowed("org-1") is False
        # Wait for window to pass
        time.sleep(1.1)
        assert limiter.is_allowed("org-1") is True

    def test_get_remaining_full(self):
        limiter = RateLimiter(default_limit=5)
        remaining = limiter.get_remaining("org-1", limit=5)
        assert remaining == 5

    def test_get_remaining_after_requests(self):
        limiter = RateLimiter(default_limit=5)
        limiter.is_allowed("org-1")
        limiter.is_allowed("org-1")
        remaining = limiter.get_remaining("org-1", limit=5)
        assert remaining == 3

    def test_get_remaining_zero(self):
        limiter = RateLimiter(default_limit=1)
        limiter.is_allowed("org-1")
        remaining = limiter.get_remaining("org-1", limit=1)
        assert remaining == 0

    def test_get_reset_time_no_requests(self):
        limiter = RateLimiter()
        reset = limiter.get_reset_time("org-1")
        assert reset == 0.0

    def test_get_reset_time_after_request(self):
        limiter = RateLimiter(window_seconds=60)
        limiter.is_allowed("org-1")
        reset = limiter.get_reset_time("org-1")
        assert 0 < reset <= 60

    def test_cleanup_removes_stale(self):
        limiter = RateLimiter()
        limiter.is_allowed("org-1")
        limiter.is_allowed("org-2")
        # Manually expire entries
        limiter._requests["org-1"] = [time.time() - 600]
        limiter._requests["org-2"] = [time.time() - 600]
        cleaned = limiter.cleanup(max_age_seconds=300)
        assert cleaned == 2
        assert "org-1" not in limiter._requests
        assert "org-2" not in limiter._requests

    def test_cleanup_keeps_active(self):
        limiter = RateLimiter()
        limiter.is_allowed("org-1")
        cleaned = limiter.cleanup(max_age_seconds=300)
        assert cleaned == 0
        assert "org-1" in limiter._requests

    def test_reset_specific_org(self):
        limiter = RateLimiter(default_limit=2)
        limiter.is_allowed("org-1")
        limiter.is_allowed("org-2")
        limiter.reset("org-1")
        assert "org-1" not in limiter._requests
        assert "org-2" in limiter._requests

    def test_reset_all(self):
        limiter = RateLimiter()
        limiter.is_allowed("org-1")
        limiter.is_allowed("org-2")
        limiter.reset()
        assert len(limiter._requests) == 0


# ── Health Check Tests ───────────────────────────────────────


class TestHealthCheck:
    """Tests for the health check response."""

    def test_health_response_format(self):
        resp = get_health_response()
        assert resp["status"] == "healthy"
        assert resp["service"] == "sovereign-venture-engine"
        assert "timestamp" in resp
        assert "version" in resp

    def test_health_timestamp_is_recent(self):
        resp = get_health_response()
        ts = datetime.fromisoformat(resp["timestamp"])
        delta = datetime.now(timezone.utc) - ts
        assert delta.total_seconds() < 2


# ── AuthContext Scope Tests (Additional) ─────────────────────


class TestScopeChecking:
    """Tests for scope-related functionality in gateway context."""

    def test_auth_context_scope_matching(self):
        ctx = AuthContext(org_id="org-1", scopes=["leads:read", "insights:write"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("insights:write") is True
        assert ctx.has_scope("insights:read") is True  # write implies read
        assert ctx.has_scope("agents:execute") is False

    def test_global_read_scope(self):
        ctx = AuthContext(org_id="org-1", scopes=["read"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("experiments:read") is True
        assert ctx.has_scope("leads:write") is False

    def test_global_write_scope(self):
        ctx = AuthContext(org_id="org-1", scopes=["write"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("leads:write") is True

    def test_empty_scopes_deny_all(self):
        ctx = AuthContext(org_id="org-1", scopes=[])
        assert ctx.has_scope("read") is False
        assert ctx.has_scope("leads:read") is False


# ── API Server Tests (using TestClient) ──────────────────────


class TestAPIServer:
    """Tests for the FastAPI app endpoints."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock DB for API server testing."""
        db = MagicMock()
        db.vertical_id = "test_vertical"

        # Default: return empty data
        mock_result = MagicMock()
        mock_result.data = []
        mock_result.execute = MagicMock(return_value=mock_result)

        db.client.table.return_value = MagicMock(
            select=MagicMock(return_value=MagicMock(
                eq=MagicMock(return_value=MagicMock(
                    order=MagicMock(return_value=MagicMock(
                        range=MagicMock(return_value=MagicMock(
                            execute=MagicMock(return_value=mock_result),
                            gte=MagicMock(return_value=MagicMock(
                                order=MagicMock(return_value=MagicMock(
                                    range=MagicMock(return_value=MagicMock(
                                        execute=MagicMock(return_value=mock_result)
                                    ))
                                ))
                            ))
                        ))
                    )),
                    gte=MagicMock(return_value=MagicMock(
                        order=MagicMock(return_value=MagicMock(
                            range=MagicMock(return_value=MagicMock(
                                execute=MagicMock(return_value=mock_result)
                            ))
                        ))
                    )),
                    limit=MagicMock(return_value=MagicMock(
                        execute=MagicMock(return_value=mock_result)
                    )),
                )),
                order=MagicMock(return_value=MagicMock(
                    execute=MagicMock(return_value=mock_result),
                    limit=MagicMock(return_value=MagicMock(
                        execute=MagicMock(return_value=mock_result)
                    )),
                    range=MagicMock(return_value=MagicMock(
                        execute=MagicMock(return_value=mock_result)
                    )),
                )),
            )),
            insert=MagicMock(return_value=MagicMock(
                execute=MagicMock(return_value=mock_result)
            )),
        )
        return db

    @pytest.fixture
    def api_app(self, mock_db):
        """Create a test API app."""
        try:
            from core.enterprise.api_server import create_api_app
            return create_api_app(db=mock_db)
        except ImportError:
            pytest.skip("FastAPI not installed")

    @pytest.fixture
    def client(self, api_app):
        """Create a test client."""
        try:
            from fastapi.testclient import TestClient
            return TestClient(api_app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_health_no_auth(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_leads_no_auth_401(self, client):
        resp = client.get("/api/v1/leads")
        assert resp.status_code == 401

    def test_leads_invalid_key_403(self, client):
        resp = client.get(
            "/api/v1/leads",
            headers={"X-API-Key": "sk_enclave_invalid_key"},
        )
        assert resp.status_code == 403

    def test_insights_no_auth_401(self, client):
        resp = client.get("/api/v1/insights")
        assert resp.status_code == 401

    def test_agents_no_auth_401(self, client):
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 401

    def test_experiments_no_auth_401(self, client):
        resp = client.get("/api/v1/experiments")
        assert resp.status_code == 401

    def test_org_no_auth_401(self, client):
        resp = client.get("/api/v1/org")
        assert resp.status_code == 401

    def test_error_response_format(self, client):
        resp = client.get("/api/v1/leads")
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data


# ── API Server Structure Tests ───────────────────────────────


class TestAPIServerStructure:
    """Tests for the API server module structure."""

    @pytest.fixture(autouse=True)
    def _require_fastapi(self):
        pytest.importorskip("fastapi", reason="FastAPI not installed")

    def test_create_api_app_importable(self):
        from core.enterprise.api_server import create_api_app
        assert callable(create_api_app)

    def test_request_models_importable(self):
        from core.enterprise.api_server import InsightCreateRequest, AgentRunRequest, APIResponse
        assert InsightCreateRequest is not None
        assert AgentRunRequest is not None
        assert APIResponse is not None

    def test_insight_create_request(self):
        from core.enterprise.api_server import InsightCreateRequest
        req = InsightCreateRequest(content="Test insight")
        assert req.content == "Test insight"
        assert req.insight_type == "market_signal"
        assert req.confidence_score == 0.7

    def test_agent_run_request(self):
        from core.enterprise.api_server import AgentRunRequest
        req = AgentRunRequest(task_input={"domain": "test.com"})
        assert req.task_input == {"domain": "test.com"}
        assert req.dry_run is False

    def test_api_response_model(self):
        from core.enterprise.api_server import APIResponse
        resp = APIResponse(data={"key": "value"}, meta={"count": 1})
        assert resp.ok is True
        assert resp.data == {"key": "value"}

    def test_api_response_error(self):
        from core.enterprise.api_server import APIResponse
        resp = APIResponse(ok=False, error="something failed")
        assert resp.ok is False
        assert resp.error == "something failed"


# ── Gateway Middleware Structure Tests ────────────────────────


class TestGatewayStructure:
    """Tests for gateway module structure."""

    def test_rate_limiter_class_exists(self):
        assert RateLimiter is not None

    def test_create_auth_dependency_callable(self):
        assert callable(create_auth_dependency)

    def test_require_scope_callable(self):
        assert callable(require_scope)

    def test_get_health_response_callable(self):
        assert callable(get_health_response)

    def test_rate_limiter_default_config(self):
        limiter = RateLimiter()
        assert limiter.default_limit == 60
        assert limiter.window_seconds == 60

    def test_rate_limiter_custom_config(self):
        limiter = RateLimiter(default_limit=100, window_seconds=120)
        assert limiter.default_limit == 100
        assert limiter.window_seconds == 120
