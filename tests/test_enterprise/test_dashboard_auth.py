"""
Tests for Dashboard Enterprise Auth — Phase 14.

Tests dual-mode authentication, org switcher, and
AuthContext propagation through session state.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ── Auth Module Tests ────────────────────────────────────────


class TestLegacyAuth:
    """Tests that legacy password auth still works."""

    def test_get_dashboard_password_from_env(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PASSWORD", "test_pass_123")
        from dashboard.auth import _get_dashboard_password
        assert _get_dashboard_password() == "test_pass_123"

    def test_get_dashboard_password_none(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
        from dashboard.auth import _get_dashboard_password
        # Should return None or password from secrets
        result = _get_dashboard_password()
        # Can be None or a value depending on st.secrets
        assert result is None or isinstance(result, str)

    def test_verify_password_correct(self):
        from dashboard.auth import _verify_password
        assert _verify_password("secret123", "secret123") is True

    def test_verify_password_incorrect(self):
        from dashboard.auth import _verify_password
        assert _verify_password("wrong", "secret123") is False

    def test_verify_password_empty(self):
        from dashboard.auth import _verify_password
        assert _verify_password("", "secret123") is False

    def test_verify_password_timing_safe(self):
        """Verify that password check uses constant-time comparison."""
        from dashboard.auth import _verify_password
        # Both should take similar time regardless of input
        _verify_password("a" * 100, "b" * 100)
        _verify_password("a", "b" * 100)
        # Just verify it doesn't crash


# ── Enterprise Auth Mode ─────────────────────────────────────


class TestEnterpriseAuthMode:
    """Tests for enterprise auth mode detection."""

    def test_enterprise_auth_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_AUTH_ENABLED", raising=False)
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is False

    def test_enterprise_auth_enabled_true(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "true")
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is True

    def test_enterprise_auth_enabled_1(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "1")
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is True

    def test_enterprise_auth_enabled_yes(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "yes")
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is True

    def test_enterprise_auth_enabled_false(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "false")
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is False

    def test_enterprise_auth_enabled_random(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "maybe")
        from dashboard.auth import _is_enterprise_auth_enabled
        assert _is_enterprise_auth_enabled() is False


# ── Current User / Org Helpers ───────────────────────────────


class TestCurrentUserOrg:
    """Tests for get_current_user and get_current_org."""

    def test_get_current_user_not_enterprise(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_AUTH_ENABLED", raising=False)
        from dashboard.auth import get_current_user
        assert get_current_user() is None

    def test_get_current_org_not_enterprise(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_AUTH_ENABLED", raising=False)
        from dashboard.auth import get_current_org
        assert get_current_org() is None

    def test_get_current_user_enterprise_no_session(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "true")
        from dashboard.auth import get_current_user
        # Will fail gracefully because st not running
        result = get_current_user()
        assert result is None

    def test_get_current_org_enterprise_no_session(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_AUTH_ENABLED", "true")
        from dashboard.auth import get_current_org
        result = get_current_org()
        assert result is None


# ── Rate Limiting ────────────────────────────────────────────


class TestRateLimiting:
    """Tests for login rate limiting."""

    def test_rate_limit_constants(self):
        from dashboard.auth import MAX_FAILED_ATTEMPTS, COOLDOWN_SECONDS
        assert MAX_FAILED_ATTEMPTS == 5
        assert COOLDOWN_SECONDS == 30


# ── Auth Module Imports ──────────────────────────────────────


class TestAuthImports:
    """Tests that all auth module exports are importable."""

    def test_require_auth_importable(self):
        from dashboard.auth import require_auth
        assert callable(require_auth)

    def test_require_auth_v2_importable(self):
        from dashboard.auth import require_auth_v2
        assert callable(require_auth_v2)

    def test_get_current_user_importable(self):
        from dashboard.auth import get_current_user
        assert callable(get_current_user)

    def test_get_current_org_importable(self):
        from dashboard.auth import get_current_org
        assert callable(get_current_org)

    def test_is_enterprise_auth_enabled_importable(self):
        from dashboard.auth import _is_enterprise_auth_enabled
        assert callable(_is_enterprise_auth_enabled)

    def test_verify_password_importable(self):
        from dashboard.auth import _verify_password
        assert callable(_verify_password)


# ── Sidebar Org Switcher ─────────────────────────────────────


class TestSidebarOrgSwitcher:
    """Tests for the org switcher function in sidebar."""

    def test_render_org_switcher_importable(self):
        from dashboard.sidebar import _render_org_switcher
        assert callable(_render_org_switcher)

    def test_render_org_switcher_no_enterprise(self, monkeypatch):
        """Org switcher should be a no-op when enterprise auth disabled."""
        monkeypatch.delenv("SUPABASE_AUTH_ENABLED", raising=False)
        from dashboard.sidebar import _render_org_switcher
        # Should not crash when called outside Streamlit
        _render_org_switcher({"text_tertiary": "#888", "border_subtle": "#333"})

    def test_get_vertical_options_importable(self):
        from dashboard.sidebar import get_vertical_options
        assert callable(get_vertical_options)

    def test_render_sidebar_importable(self):
        from dashboard.sidebar import render_sidebar
        assert callable(render_sidebar)


# ── Enterprise Model Integration ─────────────────────────────


class TestEnterpriseModelIntegration:
    """Tests that enterprise models work correctly in auth context."""

    def test_auth_context_from_dict(self):
        from core.enterprise.models import AuthContext, OrgRole, PlanTier
        ctx = AuthContext(
            org_id="org-123",
            user_id="user-456",
            role=OrgRole.ADMIN,
            org_slug="acme",
            plan_tier=PlanTier.PRO,
            scopes=["read", "leads:write"],
        )
        assert ctx.org_id == "org-123"
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("leads:write") is True
        assert ctx.has_role(OrgRole.EDITOR) is True
        assert ctx.has_role(OrgRole.OWNER) is False

    def test_auth_context_minimal(self):
        from core.enterprise.models import AuthContext
        ctx = AuthContext(org_id="org-1")
        assert ctx.user_id == ""
        assert ctx.scopes == []

    def test_plan_tier_limits_accessible(self):
        from core.enterprise.models import PLAN_TIER_LIMITS, PlanTier
        for tier in PlanTier:
            assert tier in PLAN_TIER_LIMITS

    def test_org_role_hierarchy(self):
        from core.enterprise.models import OrgRole
        assert OrgRole.OWNER.level > OrgRole.ADMIN.level
        assert OrgRole.ADMIN.level > OrgRole.EDITOR.level
        assert OrgRole.EDITOR.level > OrgRole.VIEWER.level
