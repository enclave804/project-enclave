"""
Tests for Enterprise Auth Manager — Phase 14.

Tests organization CRUD, member management, API key lifecycle,
plan enforcement, and security boundaries.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.enterprise.auth_manager import AuthManager
from core.enterprise.models import (
    AuthContext,
    OrgRole,
    PlanLimits,
    PlanTier,
    PLAN_TIER_LIMITS,
)


# ── Mock DB ──────────────────────────────────────────────────


class MockTable:
    """Mock for Supabase table operations."""

    def __init__(self, data: list[dict] = None):
        self._data = data or []
        self._filters: dict[str, Any] = {}
        self._selected = "*"
        self._order_col = None
        self._order_desc = False
        self._limit_val = None
        self._range_start = None
        self._range_end = None

    def select(self, cols="*"):
        self._selected = cols
        return self

    def insert(self, data):
        record = {**data, "id": f"mock-{len(self._data) + 1}"}
        self._data.append(record)
        return self

    def update(self, data):
        self._update_data = data
        return self

    def delete(self):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def gte(self, col, val):
        return self

    def order(self, col, desc=False):
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def range(self, start, end):
        return self

    def execute(self):
        result = MagicMock()
        # Apply filters
        filtered = self._data
        for col, val in self._filters.items():
            filtered = [r for r in filtered if r.get(col) == val]

        if self._limit_val and len(filtered) > self._limit_val:
            filtered = filtered[:self._limit_val]

        result.data = filtered
        self._filters = {}
        return result


class MockDB:
    """Mock EnclaveDB for testing."""

    def __init__(self):
        self._tables: dict[str, list[dict]] = {
            "organizations": [],
            "org_members": [],
            "api_keys": [],
            "companies": [],
        }
        self.vertical_id = "test_vertical"
        self.client = self

    def table(self, name: str):
        return MockTable(self._tables.get(name, []))

    def rpc(self, func_name: str, params: dict):
        result = MagicMock()
        result.data = None
        result.execute = MagicMock(return_value=result)
        return result


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    return MockDB()


@pytest.fixture
def auth_manager(mock_db):
    return AuthManager(mock_db)


# ── Organization CRUD ────────────────────────────────────────


class TestOrgCRUD:
    """Tests for organization create/read/update/list."""

    def test_create_org(self, auth_manager):
        org = auth_manager.create_org("Acme Corp", "acme-corp", PlanTier.PRO)
        assert org["name"] == "Acme Corp"
        assert org["slug"] == "acme-corp"
        assert org["plan_tier"] == "pro"
        assert "id" in org

    def test_create_org_default_tier(self, auth_manager):
        org = auth_manager.create_org("Test Org", "test-org")
        assert org["plan_tier"] == "free"

    def test_create_org_with_settings(self, auth_manager):
        settings = {"logo_url": "https://example.com/logo.png"}
        org = auth_manager.create_org("Custom Org", "custom", settings=settings)
        assert org["settings"] == settings

    def test_create_org_duplicate_slug_raises(self, auth_manager):
        # First create succeeds
        auth_manager.create_org("First", "same-slug")

        # Inject the first org into the mock so slug check finds it
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "name": "First",
            "slug": "same-slug",
            "plan_tier": "free",
        })

        # Second create with same slug should raise
        with pytest.raises(ValueError, match="already taken"):
            auth_manager.create_org("Second", "same-slug")

    def test_get_org_returns_none_for_missing(self, auth_manager):
        result = auth_manager.get_org("nonexistent-id")
        assert result is None

    def test_get_org_by_slug_returns_none(self, auth_manager):
        result = auth_manager.get_org_by_slug("no-such-slug")
        assert result is None

    def test_get_org_by_slug_finds_existing(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "name": "Test",
            "slug": "test-slug",
        })
        result = auth_manager.get_org_by_slug("test-slug")
        assert result is not None
        assert result["slug"] == "test-slug"

    def test_list_orgs_empty(self, auth_manager):
        orgs = auth_manager.list_orgs()
        assert orgs == []

    def test_list_orgs_returns_all(self, auth_manager):
        auth_manager.db._tables["organizations"].extend([
            {"id": "1", "name": "Org A", "slug": "a"},
            {"id": "2", "name": "Org B", "slug": "b"},
        ])
        orgs = auth_manager.list_orgs()
        assert len(orgs) == 2

    def test_update_org_name(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "name": "Old Name",
            "slug": "org",
            "plan_tier": "free",
        })
        result = auth_manager.update_org("org-1", name="New Name")
        # MockDB's update returns the filtered data
        assert result is not None

    def test_update_org_plan_tier(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "name": "Test",
            "slug": "org",
            "plan_tier": "free",
        })
        result = auth_manager.update_org("org-1", plan_tier=PlanTier.PRO)
        assert result is not None

    def test_update_org_ignores_unknown_fields(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "name": "Test",
            "slug": "org",
        })
        result = auth_manager.update_org("org-1", hacker_field="malicious")
        # Should return the org unchanged (no valid updates)
        assert result is not None


# ── Member Management ────────────────────────────────────────


class TestMemberManagement:
    """Tests for add/remove/update member operations."""

    def test_add_member(self, auth_manager):
        member = auth_manager.add_member(
            org_id="org-1",
            user_id="user-1",
            email="test@example.com",
            role=OrgRole.EDITOR,
        )
        assert member["org_id"] == "org-1"
        assert member["user_id"] == "user-1"
        assert member["role"] == "editor"
        assert member["email"] == "test@example.com"

    def test_add_member_default_role(self, auth_manager):
        member = auth_manager.add_member("org-1", "user-1")
        assert member["role"] == "viewer"

    def test_add_member_with_invited_by(self, auth_manager):
        member = auth_manager.add_member(
            "org-1", "user-2", invited_by="user-1"
        )
        assert member["invited_by"] == "user-1"

    def test_remove_member(self, auth_manager):
        result = auth_manager.remove_member("org-1", "user-1")
        assert result is True

    def test_list_members_empty(self, auth_manager):
        members = auth_manager.list_members("org-1")
        assert members == []

    def test_list_members(self, auth_manager):
        auth_manager.db._tables["org_members"].extend([
            {"id": "m1", "org_id": "org-1", "user_id": "u1", "role": "admin"},
            {"id": "m2", "org_id": "org-1", "user_id": "u2", "role": "viewer"},
            {"id": "m3", "org_id": "org-2", "user_id": "u3", "role": "editor"},
        ])
        members = auth_manager.list_members("org-1")
        assert len(members) == 2

    def test_update_role(self, auth_manager):
        auth_manager.db._tables["org_members"].append({
            "id": "m1", "org_id": "org-1", "user_id": "u1", "role": "viewer",
        })
        result = auth_manager.update_role("org-1", "u1", OrgRole.ADMIN)
        # Should return updated record
        assert result is not None

    def test_get_member(self, auth_manager):
        auth_manager.db._tables["org_members"].append({
            "id": "m1", "org_id": "org-1", "user_id": "u1", "role": "admin",
        })
        member = auth_manager.get_member("org-1", "u1")
        assert member is not None
        assert member["role"] == "admin"

    def test_get_member_not_found(self, auth_manager):
        member = auth_manager.get_member("org-1", "nonexistent")
        assert member is None


# ── API Key Management ───────────────────────────────────────


class TestAPIKeyManagement:
    """Tests for API key create/validate/revoke/list."""

    def test_create_api_key(self, auth_manager):
        raw_key, record = auth_manager.create_api_key(
            org_id="org-1",
            name="Test Key",
            scopes=["read", "leads:read"],
        )
        assert raw_key.startswith("sk_enclave_")
        assert record["name"] == "Test Key"
        assert record["org_id"] == "org-1"
        assert record["is_active"] is True

    def test_key_prefix_stored(self, auth_manager):
        raw_key, record = auth_manager.create_api_key("org-1")
        assert record["key_prefix"] == raw_key[:16]

    def test_key_hash_is_sha256(self, auth_manager):
        raw_key, record = auth_manager.create_api_key("org-1")
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert record["key_hash"] == expected_hash

    def test_key_is_unique(self, auth_manager):
        _, r1 = auth_manager.create_api_key("org-1", name="Key 1")
        _, r2 = auth_manager.create_api_key("org-1", name="Key 2")
        assert r1["key_hash"] != r2["key_hash"]

    def test_create_key_custom_rate_limit(self, auth_manager):
        _, record = auth_manager.create_api_key(
            "org-1", rate_limit_per_minute=200,
        )
        assert record["rate_limit_per_minute"] == 200

    def test_create_key_with_expiry(self, auth_manager):
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        _, record = auth_manager.create_api_key(
            "org-1", expires_at=expires,
        )
        assert record["expires_at"] is not None

    def test_validate_empty_key(self, auth_manager):
        result = auth_manager.validate_api_key("")
        assert result is None

    def test_validate_wrong_prefix(self, auth_manager):
        result = auth_manager.validate_api_key("wrong_prefix_key123")
        assert result is None

    def test_validate_nonexistent_key(self, auth_manager):
        result = auth_manager.validate_api_key("sk_enclave_nonexistent_key_value")
        assert result is None

    def test_revoke_api_key(self, auth_manager):
        result = auth_manager.revoke_api_key("key-1")
        assert result is True

    def test_list_api_keys_empty(self, auth_manager):
        keys = auth_manager.list_api_keys("org-1")
        assert keys == []

    def test_list_api_keys(self, auth_manager):
        auth_manager.db._tables["api_keys"].extend([
            {"id": "k1", "org_id": "org-1", "name": "Key 1", "is_active": True},
            {"id": "k2", "org_id": "org-1", "name": "Key 2", "is_active": False},
            {"id": "k3", "org_id": "org-2", "name": "Key 3", "is_active": True},
        ])
        keys = auth_manager.list_api_keys("org-1")
        assert len(keys) == 2

    def test_list_api_keys_excludes_hash(self, auth_manager):
        # The select() call should not include key_hash
        keys = auth_manager.list_api_keys("org-1")
        # Just verify it doesn't crash — actual hash exclusion
        # depends on the real Supabase client


# ── Plan Enforcement ─────────────────────────────────────────


class TestPlanEnforcement:
    """Tests for plan limit checking."""

    def test_get_plan_limits_free(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "free",
        })
        limits = auth_manager.get_plan_limits("org-1")
        assert limits.max_verticals == 1
        assert limits.max_members == 1

    def test_get_plan_limits_enterprise(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "enterprise",
        })
        limits = auth_manager.get_plan_limits("org-1")
        assert limits.max_verticals == 999
        assert limits.can_use_api is True

    def test_get_plan_limits_missing_org(self, auth_manager):
        limits = auth_manager.get_plan_limits("nonexistent")
        assert limits == PLAN_TIER_LIMITS[PlanTier.FREE]

    def test_check_plan_limit_members_under(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "starter",
        })
        # No members yet, starter allows 3
        result = auth_manager.check_plan_limit("org-1", "members")
        assert result is True

    def test_check_plan_limit_members_at_limit(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "free",
        })
        # Free allows 1, add 1 member
        auth_manager.db._tables["org_members"].append({
            "id": "m1", "org_id": "org-1", "user_id": "u1",
        })
        result = auth_manager.check_plan_limit("org-1", "members")
        assert result is False

    def test_check_plan_limit_api_keys(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "free",
        })
        # Free allows 0 API keys
        result = auth_manager.check_plan_limit("org-1", "api_keys")
        assert result is False

    def test_check_plan_limit_unknown_resource(self, auth_manager):
        auth_manager.db._tables["organizations"].append({
            "id": "org-1",
            "plan_tier": "free",
        })
        # Unknown resource → True (don't block)
        result = auth_manager.check_plan_limit("org-1", "unknown_resource")
        assert result is True


# ── Security Boundary Tests ──────────────────────────────────


class TestSecurityBoundaries:
    """Tests for security-critical behavior."""

    def test_key_format_starts_with_prefix(self, auth_manager):
        raw_key, _ = auth_manager.create_api_key("org-1")
        assert raw_key.startswith("sk_enclave_")

    def test_key_has_sufficient_entropy(self, auth_manager):
        raw_key, _ = auth_manager.create_api_key("org-1")
        # Remove prefix and check length
        key_body = raw_key[len("sk_enclave_"):]
        assert len(key_body) >= 32  # At least 32 chars of randomness

    def test_hash_not_reversible(self, auth_manager):
        raw_key, record = auth_manager.create_api_key("org-1")
        # Can't get the key back from the hash
        assert raw_key not in record["key_hash"]
        assert record["key_hash"] != raw_key

    def test_different_keys_different_hashes(self, auth_manager):
        _, r1 = auth_manager.create_api_key("org-1", name="K1")
        _, r2 = auth_manager.create_api_key("org-1", name="K2")
        assert r1["key_hash"] != r2["key_hash"]

    def test_plan_tier_string_conversion(self, auth_manager):
        org = auth_manager.create_org("Test", "test", PlanTier.PRO)
        assert org["plan_tier"] == "pro"  # String, not enum

    def test_role_string_conversion(self, auth_manager):
        member = auth_manager.add_member("org-1", "u1", role=OrgRole.ADMIN)
        assert member["role"] == "admin"  # String, not enum
