"""
Tests for Enterprise Models — Phase 14.

Tests Pydantic models: Organization, OrgMember, APIKey,
PlanLimits, AuthContext, PlanTier, OrgRole.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from core.enterprise.models import (
    APIKey,
    APIKeyCreate,
    AuthContext,
    Organization,
    OrgMember,
    OrgRole,
    PlanLimits,
    PlanTier,
    PLAN_TIER_LIMITS,
)


# ── PlanTier Enum ────────────────────────────────────────────


class TestPlanTier:
    """Tests for PlanTier enum."""

    def test_all_tiers_exist(self):
        assert PlanTier.FREE == "free"
        assert PlanTier.STARTER == "starter"
        assert PlanTier.PRO == "pro"
        assert PlanTier.ENTERPRISE == "enterprise"

    def test_tier_count(self):
        assert len(PlanTier) == 4

    def test_tier_from_string(self):
        assert PlanTier("free") == PlanTier.FREE
        assert PlanTier("enterprise") == PlanTier.ENTERPRISE

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            PlanTier("platinum")


# ── OrgRole Enum ─────────────────────────────────────────────


class TestOrgRole:
    """Tests for OrgRole enum."""

    def test_all_roles_exist(self):
        assert OrgRole.OWNER == "owner"
        assert OrgRole.ADMIN == "admin"
        assert OrgRole.EDITOR == "editor"
        assert OrgRole.VIEWER == "viewer"

    def test_role_count(self):
        assert len(OrgRole) == 4

    def test_role_levels_hierarchy(self):
        assert OrgRole.OWNER.level > OrgRole.ADMIN.level
        assert OrgRole.ADMIN.level > OrgRole.EDITOR.level
        assert OrgRole.EDITOR.level > OrgRole.VIEWER.level

    def test_has_at_least_owner_covers_all(self):
        assert OrgRole.OWNER.has_at_least(OrgRole.VIEWER)
        assert OrgRole.OWNER.has_at_least(OrgRole.EDITOR)
        assert OrgRole.OWNER.has_at_least(OrgRole.ADMIN)
        assert OrgRole.OWNER.has_at_least(OrgRole.OWNER)

    def test_viewer_only_covers_viewer(self):
        assert OrgRole.VIEWER.has_at_least(OrgRole.VIEWER)
        assert not OrgRole.VIEWER.has_at_least(OrgRole.EDITOR)
        assert not OrgRole.VIEWER.has_at_least(OrgRole.ADMIN)
        assert not OrgRole.VIEWER.has_at_least(OrgRole.OWNER)

    def test_editor_covers_editor_and_viewer(self):
        assert OrgRole.EDITOR.has_at_least(OrgRole.VIEWER)
        assert OrgRole.EDITOR.has_at_least(OrgRole.EDITOR)
        assert not OrgRole.EDITOR.has_at_least(OrgRole.ADMIN)

    def test_admin_covers_up_to_admin(self):
        assert OrgRole.ADMIN.has_at_least(OrgRole.VIEWER)
        assert OrgRole.ADMIN.has_at_least(OrgRole.EDITOR)
        assert OrgRole.ADMIN.has_at_least(OrgRole.ADMIN)
        assert not OrgRole.ADMIN.has_at_least(OrgRole.OWNER)


# ── Organization Model ───────────────────────────────────────


class TestOrganization:
    """Tests for Organization Pydantic model."""

    def test_basic_construction(self):
        org = Organization(name="Acme Corp", slug="acme-corp")
        assert org.name == "Acme Corp"
        assert org.slug == "acme-corp"
        assert org.plan_tier == PlanTier.FREE
        assert org.settings == {}

    def test_with_all_fields(self):
        org = Organization(
            id="uuid-123",
            name="Big Company",
            slug="big-co",
            plan_tier=PlanTier.ENTERPRISE,
            settings={"custom": True},
        )
        assert org.id == "uuid-123"
        assert org.plan_tier == PlanTier.ENTERPRISE
        assert org.settings["custom"] is True

    def test_slug_lowercase_alphanumeric(self):
        org = Organization(name="Test", slug="my-company")
        assert org.slug == "my-company"

    def test_slug_single_char(self):
        org = Organization(name="Test", slug="a")
        assert org.slug == "a"

    def test_slug_rejects_uppercase(self):
        with pytest.raises(ValueError, match="lowercase"):
            Organization(name="Test", slug="MySlug")

    def test_slug_rejects_spaces(self):
        with pytest.raises(ValueError, match="lowercase"):
            Organization(name="Test", slug="my slug")

    def test_slug_rejects_leading_hyphen(self):
        with pytest.raises(ValueError, match="cannot start"):
            Organization(name="Test", slug="-invalid")

    def test_slug_rejects_trailing_hyphen(self):
        with pytest.raises(ValueError, match="cannot start"):
            Organization(name="Test", slug="invalid-")

    def test_slug_max_length(self):
        with pytest.raises(ValueError, match="63 characters"):
            Organization(name="Test", slug="a" * 64)

    def test_name_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Organization(name="", slug="test")

    def test_name_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Organization(name="   ", slug="test")

    def test_name_max_length(self):
        with pytest.raises(ValueError, match="256 characters"):
            Organization(name="a" * 257, slug="test")

    def test_name_gets_stripped(self):
        org = Organization(name="  Acme  ", slug="acme")
        assert org.name == "Acme"


# ── OrgMember Model ──────────────────────────────────────────


class TestOrgMember:
    """Tests for OrgMember model."""

    def test_basic_construction(self):
        member = OrgMember(org_id="org-1", user_id="user-1")
        assert member.org_id == "org-1"
        assert member.user_id == "user-1"
        assert member.role == OrgRole.VIEWER
        assert member.email == ""

    def test_with_role(self):
        member = OrgMember(
            org_id="org-1",
            user_id="user-1",
            email="admin@test.com",
            role=OrgRole.ADMIN,
        )
        assert member.role == OrgRole.ADMIN
        assert member.email == "admin@test.com"

    def test_with_invited_by(self):
        member = OrgMember(
            org_id="org-1",
            user_id="user-2",
            invited_by="user-1",
        )
        assert member.invited_by == "user-1"


# ── APIKey Model ─────────────────────────────────────────────


class TestAPIKey:
    """Tests for APIKey model."""

    def test_basic_construction(self):
        key = APIKey(org_id="org-1")
        assert key.org_id == "org-1"
        assert key.name == "Default Key"
        assert key.scopes == ["read"]
        assert key.rate_limit_per_minute == 60
        assert key.is_active is True

    def test_valid_scopes(self):
        key = APIKey(org_id="org-1", scopes=["read", "leads:read", "insights:write"])
        assert len(key.scopes) == 3

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="Invalid scope"):
            APIKey(org_id="org-1", scopes=["delete_everything"])

    def test_rate_limit_too_low(self):
        with pytest.raises(ValueError, match="at least 1"):
            APIKey(org_id="org-1", rate_limit_per_minute=0)

    def test_rate_limit_too_high(self):
        with pytest.raises(ValueError, match="10,000"):
            APIKey(org_id="org-1", rate_limit_per_minute=20000)

    def test_is_expired_no_expiry(self):
        key = APIKey(org_id="org-1")
        assert key.is_expired is False

    def test_is_expired_future(self):
        key = APIKey(
            org_id="org-1",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert key.is_expired is False

    def test_is_expired_past(self):
        key = APIKey(
            org_id="org-1",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert key.is_expired is True


# ── APIKeyCreate Model ───────────────────────────────────────


class TestAPIKeyCreate:
    """Tests for APIKeyCreate input model."""

    def test_basic_construction(self):
        req = APIKeyCreate()
        assert req.name == "Default Key"
        assert req.scopes == ["read"]

    def test_custom_values(self):
        req = APIKeyCreate(
            name="Production Key",
            scopes=["read", "leads:read"],
            rate_limit_per_minute=120,
        )
        assert req.name == "Production Key"
        assert req.rate_limit_per_minute == 120

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            APIKeyCreate(name="")

    def test_name_max_length(self):
        with pytest.raises(ValueError, match="128 characters"):
            APIKeyCreate(name="a" * 129)


# ── PlanLimits ───────────────────────────────────────────────


class TestPlanLimits:
    """Tests for PlanLimits model and tier mapping."""

    def test_free_tier_limits(self):
        limits = PLAN_TIER_LIMITS[PlanTier.FREE]
        assert limits.max_verticals == 1
        assert limits.max_members == 1
        assert limits.max_api_keys == 0
        assert limits.api_rate_per_minute == 0
        assert limits.can_use_api is False
        assert limits.can_use_browser_agents is False
        assert limits.can_use_genesis is False

    def test_starter_tier_limits(self):
        limits = PLAN_TIER_LIMITS[PlanTier.STARTER]
        assert limits.max_verticals == 2
        assert limits.max_members == 3
        assert limits.can_use_api is True
        assert limits.can_use_browser_agents is False

    def test_pro_tier_limits(self):
        limits = PLAN_TIER_LIMITS[PlanTier.PRO]
        assert limits.max_verticals == 5
        assert limits.max_members == 10
        assert limits.can_use_api is True
        assert limits.can_use_browser_agents is True
        assert limits.can_use_genesis is True

    def test_enterprise_tier_limits(self):
        limits = PLAN_TIER_LIMITS[PlanTier.ENTERPRISE]
        assert limits.max_verticals == 999
        assert limits.max_api_keys == 100
        assert limits.can_use_api is True
        assert limits.features.get("sso") is True
        assert limits.features.get("audit_log") is True

    def test_all_tiers_have_limits(self):
        for tier in PlanTier:
            assert tier in PLAN_TIER_LIMITS, f"Missing limits for {tier}"

    def test_tiers_ascending_verticals(self):
        prev = 0
        for tier in [PlanTier.FREE, PlanTier.STARTER, PlanTier.PRO, PlanTier.ENTERPRISE]:
            limits = PLAN_TIER_LIMITS[tier]
            assert limits.max_verticals >= prev, f"{tier} has fewer verticals than lower tier"
            prev = limits.max_verticals


# ── AuthContext ──────────────────────────────────────────────


class TestAuthContext:
    """Tests for AuthContext model."""

    def test_basic_construction(self):
        ctx = AuthContext(org_id="org-1")
        assert ctx.org_id == "org-1"
        assert ctx.user_id == ""
        assert ctx.role == OrgRole.VIEWER
        assert ctx.plan_tier == PlanTier.FREE

    def test_has_scope_direct(self):
        ctx = AuthContext(org_id="org-1", scopes=["leads:read", "insights:write"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("insights:write") is True
        assert ctx.has_scope("agents:execute") is False

    def test_has_scope_write_implies_read(self):
        ctx = AuthContext(org_id="org-1", scopes=["insights:write"])
        assert ctx.has_scope("insights:read") is True
        assert ctx.has_scope("insights:write") is True

    def test_has_scope_global_read(self):
        ctx = AuthContext(org_id="org-1", scopes=["read"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("insights:read") is True
        assert ctx.has_scope("leads:write") is False

    def test_has_scope_global_write(self):
        ctx = AuthContext(org_id="org-1", scopes=["write"])
        assert ctx.has_scope("leads:read") is True
        assert ctx.has_scope("leads:write") is True
        assert ctx.has_scope("anything:read") is True

    def test_has_role(self):
        ctx = AuthContext(org_id="org-1", role=OrgRole.ADMIN)
        assert ctx.has_role(OrgRole.VIEWER) is True
        assert ctx.has_role(OrgRole.EDITOR) is True
        assert ctx.has_role(OrgRole.ADMIN) is True
        assert ctx.has_role(OrgRole.OWNER) is False

    def test_empty_scopes(self):
        ctx = AuthContext(org_id="org-1", scopes=[])
        assert ctx.has_scope("read") is False
        assert ctx.has_scope("leads:read") is False


# ── Migration Schema ─────────────────────────────────────────


class TestMigrationSchema:
    """Tests that verify the migration SQL structure."""

    @pytest.fixture
    def migration_sql(self):
        from pathlib import Path
        migration_path = Path(__file__).parent.parent.parent / "infrastructure" / "migrations" / "011_multi_tenant.sql"
        return migration_path.read_text()

    def test_organizations_table(self, migration_sql):
        assert "CREATE TABLE IF NOT EXISTS organizations" in migration_sql

    def test_organizations_columns(self, migration_sql):
        for col in ["name", "slug", "plan_tier", "settings", "created_at"]:
            assert col in migration_sql

    def test_plan_tier_check_constraint(self, migration_sql):
        assert "free" in migration_sql
        assert "starter" in migration_sql
        assert "pro" in migration_sql
        assert "enterprise" in migration_sql

    def test_org_members_table(self, migration_sql):
        assert "CREATE TABLE IF NOT EXISTS org_members" in migration_sql

    def test_org_members_role_constraint(self, migration_sql):
        assert "owner" in migration_sql
        assert "admin" in migration_sql
        assert "editor" in migration_sql
        assert "viewer" in migration_sql

    def test_api_keys_table(self, migration_sql):
        assert "CREATE TABLE IF NOT EXISTS api_keys" in migration_sql

    def test_api_keys_columns(self, migration_sql):
        for col in ["key_hash", "key_prefix", "scopes", "rate_limit_per_minute"]:
            assert col in migration_sql

    def test_org_id_added_to_companies(self, migration_sql):
        assert "ALTER TABLE companies" in migration_sql
        assert "org_id" in migration_sql

    def test_org_id_added_to_shared_insights(self, migration_sql):
        assert "ALTER TABLE shared_insights" in migration_sql

    def test_org_id_added_to_experiments(self, migration_sql):
        assert "ALTER TABLE experiments" in migration_sql

    def test_default_org_seed(self, migration_sql):
        assert "Default Organization" in migration_sql
        assert "'default'" in migration_sql

    def test_backfill_commands(self, migration_sql):
        # Should update existing rows to default org
        assert "UPDATE companies" in migration_sql
        assert "WHERE org_id IS NULL" in migration_sql

    def test_rls_policies(self, migration_sql):
        assert "ENABLE ROW LEVEL SECURITY" in migration_sql
        assert "CREATE POLICY" in migration_sql

    def test_get_org_stats_rpc(self, migration_sql):
        assert "get_org_stats" in migration_sql
