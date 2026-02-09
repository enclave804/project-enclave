"""
Enterprise Data Models — Phase 14.

Pydantic models for organizations, members, API keys,
plan tiers, and authentication contexts.

These models define the multi-tenancy data structures
used across the enterprise layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────


class PlanTier(str, Enum):
    """Subscription plan tiers with ascending capabilities."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class OrgRole(str, Enum):
    """Organization member roles (descending privilege)."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

    @property
    def level(self) -> int:
        """Numeric privilege level (higher = more access)."""
        return {
            OrgRole.OWNER: 100,
            OrgRole.ADMIN: 80,
            OrgRole.EDITOR: 50,
            OrgRole.VIEWER: 10,
        }[self]

    def has_at_least(self, required: OrgRole) -> bool:
        """Check if this role has at least the privilege of `required`."""
        return self.level >= required.level


# ── Organization ─────────────────────────────────────────────


class Organization(BaseModel):
    """An organization (tenant) in the platform."""

    id: str = ""
    name: str
    slug: str
    plan_tier: PlanTier = PlanTier.FREE
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Slug must be lowercase alphanumeric with hyphens."""
        import re
        if not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$", v):
            raise ValueError(
                "Slug must be lowercase alphanumeric with hyphens, "
                "cannot start/end with a hyphen"
            )
        if len(v) > 63:
            raise ValueError("Slug must be 63 characters or fewer")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Name must be non-empty and reasonable length."""
        v = v.strip()
        if not v:
            raise ValueError("Organization name cannot be empty")
        if len(v) > 256:
            raise ValueError("Organization name must be 256 characters or fewer")
        return v


# ── Organization Member ──────────────────────────────────────


class OrgMember(BaseModel):
    """A member of an organization with a specific role."""

    id: str = ""
    org_id: str
    user_id: str
    email: str = ""
    role: OrgRole = OrgRole.VIEWER
    invited_by: Optional[str] = None
    created_at: Optional[datetime] = None


# ── API Key ──────────────────────────────────────────────────


class APIKey(BaseModel):
    """An API key for programmatic access to the platform."""

    id: str = ""
    org_id: str
    key_hash: str = ""
    key_prefix: str = ""
    name: str = "Default Key"
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    rate_limit_per_minute: int = 60
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: list[str]) -> list[str]:
        """Validate that scopes are from the known set."""
        valid_scopes = {
            "read",
            "write",
            "leads:read",
            "leads:write",
            "insights:read",
            "insights:write",
            "agents:read",
            "agents:execute",
            "experiments:read",
            "experiments:write",
            "org:read",
            "org:manage",
        }
        for scope in v:
            if scope not in valid_scopes:
                raise ValueError(
                    f"Invalid scope '{scope}'. "
                    f"Valid scopes: {sorted(valid_scopes)}"
                )
        return v

    @field_validator("rate_limit_per_minute")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Rate limit must be positive and reasonable."""
        if v < 1:
            raise ValueError("Rate limit must be at least 1 request per minute")
        if v > 10000:
            raise ValueError("Rate limit cannot exceed 10,000 per minute")
        return v

    @property
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class APIKeyCreate(BaseModel):
    """Input model for creating a new API key."""

    name: str = "Default Key"
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    rate_limit_per_minute: int = 60
    expires_at: Optional[datetime] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("API key name cannot be empty")
        if len(v) > 128:
            raise ValueError("API key name must be 128 characters or fewer")
        return v


# ── Plan Limits ──────────────────────────────────────────────


class PlanLimits(BaseModel):
    """Resource limits for a subscription plan tier."""

    max_verticals: int = 1
    max_members: int = 1
    max_api_keys: int = 1
    api_rate_per_minute: int = 30
    features: dict[str, bool] = Field(default_factory=dict)

    @property
    def can_use_api(self) -> bool:
        return self.features.get("api_access", False)

    @property
    def can_use_browser_agents(self) -> bool:
        return self.features.get("browser_agents", False)

    @property
    def can_use_genesis(self) -> bool:
        return self.features.get("genesis_engine", False)


PLAN_TIER_LIMITS: dict[PlanTier, PlanLimits] = {
    PlanTier.FREE: PlanLimits(
        max_verticals=1,
        max_members=1,
        max_api_keys=0,
        api_rate_per_minute=0,
        features={
            "api_access": False,
            "browser_agents": False,
            "genesis_engine": False,
            "custom_prompts": False,
            "export_data": False,
        },
    ),
    PlanTier.STARTER: PlanLimits(
        max_verticals=2,
        max_members=3,
        max_api_keys=2,
        api_rate_per_minute=30,
        features={
            "api_access": True,
            "browser_agents": False,
            "genesis_engine": False,
            "custom_prompts": True,
            "export_data": True,
        },
    ),
    PlanTier.PRO: PlanLimits(
        max_verticals=5,
        max_members=10,
        max_api_keys=10,
        api_rate_per_minute=120,
        features={
            "api_access": True,
            "browser_agents": True,
            "genesis_engine": True,
            "custom_prompts": True,
            "export_data": True,
        },
    ),
    PlanTier.ENTERPRISE: PlanLimits(
        max_verticals=999,
        max_members=999,
        max_api_keys=100,
        api_rate_per_minute=1000,
        features={
            "api_access": True,
            "browser_agents": True,
            "genesis_engine": True,
            "custom_prompts": True,
            "export_data": True,
            "priority_support": True,
            "sso": True,
            "audit_log": True,
        },
    ),
}


# ── Auth Context ─────────────────────────────────────────────


class AuthContext(BaseModel):
    """
    Authentication context passed through API middleware.

    Contains the resolved identity after API key validation
    or Supabase Auth token verification.
    """

    org_id: str
    user_id: str = ""
    role: OrgRole = OrgRole.VIEWER
    org_slug: str = ""
    plan_tier: PlanTier = PlanTier.FREE
    scopes: list[str] = Field(default_factory=list)

    def has_scope(self, scope: str) -> bool:
        """Check if this context has a specific scope."""
        # 'write' implies 'read' for the same resource
        if scope in self.scopes:
            return True

        # Check wildcard scopes
        if "read" in self.scopes and scope.endswith(":read"):
            return True
        if "write" in self.scopes:
            return True

        # Check resource-level write implies read
        if scope.endswith(":read"):
            resource = scope.rsplit(":", 1)[0]
            if f"{resource}:write" in self.scopes:
                return True

        return False

    def has_role(self, required: OrgRole) -> bool:
        """Check if this context has at least the required role."""
        return self.role.has_at_least(required)
