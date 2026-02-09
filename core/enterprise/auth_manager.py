"""
Enterprise Auth Manager — Phase 14.

Manages organizations, members, API keys, and plan enforcement.
Built on top of the existing EnclaveDB client — all operations
go through Supabase tables with org-level isolation.

Usage:
    auth = AuthManager(db)

    # Create an organization
    org = auth.create_org("Acme Corp", "acme-corp", PlanTier.PRO)

    # Add a member
    auth.add_member(org["id"], user_id, "admin@acme.com", OrgRole.ADMIN)

    # Create an API key
    raw_key, key_record = auth.create_api_key(org["id"], "Production Key", ["read", "leads:read"])

    # Validate an API key (on every request)
    auth_ctx = auth.validate_api_key(raw_key)
    if auth_ctx:
        print(f"Authenticated as org {auth_ctx.org_slug}")
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from core.enterprise.models import (
    AuthContext,
    OrgRole,
    PlanLimits,
    PlanTier,
    PLAN_TIER_LIMITS,
)

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Enterprise authentication and authorization manager.

    Handles org CRUD, member management, API key lifecycle,
    and plan tier enforcement.
    """

    def __init__(self, db: Any):
        self.db = db

    # ── Organization CRUD ────────────────────────────────────

    def create_org(
        self,
        name: str,
        slug: str,
        plan_tier: PlanTier = PlanTier.FREE,
        settings: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Create a new organization.

        Args:
            name: Display name.
            slug: URL-safe unique identifier.
            plan_tier: Subscription tier.
            settings: Optional org-level settings.

        Returns:
            Created organization record.

        Raises:
            ValueError: If slug is already taken.
        """
        # Check slug uniqueness
        existing = self._get_org_by_slug(slug)
        if existing:
            raise ValueError(f"Organization slug '{slug}' is already taken")

        data = {
            "name": name,
            "slug": slug,
            "plan_tier": plan_tier.value if isinstance(plan_tier, PlanTier) else plan_tier,
            "settings": settings or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = (
                self.db.client.table("organizations")
                .insert(data)
                .execute()
            )
            org = result.data[0] if result.data else {}

            logger.info(
                "org_created",
                extra={
                    "org_id": org.get("id"),
                    "slug": slug,
                    "plan_tier": data["plan_tier"],
                },
            )
            return org

        except Exception as e:
            logger.error(f"Failed to create org: {e}")
            raise

    def get_org(self, org_id: str) -> Optional[dict]:
        """Get an organization by ID."""
        try:
            result = (
                self.db.client.table("organizations")
                .select("*")
                .eq("id", str(org_id))
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get org {org_id}: {e}")
            return None

    def get_org_by_slug(self, slug: str) -> Optional[dict]:
        """Get an organization by slug (public API)."""
        return self._get_org_by_slug(slug)

    def _get_org_by_slug(self, slug: str) -> Optional[dict]:
        """Internal: get org by slug."""
        try:
            result = (
                self.db.client.table("organizations")
                .select("*")
                .eq("slug", slug)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get org by slug {slug}: {e}")
            return None

    def list_orgs(self) -> list[dict]:
        """List all organizations."""
        try:
            result = (
                self.db.client.table("organizations")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list orgs: {e}")
            return []

    def update_org(self, org_id: str, **kwargs: Any) -> Optional[dict]:
        """
        Update organization fields.

        Allowed fields: name, plan_tier, settings.
        """
        allowed = {"name", "plan_tier", "settings"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_org(org_id)

        # Convert PlanTier enum to string
        if "plan_tier" in updates and isinstance(updates["plan_tier"], PlanTier):
            updates["plan_tier"] = updates["plan_tier"].value

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            result = (
                self.db.client.table("organizations")
                .update(updates)
                .eq("id", str(org_id))
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to update org {org_id}: {e}")
            return None

    # ── Member Management ────────────────────────────────────

    def add_member(
        self,
        org_id: str,
        user_id: str,
        email: str = "",
        role: OrgRole = OrgRole.VIEWER,
        invited_by: Optional[str] = None,
    ) -> dict:
        """
        Add a member to an organization.

        Args:
            org_id: Organization ID.
            user_id: Supabase auth user ID.
            email: Member's email.
            role: Role to assign.
            invited_by: User ID of the inviter.

        Returns:
            Created member record.
        """
        data = {
            "org_id": str(org_id),
            "user_id": str(user_id),
            "email": email,
            "role": role.value if isinstance(role, OrgRole) else role,
            "invited_by": str(invited_by) if invited_by else None,
        }

        try:
            result = (
                self.db.client.table("org_members")
                .insert(data)
                .execute()
            )
            member = result.data[0] if result.data else {}

            logger.info(
                "member_added",
                extra={
                    "org_id": org_id,
                    "user_id": user_id,
                    "role": data["role"],
                },
            )
            return member

        except Exception as e:
            logger.error(f"Failed to add member: {e}")
            raise

    def remove_member(self, org_id: str, user_id: str) -> bool:
        """Remove a member from an organization."""
        try:
            self.db.client.table("org_members").delete().eq(
                "org_id", str(org_id)
            ).eq("user_id", str(user_id)).execute()

            logger.info(
                "member_removed",
                extra={"org_id": org_id, "user_id": user_id},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to remove member: {e}")
            return False

    def update_role(
        self, org_id: str, user_id: str, role: OrgRole
    ) -> Optional[dict]:
        """Update a member's role."""
        try:
            result = (
                self.db.client.table("org_members")
                .update({"role": role.value if isinstance(role, OrgRole) else role})
                .eq("org_id", str(org_id))
                .eq("user_id", str(user_id))
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to update role: {e}")
            return None

    def list_members(self, org_id: str) -> list[dict]:
        """List all members of an organization."""
        try:
            result = (
                self.db.client.table("org_members")
                .select("*")
                .eq("org_id", str(org_id))
                .order("created_at")
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list members: {e}")
            return []

    def get_user_orgs(self, user_id: str) -> list[dict]:
        """
        Get all organizations a user belongs to.

        Returns org records with the user's role attached.
        """
        try:
            result = (
                self.db.client.table("org_members")
                .select("*, organizations(*)")
                .eq("user_id", str(user_id))
                .execute()
            )
            orgs = []
            for member in (result.data or []):
                org = member.get("organizations", {})
                if org:
                    org["user_role"] = member.get("role", "viewer")
                    orgs.append(org)
            return orgs
        except Exception as e:
            logger.error(f"Failed to get user orgs: {e}")
            return []

    def get_member(self, org_id: str, user_id: str) -> Optional[dict]:
        """Get a specific member record."""
        try:
            result = (
                self.db.client.table("org_members")
                .select("*")
                .eq("org_id", str(org_id))
                .eq("user_id", str(user_id))
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get member: {e}")
            return None

    # ── API Key Management ───────────────────────────────────

    def create_api_key(
        self,
        org_id: str,
        name: str = "Default Key",
        scopes: Optional[list[str]] = None,
        rate_limit_per_minute: int = 60,
        expires_at: Optional[datetime] = None,
    ) -> tuple[str, dict]:
        """
        Create a new API key.

        IMPORTANT: The raw key is returned ONLY once. After creation,
        only the hash and prefix are stored.

        Args:
            org_id: Organization this key belongs to.
            name: Human-readable key name.
            scopes: Permission scopes.
            rate_limit_per_minute: Max requests per minute.
            expires_at: Optional expiration timestamp.

        Returns:
            Tuple of (raw_key, key_record).
        """
        # Generate key
        raw_key = f"sk_enclave_{secrets.token_urlsafe(32)}"
        key_prefix = raw_key[:16]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        data = {
            "org_id": str(org_id),
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "name": name,
            "scopes": scopes or ["read"],
            "rate_limit_per_minute": rate_limit_per_minute,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_active": True,
        }

        try:
            result = (
                self.db.client.table("api_keys")
                .insert(data)
                .execute()
            )
            key_record = result.data[0] if result.data else {}

            logger.info(
                "api_key_created",
                extra={
                    "org_id": org_id,
                    "key_prefix": key_prefix,
                    "name": name,
                    "scopes": scopes or ["read"],
                },
            )

            return raw_key, key_record

        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise

    def validate_api_key(self, raw_key: str) -> Optional[AuthContext]:
        """
        Validate an API key and return auth context.

        Args:
            raw_key: The full API key string.

        Returns:
            AuthContext if valid, None if invalid/expired/inactive.
        """
        if not raw_key or not raw_key.startswith("sk_enclave_"):
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        try:
            result = (
                self.db.client.table("api_keys")
                .select("*, organizations(*)")
                .eq("key_hash", key_hash)
                .eq("is_active", True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return None

            key_record = result.data[0]
            org = key_record.get("organizations", {})

            if not org:
                return None

            # Check expiration
            expires_at = key_record.get("expires_at")
            if expires_at:
                exp_dt = datetime.fromisoformat(
                    expires_at.replace("Z", "+00:00")
                )
                if datetime.now(timezone.utc) > exp_dt:
                    return None

            # Update last_used_at (best-effort)
            try:
                self.db.client.table("api_keys").update(
                    {"last_used_at": datetime.now(timezone.utc).isoformat()}
                ).eq("id", str(key_record["id"])).execute()
            except Exception:
                pass  # Non-critical

            return AuthContext(
                org_id=str(key_record["org_id"]),
                org_slug=org.get("slug", ""),
                plan_tier=PlanTier(org.get("plan_tier", "free")),
                scopes=key_record.get("scopes", ["read"]),
            )

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke (deactivate) an API key."""
        try:
            self.db.client.table("api_keys").update(
                {"is_active": False}
            ).eq("id", str(key_id)).execute()

            logger.info("api_key_revoked", extra={"key_id": key_id})
            return True
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    def list_api_keys(self, org_id: str) -> list[dict]:
        """
        List all API keys for an organization.

        Note: key_hash is excluded from the response for security.
        """
        try:
            result = (
                self.db.client.table("api_keys")
                .select("id, org_id, key_prefix, name, scopes, rate_limit_per_minute, "
                        "expires_at, last_used_at, is_active, created_at")
                .eq("org_id", str(org_id))
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []

    # ── Plan Enforcement ─────────────────────────────────────

    def get_plan_limits(self, org_id: str) -> PlanLimits:
        """Get the plan limits for an organization."""
        org = self.get_org(org_id)
        if not org:
            return PLAN_TIER_LIMITS[PlanTier.FREE]

        tier = PlanTier(org.get("plan_tier", "free"))
        return PLAN_TIER_LIMITS.get(tier, PLAN_TIER_LIMITS[PlanTier.FREE])

    def check_plan_limit(self, org_id: str, resource: str) -> bool:
        """
        Check if an organization can use more of a resource.

        Args:
            org_id: Organization ID.
            resource: One of 'verticals', 'members', 'api_keys'.

        Returns:
            True if within limits, False if at/over limit.
        """
        limits = self.get_plan_limits(org_id)

        if resource == "verticals":
            count = self._count_verticals(org_id)
            return count < limits.max_verticals

        elif resource == "members":
            count = len(self.list_members(org_id))
            return count < limits.max_members

        elif resource == "api_keys":
            keys = self.list_api_keys(org_id)
            active_keys = sum(1 for k in keys if k.get("is_active", True))
            return active_keys < limits.max_api_keys

        return True

    def _count_verticals(self, org_id: str) -> int:
        """Count distinct verticals used by an org."""
        try:
            result = (
                self.db.client.table("companies")
                .select("vertical_id")
                .eq("org_id", str(org_id))
                .execute()
            )
            if result.data:
                return len(set(r["vertical_id"] for r in result.data))
            return 0
        except Exception:
            return 0
