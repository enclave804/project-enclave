"""
API Gateway Middleware — Phase 14.

Provides API key authentication, rate limiting, and scope checking
for the REST API gateway. Built as FastAPI middleware and dependencies.

Architecture:
    Request → RateLimiter → APIKeyAuth → ScopeCheck → Route Handler

Rate Limiting:
    - In-memory sliding window per org_id (no Redis dependency)
    - Configurable per-org limits from api_keys table
    - Falls back to plan tier default if not specified

Usage:
    from core.enterprise.gateway import create_auth_dependency, RateLimiter

    rate_limiter = RateLimiter()

    @app.get("/api/v1/leads")
    async def list_leads(auth: AuthContext = Depends(create_auth_dependency(auth_manager, rate_limiter))):
        # auth.org_id is guaranteed to be set
        ...
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.enterprise.models import AuthContext, PLAN_TIER_LIMITS, PlanTier

logger = logging.getLogger(__name__)


# ── Rate Limiter ─────────────────────────────────────────────


class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Tracks request timestamps per org_id using a sliding window
    of 60 seconds. No external dependencies (no Redis).

    Thread-safety: Not thread-safe by default. For multi-worker
    deployments, use an external rate limiter or lock.
    """

    def __init__(self, default_limit: int = 60, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        # org_id → list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, org_id: str, limit: Optional[int] = None) -> bool:
        """
        Check if a request from this org is allowed.

        Args:
            org_id: Organization identifier.
            limit: Per-minute limit override (from API key or plan tier).

        Returns:
            True if allowed, False if rate limited.
        """
        effective_limit = limit if limit is not None else self.default_limit
        if effective_limit <= 0:
            return False

        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old entries
        timestamps = self._requests[org_id]
        self._requests[org_id] = [t for t in timestamps if t > cutoff]

        # Check limit
        if len(self._requests[org_id]) >= effective_limit:
            return False

        # Record this request
        self._requests[org_id].append(now)
        return True

    def get_remaining(self, org_id: str, limit: Optional[int] = None) -> int:
        """Get remaining requests in the current window."""
        effective_limit = limit if limit is not None else self.default_limit
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = self._requests.get(org_id, [])
        active = [t for t in timestamps if t > cutoff]

        return max(0, effective_limit - len(active))

    def get_reset_time(self, org_id: str) -> float:
        """Get seconds until the rate limit window resets."""
        timestamps = self._requests.get(org_id, [])
        if not timestamps:
            return 0.0

        now = time.time()
        cutoff = now - self.window_seconds
        active = [t for t in timestamps if t > cutoff]

        if not active:
            return 0.0

        oldest = min(active)
        return max(0.0, (oldest + self.window_seconds) - now)

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """
        Remove stale entries older than max_age_seconds.

        Call periodically to prevent memory leaks for orgs
        that stop making requests.

        Returns:
            Number of org entries cleaned up.
        """
        cutoff = time.time() - max_age_seconds
        stale_orgs = []

        for org_id, timestamps in self._requests.items():
            active = [t for t in timestamps if t > cutoff]
            if not active:
                stale_orgs.append(org_id)
            else:
                self._requests[org_id] = active

        for org_id in stale_orgs:
            del self._requests[org_id]

        return len(stale_orgs)

    def reset(self, org_id: Optional[str] = None) -> None:
        """Reset rate limit state for an org or all orgs."""
        if org_id:
            self._requests.pop(org_id, None)
        else:
            self._requests.clear()


# ── API Key Authentication ───────────────────────────────────


def create_auth_dependency(
    auth_manager: Any,
    rate_limiter: RateLimiter,
) -> Callable:
    """
    Create a FastAPI dependency for API key authentication.

    Usage:
        auth_dep = create_auth_dependency(auth_manager, rate_limiter)

        @app.get("/api/v1/leads")
        async def list_leads(auth: AuthContext = Depends(auth_dep)):
            ...
    """
    from fastapi import HTTPException, Request

    async def authenticate(request: Request) -> AuthContext:
        # Extract API key from header
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Missing X-API-Key header",
                },
            )

        # Validate key
        auth_ctx = auth_manager.validate_api_key(api_key)
        if auth_ctx is None:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "forbidden",
                    "message": "Invalid or expired API key",
                },
            )

        # Rate limiting
        plan_limits = PLAN_TIER_LIMITS.get(auth_ctx.plan_tier, PLAN_TIER_LIMITS[PlanTier.FREE])
        rate_limit = plan_limits.api_rate_per_minute

        if not rate_limiter.is_allowed(auth_ctx.org_id, limit=rate_limit):
            remaining = rate_limiter.get_remaining(auth_ctx.org_id, limit=rate_limit)
            reset_time = rate_limiter.get_reset_time(auth_ctx.org_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": "Too many requests",
                    "retry_after_seconds": round(reset_time, 1),
                },
                headers={
                    "Retry-After": str(int(reset_time) + 1),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

        # Store auth context on request state for downstream use
        request.state.auth_context = auth_ctx

        logger.info(
            "api_request_authenticated",
            extra={
                "org_id": auth_ctx.org_id,
                "org_slug": auth_ctx.org_slug,
                "plan_tier": auth_ctx.plan_tier.value,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return auth_ctx

    return authenticate


# ── Scope Checking ───────────────────────────────────────────


def require_scope(scope: str) -> Callable:
    """
    Create a FastAPI dependency that checks for a specific scope.

    Usage:
        @app.post("/api/v1/insights")
        async def create_insight(
            auth: AuthContext = Depends(auth_dep),
            _scope: None = Depends(require_scope("insights:write")),
        ):
            ...
    """
    from fastapi import HTTPException, Request

    async def check_scope(request: Request) -> None:
        auth_ctx: Optional[AuthContext] = getattr(request.state, "auth_context", None)
        if auth_ctx is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )

        if not auth_ctx.has_scope(scope):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "insufficient_scope",
                    "message": f"Required scope: {scope}",
                    "current_scopes": auth_ctx.scopes,
                },
            )

    return check_scope


# ── Health Check (No Auth) ───────────────────────────────────


def get_health_response() -> dict[str, Any]:
    """Generate a health check response (no auth required)."""
    return {
        "status": "healthy",
        "service": "sovereign-venture-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }
