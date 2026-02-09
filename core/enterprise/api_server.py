"""
Enterprise API Server — Phase 14.

FastAPI application providing REST API endpoints for the
Sovereign Venture Engine platform. All endpoints are authenticated
via API keys and scoped to the requesting organization.

Usage:
    from core.enterprise.api_server import create_api_app

    app = create_api_app(db=db, embedder=embedder)
    uvicorn.run(app, host="0.0.0.0", port=8000)

Endpoints:
    GET  /api/v1/health          — Health check (no auth)
    GET  /api/v1/org             — Current org info
    GET  /api/v1/leads           — List companies
    GET  /api/v1/insights        — List shared insights
    POST /api/v1/insights        — Create insight
    GET  /api/v1/agents          — List agents
    POST /api/v1/agents/{id}/run — Trigger agent run
    GET  /api/v1/experiments     — List experiments
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.enterprise.auth_manager import AuthManager
from core.enterprise.gateway import (
    RateLimiter,
    create_auth_dependency,
    get_health_response,
    require_scope,
)
from core.enterprise.models import AuthContext

logger = logging.getLogger(__name__)


# ── Request/Response Models ──────────────────────────────────


class InsightCreateRequest(BaseModel):
    """Request body for creating an insight."""
    insight_type: str = "market_signal"
    title: str = ""
    content: str
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0)
    source_agent_id: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    """Request body for triggering an agent run."""
    task_input: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class APIResponse(BaseModel):
    """Standard API response envelope."""
    ok: bool = True
    data: Any = None
    error: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ── App Factory ──────────────────────────────────────────────


def create_api_app(
    db: Any,
    embedder: Any = None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    Create the FastAPI application with all enterprise routes.

    Args:
        db: EnclaveDB instance (used for data queries).
        embedder: Embedding model for RAG search (optional).
        cors_origins: Allowed CORS origins (default: all).

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Sovereign Venture Engine API",
        description="Enterprise REST API for the Sovereign Venture Engine platform.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth setup
    auth_manager = AuthManager(db)
    rate_limiter = RateLimiter(default_limit=60)
    auth_dep = create_auth_dependency(auth_manager, rate_limiter)

    # ── Health ───────────────────────────────────────────

    @app.get("/api/v1/health", tags=["Health"])
    async def health_check():
        """Health check endpoint (no authentication required)."""
        return get_health_response()

    # ── Organization ─────────────────────────────────────

    @app.get("/api/v1/org", tags=["Organization"])
    async def get_current_org(
        auth: AuthContext = Depends(auth_dep),
    ):
        """Get the current organization info."""
        org = auth_manager.get_org(auth.org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Remove sensitive fields
        org.pop("settings", None)

        return APIResponse(
            data=org,
            meta={"plan_tier": auth.plan_tier.value},
        )

    # ── Leads (Companies) ────────────────────────────────

    @app.get("/api/v1/leads", tags=["Leads"])
    async def list_leads(
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("leads:read")),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        industry: Optional[str] = None,
    ):
        """List companies/leads for the authenticated organization."""
        try:
            query = (
                db.client.table("companies")
                .select("*")
                .eq("org_id", auth.org_id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
            )
            if industry:
                query = query.eq("industry", industry)

            result = query.execute()

            return APIResponse(
                data=result.data or [],
                meta={
                    "count": len(result.data or []),
                    "limit": limit,
                    "offset": offset,
                },
            )
        except Exception as e:
            logger.error(f"API list_leads failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Insights ─────────────────────────────────────────

    @app.get("/api/v1/insights", tags=["Insights"])
    async def list_insights(
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("insights:read")),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        insight_type: Optional[str] = None,
        min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    ):
        """List shared insights for the authenticated organization."""
        try:
            query = (
                db.client.table("shared_insights")
                .select("*")
                .eq("org_id", auth.org_id)
                .gte("confidence_score", min_confidence)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
            )
            if insight_type:
                query = query.eq("insight_type", insight_type)

            result = query.execute()

            return APIResponse(
                data=result.data or [],
                meta={
                    "count": len(result.data or []),
                    "limit": limit,
                    "offset": offset,
                },
            )
        except Exception as e:
            logger.error(f"API list_insights failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/insights", tags=["Insights"])
    async def create_insight(
        body: InsightCreateRequest,
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("insights:write")),
    ):
        """Create a new shared insight."""
        try:
            data = {
                "insight_type": body.insight_type,
                "title": body.title or body.content[:100],
                "content": body.content,
                "confidence_score": body.confidence_score,
                "source_agent_id": body.source_agent_id,
                "metadata": body.metadata,
                "vertical_id": db.vertical_id,
                "org_id": auth.org_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            result = (
                db.client.table("shared_insights")
                .insert(data)
                .execute()
            )

            return APIResponse(
                data=result.data[0] if result.data else {},
                meta={"created": True},
            )
        except Exception as e:
            logger.error(f"API create_insight failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Agents ───────────────────────────────────────────

    @app.get("/api/v1/agents", tags=["Agents"])
    async def list_agents(
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("agents:read")),
    ):
        """List registered agents."""
        try:
            result = (
                db.client.table("agents")
                .select("agent_id, agent_type, name, enabled, shadow_mode, vertical_id, created_at")
                .order("created_at", desc=True)
                .execute()
            )

            return APIResponse(
                data=result.data or [],
                meta={"count": len(result.data or [])},
            )
        except Exception as e:
            logger.error(f"API list_agents failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/agents/{agent_id}/run", tags=["Agents"])
    async def trigger_agent_run(
        agent_id: str,
        body: AgentRunRequest,
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("agents:execute")),
    ):
        """
        Trigger an agent run.

        Note: This enqueues a task — it doesn't block until completion.
        """
        try:
            task_data = {
                "source_agent_id": "api",
                "target_agent_id": agent_id,
                "vertical_id": db.vertical_id,
                "priority": 5,
                "status": "pending",
                "input_data": body.task_input,
                "metadata": {
                    "triggered_by": "api",
                    "org_id": auth.org_id,
                    "dry_run": body.dry_run,
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            result = (
                db.client.table("task_queue")
                .insert(task_data)
                .execute()
            )

            task = result.data[0] if result.data else {}

            return APIResponse(
                data={
                    "task_id": task.get("id", task.get("task_id")),
                    "agent_id": agent_id,
                    "status": "queued",
                    "dry_run": body.dry_run,
                },
                meta={"queued": True},
            )
        except Exception as e:
            logger.error(f"API trigger_agent_run failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Experiments ──────────────────────────────────────

    @app.get("/api/v1/experiments", tags=["Experiments"])
    async def list_experiments(
        auth: AuthContext = Depends(auth_dep),
        _scope: None = Depends(require_scope("experiments:read")),
        status: Optional[str] = None,
        limit: int = Query(default=20, ge=1, le=100),
    ):
        """List experiments for the authenticated organization."""
        try:
            query = (
                db.client.table("experiments")
                .select("*")
                .eq("org_id", auth.org_id)
                .order("created_at", desc=True)
                .limit(limit)
            )
            if status:
                query = query.eq("status", status)

            result = query.execute()

            return APIResponse(
                data=result.data or [],
                meta={"count": len(result.data or [])},
            )
        except Exception as e:
            logger.error(f"API list_experiments failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Error Handlers ───────────────────────────────────

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Return consistent JSON error responses."""
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content={"ok": False, "error": detail},
                headers=getattr(exc, "headers", None),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "error": {"message": str(detail)},
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all error handler."""
        logger.error(f"Unhandled API error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {"message": "Internal server error"},
            },
        )

    return app
