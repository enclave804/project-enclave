"""
Supabase client wrapper for Project Enclave.

Provides typed operations for all database tables and pgvector-powered
RAG queries. All operations are scoped to the current vertical.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from supabase import create_client, Client


class EnclaveDB:
    """
    Database client for Project Enclave.

    All queries are automatically scoped to the configured vertical_id.
    Uses the Supabase service role key (bypasses RLS) but applies
    vertical filtering in application code for safety.
    """

    def __init__(self, vertical_id: str):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
            )
        self.client: Client = create_client(url, key)
        self.vertical_id = vertical_id

    # ------------------------------------------------------------------
    # Companies
    # ------------------------------------------------------------------

    def upsert_company(self, data: dict[str, Any]) -> dict:
        """Insert or update a company by domain + vertical_id."""
        data["vertical_id"] = self.vertical_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table("companies")
            .upsert(data, on_conflict="domain,vertical_id")
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_company_by_domain(self, domain: str) -> Optional[dict]:
        """Find a company by its domain."""
        result = (
            self.client.table("companies")
            .select("*")
            .eq("domain", domain)
            .eq("vertical_id", self.vertical_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_company(self, company_id: str) -> Optional[dict]:
        """Get a company by ID."""
        result = (
            self.client.table("companies")
            .select("*")
            .eq("id", company_id)
            .eq("vertical_id", self.vertical_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def list_companies(
        self,
        limit: int = 50,
        offset: int = 0,
        industry: Optional[str] = None,
    ) -> list[dict]:
        """List companies with optional filters."""
        query = (
            self.client.table("companies")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if industry:
            query = query.eq("industry", industry)
        return query.execute().data

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def upsert_contact(self, data: dict[str, Any]) -> dict:
        """Insert or update a contact by email + vertical_id."""
        data["vertical_id"] = self.vertical_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table("contacts")
            .upsert(data, on_conflict="email,vertical_id")
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_contact_by_email(self, email: str) -> Optional[dict]:
        """Find a contact by email."""
        result = (
            self.client.table("contacts")
            .select("*, companies(*)")
            .eq("email", email)
            .eq("vertical_id", self.vertical_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_contacts_for_company(self, company_id: str) -> list[dict]:
        """Get all contacts at a company."""
        return (
            self.client.table("contacts")
            .select("*")
            .eq("company_id", company_id)
            .eq("vertical_id", self.vertical_id)
            .execute()
        ).data

    # ------------------------------------------------------------------
    # Outreach Events
    # ------------------------------------------------------------------

    def create_outreach_event(self, data: dict[str, Any]) -> dict:
        """Record a new outreach event."""
        data["vertical_id"] = self.vertical_id
        if "content_hash" not in data and "body_preview" in data:
            data["content_hash"] = hashlib.sha256(
                data["body_preview"].encode()
            ).hexdigest()[:16]
        result = (
            self.client.table("outreach_events")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def update_outreach_event(
        self, event_id: str, updates: dict[str, Any]
    ) -> dict:
        """Update an outreach event (e.g., mark as opened/replied)."""
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table("outreach_events")
            .update(updates)
            .eq("id", event_id)
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_recent_outreach_for_contact(
        self, contact_id: str, days: int = 90
    ) -> list[dict]:
        """Get outreach events for a contact within the last N days."""
        cutoff = datetime.now(timezone.utc).isoformat()
        return (
            self.client.table("outreach_events")
            .select("*")
            .eq("contact_id", contact_id)
            .eq("vertical_id", self.vertical_id)
            .gte("sent_at", cutoff)
            .order("sent_at", desc=True)
            .execute()
        ).data

    def get_outreach_stats(
        self, days: int = 30
    ) -> dict[str, Any]:
        """Get aggregate outreach statistics for the vertical."""
        # This uses a raw SQL query via Supabase RPC
        result = self.client.rpc(
            "get_outreach_stats",
            {"p_vertical_id": self.vertical_id, "p_days": days},
        ).execute()
        return result.data[0] if result.data else {}

    # ------------------------------------------------------------------
    # Outreach Templates
    # ------------------------------------------------------------------

    def get_templates(
        self,
        approach_type: Optional[str] = None,
        persona: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """Get outreach templates with optional filters."""
        query = (
            self.client.table("outreach_templates")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("reply_rate", desc=True, nullsfirst=False)
        )
        if active_only:
            query = query.eq("active", True)
        if approach_type:
            query = query.eq("approach_type", approach_type)
        if persona:
            query = query.eq("target_persona", persona)
        return query.execute().data

    def update_template_stats(
        self, template_id: str, stats: dict[str, Any]
    ) -> dict:
        """Update performance metrics for a template."""
        stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table("outreach_templates")
            .update(stats)
            .eq("id", template_id)
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def increment_template_usage(self, template_id: str) -> None:
        """Increment the times_used counter for a template."""
        self.client.rpc(
            "increment_template_usage",
            {"p_template_id": template_id},
        ).execute()

    # ------------------------------------------------------------------
    # Opportunities
    # ------------------------------------------------------------------

    def create_opportunity(self, data: dict[str, Any]) -> dict:
        """Create a new sales opportunity."""
        data["vertical_id"] = self.vertical_id
        result = (
            self.client.table("opportunities")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def update_opportunity_stage(
        self, opportunity_id: str, stage: str, **kwargs: Any
    ) -> dict:
        """Move an opportunity to a new stage."""
        updates: dict[str, Any] = {"stage": stage, **kwargs}
        if stage == "closed_won":
            updates["won_at"] = datetime.now(timezone.utc).isoformat()
        elif stage == "closed_lost":
            updates["lost_at"] = datetime.now(timezone.utc).isoformat()
        return self.update_outreach_event(opportunity_id, updates)

    # ------------------------------------------------------------------
    # Suppression List
    # ------------------------------------------------------------------

    def is_suppressed(self, email: str) -> bool:
        """Check if an email is on the suppression list."""
        result = (
            self.client.table("suppression_list")
            .select("id")
            .eq("email", email.lower())
            .eq("vertical_id", self.vertical_id)
            .limit(1)
            .execute()
        )
        return len(result.data) > 0

    def add_to_suppression(self, email: str, reason: str = "manual") -> None:
        """Add an email to the suppression list."""
        self.client.table("suppression_list").upsert(
            {
                "email": email.lower(),
                "reason": reason,
                "vertical_id": self.vertical_id,
            },
            on_conflict="email,vertical_id",
        ).execute()

    # ------------------------------------------------------------------
    # Knowledge Chunks (RAG)
    # ------------------------------------------------------------------

    def store_knowledge_chunk(
        self,
        content: str,
        embedding: list[float],
        chunk_type: str,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
        source_type: str | None = None,
    ) -> dict:
        """Store a new knowledge chunk with its embedding."""
        data = {
            "content": content,
            "embedding": embedding,
            "chunk_type": chunk_type,
            "metadata": metadata or {},
            "source_id": source_id,
            "source_type": source_type,
            "vertical_id": self.vertical_id,
        }
        result = (
            self.client.table("knowledge_chunks")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def search_knowledge(
        self,
        query_embedding: list[float],
        chunk_type: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Semantic search over knowledge chunks using pgvector.

        Uses the match_knowledge_chunks RPC function for vector similarity.
        """
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": similarity_threshold,
            "p_vertical_id": self.vertical_id,
        }
        if chunk_type:
            params["p_chunk_type"] = chunk_type

        result = self.client.rpc(
            "match_knowledge_chunks", params
        ).execute()
        return result.data

    # ------------------------------------------------------------------
    # Pipeline Runs (Audit)
    # ------------------------------------------------------------------

    def log_pipeline_run(
        self,
        lead_id: str,
        node_name: str,
        status: str,
        input_state: dict | None = None,
        output_state: dict | None = None,
        error_message: str | None = None,
        duration_ms: int | None = None,
    ) -> dict:
        """Log a pipeline node execution for auditing."""
        data = {
            "lead_id": lead_id,
            "node_name": node_name,
            "status": status,
            "input_state": input_state or {},
            "output_state": output_state or {},
            "error_message": error_message,
            "duration_ms": duration_ms,
            "vertical_id": self.vertical_id,
        }
        result = (
            self.client.table("pipeline_runs")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_outreach_events(self) -> int:
        """Count total outreach events for this vertical."""
        result = (
            self.client.table("outreach_events")
            .select("id", count="exact")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.count or 0
