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

    # ------------------------------------------------------------------
    # Dashboard Queries
    # ------------------------------------------------------------------

    def count_companies(self) -> int:
        """Count total companies for this vertical."""
        result = (
            self.client.table("companies")
            .select("id", count="exact")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.count or 0

    def count_contacts(self) -> int:
        """Count total contacts for this vertical."""
        result = (
            self.client.table("contacts")
            .select("id", count="exact")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.count or 0

    def count_opportunities(self) -> int:
        """Count total opportunities for this vertical."""
        result = (
            self.client.table("opportunities")
            .select("id", count="exact")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.count or 0

    def get_pipeline_runs(self, limit: int = 50) -> list[dict]:
        """Get recent pipeline runs for audit/dashboard display."""
        return (
            self.client.table("pipeline_runs")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        ).data

    def get_knowledge_stats(self) -> dict[str, int]:
        """Get knowledge chunk counts grouped by type."""
        result = (
            self.client.table("knowledge_chunks")
            .select("chunk_type")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        counts: dict[str, int] = {}
        for row in result.data:
            ct = row.get("chunk_type", "unknown")
            counts[ct] = counts.get(ct, 0) + 1
        return counts

    def get_recent_outreach(self, limit: int = 20) -> list[dict]:
        """Get recent outreach events with contact info for dashboard."""
        return (
            self.client.table("outreach_events")
            .select("*, contacts(name, email, title)")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        ).data

    # ------------------------------------------------------------------
    # Agent Registry
    # ------------------------------------------------------------------

    def register_agent(self, data: dict[str, Any]) -> dict:
        """Register or update an agent in the database."""
        data["vertical_id"] = self.vertical_id
        result = (
            self.client.table("agents")
            .upsert(data, on_conflict="agent_id,vertical_id")
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_agent_record(self, agent_id: str) -> Optional[dict]:
        """Get an agent record by its agent_id."""
        result = (
            self.client.table("agents")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("vertical_id", self.vertical_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def list_agent_records(self, enabled_only: bool = False) -> list[dict]:
        """List all agents for this vertical."""
        query = (
            self.client.table("agents")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("agent_id")
        )
        if enabled_only:
            query = query.eq("enabled", True)
        return query.execute().data

    def reset_agent_errors(
        self, agent_id: str, vertical_id: str
    ) -> None:
        """Reset consecutive error counter (called on success)."""
        self.client.rpc(
            "reset_agent_errors",
            {"p_agent_id": agent_id, "p_vertical_id": vertical_id},
        ).execute()

    def record_agent_error(
        self,
        agent_id: str,
        vertical_id: str,
        error_message: Optional[str] = None,
    ) -> dict:
        """Record an error and check circuit breaker threshold."""
        result = self.client.rpc(
            "record_agent_error",
            {
                "p_agent_id": agent_id,
                "p_vertical_id": vertical_id,
                "p_error_message": error_message,
            },
        ).execute()
        return result.data if result.data else {}

    # ------------------------------------------------------------------
    # Agent Runs
    # ------------------------------------------------------------------

    def log_agent_run(
        self,
        agent_id: str,
        agent_type: str,
        run_id: str,
        status: str,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        vertical_id: Optional[str] = None,
    ) -> dict:
        """Log an agent run to the agent_runs table."""
        data: dict[str, Any] = {
            "run_id": run_id,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "status": status,
            "vertical_id": vertical_id or self.vertical_id,
        }
        if input_data is not None:
            data["input_data"] = input_data
        if output_data is not None:
            data["output_data"] = output_data
        if error_message is not None:
            data["error_message"] = error_message
        if duration_ms is not None:
            data["duration_ms"] = duration_ms

        # Upsert: "started" creates, "completed"/"failed" updates
        result = (
            self.client.table("agent_runs")
            .upsert(data, on_conflict="run_id")
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_agent_runs(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get agent runs with optional filters."""
        query = (
            self.client.table("agent_runs")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if agent_id:
            query = query.eq("agent_id", agent_id)
        if status:
            query = query.eq("status", status)
        return query.execute().data

    def get_agent_stats(
        self,
        agent_id: Optional[str] = None,
        days: int = 30,
    ) -> list[dict]:
        """Get agent statistics via RPC."""
        params: dict[str, Any] = {
            "p_vertical_id": self.vertical_id,
            "p_days": days,
        }
        if agent_id:
            params["p_agent_id"] = agent_id
        result = self.client.rpc("get_agent_stats", params).execute()
        return result.data

    # ------------------------------------------------------------------
    # Task Queue
    # ------------------------------------------------------------------

    def enqueue_task(self, data: dict[str, Any]) -> dict:
        """Add a task to the queue."""
        data["vertical_id"] = self.vertical_id
        result = (
            self.client.table("task_queue")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def claim_next_task(self, agent_id: str) -> Optional[dict]:
        """Atomically claim the next pending task (via RPC)."""
        result = self.client.rpc(
            "claim_next_task",
            {
                "p_agent_id": agent_id,
                "p_vertical_id": self.vertical_id,
            },
        ).execute()
        return result.data if result.data else None

    def complete_task(
        self,
        task_id: str,
        output_data: Optional[dict] = None,
    ) -> dict:
        """Mark a task as completed."""
        updates: dict[str, Any] = {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if output_data is not None:
            updates["output_data"] = output_data
        result = (
            self.client.table("task_queue")
            .update(updates)
            .eq("task_id", task_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def fail_task(
        self,
        task_id: str,
        error_message: str,
        retry: bool = True,
    ) -> dict:
        """Mark a task as failed, optionally re-enqueue for retry."""
        current = (
            self.client.table("task_queue")
            .select("retry_count, max_retries")
            .eq("task_id", task_id)
            .limit(1)
            .execute()
        )
        task_data = current.data[0] if current.data else {}
        retry_count = task_data.get("retry_count", 0)
        max_retries = task_data.get("max_retries", 3)

        if retry and retry_count < max_retries:
            updates: dict[str, Any] = {
                "status": "pending",
                "error_message": error_message,
                "retry_count": retry_count + 1,
                "claimed_at": None,
                "heartbeat_at": None,
            }
        else:
            updates = {
                "status": "failed",
                "error_message": error_message,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        result = (
            self.client.table("task_queue")
            .update(updates)
            .eq("task_id", task_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def heartbeat_task(self, task_id: str) -> None:
        """Update heartbeat timestamp for a running task."""
        self.client.table("task_queue").update(
            {"heartbeat_at": datetime.now(timezone.utc).isoformat()}
        ).eq("task_id", task_id).execute()

    def recover_zombie_tasks(self, stale_minutes: int = 10) -> int:
        """Recover zombie tasks via RPC. Returns count recovered."""
        result = self.client.rpc(
            "recover_zombie_tasks",
            {"p_stale_minutes": stale_minutes},
        ).execute()
        return result.data if isinstance(result.data, int) else 0

    def count_pending_tasks(self, agent_id: str) -> int:
        """Count pending tasks for an agent."""
        result = (
            self.client.table("task_queue")
            .select("id", count="exact")
            .eq("target_agent_id", agent_id)
            .eq("status", "pending")
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.count or 0

    def list_tasks(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List tasks with optional filters."""
        query = (
            self.client.table("task_queue")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if agent_id:
            query = query.eq("target_agent_id", agent_id)
        if status:
            query = query.eq("status", status)
        return query.execute().data

    # ------------------------------------------------------------------
    # Shared Insights (Cross-Agent Brain)
    # ------------------------------------------------------------------

    def store_insight(
        self,
        source_agent_id: str,
        insight_type: str,
        content: str,
        title: str = "",
        confidence_score: float = 0.5,
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        vertical_id: Optional[str] = None,
        embedding: Optional[list[float]] = None,
    ) -> dict:
        """Store a new insight in the shared brain."""
        data: dict[str, Any] = {
            "source_agent_id": source_agent_id,
            "insight_type": insight_type,
            "title": title,
            "content": content,
            "confidence_score": confidence_score,
            "metadata": metadata or {},
            "vertical_id": vertical_id or self.vertical_id,
        }
        if related_entity_id:
            data["related_entity_id"] = related_entity_id
        if related_entity_type:
            data["related_entity_type"] = related_entity_type
        if embedding:
            data["embedding"] = embedding
        result = (
            self.client.table("shared_insights")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def search_insights(
        self,
        query_embedding: list[float],
        insight_type: Optional[str] = None,
        source_agent_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """Semantic search over shared insights."""
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": similarity_threshold,
            "p_vertical_id": self.vertical_id,
        }
        if insight_type:
            params["p_insight_type"] = insight_type
        if source_agent_id:
            params["p_source_agent_id"] = source_agent_id
        result = self.client.rpc(
            "match_shared_insights", params
        ).execute()
        return result.data

    # ------------------------------------------------------------------
    # Agent Content (Generated Artifacts)
    # ------------------------------------------------------------------

    def store_content(self, data: dict[str, Any]) -> dict:
        """Store generated content (blog, proposal, ad copy, etc.)."""
        data["vertical_id"] = self.vertical_id
        result = (
            self.client.table("agent_content")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def update_content(
        self, content_id: str, updates: dict[str, Any]
    ) -> dict:
        """Update content status, body, or metadata."""
        result = (
            self.client.table("agent_content")
            .update(updates)
            .eq("id", content_id)
            .eq("vertical_id", self.vertical_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def list_content(
        self,
        agent_id: Optional[str] = None,
        content_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List generated content with optional filters."""
        query = (
            self.client.table("agent_content")
            .select("*")
            .eq("vertical_id", self.vertical_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if agent_id:
            query = query.eq("agent_id", agent_id)
        if content_type:
            query = query.eq("content_type", content_type)
        if status:
            query = query.eq("status", status)
        return query.execute().data

    # ------------------------------------------------------------------
    # Training Examples (RLHF Data Flywheel)
    # ------------------------------------------------------------------

    def store_training_example(
        self,
        agent_id: str,
        vertical_id: str,
        task_input: dict[str, Any],
        model_output: str,
        human_correction: Optional[str] = None,
        score: Optional[int] = None,
        source: str = "manual_review",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Save a training example for RLHF data collection.

        Every human correction becomes a (bad, good) pair that can be
        used for few-shot prompting, fine-tuning, or DSPy optimization.
        """
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "vertical_id": vertical_id or self.vertical_id,
            "task_input": task_input,
            "model_output": model_output,
            "source": source,
            "metadata": metadata or {},
        }
        if human_correction is not None:
            data["human_correction"] = human_correction
        if score is not None:
            data["score"] = score

        result = (
            self.client.table("training_examples")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_training_examples(
        self,
        agent_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        min_score: Optional[int] = None,
        source: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Retrieve training examples for export or few-shot construction.

        Uses the get_training_examples RPC for efficient filtered retrieval.
        """
        params: dict[str, Any] = {
            "p_limit": limit,
        }
        if agent_id:
            params["p_agent_id"] = agent_id
        if vertical_id:
            params["p_vertical_id"] = vertical_id
        elif self.vertical_id:
            params["p_vertical_id"] = self.vertical_id
        if min_score is not None:
            params["p_min_score"] = min_score
        if source:
            params["p_source"] = source

        result = self.client.rpc(
            "get_training_examples", params
        ).execute()
        return result.data

    def get_training_stats(
        self,
        vertical_id: Optional[str] = None,
    ) -> list[dict]:
        """Get training example statistics grouped by agent."""
        params: dict[str, Any] = {}
        if vertical_id:
            params["p_vertical_id"] = vertical_id
        elif self.vertical_id:
            params["p_vertical_id"] = self.vertical_id

        result = self.client.rpc(
            "get_training_stats", params
        ).execute()
        return result.data

    # ------------------------------------------------------------------
    # Shadow Agents
    # ------------------------------------------------------------------

    def get_shadow_agents(
        self,
        champion_agent_id: str,
        vertical_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Get enabled shadow agents for a champion.

        Returns agents where shadow_of=champion_agent_id,
        shadow_mode=True, and enabled=True.
        """
        query = (
            self.client.table("agents")
            .select("*")
            .eq("shadow_of", champion_agent_id)
            .eq("shadow_mode", True)
            .eq("enabled", True)
        )
        vid = vertical_id or self.vertical_id
        if vid:
            query = query.eq("vertical_id", vid)
        return query.execute().data
