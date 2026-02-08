"""
RAG retrieval engine for Project Enclave.

Provides hybrid search combining vector similarity (pgvector),
SQL filtering, and keyword matching for maximum relevance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.integrations.supabase_client import EnclaveDB
from core.rag.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from the RAG store.

    Uses a hybrid approach:
    1. Vector similarity search (pgvector cosine distance)
    2. Metadata filtering (chunk_type, source_type, etc.)
    3. Result re-ranking based on metadata relevance
    """

    def __init__(self, db: EnclaveDB, embedder: EmbeddingEngine):
        self.db = db
        self.embedder = embedder

    async def search(
        self,
        query: str,
        chunk_type: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base with a natural language query.

        Args:
            query: Natural language search query.
            chunk_type: Filter by chunk type (e.g., 'winning_pattern').
            limit: Max results to return.
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0).
            metadata_filters: Additional metadata key-value filters.

        Returns:
            List of matching knowledge chunks with similarity scores.
        """
        query_embedding = await self.embedder.embed_text(query)

        results = self.db.search_knowledge(
            query_embedding=query_embedding,
            chunk_type=chunk_type,
            limit=limit * 2,  # over-fetch for post-filtering
            similarity_threshold=similarity_threshold,
        )

        # Apply metadata filters if provided
        if metadata_filters:
            results = [
                r for r in results
                if self._matches_metadata(r.get("metadata", {}), metadata_filters)
            ]

        # Truncate to requested limit
        return results[:limit]

    async def find_previous_outreach(
        self,
        company_domain: str,
        contact_email: Optional[str] = None,
    ) -> list[dict]:
        """
        Check if we have prior outreach data for a company or contact.

        This is used by the check_duplicate node to avoid re-contacting
        recent leads and to learn from past interactions.
        """
        query = f"outreach to company {company_domain}"
        if contact_email:
            query += f" contact {contact_email}"

        results = await self.search(
            query=query,
            chunk_type="outreach_result",
            limit=10,
            similarity_threshold=0.6,
            metadata_filters={"contact_email": contact_email} if contact_email else None,
        )
        return results

    async def find_winning_patterns(
        self,
        persona: str,
        industry: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Retrieve winning outreach patterns for a specific persona/industry.

        Used by the select_strategy node to choose the best approach
        based on historical performance data.
        """
        query = f"winning outreach pattern for {persona} in {industry}"
        results = await self.search(
            query=query,
            chunk_type="winning_pattern",
            limit=limit,
            similarity_threshold=0.5,  # lower threshold for pattern matching
        )

        # Sort by win_rate (highest first)
        results.sort(
            key=lambda r: r.get("metadata", {}).get("win_rate", 0),
            reverse=True,
        )
        return results

    async def find_vulnerability_context(
        self,
        tech_stack: dict[str, Any],
        vulnerabilities: list[dict],
    ) -> list[dict]:
        """
        Find relevant vulnerability knowledge to support outreach messaging.

        Given a company's tech stack and detected vulnerabilities,
        retrieve knowledge that helps explain the risks in business terms.
        """
        # Build query from tech stack and vulnerabilities
        query_parts = []
        for tech, version in tech_stack.items():
            query_parts.append(f"{tech} {version} vulnerability")
        for vuln in vulnerabilities[:5]:
            if isinstance(vuln, dict):
                query_parts.append(vuln.get("type", "") + " " + vuln.get("description", ""))
            else:
                query_parts.append(str(vuln))

        query = " ".join(query_parts[:500])  # truncate if too long

        return await self.search(
            query=query,
            chunk_type="vulnerability_knowledge",
            limit=5,
            similarity_threshold=0.5,
        )

    async def find_objection_handling(
        self, objection_text: str
    ) -> list[dict]:
        """
        Find relevant objection handling patterns.

        Used when a prospect replies with an objection to find
        successful responses to similar objections.
        """
        return await self.search(
            query=f"objection response: {objection_text}",
            chunk_type="objection_handling",
            limit=3,
            similarity_threshold=0.6,
        )

    async def get_company_context(
        self, company_domain: str
    ) -> list[dict]:
        """
        Get all stored intelligence about a specific company.

        Useful for building context before drafting outreach.
        """
        return await self.search(
            query=f"company intelligence {company_domain}",
            chunk_type="company_intel",
            limit=5,
            similarity_threshold=0.5,
        )

    def _matches_metadata(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Check if chunk metadata matches all filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
