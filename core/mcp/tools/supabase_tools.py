"""
Supabase/RAG MCP tools for the Sovereign Venture Engine.

Wraps EnclaveDB + EmbeddingEngine methods as MCP-compatible tool functions.
These tools give agents access to the shared knowledge base and company data.

Design:
- search_knowledge: semantic vector search over knowledge_chunks
- save_insight: write to the shared_insights cross-agent brain
- query_companies: list/filter companies in the database
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def search_knowledge(
    query: str,
    chunk_type: Optional[str] = None,
    limit: int = 5,
    *,
    _db: Any = None,
    _embedder: Any = None,
) -> str:
    """
    Search the shared knowledge base using natural language.

    Uses vector similarity (pgvector) to find relevant knowledge chunks.
    This is the primary way agents access learned patterns, vulnerability
    knowledge, and past outreach results.

    Args:
        query: Natural language search query.
        chunk_type: Filter by type (e.g. "winning_pattern", "outreach_result",
                    "vulnerability_knowledge", "company_intel").
        limit: Maximum number of results (default: 5).
        _db: Injected EnclaveDB instance (for testing/DI).
        _embedder: Injected EmbeddingEngine instance (for testing/DI).

    Returns:
        JSON string with matching knowledge chunks.
    """
    if _db is None or _embedder is None:
        from core.integrations.supabase_client import EnclaveDB
        from core.rag.embeddings import EmbeddingEngine

        vertical_id = "enclave_guard"  # default; overridden by server factory
        _db = _db or EnclaveDB(vertical_id)
        _embedder = _embedder or EmbeddingEngine()

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "search_knowledge",
            "query": query[:100],
            "chunk_type": chunk_type,
        },
    )

    # Generate embedding for the query
    query_embedding = await _embedder.embed_text(query)

    # Search via pgvector
    results = _db.search_knowledge(
        query_embedding=query_embedding,
        chunk_type=chunk_type,
        limit=limit,
    )

    # Format results for the LLM
    formatted = []
    for chunk in results:
        formatted.append({
            "content": chunk.get("content", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "source_type": chunk.get("source_type", ""),
            "similarity": chunk.get("similarity"),
            "metadata": chunk.get("metadata", {}),
        })

    return json.dumps({
        "results": formatted,
        "count": len(formatted),
        "query": query,
    }, indent=2)


def save_insight(
    insight_type: str,
    content: str,
    confidence: float = 0.8,
    title: str = "",
    *,
    _db: Any = None,
) -> str:
    """
    Save an insight to the shared cross-agent brain.

    Insights are learnings that any agent can benefit from:
    - Winning outreach patterns
    - Effective messaging for specific personas
    - Industry-specific talking points
    - Objection handling strategies

    Args:
        insight_type: Category of insight (e.g. "winning_pattern",
                      "objection_handling", "persona_insight", "industry_trend").
        content: The insight text content.
        confidence: Confidence score 0.0-1.0 (default: 0.8).
        title: Short title for the insight.
        _db: Injected EnclaveDB instance (for testing/DI).

    Returns:
        JSON string confirming the save.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB

        _db = EnclaveDB("enclave_guard")

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "save_insight",
            "insight_type": insight_type,
            "confidence": confidence,
        },
    )

    result = _db.store_insight(
        source_agent_id="mcp_tool",
        insight_type=insight_type,
        content=content,
        title=title,
        confidence_score=confidence,
    )

    return json.dumps({
        "saved": True,
        "insight_type": insight_type,
        "title": title or content[:60],
        "id": result.get("id", ""),
    })


def query_companies(
    industry: Optional[str] = None,
    limit: int = 20,
    *,
    _db: Any = None,
) -> str:
    """
    Query companies from the database with optional filters.

    Args:
        industry: Filter by industry (e.g. "Information Technology").
        limit: Maximum results to return (default: 20).
        _db: Injected EnclaveDB instance (for testing/DI).

    Returns:
        JSON string with company list.
    """
    if _db is None:
        from core.integrations.supabase_client import EnclaveDB

        _db = EnclaveDB("enclave_guard")

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "query_companies",
            "industry": industry,
            "limit": limit,
        },
    )

    companies = _db.list_companies(
        limit=limit,
        industry=industry,
    )

    # Extract essential fields
    formatted = []
    for company in companies:
        formatted.append({
            "name": company.get("name", ""),
            "domain": company.get("domain", ""),
            "industry": company.get("industry", ""),
            "employee_count": company.get("employee_count"),
            "qualification_score": company.get("qualification_score"),
            "last_contacted_at": company.get("last_contacted_at"),
        })

    return json.dumps({
        "companies": formatted,
        "count": len(formatted),
    }, indent=2)
