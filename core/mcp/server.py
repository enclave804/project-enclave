"""
FastMCP server for the Sovereign Venture Engine.

Exposes Apollo, Supabase/RAG, and Email tools to agents via the
Model Context Protocol (MCP). Each tool is a thin wrapper around
the existing integration clients.

Entry points:
    python -m core.mcp           # stdio transport (for agent integration)
    create_mcp_server()          # programmatic use (for testing)

Architecture:
    FastMCP server
    +-- search_leads()       -> ApolloClient.search_people()
    +-- enrich_company()     -> ApolloClient.enrich_company()
    +-- search_knowledge()   -> EnclaveDB.search_knowledge() via embedder
    +-- save_insight()       -> EnclaveDB.store_insight()
    +-- query_companies()    -> EnclaveDB.list_companies()
    +-- send_email()         -> EmailEngine.send_email() [@sandboxed_tool]
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when fastmcp isn't installed
_server_instance = None


def create_mcp_server(
    name: str = "Enclave Tools",
    vertical_id: str = "enclave_guard",
) -> "FastMCP":
    """
    Create and configure the FastMCP server with all tools registered.

    Uses Tool.from_function() to wrap existing tool functions,
    which accepts raw async/sync functions and converts them to
    FastMCP Tool objects.

    Args:
        name: Server name for MCP protocol handshake.
        vertical_id: Default vertical for Supabase queries.

    Returns:
        Configured FastMCP server instance.
    """
    from fastmcp import FastMCP
    from fastmcp.tools import Tool

    mcp = FastMCP(name)

    # ─── Apollo Tools ─────────────────────────────────────────────
    from core.mcp.tools.apollo_tools import search_leads, enrich_company

    mcp.add_tool(Tool.from_function(search_leads))
    mcp.add_tool(Tool.from_function(enrich_company))

    # ─── Supabase / RAG Tools ────────────────────────────────────
    from core.mcp.tools.supabase_tools import (
        search_knowledge,
        save_insight,
        query_companies,
    )

    mcp.add_tool(Tool.from_function(search_knowledge))
    mcp.add_tool(Tool.from_function(save_insight))
    mcp.add_tool(Tool.from_function(query_companies))

    # ─── Email Tools (sandboxed) ─────────────────────────────────
    from core.mcp.tools.email_tools import send_email

    mcp.add_tool(Tool.from_function(send_email))

    logger.info(
        "mcp_server_created",
        extra={
            "server_name": name,
            "vertical_id": vertical_id,
            "tool_count": 6,
        },
    )

    return mcp


def get_server() -> "FastMCP":
    """Get or create the singleton MCP server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = create_mcp_server()
    return _server_instance
