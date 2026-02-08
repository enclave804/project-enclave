"""
MCP (Model Context Protocol) server for the Sovereign Venture Engine.

Exposes agent tools via FastMCP so they can be called by LLM agents
through the standardized MCP protocol.

Tool modules:
- apollo_tools: Lead search and company enrichment via Apollo.io
- supabase_tools: Knowledge base search and company queries
- email_tools: Email sending (sandboxed in non-production)

Usage:
    python -m core.mcp  # Start the MCP server (stdio transport)
"""
