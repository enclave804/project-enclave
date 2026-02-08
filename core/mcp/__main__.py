"""
CLI entry point for running the MCP server.

Usage:
    python -m core.mcp

This starts the FastMCP server with stdio transport, which is the
standard way to connect MCP servers to agents. The agent connects
to the server's stdin/stdout pipes.
"""

import logging

from core.observability.logging_config import configure_logging
from core.mcp.server import get_server

# Configure structured logging before anything else
configure_logging()

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the MCP server."""
    logger.info("Starting Enclave MCP server...")
    server = get_server()
    server.run()


if __name__ == "__main__":
    main()
