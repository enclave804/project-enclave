"""
Mock Apollo client for pipeline testing.

Returns empty enrichment data so the pipeline continues with whatever
data the mock leads already provide (graceful degradation).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockApolloClient:
    """Mock Apollo client that satisfies the interface used by PipelineNodes."""

    def __init__(self):
        logger.info("MockApolloClient initialized (test mode)")

    async def enrich_company(self, domain: str) -> dict[str, Any]:
        """Return empty enrichment — simulates Apollo being unavailable."""
        logger.info(f"[MOCK] Apollo enrich_company called for {domain} — returning empty")
        return {"organization": {}}

    async def search_people(self, **kwargs) -> list[dict[str, Any]]:
        """Not used in test-run, but satisfies the interface."""
        return []

    async def search_and_parse(self, filters: Any, page: int = 1) -> list[dict[str, Any]]:
        """Not used in test-run (mock leads supplied directly)."""
        return []
