"""
Apollo.io MCP tools for the Sovereign Venture Engine.

Wraps ApolloClient methods as MCP-compatible tool functions.
These tools are registered on the FastMCP server in core/mcp/server.py.

Each tool is a thin async wrapper that:
1. Accepts simple parameters (no complex objects)
2. Delegates to the underlying ApolloClient
3. Returns JSON-serializable results
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def search_leads(
    titles: Optional[str] = None,
    seniorities: Optional[str] = None,
    locations: Optional[str] = None,
    employee_ranges: Optional[str] = None,
    per_page: int = 25,
    *,
    _apollo_client: Any = None,
) -> str:
    """
    Search Apollo.io for leads matching the given criteria.

    Args:
        titles: Comma-separated job titles (e.g. "CTO,CISO,VP Engineering").
        seniorities: Comma-separated seniority levels (e.g. "c_suite,vp,director").
        locations: Comma-separated locations (e.g. "United States,Canada").
        employee_ranges: Comma-separated employee count ranges (e.g. "11,50|51,200").
        per_page: Number of results per page (max 100).
        _apollo_client: Injected ApolloClient instance (for testing/DI).

    Returns:
        JSON string with search results.
    """
    if _apollo_client is None:
        from core.integrations.apollo_client import ApolloClient
        _apollo_client = ApolloClient()

    # Parse comma-separated strings into lists
    title_list = [t.strip() for t in titles.split(",")] if titles else None
    seniority_list = (
        [s.strip() for s in seniorities.split(",")]
        if seniorities
        else None
    )
    location_list = (
        [loc.strip() for loc in locations.split(",")]
        if locations
        else None
    )
    employee_range_list = (
        [r.strip() for r in employee_ranges.split("|")]
        if employee_ranges
        else None
    )

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "search_leads",
            "titles": titles,
            "per_page": per_page,
        },
    )

    result = await _apollo_client.search_people(
        person_titles=title_list,
        person_seniorities=seniority_list,
        person_locations=location_list,
        organization_num_employees_ranges=employee_range_list,
        per_page=per_page,
    )

    # Extract just the essential fields for the LLM
    people = result.get("people", [])
    summary = []
    for person in people[:per_page]:
        org = person.get("organization", {}) or {}
        summary.append({
            "name": person.get("name", ""),
            "title": person.get("title", ""),
            "email": person.get("email", ""),
            "company": org.get("name", ""),
            "domain": org.get("primary_domain", ""),
            "industry": org.get("industry", ""),
            "employees": org.get("estimated_num_employees"),
        })

    pagination = result.get("pagination", {})
    return json.dumps({
        "leads": summary,
        "total_results": pagination.get("total_entries", len(summary)),
        "page": pagination.get("page", 1),
    }, indent=2)


async def enrich_company(
    domain: str,
    *,
    _apollo_client: Any = None,
) -> str:
    """
    Enrich a company by its domain using Apollo.io.

    Returns company details: industry, employee count, funding,
    tech stack, and more.

    Args:
        domain: Company domain (e.g. "acme.com").
        _apollo_client: Injected ApolloClient instance (for testing/DI).

    Returns:
        JSON string with company details.
    """
    if _apollo_client is None:
        from core.integrations.apollo_client import ApolloClient
        _apollo_client = ApolloClient()

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "enrich_company", "domain": domain},
    )

    result = await _apollo_client.enrich_company(domain)

    # Extract the organization data
    org = result.get("organization", {})
    if not org:
        return json.dumps({"error": f"No company found for domain: {domain}"})

    return json.dumps({
        "name": org.get("name", ""),
        "domain": org.get("primary_domain", ""),
        "industry": org.get("industry", ""),
        "employee_count": org.get("estimated_num_employees"),
        "annual_revenue": org.get("annual_revenue"),
        "founded_year": org.get("founded_year"),
        "linkedin_url": org.get("linkedin_url", ""),
        "website_url": org.get("website_url", ""),
        "short_description": org.get("short_description", ""),
        "tech_stack": [
            t.get("name", str(t)) if isinstance(t, dict) else str(t)
            for t in org.get("current_technologies", [])
        ][:20],  # limit tech stack to top 20
    }, indent=2)
