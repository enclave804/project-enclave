"""
Apollo.io API client for Project Enclave.

Handles lead discovery, people search, and company enrichment
via the Apollo.io REST API. Designed for the Basic plan ($49/mo).

API Reference: https://docs.apollo.io/reference
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

APOLLO_BASE_URL = "https://api.apollo.io/api/v1"


class ApolloClient:
    """
    Apollo.io API client for lead sourcing and enrichment.

    Capabilities on Basic plan:
    - People Search: find contacts by title, company, industry, location
    - Company Enrichment: tech stack, funding, employee count
    - Contact Details: verified email addresses
    - Rate limit: ~300 requests/hour
    """

    def __init__(self, api_key_env: str = "APOLLO_API_KEY"):
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise EnvironmentError(
                f"Apollo API key not found. Set {api_key_env} environment variable."
            )
        self.base_url = APOLLO_BASE_URL
        self._headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    async def search_people(
        self,
        person_titles: list[str] | None = None,
        person_seniorities: list[str] | None = None,
        organization_num_employees_ranges: list[str] | None = None,
        organization_industry_tag_ids: list[str] | None = None,
        person_locations: list[str] | None = None,
        per_page: int = 25,
        page: int = 1,
    ) -> dict[str, Any]:
        """
        Search for people matching criteria.

        Args:
            person_titles: Job title filters (e.g., ["CTO", "CISO"]).
            person_seniorities: Seniority levels (e.g., ["c_suite", "vp"]).
            organization_num_employees_ranges: Ranges like ["11,50", "51,200"].
            organization_industry_tag_ids: Industry filter IDs.
            person_locations: Location filters (e.g., ["United States"]).
            per_page: Results per page (max 100).
            page: Page number.

        Returns:
            Dict with 'people' list and 'pagination' info.
        """
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "page": page,
            "per_page": min(per_page, 100),
        }

        if person_titles:
            payload["person_titles"] = person_titles
        if person_seniorities:
            payload["person_seniorities"] = person_seniorities
        if organization_num_employees_ranges:
            payload["organization_num_employees_ranges"] = (
                organization_num_employees_ranges
            )
        if organization_industry_tag_ids:
            payload["organization_industry_tag_ids"] = (
                organization_industry_tag_ids
            )
        if person_locations:
            payload["person_locations"] = person_locations

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/mixed_people/search",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def search_people_from_config(
        self,
        filters: dict[str, Any],
        page: int = 1,
    ) -> dict[str, Any]:
        """
        Search people using filters from a vertical config.yaml.

        This is the main method called by the pipeline's
        pull_leads_from_apollo node.
        """
        return await self.search_people(
            person_titles=filters.get("person_titles"),
            person_seniorities=filters.get("person_seniorities"),
            organization_num_employees_ranges=filters.get(
                "organization_num_employees_ranges"
            ),
            organization_industry_tag_ids=filters.get(
                "organization_industry_tag_ids"
            ),
            person_locations=filters.get("person_locations"),
            per_page=filters.get("per_page", 25),
            page=page,
        )

    async def enrich_person(self, email: str) -> dict[str, Any]:
        """
        Enrich a person by email address.

        Returns detailed contact info, company data, and employment history.
        """
        payload = {
            "api_key": self.api_key,
            "email": email,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/people/match",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def enrich_company(self, domain: str) -> dict[str, Any]:
        """
        Enrich a company by domain.

        Returns company details: industry, employee count,
        funding, tech stack, etc.
        """
        payload = {
            "api_key": self.api_key,
            "domain": domain,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/organizations/enrich",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def get_person(self, apollo_id: str) -> dict[str, Any]:
        """Get a person's full profile by Apollo ID."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/people/{apollo_id}",
                params={"api_key": self.api_key},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    def parse_lead(self, raw_person: dict) -> dict[str, Any]:
        """
        Parse a raw Apollo person record into our internal format.

        Maps Apollo's data structure to our contacts + companies schema.
        """
        org = raw_person.get("organization", {}) or {}

        contact = {
            "name": raw_person.get("name", ""),
            "title": raw_person.get("title", ""),
            "email": raw_person.get("email", ""),
            "linkedin_url": raw_person.get("linkedin_url", ""),
            "phone": (
                raw_person.get("phone_numbers", [{}])[0].get("sanitized_number")
                if raw_person.get("phone_numbers")
                else None
            ),
            "seniority": raw_person.get("seniority", ""),
            "apollo_id": raw_person.get("id", ""),
        }

        company = {
            "name": org.get("name", ""),
            "domain": org.get("primary_domain", ""),
            "industry": org.get("industry", ""),
            "employee_count": org.get("estimated_num_employees"),
            "linkedin_url": org.get("linkedin_url", ""),
            "website_url": org.get("website_url", ""),
            "apollo_id": org.get("id", ""),
            "tech_stack": self._extract_tech_stack(org),
            "source": "apollo",
        }

        return {"contact": contact, "company": company}

    def _extract_tech_stack(self, org: dict) -> dict[str, str]:
        """Extract tech stack from Apollo organization data."""
        tech = {}
        # Apollo stores tech in current_technologies
        for item in org.get("current_technologies", []):
            if isinstance(item, dict):
                name = item.get("name", "")
                category = item.get("category", "other")
                if name:
                    tech[name] = category
            elif isinstance(item, str):
                tech[item] = "unknown"
        return tech

    async def search_and_parse(
        self,
        filters: dict[str, Any],
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Search for leads and parse them into our internal format.

        This is the high-level method that combines search + parsing.
        Returns a list of {"contact": {...}, "company": {...}} dicts.
        """
        result = await self.search_people_from_config(filters, page=page)
        people = result.get("people", [])

        parsed = []
        for person in people:
            if not person.get("email"):
                continue  # skip contacts without email
            try:
                parsed.append(self.parse_lead(person))
            except Exception as e:
                logger.warning(f"Failed to parse Apollo lead: {e}")
                continue

        logger.info(
            f"Apollo search returned {len(people)} results, "
            f"parsed {len(parsed)} valid leads"
        )
        return parsed
