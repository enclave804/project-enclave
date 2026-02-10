"""
Data Quality MCP tools for the Sovereign Venture Engine.

Provides tools for email/phone validation, duplicate detection,
and data freshness monitoring. Used by the Data Enrichment Agent
for CRM hygiene workflows.

Tools:
    - validate_email: Validate email format and optionally check DNS MX records
    - find_duplicates: Find potential duplicate records by fuzzy matching
    - validate_phone: Validate phone number format
    - check_data_freshness: Identify records not updated within threshold days
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# RFC 5322 simplified email regex
_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)

# E.164 phone format regex (international)
_PHONE_REGEX = re.compile(r"^\+?[1-9]\d{1,14}$")


async def validate_email(email: str) -> str:
    """
    Validate email format and optionally check DNS MX records.

    Performs:
        1. Format validation (RFC 5322 simplified)
        2. Domain extraction and basic checks
        3. MX record lookup (stub — returns placeholder)

    Returns JSON with is_valid, format_ok, domain, and mx_found.
    """
    try:
        email = email.strip().lower()
        format_ok = bool(_EMAIL_REGEX.match(email))

        domain = ""
        mx_found = False
        issues: list[str] = []

        if format_ok:
            domain = email.split("@")[1]
            # Stub: in production, perform actual DNS MX lookup
            # import dns.resolver
            # mx_records = dns.resolver.resolve(domain, 'MX')
            mx_found = True  # Placeholder
        else:
            issues.append("invalid_format")

        if not email:
            issues.append("empty_email")
            format_ok = False

        result = {
            "status": "success",
            "email": email,
            "is_valid": format_ok and mx_found,
            "format_ok": format_ok,
            "domain": domain,
            "mx_found": mx_found,
            "issues": issues,
        }

        logger.info(
            "email_validated",
            extra={
                "email_domain": domain,
                "is_valid": result["is_valid"],
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Email validation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "email": email,
        })


async def find_duplicates(
    table: str,
    field: str,
    threshold: float = 0.85,
) -> str:
    """
    Find potential duplicate records by fuzzy matching.

    Scans the specified table and field for records with similarity
    above the threshold (0.0-1.0). Returns candidate duplicate
    groups for human review.

    In production, uses pg_trgm or Levenshtein distance via
    Supabase RPC.
    """
    try:
        # Stub: return empty duplicate groups
        result = {
            "status": "success",
            "table": table,
            "field": field,
            "threshold": threshold,
            "duplicate_groups": [],
            "total_groups_found": 0,
            "total_records_affected": 0,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "duplicate_scan_complete",
            extra={
                "table": table,
                "field": field,
                "groups_found": 0,
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "table": table,
            "field": field,
        })


async def validate_phone(phone_number: str) -> str:
    """
    Validate phone number format.

    Checks against E.164 international format (+[country code][number]).
    Strips common formatting characters before validation.

    Returns JSON with is_valid, normalized number, and issues.
    """
    try:
        # Strip common formatting characters
        cleaned = re.sub(r"[\s\-\(\)\.]", "", phone_number.strip())
        format_ok = bool(_PHONE_REGEX.match(cleaned))

        issues: list[str] = []
        if not format_ok:
            if not cleaned:
                issues.append("empty_phone")
            elif not cleaned[0] == "+" and not cleaned[0].isdigit():
                issues.append("invalid_prefix")
            else:
                issues.append("invalid_format")

        result = {
            "status": "success",
            "original": phone_number,
            "normalized": cleaned,
            "is_valid": format_ok,
            "format_ok": format_ok,
            "issues": issues,
        }

        logger.info(
            "phone_validated",
            extra={"is_valid": format_ok},
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Phone validation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "phone_number": phone_number,
        })


async def check_data_freshness(
    table: str,
    days_threshold: int = 180,
) -> str:
    """
    Identify records not updated within threshold days.

    Scans the specified table for records where updated_at is
    older than days_threshold. Returns stale record count and
    sample IDs for review.

    In production, queries Supabase with a date filter.
    """
    try:
        # Stub: return empty stale records
        result = {
            "status": "success",
            "table": table,
            "days_threshold": days_threshold,
            "total_records": 0,
            "stale_records": 0,
            "stale_percentage": 0.0,
            "sample_stale_ids": [],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "data_freshness_checked",
            extra={
                "table": table,
                "stale_count": 0,
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Data freshness check failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "table": table,
        })
